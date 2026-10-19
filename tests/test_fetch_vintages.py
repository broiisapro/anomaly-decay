"""Offline tests for multi-vintage OSAP fetches (Phase 2 vintage comparison, FR-2.3).

Everything is mocked: a fake OpenAP serves hand-built frames and raises on demand, so
the tests exercise the non-primary paths -- reduced portfolio set, per-vintage canonical
directories, manifest rules, legacy header mapping and the CLI -- without network.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pytest
from anomaly_decay.config import DataConfig, FrenchConfig, OsapConfig
from anomaly_decay.data import fetch as fetch_module
from anomaly_decay.data.fetch import (
    MAX_ATTEMPTS,
    NON_PRIMARY_PORTFOLIOS,
    PORT_COLUMNS,
    FetchError,
    ManifestUpdater,
    fetch_osap,
    load_compare_vintages,
    main,
    manifest_path_for,
    read_manifest,
)

PRIMARY = 202510
COMPARE = 202410
FIXTURE_CONFIG = Path(__file__).resolve().parents[1] / "config" / "data.toml"


def make_config(root: Path, *, vintage: int = PRIMARY) -> DataConfig:
    return DataConfig(
        root=root,
        osap=OsapConfig(
            vintage=vintage,
            portfolios=("op", "deciles_ew", "deciles_vw"),
            raw_dir=root / "data" / "raw",
        ),
        french=FrenchConfig(
            ff5_url="https://example.invalid/ff5.zip",
            umd_url="https://example.invalid/umd.zip",
            pinned_dir=root / "data" / "pinned",
            downloaded_on="2026-10-16",
            ff5_sha256="0" * 64,
            umd_sha256="0" * 64,
        ),
    )


def signal_doc_frame(rows: int = 150, acronym: str = "Acronym") -> pl.DataFrame:
    acronyms = ["Size", "Mom12m"] + [f"Filler{i:03d}" for i in range(rows - 2)]
    return pl.DataFrame(
        {
            acronym: acronyms,
            "Cat.Signal": ["Predictor"] * rows,
            "SampleStartYear": [1963, 1965] + [1970] * (rows - 2),
            "SampleEndYear": [2001, 2000] + [2010] * (rows - 2),
            "Year": [1981, 1993] + [2005] * (rows - 2),
        }
    )


def port_frame(rows: int = 2000, ret: float = 0.5) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "signalname": ["Size"] * rows,
            "port": ["LS"] * rows,
            "date": ["2022-12-30"] * rows,
            "ret": [ret] * rows,
            "signallag": [1] * rows,
            "Nlong": [100] * rows,
            "Nshort": [100] * rows,
        }
    )


class FakeOpenAP:
    """Mimics the ``openassetpricing.OpenAP`` surface used by ``fetch_osap``."""

    def __init__(
        self,
        *,
        doc: pl.DataFrame,
        ports: dict[str, pl.DataFrame],
        fail_times: int = 0,
    ) -> None:
        self.doc = doc
        self.ports = ports
        self.fail_times = fail_times
        self.calls: list[str] = []

    def _maybe_fail(self, label: str) -> None:
        self.calls.append(label)
        if len(self.calls) <= self.fail_times:
            raise ConnectionError("simulated Drive quota failure")

    def dl_signal_doc(self, df_backend: str) -> pl.DataFrame:
        self._maybe_fail("doc")
        return self.doc

    def dl_port(self, data_name: str, df_backend: str) -> pl.DataFrame:
        self._maybe_fail(data_name)
        return self.ports[data_name]


def fake_for(vintage: int, *, doc: pl.DataFrame | None = None) -> FakeOpenAP:
    """A fake OpenAP whose payloads depend on the requested release."""
    ratio = 0.4 if vintage != PRIMARY else 0.5
    return FakeOpenAP(
        doc=doc if doc is not None else signal_doc_frame(),
        ports={
            "op": port_frame(ret=ratio),
            "deciles_ew": port_frame(ret=ratio),
            "deciles_vw": port_frame(ret=ratio),
        },
    )


def _updater(cfg: DataConfig, *, update: bool = False) -> ManifestUpdater:
    return ManifestUpdater(manifest_path_for(cfg, "all"), cfg.root, update=update)


# --- fetch semantics --------------------------------------------------------


def test_fetch_osap_non_primary_defaults_to_op_only(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    seen: list[int] = []

    def factory(vintage: int) -> FakeOpenAP:
        seen.append(vintage)
        return fake_for(vintage)

    written = fetch_osap(cfg, vintage=COMPARE, openap_factory=factory, sleep=lambda _: None)

    assert seen == [COMPARE]
    assert written == [
        cfg.osap.raw_dir / str(COMPARE) / "signal_doc.csv",
        cfg.osap.raw_dir / str(COMPARE) / "port_op.csv",
    ]
    assert NON_PRIMARY_PORTFOLIOS == ("op",)
    for portfolio in ("deciles_ew", "deciles_vw"):
        assert not (cfg.osap.raw_dir / str(COMPARE) / f"port_{portfolio}.csv").exists()
    parsed = pl.read_csv(cfg.osap.raw_dir / str(COMPARE) / "port_op.csv")
    assert parsed.columns == list(PORT_COLUMNS)
    assert parsed["ret"][0] == 0.4


def test_fetch_osap_non_primary_explicit_portfolios_override(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    written = fetch_osap(
        cfg,
        vintage=COMPARE,
        portfolios=("op", "deciles_ew"),
        openap_factory=fake_for,
        sleep=lambda _: None,
    )
    assert [path.name for path in written] == [
        "signal_doc.csv",
        "port_op.csv",
        "port_deciles_ew.csv",
    ]


def test_fetch_osap_primary_defaults_to_configured_portfolios(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    seen: list[int] = []

    def factory(vintage: int) -> FakeOpenAP:
        seen.append(vintage)
        return fake_for(vintage)

    written = fetch_osap(cfg, openap_factory=factory, sleep=lambda _: None)

    assert seen == [PRIMARY]
    assert written == [
        cfg.osap.raw_dir / str(PRIMARY) / "signal_doc.csv",
        cfg.osap.raw_dir / str(PRIMARY) / "port_op.csv",
        cfg.osap.raw_dir / str(PRIMARY) / "port_deciles_ew.csv",
        cfg.osap.raw_dir / str(PRIMARY) / "port_deciles_vw.csv",
    ]


def test_fetch_osap_compare_vintage_retries_quota_like_failures(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    fake = FakeOpenAP(doc=signal_doc_frame(), ports={"op": port_frame()}, fail_times=2)
    sleeps: list[float] = []
    fetch_osap(
        cfg,
        vintage=COMPARE,
        openap_factory=lambda vintage: fake,
        sleep=sleeps.append,
    )
    assert fake.calls == ["doc", "doc", "doc", "op"]
    assert sleeps == [1.0, 2.0]


def test_fetch_osap_compare_vintage_exhausted_retries_name_the_release(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    fake = FakeOpenAP(doc=signal_doc_frame(), ports={"op": port_frame()}, fail_times=99)
    with pytest.raises(FetchError, match=f"failed after {MAX_ATTEMPTS} attempts"):
        fetch_osap(cfg, vintage=COMPARE, openap_factory=lambda vintage: fake, sleep=lambda _: None)
    assert not (cfg.osap.raw_dir / str(COMPARE)).exists()


def test_fetch_osap_rejects_empty_portfolio_list(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    with pytest.raises(FetchError, match="at least one portfolio"):
        fetch_osap(cfg, vintage=COMPARE, portfolios=(), sleep=lambda _: None)


# --- manifest ---------------------------------------------------------------


def test_manifest_gains_compare_vintage_entries_without_touching_the_primary(
    tmp_path: Path,
) -> None:
    cfg = make_config(tmp_path)
    primary = _updater(cfg)
    fetch_osap(cfg, openap_factory=fake_for, sleep=lambda _: None, updater=primary)
    primary.save()
    manifest_path = tmp_path / "data" / "manifest.lock"
    before = {entry.path: (entry.sha256, entry.size) for entry in read_manifest(manifest_path)}

    compare = _updater(cfg)
    fetch_osap(
        cfg,
        vintage=COMPARE,
        openap_factory=fake_for,
        sleep=lambda _: None,
        updater=compare,
    )
    wrote = compare.save()
    assert wrote == manifest_path
    entries = {entry.path: (entry.sha256, entry.size) for entry in read_manifest(manifest_path)}
    assert set(entries) == set(before) | {
        f"data/raw/{COMPARE}/signal_doc.csv",
        f"data/raw/{COMPARE}/port_op.csv",
    }
    for path, digest in before.items():
        assert entries[path] == digest
    assert compare.updated == []
    assert sorted(compare.added) == [
        f"data/raw/{COMPARE}/port_op.csv",
        f"data/raw/{COMPARE}/signal_doc.csv",
    ]


def test_fetch_osap_compare_vintage_second_run_is_cache_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = make_config(tmp_path)
    first = _updater(cfg)
    fetch_osap(
        cfg,
        vintage=COMPARE,
        openap_factory=fake_for,
        sleep=lambda _: None,
        updater=first,
    )
    first.save()
    capsys.readouterr()

    def must_not_download(vintage: int) -> Any:
        raise AssertionError("downloader called for a cached compare vintage")

    second = _updater(cfg)
    written = fetch_osap(
        cfg,
        vintage=COMPARE,
        openap_factory=must_not_download,
        sleep=lambda _: None,
        updater=second,
    )
    assert written == []
    out = capsys.readouterr().out
    assert f"cached: data/raw/{COMPARE}/signal_doc.csv" in out
    assert f"cached: data/raw/{COMPARE}/port_op.csv" in out
    assert second.save() is None


# --- legacy signal-doc headers ----------------------------------------------


def test_legacy_signal_doc_rename_is_applied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = make_config(tmp_path)
    renamed = signal_doc_frame(acronym="SignalCode")
    fake = FakeOpenAP(doc=renamed, ports={"op": port_frame()})
    monkeypatch.setitem(fetch_module.SIGNAL_DOC_RENAMES, "SignalCode", "Acronym")

    written = fetch_osap(
        cfg, vintage=COMPARE, openap_factory=lambda vintage: fake, sleep=lambda _: None
    )
    assert written[0] == cfg.osap.raw_dir / str(COMPARE) / "signal_doc.csv"
    doc = pl.read_csv(written[0])
    assert "Acronym" in doc.columns
    assert "SignalCode" not in doc.columns


def test_unnamed_legacy_signal_doc_columns_fail_loudly(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    renamed = signal_doc_frame(acronym="SignalCode")
    fake = FakeOpenAP(doc=renamed, ports={"op": port_frame()})
    with pytest.raises(FetchError, match="missing columns.*Acronym"):
        fetch_osap(cfg, vintage=COMPARE, openap_factory=lambda vintage: fake, sleep=lambda _: None)
    assert not (cfg.osap.raw_dir / str(COMPARE)).exists()


# --- compare_vintages config --------------------------------------------------


def test_load_compare_vintages_reads_the_preference_order(tmp_path: Path) -> None:
    config_path = tmp_path / "data.toml"
    config_path.write_text(
        "[osap]\nvintage = 202510\ncompare_vintages = [202410, 202408, 2023]\n",
        encoding="utf-8",
    )
    assert load_compare_vintages(config_path) == (202410, 202408, 2023)


def test_load_compare_vintages_missing_key_is_empty(tmp_path: Path) -> None:
    config_path = tmp_path / "data.toml"
    config_path.write_text("[osap]\nvintage = 202510\n", encoding="utf-8")
    assert load_compare_vintages(config_path) == ()


def test_load_compare_vintages_rejects_non_integers(tmp_path: Path) -> None:
    config_path = tmp_path / "data.toml"
    config_path.write_text(
        '[osap]\nvintage = 202510\ncompare_vintages = ["202410"]\n', encoding="utf-8"
    )
    with pytest.raises(FetchError, match="list of integers"):
        load_compare_vintages(config_path)


def test_committed_config_lists_the_preregistered_preference_order() -> None:
    assert load_compare_vintages(FIXTURE_CONFIG) == (202410, 202408, 2023)


# --- CLI --------------------------------------------------------------------


def _write_config_file(root: Path, cfg: DataConfig) -> Path:
    config_path = root / "config" / "data.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        (
            "[osap]\n"
            f"vintage = {cfg.osap.vintage}\n"
            'portfolios = ["op", "deciles_ew", "deciles_vw"]\n'
            "compare_vintages = [202410, 202408, 2023]\n"
            'raw_dir = "data/raw"\n\n'
            "[french]\n"
            f'ff5_url = "{cfg.french.ff5_url}"\n'
            f'umd_url = "{cfg.french.umd_url}"\n'
            'pinned_dir = "data/pinned"\n'
            f'downloaded_on = "{cfg.french.downloaded_on}"\n'
            f'ff5_sha256 = "{cfg.french.ff5_sha256}"\n'
            f'umd_sha256 = "{cfg.french.umd_sha256}"\n'
        ),
        encoding="utf-8",
    )
    return config_path


def test_fetch_cli_vintage_fetches_op_only_and_updates_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = make_config(tmp_path)
    config_path = _write_config_file(tmp_path, cfg)
    monkeypatch.setattr(fetch_module, "_default_openap", fake_for)

    exit_code = main(["--osap", "--vintage", str(COMPARE), "--config", str(config_path)])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert f"wrote data/raw/{COMPARE}/signal_doc.csv" in out
    assert f"wrote data/raw/{COMPARE}/port_op.csv" in out
    assert "wrote data/manifest.lock" in out
    assert not (cfg.osap.raw_dir / str(COMPARE) / "port_deciles_ew.csv").exists()
    entries = {entry.path for entry in read_manifest(tmp_path / "data" / "manifest.lock")}
    assert entries == {
        f"data/raw/{COMPARE}/signal_doc.csv",
        f"data/raw/{COMPARE}/port_op.csv",
    }


def test_fetch_cli_vintage_with_explicit_portfolios(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = make_config(tmp_path)
    config_path = _write_config_file(tmp_path, cfg)
    monkeypatch.setattr(fetch_module, "_default_openap", fake_for)
    exit_code = main(
        [
            "--osap",
            "--vintage",
            str(COMPARE),
            "--portfolios",
            "op,deciles_ew",
            "--config",
            str(config_path),
        ]
    )
    assert exit_code == 0
    assert (cfg.osap.raw_dir / str(COMPARE) / "port_deciles_ew.csv").is_file()
    assert not (cfg.osap.raw_dir / str(COMPARE) / "port_deciles_vw.csv").exists()


def test_fetch_cli_explicit_primary_vintage_keeps_the_full_portfolio_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = make_config(tmp_path)
    config_path = _write_config_file(tmp_path, cfg)
    monkeypatch.setattr(fetch_module, "_default_openap", fake_for)

    exit_code = main(["--osap", "--vintage", str(PRIMARY), "--config", str(config_path)])
    out = capsys.readouterr().out
    assert exit_code == 0
    for name in ("signal_doc.csv", "port_op.csv", "port_deciles_ew.csv", "port_deciles_vw.csv"):
        assert f"wrote data/raw/{PRIMARY}/{name}" in out
    entries = {entry.path for entry in read_manifest(tmp_path / "data" / "manifest.lock")}
    assert entries == {
        f"data/raw/{PRIMARY}/signal_doc.csv",
        f"data/raw/{PRIMARY}/port_op.csv",
        f"data/raw/{PRIMARY}/port_deciles_ew.csv",
        f"data/raw/{PRIMARY}/port_deciles_vw.csv",
    }


def test_fetch_cli_empty_portfolios_exits_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = make_config(tmp_path)
    config_path = _write_config_file(tmp_path, cfg)
    assert main(["--osap", "--portfolios", "", "--config", str(config_path)]) == 1
    assert "--portfolios is empty" in capsys.readouterr().err


def test_fetch_cli_quota_failure_names_the_vintage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = make_config(tmp_path)
    config_path = _write_config_file(tmp_path, cfg)
    broken = FakeOpenAP(doc=signal_doc_frame(), ports={"op": port_frame()}, fail_times=99)
    monkeypatch.setattr(fetch_module, "_default_openap", lambda vintage: broken)

    assert main(["--osap", "--vintage", str(COMPARE), "--config", str(config_path)]) == 1
    err = capsys.readouterr().err
    assert f"OSAP {COMPARE} signal doc" in err
    assert "failed after" in err
