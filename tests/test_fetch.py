"""Offline tests for the fetch layer: mocked OpenAP, fake HTTP sessions, manifest round-trips."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from anomaly_decay.config import (
    ConfigError,
    DataConfig,
    FrenchConfig,
    OsapConfig,
    load_data_config,
)
from anomaly_decay.data.fetch import (
    MAX_ATTEMPTS,
    PORT_COLUMNS,
    FetchError,
    fetch_french,
    fetch_osap,
    main,
    sha256_bytes,
    verify_manifest,
    write_manifest,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures"
FF5_ZIP = (FIXTURE_DIR / "french" / "ff5.zip").read_bytes()
UMD_ZIP = (FIXTURE_DIR / "french" / "umd.zip").read_bytes()
FF5_SHA = sha256_bytes(FF5_ZIP)
UMD_SHA = sha256_bytes(UMD_ZIP)


def make_config(
    root: Path,
    *,
    ff5_sha256: str | None = None,
    umd_sha256: str | None = None,
    downloaded_on: str = "2026-10-16",
) -> DataConfig:
    return DataConfig(
        root=root,
        osap=OsapConfig(
            vintage=202510,
            portfolios=("op",),
            raw_dir=root / "data" / "raw",
        ),
        french=FrenchConfig(
            ff5_url="https://example.invalid/ff5.zip",
            umd_url="https://example.invalid/umd.zip",
            pinned_dir=root / "data" / "pinned",
            downloaded_on=downloaded_on,
            ff5_sha256=ff5_sha256 or sha256_bytes(FF5_ZIP),
            umd_sha256=umd_sha256 or sha256_bytes(UMD_ZIP),
        ),
    )


def signal_doc_frame(rows: int = 150) -> pl.DataFrame:
    acronyms = ["Size", "Mom12m"] + [f"Filler{i:03d}" for i in range(rows - 2)]
    return pl.DataFrame(
        {
            "Acronym": acronyms,
            "Cat.Signal": ["Predictor"] * rows,
            "SampleStartYear": [1963, 1965] + [1970] * (rows - 2),
            "SampleEndYear": [2001, 2000] + [2010] * (rows - 2),
            "Year": [1981, 1993] + [2005] * (rows - 2),
            "Journal": ["JFE", "JF"] + ["JF"] * (rows - 2),
        }
    )


def port_frame(rows: int = 2000) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "signalname": ["Size"] * rows,
            "port": ["LS"] * rows,
            "date": pl.date_range(
                pl.date(1963, 7, 1), pl.date(1963, 7, 1), interval="1d", eager=True
            ).to_list()
            * rows,
            "ret": [0.5] * rows,
            "signallag": [1] * rows,
            "Nlong": [100] * rows,
            "Nshort": [100] * rows,
        }
    )


class FakeOpenAP:
    """Mimics the parts of openassetpricing.OpenAP that fetch_osap uses."""

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
            raise ConnectionError("simulated Drive failure")

    def dl_signal_doc(self, df_backend: str) -> pl.DataFrame:
        self._maybe_fail("doc")
        return self.doc

    def dl_port(self, data_name: str, df_backend: str) -> pl.DataFrame:
        self._maybe_fail(data_name)
        return self.ports[data_name]


class FakeResponse:
    def __init__(self, content: bytes, content_type: str = "application/zip") -> None:
        self.content = content
        self.headers = {"content-type": content_type}

    def raise_for_status(self) -> None:
        return None


class FakeSession:
    def __init__(self, responses: dict[str, bytes]) -> None:
        self.responses = responses

    def get(self, url: str, timeout: float | None = None) -> FakeResponse:
        if url not in self.responses:
            raise ConnectionError(f"no fake response for {url}")
        return FakeResponse(self.responses[url])


# --- OSAP fetch -------------------------------------------------------------


def test_fetch_osap_writes_canonical_files_under_vintage(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    fake = FakeOpenAP(doc=signal_doc_frame(), ports={"op": port_frame()})
    written = fetch_osap(cfg, openap_factory=lambda vintage: fake, sleep=lambda _: None)

    assert written == [
        cfg.osap.raw_dir / "202510" / "signal_doc.csv",
        cfg.osap.raw_dir / "202510" / "port_op.csv",
    ]
    parsed = pl.read_csv(cfg.osap.raw_dir / "202510" / "port_op.csv")
    assert parsed.columns == list(PORT_COLUMNS)
    assert parsed.height == 2000
    doc = pl.read_csv(cfg.osap.raw_dir / "202510" / "signal_doc.csv")
    assert set(doc["Acronym"].to_list()) >= {"Size", "Mom12m"}
    assert doc.height == 150


def test_fetch_osap_quota_like_response_raises_fetch_error(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    bad = pl.DataFrame({"Acronym": ["Size"]})  # what a quota HTML page can degrade to
    fake = FakeOpenAP(doc=signal_doc_frame(), ports={"op": bad})
    with pytest.raises(FetchError, match="quota"):
        fetch_osap(cfg, openap_factory=lambda vintage: fake, sleep=lambda _: None)
    assert fake.calls.count("op") == MAX_ATTEMPTS
    assert not (cfg.osap.raw_dir / "202510" / "port_op.csv").exists()


def test_fetch_osap_retries_with_exponential_backoff(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    fake = FakeOpenAP(doc=signal_doc_frame(), ports={"op": port_frame()}, fail_times=2)
    sleeps: list[float] = []
    fetch_osap(cfg, openap_factory=lambda vintage: fake, sleep=sleeps.append)
    assert fake.calls == ["doc", "doc", "doc", "op"]
    assert sleeps == [1.0, 2.0]


def test_fetch_osap_exhausts_retries(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    fake = FakeOpenAP(doc=signal_doc_frame(), ports={"op": port_frame()}, fail_times=99)
    sleeps: list[float] = []
    with pytest.raises(FetchError, match=f"failed after {MAX_ATTEMPTS} attempts"):
        fetch_osap(cfg, openap_factory=lambda vintage: fake, sleep=sleeps.append)
    assert len(sleeps) == MAX_ATTEMPTS - 1
    assert sleeps == [1.0, 2.0, 4.0, 8.0]


def test_fetch_osap_retries_the_open_call(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    attempts: list[int] = []

    def factory(vintage: int) -> FakeOpenAP:
        attempts.append(vintage)
        if len(attempts) < 3:
            raise ConnectionError("simulated Drive failure")
        return FakeOpenAP(doc=signal_doc_frame(), ports={"op": port_frame()})

    fetch_osap(cfg, openap_factory=factory, sleep=lambda _: None)
    assert attempts == [202510, 202510, 202510]


# --- Ken French fetch -------------------------------------------------------


def test_fetch_french_verifies_hashes_parses_and_caches_zip(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    session = FakeSession(
        {"https://example.invalid/ff5.zip": FF5_ZIP, "https://example.invalid/umd.zip": UMD_ZIP}
    )
    written = fetch_french(cfg, session=session, sleep=lambda _: None)  # type: ignore[arg-type]
    assert written == [
        cfg.french.pinned_dir / "ff5_monthly.csv",
        cfg.french.pinned_dir / "umd_monthly.csv",
    ]
    ff5 = pl.read_csv(cfg.french.pinned_dir / "ff5_monthly.csv")
    assert ff5.columns == ["yyyymm", "mkt_rf", "smb", "hml", "rmw", "cma", "rf"]
    assert ff5.height == 758
    cached = cfg.osap.raw_dir / "ff" / "2026-10-16" / "ff5.zip"
    assert cached.read_bytes() == FF5_ZIP


def test_fetch_french_hash_mismatch_raises(tmp_path: Path) -> None:
    cfg = make_config(tmp_path, ff5_sha256="0" * 64)
    session = FakeSession(
        {"https://example.invalid/ff5.zip": FF5_ZIP, "https://example.invalid/umd.zip": UMD_ZIP}
    )
    with pytest.raises(FetchError, match="refresh-french"):
        fetch_french(cfg, session=session, sleep=lambda _: None)  # type: ignore[arg-type]


def test_fetch_french_non_zip_response_raises(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    session = FakeSession(
        {
            "https://example.invalid/ff5.zip": b"<html>quota exceeded</html>",
            "https://example.invalid/umd.zip": UMD_ZIP,
        }
    )
    with pytest.raises(FetchError, match="not a zip"):
        fetch_french(cfg, session=session, sleep=lambda _: None)  # type: ignore[arg-type]


def test_fetch_french_refresh_updates_config(tmp_path: Path) -> None:
    zeros = "0" * 64
    cfg = make_config(tmp_path, ff5_sha256=zeros, umd_sha256=zeros)
    config_path = tmp_path / "config" / "data.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        (
            "[osap]\n"
            'vintage = 202510\nportfolios = ["op"]\nraw_dir = "data/raw"\n\n'
            "[french]\n"
            'ff5_url = "https://example.invalid/ff5.zip"\n'
            'umd_url = "https://example.invalid/umd.zip"\n'
            'pinned_dir = "data/pinned"\n'
            'downloaded_on = "2026-01-01"\n'
            f'ff5_sha256 = "{zeros}"\n'
            f'umd_sha256 = "{zeros}"\n'
        ),
        encoding="utf-8",
    )
    session = FakeSession(
        {"https://example.invalid/ff5.zip": FF5_ZIP, "https://example.invalid/umd.zip": UMD_ZIP}
    )
    fetch_french(cfg, refresh=True, session=session, sleep=lambda _: None)  # type: ignore[arg-type]
    text = config_path.read_text(encoding="utf-8")
    assert f'ff5_sha256 = "{sha256_bytes(FF5_ZIP)}"' in text
    assert f'umd_sha256 = "{sha256_bytes(UMD_ZIP)}"' in text
    assert f'downloaded_on = "{time.strftime("%Y-%m-%d")}"' in text
    reloaded = load_data_config(config_path)
    assert reloaded.french.ff5_sha256 == sha256_bytes(FF5_ZIP)
    assert reloaded.french.downloaded_on == time.strftime("%Y-%m-%d")


# --- manifest ---------------------------------------------------------------


def test_manifest_roundtrip_and_tamper_detection(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    (cfg.french.pinned_dir).mkdir(parents=True)
    held = cfg.french.pinned_dir / "ff5_monthly.csv"
    held.write_bytes(b"yyyymm,mkt_rf\n196307,-0.39\n")
    osap_file = cfg.osap.raw_dir / "202510" / "port_op.csv"
    osap_file.parent.mkdir(parents=True)
    osap_file.write_bytes(b"signalname,port\nSize,LS\n")

    manifest = write_manifest(cfg, "all")
    assert manifest == tmp_path / "data" / "manifest.lock"
    lines = manifest.read_text(encoding="utf-8").splitlines()
    assert lines == sorted(lines, key=lambda line: line.split("  ")[1])
    assert all(len(line.split("  ")) == 3 for line in lines)

    result = verify_manifest(manifest, cfg.root)
    assert result.ok
    assert result.checked == 2

    held.write_bytes(b"tampered")
    result = verify_manifest(manifest, cfg.root)
    assert not result.ok
    assert result.mismatches == ("data/pinned/ff5_monthly.csv",)


def test_missing_osap_cache_is_not_cached_not_a_failure(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    (cfg.french.pinned_dir).mkdir(parents=True)
    (cfg.french.pinned_dir / "ff5_monthly.csv").write_bytes(b"a,b\n1,2\n")
    osap_file = cfg.osap.raw_dir / "202510" / "port_op.csv"
    osap_file.parent.mkdir(parents=True)
    osap_file.write_bytes(b"x\n")
    manifest = write_manifest(cfg, "all")

    osap_file.unlink()
    result = verify_manifest(manifest, cfg.root)
    assert result.ok
    assert result.not_cached == ("data/raw/202510/port_op.csv",)
    assert result.missing == ()


def test_missing_pinned_file_fails(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    (cfg.french.pinned_dir).mkdir(parents=True)
    held = cfg.french.pinned_dir / "ff5_monthly.csv"
    held.write_bytes(b"a,b\n1,2\n")
    manifest = write_manifest(cfg, "all")
    held.unlink()
    result = verify_manifest(manifest, cfg.root)
    assert not result.ok
    assert result.missing == ("data/pinned/ff5_monthly.csv",)


def test_write_manifest_preserves_absent_entries(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    (cfg.french.pinned_dir).mkdir(parents=True)
    (cfg.french.pinned_dir / "ff5_monthly.csv").write_bytes(b"a,b\n1,2\n")
    osap_file = cfg.osap.raw_dir / "202510" / "port_op.csv"
    osap_file.parent.mkdir(parents=True)
    osap_file.write_bytes(b"x\n")
    write_manifest(cfg, "all")
    osap_file.unlink()
    rewritten = write_manifest(cfg, "all")
    result = verify_manifest(rewritten, cfg.root)
    assert result.not_cached == ("data/raw/202510/port_op.csv",)


# --- CLI --------------------------------------------------------------------


def _write_config_file(root: Path, cfg: DataConfig) -> Path:
    config_path = root / "config" / "data.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        (
            "[osap]\n"
            f"vintage = {cfg.osap.vintage}\n"
            'portfolios = ["op"]\n'
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


def test_verify_cli_without_osap_cache_prints_all_hashes_match(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = make_config(tmp_path)
    config_path = _write_config_file(tmp_path, cfg)
    (cfg.french.pinned_dir).mkdir(parents=True)
    (cfg.french.pinned_dir / "ff5_monthly.csv").write_bytes(b"a,b\n1,2\n")
    osap_file = cfg.osap.raw_dir / "202510" / "port_op.csv"
    osap_file.parent.mkdir(parents=True)
    osap_file.write_bytes(b"x\n")
    write_manifest(cfg, "all")
    osap_file.unlink()

    exit_code = main(["--verify", "--config", str(config_path)])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "not cached (run make fetch): data/raw/202510/port_op.csv" in out
    assert out.splitlines()[-1] == "all hashes match"


def test_verify_cli_reports_mismatch_with_nonzero_exit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = make_config(tmp_path)
    config_path = _write_config_file(tmp_path, cfg)
    (cfg.french.pinned_dir).mkdir(parents=True)
    held = cfg.french.pinned_dir / "ff5_monthly.csv"
    held.write_bytes(b"a,b\n1,2\n")
    write_manifest(cfg, "all")
    held.write_bytes(b"tampered")

    exit_code = main(["--verify", "--config", str(config_path)])
    out = capsys.readouterr().out
    assert exit_code == 1
    assert "mismatch: data/pinned/ff5_monthly.csv" in out
    assert "all hashes match" not in out


# --- config loader ----------------------------------------------------------


def test_load_data_config_reads_pinned_values(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    config_path = _write_config_file(tmp_path, cfg)
    loaded = load_data_config(config_path)
    assert loaded.root == tmp_path
    assert loaded.osap.vintage == 202510
    assert loaded.osap.portfolios == ("op",)
    assert loaded.osap.raw_dir == tmp_path / "data" / "raw"
    assert loaded.french.ff5_sha256 == sha256_bytes(FF5_ZIP)
    assert loaded.french.pinned_dir == tmp_path / "data" / "pinned"


@pytest.mark.parametrize(
    ("drop", "match"),
    [
        ("vintage = 202510\n", "vintage"),
        ('downloaded_on = "2026-10-16"\n', "downloaded_on"),
        (f'ff5_sha256 = "{FF5_SHA}"\n', "ff5_sha256"),
    ],
)
def test_load_data_config_missing_key_raises(tmp_path: Path, drop: str, match: str) -> None:
    cfg = make_config(tmp_path)
    config_path = _write_config_file(tmp_path, cfg)
    text = config_path.read_text(encoding="utf-8").replace(drop, "", 1)
    config_path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match=match):
        load_data_config(config_path)


def test_load_data_config_bad_hash_raises(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    config_path = _write_config_file(tmp_path, cfg)
    text = config_path.read_text(encoding="utf-8").replace(cfg.french.ff5_sha256, "nothex", 1)
    config_path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="SHA-256"):
        load_data_config(config_path)


def test_load_data_config_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_data_config(tmp_path / "nope.toml")


def test_openap_factory_returning_broken_api_is_retried(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    calls: list[int] = []

    def broken_factory(vintage: int) -> Any:
        calls.append(vintage)
        raise RuntimeError("Drive scrape failed")

    with pytest.raises(FetchError, match="open OSAP release 202510"):
        fetch_osap(cfg, openap_factory=broken_factory, sleep=lambda _: None)
    assert len(calls) == MAX_ATTEMPTS
