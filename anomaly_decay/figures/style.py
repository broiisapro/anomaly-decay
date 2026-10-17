"""Shared figure styling and byte-deterministic saving.

Every figure module imports from here so that the committed files under
``results/figures/`` are reproducible byte for byte: the Agg backend is selected
before pyplot is used, one font family bundled with Matplotlib is pinned, the SVG
hash salt is fixed (otherwise the element ids change per process), and date/tool
metadata is suppressed at save time.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import rcParams
from matplotlib.figure import Figure

__all__ = [
    "FONT_FAMILY",
    "PALETTE",
    "PNG_DPI",
    "apply_style",
    "save_figure",
]

FONT_FAMILY = "DejaVu Sans"
"""Font bundled with Matplotlib, so macOS and the CI runners render identically."""

PALETTE: tuple[str, ...] = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
)
"""Colour-blind-safe palette (Okabe and Ito), used in order."""

PNG_DPI = 200
"""Resolution of the committed PNG figures."""

_SVG_METADATA = {"Date": None, "Creator": None}
_PNG_METADATA = {"Software": None}


def apply_style() -> None:
    """Apply the shared, deterministic Matplotlib defaults.

    Idempotent; call before building any figure. Sets the pinned font family,
    the fixed SVG hash salt, and the shared sizes used by every panel.
    """
    rcParams["font.family"] = FONT_FAMILY
    rcParams["font.size"] = 10
    rcParams["axes.labelsize"] = 10
    rcParams["axes.titlesize"] = 11
    rcParams["xtick.labelsize"] = 9
    rcParams["ytick.labelsize"] = 9
    rcParams["legend.fontsize"] = 8
    rcParams["axes.grid"] = True
    rcParams["grid.alpha"] = 0.25
    rcParams["grid.linewidth"] = 0.6
    rcParams["axes.spines.top"] = False
    rcParams["axes.spines.right"] = False
    rcParams["svg.hashsalt"] = "anomaly-decay"


def save_figure(fig: Figure, out_dir: Path, stem: str) -> list[Path]:
    """Write *fig* as ``<stem>.png`` (200 dpi) and ``<stem>.svg`` under *out_dir*.

    Returns ``[png_path, svg_path]``. Date and tool metadata are suppressed so
    successive renders are byte-identical.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{stem}.png"
    svg_path = out_dir / f"{stem}.svg"
    fig.savefig(png_path, dpi=PNG_DPI, metadata=_PNG_METADATA)
    fig.savefig(svg_path, metadata=_SVG_METADATA)
    return [png_path, svg_path]
