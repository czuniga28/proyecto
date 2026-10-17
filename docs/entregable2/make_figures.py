"""Regenerate the figures of Entregable 2 from the pipeline outputs.

Run from proyecto/:  python -m docs.entregable2.make_figures
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.io import load_volume
from src.data.transforms import normalize_volume

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figs"
RAW = ROOT / "datasets/raw/volumetric_data"

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3de"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "legend.frameon": False,
})


def normalization_figure(manifest: pd.DataFrame) -> None:
    # Brightest-typical vs dimmest volume: shows why the z-score is per volume
    typical = manifest.iloc[(manifest.intensity_max - manifest.intensity_max.median()).abs().argmin()]
    dim = manifest.loc[manifest.intensity_max.idxmin()]
    cases = [(typical, BLUE, f"Típico (máx. {typical.intensity_max})"),
             (dim, ORANGE, f"Tenue (máx. {dim.intensity_max})")]

    fig, (ax_raw, ax_norm) = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for row, color, label in cases:
        raw = load_volume(RAW / row.path)
        norm = normalize_volume(raw, 0.5, 99.5)
        ax_raw.hist(raw.ravel(), bins=120, range=(0, 3000), density=True,
                    histtype="step", linewidth=2, color=color, label=label)
        ax_norm.hist(norm.ravel(), bins=120, range=(-2, 6), density=True,
                     histtype="step", linewidth=2, color=color, label=label)

    ax_raw.set(title="Intensidad cruda (uint16)", xlabel="Intensidad", ylabel="Densidad")
    ax_norm.set(title="Tras recorte P0.5–P99.5 y puntaje z", xlabel="Intensidad normalizada")
    for ax in (ax_raw, ax_norm):
        ax.set_yscale("log")
        ax.title.set_color(INK)
    ax_raw.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "normalizacion.pdf")
    fig.savefig(OUT / "normalizacion.png", dpi=200)


def inspection_grid(slices: pd.DataFrame, per_class: int = 10, seed: int = 0) -> pd.DataFrame:
    """Central positive slice of `per_class` random volumes per diagnosis, with its box."""
    from matplotlib.patches import Rectangle

    pos = slices[slices.contains_acl].sort_values(['examId', 'seriesNo', 'slice_idx'])
    # Middle positive slice of each volume
    rank = pos.groupby(['examId', 'seriesNo']).cumcount()
    size = pos.groupby(['examId', 'seriesNo'])['slice_idx'].transform('size')
    mid = pos[rank == size // 2]
    sample = (mid.groupby('aclDiagnosis').sample(per_class, random_state=seed)
                 .sort_values(['aclDiagnosis', 'examId']).reset_index(drop=True))

    fig, axes = plt.subplots(6, 5, figsize=(7.0, 8.6))
    names = {0: "sano", 1: "parcial", 2: "roto"}
    for ax, (_, r) in zip(axes.ravel(), sample.iterrows()):
        vol = np.load(ROOT / "datasets/processed" / r.volume_file)
        ax.imshow(vol[int(r.slice_idx)].astype(np.float32), cmap="gray", vmin=-2, vmax=4)
        ax.add_patch(Rectangle((r.x0, r.y0), r.x1 - r.x0, r.y1 - r.y0,
                               fill=False, edgecolor=ORANGE, linewidth=1.2))
        ax.set_title(f"{names[r.aclDiagnosis]} · {r.examId}-{r.seriesNo} · z={r.slice_idx}",
                     fontsize=6, color=INK)
        ax.set_axis_off()
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "inspeccion_30.png", dpi=220)
    return sample[['examId', 'seriesNo', 'aclDiagnosis', 'slice_idx']]


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(ROOT / "datasets/interim/manifest.csv")
    normalization_figure(manifest)
    slices = pd.read_csv(ROOT / "datasets/processed/slices.csv")
    print(inspection_grid(slices).to_string(index=False))
