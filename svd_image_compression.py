#!/usr/bin/env python3
"""
Singular Value Decomposition (SVD) and Image Compression
=========================================================

Applied Linear Algebra project — complete, self-contained pipeline.

What it does
------------
1. Loads a grayscale image and an RGB image (your own files, or the standard
   512x512 test images "cameraman" and "astronaut" from scikit-image).
2. Computes a full SVD with numpy.linalg.svd and builds rank-k truncations:
     - one matrix for grayscale,
     - three independent matrices (R, G, B) for colour.
3. For every k it reports:
     - storage: k(M+N+1) values per channel vs. M*N,
     - compression ratio and storage savings (%),
     - MSE and PSNR (dB) on the 8-bit reconstruction,
     - cumulative energy  sum_{i<=k} s_i^2 / sum_i s_i^2,
     - byte cost if the factors are stored as float32 / float16 / uint8.
4. Verifies the Eckart-Young-Mirsky identity ||A - A_k||_F^2 = sum_{i>k} s_i^2.
5. Saves two publication-ready figures, every reconstruction as PNG, and a
   CSV of all metrics.

Run
---
    python svd_image_compression.py                       # built-in test images
    python svd_image_compression.py --gray my.png --rgb photo.jpg --ks 5 20 50 100
    python svd_image_compression.py --size 512            # resize inputs to 512x512

In Jupyter: paste the file into a cell and call  main([])  or
main(["--gray", "my.png"]).

Dependencies: numpy, scipy, matplotlib, pillow  (scikit-image optional, only
for the built-in test images).
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass, asdict

import numpy as np
import matplotlib

if not os.environ.get("DISPLAY") and "ipykernel" not in os.sys.modules:
    matplotlib.use("Agg")  # headless-safe; Jupyter keeps its inline backend
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from PIL import Image

DEFAULT_KS = (5, 20, 50, 100)
MAX_PIXEL = 255.0


# --------------------------------------------------------------------------
# 1. Image loading
# --------------------------------------------------------------------------
def _resize(arr: np.ndarray, size: int | None) -> np.ndarray:
    if size is None:
        return arr
    img = Image.fromarray(arr)
    return np.asarray(img.resize((size, size), Image.LANCZOS))


def _synthetic(shape: tuple[int, ...]) -> np.ndarray:
    """Deterministic fallback test pattern (gradients + disc + text-like bars)."""
    m, n = shape[:2]
    y, x = np.mgrid[0:m, 0:n] / max(m, n)
    base = 0.5 + 0.25 * np.sin(8 * np.pi * x) * np.cos(6 * np.pi * y)
    disc = ((x - 0.6) ** 2 + (y - 0.4) ** 2) < 0.04
    bars = (np.floor(x * 40) % 2 == 0) & (y > 0.75) & (y < 0.85)
    g = np.clip(base + 0.3 * disc - 0.3 * bars, 0, 1)
    if len(shape) == 2:
        return (g * 255).astype(np.uint8)
    rgb = np.stack([g, np.roll(g, m // 7, 0), np.roll(g, n // 5, 1)], axis=-1)
    return (rgb * 255).astype(np.uint8)


def load_grayscale(path: str | None, size: int | None = None) -> tuple[np.ndarray, str]:
    if path:
        arr = np.asarray(Image.open(path).convert("L"))
        return _resize(arr, size), os.path.splitext(os.path.basename(path))[0]
    try:
        from skimage import data
        return _resize(data.camera(), size), "cameraman"
    except Exception:
        return _synthetic((size or 512, size or 512)), "synthetic_gray"


def load_rgb(path: str | None, size: int | None = None) -> tuple[np.ndarray, str]:
    if path:
        arr = np.asarray(Image.open(path).convert("RGB"))
        return _resize(arr, size), os.path.splitext(os.path.basename(path))[0]
    try:
        from skimage import data
        return _resize(data.astronaut(), size), "astronaut"
    except Exception:
        return _synthetic((size or 512, size or 512, 3)), "synthetic_rgb"


# --------------------------------------------------------------------------
# 2. SVD and truncated reconstruction
# --------------------------------------------------------------------------
@dataclass
class ChannelSVD:
    U: np.ndarray      # M x r
    s: np.ndarray      # r
    Vt: np.ndarray     # r x N

    @classmethod
    def of(cls, A: np.ndarray) -> "ChannelSVD":
        U, s, Vt = np.linalg.svd(A.astype(np.float64), full_matrices=False)
        return cls(U, s, Vt)

    def reconstruct(self, k: int) -> np.ndarray:
        """A_k = U_k Sigma_k V_k^T  (float, unclipped)."""
        k = min(k, self.s.size)
        return (self.U[:, :k] * self.s[:k]) @ self.Vt[:k, :]

    def energy(self, k: int) -> float:
        e = self.s ** 2
        return float(e[:k].sum() / e.sum())

    def tail_energy(self, k: int) -> float:
        """Eckart-Young-Mirsky: ||A - A_k||_F^2 = sum_{i>k} s_i^2."""
        return float((self.s[k:] ** 2).sum())


def decompose(img: np.ndarray) -> list[ChannelSVD]:
    """One SVD for a 2-D image, three independent SVDs for an H x W x 3 image."""
    if img.ndim == 2:
        return [ChannelSVD.of(img)]
    return [ChannelSVD.of(img[:, :, c]) for c in range(img.shape[2])]


def reconstruct(svds: list[ChannelSVD], k: int, is_rgb: bool) -> tuple[np.ndarray, np.ndarray]:
    """Returns (float reconstruction, uint8 displayable reconstruction)."""
    planes = [ch.reconstruct(k) for ch in svds]
    raw = np.stack(planes, axis=-1) if is_rgb else planes[0]
    return raw, np.clip(np.rint(raw), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# 3. Metrics
# --------------------------------------------------------------------------
@dataclass
class Metrics:
    image: str
    mode: str
    M: int
    N: int
    channels: int
    k: int
    original_values: int          # M*N*C
    compressed_values: int        # k(M+N+1)*C
    compression_ratio: float      # original / compressed
    storage_savings_pct: float    # 1 - compressed/original
    original_bytes_uint8: int
    compressed_bytes_float32: int
    compressed_bytes_float16: int
    mse: float
    psnr_db: float
    energy_fraction: float        # mean over channels
    ey_mse_predicted: float       # tail energy / (M N C), before clipping
    ey_identity_error: float      # | ||A-A_k||_F^2 - sum_{i>k} s_i^2 | / ||A||_F^2
    jpeg_quality_same_psnr: int = -1   # lowest JPEG quality reaching >= this PSNR
    jpeg_bytes_same_psnr: int = -1     # its file size (practical baseline)


def storage_values(M: int, N: int, k: int, C: int = 1) -> tuple[int, int]:
    return M * N * C, k * (M + N + 1) * C


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def psnr(m: float, peak: float = MAX_PIXEL) -> float:
    return float("inf") if m == 0 else float(10 * np.log10(peak ** 2 / m))


def evaluate(img: np.ndarray, name: str, svds: list[ChannelSVD], k: int) -> tuple[Metrics, np.ndarray]:
    is_rgb = img.ndim == 3
    M, N = img.shape[:2]
    C = img.shape[2] if is_rgb else 1
    k = max(1, min(int(k), min(M, N)))  # a rank cannot exceed min(M, N)
    raw, rec = reconstruct(svds, k, is_rgb)

    orig_vals, comp_vals = storage_values(M, N, k, C)
    m = mse(img, rec)

    # Eckart-Young-Mirsky check on the unclipped float reconstruction
    resid = float(np.sum((img.astype(np.float64) - raw) ** 2))
    tail = sum(ch.tail_energy(k) for ch in svds)
    total = float(np.sum(img.astype(np.float64) ** 2))

    met = Metrics(
        image=name, mode="RGB" if is_rgb else "Grayscale", M=M, N=N, channels=C, k=k,
        original_values=orig_vals, compressed_values=comp_vals,
        compression_ratio=orig_vals / comp_vals,
        storage_savings_pct=100 * (1 - comp_vals / orig_vals),
        original_bytes_uint8=orig_vals,
        compressed_bytes_float32=4 * comp_vals,
        compressed_bytes_float16=2 * comp_vals,
        mse=m, psnr_db=psnr(m),
        energy_fraction=float(np.mean([ch.energy(k) for ch in svds])),
        ey_mse_predicted=tail / (M * N * C),
        ey_identity_error=abs(resid - tail) / total,
    )
    return met, rec


def jpeg_baseline(img: np.ndarray, target_psnr: float) -> tuple[int, int]:
    """Smallest JPEG (quality 5..100) whose PSNR >= target. Returns (quality, bytes)."""
    import io
    for q in range(5, 101):
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, "JPEG", quality=q)
        dec = np.asarray(Image.open(io.BytesIO(buf.getvalue())))
        if psnr(mse(img, dec)) >= target_psnr:
            return q, buf.tell()
    return -1, -1


def breakeven_rank(M: int, N: int, bytes_per_value: float = 1.0) -> float:
    """Rank above which the truncated factors cost more than the 8-bit image."""
    return M * N / ((M + N + 1) * bytes_per_value)


# --------------------------------------------------------------------------
# 4. Figures
# --------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200,
})
GRAY, ACCENT, ACCENT2 = "#6b7280", "#1f5fbf", "#c2410c"


def comparison_grid(rows: list[tuple[np.ndarray, str, list[tuple[Metrics, np.ndarray]]]],
                    path: str) -> None:
    """Original + one reconstruction per k, one row per image."""
    ncols = 1 + len(rows[0][2])
    fig, axes = plt.subplots(len(rows), ncols, figsize=(2.7 * ncols, 3.3 * len(rows)),
                             squeeze=False, gridspec_kw={"hspace": 0.32, "wspace": 0.06})
    for r, (img, label, results) in enumerate(rows):
        cmap = "gray" if img.ndim == 2 else None
        ax = axes[r, 0]
        ax.imshow(img, cmap=cmap, vmin=0, vmax=255)
        ax.set_title(f"Original ({label})\n{img.shape[1]}×{img.shape[0]}, full rank",
                     fontweight="bold")
        for c, (met, rec) in enumerate(results, start=1):
            ax = axes[r, c]
            ax.imshow(rec, cmap=cmap, vmin=0, vmax=255)
            ax.set_title(f"k = {met.k}\nPSNR {met.psnr_db:.2f} dB · "
                         f"{met.storage_savings_pct:.1f}% saved", fontsize=9)
        for ax in axes[r]:
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
    fig.suptitle("Rank-k SVD reconstructions (title: k · PSNR · coefficient storage saved)",
                 fontsize=12, fontweight="bold", y=0.99)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def metric_panels(curves: list[dict], ks: list[int], path: str) -> None:
    """Left: singular-value spectrum + cumulative energy. Right: PSNR vs compression."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 5.2))
    styles = ["-", "--"]

    # Left panel — spectrum (log-log) and cumulative energy (twin axis)
    axE = axL.twinx()
    axE.spines["right"].set_visible(True)
    for i, cv in enumerate(curves):
        s = cv["s"]
        idx = np.arange(1, s.size + 1)
        axL.loglog(idx, s / s[0], color=GRAY, ls=styles[i % 2], lw=1.2,
                   label=f"σᵢ/σ₁ — {cv['label']}")
        cum = np.cumsum(s ** 2) / np.sum(s ** 2)
        axE.semilogx(idx, cum, color=ACCENT, ls=styles[i % 2], lw=1.8,
                     label=f"cumulative energy — {cv['label']}")
    for k in ks:
        axL.axvline(k, color="#d1d5db", lw=0.8, zorder=0)
        axL.text(k, 1.02, f"k={k}", ha="center", va="bottom", fontsize=8, color=GRAY,
                 transform=axL.get_xaxis_transform())
    axL.set_xlabel("singular value index i (log)")
    axL.set_ylabel("normalised singular value σᵢ / σ₁ (log)")
    axE.set_ylabel("cumulative energy Σᵢ≤ₖ σᵢ² / Σ σᵢ²", color=ACCENT)
    axE.set_ylim(0.75, 1.0)
    axE.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axL.set_xlim(1, max(cv["s"].size for cv in curves))
    axL.set_title("Singular value spectrum and energy captured", fontweight="bold", pad=18)
    h1, l1 = axL.get_legend_handles_labels(); h2, l2 = axE.get_legend_handles_labels()
    axL.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.14),
               ncol=2, fontsize=8, frameon=False)

    # Right panel — PSNR vs compression ratio (dense k sweep up to value break-even)
    for i, cv in enumerate(curves):
        colour = ACCENT if i == 0 else ACCENT2
        axR.plot(cv["cr"], cv["psnr"], color=colour, lw=1.8, ls=styles[i % 2],
                 label=cv["label"])
        for k in ks:
            j = int(np.flatnonzero(cv["k"] == k)[0])
            axR.plot(cv["cr"][j], cv["psnr"][j], "o", color=colour, ms=5)
            axR.annotate(f"k={k}", (cv["cr"][j], cv["psnr"][j]), textcoords="offset points",
                         xytext=(-4, 7) if i == 0 else (4, -13), fontsize=8, color=colour,
                         ha="right" if i == 0 else "left")
    axR.set_xscale("log")
    axR.invert_xaxis()
    axR.axhspan(30, 40, color="#ecfdf5", zorder=0)
    axR.text(0.02, 35, "30–40 dB: typical acceptable web quality", ha="left", va="center",
             fontsize=8, color="#047857", transform=axR.get_yaxis_transform())
    for cr, lab, yy in ((4.0, "float32 factors = raw 8-bit size (CR 4)", 0.10),
                        (2.0, "float16 factors = raw 8-bit size (CR 2)", 0.03)):
        axR.axvline(cr, color=GRAY, lw=0.8, ls=":")
        axR.text(cr, yy, lab + " ", fontsize=7.5, color=GRAY, ha="right",
                 transform=axR.get_xaxis_transform())
    axR.set_xlabel("compression ratio  M·N / k(M+N+1)  (log; ← more compression)")
    axR.set_ylabel("PSNR (dB)")
    axR.set_title("Rate–distortion: PSNR vs compression ratio", fontweight="bold", pad=18)
    axR.legend(frameon=False, fontsize=9, loc="upper left")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# 5. Output helpers
# --------------------------------------------------------------------------
def write_csv(rows: list[Metrics], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for m in rows:
            d = asdict(m)
            for key, v in d.items():
                if isinstance(v, float):
                    d[key] = f"{v:.6g}"
            w.writerow(d)


def print_table(rows: list[Metrics]) -> None:
    hdr = (f"{'image':<12}{'k':>5}{'values':>10}{'CR':>8}{'saved%':>8}"
           f"{'MSE':>10}{'PSNR':>8}{'energy%':>10}{'EY err':>10}{'f16 B':>9}{'JPEG B':>9}")
    print(hdr); print("-" * len(hdr))
    for m in rows:
        print(f"{m.image:<12}{m.k:>5}{m.compressed_values:>10}{m.compression_ratio:>8.2f}"
              f"{m.storage_savings_pct:>8.2f}{m.mse:>10.2f}{m.psnr_db:>8.2f}"
              f"{100 * m.energy_fraction:>10.3f}{m.ey_identity_error:>10.1e}"
              f"{m.compressed_bytes_float16:>9}{m.jpeg_bytes_same_psnr:>9}")


def sweep(img: np.ndarray, name: str, svds: list[ChannelSVD], k_max: int,
          must_include: list[int] = ()) -> dict:
    ks = np.unique(np.concatenate([np.round(np.geomspace(1, k_max, 60)).astype(int),
                                   np.asarray(must_include, dtype=int)]))
    psnrs, crs = [], []
    for k in ks:
        met, _ = evaluate(img, name, svds, int(k))
        psnrs.append(met.psnr_db); crs.append(met.compression_ratio)
    s = np.mean([ch.s for ch in svds], axis=0) if len(svds) > 1 else svds[0].s
    return {"label": name, "k": ks, "psnr": np.array(psnrs), "cr": np.array(crs), "s": s}


# --------------------------------------------------------------------------
# 6. Main
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> list[Metrics]:
    p = argparse.ArgumentParser(description="SVD image compression study")
    p.add_argument("--gray", help="path to grayscale image (any format; converted to L)")
    p.add_argument("--rgb", help="path to colour image")
    p.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    p.add_argument("--size", type=int, default=None, help="optional square resize, e.g. 512")
    p.add_argument("--out", default="svd_outputs")
    p.add_argument("--show", action="store_true", help="also display figures")
    p.add_argument("--no-jpeg", action="store_true", help="skip the JPEG baseline")
    args, _ = p.parse_known_args(argv)  # tolerant of Jupyter's own argv
    if any(k < 1 for k in args.ks):
        p.error("every k must be at least 1 (rank 0 is just a black image)")

    os.makedirs(os.path.join(args.out, "reconstructions"), exist_ok=True)
    gray, gname = load_grayscale(args.gray, args.size)
    rgb, cname = load_rgb(args.rgb, args.size)
    print(f"Grayscale: {gname} {gray.shape}   RGB: {cname} {rgb.shape}")

    all_metrics: list[Metrics] = []
    grid_rows, curves = [], []
    for img, name in ((gray, gname), (rgb, cname)):
        svds = decompose(img)
        results = []
        for k in args.ks:
            met, rec = evaluate(img, name, svds, k)
            if not args.no_jpeg:
                met.jpeg_quality_same_psnr, met.jpeg_bytes_same_psnr = jpeg_baseline(img, met.psnr_db)
            results.append((met, rec))
            all_metrics.append(met)
            Image.fromarray(rec).save(
                os.path.join(args.out, "reconstructions", f"{name}_k{k:03d}.png"))
        grid_rows.append((img, name, results))
        M, N = img.shape[:2]
        curves.append(sweep(img, name, svds, int(breakeven_rank(M, N)), args.ks))
        print(f"\n{name}: break-even rank vs 8-bit pixels — "
              f"{breakeven_rank(M, N, 1):.1f} values/uint8, "
              f"{breakeven_rank(M, N, 2):.1f} @float16, {breakeven_rank(M, N, 4):.1f} @float32")

    print()
    print_table(all_metrics)

    comparison_grid(grid_rows, os.path.join(args.out, "fig1_comparison_grid.png"))
    metric_panels(curves, args.ks, os.path.join(args.out, "fig2_metrics.png"))
    write_csv(all_metrics, os.path.join(args.out, "svd_metrics.csv"))
    print(f"\nSaved figures, reconstructions and CSV to ./{args.out}/")

    if args.show:
        for f in ("fig1_comparison_grid.png", "fig2_metrics.png"):
            plt.figure(figsize=(12, 6)); plt.axis("off")
            plt.imshow(plt.imread(os.path.join(args.out, f))); plt.show()
    return all_metrics


if __name__ == "__main__":
    main()
