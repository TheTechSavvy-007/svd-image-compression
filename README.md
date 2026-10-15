# Singular Value Decomposition (SVD) and Image Compression

MFAD Mini Project (Applied Linear Algebra), PES University.

**Team:** PRAGUNA KRISHNA A-PES1UG25AM542, SHREYAS P ATHREYA-PES1UG25AM545, NAVEEN H-PES1UG26AM813, TANUSH GIRISH-PES1UG25AM530

This project compresses images by keeping only the `k` most important layers of the image matrix, using the truncated Singular Value Decomposition, and measures what is saved against what is lost.

![Rank-k reconstructions](results/fig1_comparison_grid.png)

## Method

Any image matrix `A` factors as `A = UΣVᵀ`, which is a sum of rank-1 layers `σᵢ uᵢ vᵢᵀ` ordered from most to least important. Keeping the first `k` layers gives the rank-`k` approximation `A_k`:

- It needs `k(M+N+1)` numbers instead of `M×N`.
- By the Eckart–Young–Mirsky theorem it is the best possible rank-`k` approximation, and its squared error equals the sum of the dropped `σᵢ²`. The program verifies this identity on every run (agreement to about 1e-17).
- A grayscale image is one matrix with one SVD. A colour image is split into its R, G and B planes, and each plane gets its own independent SVD.

Metrics reported for each `k`: storage, compression ratio and savings, MSE, PSNR (dB), cumulative energy captured, storage in bytes (float32 and float16), and the size of the smallest JPEG with the same PSNR.

## Results (512 × 512 test images)

Grayscale (cameraman) and colour (astronaut), from `results/svd_metrics.csv`:

| k | Values stored (gray) | Saved | PSNR gray | PSNR colour | Energy (gray) |
|---|---|---|---|---|---|
| 5 | 5,125 | 98.0% | 20.00 dB | 16.17 dB | 97.04% |
| 20 | 20,500 | 92.2% | 24.62 dB | 22.19 dB | 98.98% |
| 50 | 51,250 | 80.4% | 28.64 dB | 27.41 dB | 99.60% |
| 100 | 102,500 | 60.9% | 32.81 dB | 32.86 dB | 99.85% |

![Metrics](results/fig2_metrics.png)

Main findings:

- Quality always improves with `k`, but with diminishing returns. The gain per 1,000 extra stored values falls from 0.30 dB to 0.13 dB to 0.08 dB.
- Energy captured is a poor measure of quality: 97% of the energy at `k = 5` still gives an unrecognisable image, because `σ₁` mostly encodes average brightness.
- The smallest `k` reaching 30 dB is 64 for the grayscale image and 72 for the colour image. `k ≈ 50` is a good rank for previews.
- In real bytes, SVD is not competitive with JPEG: JPEG reaches the same PSNR in 9–24× fewer bytes. At `k = 100` in float32 the factors are larger than the raw image. SVD is useful for progressive loading, denoising and low-rank data, and as the idea behind transform codecs.

## Run it

Requires Python 3.9 or newer.

```bash
pip install -r requirements.txt

# built-in test images (cameraman, astronaut), k = 5 20 50 100
python svd_image_compression.py

# your own image(s), resized to 512 x 512, custom ranks, custom output folder
python svd_image_compression.py --gray photo.jpg --rgb photo.jpg --size 512 --ks 10 30 80 --out my_results
```

Options:

| Flag | Meaning |
|---|---|
| `--gray PATH` | Grayscale image (any format; converted to grayscale). Default: cameraman. |
| `--rgb PATH` | Colour image. Default: astronaut. |
| `--ks 5 20 50 100` | Ranks to evaluate (each must be at least 1; values above `min(M, N)` are capped). |
| `--size 512` | Resize inputs to a square of this size (keeps large photos fast). |
| `--out DIR` | Output folder (default `svd_outputs`). |
| `--no-jpeg` | Skip the JPEG size comparison. |
| `--show` | Also display the figures (useful in Jupyter). |

Each run writes to the output folder: `fig1_comparison_grid.png`, `fig2_metrics.png`, `svd_metrics.csv`, and `reconstructions/` with one PNG per image and `k`.

In Jupyter, paste the script into a cell and call `main([])` or `main(["--gray", "my.png"])`.

## Repository contents

```
svd_image_compression.py   the full program (loading, SVD, metrics, figures, CSV)
requirements.txt           Python dependencies
results/                   figures, reconstructions and metrics from the default run
docs/SVD_Image_Compression.pptx   presentation slides with speaker notes
```
