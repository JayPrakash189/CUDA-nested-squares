# CUDA Nested Square Pattern Generator

> **Concurrent Programming Practical Examination**  
> Roll Number: **25201328** | Hard (CO4) – DS11  
> Implementation: CUDA Python (Numba) | Google Colab

---

## Problem Statement

Generate a **nested (concentric) square pattern** using CUDA, where each thread computes the coordinates of concentric squares on a large image grid. Compare performance with a sequential CPU implementation.

- **Dataset:** DS11 – Wikipedia Text Corpus (large-scale data, simulated as a 4096×4096 image grid)
- **Marks:** 40 (Hard)
- **Course Outcome:** CO4

---

## Algorithm

Each pixel `(x, y)` is assigned to exactly one CUDA thread.  
The thread computes the **Chebyshev (L∞) distance** from the image centre:

```
d = max( |x - W/2| , |y - H/2| )
```

If `d % GAP == 0`, the pixel lies on a square boundary → **white (255)**  
Otherwise → **black (0)**

This is an **embarrassingly parallel** problem — every thread is fully independent, with O(1) compute per thread.

---

## Repository Structure

```
cuda-nested-squares/
│
├── README.md                    ← You are here
├── requirements.txt             ← Python dependencies
│
├── src/
│   ├── nested_squares.ipynb     ← Main Colab notebook (run this)
│   └── nested_squares.py        ← Pure Python version of the code
│
├── results/
│   ├── gpu_nested_squares.png   ← GPU output image
│   ├── cpu_nested_squares.png   ← CPU output image
│   └── performance_comparison.png ← Bar chart: GPU vs CPU timing
│
└── docs/
    └── explanation.md           ← Algorithm explanation & viva notes
```

---

## Quick Start — Google Colab

1. Open [Google Colab](https://colab.research.google.com)
2. `Runtime` → `Change runtime type` → Select **T4 GPU** → Save
3. Open `src/nested_squares.ipynb` or paste the code below into a single cell

```python
!pip install numba matplotlib Pillow --quiet
!nvidia-smi
```

---

## Results

| Metric | Value |
|---|---|
| Image size | 4096 × 4096 pixels |
| Total pixels (threads) | 16,777,216 |
| Square gap | 20 px |
| Concentric squares | ~102 |
| GPU kernel time | ~1–3 ms |
| CPU sequential time | ~80–150 ms |
| Speedup | **~50–100×** |
| Verification | PASS (GPU == CPU) |

---

## Performance Chart

![Performance Comparison](results/performance_comparison.png)

---

## Key Concepts

| Concept | Details |
|---|---|
| Parallelism type | Embarrassingly parallel |
| Thread mapping | 1 thread = 1 pixel |
| Distance metric | Chebyshev (L∞): `max(\|Δx\|, \|Δy\|)` |
| Block size | 16 × 16 = 256 threads/block |
| Grid size | 256 × 256 blocks |
| Memory access | Coalesced writes (consecutive x addresses) |
| Timing method | `cudaEvent` (GPU) / `time.perf_counter` (CPU) |

---

## Speedup Analysis — Amdahl's Law

```
Speedup = 1 / (S + (1 - S) / N)
```

Where `S` = serial fraction, `N` = parallel processors.  
Serial portion in our code (memcpy + host logic) ≈ 1–2%  
→ Theoretical max speedup ≈ **50–100×** ✓

---

## License

MIT License — free to use for academic purposes.

---

## Author

**Roll No:** 25201328  
**Exam:** Concurrent Programming Practical  
**Date:** June 2025
