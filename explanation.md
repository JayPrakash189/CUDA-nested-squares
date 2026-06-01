# Algorithm Explanation & Viva Notes

## How the Kernel Works

Each CUDA thread is assigned one pixel `(x, y)` on the image grid.

**Step 1 — Translate to centre-relative coordinates:**
```
cx = x - WIDTH / 2
cy = y - HEIGHT / 2
```

**Step 2 — Compute Chebyshev distance:**
```
dist = max(|cx|, |cy|)
```

**Step 3 — Decide pixel colour:**
```
pixel = 255 (white) if dist % GAP == 0
pixel = 0   (black) otherwise
```

## Why Chebyshev Distance?

| Distance Metric | Formula | Shape at radius r |
|---|---|---|
| Euclidean (L2) | sqrt(x^2+y^2) | Circle |
| Manhattan (L1) | |x|+|y| | Diamond |
| Chebyshev (L-inf) | max(|x|,|y|) | **Square** (we use this) |

## Why This is Embarrassingly Parallel

- Each thread uses ONLY its own (x, y) coordinates
- No thread reads another thread's output
- No __syncthreads() needed
- O(1) arithmetic per thread
- Parallel fraction is approximately 100%

## Viva Quick Reference

**Q: Why 16x16 blocks?**
A: 256 threads/block — standard sweet spot for SM occupancy. Matches 2D image layout.

**Q: What is coalesced access?**
A: Threads in the same warp write consecutive addresses, enabling a single memory transaction for peak bandwidth.

**Q: Why warm-up launch before timing?**
A: First launch incurs JIT compilation (50-300ms). Warm-up absorbs this so timed run measures only compute.

**Q: How did you verify correctness?**
A: np.array_equal(gpu_img, cpu_img) — bit-exact match confirms PASS.
