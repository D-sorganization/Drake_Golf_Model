## 2024-05-22 - [Optimizing Spatial Vector Operations]
**Learning:** `np.cross` in Python has significant overhead for small arrays (3-element vectors) compared to manual scalar arithmetic. In performance-critical loops like spatial algebra, manual unrolling can yield 10x+ speedups for individual operations.
**Action:** For small, fixed-size vector operations in tight loops (like 3D cross products), prefer manual implementation over generic numpy functions if profiling shows a bottleneck.
