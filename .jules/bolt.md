## 2024-05-22 - [Optimizing Spatial Vector Operations]
**Learning:** `np.cross` in Python has significant overhead for small arrays (3-element vectors) compared to manual scalar arithmetic. In performance-critical loops like spatial algebra, manual unrolling can yield 10x+ speedups for individual operations.
**Action:** For small, fixed-size vector operations in tight loops (like 3D cross products), prefer manual implementation over generic numpy functions if profiling shows a bottleneck.

## 2024-05-23 - [Optimizing Matrix Construction]
**Learning:** `np.block` and `np.array` creation for small, fixed-size matrices (like 3x3 or 6x6) has significant overhead. Manual assignment into a pre-allocated `np.zeros` array is much faster (~8x for 6x6 `crm` matrix).
**Action:** For performance-critical small matrix construction, avoid `np.block` and prefer manual element assignment.
