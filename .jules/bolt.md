## 2024-05-22 - [Optimizing Spatial Vector Operations]
**Learning:** `np.cross` in Python has significant overhead for small arrays (3-element vectors) compared to manual scalar arithmetic. In performance-critical loops like spatial algebra, manual unrolling can yield 10x+ speedups for individual operations.
**Action:** For small, fixed-size vector operations in tight loops (like 3D cross products), prefer manual implementation over generic numpy functions if profiling shows a bottleneck.

## 2024-05-23 - [Optimizing Matrix Construction]
**Learning:** `np.block` and `np.array` creation for small, fixed-size matrices (like 3x3 or 6x6) has significant overhead. Manual assignment into a pre-allocated `np.zeros` array is much faster (~8x for 6x6 `crm` matrix).
**Action:** For performance-critical small matrix construction, avoid `np.block` and prefer manual element assignment.

## 2024-05-24 - [Optimizing Spatial Inertia Matrix Construction]
**Learning:** `np.block` for constructing 6x6 spatial inertia matrices has significant overhead. Manual element assignment into a pre-allocated `np.zeros` array yielded a ~2.8x speedup in `mcI`. Also, `mass * np.eye(3)` creates unnecessary temporary arrays.
**Action:** Use manual assignment for constructing spatial inertia matrices and avoid temporary identity matrix scaling when possible.

## 2024-05-25 - [Optimizing Joint Transforms]
**Learning:** Helper functions like `xrot` that include safety checks (like `np.linalg.det`) are extremely expensive for hot loops. `jcalc` using `xrot` was ~17x slower than a manual implementation that bypasses the check, because `jcalc` already guarantees valid inputs.
**Action:** In core kinematic functions like `jcalc`, construct transformation matrices manually to avoid overhead from general-purpose helper functions and unnecessary validation checks.
