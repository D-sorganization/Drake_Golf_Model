## 2024-05-22 - [Realtime Rate Control for Physics Simulation]
**Learning:** In scientific simulations using Meshcat/Drake, users often need to slow down visualization to understand rapid dynamic events (like a golf swing). Adding a runtime control for `realtime_rate` directly in the visualizer is a high-value, low-code UX win compared to restarting with different parameters.
**Action:** Always expose time/speed controls in physics-based visualizations by default.
