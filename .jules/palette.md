## 2024-05-22 - [Realtime Rate Control for Physics Simulation]
**Learning:** In scientific simulations using Meshcat/Drake, users often need to slow down visualization to understand rapid dynamic events (like a golf swing). Adding a runtime control for `realtime_rate` directly in the visualizer is a high-value, low-code UX win compared to restarting with different parameters.
**Action:** Always expose time/speed controls in physics-based visualizations by default.

## 2025-12-12 - Meshcat Button State Tracking
**Learning:** Meshcat `AddButton` controls are stateless counters. To implement standard UI patterns like "Toggle" or "Trigger", you must manually track the previous click count and detect changes.
**Action:** Use a `previous_clicks` variable for each button and check `current > previous` to detect events.

## 2024-05-23 - [Visual Feedback for Desktop Simulation Controls]
**Learning:** In desktop GUI apps (PyQt) controlling background physics simulations, simply changing button text (Run/Stop) is insufficient for quick status recognition. Adding color-coded states (Green/Red) and a status bar significantly improves the user's ability to instantly parse the system state without reading labels.
**Action:** Use color encoding and redundant status messages for binary system states (Running/Stopped, Connected/Disconnected).

## 2025-10-27 - [Keyboard Shortcuts for Simulation Control]
**Learning:** For physics simulations where users frequently start/stop/reset to iterate, mouse interactions become tedious. Adding standard keyboard shortcuts (Space for Toggle, Ctrl+R for Reset) drastically improves the iteration loop speed and feels "pro".
**Action:** Always map primary simulation actions (Play/Pause, Reset) to standard keyboard shortcuts.
