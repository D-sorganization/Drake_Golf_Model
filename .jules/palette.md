## 2025-12-12 - Meshcat Button State Tracking
**Learning:** Meshcat `AddButton` controls are stateless counters. To implement standard UI patterns like "Toggle" or "Trigger", you must manually track the previous click count and detect changes.
**Action:** Use a `previous_clicks` variable for each button and check `current > previous` to detect events.
