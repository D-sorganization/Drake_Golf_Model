"""Golf Analysis Suite GUI Entry Point."""

import logging
import time

from pydrake.all import (
    Context,
    Diagram,
    Meshcat,
    Simulator,
    StartMeshcat,
)

try:
    from .drake_golf_model import (
        GolfModelParams,
        build_golf_swing_diagram,
    )
    from .logger_utils import setup_logging
except ImportError:
    # Fallback imports for direct script execution (when run as __main__)
    from drake_golf_model import (  # type: ignore[no-redef]
        GolfModelParams,
        build_golf_swing_diagram,
    )
    from logger_utils import setup_logging  # type: ignore[no-redef]

logger = logging.getLogger(__name__)


def _run_simulation_loop(
    meshcat: Meshcat,
    simulator: Simulator,
    diagram: Diagram,
    context: Context,
    duration: float,
) -> None:
    """Run the main simulation loop with UX controls."""
    reset_clicks = 0
    pause_clicks = 0
    is_paused = False
    step_size = 0.1

    logger.info("Running simulation loop. Use GUI controls to Reset/Pause.")

    while True:
        # UX: Check Reset
        current_reset_clicks = meshcat.GetButtonClicks("Reset")
        if current_reset_clicks > reset_clicks:
            reset_clicks = current_reset_clicks
            logger.info("UX: Resetting simulation...")
            context.SetTime(0.0)
            diagram.SetDefaultContext(context)
            simulator.Initialize()

        # UX: Check Pause
        current_pause_clicks = meshcat.GetButtonClicks("Pause")
        if current_pause_clicks > pause_clicks:
            pause_clicks = current_pause_clicks
            is_paused = not is_paused
            logger.info("UX: Simulation %s", "Paused" if is_paused else "Resumed")

        # UX: Update Realtime Rate from Meshcat slider
        rate = meshcat.GetSliderValue("Realtime Rate")
        simulator.set_target_realtime_rate(rate)

        # Advance simulation if not paused and within duration
        if not is_paused and context.get_time() < duration:
            next_time = min(context.get_time() + step_size, duration)
            simulator.AdvanceTo(next_time)
        else:
            # Sleep briefly to avoid busy loop when paused or finished
            time.sleep(0.05)


def main() -> None:
    """Run the Golf Analysis GUI."""
    setup_logging()
    logger.info("Starting Golf Analysis Suite...")

    # Start Meshcat
    try:
        meshcat = StartMeshcat()
        logger.info("Meshcat server started at: %s", meshcat.web_url())
    except Exception:
        logger.exception(
            "Failed to start Meshcat. Common causes include:\n"
            "- Port already in use (try closing other Meshcat sessions or rebooting)\n"
            "- Missing or incompatible Meshcat/Drake dependencies\n"
            "- Firewall or network restrictions"
        )
        return

    # Build Diagram with Visualization
    params = GolfModelParams()
    diagram, _, _ = build_golf_swing_diagram(params, meshcat=meshcat)

    # create a simulator
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)

    # Initialize
    context = simulator.get_mutable_context()

    logger.info("Simulation initialized. Ready to run.")

    # Add UX controls
    meshcat.AddSlider("Realtime Rate", 0.1, 2.0, 0.1, 1.0)
    meshcat.AddButton("Reset")
    meshcat.AddButton("Pause")

    # Run simulation
    duration = 2.0
    _run_simulation_loop(meshcat, simulator, diagram, context, duration)


if __name__ == "__main__":
    main()
