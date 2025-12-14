"""Golf Analysis Suite GUI Entry Point."""

import logging
import time

from pydrake.all import (
    Meshcat,
    MeshcatParams,
    RigidTransform,
    Simulator,
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

# Constants
# [s] 20Hz update rate for smooth visualization without overloading
# Value derived from 1.0 / 20.0 Hz
STEP_SIZE_S = 0.05

# [m] Approx standing height
# Source: Winter, D. A. (2009). Biomechanics and motor control of human movement.
#         Approx 50th percentile male pelvis height.
PELVIS_HEIGHT_M = 1.0

# [s] Sleep duration when paused to prevent CPU spin
# Source: 10ms sleep -> ~100Hz polling rate
PAUSE_SLEEP_S = 0.01


def main() -> None:  # noqa: PLR0915
    """Run the Golf Analysis GUI."""
    setup_logging()
    logger.info("Starting Golf Analysis Suite...")

    # Start Meshcat
    try:
        # Security: Bind to localhost to prevent exposure to the network
        meshcat_params = MeshcatParams()
        meshcat_params.host = "localhost"
        meshcat = Meshcat(meshcat_params)

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
    diagram, plant, _ = build_golf_swing_diagram(params, meshcat=meshcat)

    # create a simulator
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)

    # Initialize
    context = simulator.get_mutable_context()

    # Set initial pose (standing up, feet on ground approx)
    plant_context = diagram.GetMutableSubsystemContext(plant, context)
    pelvis = plant.GetBodyByName("pelvis")
    plant.SetFreeBodyPose(
        plant_context, pelvis, RigidTransform([0.0, 0.0, PELVIS_HEIGHT_M])
    )

    logger.info("Simulation initialized. Ready to run.")

    # Add UX controls
    meshcat.AddSlider("Realtime Rate", 0.1, 2.0, 0.1, 1.0)
    meshcat.AddButton("Reset")
    meshcat.AddButton("Pause")
    meshcat.AddButton("Stop")

    logger.info("Simulation running. Use Meshcat controls to interact.")
    logger.info("Click 'Stop' in Meshcat or press Ctrl+C in terminal to exit.")

    # Save initial state for reset (Cloning context captures continuous & discrete state)
    initial_context = context.Clone()

    # State tracking for buttons
    reset_clicks = 0
    pause_clicks = 0
    stop_clicks = 0
    is_paused = False

    while True:
        # Check Stop condition
        curr_stop = meshcat.GetButtonClicks("Stop")
        if curr_stop > stop_clicks:
            logger.info("Visualizer: Stop triggered. Exiting...")
            break

        # 1. Handle Reset
        # GetButtonClicks returns total clicks since creation
        curr_reset = meshcat.GetButtonClicks("Reset")
        if curr_reset > reset_clicks:
            reset_clicks = curr_reset
            # Restore state from cloned context
            context.SetTimeStateAndParametersFrom(initial_context)
            simulator.Initialize()
            # Force publish so view updates even if paused
            diagram.Publish(context)
            logger.info("Visualizer: Reset triggered.")

        # 2. Handle Pause
        curr_pause = meshcat.GetButtonClicks("Pause")
        if curr_pause > pause_clicks:
            clicks_diff = curr_pause - pause_clicks
            pause_clicks = curr_pause
            # Only toggle if an odd number of clicks occurred (e.g., 1 click = toggle, 2 clicks = no net change)
            if clicks_diff % 2 == 1:
                is_paused = not is_paused
                logger.info("Visualizer: Paused = %s", is_paused)

        # 3. Update Rate
        rate = meshcat.GetSliderValue("Realtime Rate")
        simulator.set_target_realtime_rate(rate)

        # 4. Step Simulation
        if not is_paused:
            target_time = context.get_time() + STEP_SIZE_S
            simulator.AdvanceTo(target_time)
        else:
            # Prevent CPU spin when paused
            time.sleep(PAUSE_SLEEP_S)


if __name__ == "__main__":
    main()
