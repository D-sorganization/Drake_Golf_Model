"""Golf Analysis Suite GUI Entry Point."""

import contextlib
import logging

from pydrake.all import (
    Simulator,
    StartMeshcat,
    RigidTransform,
    RotationMatrix,
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
    # Build Diagram with Visualization
    params = GolfModelParams()
    diagram, plant, _ = build_golf_swing_diagram(params, meshcat=meshcat)

    # create a simulator
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)

    # Initialize
    context = simulator.get_mutable_context()
    
    # Set initial pose (standing up, feet on ground approx)
    # Pelvis is approx 1m high for a standing adult
    plant_context = diagram.GetMutableSubsystemContext(plant, context)
    pelvis = plant.GetBodyByName("pelvis")
    plant.SetFreeBodyPose(
        plant_context, 
        pelvis, 
        RigidTransform([0.0, 0.0, 1.0])
    )

    logger.info("Simulation initialized. Ready to run.")

    # Add UX controls
    meshcat.AddSlider("Realtime Rate", 0.1, 2.0, 0.1, 1.0)
    meshcat.AddButton("Reset")
    meshcat.AddButton("Pause")

    logger.info("Simulation running. Use Meshcat controls to interact.")
    logger.info("Press Ctrl+C in terminal to exit.")

    # Save initial state for reset
    context = simulator.get_mutable_context()
    initial_state = context.get_continuous_state_vector().CopyToVector()
    initial_time = context.get_time()

    # State tracking for buttons
    reset_clicks = 0
    pause_clicks = 0
    is_paused = False

    step_size = 0.05  # Visualization update rate

    while True:
        # 1. Handle Reset
        # GetButtonClicks returns total clicks since creation
        curr_reset = meshcat.GetButtonClicks("Reset")
        if curr_reset > reset_clicks:
            reset_clicks = curr_reset
            context.SetTime(initial_time)
            context.SetContinuousState(initial_state)
            simulator.Initialize()
            logger.info("Visualizer: Reset triggered.")

        # 2. Handle Pause
        curr_pause = meshcat.GetButtonClicks("Pause")
        if curr_pause > pause_clicks:
            pause_clicks = curr_pause
            is_paused = not is_paused
            logger.info("Visualizer: Paused = %s", is_paused)

        # 3. Update Rate
        rate = meshcat.GetSliderValue("Realtime Rate")
        simulator.set_target_realtime_rate(rate)

        # 4. Step Simulation
        if not is_paused:
            target_time = context.get_time() + step_size
            simulator.AdvanceTo(target_time)
        else:
            # If paused, we still need to yield time to Meshcat/system (though python is single threaded)
            # Just sleeping a tiny bit prevents CPU spin
            pass


if __name__ == "__main__":
    main()
