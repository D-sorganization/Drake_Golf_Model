"""Golf Analysis Suite GUI Entry Point."""

import logging
import time
import typing

from pydrake.all import (
    Context,
    Diagram,
    Meshcat,
    MeshcatParams,
    MultibodyPlant,
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

STEP_SIZE_S: typing.Final[float] = 0.01  # [s] Simulation time step for 100Hz updates
SLEEP_DURATION_S: typing.Final[float] = 0.05  # [s] Sleep to prevent busy-wait (20Hz)


def initialize_simulation_state(plant: MultibodyPlant, context: Context) -> None:
    """Initialize the simulation state with the model above ground.

    Args:
        plant: The MultibodyPlant of the model.
        context: The root context of the Diagram (will be used to get plant context).
    """
    plant_context = plant.GetMyContextFromRoot(context)

    # Get the pelvis body (root link)
    # Note: Using GetBodyByName might fail if the model instance is strict,
    # but we only have one model here.
    pelvis = plant.GetBodyByName("pelvis")

    # Set initial pose: 1.0m above ground to prevent intersection
    pelvis_pose = RigidTransform(p=[0, 0, 1.0])
    plant.SetFreeBodyPose(plant_context, pelvis, pelvis_pose)


def _reset_simulation(
    simulator: Simulator,
    diagram: Diagram,
    plant: MultibodyPlant,
    context: Context,
) -> None:
    """Reset the simulation to its initial state."""
    logger.info("UX: Resetting simulation...")

    # 1) Set time to start (0.0)
    context.SetTime(0.0)

    # 2) Restore diagram default state
    diagram.SetDefaultContext(context)

    # 3) Reinitialize simulator integrator
    simulator.Initialize()

    # 4) Initialize model state (lift pelvis)
    initialize_simulation_state(plant, context)

    # 5) Publish diagram
    diagram.Publish(context)


def _run_simulation_loop(
    meshcat: Meshcat,
    simulator: Simulator,
    diagram: Diagram,
    plant: MultibodyPlant,
    duration: float,
) -> None:
    """Run the main simulation loop with UX controls."""
    context = simulator.get_mutable_context()
    reset_clicks = 0
    pause_clicks = 0
    stop_clicks = 0
    is_paused = False

    logger.info("Running simulation loop. Use GUI controls to Reset/Pause/Stop.")

    while True:
        # UX: Check Stop
        current_stop_clicks = meshcat.GetButtonClicks("Stop")
        if current_stop_clicks > stop_clicks:
            logger.info("UX: Stop button clicked. Exiting simulation.")
            break

        # UX: Check Reset
        current_reset_clicks = meshcat.GetButtonClicks("Reset")
        if current_reset_clicks > reset_clicks:
            reset_clicks = current_reset_clicks
            _reset_simulation(simulator, diagram, plant, context)

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
            next_time = min(context.get_time() + STEP_SIZE_S, duration)
            simulator.AdvanceTo(next_time)
        else:
            # Sleep briefly to avoid busy loop when paused or finished
            time.sleep(SLEEP_DURATION_S)


def main() -> None:
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
            "- Firewall or network restrictions",
        )
        return

    # Build Diagram with Visualization
    params = GolfModelParams()
    diagram, plant, _ = build_golf_swing_diagram(params, meshcat=meshcat)

    # create a simulator
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)

    # Initialize
    # Ensure simulator is initialized before use
    simulator.Initialize()
    context = simulator.get_mutable_context()

    # Set initial state
    initialize_simulation_state(plant, context)

    logger.info("Simulation initialized. Ready to run.")

    # Add UX controls
    meshcat.AddSlider("Realtime Rate", 0.1, 2.0, 0.1, 1.0)
    meshcat.AddButton("Reset", "KeyR")
    meshcat.AddButton("Pause", "KeyP")
    meshcat.AddButton("Stop", "KeyQ")

    # Run simulation
    duration = 2.0
    _run_simulation_loop(meshcat, simulator, diagram, plant, duration)


if __name__ == "__main__":
    main()
