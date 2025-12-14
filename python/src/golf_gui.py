"""Golf Analysis Suite GUI Entry Point."""

import logging
import os
import time
import typing
from pathlib import Path

from pydrake.all import (
    Context,
    Diagram,
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

# [s] Simulation time step for 100Hz updates (smoother visualization)
STEP_SIZE_S: typing.Final[float] = 0.01

# [s] Sleep to prevent busy-wait (20Hz)
SLEEP_DURATION_S: typing.Final[float] = 0.05

# [m] Design value for standing height
# Source: Winter (2009) data for 50th percentile male; set to 1.0m for simulation design.
PELVIS_HEIGHT_M: typing.Final[float] = 1.0


def poll_ui_state(  # noqa: PLR0913
    meshcat: Meshcat,
    context: Context,
    simulator: Simulator,
    diagram: Diagram,
    initial_context: Context,
    *,
    is_paused: bool,
    pause_clicks: int,
    stop_clicks: int,
    reset_clicks: int,
) -> tuple[bool, bool, int, int, int]:
    """Handle UI events from Meshcat buttons."""
    # Check Stop condition
    curr_stop = meshcat.GetButtonClicks("Stop")
    if curr_stop > stop_clicks:
        logger.info("Visualizer: Stop triggered. Exiting...")
        return True, is_paused, pause_clicks, curr_stop, reset_clicks

    # 1. Handle Reset
    curr_reset = meshcat.GetButtonClicks("Reset")
    if curr_reset > reset_clicks:
        reset_clicks = curr_reset
        # Restore state from cloned context
        # Reset simulation state:
        #   1) Set time to start (0.0) so simulation restarts from the beginning.
        #   2) Restore diagram default state to reset all model variables and parameters.
        #   3) Reinitialize simulator integrator to ensure numerical state is consistent.
        context.SetTimeStateAndParametersFrom(initial_context)
        simulator.Initialize()
        # Force publish so view updates even if paused
        diagram.Publish(context)
        # BUG FIX: Ensure we unpause on reset so playback starts immediately
        is_paused = False
        logger.info("Visualizer: Reset triggered.")

    # 2. Handle Pause
    curr_pause = meshcat.GetButtonClicks("Pause")
    if curr_pause > pause_clicks:
        is_paused = not is_paused
        pause_clicks = curr_pause
        logger.info("Visualizer: Paused = %s", is_paused)

    return False, is_paused, pause_clicks, stop_clicks, reset_clicks


def _is_running_in_docker() -> bool:
    """Detect if running inside a Docker container."""
    if Path("/.dockerenv").exists():
        return True
    try:
        content = Path("/proc/1/cgroup").read_text(encoding="utf-8")
        if "docker" in content or "containerd" in content:
            return True
    except OSError:
        pass
    return False


def main() -> None:  # noqa: PLR0915
    """Run the Golf Analysis GUI."""
    setup_logging()
    logger.info("Starting Golf Analysis Suite...")

    # Start Meshcat
    try:
        # Security: Validate host bind
        meshcat_params = MeshcatParams()
        env_host = os.environ.get("MESHCAT_HOST", "localhost")
        safe_hosts = {"localhost", "127.0.0.1"}

        if env_host not in safe_hosts and not _is_running_in_docker():
            logger.warning(
                "Unsafe MESHCAT_HOST value '%s' ignored outside Docker. "
                "Defaulting to 'localhost'.",
                env_host,
            )
            meshcat_params.host = "localhost"
        else:
            meshcat_params.host = env_host

        # Use an ephemeral port to avoid collisions
        meshcat_params.port = 0
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
    diagram, plant, _, model_instance = build_golf_swing_diagram(
        params, meshcat=meshcat
    )

    # create a simulator
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)

    # Initialize
    context = simulator.get_mutable_context()

    # Set initial pose (standing up, feet on ground)
    plant_context = diagram.GetMutableSubsystemContext(plant, context)
    pelvis = plant.GetBodyByName("pelvis", model_instance)
    plant.SetFreeBodyPose(
        plant_context, pelvis, RigidTransform([0.0, 0.0, PELVIS_HEIGHT_M])
    )

    # Initialize BEFORE cloning to capture post-init state
    simulator.Initialize()
    logger.info("Simulation initialized. Ready to run.")

    # Add UX controls
    meshcat.AddSlider("Realtime Rate", 0.1, 2.0, 0.1, 1.0)
    meshcat.AddButton("Reset", "KeyR")
    meshcat.AddButton("Pause", "KeyP")
    meshcat.AddButton("Stop", "KeyQ")

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
        try:
            stop_triggered, is_paused, pause_clicks, stop_clicks, reset_clicks = (
                poll_ui_state(
                    meshcat,
                    context,
                    simulator,
                    diagram,
                    initial_context,
                    is_paused=is_paused,
                    pause_clicks=pause_clicks,
                    stop_clicks=stop_clicks,
                    reset_clicks=reset_clicks,
                )
            )

            if stop_triggered:
                break

            # 3. Update Rate
            rate = meshcat.GetSliderValue("Realtime Rate")
            simulator.set_target_realtime_rate(rate)

            # 4. Step Simulation
            if not is_paused:
                target_time = context.get_time() + STEP_SIZE_S
                simulator.AdvanceTo(target_time)
            else:
                # Prevent CPU spin when paused
                time.sleep(SLEEP_DURATION_S)

        except (RuntimeError, ValueError):
            logger.exception("A simulation error occurred")
            break
        except Exception:
            # Catch-all for other non-exit exceptions (like KeyboardInterrupt if not handled)
            # but allow clean exit if needed.
            logger.exception("An unexpected error occurred.")
            break


if __name__ == "__main__":
    main()
