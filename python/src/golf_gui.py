"""Golf Analysis Suite GUI Entry Point."""

import logging

from pydrake.all import (
    Simulator,
    StartMeshcat,
)

from .drake_golf_model import (
    GolfModelParams,
    build_golf_swing_diagram,
)
from .logger_utils import setup_logging

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
    params = GolfModelParams()
    diagram, _, _ = build_golf_swing_diagram(params, meshcat=meshcat)

    # create a simulator
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)

    # Initialize
    _ = simulator.get_mutable_context()

    logger.info("Simulation initialized. Ready to run.")

    # Run simulation
    duration = 2.0
    logger.info("Running simulation for %.1f seconds...", duration)
    simulator.AdvanceTo(duration)
    logger.info("Simulation complete.")


if __name__ == "__main__":
    main()
