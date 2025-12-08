"""Golf Analysis Suite GUI Entry Point."""

import logging
import sys
from pathlib import Path

from pydrake.all import (
    Simulator,
    StartMeshcat,
)

# Add repo root to path to import drake_golf_model
# Assuming this file is in python/src/
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drake_golf_model import (  # noqa: E402
    GolfModelParams,
    build_golf_swing_diagram,
)
from python.src.logger_utils import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> None:
    """Run the Golf Analysis GUI."""
    setup_logging()
    logger.info("Starting Golf Analysis Suite...")

    # Start Meshcat
    try:
        meshcat = StartMeshcat()
        logger.info("Meshcat server started at: %s", meshcat.web_url())
    except Exception:  # noqa: BLE001
        logger.exception("Failed to start Meshcat")
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
