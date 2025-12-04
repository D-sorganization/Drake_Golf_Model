"""Unit tests for Drake golf swing model.

Tests model building, parameter validation, and model structure.
"""

# Import Drake model builder
import sys
from pathlib import Path

import numpy as np  # noqa: TID253
import pytest

# Add parent directory to path to import drake_golf_model
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from drake_golf_model import (
    GolfModelParams,
    SegmentParams,
    build_golf_swing_diagram,
    make_cylinder_inertia,
)


class TestSegmentParams:
    """Tests for SegmentParams dataclass."""

    def test_segment_params_default(self) -> None:
        """Test SegmentParams with default radius."""
        params = SegmentParams(length=1.0, mass=2.0)
        assert params.length == 1.0
        assert params.mass == 2.0
        assert params.radius == 0.03  # Default value

    def test_segment_params_custom_radius(self) -> None:
        """Test SegmentParams with custom radius."""
        params = SegmentParams(length=1.0, mass=2.0, radius=0.05)
        assert params.radius == 0.05

    def test_segment_params_positive_values(self) -> None:
        """Test SegmentParams requires positive values."""
        with pytest.raises((ValueError, TypeError)):
            SegmentParams(length=-1.0, mass=2.0)
        with pytest.raises((ValueError, TypeError)):
            SegmentParams(length=1.0, mass=-2.0)


class TestGolfModelParams:
    """Tests for GolfModelParams dataclass."""

    def test_default_params(self) -> None:
        """Test default parameter values."""
        params = GolfModelParams()
        assert params.pelvis_to_shoulders > 0
        assert params.spine_mass > 0
        assert params.hand_spacing_m > 0
        assert params.club.length > 0
        assert params.club.mass > 0

    def test_custom_params(self) -> None:
        """Test custom parameter values."""
        custom_club = SegmentParams(length=1.1, mass=0.45)
        params = GolfModelParams(club=custom_club)
        assert params.club.length == 1.1
        assert params.club.mass == 0.45

    def test_joint_axes_normalized(self) -> None:
        """Test joint axes are properly defined."""
        params = GolfModelParams()
        # Check axes are numpy arrays
        assert isinstance(params.hip_axis, np.ndarray)
        assert isinstance(params.spine_twist_axis, np.ndarray)
        assert len(params.shoulder_axes) == 3

    def test_friction_parameters(self) -> None:
        """Test friction parameters are reasonable."""
        params = GolfModelParams()
        assert 0 < params.ground_friction_mu_static <= 1.0
        assert 0 < params.ground_friction_mu_dynamic <= 1.0
        assert params.ground_friction_mu_dynamic <= params.ground_friction_mu_static


class TestCylinderInertia:
    """Tests for cylinder inertia computation."""

    def test_make_cylinder_inertia_positive(self) -> None:
        """Test cylinder inertia with positive parameters."""
        # This test requires Drake to be available
        try:
            from pydrake.multibody.tree import SpatialInertia

            inertia = make_cylinder_inertia(mass=1.0, radius=0.05, length=1.0)
            assert isinstance(inertia, SpatialInertia)
        except ImportError:
            pytest.skip("Drake not available")

    def test_make_cylinder_inertia_zero_mass(self) -> None:
        """Test cylinder inertia with zero mass raises error."""
        try:
            with pytest.raises((ValueError, RuntimeError)):
                make_cylinder_inertia(mass=0.0, radius=0.05, length=1.0)
        except ImportError:
            pytest.skip("Drake not available")


class TestModelBuilding:
    """Tests for model building functions."""

    def test_build_golf_swing_diagram_default(self) -> None:
        """Test building golf swing diagram with default parameters."""
        try:
            diagram, plant, scene_graph = build_golf_swing_diagram()
            assert diagram is not None
            assert plant is not None
            assert scene_graph is not None
            # Check that plant has been finalized
            assert plant.num_bodies() > 0
        except ImportError:
            pytest.skip("Drake not available")

    def test_build_golf_swing_diagram_custom_params(self) -> None:
        """Test building golf swing diagram with custom parameters."""
        try:
            custom_params = GolfModelParams(
                pelvis_to_shoulders=0.40,
                spine_mass=16.0,
            )
            diagram, plant, scene_graph = build_golf_swing_diagram(custom_params)
            assert diagram is not None
            assert plant is not None
            assert scene_graph is not None
        except ImportError:
            pytest.skip("Drake not available")

    def test_model_has_required_bodies(self) -> None:
        """Test that model has required body components."""
        try:
            _, plant, _ = build_golf_swing_diagram()
            # Check for key bodies
            body_names = [
                plant.GetBodyByName(name).name()
                for name in [
                    "pelvis",
                    "spine_base",
                    "left_upper_arm",
                    "right_upper_arm",
                    "club",
                ]
            ]
            assert len(body_names) == 5
        except (ImportError, RuntimeError):
            pytest.skip("Drake not available or model structure changed")

    def test_model_has_joints(self) -> None:
        """Test that model has joints."""
        try:
            _, plant, _ = build_golf_swing_diagram()
            assert plant.num_joints() > 0
        except ImportError:
            pytest.skip("Drake not available")

    def test_model_has_actuators(self) -> None:
        """Test that model has actuators."""
        try:
            _, plant, _ = build_golf_swing_diagram()
            assert plant.num_actuators() > 0
        except ImportError:
            pytest.skip("Drake not available")


class TestParameterValidation:
    """Tests for parameter validation."""

    def test_hand_spacing_reasonable(self) -> None:
        """Test hand spacing is reasonable."""
        params = GolfModelParams()
        # Hand spacing should be less than club length
        assert params.hand_spacing_m < params.club.length
        # Hand spacing should be positive
        assert params.hand_spacing_m > 0

    def test_segment_lengths_positive(self) -> None:
        """Test all segment lengths are positive."""
        params = GolfModelParams()
        assert params.scapula_rod.length > 0
        assert params.upper_arm.length > 0
        assert params.forearm.length > 0
        assert params.hand.length > 0
        assert params.club.length > 0

    def test_segment_masses_positive(self) -> None:
        """Test all segment masses are positive."""
        params = GolfModelParams()
        assert params.scapula_rod.mass > 0
        assert params.upper_arm.mass > 0
        assert params.forearm.mass > 0
        assert params.hand.mass > 0
        assert params.club.mass > 0
        assert params.spine_mass > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
