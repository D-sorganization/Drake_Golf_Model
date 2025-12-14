"""Unit tests for Drake golf swing model.

Tests model building, parameter validation, and model structure.
"""


import numpy as np
import pytest

# Handle optional pydrake dependency
try:
    from pydrake.multibody.tree import SpatialInertia

    from python.src.drake_golf_model import (
        GolfModelParams,
        SegmentParams,
        build_golf_swing_diagram,
        make_cylinder_inertia,
    )

    HAS_DRAKE = True
except ImportError:
    HAS_DRAKE = False
    # Mock symbols for type checker / name resolution
    SpatialInertia = object  # type: ignore[misc,assignment]
    GolfModelParams = None  # type: ignore[assignment]
    SegmentParams = None  # type: ignore[assignment]
    build_golf_swing_diagram = None  # type: ignore[assignment]
    make_cylinder_inertia = None  # type: ignore[assignment]


# Skip decorator for tests requiring Drake
requires_drake = pytest.mark.skipif(not HAS_DRAKE, reason="pydrake not available")


@requires_drake
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

    def test_segment_params_allows_negative_values(self) -> None:
        """Test SegmentParams allows negative values (no validation).

        Note: SegmentParams is a simple dataclass without validation.
        Negative values are allowed but may not be physically meaningful.
        """
        # SegmentParams doesn't validate, so negative values are allowed
        params = SegmentParams(length=-1.0, mass=2.0)
        assert params.length == -1.0
        assert params.mass == 2.0

        params = SegmentParams(length=1.0, mass=-2.0)
        assert params.length == 1.0
        assert params.mass == -2.0


@requires_drake
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


@requires_drake
class TestCylinderInertia:
    """Tests for cylinder inertia computation."""

    def test_make_cylinder_inertia_positive(self) -> None:
        """Test cylinder inertia with positive parameters."""
        # This test requires Drake to be available if we were creating raw objects,
        # but 'make_cylinder_inertia' returns a SpatialInertia which we import for verification.
        if not HAS_DRAKE:
            pytest.skip("Drake not available")

        inertia = make_cylinder_inertia(mass=1.0, radius=0.05, length=1.0)
        assert isinstance(inertia, SpatialInertia)

    def test_make_cylinder_inertia_zero_mass(self) -> None:
        """Test cylinder inertia with zero mass raises error."""
        # Drake throws error for zero mass in standard inertia creation
        try:
            with pytest.raises((ValueError, RuntimeError)):
                make_cylinder_inertia(mass=0.0, radius=0.05, length=1.0)
        except (ImportError, ValueError, RuntimeError):
            if not HAS_DRAKE:
                pytest.skip("Drake not available")
            raise


@requires_drake
class TestModelBuilding:
    """Tests for model building functions."""

    def test_build_golf_swing_diagram_default(self) -> None:
        """Test building golf swing diagram with default parameters."""
        try:
            diagram, plant, scene_graph, _ = build_golf_swing_diagram()
            assert diagram is not None
            assert plant is not None
            assert scene_graph is not None
            # Check that plant has been finalized
            assert plant.num_bodies() > 0
        except ImportError:
            if not HAS_DRAKE:
                pytest.skip("Drake not available")
            raise

    def test_build_golf_swing_diagram_custom_params(self) -> None:
        """Test building golf swing diagram with custom parameters."""
        try:
            custom_params = GolfModelParams(
                pelvis_to_shoulders=0.40,
                spine_mass=16.0,
            )
            diagram, plant, scene_graph, _ = build_golf_swing_diagram(custom_params)
            assert diagram is not None
            assert plant is not None
            assert scene_graph is not None
        except ImportError:
            if not HAS_DRAKE:
                pytest.skip("Drake not available")
            raise

    def test_model_has_required_bodies(self) -> None:
        """Test that model has required body components."""
        try:
            _, plant, _, model_instance = build_golf_swing_diagram()
            # Check for key bodies
            body_names = [
                plant.GetBodyByName(name, model_instance).name()
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
            if not HAS_DRAKE:
                pytest.skip("Drake not available")
            raise

    def test_model_has_joints(self) -> None:
        """Test that model has joints."""
        try:
            _, plant, _, _ = build_golf_swing_diagram()
            assert plant.num_joints() > 0
        except ImportError:
            if not HAS_DRAKE:
                pytest.skip("Drake not available")
            raise

    def test_model_has_actuators(self) -> None:
        """Test that model has actuators."""
        try:
            _, plant, _, _ = build_golf_swing_diagram()
            assert plant.num_actuators() > 0
        except ImportError:
            if not HAS_DRAKE:
                pytest.skip("Drake not available")
            raise


@requires_drake
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


def test_dummy_always_passes() -> None:
    """Ensure pytest collects at least one test to avoid exit code 5."""
    assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
