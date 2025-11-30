# golf_swing_drake_model.py

from dataclasses import dataclass
import numpy as np

from pydrake.all import (
    DiagramBuilder,
    MultibodyPlant,
    SceneGraph,
    AddMultibodyPlantSceneGraph,
    RigidTransform,
    SpatialInertia,
    UnitInertia,
    HalfSpace,
    CoulombFriction,
    Sphere,
)


# -----------------------------
# Parameter containers
# -----------------------------

@dataclass
class SegmentParams:
    length: float
    mass: float
    radius: float = 0.03  # default cylinder radius


@dataclass
class GolfModelParams:
    # Rough anthropometrics (replace with real values later).
    pelvis_to_shoulders: float = 0.35
    spine_mass: float = 15.0

    scapula_rod: SegmentParams = SegmentParams(length=0.12, mass=1.0)
    upper_arm: SegmentParams = SegmentParams(length=0.30, mass=2.0)
    forearm: SegmentParams = SegmentParams(length=0.27, mass=1.5)
    hand: SegmentParams = SegmentParams(length=0.10, mass=0.5)

    club: SegmentParams = SegmentParams(length=1.05, mass=0.40)

    # Distance between hand attachment points along club (meters)
    # ~3 inches = 0.0762 m
    hand_spacing_m: float = 0.0762

    # Joint axes
    hip_axis: np.ndarray = np.array([0.0, 0.0, 1.0])
    spine_twist_axis: np.ndarray = np.array([0.0, 0.0, 1.0])

    spine_universal_axis_1: np.ndarray = np.array([1.0, 0.0, 0.0])  # flex/extend
    spine_universal_axis_2: np.ndarray = np.array([0.0, 1.0, 0.0])  # side-bend

    scap_axis_1: np.ndarray = np.array([1.0, 0.0, 0.0])
    scap_axis_2: np.ndarray = np.array([0.0, 1.0, 0.0])

    wrist_axis_1: np.ndarray = np.array([1.0, 0.0, 0.0])
    wrist_axis_2: np.ndarray = np.array([0.0, 1.0, 0.0])

    shoulder_axes: tuple = (
        np.array([0.0, 0.0, 1.0]),  # yaw
        np.array([0.0, 1.0, 0.0]),  # pitch
        np.array([1.0, 0.0, 0.0]),  # roll
    )

    elbow_axis: np.ndarray = np.array([0.0, 1.0, 0.0])

    # Contact / ground
    ground_friction_mu_static: float = 0.8
    ground_friction_mu_dynamic: float = 0.6
    clubhead_radius: float = 0.025  # m


# -----------------------------
# Helper functions
# -----------------------------

def make_cylinder_inertia(mass: float, radius: float, length: float):
    """
    Uniform solid cylinder aligned with +z, COM at origin.
    """
    I = UnitInertia.Cylinder(radius, length)
    return SpatialInertia(mass, np.zeros(3), I * mass)


def add_body_with_inertia(plant: MultibodyPlant, name: str, params: SegmentParams):
    inertia = make_cylinder_inertia(params.mass, params.radius, params.length)
    body = plant.AddRigidBody(name, inertia)
    return body


def add_free_base_with_hip(plant: MultibodyPlant, params: GolfModelParams):
    """
    Adds a 6-DoF pelvis base (FreeJoint) and a revolute hip joint
    to a 'spine_base' body.
    """
    pelvis_inertia = SpatialInertia(
        params.spine_mass, np.zeros(3),
        UnitInertia.SolidBox(0.3, 0.2, 0.2) * params.spine_mass
    )
    pelvis = plant.AddRigidBody("pelvis", pelvis_inertia)

    # 6-DoF between world and pelvis
    plant.AddJointFreeBody(pelvis)

    # Revolute hip joint
    spine_base_inertia = SpatialInertia(
        1.0, np.zeros(3),
        UnitInertia.SolidBox(0.1, 0.1, 0.1)
    )
    spine_base = plant.AddRigidBody("spine_base", spine_base_inertia)

    axis = params.hip_axis / np.linalg.norm(params.hip_axis)

    plant.AddJointRevolute(
        "hip_yaw",
        parent=pelvis,
        child=spine_base,
        pose_in_parent=RigidTransform(),  # hip at pelvis origin
        pose_in_child=RigidTransform(),
        axis=axis
    )

    return pelvis, spine_base


def add_spine_stack(plant: MultibodyPlant, spine_base, params: GolfModelParams):
    """
    Universal + twist revolute -> upper torso hub.
    """
    # Lower spine (universal)
    lower_spine_inertia = SpatialInertia(
        params.spine_mass * 0.5,
        np.zeros(3),
        UnitInertia.SolidBox(0.2, 0.2, params.pelvis_to_shoulders * 0.5)
        * (params.spine_mass * 0.5)
    )
    lower_spine = plant.AddRigidBody("lower_spine", lower_spine_inertia)

    a1 = params.spine_universal_axis_1 / np.linalg.norm(params.spine_universal_axis_1)
    a2 = params.spine_universal_axis_2 / np.linalg.norm(params.spine_universal_axis_2)

    plant.AddJointUniversal(
        "spine_universal",
        parent=spine_base,
        child=lower_spine,
        pose_in_parent=RigidTransform(),
        pose_in_child=RigidTransform(),
        axis1=a1,
        axis2=a2
    )

    # Upper spine twist
    upper_spine_inertia = SpatialInertia(
        params.spine_mass * 0.5,
        np.zeros(3),
        UnitInertia.SolidBox(0.2, 0.2, params.pelvis_to_shoulders * 0.5)
        * (params.spine_mass * 0.5)
    )
    upper_spine = plant.AddRigidBody("upper_spine", upper_spine_inertia)

    twist_axis = params.spine_twist_axis / np.linalg.norm(params.spine_twist_axis)

    plant.AddJointRevolute(
        "spine_twist",
        parent=lower_spine,
        child=upper_spine,
        pose_in_parent=RigidTransform(
            p=[0.0, 0.0, params.pelvis_to_shoulders * 0.25]
        ),
        pose_in_child=RigidTransform(
            p=[0.0, 0.0, -params.pelvis_to_shoulders * 0.25]
        ),
        axis=twist_axis
    )

    # Upper torso hub
    hub_inertia = SpatialInertia(
        5.0, np.zeros(3),
        UnitInertia.SolidBox(0.3, 0.3, 0.2) * 5.0
    )
    upper_torso = plant.AddRigidBody("upper_torso_hub", hub_inertia)

    plant.WeldFrames(
        upper_spine.body_frame(),
        upper_torso.body_frame(),
        RigidTransform(p=[0.0, 0.0, params.pelvis_to_shoulders * 0.25])
    )

    return upper_torso


def add_scapula_and_shoulder_chain(
    plant: MultibodyPlant,
    upper_torso,
    side: str,
    params: GolfModelParams
):
    """
    Scapula universal + rod, then 3-DOF shoulder (gimbal from 3 revolutes),
    then upper arm.
    """
    sign = 1.0 if side == "right" else -1.0

    scap_body = add_body_with_inertia(
        plant, f"{side}_scapula_rod", params.scapula_rod
    )

    a1 = params.scap_axis_1 / np.linalg.norm(params.scap_axis_1)
    a2 = params.scap_axis_2 / np.linalg.norm(params.scap_axis_2)

    scap_offset = [0.0, sign * 0.18, 0.10]
    plant.AddJointUniversal(
        f"{side}_scapula_universal",
        parent=upper_torso,
        child=scap_body,
        pose_in_parent=RigidTransform(p=scap_offset),
        pose_in_child=RigidTransform(
            p=[0.0, 0.0, -params.scapula_rod.length / 2.0]
        ),
        axis1=a1,
        axis2=a2
    )

    # Shoulder gimbal: yaw -> pitch -> roll
    yaw_inertia = SpatialInertia(
        0.1, np.zeros(3),
        UnitInertia.SolidSphere(0.05) * 0.1
    )
    yaw_link = plant.AddRigidBody(f"{side}_shoulder_yaw_link", yaw_inertia)

    pitch_inertia = SpatialInertia(
        0.1, np.zeros(3),
        UnitInertia.SolidSphere(0.05) * 0.1
    )
    pitch_link = plant.AddRigidBody(f"{side}_shoulder_pitch_link", pitch_inertia)

    roll_inertia = SpatialInertia(
        0.1, np.zeros(3),
        UnitInertia.SolidSphere(0.05) * 0.1
    )
    roll_link = plant.AddRigidBody(f"{side}_shoulder_roll_link", roll_inertia)

    a_yaw, a_pitch, a_roll = params.shoulder_axes

    plant.AddJointRevolute(
        f"{side}_shoulder_yaw",
        parent=scap_body,
        child=yaw_link,
        pose_in_parent=RigidTransform(
            p=[0.0, 0.0, params.scapula_rod.length / 2.0]
        ),
        pose_in_child=RigidTransform(),
        axis=a_yaw / np.linalg.norm(a_yaw)
    )

    plant.AddJointRevolute(
        f"{side}_shoulder_pitch",
        parent=yaw_link,
        child=pitch_link,
        pose_in_parent=RigidTransform(),
        pose_in_child=RigidTransform(),
        axis=a_pitch / np.linalg.norm(a_pitch)
    )

    plant.AddJointRevolute(
        f"{side}_shoulder_roll",
        parent=pitch_link,
        child=roll_link,
        pose_in_parent=RigidTransform(),
        pose_in_child=RigidTransform(),
        axis=a_roll / np.linalg.norm(a_roll)
    )

    upper_arm = add_body_with_inertia(
        plant, f"{side}_upper_arm", params.upper_arm
    )

    plant.WeldFrames(
        roll_link.body_frame(),
        upper_arm.body_frame(),
        RigidTransform(
            p=[0.0, 0.0, -params.upper_arm.length / 2.0]
        )
    )

    return upper_arm


def add_elbow_and_forearm(
    plant: MultibodyPlant,
    upper_arm,
    side: str,
    params: GolfModelParams
):
    forearm = add_body_with_inertia(
        plant, f"{side}_forearm", params.forearm
    )

    axis = params.elbow_axis / np.linalg.norm(params.elbow_axis)

    plant.AddJointRevolute(
        f"{side}_elbow",
        parent=upper_arm,
        child=forearm,
        pose_in_parent=RigidTransform(
            p=[0.0, 0.0, params.upper_arm.length / 2.0]
        ),
        pose_in_child=RigidTransform(
            p=[0.0, 0.0, -params.forearm.length / 2.0]
        ),
        axis=axis
    )

    return forearm


def add_wrist_and_hand(
    plant: MultibodyPlant,
    forearm,
    side: str,
    params: GolfModelParams
):
    hand = add_body_with_inertia(
        plant, f"{side}_hand", params.hand
    )

    a1 = params.wrist_axis_1 / np.linalg.norm(params.wrist_axis_1)
    a2 = params.wrist_axis_2 / np.linalg.norm(params.wrist_axis_2)

    plant.AddJointUniversal(
        f"{side}_wrist_universal",
        parent=forearm,
        child=hand,
        pose_in_parent=RigidTransform(
            p=[0.0, 0.0, params.forearm.length / 2.0]
        ),
        pose_in_child=RigidTransform(
            p=[0.0, 0.0, -params.hand.length / 2.0]
        ),
        axis1=a1,
        axis2=a2
    )

    return hand


def add_club_with_dual_hand_constraints(
    plant: MultibodyPlant,
    left_hand,
    right_hand,
    params: GolfModelParams
):
    """
    Create the club body and attach each hand to *different* points on the shaft
    using ball constraints.

    Convention:
      - Club's +z axis runs from butt toward head.
      - Club COM is at z = 0.
      - Left (lead) hand attaches at z = -hand_spacing/2
      - Right (trail) hand attaches at z = +hand_spacing/2
      - Clubhead (collision sphere) is at z = +club.length/2
    """
    # Ensure spacing isn't bigger than club length (just in case)
    spacing = min(params.hand_spacing_m, 0.5 * params.club.length)

    club = add_body_with_inertia(plant, "club", params.club)

    # Points on club where hands attach (in club frame)
    p_club_lead = [0.0, 0.0, -0.5 * spacing]   # left hand (lead) nearer butt
    p_club_trail = [0.0, 0.0, +0.5 * spacing]  # right hand (trail) nearer head

    # Points on hands (distal end along +z in hand frame)
    p_left_hand = [0.0, 0.0, params.hand.length / 2.0]
    p_right_hand = [0.0, 0.0, params.hand.length / 2.0]

    # Ball constraint: left hand <-> proximal point on club
    plant.AddBallConstraint(
        frameA=left_hand.body_frame(),
        p_AP=p_left_hand,
        frameB=club.body_frame(),
        p_BQ=p_club_lead
    )

    # Ball constraint: right hand <-> distal point on club
    plant.AddBallConstraint(
        frameA=right_hand.body_frame(),
        p_AP=p_right_hand,
        frameB=club.body_frame(),
        p_BQ=p_club_trail
    )

    return club


def add_ground_and_club_contact(
    plant: MultibodyPlant,
    scene_graph: SceneGraph,
    club,
    params: GolfModelParams
):
    """
    Adds a ground half-space, and a spherical collision at the clubhead.
    """
    world_body = plant.world_body()
    X_WG = RigidTransform()  # z=0 plane

    friction = CoulombFriction(
        params.ground_friction_mu_static,
        params.ground_friction_mu_dynamic
    )

    # Ground collision + visual
    plant.RegisterCollisionGeometry(
        world_body,
        X_WG,
        HalfSpace(),
        "ground_collision",
        friction
    )
    plant.RegisterVisualGeometry(
        world_body,
        X_WG,
        HalfSpace(),
        "ground_visual"
    )

    # Clubhead collision sphere at distal end of club (+z)
    X_C_H = RigidTransform(
        p=[0.0, 0.0, params.club.length / 2.0]
    )
    plant.RegisterCollisionGeometry(
        club,
        X_C_H,
        Sphere(params.clubhead_radius),
        "clubhead_collision",
        friction
    )
    plant.RegisterVisualGeometry(
        club,
        X_C_H,
        Sphere(params.clubhead_radius),
        "clubhead_visual"
    )


def add_joint_actuators(plant: MultibodyPlant):
    """
    Add actuators for ALL joints in the plant.
    This makes it easy to use InverseDynamicsController.
    """
    for joint_index in range(plant.num_joints()):
        joint = plant.get_joint(joint_index)
        if joint.num_velocities() == 0:
            continue
        plant.AddJointActuator(f"{joint.name()}_act", joint)


# -----------------------------
# Main model builder
# -----------------------------

def build_golf_swing_diagram(params: GolfModelParams = GolfModelParams()):
    """
    Builds the full multibody model + scene graph:
      - Free pelvis + hip
      - Spine (universal + twist)
      - Upper torso hub
      - Left and right arm chains
      - Club attached to both hands with 3" separation
      - Ground contact + clubhead collision
      - Joint actuators for all non-weld joints
    Returns (diagram, plant, scene_graph).
    """
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)

    # Base and spine
    pelvis, spine_base = add_free_base_with_hip(plant, params)
    upper_torso = add_spine_stack(plant, spine_base, params)

    # Arms
    left_upper_arm = add_scapula_and_shoulder_chain(
        plant, upper_torso, side="left", params=params
    )
    left_forearm = add_elbow_and_forearm(
        plant, left_upper_arm, side="left", params=params
    )
    left_hand = add_wrist_and_hand(
        plant, left_forearm, side="left", params=params
    )

    right_upper_arm = add_scapula_and_shoulder_chain(
        plant, upper_torso, side="right", params=params
    )
    right_forearm = add_elbow_and_forearm(
        plant, right_upper_arm, side="right", params=params
    )
    right_hand = add_wrist_and_hand(
        plant, right_forearm, side="right", params=params
    )

    # Club with two separate hand constraints (= parallel grip with spacing)
    club = add_club_with_dual_hand_constraints(
        plant, left_hand, right_hand, params
    )

    # Ground and contact
    add_ground_and_club_contact(plant, scene_graph, club, params)

    # Actuators
    add_joint_actuators(plant)

    plant.Finalize()

    diagram = builder.Build()
    return diagram, plant, scene_graph
