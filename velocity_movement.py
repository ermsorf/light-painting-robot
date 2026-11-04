#!/usr/bin/env python3
"""Follow a Cartesian CSV path using the Interbotix velocity operating mode."""

from pathlib import Path

import numpy as np
from rclpy.duration import Duration

from interbotix_common_modules.common_robot.robot import robot_shutdown, robot_startup
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS

if __package__:
    from .projection import adjust_3d_to_robot_frame, project_2d_to_sphere
else:
    from projection import adjust_3d_to_robot_frame, project_2d_to_sphere


def precompute_ik_path(bot, path, roll=0.0, pitch=0.0, yaw=None):
    """Solve all Cartesian points before changing the arm to velocity mode."""
    joint_path = []
    custom_guess = None
    current_pose = np.array(bot.arm.T_sb[:3, 3], dtype=float)

    for index, point in enumerate(path, start=1):
        point = np.asarray(point, dtype=float)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError(f"Path point {index} contains invalid coordinates: {point}")

        current_pose = point
        joint_positions, success = bot.arm.set_ee_pose_components(
            x=current_pose[0],
            y=current_pose[1],
            z=current_pose[2],
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            custom_guess=custom_guess,
            execute=False,
        )
        if not success:
            raise ValueError(f"IK failed for path point {index}: {point}")

        joint_path.append(np.asarray(joint_positions, dtype=float))
        custom_guess = joint_positions

    return joint_path


def _current_joint_positions(bot):
    arm = bot.arm
    with arm.core.js_mutex:
        return np.asarray(
            [
                arm.core.joint_states.position[arm.core.js_index_map[name]]
                for name in arm.group_info.joint_names
            ],
            dtype=float,
        )


def make_velocity_trajectory(
    joint_path,
    start_positions,
    max_joint_speed=0.8,
    minimum_period=0.03,
    stop_time=0.15,
):
    """Create timed joint velocity commands from a position path.

    The path is not expected to be hit exactly. Times are based on joint-space
    distance, and interior velocities use centered differences so they do not
    intentionally go to zero at every waypoint.
    """
    if max_joint_speed <= 0 or minimum_period <= 0 or stop_time <= 0:
        raise ValueError("Velocity trajectory timing values must be positive")

    positions = np.asarray([start_positions, *joint_path], dtype=float)
    if positions.ndim != 2 or positions.shape[0] < 2:
        raise ValueError("At least one path point is required")
    if not np.all(np.isfinite(positions)):
        raise ValueError("Joint path contains non-finite values")

    segment_distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    segment_times = np.maximum(segment_distances / max_joint_speed, minimum_period)
    times = np.concatenate(([0.0], np.cumsum(segment_times)))

    velocities = np.empty_like(positions)
    velocities[0] = (positions[1] - positions[0]) / segment_times[0]
    velocities[-1] = (positions[-1] - positions[-2]) / segment_times[-1]
    for index in range(1, len(positions) - 1):
        velocities[index] = (
            positions[index + 1] - positions[index - 1]
        ) / (times[index + 1] - times[index - 1])

    # Limit the whole joint vector together so clipping one joint does not
    # change the direction of motion at a waypoint.
    speed = np.linalg.norm(velocities, axis=1)
    scale = np.minimum(1.0, max_joint_speed / np.maximum(speed, 1e-12))
    velocities *= scale[:, None]

    trajectory = [
        {float(time): list(velocity)}
        for time, velocity in zip(times, velocities)
    ]
    trajectory.append({float(times[-1] + stop_time): [0.0] * positions.shape[1]})
    return trajectory, float(times[-1] + stop_time)


def follow_velocity_path(
    bot,
    path,
    max_joint_speed=0.8,
    minimum_period=0.03,
    stop_time=0.15,
    position_profile_velocity=2000,
    position_profile_acceleration=300,
):
    """Follow a Cartesian path approximately with continuous joint velocity."""
    joint_path = precompute_ik_path(bot, path)
    start_positions = _current_joint_positions(bot)
    trajectory, total_time = make_velocity_trajectory(
        joint_path,
        start_positions,
        max_joint_speed=max_joint_speed,
        minimum_period=minimum_period,
        stop_time=stop_time,
    )

    arm = bot.arm
    arm.core.robot_set_operating_modes(
        "group",
        arm.group_name,
        "velocity",
        profile_type="velocity",
    )
    try:
        arm.core.robot_write_trajectory(
            "group",
            arm.group_name,
            "velocity",
            trajectory,
        )
        print(
            f"Following {len(joint_path)} points in velocity mode for "
            f"{total_time:.2f}s (max joint speed={max_joint_speed:.2f} rad/s)"
        )
        arm.core.get_node().get_clock().sleep_for(Duration(seconds=total_time))
    finally:
        # Zero velocity does not hold a gravity-loaded arm. Capture the actual
        # pose, then restore position mode and command that pose as the hold
        # target before returning control to any other motion command.
        arm.core.robot_write_commands(arm.group_name, [0.0] * len(start_positions))
        arm.core.get_node().get_clock().sleep_for(Duration(seconds=0.05))
        hold_positions = _current_joint_positions(bot)
        arm.core.robot_set_operating_modes(
            "group",
            arm.group_name,
            "position",
            profile_type="time",
            profile_velocity=position_profile_velocity,
            profile_acceleration=position_profile_acceleration,
        )
        arm.core.robot_write_commands(arm.group_name, list(hold_positions))
        arm.joint_commands = list(hold_positions)
        arm._update_Tsb()
        print(
            "Velocity mode stopped; position mode is holding the measured "
            "final joint state. Do not send a home command automatically."
        )


def main():
    trajectory_file = Path(__file__).resolve().parents[1] / "trajectories" / "deer.csv"
    fit_path_to_sphere = False

    path = np.loadtxt(
        trajectory_file,
        delimiter=",",
        skiprows=1,
        usecols=(2, 3, 4),
    )
    if fit_path_to_sphere:
        path_2d = np.column_stack((-path[:, 1], path[:, 2]))
        path = adjust_3d_to_robot_frame(
            project_2d_to_sphere(
                path_2d,
                sphere_radius=0.33,
                camera_distance=2,
                image_distance=0.0,
            )
        )
    else:
        path = np.column_stack((-path[:, 1], path[:, 0], path[:, 2]))

    bot = InterbotixManipulatorXS(
        robot_model="rx150",
        group_name="arm",
        gripper_name="gripper",
    )

    robot_startup()
    try:
        bot.arm.go_to_home_pose()
        follow_velocity_path(
            bot,
            path,
            max_joint_speed=0.3,
            minimum_period=0.03,
            stop_time=0.15,
        )
    finally:
        robot_shutdown()


if __name__ == "__main__":
    main()
