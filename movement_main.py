#!/usr/bin/env python3

# Copyright 2024 Trossen Robotics
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#    * Neither the name of the copyright holder nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THinterbotix_common_modulesIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

if __package__:
    from .projection import adjust_3d_to_robot_frame, project_2d_to_sphere
else:
    from projection import adjust_3d_to_robot_frame, project_2d_to_sphere
from interbotix_common_modules.common_robot.robot import robot_shutdown, robot_startup
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from pathlib import Path
from dataclasses import dataclass

import serial
import numpy as np
import math


@dataclass(frozen=True)
class Parameters:
    trajectory_file: str = 'butterfly_trajectory.csv'
    robot_model: str = 'rx150'
    wp_period: float = 0.5
    wp_moving_time: float = 0.5
    cartesian_speed: float | None = 0.25
    initial_moving_time: float = 3.0
    acceleration_fraction: float = 1 / 32
    home_acceleration_fraction: float = 0.5
    minimum_period: float = 0.3
    fit_path_to_sphere: bool = False
    sphere_radius: float = 0.33
    camera_distance: float = 2.0
    image_distance: float = 0.0
    path_points_are_relative: bool = False
    precompute_ik: bool = True
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float | None = None
    light_enabled: bool = True
    color: tuple[int, int, int] = (255, 127, 0)
    start_color: tuple[int, int, int] = (0, 0, 0)
    serial_port: str = '/dev/ttyACM0'
    serial_baudrate: int = 115200
    serial_timeout: float = 1.0


def _cartesian_waypoints(bot, path, relative):
    """Resolve path points to absolute Cartesian waypoints."""
    current_pose = np.array(bot.arm.T_sb[:3, 3], dtype=float)
    waypoints = []
    for index, point in enumerate(path, start=1):
        point = np.asarray(point, dtype=float)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError(f"Path point {index} contains invalid coordinates: {point}")
        current_pose = current_pose + point if relative else point
        waypoints.append(current_pose.copy())
    return waypoints


def _cartesian_segment_periods(start_pose, waypoints, cartesian_speed, minimum_period):
    """Calculate waypoint intervals for an approximately constant Cartesian speed."""
    if cartesian_speed <= 0 or minimum_period <= 0:
        raise ValueError("Cartesian speed and minimum period must be positive")

    poses = np.asarray([start_pose, *waypoints], dtype=float)
    distances = np.linalg.norm(np.diff(poses, axis=0), axis=1)
    return np.maximum(distances / cartesian_speed, minimum_period).tolist()


def precompute_ik_path(bot, path, params):
    """Solve every Cartesian point before sending the first movement command."""
    joint_path = []
    custom_guess = None
    current_pose = np.array(bot.arm.T_sb[:3, 3], dtype=float)

    for index, point in enumerate(path, start=1):
        if len(point) != 3 or not all(
            math.isfinite(float(coordinate)) for coordinate in point
        ):
            raise ValueError(f"Path point {index} contains invalid coordinates: {point}")

        point = np.asarray(point, dtype=float)
        if params.path_points_are_relative:
            current_pose = current_pose + point
        else:
            current_pose = point

        joint_positions, success = bot.arm.set_ee_pose_components(
            x=current_pose[0],
            y=current_pose[1],
            z=current_pose[2],
            roll=params.roll,
            pitch=params.pitch,
            yaw=params.yaw,
            custom_guess=custom_guess,
            execute=False,
        )
        if not success:
            raise ValueError(f"IK failed for path point {index}: {point}")

        joint_path.append(joint_positions)
        custom_guess = joint_positions

    return joint_path


def _send_color(serial_connection, color):
    serial_connection.write(f"{color[0]},{color[1]},{color[2]}\n".encode())


def follow_path(
    bot,
    path,
    params,
    light_serial=None,
):
    """Follow Cartesian points, optionally treating them as relative moves.

    When ``cartesian_speed`` is set, each waypoint interval is calculated as
    ``max(distance / cartesian_speed, minimum_period)``. The precomputed path
    uses those intervals as trajectory timestamps; direct execution uses each
    calculated interval as that waypoint's moving time.
    """
    path = list(path)

    if params.precompute_ik:
        waypoint_periods = None
        if params.cartesian_speed is not None:
            cartesian_waypoints = _cartesian_waypoints(
                bot, path, params.path_points_are_relative
            )
            waypoint_periods = _cartesian_segment_periods(
                bot.arm.T_sb[:3, 3],
                cartesian_waypoints,
                params.cartesian_speed,
                params.minimum_period,
            )
        joint_path = precompute_ik_path(
            bot, path, params=params
        )
        follow_joint_path(
            bot,
            joint_path,
            params=params,
            waypoint_periods=waypoint_periods,
            light_serial=light_serial,
        )
        return

    current_pose = np.array(bot.arm.T_sb[:3, 3], dtype=float)
    if params.cartesian_speed is not None and (
        params.cartesian_speed <= 0 or params.minimum_period <= 0
    ):
        raise ValueError("Cartesian speed and minimum period must be positive")
    if params.initial_moving_time <= 0:
        raise ValueError("first_moving_time must be positive")
    for index, point in enumerate(
        _cartesian_waypoints(bot, path, params.path_points_are_relative)
    ):
        moving_time = params.wp_moving_time
        if index == 0:
            moving_time = params.initial_moving_time
        elif params.cartesian_speed is not None:
            distance = np.linalg.norm(point - current_pose)
            moving_time = max(
                distance / params.cartesian_speed, params.minimum_period
            )
        point = np.asarray(point, dtype=float)
        print(f"{point}")
        bot.arm.set_ee_pose_components(
            x=point[0],
            y=point[1],
            z=point[2],
            roll=params.roll,
            pitch=params.pitch,
            yaw=params.yaw,
            moving_time=moving_time,
        )
        current_pose = point



def follow_joint_path(
    bot,
    joint_path,
    params,
    waypoint_periods=None,
    light_serial=None,
):
    """Execute precomputed IK points with blocking joint-position commands.

    When ``waypoint_periods`` is supplied, each waypoint uses its own period.
    This keeps Cartesian-speed timing proportional to each segment instead of
    using the shortest segment duration for the whole path.
    """
    if not joint_path:
        raise ValueError("Cannot execute an empty joint path")
    if params.wp_period <= 0 or params.wp_moving_time <= 0:
        raise ValueError("Trajectory timing values must be positive")
    if params.initial_moving_time <= 0:
        raise ValueError("first_moving_time must be positive")
    if not 0 < params.acceleration_fraction <= 1:
        raise ValueError("acceleration_fraction must be between 0 and 1")
    has_waypoint_periods = waypoint_periods is not None
    if waypoint_periods is None:
        waypoint_periods = [params.wp_period] * len(joint_path)
    if len(waypoint_periods) != len(joint_path) or any(
        period <= 0 for period in waypoint_periods
    ):
        raise ValueError("There must be one positive period for each waypoint")
    first = True

    for index, joint_positions in enumerate(joint_path, start=1):
        print(f"Moving to precomputed point {index}: {joint_positions}")
        moving_time = (
            params.initial_moving_time
            if index == 1
            else waypoint_periods[index - 1]
            if has_waypoint_periods
            else params.wp_moving_time
        )
        if not bot.arm.set_joint_positions(
            joint_positions,
            moving_time=moving_time,
            blocking=True,
            accel_time=moving_time * params.acceleration_fraction,
        ):
            raise RuntimeError(f"Robot rejected precomputed point {index}")
        if first:
            if light_serial is not None:
                _send_color(light_serial, params.color)
            first = False




    

def main():
    params = Parameters(
        robot_model='rx150',
        trajectory_file='deer.csv',
        roll=0.0,
        pitch=0.0,
        yaw=None,

        wp_period=0.5,
        wp_moving_time=0.5,
        cartesian_speed=0.25,
        initial_moving_time=3.0,
        acceleration_fraction=1 / 8,
        minimum_period=0.25,

        home_acceleration_fraction=0.5,
        light_enabled=True,
        color=(0, 255, 0),
        serial_baudrate=115200,
        serial_timeout=1.0,
        serial_port='/dev/ttyACM0',
        fit_path_to_sphere=False,

        path_points_are_relative=False,
        precompute_ik=True,
    )

    trajectory_file = Path(__file__).resolve().parents[1] / 'trajectories' / params.trajectory_file
    light_serial = None
    if params.light_enabled:
        light_serial = serial.Serial(
            params.serial_port,
            params.serial_baudrate,
            timeout=params.serial_timeout,
        )
        _send_color(light_serial, params.start_color)

    # import path csv file
    path = np.loadtxt(
        trajectory_file,
        delimiter=',',
        skiprows=1,
        usecols=(2, 3, 4),
    )
    if params.fit_path_to_sphere:
        # The contour is stored in the Y/Z plane; use it as the projector's
        # horizontal/vertical image coordinates. Center the contour first:
        # the sphere is centered at the robot-frame offset, not at the CSV
        # coordinate origin.
        path_2d = np.column_stack(
            (
                -path[:, 1],
                path[:, 2] - np.mean(path[:, 2]),
            )
        )
        path = project_2d_to_sphere(
            path_2d,
                sphere_radius=0.33,
                camera_distance=2.0,
                image_distance=0.0,
        )
        path = adjust_3d_to_robot_frame(path)
    else:
        path = np.column_stack((-path[:, 1], path[:, 0], path[:, 2]))

    # initialize robot
    bot = InterbotixManipulatorXS(
        robot_model=params.robot_model,
        group_name='arm',
        gripper_name='gripper',
    )

    robot_startup()
    
    # save original moving time and accel time to restore later
    original_moving_time = bot.arm.moving_time
    original_accel_time = bot.arm.accel_time
    try:
        bot.arm.set_trajectory_time(
            params.initial_moving_time,
            params.initial_moving_time * params.home_acceleration_fraction,
        )
        bot.arm.go_to_home_pose()
    finally:
        bot.arm.set_trajectory_time(original_moving_time, original_accel_time)
    print("starting")
    print(path)

    follow_path(
        bot,
        path,
        params=params,
        light_serial=light_serial,
    )
    #turn off light at last point before returning to home pose
    if light_serial is not None:
        _send_color(light_serial, params.start_color)

    # Return the robot to a safe position slowly before shutting down.
    final_moving_time = params.initial_moving_time
    final_accel_time = params.initial_moving_time * params.home_acceleration_fraction
    original_moving_time = bot.arm.moving_time
    original_accel_time = bot.arm.accel_time
    try:
        bot.arm.set_trajectory_time(final_moving_time, final_accel_time)
        bot.arm.go_to_home_pose()
        bot.arm.go_to_sleep_pose()
    finally:
        bot.arm.set_trajectory_time(original_moving_time, original_accel_time)
        robot_shutdown()


if __name__ == '__main__':
    main()
    