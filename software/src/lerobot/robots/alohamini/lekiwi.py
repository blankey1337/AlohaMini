#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import logging
import os
import time
from functools import cached_property
from itertools import chain
from typing import Any
import sys

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_lekiwi import LeKiwiConfig, LeKiwiClientConfig

logger = logging.getLogger(__name__)

from .lift_axis import LiftAxis, LiftAxisConfig


class LeKiwi(Robot):
    """
    The robot includes a three omniwheel mobile base and a remote follower arm.
    The leader arm is connected locally (on the laptop) and its joint positions are recorded and then
    forwarded to the remote follower arm (after applying a safety clamp).
    In parallel, keyboard teleoperation is used to generate raw velocity commands for the wheels.
    """

    config_class = LeKiwiConfig
    name = "lekiwi"

    def __init__(self, config: LeKiwiConfig):
        super().__init__(config)
        self.config = config
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100


        self.left_bus = FeetechMotorsBus(
            port=self.config.left_port,
            motors={
                # arm
                "arm_left_shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "arm_left_shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "arm_left_elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "arm_left_wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "arm_left_wrist_roll": Motor(5, "sts3215", norm_mode_body),
                "arm_left_gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
                # base
                "base_left_wheel": Motor(8, "sts3215", MotorNormMode.RANGE_M100_100),
                "base_back_wheel": Motor(9, "sts3215", MotorNormMode.RANGE_M100_100),
                "base_right_wheel": Motor(10, "sts3215", MotorNormMode.RANGE_M100_100),
                "lift_axis": Motor(11, "sts3215", MotorNormMode.DEGREES),
            },
            calibration=self.calibration,
        )

        self.right_bus = FeetechMotorsBus(
            port=self.config.right_port,
            motors={
                # arm
                "arm_right_shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "arm_right_shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "arm_right_elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "arm_right_wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "arm_right_wrist_roll": Motor(5, "sts3215", norm_mode_body),
                "arm_right_gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )


        self.left_arm_motors  = [m for m in self.left_bus.motors        if m.startswith("arm_left_")]
        self.base_motors      = [m for m in self.left_bus.motors        if m.startswith("base_")]

        self.right_arm_motors = [m for m in (self.right_bus.motors if self.right_bus else []) if m.startswith("arm_right_")]

        self.cameras = make_cameras_from_configs(config.cameras)


        self.lift = LiftAxis(
        LiftAxisConfig(),        
        bus_left=self.left_bus,
        bus_right=self.right_bus,
)


    @property
    def _state_ft(self) -> dict[str, type]:
        return dict.fromkeys(
            (
                "arm_left_shoulder_pan.pos",
                "arm_left_shoulder_lift.pos",
                "arm_left_elbow_flex.pos",
                "arm_left_wrist_flex.pos",
                #"left_wrist_yaw.pos",
                "arm_left_wrist_roll.pos",
                "arm_left_gripper.pos",
                "arm_right_shoulder_pan.pos",
                "arm_right_shoulder_lift.pos",
                "arm_right_elbow_flex.pos",
                "arm_right_elbow_flex.pos", # Duplicate removed? No, check original
                "arm_right_wrist_flex.pos",
                #"right_wrist_yaw.pos",
                "arm_right_wrist_roll.pos",
                "arm_right_gripper.pos",
                "x.vel",
                "y.vel",
                "theta.vel",
                "lift_axis.height_mm",   # Added
                #"lift_axis.vel",         # Added (optional, for debugging)
            ),
            float,
        )

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._state_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._state_ft

    @property
    def is_connected(self) -> bool:
        cams_ok = all(cam.is_connected for cam in self.cameras.values())
        return self.left_bus.is_connected and (self.right_bus.is_connected if self.right_bus else True) and cams_ok



    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.left_bus.connect()
        self.right_bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

        self.lift.home()
        logger.info("Lift axis homed to 0mm.")

        

    @property
    def is_calibrated(self) -> bool:
        return self.left_bus.is_calibrated

    def calibrate(self) -> None:
        """
        Dual-arm calibration (left arm + chassis on self.left_bus, right arm on self.right_bus):
        - Left arm: position mode → half-turn homing → collect ROM
        - Chassis: no homing; ROM fixed to 0–4095
        - Right arm (if present): position mode → half-turn homing → collect ROM
        - Merge into a single self.calibration, split by bus, write back to both buses, and save
        """
        # If a calibration file already exists: load it and write back, filtering for each bus separately
        if self.calibration:
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, "
                f"or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info("Writing existing calibration to both buses (trim per-bus caches)")

                calib_left = {k: v for k, v in self.calibration.items() if k in self.left_bus.motors}
                self.left_bus.write_calibration(calib_left, cache=False)
                self.left_bus.calibration = calib_left

                if getattr(self, "right_bus", None):
                    calib_right = {k: v for k, v in self.calibration.items() if k in self.right_bus.motors}
                    self.right_bus.write_calibration(calib_right, cache=False)
                    self.right_bus.calibration = calib_right

                return

        logger.info(f"\nRunning calibration of {self} (dual-bus if right_bus present)")

        if not getattr(self, "left_arm_motors", None):
            raise RuntimeError("left_arm_motors is empty; expected names starting with 'left_arm_'")

        self.left_bus.disable_torque(self.left_arm_motors)
        for name in self.left_arm_motors:
            self.left_bus.write("Operating_Mode", name, OperatingMode.POSITION.value)

        input("Move LEFT arm to the middle of its range of motion, then press ENTER...")
        left_homing = self.left_bus.set_half_turn_homings(self.left_arm_motors)  # Only left arm items

        for wheel in self.base_motors:
            left_homing[wheel] = 0

        motors_left_all = self.left_arm_motors + self.base_motors
        full_turn_left = [m for m in motors_left_all if m.startswith("base_")]  # 三个轮子
        unknown_left = [m for m in motors_left_all if m not in full_turn_left]

        print("Move LEFT arm joints sequentially through full ROM. Press ENTER to stop...")
        l_mins, l_maxs = self.left_bus.record_ranges_of_motion(unknown_left)
        for m in full_turn_left:
            l_mins[m] = 0
            l_maxs[m] = 4095

        right_homing = {}
        r_mins, r_maxs = {}, {}

        if getattr(self, "right_bus", None) and getattr(self, "right_arm_motors", None):
            self.right_bus.disable_torque(self.right_arm_motors)
            for name in self.right_arm_motors:
                self.right_bus.write("Operating_Mode", name, OperatingMode.POSITION.value)

            input("Move RIGHT arm to the middle of its range of motion, then press ENTER...")
            right_homing = self.right_bus.set_half_turn_homings(self.right_arm_motors)

            print("Move RIGHT arm joints sequentially through full ROM. Press ENTER to stop...")
            r_mins, r_maxs = self.right_bus.record_ranges_of_motion(self.right_arm_motors)

        # Merge → filter by bus and write back → save as a single file
        self.calibration = {}

        for name, motor in self.left_bus.motors.items():
            self.calibration[name] = MotorCalibration(
                id=motor.id,
                drive_mode=0,
                homing_offset=left_homing.get(name, 0),
                range_min=l_mins.get(name, 0),
                range_max=l_maxs.get(name, 4095),
            )

        if getattr(self, "right_bus", None):
            for name, motor in self.right_bus.motors.items():
                self.calibration[name] = MotorCalibration(
                    id=motor.id,
                    drive_mode=0,
                    homing_offset=right_homing.get(name, 0),
                    range_min=r_mins.get(name, 0),
                    range_max=r_maxs.get(name, 4095),
                )

        # Write back: each bus only writes its own entries to avoid KeyError
        calib_left = {k: v for k, v in self.calibration.items() if k in self.left_bus.motors}
        self.left_bus.write_calibration(calib_left, cache=False)
        self.left_bus.calibration = calib_left

        if getattr(self, "right_bus", None):
            calib_right = {k: v for k, v in self.calibration.items() if k in self.right_bus.motors}
            self.right_bus.write_calibration(calib_right, cache=False)
            self.right_bus.calibration = calib_right

        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)





    def configure(self):
        # Set-up arm actuators (position mode)
        # We assume that at connection time, arm is in a rest position,
        # and torque can be safely disabled to run calibration.
        self.left_bus.disable_torque()
        self.left_bus.configure_motors()
        for name in self.left_arm_motors:
            self.left_bus.write("Operating_Mode", name, OperatingMode.POSITION.value)
            # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
            self.left_bus.write("P_Coefficient", name, 16)
            # Set I_Coefficient and D_Coefficient to default value 0 and 32
            self.left_bus.write("I_Coefficient", name, 0)
            self.left_bus.write("D_Coefficient", name, 32)

        for name in self.base_motors:
            self.left_bus.write("Operating_Mode", name, OperatingMode.VELOCITY.value)

        self.right_bus.disable_torque()
        self.right_bus.configure_motors()
        for name in self.right_arm_motors:
            self.right_bus.write("Operating_Mode", name, OperatingMode.POSITION.value)
            self.right_bus.write("P_Coefficient", name, 16)
            self.right_bus.write("I_Coefficient", name, 0)
            self.right_bus.write("D_Coefficient", name, 32)





    def setup_motors(self) -> None:
        for motor in chain(reversed(self.arm_motors), reversed(self.base_motors)):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.left_bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.left_bus.motors[motor].id}")

    @staticmethod
    def _degps_to_raw(degps: float) -> int:
        steps_per_deg = 4096.0 / 360.0
        speed_in_steps = degps * steps_per_deg
        speed_int = int(round(speed_in_steps))
        # Cap the value to fit within signed 16-bit range (-32768 to 32767)
        if speed_int > 0x7FFF:
            speed_int = 0x7FFF  # 32767 -> maximum positive value
        elif speed_int < -0x8000:
            speed_int = -0x8000  # -32768 -> minimum negative value
        return speed_int

    @staticmethod
    def _raw_to_degps(raw_speed: int) -> float:
        steps_per_deg = 4096.0 / 360.0
        magnitude = raw_speed
        degps = magnitude / steps_per_deg
        return degps

    def _body_to_wheel_raw(
        self,
        x: float,
        y: float,
        theta: float,
        wheel_radius: float = 0.05,
        base_radius: float = 0.125,
        max_raw: int = 3000,
    ) -> dict:
        """
        Convert desired body-frame velocities into wheel raw commands.

        Parameters:
          x_cmd      : Linear velocity in x (m/s).
          y_cmd      : Linear velocity in y (m/s).
          theta_cmd  : Rotational velocity (deg/s).
          wheel_radius: Radius of each wheel (meters).
          base_radius : Distance from the center of rotation to each wheel (meters).
          max_raw    : Maximum allowed raw command (ticks) per wheel.

        Returns:
          A dictionary with wheel raw commands:
             {"base_left_wheel": value, "base_back_wheel": value, "base_right_wheel": value}.

        Notes:
          - Internally, the method converts theta_cmd to rad/s for the kinematics.
          - The raw command is computed from the wheels angular speed in deg/s
            using _degps_to_raw(). If any command exceeds max_raw, all commands
            are scaled down proportionally.
        """
        # Convert rotational velocity from deg/s to rad/s.
        theta_rad = theta * (np.pi / 180.0)
        # Create the body velocity vector [x, y, theta_rad].
        velocity_vector = np.array([-x, -y, theta_rad])

        # Define the wheel mounting angles with a -90° offset.
        angles = np.radians(np.array([240, 0, 120]) - 90)
        # Build the kinematic matrix: each row maps body velocities to a wheel’s linear speed.
        # The third column (base_radius) accounts for the effect of rotation.
        m = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])

        # Compute each wheel’s linear speed (m/s) and then its angular speed (rad/s).
        wheel_linear_speeds = m.dot(velocity_vector)
        wheel_angular_speeds = wheel_linear_speeds / wheel_radius

        # Convert wheel angular speeds from rad/s to deg/s.
        wheel_degps = wheel_angular_speeds * (180.0 / np.pi)

        # Scaling
        steps_per_deg = 4096.0 / 360.0
        raw_floats = [abs(degps) * steps_per_deg for degps in wheel_degps]
        max_raw_computed = max(raw_floats)
        if max_raw_computed > max_raw:
            scale = max_raw / max_raw_computed
            wheel_degps = wheel_degps * scale

        # Convert each wheel’s angular speed (deg/s) to a raw integer.
        wheel_raw = [self._degps_to_raw(deg) for deg in wheel_degps]

        return {
            "base_left_wheel": wheel_raw[0],
            "base_back_wheel": wheel_raw[1],
            "base_right_wheel": wheel_raw[2],
        }

    def _wheel_raw_to_body(
        self,
        left_wheel_speed,
        back_wheel_speed,
        right_wheel_speed,
        wheel_radius: float = 0.05,
        base_radius: float = 0.125,
    ) -> dict[str, Any]:
        """
        Convert wheel raw command feedback back into body-frame velocities.

        Parameters:
          wheel_raw   : Vector with raw wheel commands ("base_left_wheel", "base_back_wheel", "base_right_wheel").
          wheel_radius: Radius of each wheel (meters).
          base_radius : Distance from the robot center to each wheel (meters).

        Returns:
          A dict (x.vel, y.vel, theta.vel) all in m/s
        """

        # Convert each raw command back to an angular speed in deg/s.
        wheel_degps = np.array(
            [
                self._raw_to_degps(left_wheel_speed),
                self._raw_to_degps(back_wheel_speed),
                self._raw_to_degps(right_wheel_speed),
            ]
        )

        # Convert from deg/s to rad/s.
        wheel_radps = wheel_degps * (np.pi / 180.0)
        # Compute each wheel’s linear speed (m/s) from its angular speed.
        wheel_linear_speeds = wheel_radps * wheel_radius

        # Define the wheel mounting angles with a -90° offset.
        angles = np.radians(np.array([240, 0, 120]) - 90)
        m = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])

        # Solve the inverse kinematics: body_velocity = M⁻¹ · wheel_linear_speeds.
        m_inv = np.linalg.inv(m)
        velocity_vector = m_inv.dot(wheel_linear_speeds)
        x, y, theta_rad = velocity_vector
        
        theta = theta_rad * (180.0 / np.pi)
        return {
            "x.vel": x,
            "y.vel": y,
            "theta.vel": theta,
        }  # m/s and deg/s
    
    def _raw_to_ma(raw):
        try:
            return float(raw) * 6.5
        except Exception:
            return 0.0
        
    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read actuators position for arm and vel for base
        start = time.perf_counter()
        
        left_pos = self.left_bus.sync_read("Present_Position", self.left_arm_motors)   # left_arm_*


        base_wheel_vel = self.left_bus.sync_read("Present_Velocity", self.base_motors)

        base_vel = self._wheel_raw_to_body(
            base_wheel_vel["base_left_wheel"],
            base_wheel_vel["base_back_wheel"],
            base_wheel_vel["base_right_wheel"],
        )

        right_pos = self.right_bus.sync_read("Present_Position", self.right_arm_motors)  # right_arm_*


        left_arm_state = {f"{k}.pos": v for k, v in left_pos.items()}
        right_arm_state = {f"{k}.pos": v for k, v in right_pos.items()}

        obs_dict = {**left_arm_state, **right_arm_state,**base_vel}
        self.lift.contribute_observation(obs_dict)

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # currents protection
        self.read_and_check_currents(limit_ma=2000, print_currents=True)

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Command AlohaMini to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Raises:
            RobotDeviceNotConnectedError: if robot is not connected.

        Returns:
            np.ndarray: the action sent to the motors, potentially clipped.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # arm_goal_pos = {k: v for k, v in action.items() if k.endswith(".pos")}
        left_pos  = {k: v for k, v in action.items() if k.endswith(".pos") and k.startswith("arm_left_")}
        right_pos = {k: v for k, v in action.items() if k.endswith(".pos") and k.startswith("arm_right_")}


        base_goal_vel = {k: v for k, v in action.items() if k.endswith(".vel")}

        base_wheel_goal_vel = self._body_to_wheel_raw(
            base_goal_vel["x.vel"], base_goal_vel["y.vel"], base_goal_vel["theta.vel"]
        )

        # Cap goal position when too far away from present position.
        # /!\ Slower fps expected due to reading from the follower.
        
        self.lift.apply_action(action)

        if left_pos and self.config.max_relative_target is not None:
            present_left = self.left_bus.sync_read("Present_Position", self.left_arm_motors)  # left_arm_*
            gp_left = {k: (v, present_left[k.replace(".pos", "")]) for k, v in left_pos.items()}
            left_pos = ensure_safe_goal_position(gp_left, self.config.max_relative_target)

        if self.right_bus and right_pos and self.config.max_relative_target is not None:
            present_right = self.right_bus.sync_read("Present_Position", self.right_arm_motors)
            gp_right = {k: (v, present_right[k.replace(".pos", "")]) for k, v in right_pos.items()}
            right_pos = ensure_safe_goal_position(gp_right, self.config.max_relative_target)


        # Send goal position to the actuators
    
        if left_pos:
            self.left_bus.sync_write("Goal_Position", {k.replace(".pos", ""): v for k, v in left_pos.items()})
        if self.right_bus and right_pos:
            self.right_bus.sync_write("Goal_Position", {k.replace(".pos", ""): v for k, v in right_pos.items()})
        self.left_bus.sync_write("Goal_Velocity", base_wheel_goal_vel)

        lift_sent = {k: v for k, v in action.items() if k.startswith("lift_axis.")}
        return {**left_pos, **right_pos, **base_goal_vel, **lift_sent}


    def stop_base(self):
        self.left_bus.sync_write("Goal_Velocity", dict.fromkeys(self.base_motors, 0), num_retry=0)
        logger.info("Base motors stopped")

    def read_and_check_currents(self, limit_ma, print_currents):
        """Read left/right bus currents (mA), print them, and enforce overcurrent protection"""
        scale = 6.5  # sts3215 current unit conversion coefficient
        left_curr_raw = {}
        left_curr_raw = self.left_bus.sync_read("Present_Current", list(self.left_bus.motors.keys()))
        right_curr_raw = {}
        if getattr(self, "right_bus", None):
            right_curr_raw = self.right_bus.sync_read("Present_Current", list(self.right_bus.motors.keys()))

        if print_currents:
            left_line = "{" + ",".join(str(int(v * scale)) for v in left_curr_raw.values()) + "}"
            logger.info(f"Left Bus currents: {left_line}")
            if right_curr_raw:
                right_line = "{" + ",".join(str(int(v * scale)) for v in right_curr_raw.values()) + "}"
                logger.info(f"Right Bus currents: {right_line}")

        for name, raw in {**left_curr_raw, **right_curr_raw}.items():
            current_ma = float(raw) * scale
            if current_ma > limit_ma:
                logger.error(f"[Overcurrent] {name}: {current_ma:.1f} mA > {limit_ma:.1f} mA, disconnecting!")
                try:
                    self.stop_base()
                except Exception:
                    pass
                try:
                    self.disconnect()
                except Exception as e:
                    logger.error(f"[Overcurrent] disconnect error: {e}")
                sys.exit(1)

        return {k: round(v * scale, 1) for k, v in {**left_curr_raw, **right_curr_raw}.items()}

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.stop_base()
        self.left_bus.disconnect(self.config.disable_torque_on_disconnect)
        self.right_bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")


class LeKiwiSim(Robot):
    """
    Simulation version of LeKiwi for testing without hardware.
    Maintains a kinematic state and updates it based on velocity commands.
    """
    config_class = LeKiwiClientConfig
    name = "lekiwi_sim"

    def __init__(self, config: LeKiwiClientConfig):
        super().__init__(config)
        self.config = config
        self._is_connected = False
        
        # State
        self.state = {
            "x": 0.0,
            "y": 0.0,
            "theta": 0.0,
        }
        
        # Joint positions
        self.joints = {}
        # Populate with initial 0s for all expected joints from LeKiwi definition
        # Manually listing keys to ensure consistency
        joint_names = [
                "arm_left_shoulder_pan",
                "arm_left_shoulder_lift",
                "arm_left_elbow_flex",
                "arm_left_wrist_flex",
                "arm_left_wrist_roll",
                "arm_left_gripper",
                "arm_right_shoulder_pan",
                "arm_right_shoulder_lift",
                "arm_right_elbow_flex",
                "arm_right_wrist_flex",
                "arm_right_wrist_roll",
                "arm_right_gripper",
        ]
        self.joints.update({k: 0.0 for k in joint_names})
        self.joints["lift_axis"] = 0.0
        
        # Teleop config
        self.teleop_keys = config.teleop_keys
        self.speed_levels = [
            {"xy": 0.15, "theta": 45},
            {"xy": 0.2, "theta": 60},
            {"xy": 0.25, "theta": 75},
        ]
        self.speed_index = 0
        
        # Last update time
        self.last_update = time.perf_counter()

    @property
    def _state_ft(self) -> dict[str, type]:
        return dict.fromkeys(
            (
                "arm_left_shoulder_pan.pos",
                "arm_left_shoulder_lift.pos",
                "arm_left_elbow_flex.pos",
                "arm_left_wrist_flex.pos",
                "arm_left_wrist_roll.pos",
                "arm_left_gripper.pos",
                "arm_right_shoulder_pan.pos",
                "arm_right_shoulder_lift.pos",
                "arm_right_elbow_flex.pos",
                "arm_right_wrist_flex.pos",
                "arm_right_wrist_roll.pos",
                "arm_right_gripper.pos",
                "x.vel",
                "y.vel",
                "theta.vel",
                "lift_axis.height_mm",
            ),
            float,
        )

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.config.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._state_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._state_ft

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self):
        logger.info(f"{self} simulated connection established.")
        self._is_connected = True

    def disconnect(self):
        logger.info(f"{self} simulated connection closed.")
        self._is_connected = False
        
    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Return current state
        obs = {}
        # Joints
        for k, v in self.joints.items():
            if k == "lift_axis": continue
            obs[f"{k}.pos"] = v
            
        # Base vel (simulated as instantaneous match to command for now, or 0 if stopped)
        obs["x.vel"] = 0.0 
        obs["y.vel"] = 0.0
        obs["theta.vel"] = 0.0
        
        # Lift
        obs["lift_axis.height_mm"] = self.joints["lift_axis"]
        
        # Cameras (blank)
        for cam in self.config.cameras:
            h, w = self.config.cameras[cam].height, self.config.cameras[cam].width
            obs[cam] = np.zeros((h, w, 3), dtype=np.uint8)

        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
            
        now = time.perf_counter()
        dt = now - self.last_update
        self.last_update = now
        
        # Update base pose (simple integration)
        vx = action.get("x.vel", 0.0)
        vy = action.get("y.vel", 0.0)
        vth = action.get("theta.vel", 0.0)
        
        # Rotate velocity to world frame
        rad = np.radians(self.state["theta"])
        c, s = np.cos(rad), np.sin(rad)
        dx = (vx * c - vy * s) * dt
        dy = (vx * s + vy * c) * dt
        dth = vth * dt
        
        self.state["x"] += dx
        self.state["y"] += dy
        self.state["theta"] += dth
        
        # Update joints (instant move to target for simplicity, or LPF)
        for k, v in action.items():
            if k.endswith(".pos"):
                joint = k.replace(".pos", "")
                self.joints[joint] = v
            elif k == "lift_axis.height_mm":
                self.joints["lift_axis"] = v
                
        return action
        
    def _from_keyboard_to_base_action(self, pressed_keys: np.ndarray):
        # Speed control
        if self.teleop_keys["speed_up"] in pressed_keys:
            self.speed_index = min(self.speed_index + 1, 2)
        if self.teleop_keys["speed_down"] in pressed_keys:
            self.speed_index = max(self.speed_index - 1, 0)
        speed_setting = self.speed_levels[self.speed_index]
        xy_speed = speed_setting["xy"]
        theta_speed = speed_setting["theta"]

        x_cmd = 0.0
        y_cmd = 0.0
        theta_cmd = 0.0

        if self.teleop_keys["forward"] in pressed_keys:
            x_cmd += xy_speed
        if self.teleop_keys["backward"] in pressed_keys:
            x_cmd -= xy_speed
        if self.teleop_keys["left"] in pressed_keys:
            y_cmd += xy_speed
        if self.teleop_keys["right"] in pressed_keys:
            y_cmd -= xy_speed
        if self.teleop_keys["rotate_left"] in pressed_keys:
            theta_cmd += theta_speed
        if self.teleop_keys["rotate_right"] in pressed_keys:
            theta_cmd -= theta_speed

        return {
            "x.vel": x_cmd,
            "y.vel": y_cmd,
            "theta.vel": theta_cmd,
        }
    
    def _from_keyboard_to_lift_action(self, pressed_keys: np.ndarray):
        up_pressed = self.teleop_keys.get("lift_up", "u") in pressed_keys
        dn_pressed = self.teleop_keys.get("lift_down", "j") in pressed_keys

        h_now = float(self.joints["lift_axis"])

        if not (up_pressed or dn_pressed):
            return {"lift_axis.height_mm": h_now}

        # Increment on each key press
        if up_pressed and not dn_pressed:
            h_target = h_now + LiftAxisConfig.step_mm
        elif dn_pressed and not up_pressed:
            h_target = h_now - LiftAxisConfig.step_mm
        else:
            h_target = h_now

        return {"lift_axis.height_mm": h_target}
