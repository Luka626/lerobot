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

"""
Simple script to control a robot from teleoperation.

Example:

```shell
python -m lerobot.teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=blue \
    --display_data=true
```
"""

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat

import draccus
import numpy as np
import rerun as rr

from lerobot.common.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.common.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.common.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.common.teleoperators import (
    Teleoperator,
    TeleoperatorConfig,
    make_teleoperator_from_config,
)
from lerobot.common.utils.robot_utils import busy_wait
from lerobot.common.utils.utils import init_logging, move_cursor_up
from lerobot.common.utils.visualization_utils import _init_rerun

from .common.teleoperators import koch_leader, so100_leader, so101_leader  # noqa: F401


@dataclass
class TeleoperateConfig:
    teleop_1: TeleoperatorConfig
    teleop_2: TeleoperatorConfig
    robot_1: RobotConfig
    robot_2: RobotConfig
    # Limit the maximum frames per second.
    fps: int = 60
    teleop_time_s: float | None = None
    # Display all cameras on screen
    display_data: bool = False


def teleop_loop(
    teleop_1: Teleoperator, teleop_2: Teleoperator, robot_1: Robot, robot_2: Robot, fps: int, display_data: bool = False, duration: float | None = None
):
    # display_len = max(len(key) for key in robot.action_features)
    display_len = max(len(key) for robot in (robot_1, robot_2) for key in robot.action_features)
    start = time.perf_counter()
    while True:
        loop_start = time.perf_counter()
        action_1 = teleop_1.get_action()
        action_2 = teleop_2.get_action()
        actions = (action_1, action_2)
        robots = (robot_1, robot_2)
        if display_data:
            for robot,action in zip(robots, actions):
                observation = robot.get_observation()
                for obs, val in observation.items():
                    if isinstance(val, float):
                        rr.log(f"observation_{robot.id}_{obs}", rr.Scalar(val))
                    elif isinstance(val, np.ndarray):
                        rr.log(f"observation_{robot.id}_{obs}", rr.Image(val), static=True)
            for act, val in action.items():
                if isinstance(val, float):
                    rr.log(f"action_{act}", rr.Scalar(val))

        for robot, action in zip(robots, actions):
            robot.send_action(action)
        dt_s = time.perf_counter() - loop_start
        busy_wait(1 / fps - dt_s)

        loop_s = time.perf_counter() - loop_start

        print("\n" + "-" * (display_len + 10))
        print(f"{'NAME':<{display_len}} | {'NORM':>7}")
        for motor, value in action.items():
            print(f"{motor:<{display_len}} | {value:>7.2f}")
        print(f"\ntime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")

        if duration is not None and time.perf_counter() - start >= duration:
            return

        move_cursor_up(len(action) + 5)


@draccus.wrap()
def teleoperate(cfg: TeleoperateConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        _init_rerun(session_name="teleoperation")

    teleop_1 = make_teleoperator_from_config(cfg.teleop_1)
    teleop_2 = make_teleoperator_from_config(cfg.teleop_2)
    robot_1 = make_robot_from_config(cfg.robot_1)
    robot_2 = make_robot_from_config(cfg.robot_2)

    teleop_1.connect()
    teleop_2.connect()
    robot_1.connect()
    robot_2.connect()

    try:
        teleop_loop(teleop_1, teleop_2, robot_1, robot_2, cfg.fps, display_data=cfg.display_data, duration=cfg.teleop_time_s)
    except KeyboardInterrupt:
        pass
    finally:
        if cfg.display_data:
            rr.rerun_shutdown()
        teleop_1.disconnect()
        teleop_2.disconnect()
        robot_1.disconnect()
        robot_2.disconnect()


if __name__ == "__main__":
    teleoperate()

