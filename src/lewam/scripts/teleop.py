#!/usr/bin/env python3
"""Launch SO101 teleoperation with cameras resolved by default resolution at runtime.

Identifies cameras via OpenCV default resolution (no ffmpeg, no race condition).
"""

import cv2

import rerun as rr
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO101Follower
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader
from lerobot.teleoperators.so_leader.config_so_leader import SO101LeaderConfig
from lerobot.processor import make_default_processors
from lerobot.scripts.lerobot_teleoperate import teleop_loop
from lerobot.utils.visualization_utils import init_rerun

ROBOT_PORT = "/dev/tty.usbmodem5B141136531"
LEADER_PORT = "/dev/tty.usbmodem5B141125311"

# Map camera name -> (default_w, default_h, capture_w, capture_h)
CAMERAS = {
    "camera2": (1920, 1080, 640, 480),   # USB2.0_CAM1
    "camera3": (1280, 960, 640, 480),     # Logitech (VID:1133 PID:2085)
}


def find_camera_by_resolution(target_w: int, target_h: int) -> int:
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            break
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if w == target_w and h == target_h:
            return i
    raise RuntimeError(f"No camera found with default resolution {target_w}x{target_h}")


def main():
    camera_configs = {}
    for cam_key, (default_w, default_h, cap_w, cap_h) in CAMERAS.items():
        idx = find_camera_by_resolution(default_w, default_h)
        print(f"{cam_key}: default {default_w}x{default_h} -> OpenCV index {idx}, capture {cap_w}x{cap_h}")
        camera_configs[cam_key] = OpenCVCameraConfig(
            index_or_path=idx, width=cap_w, height=cap_h, fps=30,
        )

    robot = SO101Follower(SO101FollowerConfig(
        port=ROBOT_PORT,
        id="eth_rl_so101_follower",
        cameras=camera_configs,
    ))
    teleop = SO101Leader(SO101LeaderConfig(
        port=LEADER_PORT,
        id="eth_rl_so101_leader",
    ))

    init_rerun(session_name="teleoperation")
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    teleop.connect()
    robot.connect()

    try:
        teleop_loop(
            teleop=teleop,
            robot=robot,
            fps=30,
            display_data=True,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
        )
    except KeyboardInterrupt:
        pass
    finally:
        rr.rerun_shutdown()
        teleop.disconnect()
        robot.disconnect()


if __name__ == "__main__":
    main()
