#!/usr/bin/env python3
"""Record episodes for multiple tasks with camera indices resolved at runtime.

After each episode, choose a task: type a new one, pick from history, or keep the current one.

Controls (during recording):
  Right arrow  -> end episode early
  Left arrow   -> discard and re-record episode
  Escape       -> stop recording entirely
"""

import cv2

import rerun as rr
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.feature_utils import hw_to_dataset_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.so_follower import SO101Follower
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader
from lerobot.teleoperators.so_leader.config_so_leader import SO101LeaderConfig
from lerobot.processor import make_default_processors
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

ROBOT_PORT = "/dev/tty.usbmodem5B141136531"
LEADER_PORT = "/dev/tty.usbmodem5B141125311"

REPO_ID = "ehalicki/so101_multitask"
FPS = 30
EPISODE_TIME_S = 180
RESET_TIME_S = 5
NUM_EPISODES = 100

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


def pick_task(task_history: list[str], current_task: str | None) -> str:
    while True:
        print("\n" + "=" * 50)
        print("TASK SELECTION")
        if current_task:
            print(f"  Current: {current_task}")
        if task_history:
            print("  Previous tasks:")
            for i, t in enumerate(task_history):
                print(f"    [{i + 1}] {t}")
        print("  [Enter] Keep current task" if current_task else "  Type a new task description")
        if task_history:
            print("  [number] Select a previous task")
        print("  [text] Type a new task")
        print("=" * 50)

        choice = input("Task> ").strip()

        if not choice:
            if current_task:
                return current_task
            print("No task set. Please type a task description.")
            continue

        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(task_history):
                return task_history[idx]
            print(f"Invalid number. Choose 1-{len(task_history)}.")
            continue

        if len(choice) < 5:
            print(f"Task too short, please enter a real description.")
            continue

        return choice


def main():
    camera_configs = {}
    for cam_key, (default_w, default_h, cap_w, cap_h) in CAMERAS.items():
        idx = find_camera_by_resolution(default_w, default_h)
        print(f"{cam_key}: default {default_w}x{default_h} -> OpenCV index {idx}, capture {cap_w}x{cap_h}")
        camera_configs[cam_key] = OpenCVCameraConfig(
            index_or_path=idx, width=cap_w, height=cap_h, fps=FPS,
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

    from huggingface_hub.utils import RepositoryNotFoundError

    try:
        print(f"Attempting to resume dataset '{REPO_ID}' (local or Hub)...")
        dataset = LeRobotDataset.resume(
            repo_id=REPO_ID,
            image_writer_threads=4,
            streaming_encoding=True,
            vcodec="h264_videotoolbox",
        )
        print(f"Resumed dataset with {dataset.meta.total_episodes} episodes")
    except (RepositoryNotFoundError, FileNotFoundError):
        print(f"No existing dataset found, creating new one")
        action_features = hw_to_dataset_features(robot.action_features, "action")
        obs_features = hw_to_dataset_features(robot.observation_features, "observation")
        dataset_features = {**action_features, **obs_features}
        dataset = LeRobotDataset.create(
            repo_id=REPO_ID,
            fps=FPS,
            features=dataset_features,
            robot_type=robot.name,
            use_videos=True,
            image_writer_threads=4,
            streaming_encoding=True,
            vcodec="h264_videotoolbox",
        )

    listener, events = init_keyboard_listener()
    init_rerun(session_name="recording")

    robot.connect()
    teleop.connect()

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    task_history: list[str] = []
    current_task: str | None = None
    episode_idx = 0

    try:
        while episode_idx < NUM_EPISODES and not events["stop_recording"]:
            current_task = pick_task(task_history, current_task)
            if current_task not in task_history:
                task_history.append(current_task)

            log_say(f"Recording episode {episode_idx + 1}")
            print(f"Task: \"{current_task}\"")

            record_loop(
                robot=robot,
                events=events,
                fps=FPS,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                teleop=teleop,
                dataset=dataset,
                control_time_s=EPISODE_TIME_S,
                single_task=current_task,
                display_data=True,
            )

            if not events["stop_recording"] and (
                episode_idx < NUM_EPISODES - 1 or events["rerecord_episode"]
            ):
                log_say("Reset the environment")
                record_loop(
                    robot=robot,
                    events=events,
                    fps=FPS,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    control_time_s=RESET_TIME_S,
                    single_task=current_task,
                    display_data=True,
                )

            if events["rerecord_episode"]:
                log_say("Re-recording episode")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            dataset.save_episode()
            episode_idx += 1
            print(f"Saved episode {episode_idx}/{NUM_EPISODES}")

    except KeyboardInterrupt:
        pass
    finally:
        log_say("Stop recording")
        rr.rerun_shutdown()
        dataset.finalize()
        if robot.is_connected:
            robot.disconnect()
        if teleop.is_connected:
            teleop.disconnect()
        if listener:
            listener.stop()

    print(f"\nDone. Recorded {episode_idx} episodes across {len(task_history)} tasks.")
    print("Tasks recorded:")
    for t in task_history:
        print(f"  - {t}")


if __name__ == "__main__":
    main()
