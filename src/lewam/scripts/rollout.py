#!/usr/bin/env python3
"""LeWAM rollout client.

Captures observations from the robot, maintains a temporal context buffer,
sends full context to a remote inference server, and executes action chunks.
Task can be changed on the fly by typing a new task and pressing Enter.

Usage:
    python src/lewam/scripts/rollout.py
    python src/lewam/scripts/rollout.py --server eryk-pc --port 8080
"""

import argparse
import pickle
import socket
import struct
import threading
import time
from collections import deque

import cv2
import numpy as np
import rerun as rr

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.robots.so_follower.so_follower import SO101Follower

CAMERAS = {
    "image1": (1920, 1080, 640, 480),
    "image2": (1280, 960, 640, 480),
}
MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
VIDEO_STRIDE = 6  # 30fps native / 5fps model
N_CONTEXT = 8
N_ACTION_STEPS = 48
CONTROL_FPS = 30
GRIPPER_OFFSET = -3.0


def _recvall(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 4 * 1024 * 1024))
        if not chunk:
            raise ConnectionError("Connection closed")
        buf.extend(chunk)
    return bytes(buf)


def recv_msg(sock):
    raw_len = _recvall(sock, 4)
    length = struct.unpack(">I", raw_len)[0]
    return pickle.loads(_recvall(sock, length))


def send_msg(sock, obj):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack(">I", len(data)))
    sock.sendall(data)


def find_camera_by_resolution(target_w, target_h):
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            break
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if w == target_w and h == target_h:
            return i
    raise RuntimeError(f"No camera with default resolution {target_w}x{target_h}")


def obs_to_state(obs):
    state = np.array([obs[f"{m}.pos"] for m in MOTOR_NAMES], dtype=np.float32)
    #state[MOTOR_NAMES.index("gripper")] -= GRIPPER_OFFSET
    return state


def action_to_dict(action):
    return {f"{m}.pos": float(action[i]) for i, m in enumerate(MOTOR_NAMES)}


def encode_frames(buf, cam_keys):
    """JPEG-encode context frames for network transfer. Caches duplicates from padding."""
    result = {cam: [] for cam in cam_keys}
    jpg_cache = {}
    for frame_dict in buf:
        dict_id = id(frame_dict)
        if dict_id not in jpg_cache:
            encoded = {}
            for cam in cam_keys:
                bgr = cv2.cvtColor(frame_dict[cam], cv2.COLOR_RGB2BGR)
                _, jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                encoded[cam] = jpg.tobytes()
            jpg_cache[dict_id] = encoded
        for cam in cam_keys:
            result[cam].append(jpg_cache[dict_id][cam])
    return result


def task_input_loop(task_ref):
    while True:
        try:
            new = input()
            if new.strip():
                task_ref[0] = new.strip()
                print(f'Task changed to: "{task_ref[0]}"')
        except EOFError:
            break


def parse_args():
    p = argparse.ArgumentParser(description="LeWAM rollout client")
    p.add_argument("--server", default="eryk-pc")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--robot-port", default="/dev/tty.usbmodem5B141136531")
    p.add_argument("--ode-steps", type=int, default=None, help="Override ODE steps on server")
    p.add_argument("--cfg-scale", type=float, default=1.0, help="Classifier-free guidance scale (1.0=off, 2-5=stronger task conditioning)")
    return p.parse_args()


def main():
    args = parse_args()
    cam_keys = sorted(CAMERAS.keys())

    camera_configs = {}
    for cam_key, (default_w, default_h, cap_w, cap_h) in CAMERAS.items():
        idx = find_camera_by_resolution(default_w, default_h)
        print(f"{cam_key}: index {idx}, capture {cap_w}x{cap_h}")
        camera_configs[cam_key] = OpenCVCameraConfig(
            index_or_path=idx, width=cap_w, height=cap_h, fps=30,
        )

    robot = SO101Follower(SO101FollowerConfig(
        port=args.robot_port,
        id="eth_rl_so101_follower",
        cameras=camera_configs,
    ))
    robot.connect()
    print("Robot connected.")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.connect((args.server, args.port))
    print(f"Connected to server {args.server}:{args.port}")

    rr.init("rollout")
    rr.spawn(memory_limit="512MB")

    task = input("Enter task> ").strip()
    while not task:
        task = input("Enter task> ").strip()
    task_ref = [task]
    print(f'Task: "{task_ref[0]}"')
    print("Type a new task + Enter to change. Ctrl+C to stop.\n")

    threading.Thread(target=task_input_loop, args=(task_ref,), daemon=True).start()

    frame_buffer = deque(maxlen=N_CONTEXT)
    action_queue = deque()
    step = 0

    try:
        while True:
            t0 = time.perf_counter()
            obs = robot.get_observation()

            for cam in cam_keys:
                rr.log(f"cameras/{cam}", rr.Image(obs[cam]))

            if len(action_queue) > 0 and step < N_ACTION_STEPS:
                if step % VIDEO_STRIDE == 0:
                    frame_buffer.append({cam: obs[cam] for cam in cam_keys})
                action = action_queue.popleft()
                #action[MOTOR_NAMES.index("gripper")] += GRIPPER_OFFSET
                #if action[MOTOR_NAMES.index("gripper")] <= 23.0:
                    #action[MOTOR_NAMES.index("gripper")] = 10

                robot.send_action(action_to_dict(action))
                for i, motor in enumerate(MOTOR_NAMES):
                    rr.log(f"actions/{motor}", rr.Scalars(float(action[i])))
                step += 1
                print(f"Executed {step}/{N_ACTION_STEPS} actions")
            else:
                if len(frame_buffer) == 0:
                    frame_buffer.append({cam: obs[cam] for cam in cam_keys})

                buf = list(frame_buffer)
                while len(buf) < N_CONTEXT:
                    buf.insert(0, buf[0])

                msg = {
                    "frames": encode_frames(buf, cam_keys),
                    "state": obs_to_state(obs),
                    "task": task_ref[0],
                }
                if args.ode_steps is not None:
                    msg["ode_steps"] = args.ode_steps
                if args.cfg_scale != 1.0:
                    msg["cfg_scale"] = args.cfg_scale
                send_msg(sock, msg)
                resp = recv_msg(sock)
                action_queue.clear()
                action_queue.extend(resp["actions"])
                step = 0
                print(f"Got {len(resp['actions'])} actions")

                if "future_viz" in resp:
                    viz = resp["future_viz"]
                    for t in range(viz.shape[0]):
                        rr.log(f"predicted_future/t{t}", rr.Image(viz[t]))

            elapsed = time.perf_counter() - t0
            time.sleep(max(0, 1.0 / CONTROL_FPS - elapsed))
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        robot.disconnect()
        print("Stopped.")


if __name__ == "__main__":
    main()
