import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image
from si190c_interfaces.srv import SetPosition

from handeye_calibration.aruco_detector import detect_aruco_pose
from handeye_calibration.utils import (
    load_calibration_data,
    load_intrinsics,
    pose_to_matrix,
    save_calibration_data,
)


DEFAULT_POSES = [
    {'position': [-0.0001, -0.0020, 0.3323], 'orientation': [0.2710, 0.2712, -0.6526, 0.6536]},
]


class DataCollector(Node):
    def __init__(self, args):
        super().__init__('handeye_data_collector')
        self.args = args
        self.bridge = CvBridge()
        self.camera_matrix, self.dist_coeffs = load_intrinsics(args.intrinsics)
        self.latest_image = None
        self.latest_fk_pose = None
        self.base_to_ee = []
        self.cam_to_marker = []

        if args.append and Path(args.output).exists():
            self.base_to_ee, self.cam_to_marker = load_calibration_data(args.output)
            self.base_to_ee = list(self.base_to_ee)
            self.cam_to_marker = list(self.cam_to_marker)
            self.get_logger().info(f'Loaded {len(self.base_to_ee)} existing samples.')

        self.image_sub = self.create_subscription(Image, args.image_topic, self.image_callback, 10)
        self.fk_sub = self.create_subscription(Pose, args.fk_topic, self.fk_callback, 10)
        self.move_client = self.create_client(SetPosition, args.service_name)
        self.get_logger().info(f'Listening on {args.image_topic} and {args.fk_topic}')

    def image_callback(self, msg):
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def fk_callback(self, msg):
        self.latest_fk_pose = msg

    def move_to_pose(self, position, orientation, timeout_sec=5.0):
        if not self.move_client.wait_for_service(timeout_sec=timeout_sec):
            self.get_logger().error(f'Service {self.args.service_name} is not available.')
            return False

        request = SetPosition.Request()
        request.x, request.y, request.z = [float(v) for v in position]
        request.qx, request.qy, request.qz, request.qw = [float(v) for v in orientation]
        future = self.move_client.call_async(request)
        start = time.time()
        while rclpy.ok() and not future.done():
            if time.time() - start > timeout_sec:
                self.get_logger().error('Timed out while waiting for /set_position response.')
                return False
            time.sleep(0.05)

        response = future.result()
        if response is None or not response.success:
            self.get_logger().error(f'Move request failed: {response}')
            return False

        self.get_logger().info(response.message)
        return True

    def detect_latest_marker(self):
        if self.latest_image is None:
            return None, None, None
        image = self.latest_image.copy()
        T_cm, ids, _, pose = detect_aruco_pose(
            image,
            self.camera_matrix,
            self.dist_coeffs,
            marker_length=self.args.marker_length,
            dictionary_name=self.args.dictionary,
            marker_id=self.args.marker_id,
        )
        reprojection = pose[2] if pose is not None and len(pose) > 2 else None
        return T_cm, ids, reprojection

    def capture_stable_marker_pose(self):
        transforms = []
        reprojections = []
        last_ids = None

        for _ in range(self.args.stable_frames):
            T_cm, ids, reprojection = self.detect_latest_marker()
            if ids is not None:
                last_ids = ids
            if T_cm is not None:
                transforms.append(T_cm)
                if reprojection is not None:
                    reprojections.append(float(reprojection))
            time.sleep(self.args.stable_delay)

        if len(transforms) < self.args.min_stable_frames:
            self.get_logger().warning(
                f'Only {len(transforms)}/{self.args.stable_frames} valid ArUco frames; sample skipped.'
            )
            return None, last_ids

        translations = np.asarray([T[:3, 3] for T in transforms], dtype=np.float64)
        translation_std = np.std(translations, axis=0)
        z_span = float(np.max(translations[:, 2]) - np.min(translations[:, 2]))
        std_norm = float(np.linalg.norm(translation_std))
        mean_reprojection = float(np.mean(reprojections)) if reprojections else 0.0

        if std_norm > self.args.max_translation_std or z_span > self.args.max_z_span:
            self.get_logger().warning(
                'ArUco pose is not stable; sample skipped. '
                f'std_norm={std_norm:.4f} m, z_span={z_span:.4f} m, '
                f'mean_reprojection={mean_reprojection:.2f} px'
            )
            return None, last_ids

        if self.args.max_reprojection_error > 0.0 and mean_reprojection > self.args.max_reprojection_error:
            self.get_logger().warning(
                'ArUco reprojection error is too high; sample skipped. '
                f'mean_reprojection={mean_reprojection:.2f} px'
            )
            return None, last_ids

        median_translation = np.median(translations, axis=0)
        selected = int(np.argmin(np.linalg.norm(translations - median_translation, axis=1)))
        selected_T = transforms[selected]
        pos = selected_T[:3, 3]
        self.get_logger().info(
            'Stable ArUco pose accepted: '
            f'x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f} m, '
            f'std_norm={std_norm:.4f} m, z_span={z_span:.4f} m, '
            f'mean_reprojection={mean_reprojection:.2f} px'
        )
        return selected_T, last_ids

    def capture_sample(self):
        if self.latest_fk_pose is None:
            self.get_logger().warning('No /fk_pose received yet.')
            return False
        if self.latest_image is None:
            self.get_logger().warning('No camera image received yet.')
            return False

        T_cm, ids = self.capture_stable_marker_pose()
        if T_cm is None:
            self.get_logger().warning('No stable requested ArUco marker detected; sample skipped.')
            return False

        T_be = pose_to_matrix(self.latest_fk_pose)
        self.base_to_ee.append(T_be)
        self.cam_to_marker.append(T_cm)
        save_calibration_data(self.args.output, self.base_to_ee, self.cam_to_marker)

        marker_info = ids.tolist() if ids is not None else []
        self.get_logger().info(
            f'Saved sample {len(self.base_to_ee)} to {self.args.output}; '
            f'detected ids={marker_info}'
        )
        return True


def load_pose_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def run_manual_collection(node):
    print('Manual collection mode.')
    print('Move the robot to a calibration pose, make sure ArUco is visible, then press Enter.')
    print('Commands: Enter=capture, q=quit')
    while rclpy.ok():
        command = input('capture> ').strip().lower()
        if command in ('q', 'quit', 'exit'):
            break
        node.capture_sample()


def run_auto_collection(node, poses):
    for index, pose in enumerate(poses, start=1):
        position = pose['position']
        orientation = pose['orientation']
        print(f'Moving to pose {index}/{len(poses)}: pos={position}, quat={orientation}')
        if not node.move_to_pose(position, orientation, timeout_sec=node.args.service_timeout):
            continue
        time.sleep(node.args.settle_time)
        node.capture_sample()


def parse_args(args=None):
    parser = argparse.ArgumentParser(description='Collect FK and ArUco pose pairs for hand-eye calibration.')
    parser.add_argument('--image-topic', default='/camera/image_raw')
    parser.add_argument('--fk-topic', default='/fk_pose')
    parser.add_argument('--service-name', default='/set_position')
    parser.add_argument('--intrinsics', default='results/camera_intrinsics.npz')
    parser.add_argument('--output', default='results/calibration_data.npz')
    parser.add_argument('--marker-length', type=float, default=0.094)
    parser.add_argument('--marker-id', type=int, default=6)
    parser.add_argument('--dictionary', default='DICT_ARUCO_ORIGINAL')
    parser.add_argument('--append', action='store_true')
    parser.add_argument('--auto', action='store_true', help='Move through poses instead of manual capture.')
    parser.add_argument('--poses-file', default=None, help='JSON list of {"position": [...], "orientation": [...]}')
    parser.add_argument('--settle-time', type=float, default=2.0)
    parser.add_argument('--service-timeout', type=float, default=5.0)
    parser.add_argument('--stable-frames', type=int, default=10)
    parser.add_argument('--min-stable-frames', type=int, default=6)
    parser.add_argument('--stable-delay', type=float, default=0.05)
    parser.add_argument('--max-translation-std', type=float, default=0.02)
    parser.add_argument('--max-z-span', type=float, default=0.05)
    parser.add_argument('--max-reprojection-error', type=float, default=3.0)
    return parser.parse_args(args)


def main(args=None):
    parsed = parse_args(args)
    rclpy.init(args=None)
    node = DataCollector(parsed)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    try:
        if parsed.auto:
            poses = load_pose_file(parsed.poses_file) if parsed.poses_file else DEFAULT_POSES
            run_auto_collection(node, poses)
        else:
            run_manual_collection(node)
    finally:
        executor.shutdown()
        thread.join(timeout=1.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
