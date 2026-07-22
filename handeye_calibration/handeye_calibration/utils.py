from pathlib import Path

import cv2
import numpy as np
from geometry_msgs.msg import Pose


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_quaternion(qx, qy, qz, qw):
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError('Quaternion norm is zero')
    return q / norm


def quaternion_to_matrix(qx, qy, qz, qw):
    x, y, z, w = normalize_quaternion(qx, qy, qz, qw)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ], dtype=np.float64)


def pose_to_matrix(pose):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quaternion_to_matrix(
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    )
    T[:3, 3] = [
        pose.position.x,
        pose.position.y,
        pose.position.z,
    ]
    return T


def pose_fields_to_matrix(position, orientation):
    pose = Pose()
    pose.position.x = float(position[0])
    pose.position.y = float(position[1])
    pose.position.z = float(position[2])
    pose.orientation.x = float(orientation[0])
    pose.orientation.y = float(orientation[1])
    pose.orientation.z = float(orientation[2])
    pose.orientation.w = float(orientation[3])
    return pose_to_matrix(pose)


def rvec_tvec_to_matrix(rvec, tvec):
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return T


def invert_transform(T):
    T = np.asarray(T, dtype=np.float64)
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = T[:3, :3].T
    inv[:3, 3] = -inv[:3, :3] @ T[:3, 3]
    return inv


def rotation_error_deg(R_a, R_b):
    R_err = np.asarray(R_a) @ np.asarray(R_b).T
    cos_angle = (np.trace(R_err) - 1.0) / 2.0
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return float(np.degrees(angle))


def save_intrinsics(path, camera_matrix, dist_coeffs, rms=None, image_size=None):
    ensure_dir(Path(path).parent)
    payload = {
        'camera_matrix': np.asarray(camera_matrix, dtype=np.float64),
        'dist_coeffs': np.asarray(dist_coeffs, dtype=np.float64),
    }
    if rms is not None:
        payload['rms'] = np.asarray([rms], dtype=np.float64)
    if image_size is not None:
        payload['image_size'] = np.asarray(image_size, dtype=np.int32)
    np.savez(path, **payload)


def load_intrinsics(path):
    data = np.load(path)
    return data['camera_matrix'], data['dist_coeffs']


def save_calibration_data(path, base_to_ee, cam_to_marker):
    ensure_dir(Path(path).parent)
    np.savez(
        path,
        base_to_ee=np.asarray(base_to_ee, dtype=np.float64),
        cam_to_marker=np.asarray(cam_to_marker, dtype=np.float64),
    )


def load_calibration_data(path):
    data = np.load(path)
    return data['base_to_ee'], data['cam_to_marker']
