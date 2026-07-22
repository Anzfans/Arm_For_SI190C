import argparse
from pathlib import Path

import cv2
import numpy as np

from handeye_calibration.utils import load_calibration_data, rotation_error_deg


HAND_EYE_METHODS = {
    'tsai': cv2.CALIB_HAND_EYE_TSAI,
    'park': cv2.CALIB_HAND_EYE_PARK,
    'horaud': cv2.CALIB_HAND_EYE_HORAUD,
    'andreff': cv2.CALIB_HAND_EYE_ANDREFF,
    'daniilidis': cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def compute_motion_pairs(base_to_ee, cam_to_marker):
    A_mats = []
    B_mats = []
    for i in range(1, len(base_to_ee)):
        A = np.linalg.inv(base_to_ee[i]) @ base_to_ee[i - 1]
        B = cam_to_marker[i] @ np.linalg.inv(cam_to_marker[i - 1])
        A_mats.append(A)
        B_mats.append(B)
    return A_mats, B_mats


def solve_hand_eye_from_absolute(base_to_ee, cam_to_marker, method='tsai'):
    if len(base_to_ee) != len(cam_to_marker):
        raise ValueError('base_to_ee and cam_to_marker must have the same length.')
    if len(base_to_ee) < 4:
        raise ValueError('Need at least 4 absolute pose pairs; 10-20 is recommended.')
    if method not in HAND_EYE_METHODS:
        raise ValueError(f'Unknown method {method}. Choose from {sorted(HAND_EYE_METHODS)}')

    R_gripper2base = [np.asarray(T[:3, :3], dtype=np.float64) for T in base_to_ee]
    t_gripper2base = [np.asarray(T[:3, 3], dtype=np.float64).reshape(3, 1) for T in base_to_ee]
    R_target2cam = [np.asarray(T[:3, :3], dtype=np.float64) for T in cam_to_marker]
    t_target2cam = [np.asarray(T[:3, 3], dtype=np.float64).reshape(3, 1) for T in cam_to_marker]

    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base,
        t_gripper2base,
        R_target2cam,
        t_target2cam,
        method=HAND_EYE_METHODS[method],
    )

    T_ec = np.eye(4, dtype=np.float64)
    T_ec[:3, :3] = R_cam2gripper
    T_ec[:3, 3] = t_cam2gripper.reshape(3)
    return T_ec


def summarize_motion_diversity(base_to_ee, cam_to_marker):
    A_mats, B_mats = compute_motion_pairs(base_to_ee, cam_to_marker)
    ee_trans = [np.linalg.norm(A[:3, 3]) for A in A_mats]
    cam_trans = [np.linalg.norm(B[:3, 3]) for B in B_mats]
    ee_rot = [rotation_error_deg(A[:3, :3], np.eye(3)) for A in A_mats]
    cam_rot = [rotation_error_deg(B[:3, :3], np.eye(3)) for B in B_mats]
    return {
        'num_motion_pairs': len(A_mats),
        'ee_translation_m': ee_trans,
        'camera_translation_m': cam_trans,
        'ee_rotation_deg': ee_rot,
        'camera_rotation_deg': cam_rot,
    }


def parse_args(args=None):
    parser = argparse.ArgumentParser(description='Solve hand-eye calibration from collected pose pairs.')
    parser.add_argument('--data', default='results/calibration_data.npz')
    parser.add_argument('--output', default='results/handeye_transform.npy')
    parser.add_argument('--method', default='tsai', choices=sorted(HAND_EYE_METHODS))
    return parser.parse_args(args)


def main(args=None):
    parsed = parse_args(args)
    base_to_ee, cam_to_marker = load_calibration_data(parsed.data)
    print(f'Loaded {len(base_to_ee)} pose pairs from {parsed.data}')
    if len(base_to_ee) < 10:
        print('WARNING: fewer than 10 pose pairs. Hand-eye result may be weak.')

    diversity = summarize_motion_diversity(base_to_ee, cam_to_marker)
    print('Motion diversity summary:')
    print(f"  motion pairs: {diversity['num_motion_pairs']}")
    if diversity['ee_translation_m']:
        print(f"  EE translation range: {min(diversity['ee_translation_m']):.3f} - "
              f"{max(diversity['ee_translation_m']):.3f} m")
        print(f"  EE rotation range: {min(diversity['ee_rotation_deg']):.1f} - "
              f"{max(diversity['ee_rotation_deg']):.1f} deg")

    T_ec = solve_hand_eye_from_absolute(base_to_ee, cam_to_marker, method=parsed.method)
    Path(parsed.output).parent.mkdir(parents=True, exist_ok=True)
    np.save(parsed.output, T_ec)

    print('Estimated E_T_C (camera frame to end-effector frame):')
    np.set_printoptions(precision=6, suppress=True)
    print(T_ec)
    print(f'Saved transform to {parsed.output}')


if __name__ == '__main__':
    main()
