import argparse

import numpy as np

from handeye_calibration.utils import load_calibration_data, rotation_error_deg


def verify_stationary_target(base_to_ee, cam_to_marker, T_ec):
    base_to_marker = []
    for T_be, T_cm in zip(base_to_ee, cam_to_marker):
        base_to_marker.append(T_be @ T_ec @ T_cm)

    reference = base_to_marker[0]
    position_errors_mm = []
    orientation_errors_deg = []
    for T_bo in base_to_marker:
        position_errors_mm.append(float(np.linalg.norm(T_bo[:3, 3] - reference[:3, 3]) * 1000.0))
        orientation_errors_deg.append(rotation_error_deg(T_bo[:3, :3], reference[:3, :3]))

    positions = np.asarray([T[:3, 3] for T in base_to_marker], dtype=np.float64)
    return {
        'base_to_marker': np.asarray(base_to_marker),
        'position_errors_mm': np.asarray(position_errors_mm),
        'orientation_errors_deg': np.asarray(orientation_errors_deg),
        'position_std_mm': positions.std(axis=0) * 1000.0,
    }


def parse_args(args=None):
    parser = argparse.ArgumentParser(description='Verify hand-eye calibration consistency.')
    parser.add_argument('--data', default='results/calibration_data.npz')
    parser.add_argument('--transform', default='results/handeye_transform.npy')
    parser.add_argument('--save-base-marker', default='results/base_to_marker_estimates.npy')
    return parser.parse_args(args)


def main(args=None):
    parsed = parse_args(args)
    base_to_ee, cam_to_marker = load_calibration_data(parsed.data)
    T_ec = np.load(parsed.transform)
    result = verify_stationary_target(base_to_ee, cam_to_marker, T_ec)
    np.save(parsed.save_base_marker, result['base_to_marker'])

    pos_errors = result['position_errors_mm']
    rot_errors = result['orientation_errors_deg']
    print(f'Verified {len(pos_errors)} samples against the first sample as reference.')
    print(f'Mean position error: {pos_errors.mean():.2f} mm')
    print(f'Max position error: {pos_errors.max():.2f} mm')
    print(f'Mean orientation error: {rot_errors.mean():.2f} deg')
    print(f'Max orientation error: {rot_errors.max():.2f} deg')
    print(f'Per-axis position std: {result["position_std_mm"]} mm')
    print(f'Saved base-to-marker estimates to {parsed.save_base_marker}')


if __name__ == '__main__':
    main()
