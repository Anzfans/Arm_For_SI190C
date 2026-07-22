import argparse
import glob
from pathlib import Path

import cv2
import numpy as np

from handeye_calibration.utils import ensure_dir, load_intrinsics, save_intrinsics


def load_camera_intrinsics(path='results/camera_intrinsics.npz'):
    return load_intrinsics(path)


def calibrate_camera_intrinsics(image_paths, chessboard_size=(9, 6), square_size=0.025, preview=False):
    if len(image_paths) < 3:
        raise ValueError('Need at least 3 chessboard images; 15-20 is recommended.')

    cols, rows = chessboard_size
    objp = np.zeros((1, rows * cols, 3), np.float64)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp[0, :, :2] = grid * square_size

    obj_points = []
    img_points = []
    image_size = None

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f'WARNING: failed to read {img_path}')
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = gray.shape[::-1]
        elif image_size != gray.shape[::-1]:
            print(f'WARNING: skipping {img_path}; image size differs from first image.')
            continue

        found, corners = cv2.findChessboardCorners(
            gray,
            chessboard_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if not found:
            print(f'No chessboard found: {img_path}')
            continue

        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        obj_points.append(objp.copy())
        img_points.append(corners.reshape(1, -1, 2).astype(np.float64))
        print(f'Accepted: {img_path}')

        if preview:
            vis = img.copy()
            cv2.drawChessboardCorners(vis, chessboard_size, corners, found)
            cv2.imshow('chessboard_corners', vis)
            cv2.waitKey(300)

    if preview:
        cv2.destroyAllWindows()

    if len(obj_points) < 3:
        raise RuntimeError(f'Only {len(obj_points)} valid chessboard images found.')

    K = np.eye(3, dtype=np.float64)
    K[0, 0] = image_size[0] / 2.0
    K[1, 1] = image_size[1] / 2.0
    K[0, 2] = image_size[0] / 2.0
    K[1, 2] = image_size[1] / 2.0
    D = np.zeros((4, 1), dtype=np.float64)

    flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        + cv2.fisheye.CALIB_CHECK_COND
        + cv2.fisheye.CALIB_FIX_SKEW
    )

    rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        obj_points,
        img_points,
        image_size,
        K,
        D,
        flags=flags,
        criteria=(
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            100,
            1e-6,
        ),
    )

    return rms, K, D, rvecs, tvecs, image_size


def undistort_image(image, camera_matrix, dist_coeffs, balance=1.0):
    h, w = image.shape[:2]
    new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        camera_matrix,
        dist_coeffs,
        (w, h),
        np.eye(3),
        balance=balance,
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        camera_matrix,
        dist_coeffs,
        np.eye(3),
        new_camera_matrix,
        (w, h),
        cv2.CV_16SC2,
    )
    return cv2.remap(image, map1, map2, cv2.INTER_LINEAR)


def parse_args(args=None):
    parser = argparse.ArgumentParser(description='Calibrate fisheye camera intrinsics.')
    parser.add_argument('--images', default='calibration_images/chessboard_*.png')
    parser.add_argument('--cols', type=int, default=9, help='Inner corner columns.')
    parser.add_argument('--rows', type=int, default=6, help='Inner corner rows.')
    parser.add_argument('--square-size', type=float, default=0.025, help='Square size in meters.')
    parser.add_argument('--output', default='results/camera_intrinsics.npz')
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--undistort-dir', default=None)
    return parser.parse_args(args)


def main(args=None):
    parsed = parse_args(args)
    image_paths = sorted(glob.glob(parsed.images))
    print(f'Found {len(image_paths)} candidate images.')
    if len(image_paths) < 10:
        print('WARNING: fewer than 10 images. Calibration quality may be poor.')

    rms, K, D, _, _, image_size = calibrate_camera_intrinsics(
        image_paths,
        chessboard_size=(parsed.cols, parsed.rows),
        square_size=parsed.square_size,
        preview=parsed.preview,
    )
    save_intrinsics(parsed.output, K, D, rms=rms, image_size=image_size)

    print(f'RMS reprojection error: {rms:.4f} pixels')
    print('Camera matrix:')
    print(K)
    print('Fisheye distortion coefficients [k1, k2, k3, k4]:')
    print(D.ravel())
    print(f'Saved intrinsics to {parsed.output}')

    if parsed.undistort_dir:
        out_dir = ensure_dir(parsed.undistort_dir)
        camera_matrix, dist_coeffs = load_intrinsics(parsed.output)
        for img_path in image_paths:
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            undistorted = undistort_image(image, camera_matrix, dist_coeffs)
            cv2.imwrite(str(out_dir / Path(img_path).name), undistorted)
        print(f'Saved undistorted images to {out_dir}')


if __name__ == '__main__':
    main()
