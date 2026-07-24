import argparse

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

from handeye_calibration.utils import load_intrinsics, rvec_tvec_to_matrix


ARUCO_DICTIONARIES = {
    name: value
    for name, value in vars(cv2.aruco).items()
    if name.startswith('DICT_')
}


def get_aruco_dictionary(name):
    if name not in ARUCO_DICTIONARIES:
        valid = ', '.join(sorted(ARUCO_DICTIONARIES))
        raise ValueError(f'Unknown ArUco dictionary {name}. Valid names: {valid}')
    return cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARIES[name])


def create_detector_parameters():
    if hasattr(cv2.aruco, 'DetectorParameters_create'):
        params = cv2.aruco.DetectorParameters_create()
    else:
        params = cv2.aruco.DetectorParameters()

    tuned_values = {
        'adaptiveThreshWinSizeMin': 3,
        'adaptiveThreshWinSizeMax': 53,
        'adaptiveThreshWinSizeStep': 4,
        'minMarkerPerimeterRate': 0.03,
        'maxMarkerPerimeterRate': 4.0,
        'polygonalApproxAccuracyRate': 0.03,
        'minCornerDistanceRate': 0.03,
        'minDistanceToBorder': 3,
        'cornerRefinementMethod': cv2.aruco.CORNER_REFINE_SUBPIX,
        'cornerRefinementWinSize': 7,
        'cornerRefinementMaxIterations': 50,
        'cornerRefinementMinAccuracy': 0.001,
        'errorCorrectionRate': 0.6,
    }
    for name, value in tuned_values.items():
        if hasattr(params, name):
            setattr(params, name, value)
    return params


def marker_object_points(marker_length):
    half = marker_length / 2.0
    return np.array([
        [-half, half, 0.0],
        [half, half, 0.0],
        [half, -half, 0.0],
        [-half, -half, 0.0],
    ], dtype=np.float64)


def use_fisheye_model(dist_coeffs):
    return np.asarray(dist_coeffs).size == 4


def undistort_marker_corners(corners, camera_matrix, dist_coeffs):
    image_points = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    if use_fisheye_model(dist_coeffs):
        undistorted = cv2.fisheye.undistortPoints(
            image_points.reshape(-1, 1, 2),
            camera_matrix,
            np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1),
            P=camera_matrix,
        )
        return undistorted.reshape(4, 2), None
    return image_points, dist_coeffs


def reprojection_error(object_points, image_points, rvec, tvec, camera_matrix, dist_coeffs):
    projected, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )
    projected = projected.reshape(-1, 2)
    image_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum((projected - image_points) ** 2, axis=1))))


def estimate_marker_pose(corners, marker_length, camera_matrix, dist_coeffs):
    object_points = marker_object_points(marker_length)
    image_points, pose_dist_coeffs = undistort_marker_corners(
        corners,
        camera_matrix,
        dist_coeffs,
    )

    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        pose_dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        raise RuntimeError('solvePnP failed for detected ArUco marker')

    error = reprojection_error(
        object_points,
        image_points,
        rvec,
        tvec,
        camera_matrix,
        pose_dist_coeffs,
    )
    return rvec, tvec, error


def draw_pose_axes(image, camera_matrix, dist_coeffs, rvec, tvec, axis_length):
    if use_fisheye_model(dist_coeffs):
        axis_points = np.array([
            [0.0, 0.0, 0.0],
            [axis_length, 0.0, 0.0],
            [0.0, axis_length, 0.0],
            [0.0, 0.0, axis_length],
        ], dtype=np.float64).reshape(-1, 1, 3)
        points, _ = cv2.fisheye.projectPoints(
            axis_points,
            rvec,
            tvec,
            camera_matrix,
            np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1),
        )
        points = np.round(points.reshape(-1, 2)).astype(int)
        origin = tuple(points[0])
        cv2.line(image, origin, tuple(points[1]), (0, 0, 255), 2)
        cv2.line(image, origin, tuple(points[2]), (0, 255, 0), 2)
        cv2.line(image, origin, tuple(points[3]), (255, 0, 0), 2)
    else:
        cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvec, tvec, axis_length)


def detect_aruco_pose(
    image,
    camera_matrix,
    dist_coeffs,
    marker_length=0.094,
    dictionary_name='DICT_ARUCO_ORIGINAL',
    marker_id=6,
):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = get_aruco_dictionary(dictionary_name)
    params = create_detector_parameters()

    if hasattr(cv2.aruco, 'ArucoDetector'):
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)

    if ids is None or len(ids) == 0:
        return None, None, None, None

    ids_flat = ids.flatten()
    selected = 0
    if marker_id is not None:
        matches = np.where(ids_flat == marker_id)[0]
        if len(matches) == 0:
            return None, ids_flat, None, None
        selected = int(matches[0])

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    for corner in corners:
        cv2.cornerSubPix(gray, corner, (5, 5), (-1, -1), criteria)

    selected_corners = [corners[selected]]
    rvec, tvec, error = estimate_marker_pose(
        selected_corners[0],
        marker_length,
        camera_matrix,
        dist_coeffs,
    )
    T_cm = rvec_tvec_to_matrix(rvec, tvec)
    return T_cm, ids_flat, selected_corners[0], (rvec, tvec, error)


def draw_detection(image, ids, corners, pose, camera_matrix, dist_coeffs, axis_length):
    vis = image.copy()
    if ids is not None and corners is not None:
        draw_ids = np.asarray(ids, dtype=np.int32).reshape(-1, 1)
        cv2.aruco.drawDetectedMarkers(vis, [corners], draw_ids[:1])
    if pose is not None:
        rvec, tvec = pose[:2]
        draw_pose_axes(vis, camera_matrix, dist_coeffs, rvec, tvec, axis_length)
    return vis


class ArucoDetectorNode(Node):
    def __init__(self, args):
        super().__init__('aruco_detector')
        self.bridge = CvBridge()
        self.args = args
        self.camera_matrix, self.dist_coeffs = load_intrinsics(args.intrinsics)
        self.sub = self.create_subscription(Image, args.image_topic, self.image_callback, 10)
        self.get_logger().info(f'Listening on {args.image_topic}')
        self.get_logger().info(f'Using intrinsics from {args.intrinsics}')

    def image_callback(self, msg):
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        T_cm, ids, corners, pose = detect_aruco_pose(
            image,
            self.camera_matrix,
            self.dist_coeffs,
            marker_length=self.args.marker_length,
            dictionary_name=self.args.dictionary,
            marker_id=self.args.marker_id,
        )

        if T_cm is None:
            if ids is None:
                self.get_logger().warning('No ArUco marker detected.')
            else:
                self.get_logger().warning(f'No requested ArUco marker detected; detected ids={ids.tolist()}')
        else:
            pos = T_cm[:3, 3]
            reproj = pose[2] if pose is not None and len(pose) > 2 else float('nan')
            self.get_logger().info(
                f'Detected marker. C_T_O translation: '
                f'x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f} m, '
                f'reprojection_error={reproj:.2f} px'
            )

        if not self.args.no_preview:
            vis = image
            if corners is not None:
                selected_id = [self.args.marker_id if self.args.marker_id is not None else ids[0]]
                vis = draw_detection(
                    image,
                    selected_id,
                    corners,
                    pose,
                    self.camera_matrix,
                    self.dist_coeffs,
                    self.args.axis_length,
                )
            cv2.imshow('aruco_detector', vis)
            cv2.waitKey(1)


def parse_args(args=None):
    parser = argparse.ArgumentParser(description='Detect ArUco markers and estimate pose.')
    parser.add_argument('--image-topic', default='/camera/image_raw')
    parser.add_argument('--intrinsics', default='results/camera_intrinsics.npz')
    parser.add_argument('--marker-length', type=float, default=0.094)
    parser.add_argument('--marker-id', type=int, default=6)
    parser.add_argument('--dictionary', default='DICT_ARUCO_ORIGINAL')
    parser.add_argument('--axis-length', type=float, default=0.03)
    parser.add_argument('--no-preview', action='store_true')
    return parser.parse_args(args)


def main(args=None):
    parsed = parse_args(args)
    rclpy.init(args=None)
    node = ArucoDetectorNode(parsed)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
