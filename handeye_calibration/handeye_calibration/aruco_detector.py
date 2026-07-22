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
    if hasattr(cv2.aruco, 'DetectorParameters'):
        return cv2.aruco.DetectorParameters()
    return cv2.aruco.DetectorParameters_create()


def detect_aruco_pose(
    image,
    camera_matrix,
    dist_coeffs,
    marker_length=0.05,
    dictionary_name='DICT_4X4_50',
    marker_id=None,
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
            return None, ids_flat, corners, None
        selected = int(matches[0])

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    for corner in corners:
        cv2.cornerSubPix(gray, corner, (5, 5), (-1, -1), criteria)

    selected_corners = [corners[selected]]
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        selected_corners,
        marker_length,
        camera_matrix,
        dist_coeffs,
    )
    T_cm = rvec_tvec_to_matrix(rvecs[0], tvecs[0])
    return T_cm, ids_flat, selected_corners[0], (rvecs[0], tvecs[0])


def draw_detection(image, ids, corners, pose, camera_matrix, dist_coeffs, axis_length):
    vis = image.copy()
    if ids is not None and corners is not None:
        draw_ids = np.asarray(ids, dtype=np.int32).reshape(-1, 1)
        cv2.aruco.drawDetectedMarkers(vis, [corners], draw_ids[:1])
    if pose is not None:
        rvec, tvec = pose
        cv2.drawFrameAxes(vis, camera_matrix, dist_coeffs, rvec, tvec, axis_length)
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
            self.get_logger().warning('No requested ArUco marker detected.')
        else:
            pos = T_cm[:3, 3]
            self.get_logger().info(
                f'Detected marker. C_T_O translation: '
                f'x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f} m'
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
    parser.add_argument('--marker-length', type=float, default=0.05)
    parser.add_argument('--marker-id', type=int, default=None)
    parser.add_argument('--dictionary', default='DICT_4X4_50')
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
