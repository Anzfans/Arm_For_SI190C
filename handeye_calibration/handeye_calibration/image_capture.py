import argparse
from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

from handeye_calibration.utils import ensure_dir


class ImageCapture(Node):
    def __init__(self, image_topic, output_dir, prefix):
        super().__init__('handeye_image_capture')
        self.bridge = CvBridge()
        self.output_dir = ensure_dir(output_dir)
        self.prefix = prefix
        self.latest_image = None
        self.counter = self._next_index()
        self.sub = self.create_subscription(Image, image_topic, self.image_callback, 10)
        self.get_logger().info(f'Listening on {image_topic}')
        self.get_logger().info(f'Saving images to {self.output_dir}')

    def image_callback(self, msg):
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def _next_index(self):
        existing = sorted(Path(self.output_dir).glob(f'{self.prefix}_*.png'))
        if not existing:
            return 1
        indices = []
        for path in existing:
            try:
                indices.append(int(path.stem.split('_')[-1]))
            except ValueError:
                pass
        return max(indices, default=0) + 1

    def save_latest(self):
        if self.latest_image is None:
            self.get_logger().warning('No image received yet.')
            return False
        path = self.output_dir / f'{self.prefix}_{self.counter:03d}.png'
        cv2.imwrite(str(path), self.latest_image)
        self.get_logger().info(f'Saved {path}')
        self.counter += 1
        return True


def parse_args(args=None):
    parser = argparse.ArgumentParser(description='Capture images from a ROS image topic.')
    parser.add_argument('--image-topic', default='/camera/image_raw')
    parser.add_argument('--output-dir', default='calibration_images')
    parser.add_argument('--prefix', default='chessboard')
    parser.add_argument('--once', action='store_true', help='Save one image and exit.')
    parser.add_argument('--no-preview', action='store_true', help='Disable OpenCV preview window.')
    return parser.parse_args(args)


def main(args=None):
    parsed = parse_args(args)
    rclpy.init(args=None)
    node = ImageCapture(parsed.image_topic, parsed.output_dir, parsed.prefix)
    try:
        if parsed.once:
            while rclpy.ok() and node.latest_image is None:
                rclpy.spin_once(node, timeout_sec=0.1)
            node.save_latest()
            return

        print('Press s to save, q or Esc to quit.')
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            if not parsed.no_preview and node.latest_image is not None:
                cv2.imshow('image_capture', node.latest_image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    node.save_latest()
                elif key == ord('q') or key == 27:
                    break
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
