import math

import rclpy
from geometry_msgs.msg import Pose
from rclpy.node import Node
from si190c_interfaces.srv import SetPosition


class PositionService(Node):
    def __init__(self):
        super().__init__('position_service')
        self.current_pose = None
        self.pose_pub = self.create_publisher(Pose, '/gui_pose', 10)
        self.fk_sub = self.create_subscription(
            Pose,
            '/fk_pose',
            self.fk_pose_callback,
            10,
        )
        self.srv = self.create_service(
            SetPosition,
            '/set_position',
            self.set_position_callback,
        )
        self.get_logger().info('Service ready: /set_position')

    def fk_pose_callback(self, msg):
        self.current_pose = msg

    def set_position_callback(self, request, response):
        target_distance = math.sqrt(request.x ** 2 + request.y ** 2 + request.z ** 2)
        quat_norm = math.sqrt(
            request.qx ** 2
            + request.qy ** 2
            + request.qz ** 2
            + request.qw ** 2
        )

        if quat_norm < 1e-6:
            response.success = False
            response.message = 'Rejected: quaternion norm is zero.'
            self.get_logger().error(response.message)
            return response

        pose = Pose()
        pose.position.x = request.x
        pose.position.y = request.y
        pose.position.z = request.z
        pose.orientation.x = request.qx
        pose.orientation.y = request.qy
        pose.orientation.z = request.qz
        pose.orientation.w = request.qw
        self.pose_pub.publish(pose)

        message = f'Published target pose. Distance from origin: {target_distance:.3f} m.'
        if self.current_pose is not None:
            current_distance = self._distance_to_current(request)
            message += f' Distance from current FK pose: {current_distance:.3f} m.'
        else:
            message += ' No /fk_pose received yet.'

        response.success = True
        response.message = message
        self.get_logger().info(
            'SetPosition x=%.3f y=%.3f z=%.3f q=(%.3f, %.3f, %.3f, %.3f): %s',
            request.x,
            request.y,
            request.z,
            request.qx,
            request.qy,
            request.qz,
            request.qw,
            message,
        )
        return response

    def _distance_to_current(self, request):
        dx = request.x - self.current_pose.position.x
        dy = request.y - self.current_pose.position.y
        dz = request.z - self.current_pose.position.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)


def main(args=None):
    rclpy.init(args=args)
    node = PositionService()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
