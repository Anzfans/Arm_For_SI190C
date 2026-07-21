import math
import time

import rclpy
from geometry_msgs.msg import Pose
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from si190c_interfaces.action import MoveArm


class ArmActionServer(Node):
    def __init__(self):
        super().__init__('arm_action_server')
        self.current_pose = None
        self.callback_group = ReentrantCallbackGroup()

        self.declare_parameter('position_tolerance', 0.01)
        self.declare_parameter('feedback_period', 0.2)
        self.declare_parameter('timeout_sec', 30.0)
        self.declare_parameter('simulate_without_fk', True)
        self.declare_parameter('simulation_duration_sec', 5.0)

        self.pose_pub = self.create_publisher(Pose, '/gui_pose', 10)
        self.fk_sub = self.create_subscription(
            Pose,
            '/fk_pose',
            self.fk_pose_callback,
            10,
            callback_group=self.callback_group,
        )
        self.action_server = ActionServer(
            self,
            MoveArm,
            '/move_arm',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.callback_group,
        )
        self.get_logger().info('Action server ready: /move_arm')

    def fk_pose_callback(self, msg):
        self.current_pose = msg

    def goal_callback(self, goal_request):
        quat_norm = math.sqrt(
            goal_request.tqx ** 2
            + goal_request.tqy ** 2
            + goal_request.tqz ** 2
            + goal_request.tqw ** 2
        )
        if quat_norm < 1e-6:
            self.get_logger().error('Rejected goal: quaternion norm is zero.')
            return GoalResponse.REJECT

        self.get_logger().info(
            f'Accepted goal x={goal_request.target_x:.3f} '
            f'y={goal_request.target_y:.3f} z={goal_request.target_z:.3f}'
        )
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info('Cancel request accepted.')
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        target = goal_handle.request
        target_pose = self._make_pose(target)
        self.pose_pub.publish(target_pose)

        tolerance = self.get_parameter('position_tolerance').value
        feedback_period = self.get_parameter('feedback_period').value
        timeout_sec = self.get_parameter('timeout_sec').value
        simulate_without_fk = self.get_parameter('simulate_without_fk').value
        simulation_duration = self.get_parameter('simulation_duration_sec').value

        start_time = self.get_clock().now()
        start_pose = self.current_pose
        start_distance = self._distance_to_target(target, start_pose)
        if start_distance is None or start_distance < tolerance:
            start_distance = 1.0

        while rclpy.ok():
            elapsed = (self.get_clock().now() - start_time).nanoseconds / 1e9

            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return self._result(False, target, 'Canceled before reaching target.')

            if elapsed > timeout_sec:
                goal_handle.abort()
                return self._result(False, target, 'Timed out before reaching target.')

            self.pose_pub.publish(target_pose)

            if self.current_pose is not None:
                distance = self._distance_to_target(target, self.current_pose)
                progress = max(0.0, min(100.0, 100.0 * (1.0 - distance / start_distance)))
                self._publish_feedback(goal_handle, self.current_pose, progress)

                if distance <= tolerance:
                    goal_handle.succeed()
                    return self._result(True, target, 'Reached target pose.')
            elif simulate_without_fk:
                progress = min(100.0, elapsed / simulation_duration * 100.0)
                simulated_pose = self._interpolate_from_start(target, progress / 100.0)
                self._publish_feedback(goal_handle, simulated_pose, progress)

                if progress >= 100.0:
                    goal_handle.succeed()
                    return self._result(True, target, 'Simulated target reached; no /fk_pose received.')
            else:
                self._publish_feedback(goal_handle, target_pose, 0.0)

            time.sleep(feedback_period)

        goal_handle.abort()
        return self._result(False, target, 'ROS shutdown before reaching target.')

    def _make_pose(self, target):
        pose = Pose()
        pose.position.x = target.target_x
        pose.position.y = target.target_y
        pose.position.z = target.target_z
        pose.orientation.x = target.tqx
        pose.orientation.y = target.tqy
        pose.orientation.z = target.tqz
        pose.orientation.w = target.tqw
        return pose

    def _distance_to_target(self, target, pose):
        if pose is None:
            return None
        dx = target.target_x - pose.position.x
        dy = target.target_y - pose.position.y
        dz = target.target_z - pose.position.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _interpolate_from_start(self, target, ratio):
        pose = Pose()
        start = self.current_pose
        sx = start.position.x if start is not None else 0.0
        sy = start.position.y if start is not None else 0.0
        sz = start.position.z if start is not None else 0.0
        pose.position.x = sx + (target.target_x - sx) * ratio
        pose.position.y = sy + (target.target_y - sy) * ratio
        pose.position.z = sz + (target.target_z - sz) * ratio
        pose.orientation.x = target.tqx
        pose.orientation.y = target.tqy
        pose.orientation.z = target.tqz
        pose.orientation.w = target.tqw
        return pose

    def _publish_feedback(self, goal_handle, pose, progress):
        feedback = MoveArm.Feedback()
        feedback.current_x = pose.position.x
        feedback.current_y = pose.position.y
        feedback.current_z = pose.position.z
        feedback.progress = progress
        goal_handle.publish_feedback(feedback)

    def _result(self, success, target, status):
        result = MoveArm.Result()
        result.success = success
        result.final_x = target.target_x
        result.final_y = target.target_y
        result.final_z = target.target_z
        result.status = status
        self.get_logger().info(status)
        return result


def main(args=None):
    rclpy.init(args=args)
    node = ArmActionServer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
