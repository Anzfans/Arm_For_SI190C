/*
 * File: pose_publisher.cpp
 * Author: Zhenghao Li
 * Email: lizhenghao@shanghaitech.edu.cn
 * Institute: SIST
 * Created: 2025-06-12
 * Last Modified: 2026-07-13
 */
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <vector>

class PosePublisher : public rclcpp::Node
{
public:
    PosePublisher()
        : Node("pose_publisher"),
          current_index_(0)
    {
        publisher_ =
            this->create_publisher<geometry_msgs::msg::Pose>("/gui_pose", 10);

        // 三个给定点：A、B、C
        waypoints_ = {
            createPose(
                  0.125, -0.194, -0.049,
                 -0.629, 0.704, -0.292, -0.151),
            createPose(     
                  0.014, -0.307, -0.03,
                 0.835, -0.429, 0.305, 0.160),
                 
            createPose(
                  0.088, -0.034, 0.177,
                 0.703, 0.017,  0.669, 0.241)
//           createPose(
//                0.248, -0.004, -0.049,
//                0.826, 0.544, 0.113, 0.098
//                ),
//
//            createPose(
//                0.146, -0.2565, 0.005,
//                0.908, 0.201, 0.259, 0.261),
//
//            createPose(
//                -0.038, 0.167, 0.237,
//                0.079, 0.365, 0.368, 0.852)
        };

        // 在 A->B、B->C、C->A 之间分别插入20个点
        generateTrajectory(20);

        // 每1秒发布一个点
        timer_ = this->create_wall_timer(
            std::chrono::seconds(1),
            std::bind(&PosePublisher::publishPose, this));

        RCLCPP_INFO(
            this->get_logger(),
            "Trajectory generated: %zu points, publish interval: 1 second",
            trajectory_.size());
    }

private:
    geometry_msgs::msg::Pose createPose(
        double x,
        double y,
        double z,
        double qx,
        double qy,
        double qz,
        double qw)
    {
        geometry_msgs::msg::Pose pose;

        pose.position.x = x;
        pose.position.y = y;
        pose.position.z = z;

        // 对输入四元数进行归一化
        const double norm =
            std::sqrt(qx * qx + qy * qy + qz * qz + qw * qw);

        if (norm > 1e-12) {
            pose.orientation.x = qx / norm;
            pose.orientation.y = qy / norm;
            pose.orientation.z = qz / norm;
            pose.orientation.w = qw / norm;
        } else {
            pose.orientation.x = 0.0;
            pose.orientation.y = 0.0;
            pose.orientation.z = 0.0;
            pose.orientation.w = 1.0;
        }

        return pose;
    }

    geometry_msgs::msg::Quaternion slerp(
        const geometry_msgs::msg::Quaternion &q0,
        const geometry_msgs::msg::Quaternion &q1,
        double t)
    {
        geometry_msgs::msg::Quaternion qa = q0;
        geometry_msgs::msg::Quaternion qb = q1;

        double dot =
            qa.x * qb.x +
            qa.y * qb.y +
            qa.z * qb.z +
            qa.w * qb.w;

        // 保证沿四元数最短路径插值
        if (dot < 0.0) {
            qb.x = -qb.x;
            qb.y = -qb.y;
            qb.z = -qb.z;
            qb.w = -qb.w;
            dot = -dot;
        }

        dot = std::clamp(dot, -1.0, 1.0);

        geometry_msgs::msg::Quaternion result;

        // 两个四元数非常接近时，使用线性插值避免除零
        if (dot > 0.9995) {
            result.x = qa.x + t * (qb.x - qa.x);
            result.y = qa.y + t * (qb.y - qa.y);
            result.z = qa.z + t * (qb.z - qa.z);
            result.w = qa.w + t * (qb.w - qa.w);
        } else {
            const double theta_0 = std::acos(dot);
            const double sin_theta_0 = std::sin(theta_0);
            const double theta = theta_0 * t;

            const double s0 =
                std::sin(theta_0 - theta) / sin_theta_0;
            const double s1 =
                std::sin(theta) / sin_theta_0;

            result.x = s0 * qa.x + s1 * qb.x;
            result.y = s0 * qa.y + s1 * qb.y;
            result.z = s0 * qa.z + s1 * qb.z;
            result.w = s0 * qa.w + s1 * qb.w;
        }

        // 再次归一化
        const double norm = std::sqrt(
            result.x * result.x +
            result.y * result.y +
            result.z * result.z +
            result.w * result.w);

        if (norm > 1e-12) {
            result.x /= norm;
            result.y /= norm;
            result.z /= norm;
            result.w /= norm;
        } else {
            result.x = 0.0;
            result.y = 0.0;
            result.z = 0.0;
            result.w = 1.0;
        }

        return result;
    }

    geometry_msgs::msg::Pose interpolatePose(
        const geometry_msgs::msg::Pose &start,
        const geometry_msgs::msg::Pose &end,
        double t)
    {
        geometry_msgs::msg::Pose result;

        // 位置线性插值
        result.position.x =
            start.position.x +
            t * (end.position.x - start.position.x);

        result.position.y =
            start.position.y +
            t * (end.position.y - start.position.y);

        result.position.z =
            start.position.z +
            t * (end.position.z - start.position.z);

        // 姿态使用四元数球面插值
        result.orientation =
            slerp(start.orientation, end.orientation, t);

        return result;
    }

    void generateTrajectory(std::size_t interpolation_count)
    {
        trajectory_.clear();

        if (waypoints_.size() < 2) {
            trajectory_ = waypoints_;
            return;
        }

        // 包含 C->A，因此这里对所有三个段进行循环处理
        for (std::size_t segment = 0;
             segment < waypoints_.size();
             ++segment)
        {
            const auto &start = waypoints_[segment];
            const auto &end =
                waypoints_[(segment + 1) % waypoints_.size()];

            // j = 0：当前段起点
            // j = 1~20：20个中间插补点
            //
            // 不在当前段加入终点，因为终点会成为下一段的起点，
            // 这样可以避免重复发布 B、C、A。
            for (std::size_t j = 0;
                 j <= interpolation_count;
                 ++j)
            {
                const double t =
                    static_cast<double>(j) /
                    static_cast<double>(interpolation_count + 1);

                trajectory_.push_back(
                    interpolatePose(start, end, t));
            }
        }
    }

    void publishPose()
    {
        if (trajectory_.empty()) {
            RCLCPP_WARN(this->get_logger(), "Trajectory is empty");
            return;
        }

        const auto &pose = trajectory_[current_index_];
        publisher_->publish(pose);

        RCLCPP_INFO(
            this->get_logger(),
            "Publishing point [%zu/%zu], position: "
            "x=%.4f, y=%.4f, z=%.4f",
            current_index_ + 1,
            trajectory_.size(),
            pose.position.x,
            pose.position.y,
            pose.position.z);

        current_index_ =
            (current_index_ + 1) % trajectory_.size();
    }

    rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;

    // 原始的A、B、C三个点
    std::vector<geometry_msgs::msg::Pose> waypoints_;

    // 插补后用于循环发布的轨迹
    std::vector<geometry_msgs::msg::Pose> trajectory_;

    std::size_t current_index_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PosePublisher>());
    rclcpp::shutdown();
    return 0;
}
