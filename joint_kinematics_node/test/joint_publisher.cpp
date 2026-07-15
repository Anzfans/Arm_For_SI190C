/*
 * File: pose_publisher.cpp
 * Author: Zhenghao Li
 * Email: lizhenghao@shanghaitech.edu.cn
 * Institute: SIST
 * Created: 2025-06-12
 * Last Modified: 2026-07-12
 */
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <cmath>
#include <vector>
#include <string>

class JointStateTestNode : public rclcpp::Node
{
public:
    JointStateTestNode()
        : Node("joint_state_kinematics_test"),
          current_angle_(0.0),
          target_angle_(2 * M_PI),
          step_size_(0.1)  // 每次递增的弧度值，可根据需要调整
    {
        // 创建发布者，队列深度10
        publisher_ = this->create_publisher<sensor_msgs::msg::JointState>(
            "/joint_states", 10);

        // 定时器，50Hz 发布频率
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(200),
            std::bind(&JointStateTestNode::timer_callback, this));

        // 初始化6个关节名称
        joint_names_ = {
            "joint1", "joint2", "joint3",
            "joint4", "joint5", "joint6"
        };
        dir = {-1, 1, -1, -1, 1, 1};

        RCLCPP_INFO(this->get_logger(),
            "Joint State Kinematics Test Node started. "
            "Publishing from 0.0 to PI with step=%.4f at 50Hz", step_size_);
    }

private:
    void timer_callback()
    {
        auto msg = sensor_msgs::msg::JointState();
        msg.header.stamp = this->now();
        msg.name = joint_names_;

        // 所有6个关节使用相同的当前角度值
        std::vector<double> positions(6, 0);
        std::vector<double> velocities(6, 0.0);
        std::vector<double> efforts(6, 0.0);
        positions[4] = current_angle_;
        for(int i = 0; i < 6; ++i)
            positions[i] = positions[i] * dir[i];

        msg.position = positions;
        msg.velocity = velocities;
        msg.effort = efforts;

        publisher_->publish(msg);

        RCLCPP_DEBUG(this->get_logger(),
            "Published joint angles: %.4f / %.4f", current_angle_, target_angle_);

        // 递增角度，到达PI后停止或重置
        if (current_angle_ < target_angle_) {
            current_angle_ += step_size_;
            // 防止超过目标值
            if (current_angle_ > target_angle_) {
                current_angle_ = target_angle_;
            }
        } else {
            // 到达PI后保持在PI，也可以选择重置为0循环测试
            current_angle_ = 0.0;  // 取消注释可循环测试
            static bool warned = false;
            if (!warned) {
                RCLCPP_INFO(this->get_logger(),
                    "Reached target angle PI. Holding position. "
                    "(Uncomment reset line in code to loop)");
                warned = true;
            }
        }
    }

    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::vector<std::string> joint_names_;
    double current_angle_;
    double target_angle_;
    double step_size_;
    std::vector<int> dir;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<JointStateTestNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
