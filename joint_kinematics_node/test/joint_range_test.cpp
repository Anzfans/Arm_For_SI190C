/*
 * File: joint_range_test.cpp
 * Author: Zhenghao Li
 * Email: lizhenghao@shanghaitech.edu.cn
 * Institute: SIST
 * Created: 2026-07-12
 * Last Modified: 2026-07-13
 *
 * Description: Test node that subscribes to /joint_states and validates
 *              each received JointState message against per-joint angle
 *              ranges using the inRange() utility function.
 */

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include "utility/utility.hpp"
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>

class JointRangeTestNode : public rclcpp::Node
{
public:
    JointRangeTestNode()
        : Node("joint_range_test_node"), range_(6)
    {
        // ---------------------------------------------------------------
        // Hardcoded defaults — mirror the range constraints in
        // joint_kinematics_node.cpp so the test is meaningful out of
        // the box.  All joints default to [-π, π] (full circle); only
        // joint 2 (index 1) is explicitly narrowed to [0, π].
        // ---------------------------------------------------------------
        range_[0].set(M_PI / 2,  2 * M_PI);
        range_[1].set(0, M_PI);
        range_[2].set(M_PI, 2 * M_PI);
        range_[3].set(M_PI / 2, 2 *M_PI);
        range_[4].set(3 * M_PI / 2,  M_PI / 2);

        // ---------------------------------------------------------------
        // ROS 2 parameters — allow per-joint override without recompiling.
        // Each parameter "joint_N_range" is a 2‑element double array
        // [left, right] in radians.
        // ---------------------------------------------------------------
        for (int i = 0; i < 6; ++i) {
            std::string name = "joint_" + std::to_string(i + 1) + "_range";
            this->declare_parameter(name,
                rclcpp::ParameterValue(
                    std::vector<double>{range_[i].left, range_[i].right}));
        }

        // Read back parameters (may differ from defaults if user overrode)
        for (int i = 0; i < 6; ++i) {
            std::string name = "joint_" + std::to_string(i + 1) + "_range";
            auto val = this->get_parameter(name).as_double_array();
            if (val.size() == 2) {
                range_[i].set(val[0], val[1]);
            }
        }

        // Subscription
        subscription_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states", 10,
            std::bind(&JointRangeTestNode::callback, this,
                      std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(),
            "Joint Range Test Node started.  Listening on /joint_states");
        log_ranges();
    }

private:
    void log_ranges() const
    {
        RCLCPP_INFO(this->get_logger(), "Configured joint ranges (radians):");
        for (size_t i = 0; i < range_.size(); ++i) {
            RCLCPP_INFO(this->get_logger(),
                "  Joint %zu: [%.3f, %.3f]  (%s)",
                i + 1,
                range_[i].left,
                range_[i].right,
                (std::abs(range_[i].right - range_[i].left) >=
                       M_PI * 2.0 - epsilon)
                    ? "full circle"
                    : "");
        }
    }

    void callback(const sensor_msgs::msg::JointState::SharedPtr msg)
    {
        // Guard: the node is designed for a 6‑joint arm
        if (msg->position.size() != range_.size()) {
            RCLCPP_WARN(this->get_logger(),
                "Size mismatch: msg->position.size()=%zu, "
                "range_.size()=%zu.  Skipping this message.",
                msg->position.size(), range_.size());
            return;
        }

        // Per‑joint validation against individual ranges
        std::vector<std::string> failed_joints;
        std::ostringstream detail;
        detail << std::fixed << std::setprecision(4);

        for (size_t i = 0; i < range_.size(); ++i) {
            double angle = msg->position[i];
            bool   valid = inRange(angle, range_[i].left, range_[i].right);

            if (!valid) {
                std::string jname =
                    (i < msg->name.size()) ? msg->name[i]
                                           : ("joint_" + std::to_string(i + 1));
                failed_joints.push_back(jname);
                detail << "  [" << jname << "]  value=" << angle
                       << "  range=[" << range_[i].left
                       << ", " << range_[i].right << "]\n";
            }
        }

        // Log outcome
        if (failed_joints.empty()) {
            RCLCPP_INFO(this->get_logger(),
                "[PASS]  All %zu joints are within their configured ranges.",
                range_.size());
        } else {
            RCLCPP_WARN(this->get_logger(),
                "[FAIL]  %zu / %zu joint(s) OUT OF RANGE:\n%s",
                failed_joints.size(), range_.size(),
                detail.str().c_str());
        }
    }

    // ---- data members ----
    std::vector<axisRange> range_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr
        subscription_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<JointRangeTestNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
