/*
 * File: joint_comm_node.cpp
 * Author: Zhenghao Li
 * Email: lizhenghao@shanghaitech.edu.cn
 * Institute: SIST
 * Created: 2025-05-29
 * Last Modified: 2026-07-13
 */

#include "io_context/io_context.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "serial_driver/serial_driver.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <cmath>
#include <queue>
#include <rclcpp/logger.hpp>
#include <rclcpp/logging.hpp>
#include <sys/types.h>
#include <thread>

using namespace drivers::serial_driver;


double normalizeAngle(double angle)
{
    angle = std::fmod(angle, 2 * M_PI); // 先模到 (-2pi, 2pi)
    if (angle <= 0)
        angle += 2 * M_PI;  // 转到 (0, 2pi)
    return angle;    // 映射到 (0, 2pi]
}

double normalizeAnglePI(double angle)
{
    angle = std::fmod(angle, 2 * M_PI);
    if (angle > M_PI) angle -= 2 * M_PI;
    if (angle <= -M_PI) angle += 2 * M_PI;
    return angle;    // 映射到 (-pi, pi]
}


class JointSerialComm : public rclcpp::Node {

  std::queue<std::vector<uint8_t>> commands;
  std::vector<uint8_t> current_command;
  std::mutex state_mutex_;
  std::thread t_;
  std::shared_ptr<drivers::common::IoContext> io_context_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::TimerBase::SharedPtr command_timer_;
  int reducer_ratio[6];
  uint8_t motor_dir[6];
  double axis_dir[6];
  bool bipolar[6];
  double joint_limit_min[6];
  double joint_limit_max[6];
  double command_rate;
  double target_tolerance;
  int command_joint_delay_ms;
  std::vector<double> rcv_pos;
  std::vector<double> snd_pos;
  std::vector<double> crt_pos;
  std::vector<double> max_joint_velocity;
  std::vector<double> target_pos;
  std::vector<double> command_pos;
  std::vector<bool> target_active;
  std::vector<bool> feedback_received;
  std::vector<bool> missing_feedback_warned;

  void parse(const std::vector<uint8_t> &arr){
      uint32_t i = 0;
      while(i < arr.size()){
          if(arr[i] != 0 || !current_command.empty())
              current_command.push_back(arr[i]);
          if(arr[i] == 0x6B){
              commands.push(current_command);
              current_command.clear();
          }
          ++i;
      }
  }

public:
  JointSerialComm()
  : Node("joint_serial_comm"),
    rcv_pos(6, 0),
    snd_pos(6, 0),
    crt_pos(6, 0),
    max_joint_velocity(6, 0.08),
    target_pos(6, 0),
    command_pos(6, 0),
    target_active(6, false),
    feedback_received(6, false),
    missing_feedback_warned(6, false)
  {
    // Declare and read parameters from config file
    this->declare_parameter<std::string>("serial_port", "/dev/Motor");
    this->declare_parameter<int>("baud_rate", 115200);
    this->declare_parameter<std::vector<long int>>("reducer_ratios", {30L, 50L, 20L, 20L, 20L, 1L});
    this->declare_parameter<std::vector<long int>>("motor_directions", {0L, 0L, 0L, 0L, 1L, 0L});
    this->declare_parameter<std::vector<double>>("axis_directions", {-1.0, 1.0, -1.0, -1.0, 1.0, 1.0});
    this->declare_parameter<std::vector<long int>>("bipolar_joints", {0L, 0L, 0L, 0L, 1L, 0L});
    this->declare_parameter<std::vector<double>>("joint_limits_min", {-6.20, 0.0, -6.20, -6.20, -3.09, 0.0});
    this->declare_parameter<std::vector<double>>("joint_limits_max", {0.0, 6.20, 0.0, 0.0, 3.09, 6.20});
    this->declare_parameter<double>("command_rate", 5.0);
    this->declare_parameter<std::vector<double>>("max_joint_velocity", {0.08, 0.06, 0.06, 0.08, 0.08, 0.10});
    this->declare_parameter<double>("target_tolerance", 0.01);
    this->declare_parameter<int>("command_joint_delay_ms", 20);

    std::string port;
    this->get_parameter("serial_port", port);

    int baud_rate;
    this->get_parameter("baud_rate", baud_rate);

    std::vector<long int> reducer_ratios_vec;
    this->get_parameter("reducer_ratios", reducer_ratios_vec);

    std::vector<long int> motor_dir_vec;
    this->get_parameter("motor_directions", motor_dir_vec);

    std::vector<double> axis_dir_vec;
    this->get_parameter("axis_directions", axis_dir_vec);

    std::vector<long int> bipolar_vec;
    this->get_parameter("bipolar_joints", bipolar_vec);

    std::vector<double> joint_limits_min_vec;
    this->get_parameter("joint_limits_min", joint_limits_min_vec);

    std::vector<double> joint_limits_max_vec;
    this->get_parameter("joint_limits_max", joint_limits_max_vec);

    this->get_parameter("command_rate", command_rate);
    command_rate = std::max(command_rate, 1.0);

    std::vector<double> max_joint_velocity_vec;
    this->get_parameter("max_joint_velocity", max_joint_velocity_vec);

    this->get_parameter("target_tolerance", target_tolerance);
    target_tolerance = std::max(target_tolerance, 0.001);

    this->get_parameter("command_joint_delay_ms", command_joint_delay_ms);
    command_joint_delay_ms = std::max(command_joint_delay_ms, 0);

    // Copy into fixed-size arrays
    for (int i = 0; i < 6; ++i) {
      reducer_ratio[i] = static_cast<int>(reducer_ratios_vec[i]);
      motor_dir[i]     = static_cast<uint8_t>(motor_dir_vec[i]);
      axis_dir[i]      = axis_dir_vec[i];
      bipolar[i]       = static_cast<bool>(bipolar_vec[i]);
      joint_limit_min[i] = joint_limits_min_vec[i];
      joint_limit_max[i] = joint_limits_max_vec[i];
      max_joint_velocity[i] = max_joint_velocity_vec[i];
    }

    // 创建串口驱动实例
    io_context_ = std::make_shared<drivers::common::IoContext>(1);
    serial_port_ = std::make_unique<SerialDriver>(*io_context_);
    SerialPortConfig config(baud_rate, FlowControl::NONE, Parity::NONE, StopBits::ONE);
    serial_port_->init_port(port, config);
    serial_port_->port()->open();
    subscription_ = this->create_subscription<sensor_msgs::msg::JointState>(
      "joint_states", 10,
      std::bind(&JointSerialComm::joint_callback, this, std::placeholders::_1)
    );
    joint_state_pub_ = this->create_publisher<sensor_msgs::msg::JointState>("joint_states_current", 10);

    timer_ = this->create_wall_timer(std::chrono::milliseconds(200),
                                    std::bind(&JointSerialComm::require_position, this));
    const auto command_period = std::chrono::milliseconds(
        std::max(1, static_cast<int>(1000.0 / command_rate)));
    command_timer_ = this->create_wall_timer(command_period,
                                    std::bind(&JointSerialComm::command_timer_callback, this));
    RCLCPP_INFO(this->get_logger(), "JointSerialComm node started, serial initialized on %s", port.c_str());
    RCLCPP_INFO(this->get_logger(), "Interpolated joint commands enabled: rate=%.1f Hz tolerance=%.3f rad",
                command_rate, target_tolerance);
    async_receive();

    home();
  }

private:
  std::unique_ptr<SerialDriver> serial_port_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr subscription_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_pub_;

  int angle_to_pulse(double angle_deg) {
    return static_cast<int>(angle_deg / (2 * M_PI) * 3200);  // 假设 1 度 = 100 脉冲
  }

  double canonicalize_logical_angle(size_t index, double angle) {
    if (bipolar[index]) {
      return normalizeAnglePI(angle);
    }

    if (joint_limit_max[index] <= 0.0 && angle > M_PI) {
      return angle - 2 * M_PI;
    }

    if (joint_limit_min[index] >= 0.0 && angle < -M_PI) {
      return angle + 2 * M_PI;
    }

    return angle;
  }

  bool in_joint_limit(size_t index, double angle) {
    constexpr double kLimitEpsilon = 1e-6;
    return angle >= joint_limit_min[index] - kLimitEpsilon &&
           angle <= joint_limit_max[index] + kLimitEpsilon;
  }

  bool validate_logical_command(size_t index, double logical_angle) {
    if (!in_joint_limit(index, logical_angle)) {
      RCLCPP_ERROR(this->get_logger(),
        "joint%lu command %.3f rad is outside software limit [%.3f, %.3f]; ignored",
        index + 1, logical_angle, joint_limit_min[index], joint_limit_max[index]);
      return false;
    }
    return true;
  }

  void async_receive(){
    auto port = serial_port_->port();
    port->async_receive([this](const std::vector<uint8_t> &data, const size_t &size){
      if(size > 0){
        parse(data);
        while(!commands.empty()){
            std::vector<uint8_t> cmd = commands.front();
            get_pos(cmd);
            commands.pop();
        }
        usleep(2000);
        RCLCPP_DEBUG(this->get_logger(), "async receive %lu", size);
      }
      async_receive();
    });
  }

  void require_position(){
    for(uint8_t i = 1; i <= 6; ++i){
        std::vector<uint8_t> frame = {i, 0x30, 0x6B};
        serial_port_->port()->send(frame);
        usleep(10000);
    }
  }

  void send_frame(uint8_t joint_id, int pulse, uint8_t dir = 0) {
    std::vector<uint8_t> frame = {0x01, 0xFD, 0x00, 0x13, 0x50, 0x0A,
                         0, 0, 0, 0, 0x01, 0x00, 0x6B};

    frame[0] = joint_id;
    frame[2] = dir;
    frame[6] = (pulse >> 24) & 0xFF;
    frame[7] = (pulse >> 16) & 0xFF;
    frame[8] = (pulse >> 8) & 0xFF;
    frame[9] = (pulse) & 0xFF;

    //std::string data(reinterpret_cast<char*>(frame), sizeof(frame));
    serial_port_->port()->send(frame);
  }

  bool send_joint_command(size_t i, double logical_angle) {
      if (!validate_logical_command(i, logical_angle)) {
        return false;
      }
      const double motor_angle = logical_angle * axis_dir[i];
      if (bipolar[i]) {
        // Bipolar joint: range is (-pi, pi], direction encoded by sign
        double cmd_angle = normalizeAnglePI(motor_angle);
        if (std::abs(cmd_angle) < 1e-3) cmd_angle = 0;
        double abs_angle = std::abs(cmd_angle);
        if (std::abs(snd_pos[i] - cmd_angle) < 0.01) return false;

        RCLCPP_INFO(this->get_logger(), "joint%lu:%f logical=%f (bipolar)",
                    i+1, cmd_angle, logical_angle);
        int pulse = angle_to_pulse(abs_angle * reducer_ratio[i]);
        // Direction: positive angle uses motor_dir, negative reverses it
        uint8_t dir = (cmd_angle < 0) ? motor_dir[i] : (1 - motor_dir[i]);
        send_frame(i + 1, pulse, dir);
        snd_pos[i] = cmd_angle;
      } else {
        // Unipolar joint: range is (0, 2pi]
        double cmd_angle = motor_angle;
        if (std::abs(cmd_angle) < 1e-3 || std::abs(cmd_angle - 2 * M_PI) < 1e-3)
          cmd_angle = 0;
        if (std::abs(snd_pos[i] - cmd_angle) < 0.01) return false;

        if (cmd_angle < 0) {
          RCLCPP_ERROR(this->get_logger(),
            "joint%lu motor command %.3f rad is negative after axis conversion; ignored",
            i+1, cmd_angle);
          return false;
        }
        RCLCPP_INFO(this->get_logger(), "joint%lu:%f logical=%f", i+1, cmd_angle, logical_angle);
        int pulse = angle_to_pulse(cmd_angle * reducer_ratio[i]);
        send_frame(i + 1, pulse, motor_dir[i]);
        snd_pos[i] = cmd_angle;
      }
      return true;
  }

  void joint_callback(const sensor_msgs::msg::JointState::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    RCLCPP_INFO(this->get_logger(), "Get target command");
    for (size_t i = 0; i < std::min(msg->position.size(), size_t(6)); ++i) {
      const double logical_angle = canonicalize_logical_angle(i, msg->position[i]);
      if (!validate_logical_command(i, logical_angle)) {
        target_active[i] = false;
        continue;
      }

      target_pos[i] = logical_angle;
      target_active[i] = true;
      missing_feedback_warned[i] = false;
      if (feedback_received[i]) {
        command_pos[i] = crt_pos[i];
      }
      RCLCPP_INFO(this->get_logger(), "joint%lu target queued: %.3f rad", i + 1, logical_angle);
    }
  }

  void command_timer_callback() {
    std::lock_guard<std::mutex> lock(state_mutex_);
    for (size_t i = 0; i < 6; ++i) {
      if (!target_active[i]) {
        continue;
      }

      if (!feedback_received[i]) {
        if (!missing_feedback_warned[i]) {
          RCLCPP_WARN(this->get_logger(),
            "joint%lu target is waiting for /joint_states_current feedback before motion",
            i + 1);
          missing_feedback_warned[i] = true;
        }
        continue;
      }

      const double error = target_pos[i] - command_pos[i];
      if (std::abs(error) <= target_tolerance) {
        command_pos[i] = target_pos[i];
        send_joint_command(i, command_pos[i]);
        target_active[i] = false;
        RCLCPP_INFO(this->get_logger(), "joint%lu reached target %.3f rad", i + 1, target_pos[i]);
        continue;
      }

      const double max_step = max_joint_velocity[i] / command_rate;
      const double step = std::copysign(std::min(std::abs(error), max_step), error);
      command_pos[i] += step;
      if (send_joint_command(i, command_pos[i]) && command_joint_delay_ms > 0) {
        usleep(command_joint_delay_ms * 1000);
      }
    }
  }

  void home(){
    std::vector<uint8_t> homing_setting(20, 0);
    homing_setting[0] = 0x01;
    homing_setting[1] = 0x4C;
    homing_setting[2] = 0xAE;
    homing_setting[3] = 0x01;
    homing_setting[4] = 0x02;
    homing_setting[5] = 0x01;
    homing_setting[6] = (30 >> 8) & 0xff;
    homing_setting[7] = 30 & 0xff;
    homing_setting[8] = (10000 >> 24) & 0xff;
    homing_setting[9] = (10000 >> 16) & 0xff;
    homing_setting[10] = (10000 >> 8) & 0xff;
    homing_setting[11] = 10000 & 0xff;
    homing_setting[12] = (4000 >> 8) & 0xff;
    homing_setting[13] = 4000 & 0xff;
    homing_setting[14] = (400 >> 8) & 0xff;
    homing_setting[15] = 400 & 0xff;
    homing_setting[16] = (60 >> 8) & 0xff;
    homing_setting[17] = 60 & 0xff;
    homing_setting[18] = 0x01;
    homing_setting[19] = 0x6B;

    //std::string data(reinterpret_cast<char*>(frame), sizeof(frame));
    serial_port_->port()->send(homing_setting);
    usleep(100000);
  }

  void get_pos(std::vector<uint8_t> &cmd)
  {
    if(cmd.size() < 2)
      return;
    if(cmd[1] == 0x30 && cmd.size() == 8){

      int pos = (int)cmd[6] + ((int)cmd[5] << 8) + ((int)cmd[4] << 16) + ((int)cmd[3] << 24);
      int index = cmd[0] - 1;
      std::vector<double> publish_pos;
      {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if(index >= 0 && index < 6){
          crt_pos[index] = double(pos) / 3200.0 / reducer_ratio[index] * 2 * M_PI;
          // For bipolar joints: cmd[2] from motor response indicates sign
          if(bipolar[index] && cmd[2] == 0x00){
              crt_pos[index] *= -1;
          }
          crt_pos[index] *= axis_dir[index];
          if(!feedback_received[index]){
            command_pos[index] = crt_pos[index];
          }
          feedback_received[index] = true;
        }
        publish_pos = crt_pos;
      }

      // Publish current joint state
      auto msg = sensor_msgs::msg::JointState();
      msg.header.stamp = this->now();
      msg.name = {"joint1", "joint2", "joint3", "joint4", "joint5", "joint6"};
      msg.position = publish_pos;
      joint_state_pub_->publish(msg);
    }
    if(cmd[1] == 0x00){
      RCLCPP_ERROR(this->get_logger(), "failed");
    }
    if(cmd[1] == 0xF5){
      RCLCPP_ERROR(this->get_logger(), "torque back");
    }
  }

  void read_loop(){
      while(rclcpp::ok()){
          //async_receive();
          std::vector<uint8_t> frame;
          frame.resize(8);
          //RCLCPP_INFO(this->get_logger(), "receive %lu", serial_port_->port()->receive(frame));
          parse(frame);
          while(!commands.empty()){
            std::vector<uint8_t> cmd = commands.front();
            get_pos(cmd);
            commands.pop();
          }
          usleep(2000);
      }

  }
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<JointSerialComm>());
  rclcpp::shutdown();
  return 0;
}
