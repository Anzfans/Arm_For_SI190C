/*
 * File: DH_server.cpp
 * Author: Zhenghao Li
 * Email: lizhenghao@shanghaitech.edu.cn
 * Institute: SIST
 * Created: 2025-04-29
 * Last Modified: 2026-07-14
 */
#include "joint_kinematics_node/DH_server.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
Eigen::Matrix4d make_dh_transform(double a, double alpha, double d, double theta)
{
    const double ct = std::cos(theta);
    const double st = std::sin(theta);
    const double ca = std::cos(alpha);
    const double sa = std::sin(alpha);

    Eigen::Matrix4d A;
    A << ct, -st * ca,  st * sa, a * ct,
         st,  ct * ca, -ct * sa, a * st,
        0.0,       sa,       ca,      d,
        0.0,      0.0,      0.0,    1.0;
    return A;
}
}

DHServer::DHServer(): theta_delta(6, 0.0)
{
}

void DHServer::set_DH_params(int x, const std::vector<double> &&params)
{
    this->joint[x] << params[0], params[1], params[2], params[3];
}
void DHServer::set_theta(int index, double theta)
{
    theta_delta[index] = theta;
}
void DHServer::get_transform(Eigen::Matrix4d &T){
    T = Eigen::Matrix4d::Identity();
    for(size_t i = 0; i < 6; ++i){
        T *= make_dh_transform(get_a(i), get_alpha(i), get_d(i), get_theta(i));
    }
}

void DHServer::get_transform(Eigen::Matrix4d &T, std::vector<double> &t){
    if(t.size() < 6)
        throw std::runtime_error("get_transform requires 6 joint angles");

    T = Eigen::Matrix4d::Identity();
    for(size_t i = 0; i < 6; ++i){
        const double theta = this->joint[i][3] + t[i];
        T *= make_dh_transform(get_a(i), get_alpha(i), get_d(i), theta);
    }
}

void DHServer::get_A03(Eigen::Matrix4d &T, std::vector<double> &t)
{
    if(t.size() < 3)
        throw std::runtime_error("get_A03 requires at least 3 joint angles");

    T = Eigen::Matrix4d::Identity();
    for(size_t i = 0; i < 3; ++i){
        T *= make_dh_transform(get_a(i), get_alpha(i), get_d(i), t[i]);
    }
}

void DHServer::get_A03(Eigen::Matrix4d &T)
{
    T = Eigen::Matrix4d::Identity();
    for(size_t i = 0; i < 3; ++i){
        T *= make_dh_transform(get_a(i), get_alpha(i), get_d(i), get_theta(i));
    }
}



double DHServer::get_a(int joint)
{
    return this->joint[joint][0];
}

double DHServer::get_alpha(int joint)
{
    return this->joint[joint][1];
}

double DHServer::get_d(int joint)
{
    return this->joint[joint][2];
}

double DHServer::get_theta(int joint)
{
    return this->joint[joint][3] + this->theta_delta[joint];
}
double DHServer::get_delta(int joint)
{
    return this->theta_delta[joint];
}
double DHServer::get_d0(int joint)
{
    return this->joint[joint][3];
}

