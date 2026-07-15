/*
 * File: utility.cpp
 * Author: Zhenghao Li
 * Email: lizhenghao@shanghaitech.edu.cn
 * Institute: SIST
 * Created: 2025-05-08
 * Last Modified: 2026-07-12
 */

#include "utility/utility.hpp"

namespace {
    constexpr double TWO_PI = M_PI * 2.0;
}

double angle_distance(double a, double b) {
    double d = std::fmod(std::fabs(a - b), TWO_PI);
    // fabs returns non-negative → fmod result is non-negative → d < 0 impossible
    if (d > M_PI) d = TWO_PI - d;
    return d;
}

double angle_diff_signed(double a, double b) {
    double d = std::fmod(a - b, TWO_PI);
    if (d < -M_PI) d += TWO_PI;
    else if (d >  M_PI) d -= TWO_PI;
    return d;
}

void axisRange::set(double a, double b){
    left = a;
    right = b;
}

double distance(double target, double angle)
{
    double diff = std::fabs(angle - target);
    return std::fmin(diff, TWO_PI - diff);
}

bool inRange(double angle, double minAngle, double maxAngle) {
    // Normalize all three angles to [0, 2π)
    const double a  = normalizeAngle(angle);
    const double lo = normalizeAngle(minAngle);
    const double hi = normalizeAngle(maxAngle);

    // CCW span from lo to hi
    double span = hi - lo;
    if (span < 0.0) span += TWO_PI;

    // Full-circle check #1: normalized span covers (almost) the whole circle
    if (span >= TWO_PI - 2.0 * epsilon) {
        return true;
    }

    // Full-circle check #2: degenerate case where lo ≈ hi but the raw
    // values differ by ~2π (e.g. range [-π, π], the default unlimited joint).
    // When lo and hi are the same point on the circle but minAngle and
    // maxAngle are distinct real numbers ~2π apart, the range is 360°.
    if (span < epsilon) {
        double rawDiff = std::abs(maxAngle - minAngle);
        if (rawDiff >= TWO_PI - epsilon) {
            return true;
        }
    }

    // CCW distance from lo to angle
    double dist = a - lo;
    if (dist < 0.0) dist += TWO_PI;

    // Within [0, span] with epsilon tolerance on both boundaries:
    //   dist <= span + ε  →  catches angles up to ε past maxAngle
    //   dist >= 2π - ε    →  catches angles up to ε before minAngle
    return dist <= span + epsilon || dist >= TWO_PI - epsilon;
}

bool inRange(std::vector<double> &solutions, std::vector<axisRange> &range){
    // Guard against size mismatch (robustness)
    size_t n = std::min(solutions.size(), range.size());
    for(size_t i = 0; i < n; ++i){
        if(!inRange(solutions[i], range[i].left, range[i].right))
            return false;
    }
    return true;
}

//double getAngleInRange(double angle1, double angle2) {
//    bool in1 = inRange(angle1);
//    bool in2 = inRange(angle2);
//
//    if (in1 && !in2) return angle1;
//    if (!in1 && in2) return angle2;
//
//    throw std::invalid_argument("Both angles are either in or out of (-pi, pi].");
//}


double normalizeAngle(double angle)
{
    angle = std::fmod(angle, TWO_PI); // 先模到 (-2pi, 2pi)
    if (angle < 0.0)
        angle += TWO_PI;              // 转到 [0, 2pi)
    // Avoid negative zero from fmod of exact multiples (-2π, -4π, …)
    return (angle == 0.0) ? 0.0 : angle;
}
double normalizeToPI(double angle)
{
    constexpr double twoPi = 2.0 * M_PI;

    angle = std::remainder(angle, twoPi); // 结果通常在 [-π, π]

    // 将 +π 统一映射成 -π，保证区间为 [-π, π)
    if (angle >= M_PI) {
        angle -= twoPi;
    }

    return angle;
}


double selectBest(std::vector<double> &angle, std::vector<double> &range, double current)
{
    std::vector<bool> flag(angle.size(), 0);
    double closest = TWO_PI;
    int index = -1;
    for(size_t i = 0; i < angle.size(); ++i){
        if(inRange(angle[i], range[0] , range[1])){
            double diff = distance(current, angle[i]);
            if(diff < closest){
                closest = diff;
                index = i;
            }
        }
    }
    if(index == -1)
        throw std::runtime_error("Out of range");
    return angle[index];
}

bool isZero(double value)
{
    if(fabs(value) < 0.001)
        return true;
    return false;
}
