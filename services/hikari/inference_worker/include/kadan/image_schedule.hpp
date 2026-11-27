#pragma once
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace kadan::image {
// FLUX.2 FlowMatchEulerDiscreteScheduler: empirical mu shift with exponential
// time shifting and deterministic Euler integration.
struct Schedule {
    std::array<float, 101> sigma{};
    std::size_t steps = 0;
};

inline double compute_empirical_mu(std::size_t tokens, std::size_t steps) {
    constexpr double a1 = 8.73809524e-5, b1 = 1.89833333;
    constexpr double a2 = 0.00016927, b2 = 0.45666666;
    if (tokens > 4300)
        return a2 * double(tokens) + b2;
    const double m_200 = a2 * double(tokens) + b2;
    const double m_10 = a1 * double(tokens) + b1;
    const double a = (m_200 - m_10) / 190.0;
    const double b = m_200 - 200.0 * a;
    return a * double(steps) + b;
}

inline Schedule schedule(std::size_t tokens, std::size_t steps) {
    if (tokens == 0 || tokens > 18432 || steps < 2 || steps > 100)
        throw std::invalid_argument("image_schedule_bounds");
    Schedule s;
    s.steps = steps;
    const double mu = compute_empirical_mu(tokens, steps);
    const double e = std::exp(mu);
    for (std::size_t i = 0; i < steps; ++i) {
        const double t =
            1.0 + (1.0 / double(steps) - 1.0) * double(i) / double(steps - 1);
        s.sigma[i] = t >= 1.0 ? 1.f : float(e / (e + (1.0 / t - 1.0)));
    }
    s.sigma[steps] = 0.f;
    return s;
}
}  // namespace kadan::image
