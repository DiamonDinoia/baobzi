#ifndef BAOBZI_DETAIL_NUMERICS_HPP
#define BAOBZI_DETAIL_NUMERICS_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <tuple>

#include <polyfit/polyeval.hpp>

#include <baobzi/detail/tol_kind.hpp>
#include <baobzi/detail/value.hpp>

namespace baobzi::detail {

using index_t = std::size_t; ///< Type specifying indexing into flattened tree.

/// Compile-time integer power, `base ** EXP`, via repeated squaring. Used in
/// place of `std::pow` for ND fan-out counts (e.g. `Degree ** input_dim`)
/// where the exponent is a compile-time constant.
template <int EXP, typename T>
constexpr auto powi(T base) -> T {
    if constexpr (EXP == 0) {
        return T{1};
    } else if constexpr (EXP % 2 == 0) {
        const auto half = powi<EXP / 2>(base);
        return half * half;
    } else {
        return base * powi<EXP - 1>(base);
    }
}

/// Number of scalar components in a fit input/output type: 1 for arithmetic
/// scalars, `std::tuple_size_v<T>` for `std::array`-like tuples. Implemented
/// as a function template to keep `std::tuple_size_v<T>` out of the
/// instantiation when `T` is a scalar.
template <typename T>
constexpr auto value_dim() -> std::size_t {
    if constexpr (poly_eval::detail::hasTupleSize_v<T>)
        return std::tuple_size_v<T>;
    else
        return 1;
}

template <typename T>
inline constexpr std::size_t value_dim_v = value_dim<T>();

/// Geometric portion of Baobzi nodes.
template <typename T, std::size_t Dim>
struct Box {
    Value<T, Dim> center;
    Value<T, Dim> half_length;

    Box(const auto &x, const auto &hl) : center{x}, half_length{hl} {}
};

/// True when the leading-coefficient tail estimate of `polyfit` exceeds
/// `tol` for the given relative/absolute kind. 1D-only — the tail estimate
/// is read off the first/last Chebyshev coefficients, which generalise
/// poorly to ND. Sample-based kinds dispatch to `sample_error_below_tolerance`.
template <class Polyfit>
auto tail_error_below_tolerance(TolKind tol_type, double tol,
                                const Polyfit &polyfit) -> bool {
    constexpr std::size_t input_dim  = value_dim_v<typename Polyfit::InputType>;
    constexpr std::size_t output_dim = value_dim_v<typename Polyfit::OutputType>;
    using T = poly_eval::detail::value_type_or_t<typename Polyfit::InputType>;
    if constexpr (input_dim != 1)
        throw std::runtime_error(
            "Baobzi fit error: tail_error check is currently only implemented for 1D input; "
            "use a sample-based tol_type (RELATIVE_MAX, ABSOLUTE_MAX, RELATIVE_L2, ABSOLUTE_L2) instead");

    T maxcoeff{0.0};
    T scaling_factor{1.0};

    if constexpr (input_dim == 1) {
        const auto &coeffs = polyfit.coeffs();
        constexpr std::size_t N = Polyfit::NCOEFFS;
        static_assert(output_dim == 1, "tail_error only implemented for single output in 1D");

        for (std::size_t i = 0; i < 2; ++i)
            maxcoeff = std::max(std::abs(coeffs[i]), maxcoeff);
        scaling_factor = std::max(scaling_factor, std::abs(coeffs[N - 1]));
    }

    if (tol_type == TolKind::RelativeL2 || tol_type == TolKind::RelativeMax)
        return maxcoeff / scaling_factor > tol;
    else
        return maxcoeff > tol;
}

/// True when the maximum/L2 error of `polyfit` measured on a uniform
/// `n_sample_1d`-per-axis grid exceeds `tol`. The chosen `tol_type`
/// selects between max-abs vs. L2 and relative vs. absolute.
template <class Func, class Polyfit>
inline auto sample_error_below_tolerance(int n_sample_1d, TolKind tol_type, double tol,
                                         const typename Polyfit::InputType &center_in,
                                         const typename Polyfit::InputType &half_length_in,
                                         const Func &func, const Polyfit &polyfit) -> bool {
    constexpr std::size_t input_dim  = value_dim_v<typename Polyfit::InputType>;
    constexpr std::size_t output_dim = value_dim_v<typename Polyfit::OutputType>;
    const auto n_sample_1d_sz = static_cast<std::size_t>(n_sample_1d);
    const std::size_t n_samples = powi<static_cast<int>(input_dim)>(n_sample_1d_sz);
    const Value half_length = half_length_in;
    const Value center = center_in;

    double max_abs_err{0.0};
    double max_rel_err{0.0};
    double abs_err_l2{0.0};
    double direct_sum{0.0};
    for (std::size_t linear_index = 0; linear_index < n_samples; ++linear_index) {
        Value<double, input_dim> sample_point;
        std::size_t curr_index = linear_index;

        for (std::size_t dim = 0; dim < input_dim; ++dim) {
            const double dx = 2.0 * half_length[dim] / static_cast<double>(n_sample_1d_sz);
            sample_point[dim] = center[dim] - half_length[dim] + dx / 2.0 +
                                dx * static_cast<double>(curr_index % n_sample_1d_sz);
            curr_index /= n_sample_1d_sz;
        }

        Value<double, output_dim> actual = func(sample_point);
        Value<double, output_dim> approx = polyfit(sample_point);

        for (std::size_t i = 0; i < output_dim; ++i) {
            const double abs_err = std::abs(approx[i] - actual[i]);
            max_abs_err = std::max(max_abs_err, abs_err);
            if (actual[i] != 0.0)
                max_rel_err = std::max(max_rel_err, std::abs(abs_err / actual[i]));
            abs_err_l2 += powi<2>(abs_err);
            direct_sum += powi<2>(actual[i]);
        }
    }

    switch (tol_type) {
    case TolKind::RelativeL2:
        return std::sqrt(abs_err_l2 / direct_sum) > tol;
    case TolKind::AbsoluteL2:
        return std::sqrt(abs_err_l2) / static_cast<double>(n_samples * output_dim) > tol;
    case TolKind::RelativeMax:
        return max_rel_err > tol;
    case TolKind::AbsoluteMax:
        return max_abs_err > tol;
    default:
        throw std::runtime_error("Baobzi fit error: unknown tolerance type for sampling");
    }
}

} // namespace baobzi::detail

#endif // BAOBZI_DETAIL_NUMERICS_HPP
