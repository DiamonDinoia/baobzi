#ifndef BAOBZI_DETAIL_VALUE_HPP
#define BAOBZI_DETAIL_VALUE_HPP

#include <array>
#include <cstddef>
#include <type_traits>

#include <poet/poet.hpp>

namespace baobzi::detail {

/// Scalar/array uniform wrapper. Behaves as a scalar when `N == 1` and as a
/// `std::array<T, N>` otherwise, exposing the same arithmetic operators in
/// both forms so the rest of baobzi can be written dim-agnostic. The four
/// elementwise binary operators all funnel through `apply` / `apply_scalar`,
/// which lets the optimiser see one inlined loop per op rather than four
/// duplicated bodies.
template <typename T, std::size_t N>
class Value {
    using storage_t = std::conditional_t<N == 1, T, std::array<T, N>>;
    storage_t data_{};

    /// Apply binary `op` elementwise against another `Value`.
    template <class Op>
    constexpr Value apply(const Value &rhs, Op op) const {
        if constexpr (N == 1) {
            return Value(static_cast<T>(op(data_, rhs.data_)));
        } else {
            std::array<T, N> r{};
            for (std::size_t i = 0; i < N; ++i)
                r[i] = op(data_[i], rhs.data_[i]);
            return Value(r);
        }
    }

    /// Apply binary `op` against a broadcast scalar.
    template <class Op>
    constexpr Value apply_scalar(const T &rhs, Op op) const {
        if constexpr (N == 1) {
            return Value(static_cast<T>(op(data_, rhs)));
        } else {
            std::array<T, N> r{};
            for (std::size_t i = 0; i < N; ++i)
                r[i] = op(data_[i], rhs);
            return Value(r);
        }
    }

  public:
    template <std::size_t M = N, typename = std::enable_if_t<M == 1>>
    Value(const T &val) : data_(val) {}
    Value(const std::array<T, 1> &arr) { data_ = arr[0]; }

    template <std::size_t M = N, typename = std::enable_if_t<M != 1>>
    Value(const std::array<T, N> &arr) : data_(arr) {}
    Value() = default;
    Value(const Value &) = default;
    Value(Value &&) noexcept = default;
    Value &operator=(const Value &) = default;
    Value &operator=(Value &&) noexcept = default;
    ~Value() = default;

    Value(const T *arr) {
        if constexpr (N == 1) {
            data_ = arr[0];
        } else {
            poet::static_for<N>([&](auto I) {
                constexpr std::size_t i = I;
                data_[i] = arr[i];
            });
        }
    }

    Value operator+(const Value &rhs) const { return apply(rhs, [](T a, T b) { return a + b; }); }
    Value operator-(const Value &rhs) const { return apply(rhs, [](T a, T b) { return a - b; }); }
    Value operator*(const Value &rhs) const { return apply(rhs, [](T a, T b) { return a * b; }); }
    Value operator/(const Value &rhs) const { return apply(rhs, [](T a, T b) { return a / b; }); }

    Value operator+(const T &rhs) const { return apply_scalar(rhs, [](T a, T b) { return a + b; }); }
    Value operator-(const T &rhs) const { return apply_scalar(rhs, [](T a, T b) { return a - b; }); }
    Value operator*(const T &rhs) const { return apply_scalar(rhs, [](T a, T b) { return a * b; }); }
    Value operator/(const T &rhs) const { return apply_scalar(rhs, [](T a, T b) { return a / b; }); }

    constexpr T &operator[](std::size_t idx) {
        if constexpr (N == 1) {
            static_cast<void>(idx);
            return data_;
        } else {
            return data_[idx];
        }
    }

    [[nodiscard]] constexpr const T &operator[](std::size_t idx) const {
        if constexpr (N == 1) {
            static_cast<void>(idx);
            return data_;
        } else {
            return data_[idx];
        }
    }

    constexpr T *begin() {
        if constexpr (N == 1) return &data_;
        else                  return data_.data();
    }
    constexpr T *end() {
        if constexpr (N == 1) return &data_ + 1;
        else                  return data_.data() + N;
    }
    [[nodiscard]] constexpr const T *begin() const {
        if constexpr (N == 1) return &data_;
        else                  return data_.data();
    }
    [[nodiscard]] constexpr const T *end() const {
        if constexpr (N == 1) return &data_ + 1;
        else                  return data_.data() + N;
    }

    [[nodiscard]] T prod() const {
        if constexpr (N == 1) {
            return data_;
        } else {
            T result = T{1};
            for (const auto &val : data_)
                result *= val;
            return result;
        }
    }

    [[nodiscard]] storage_t get() const { return data_; }

    operator T() const {
        static_assert(N == 1, "Can only cast to scalar if N == 1");
        return data_;
    }

    operator std::array<T, N>() const {
        static_assert(N != 1, "Can only cast to array if N != 1");
        return data_;
    }

    [[nodiscard]] const T &scalar() const {
        static_assert(N == 1, "Not a scalar");
        return data_;
    }
    [[nodiscard]] const std::array<T, N> &array() const {
        static_assert(N != 1, "Not an array");
        return data_;
    }

    /// Always-array view: useful for passing the underlying coordinates to
    /// generic vector-of-double sinks (e.g. exception ctors) without
    /// branching on `N` at the call site.
    [[nodiscard]] std::array<T, N> as_array() const {
        if constexpr (N == 1) {
            return std::array<T, N>{data_};
        } else {
            return data_;
        }
    }
};

template <typename T>
Value(const T &) -> Value<T, 1>;

template <typename T, std::size_t N>
Value(const std::array<T, N> &) -> Value<T, N>;

} // namespace baobzi::detail

#endif // BAOBZI_DETAIL_VALUE_HPP
