#include <baobzi/baobzi.hpp>

#include <catch2/catch_test_macros.hpp>
#include <cmath>

TEST_CASE("1D1 evaluations", "[baobzi_template]") {
    const double scale_factor = 1.5;
    auto testfun_1d1 = [scale_factor](const double x) { return scale_factor * std::log(x); };

    const double half_l = 1.0;
    const double center = 3.0;
    const double a = center - half_l;
    const double b = center + half_l;
    constexpr double tol = 1e-10;

    auto baobzifunc = baobzi::fit<8>(testfun_1d1, tol, a, b, baobzi::options{},
                                     baobzi::SplitMultiEvalOff);

    SECTION("evaluations at lower left") {
        const double x = a;
        const double y_appx = baobzifunc(x);
        const double y_exact = testfun_1d1(x);
        REQUIRE(std::fabs((y_appx - y_exact) / y_exact) < tol);
    }

    SECTION("evaluations at center") {
        const double y_appx = baobzifunc(center);
        const double y_exact = testfun_1d1(center);
        REQUIRE(std::fabs((y_appx - y_exact) / y_exact) < tol);
    }
}
