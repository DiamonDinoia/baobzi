/// Factory TU for (value_type = double, input_dim = 2). See dispatch_f64_dim1.

#include <baobzi/detail/c_binding.hpp>

namespace baobzi::capi {

auto make_eval_f64_dim2(int degree, int output_dim, baobzi_func_f64_t f,
                        void *data, const double *a, const double *b,
                        double tol, const baobzi::options &opts)
    -> IEval<double> * {
    return make_eval_impl<double, 2>(degree, output_dim, f, data, a, b, tol,
                                     opts);
}

} // namespace baobzi::capi
