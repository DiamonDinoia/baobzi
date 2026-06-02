/// Factory TU for (value_type = double, input_dim = 1). Instantiating
/// `make_eval_impl<double, 1>` here forces the EvalImpl<double, Deg, 1, OUT>
/// instantiations (and their baobzi::fit chains) into this object file, so
/// the per-shape codegen compiles in parallel with the other dispatch TUs.

#include <baobzi/detail/c_binding.hpp>

namespace baobzi::capi {

auto make_eval_f64_dim1(int degree, int output_dim, baobzi_func_f64_t f,
                        void *data, const double *a, const double *b,
                        double tol, const baobzi::options &opts)
    -> IEval<double> * {
    return make_eval_impl<double, 1>(degree, output_dim, f, data, a, b, tol,
                                     opts);
}

} // namespace baobzi::capi
