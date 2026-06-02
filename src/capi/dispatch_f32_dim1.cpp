/// Factory TU for (value_type = float, input_dim = 1). See dispatch_f64_dim1.

#include <baobzi/detail/c_binding.hpp>

namespace baobzi::capi {

auto make_eval_f32_dim1(int degree, int output_dim, baobzi_func_f32_t f,
                        void *data, const float *a, const float *b,
                        double tol, const baobzi::options &opts)
    -> IEval<float> * {
    return make_eval_impl<float, 1>(degree, output_dim, f, data, a, b, tol,
                                    opts);
}

} // namespace baobzi::capi
