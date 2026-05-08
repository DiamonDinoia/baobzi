#ifndef BAOBZI_DETAIL_TOL_KIND_HPP
#define BAOBZI_DETAIL_TOL_KIND_HPP

namespace baobzi {

/// Tolerance interpretation for the tree's adaptive refinement.
enum class TolKind : int {
    RelativeTail = 0,   ///< relative tail-coefficient estimate (1D only)
    AbsoluteTail = 1,   ///< absolute tail-coefficient estimate (1D only)
    RelativeMax  = 2,   ///< sample-based, max-abs relative error
    AbsoluteMax  = 3,   ///< sample-based, max-abs absolute error
    RelativeL2   = 4,   ///< sample-based, L2 relative error
    AbsoluteL2   = 5,   ///< sample-based, L2 absolute error
};

namespace detail {
/// Internal fit-time configuration; mirror of `baobzi::options` plus the
/// shape parameters resolved at the public API boundary.
struct TreeInput {
    int     input_dim              = 0;
    int     output_dim             = 1;
    int     degree                 = 8;
    double  tol                    = 0.0;
    int     max_depth              = 50;
    int     max_memory_mib         = 4;
    bool    allow_max_depth_leaves = false;
    TolKind tol_kind               = TolKind::RelativeMax;
};

/// Sample-grid resolution per axis used by sample-based tolerance kinds
/// (`RelativeMax`, `AbsoluteMax`, `RelativeL2`, `AbsoluteL2`). 8 samples
/// per axis is dense enough to expose ringing from a degree-8 fit while
/// keeping ND fit cost bounded (8^Dim per panel).
inline constexpr int kFitSamplesPerDim = 8;
} // namespace detail

} // namespace baobzi

#endif // BAOBZI_DETAIL_TOL_KIND_HPP
