"""
    Baobzi

Julia wrapper for the baobzi piecewise-Chebyshev function approximator.

# Overview

`Baobzi` wraps the `libbaobzi_c` C ABI to provide fast, accurate approximations
of smooth functions over box-shaped domains in 1–3 dimensions with 1–3 output
components.

## User-function calling convention

The function `f` passed to `fit` must accept:
- **dim == 1**: a single `Float64` (or `Float32`) scalar.
- **dim > 1**: a `NTuple{dim, Float64}` (or `Float32`) of coordinates.

It must return:
- **out_dim == 1**: a scalar.
- **out_dim > 1**: an indexable collection of length `out_dim`.

## Example

```julia
using Baobzi

# 1D scalar
b = fit(x -> exp(0.5x) + sin(3x), 0.0, 1.0, 1e-8)
b(0.5)                       # scalar result

# 2D -> 3D vector
b2 = fit((x, y) -> (sin(x)*cos(y), x+y, x*y), [0.0,0.0], [1.0,1.0], 1e-6; out_dim=3)
b2([0.5, 0.3])               # Vector{Float64} of length 3

# Batch eval
X = rand(100, 2)             # 100 points, 2 coordinates each
R = eval_multi(b2, X)        # 100×3 matrix
```

## Library resolution

`libbaobzi_c` is located, in order:

1. the `LIBBAOBZI_C` environment variable (an explicit path — always wins);
2. `deps/deps.jl`, written by `deps/build.jl` (run via `Pkg.build("Baobzi")`);
3. the loader search path (`Libdl.find_library`);
4. a sibling CMake `build*/` tree found by walking up from the package dir.

So an in-repo `using Baobzi` / `Pkg.test()` works without any env var as long
as the project has been built once (a `build*/libbaobzi_c.<ext>` exists).
"""
module Baobzi

using Libdl

# ---------------------------------------------------------------------------
# Library path
# ---------------------------------------------------------------------------

# Walk up from `start`, returning the first `build*/libbaobzi_c.<dlext>` found
# (the canonical location of an in-repo CMake build), or `nothing`.
function _find_in_build_tree(start::AbstractString)
    libname = "libbaobzi_c." * Libdl.dlext
    dir = abspath(start)
    while true
        for bd in sort(filter(isdir, readdir(dir; join = true)))
            if startswith(basename(bd), "build")
                cand = joinpath(bd, libname)
                isfile(cand) && return cand
            end
        end
        parent = dirname(dir)
        parent == dir && return nothing   # reached filesystem root
        dir = parent
    end
end

function _resolve_libbaobzi()
    # 1. explicit override
    env = get(ENV, "LIBBAOBZI_C", "")
    isempty(env) || return env

    # 2. path baked in by deps/build.jl (`Pkg.build("Baobzi")`)
    depsjl = normpath(joinpath(@__DIR__, "..", "deps", "deps.jl"))
    if isfile(depsjl)
        path = include(depsjl)
        path isa AbstractString && isfile(path) && return path
    end

    # 3. the loader search path
    found = Libdl.find_library(["libbaobzi_c", "baobzi_c"])
    isempty(found) || return found

    # 4. a sibling CMake build tree
    walked = _find_in_build_tree(@__DIR__)
    walked === nothing || return walked

    # Nothing found — return the bare soname so the __init__ warning is clear.
    return "libbaobzi_c." * Libdl.dlext
end

const LIBBAOBZI = _resolve_libbaobzi()

# Verify at load time so the error is clear.
function __init__()
    if !isfile(LIBBAOBZI) && Libdl.find_library([LIBBAOBZI]) == ""
        @warn "Baobzi: could not locate libbaobzi_c ($LIBBAOBZI). Set the " *
              "LIBBAOBZI_C env var, run Pkg.build(\"Baobzi\"), or build the " *
              "project (a build*/ dir next to the repo root)."
    end
end

# ---------------------------------------------------------------------------
# Enumerations (mirrored as Cint constants)
# ---------------------------------------------------------------------------

# baobzi_tol_kind_t
const BAOBZI_RELATIVE_TAIL  = Cint(0)
const BAOBZI_ABSOLUTE_TAIL  = Cint(1)
const BAOBZI_RELATIVE_MAX   = Cint(2)
const BAOBZI_ABSOLUTE_MAX   = Cint(3)
const BAOBZI_RELATIVE_L2    = Cint(4)
const BAOBZI_ABSOLUTE_L2    = Cint(5)

# baobzi_eval_policy_t
const BAOBZI_LATENCY    = Cint(0)
const BAOBZI_THROUGHPUT = Cint(1)
const BAOBZI_BALANCED   = Cint(2)

# baobzi_dtype_t
const BAOBZI_F64 = Cint(0)
const BAOBZI_F32 = Cint(1)

# ---------------------------------------------------------------------------
# Options struct (maps directly to baobzi_options_t)
# ---------------------------------------------------------------------------

"""
    BaobziOptions(; tol_kind, max_depth, max_memory_mib, allow_max_depth_leaves, min_uniform_depth)

Mirror of the C `baobzi_options_t` struct.  All fields are `Cint` so the struct
is blittable for `ccall`.

Defaults match `baobzi_default_options` = {RELATIVE_MAX, 50, 4, 0, 0}.
"""
struct BaobziOptions
    tol_kind              :: Cint   # baobzi_tol_kind_t
    max_depth             :: Cint
    max_memory_mib        :: Cint
    allow_max_depth_leaves:: Cint
    min_uniform_depth     :: Cint
end

function BaobziOptions(;
        tol_kind               = BAOBZI_RELATIVE_MAX,
        max_depth              = Cint(50),
        max_memory_mib         = Cint(4),
        allow_max_depth_leaves = Cint(0),
        min_uniform_depth      = Cint(0))
    BaobziOptions(Cint(tol_kind), Cint(max_depth), Cint(max_memory_mib),
                  Cint(allow_max_depth_leaves), Cint(min_uniform_depth))
end

# ---------------------------------------------------------------------------
# Baobzi handle struct
# ---------------------------------------------------------------------------

"""
    Baobzi{T}

Opaque wrapper around a `baobzi_t` handle.  `T` is `Float64` or `Float32`.
Call the handle directly as a function, or use `eval_multi` / `eval_sorted`.

Do not copy; memory is managed by a finalizer.
"""
mutable struct BaobziFn{T}
    ptr     :: Ptr{Cvoid}   # baobzi_t  (opaque struct*)
    dim     :: Int
    out_dim :: Int
    # Keep the CFunction alive for the lifetime of the object.
    # (The cfun is only needed during fit, but pinning it on the struct is the
    # safest idiom against premature GC.)
    _cfun   :: Any
end

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

@inline function _last_error()::String
    ptr = ccall((:baobzi_last_error, LIBBAOBZI), Cstring, ())
    ptr == C_NULL ? "(no error message)" : unsafe_string(ptr)
end

# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------

"""
    fit(f, a, b, tol; dim=length(a), out_dim=1, degree=11, dtype=Float64,
        policy=BAOBZI_BALANCED, options=BaobziOptions())  -> BaobziFn{dtype}

Approximate `f` over the axis-aligned box `[a,b]` to tolerance `tol`.

`a` and `b` can be scalars (for dim==1) or `AbstractVector`s.

The user function `f` is called as:
- dim == 1 : `f(x::T)` where `x` is a scalar.
- dim > 1  : `f(x1::T, x2::T, ...)` — one scalar argument per dimension,
             dispatched via `f(coords...)` where `coords` is an `NTuple`.

Return value of `f`:
- out_dim == 1 : a scalar convertible to `T`.
- out_dim > 1  : an indexable object of length `out_dim`.

Supported combinations: `dim` ∈ {1,2,3}, `out_dim` ∈ {1,2,3},
`degree` ∈ {7,11,15}.
"""
function fit(f, a, b, tol::Real;
             dim     = (a isa AbstractVector ? length(a) : 1),
             out_dim = 1,
             degree  = 11,
             dtype   = Float64,
             policy  = BAOBZI_BALANCED,
             options :: BaobziOptions = BaobziOptions())

    T   = (dtype == Float32 ? Float32 : Float64)
    CT  = (T == Float64 ? Cdouble : Cfloat)

    # Coerce corners to same-type vectors.
    av  = a isa AbstractVector ? CT[CT(x) for x in a] : CT[CT(a)]
    bv  = b isa AbstractVector ? CT[CT(x) for x in b] : CT[CT(b)]
    @assert length(av) == dim && length(bv) == dim

    # -----------------------------------------------------------------------
    # Build the trampoline closure.
    # -----------------------------------------------------------------------
    err = Ref{Any}(nothing)

    # Capture `dim` and `out_dim` by value through closure.
    _dim     = dim
    _out_dim = out_dim

    if T == Float64
        function trampoline64(xptr::Ptr{Cdouble}, yptr::Ptr{Cdouble}, ::Ptr{Cvoid})
            if err[] !== nothing
                for j in 1:_out_dim; unsafe_store!(yptr, NaN, j); end
                return nothing
            end
            try
                if _dim == 1
                    coords = unsafe_load(xptr, 1)
                    result = f(coords)
                else
                    coords = ntuple(i -> unsafe_load(xptr, i), _dim)
                    result = f(coords...)
                end
                if _out_dim == 1
                    unsafe_store!(yptr, Cdouble(result), 1)
                else
                    for j in 1:_out_dim
                        unsafe_store!(yptr, Cdouble(result[j]), j)
                    end
                end
            catch e
                err[] = e
                for j in 1:_out_dim; unsafe_store!(yptr, NaN, j); end
            end
            return nothing
        end
        cfun = @cfunction($trampoline64, Cvoid, (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cvoid}))
        ptr = GC.@preserve cfun begin
            ccall((:baobzi_fit_f64, LIBBAOBZI), Ptr{Cvoid},
                  (Ptr{Cvoid}, Ptr{Cvoid}, Cint, Cint, Cint,
                   Ptr{Cdouble}, Ptr{Cdouble}, Cdouble, Cint, Ref{BaobziOptions}),
                  cfun, C_NULL, Cint(dim), Cint(out_dim), Cint(degree),
                  av, bv, Cdouble(tol), Cint(policy), options)
        end
    else
        function trampoline32(xptr::Ptr{Cfloat}, yptr::Ptr{Cfloat}, ::Ptr{Cvoid})
            if err[] !== nothing
                for j in 1:_out_dim; unsafe_store!(yptr, NaN32, j); end
                return nothing
            end
            try
                if _dim == 1
                    coords = unsafe_load(xptr, 1)
                    result = f(coords)
                else
                    coords = ntuple(i -> unsafe_load(xptr, i), _dim)
                    result = f(coords...)
                end
                if _out_dim == 1
                    unsafe_store!(yptr, Cfloat(result), 1)
                else
                    for j in 1:_out_dim
                        unsafe_store!(yptr, Cfloat(result[j]), j)
                    end
                end
            catch e
                err[] = e
                for j in 1:_out_dim; unsafe_store!(yptr, NaN32, j); end
            end
            return nothing
        end
        cfun = @cfunction($trampoline32, Cvoid, (Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cvoid}))
        ptr = GC.@preserve cfun begin
            ccall((:baobzi_fit_f32, LIBBAOBZI), Ptr{Cvoid},
                  (Ptr{Cvoid}, Ptr{Cvoid}, Cint, Cint, Cint,
                   Ptr{Cfloat}, Ptr{Cfloat}, Cdouble, Cint, Ref{BaobziOptions}),
                  cfun, C_NULL, Cint(dim), Cint(out_dim), Cint(degree),
                  av, bv, Cdouble(tol), Cint(policy), options)
        end
    end

    # If the user function threw, re-raise that (more informative).
    if err[] !== nothing
        throw(err[])
    end

    if ptr == C_NULL
        error("baobzi fit failed: " * _last_error())
    end

    handle = BaobziFn{T}(ptr, dim, out_dim, cfun)
    finalizer(handle) do h
        if h.ptr != C_NULL
            h.ptr = ccall((:baobzi_free, LIBBAOBZI), Ptr{Cvoid}, (Ptr{Cvoid},), h.ptr)
        end
    end
    return handle
end

# ---------------------------------------------------------------------------
# Single-point eval — make BaobziFn callable
# ---------------------------------------------------------------------------

"""
    (b::BaobziFn{Float64})(x) -> scalar or Vector{Float64}

Evaluate at a single point `x` (scalar for dim==1, or an `AbstractVector`
of length `dim` for dim > 1).  Returns a scalar for out_dim==1 or a
`Vector{Float64}` for out_dim > 1.
"""
function (b::BaobziFn{Float64})(x)
    xv = _coerce_x(x, b.dim, Float64)
    y  = Vector{Float64}(undef, b.out_dim)
    ccall((:baobzi_eval_f64, LIBBAOBZI), Cvoid,
          (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}),
          b.ptr, xv, y)
    return b.out_dim == 1 ? y[1] : y
end

function (b::BaobziFn{Float32})(x)
    xv = _coerce_x(x, b.dim, Float32)
    y  = Vector{Float32}(undef, b.out_dim)
    ccall((:baobzi_eval_f32, LIBBAOBZI), Cvoid,
          (Ptr{Cvoid}, Ptr{Cfloat}, Ptr{Cfloat}),
          b.ptr, xv, y)
    return b.out_dim == 1 ? y[1] : y
end

# ---------------------------------------------------------------------------
# eval_multi  (AoS batch)
# ---------------------------------------------------------------------------

"""
    eval_multi(b::BaobziFn, X) -> result

Evaluate at `n` points.  `X` can be:
- A length-`n` `AbstractVector` (dim==1 only).
- An `n × dim` `AbstractMatrix` (row = one point, column-major storage is
  handled by transposing to produce point-major C buffers).

Returns:
- A `Vector` of length `n` when `out_dim == 1`.
- An `n × out_dim` `Matrix` when `out_dim > 1`.
"""
function eval_multi(b::BaobziFn{Float64}, X)
    xbuf, n = _pack_x(X, b.dim, Float64)
    res = Vector{Float64}(undef, n * b.out_dim)
    ccall((:baobzi_eval_multi_f64, LIBBAOBZI), Cvoid,
          (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Csize_t),
          b.ptr, xbuf, res, Csize_t(n))
    return _unpack_y(res, n, b.out_dim)
end

function eval_multi(b::BaobziFn{Float32}, X)
    xbuf, n = _pack_x(X, b.dim, Float32)
    res = Vector{Float32}(undef, n * b.out_dim)
    ccall((:baobzi_eval_multi_f32, LIBBAOBZI), Cvoid,
          (Ptr{Cvoid}, Ptr{Cfloat}, Ptr{Cfloat}, Csize_t),
          b.ptr, xbuf, res, Csize_t(n))
    return _unpack_y(res, n, b.out_dim)
end

# ---------------------------------------------------------------------------
# eval_sorted  (sorted 1D only)
# ---------------------------------------------------------------------------

"""
    eval_sorted(b::BaobziFn, x::AbstractVector) -> result

Evaluate at `n` sorted 1D points.  `dim` must be 1.  The caller guarantees
`x[i] ≤ x[i+1]`.  Returns the same shape as `eval_multi`.
"""
function eval_sorted(b::BaobziFn{Float64}, x::AbstractVector)
    b.dim == 1 || error("eval_sorted requires dim == 1")
    n   = length(x)
    xv  = Vector{Float64}(x)
    res = Vector{Float64}(undef, n * b.out_dim)
    ccall((:baobzi_eval_sorted_f64, LIBBAOBZI), Cvoid,
          (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Csize_t),
          b.ptr, xv, res, Csize_t(n))
    return _unpack_y(res, n, b.out_dim)
end

function eval_sorted(b::BaobziFn{Float32}, x::AbstractVector)
    b.dim == 1 || error("eval_sorted requires dim == 1")
    n   = length(x)
    xv  = Vector{Float32}(x)
    res = Vector{Float32}(undef, n * b.out_dim)
    ccall((:baobzi_eval_sorted_f32, LIBBAOBZI), Cvoid,
          (Ptr{Cvoid}, Ptr{Cfloat}, Ptr{Cfloat}, Csize_t),
          b.ptr, xv, res, Csize_t(n))
    return _unpack_y(res, n, b.out_dim)
end

# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------

"""
    memory_usage(b::BaobziFn) -> Int

Return the number of bytes occupied by the approximation tree.
"""
memory_usage(b::BaobziFn) =
    Int(ccall((:baobzi_memory_usage, LIBBAOBZI), Csize_t, (Ptr{Cvoid},), b.ptr))

"""
    print_stats(b::BaobziFn)

Print internal statistics via the C library (to stdout).
"""
print_stats(b::BaobziFn) =
    ccall((:baobzi_print_stats, LIBBAOBZI), Cvoid, (Ptr{Cvoid},), b.ptr)

function Base.show(io::IO, b::BaobziFn{T}) where T
    bytes = memory_usage(b)
    print(io, "BaobziFn{$T}(dim=$(b.dim), out_dim=$(b.out_dim), $(bytes) bytes)")
end

# ---------------------------------------------------------------------------
# Internal utility helpers
# ---------------------------------------------------------------------------

# Coerce a user-supplied point to a Vector{T} of length dim.
function _coerce_x(x, dim::Int, ::Type{T}) where T
    if dim == 1
        return T[T(x isa AbstractVector ? x[1] : x)]
    else
        xv = x isa AbstractVector ? x : (x,)
        length(xv) == dim || error("Expected $dim coordinates, got $(length(xv))")
        return T[T(c) for c in xv]
    end
end

# Pack an n-point array into a point-major C buffer of type T.
# Returns (buf::Vector{T}, n::Int).
function _pack_x(X, dim::Int, ::Type{T}) where T
    if X isa AbstractVector && dim == 1
        n = length(X)
        return Vector{T}(X), n
    elseif X isa AbstractMatrix
        # X is n × dim in Julia (column-major).  We need a point-major
        # flat buffer: [p0_x0, p0_x1, ..., p0_x{dim-1}, p1_x0, ...].
        # X[i,j] is the j-th coordinate of the i-th point.
        n, d = size(X)
        d == dim || error("Matrix has $d columns but dim=$dim")
        # Build row-major flat buffer explicitly.
        buf = Vector{T}(undef, n * dim)
        for i in 1:n
            for j in 1:dim
                buf[(i-1)*dim + j] = T(X[i, j])
            end
        end
        return buf, n
    else
        error("X must be a Vector (dim==1) or an n×dim Matrix")
    end
end

# Reshape the flat AoS result buffer.
function _unpack_y(res::Vector{T}, n::Int, out_dim::Int) where T
    if out_dim == 1
        return res   # already length-n
    else
        # res is [p0_y0..p0_y{out-1}, p1_y0...] — point-major.
        # Return an n × out_dim matrix (Julia column-major).
        M = Matrix{T}(undef, n, out_dim)
        for i in 1:n
            for j in 1:out_dim
                M[i, j] = res[(i-1)*out_dim + j]
            end
        end
        return M
    end
end

# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

export BaobziOptions, BaobziFn, fit, eval_multi, eval_sorted, memory_usage, print_stats
export BAOBZI_RELATIVE_TAIL, BAOBZI_ABSOLUTE_TAIL, BAOBZI_RELATIVE_MAX
export BAOBZI_ABSOLUTE_MAX, BAOBZI_RELATIVE_L2, BAOBZI_ABSOLUTE_L2
export BAOBZI_LATENCY, BAOBZI_THROUGHPUT, BAOBZI_BALANCED
export BAOBZI_F64, BAOBZI_F32

end # module Baobzi
