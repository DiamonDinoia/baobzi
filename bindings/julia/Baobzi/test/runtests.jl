using Test
using Baobzi

# ---------------------------------------------------------------------------
# 1. Accuracy: 1D scalar fit
# ---------------------------------------------------------------------------
@testset "1D scalar accuracy" begin
    f1d = x -> exp(0.5x) + sin(3x)
    b   = fit(f1d, 0.0, 1.0, 1e-8)
    @test b isa BaobziFn{Float64}
    # Use strictly interior points — baobzi uses open-interval domain check;
    # exact boundary coordinates may yield NaN.
    for x in LinRange(0.02, 0.98, 50)
        @test abs(b(x) - f1d(x)) < 1e-5
    end
end

# ---------------------------------------------------------------------------
# 2. 2D -> 3D vector fit
# ---------------------------------------------------------------------------
@testset "2D->3D vector accuracy" begin
    f2d3 = (x, y) -> (sin(x)*cos(y), x + y, x*y)
    b    = fit(f2d3, [0.0, 0.0], [1.0, 1.0], 1e-6; out_dim=3)
    @test b isa BaobziFn{Float64}
    @test b.dim     == 2
    @test b.out_dim == 3

    for (xi, yi) in ((0.1, 0.2), (0.5, 0.7), (0.9, 0.3))
        ref = collect(f2d3(xi, yi))
        got = b([xi, yi])
        @test got isa Vector{Float64}
        @test length(got) == 3
        for k in 1:3
            @test abs(got[k] - ref[k]) < 1e-4
        end
    end
end

# ---------------------------------------------------------------------------
# 3. eval_sorted vs eval_multi parity (dim==1)
# ---------------------------------------------------------------------------
@testset "sorted vs multi parity" begin
    f1d = x -> sin(x) + 0.5cos(2x)
    b   = fit(f1d, 0.0, Float64(pi), 1e-8)
    xs  = sort(rand(200) .* pi)

    r_multi  = eval_multi(b, xs)
    r_sorted = eval_sorted(b, xs)

    @test r_multi == r_sorted
end

# ---------------------------------------------------------------------------
# 4. NaN out-of-domain
# ---------------------------------------------------------------------------
@testset "NaN out-of-domain" begin
    b = fit(x -> x^2, 0.0, 1.0, 1e-8)
    @test isnan(b(2.0))
    @test isnan(b(-1.0))
end

# ---------------------------------------------------------------------------
# 5. Raising Julia closure propagates as Julia exception (does NOT crash)
# ---------------------------------------------------------------------------
@testset "raising closure throws Julia exception" begin
    boom = x -> error("intentional test error")
    @test_throws ErrorException fit(boom, 0.0, 1.0, 1e-8)
end

# ---------------------------------------------------------------------------
# 6. Too-tight tolerance or too-shallow depth raises a descriptive error
# ---------------------------------------------------------------------------
@testset "fit failure raises informative error" begin
    # Very tight tolerance + tiny max_depth forces MaxDepthExceeded.
    opts = BaobziOptions(max_depth=1, max_memory_mib=4)
    # Use a rapidly oscillating function to ensure the approximation can't
    # converge quickly.
    noisy = x -> sin(1000x) + cos(997x)
    err = try
        fit(noisy, 0.0, 1.0, 1e-14; options=opts)
        nothing
    catch e
        e
    end
    @test err !== nothing
    msg = sprint(showerror, err)
    @test occursin(r"MaxDepth|MemoryBudget|fit failed"i, msg)
end

# ---------------------------------------------------------------------------
# 7. eval_multi matrix interface
# ---------------------------------------------------------------------------
@testset "eval_multi matrix" begin
    f2d = (x, y) -> sin(x + y)
    b   = fit(f2d, [0.0, 0.0], [1.0, 1.0], 1e-7)
    n   = 20
    X   = rand(n, 2)
    R   = eval_multi(b, X)
    @test size(R) == (n,)   # out_dim==1 returns Vector of length n
    for i in 1:n
        @test abs(R[i] - sin(X[i,1] + X[i,2])) < 1e-5
    end
end

# ---------------------------------------------------------------------------
# 8. memory_usage is positive
# ---------------------------------------------------------------------------
@testset "memory_usage" begin
    b = fit(x -> cos(x), 0.0, 2.0, 1e-9)
    @test memory_usage(b) > 0
end

# ---------------------------------------------------------------------------
# 9. Float32 fit
# ---------------------------------------------------------------------------
@testset "Float32 fit" begin
    b = fit(x -> Float32(exp(x)), Float32(0), Float32(1), 1e-5; dtype=Float32)
    @test b isa BaobziFn{Float32}
    # Evaluate at interior point (avoid boundary NaN)
    val = b(Float32(0.5))
    @test abs(val - Float32(exp(0.5))) < Float32(1e-3)
end

println("All Baobzi tests passed.")
