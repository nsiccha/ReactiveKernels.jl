# Shared factors and terms leave a summed plate's sum (`src/summed_plate_factors.jl`,
# snag `normal-plate-kee-8322185b`). A Normal observation plate with one scale
# sums `(x - location)^2` per cell and takes `-0.5 / scale^2` and the additive
# `-log(scale) - 0.5 * log(2pi)` once per call; the plate's own loop still runs
# whenever a shared guard holds (`scale == 0`) or the factored total would lose
# what the cells keep. The reference for every value and derivative below is
# the literal program (`simplify = false`), which keeps the plate's own loop.
module SummedPlateFactorsTest

using ReactiveKernels, Enzyme, DifferentiationInterface, Test
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    normal, laplace, cauchy, logistic, student_t
include("type_stability_audit.jl")
using .TypeStabilityAudit: findings, ad_call_findings

const RK = ReactiveKernels

@kernel shared_normal(y::Vector{Float64}, mu::Vector{Float64}, sigma::Float64) = begin
    lp = plate(y, mu) do yi, mi
        normal(mi, sigma).logpdf(yi)
    end
    total::Float64 = sum(lp)
    return total
end
@kernel scalar_location_normal(y::Vector{Float64}, mu::Float64, sigma::Float64) = begin
    lp = plate(y) do yi
        normal(mu, sigma).logpdf(yi)
    end
    total::Float64 = sum(lp)
    return total
end
@kernel shared_laplace(y::Vector{Float64}, mu::Vector{Float64}, sigma::Float64) = begin
    lp = plate(y, mu) do yi, mi
        laplace(mi, sigma).logpdf(yi)
    end
    total::Float64 = sum(lp)
    return total
end
@kernel shared_cauchy(y::Vector{Float64}, mu::Vector{Float64}, sigma::Float64) = begin
    lp = plate(y, mu) do yi, mi
        cauchy(mi, sigma).logpdf(yi)
    end
    total::Float64 = sum(lp)
    return total
end
@kernel shared_logistic(y::Vector{Float64}, mu::Vector{Float64}, sigma::Float64) = begin
    lp = plate(y, mu) do yi, mi
        logistic(mi, sigma).logpdf(yi)
    end
    total::Float64 = sum(lp)
    return total
end
@kernel shared_student(y::Vector{Float64}, mu::Vector{Float64}, sigma::Float64) = begin
    lp = plate(y, mu) do yi, mi
        student_t(4.0, mi, sigma).logpdf(yi)
    end
    total::Float64 = sum(lp)
    return total
end
# An authored cell: the same algebra without a law.
@kernel authored_normal(y::Vector{Float64}, mu::Vector{Float64}, sigma::Float64) = begin
    lp = plate(y, mu) do yi, mi
        -0.5 * ((yi - mi) / sigma)^2 - log(sigma)
    end
    total::Float64 = sum(lp)
    return total
end
# Nothing to take out: a literal factor of the cell's own square, and a scale
# that is the cell's own.
@kernel literal_factor(y::Vector{Float64}, mu::Vector{Float64}, sigma::Float64) = begin
    lp = plate(y, mu) do yi, mi
        -0.5 * (yi - mi)^2
    end
    total::Float64 = sum(lp) / sigma
    return total
end
@kernel cell_scale(y::Vector{Float64}, mu::Vector{Float64}, sigma::Vector{Float64}) = begin
    lp = plate(y, mu, sigma) do yi, mi, si
        normal(mi, si).logpdf(yi)
    end
    total::Float64 = sum(lp)
    return total
end
# A location gathered through a sorted index: index runs group the factored
# cells as they group the plate's own.
@kernel gathered_normal(q::Vector{Float64}, idx::Vector{Int}, y::Vector{Float64}, sigma::Float64) = begin
    location = ReactiveKernels._readonly_gather(q, idx)
    lp = plate(y, location) do yi, mi
        normal(mi, sigma).logpdf(yi)
    end
    total::Float64 = sum(lp)
    return total
end

_code(k) = string(code_expr(k))
_factored(k) = occursin("#factored", _code(k))
function _loops(ex)
    ex isa Expr || return 0
    (ex.head === :while) + sum(_loops, ex.args; init = 0)
end
# The ordinary reverse gradient in `mu` and `sigma`.
function _gradient(k, y, mu, sigma)
    dmu = zero(mu)
    dsigma = Enzyme.autodiff(Enzyme.Reverse, Enzyme.Const(k), Enzyme.Active, Enzyme.Const(y),
        Enzyme.Duplicated(copy(mu), dmu), Enzyme.Active(sigma))[1][3]
    (dmu, dsigma)
end
_agrees(a, b; rtol) = isequal(a, b) || isapprox(a, b; rtol)

const Y = [0.3, -0.2, 1.1, 2.5, -1.0, 0.7, 0.0]
const MU = [0.1, 0.0, 0.9, 2.0, -0.5, 0.7, -0.3]

@testset "location-scale laws with a shared scale" begin
    for kernel in (shared_normal, shared_laplace, shared_cauchy, shared_logistic,
                   shared_student, authored_normal)
        k = prepare(kernel)
        literal = prepare(kernel; simplify = false)
        @test _factored(k)
        @test !_factored(literal)
        for sigma in (0.7, 2.0, 1e-3, 3e5)
            @test k(Y, MU, sigma) ≈ literal(Y, MU, sigma) rtol = 1e-14
            g, gl = _gradient(k, Y, MU, sigma), _gradient(literal, Y, MU, sigma)
            @test g[1] ≈ gl[1] rtol = 1e-13
            @test g[2] ≈ gl[2] rtol = 1e-13
        end
        # Extreme scales and data: the plate's own loop decides each of these,
        # and every finite value and derivative is the literal one.
        for (y, mu, sigma) in ((Y, MU, 0.0), (Y, MU, Inf), (Y, MU, 1e-160), (Y, MU, 1e160),
                               ([1.0, 1.0], [1.0, 1.0], 0.0), ([NaN, 1.0], [0.0, 0.0], 0.7),
                               ([Inf, 1.0], [0.0, 0.0], 0.7), ([1e160, 1.0], [0.0, 0.0], 1e10))
            v, vl = k(y, mu, sigma), literal(y, mu, sigma)
            @test _agrees(v, vl; rtol = 1e-14)
            if isfinite(vl)
                g, gl = _gradient(k, y, mu, sigma), _gradient(literal, y, mu, sigma)
                @test all(_agrees.(g[1], gl[1]; rtol = 1e-13))
                @test _agrees(g[2], gl[2]; rtol = 1e-13)
            end
        end
        @test k(Float64[], Float64[], 0.7) == literal(Float64[], Float64[], 0.7) == 0.0
    end
    # The Normal plate's cells read only their residual: its square is all a
    # cell adds, and the scale enters once.
    code = _code(prepare(shared_normal))
    @test occursin(r"factored_term\" = \(var\"[^\"]*factored_read_1\" - var\"[^\"]*factored_read_2\"\) \^ 2", code)
end

@testset "direct reads and broadcast extrusion both factor" begin
    k = prepare(scalar_location_normal)
    literal = prepare(scalar_location_normal; simplify = false)
    @test _factored(k)
    @test _loops(k.ast) == 2 * _loops(literal.ast)
    for y in (Y, Float64[])
        @test k(y, 0.2, 0.7) ≈ literal(y, 0.2, 0.7) rtol = 1e-14
        g = Enzyme.autodiff(Enzyme.Reverse, Enzyme.Const(k), Enzyme.Active,
            Enzyme.Const(y), Enzyme.Active(0.2), Enzyme.Active(0.7))[1]
        gl = Enzyme.autodiff(Enzyme.Reverse, Enzyme.Const(literal), Enzyme.Active,
            Enzyme.Const(y), Enzyme.Active(0.2), Enzyme.Active(0.7))[1]
        @test g[2] ≈ gl[2] rtol = 1e-13
        @test g[3] ≈ gl[3] rtol = 1e-13
    end
    k, literal = prepare(shared_normal), prepare(shared_normal; simplify = false)
    @test _loops(k.ast) == 2 * _loops(literal.ast)
    for mu in (MU, [0.2])
        @test k(Y, mu, 0.7) ≈ literal(Y, mu, 0.7) rtol = 1e-14
        g, gl = _gradient(k, Y, mu, 0.7), _gradient(literal, Y, mu, 0.7)
        @test g[1] ≈ gl[1] rtol = 1e-13
        @test g[2] ≈ gl[2] rtol = 1e-13
    end
end

@testset "nothing to take out keeps the plate's loop" begin
    for kernel in (literal_factor, cell_scale)
        k = prepare(kernel)
        @test !_factored(k)
    end
    y, mu, s = [0.3, -0.2, 1.1], [0.1, 0.0, 0.9], [0.5, 1.0, 2.0]
    @test prepare(cell_scale)(y, mu, s) ≈ prepare(cell_scale; simplify = false)(y, mu, s) rtol = 1e-14
end

@testset "index runs group the factored cells" begin
    q = [0.1, -0.4, 0.3]
    idx = repeat([1, 2, 3], inner = 16)
    y = [sin(i / 3) / 2 for i in eachindex(idx)]
    # Bound least-squares statistics (`bound_statistics.jl`) would sum this
    # plate's squared residuals without a loop; the factored loop is the one
    # that runs without them.
    k = prepare(gathered_normal; bound = (; idx, y), bound_statistics = false)
    literal = prepare(gathered_normal; bound = (; idx, y), simplify = false)
    @test _factored(k)
    @test occursin("run_partial", _code(k))
    for sigma in (0.4, 1.3)
        @test k(q, sigma) ≈ literal(q, sigma) rtol = 1e-14
        dq, dql = zero(q), zero(q)
        ds = Enzyme.autodiff(Enzyme.Reverse, Enzyme.Const(k), Enzyme.Active,
            Enzyme.Duplicated(copy(q), dq), Enzyme.Active(sigma))[1][2]
        dsl = Enzyme.autodiff(Enzyme.Reverse, Enzyme.Const(literal), Enzyme.Active,
            Enzyme.Duplicated(copy(q), dql), Enzyme.Active(sigma))[1][2]
        @test dq ≈ dql rtol = 1e-12
        @test ds ≈ dsl rtol = 1e-12
    end
    # Both passes leave a lowered body as it is.
    @test RK._lower_summed_plate_factors_native(k.ast, RK._kernel_ops(k)) == k.ast
    @test RK._lower_index_runs_native(k.ast, RK._kernel_ops(k)) == k.ast
end

@testset "data length does not grow the program" begin
    loops = map((3, 3000)) do n
        y, mu = collect(range(-1, 1; length = n)), zeros(n)
        k = prepare(shared_normal; bound = (; y, mu))
        @test k(0.7) ≈ prepare(shared_normal; bound = (; y, mu), simplify = false)(0.7) rtol = 1e-13
        _loops(k.ast)
    end
    @test loops[1] == loops[2]
    # The factored loop beside the plate's own.
    @test loops[1] == 2 * _loops(prepare(shared_normal; bound = (; y = Y, mu = MU), simplify = false).ast)
end

@testset "concrete program and ordinary reverse" begin
    k = prepare(shared_normal)
    @test isempty(findings(k, (Vector{Float64}, Vector{Float64}, Float64)))
    ad = prepare_ad(k, AutoEnzyme(; mode = Enzyme.Reverse), Y, MU, 0.7;
        active = (:mu, :sigma), native_forward_batch = false)
    @test isempty(ad_call_findings(ad, Y, MU, 0.7))
end

@testset "the opt-in plate reverse keeps the plate's loop" begin
    k = RK.with_plate_reverse() do
        prepare(shared_normal; bound = (; y = Y))
    end
    @test !_factored(k)
end

end
