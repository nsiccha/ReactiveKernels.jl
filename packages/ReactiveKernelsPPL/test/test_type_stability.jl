# Prepared RKPPL samplers are concretely typed (RK principle review
# `ReactiveKernels:snag.joint-nlme-enzym-f8d3cb5f/decisions/2026-10-12T20-09-11-062-114x8kz`;
# audit and warning-only client check, user decision `…/2026-10-12T20-13-15-811-1hui2qi`).
# Programs from the canonical corpus (`test/corpus`) cover the surface's
# lowerings with concretely typed data: the sampler's native kernel and its
# gradient call hold no non-concrete value and no run-time dispatched call, and
# preparation emits no type-stability warning.
using ReactiveKernels, ReactiveKernelsPPL, DifferentiationInterface, Enzyme
using Test

module RKPPLTypeStability
import ReactiveKernels
include(joinpath(pkgdir(ReactiveKernels), "test", "type_stability_audit.jl"))
end
using .RKPPLTypeStability.TypeStabilityAudit: findings, ad_call_findings

const _TS_N = 12
const _TS_X = [0.1i - 0.6 for i in 1:_TS_N]
const _TS_G = [mod1(i, 3) for i in 1:_TS_N]

# (corpus program, data)
const _TS_PROGRAMS = (
    ("01_gaussian", (; y = [0.4 + 0.8x + 0.2sin(3x) for x in _TS_X], x = _TS_X)),
    ("02_bernoulli_logit", (; y = [isodd(i) ? 1 : 0 for i in 1:_TS_N], x = _TS_X)),
    ("03_poisson_exp", (; y = [mod(3i, 5) for i in 1:_TS_N], x = _TS_X)),
    ("09_plate_loop", (; y = [0.4 + 0.8x for x in _TS_X], x = _TS_X)),
    ("10_levels_prior", (; y = [0.3sin(i) + 0.1i for i in 1:_TS_N], g = _TS_G,
                          o = [0.05i for i in 1:_TS_N])),
    ("20_ordered_logistic", (; y = [mod1(i, 3) for i in 1:_TS_N], x = _TS_X)),
    ("23_categorical_simplex", (; y = [mod1(2i, 3) for i in 1:_TS_N])),
    ("46_matrix_gaussian", (; y = [0.2 + x for x in _TS_X], x1 = _TS_X,
                             x2 = [cos(0.7i) for i in 1:_TS_N])),
    ("54_mixture", (; y = [0.4 + 0.8x + 0.2sin(3x) for x in _TS_X], x = _TS_X)),
    ("72_centered_levels", (; y = [0.3sin(i) for i in 1:_TS_N], g = _TS_G)),
    ("98_array_rows_mvnormal", (; y = [0.1i for i in 1:_TS_N], x = _TS_X, g = _TS_G)),
)

function _ts_report(name, found)
    isempty(found) && return true
    println(stderr, "type-stability findings for ", name, ":")
    foreach(f -> println(stderr, "  ", f), found)
    false
end

@testset "RKPPL samplers are concretely typed ($name)" for (name, data) in _TS_PROGRAMS
    program = Meta.parse(read(joinpath(@__DIR__, "corpus", name * ".jl"), String))
    columns = Dict{Symbol,AbstractVector}(k => collect(v) for (k, v) in pairs(data))
    bound = bind_data(lower_rkppl(program, keys(columns); conditioned = keys(columns)),
                      columns)
    built = build_kernel(bound)
    u = [0.05 * sin(i) for i in 1:built.layout.total]
    q = @test_logs min_level = Base.CoreLogging.Warn prepare_sampler(
        built, bound, u; backend = AutoEnzyme(; mode = Enzyme.Reverse))
    @test isfinite(q(u))
    @test _ts_report("$name (native kernel)",
                     findings(q.kernel, (Vector{Float64},)))
    @test _ts_report("$name (gradient call)", ad_call_findings(q.ad, u))
end
