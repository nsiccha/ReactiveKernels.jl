# A half-Student prior on a positive vector (snag `measured-the-rem-97a93f3f`):
# ReactiveKernels' stable rewrites compute its kernel from the unconstrained
# coordinate `u` of `w = exp(u)`, with the literal program's values and
# gradients and a non-allocating gradient.
using DifferentiationInterface: AutoEnzyme
using Enzyme: Enzyme
using ReactiveKernels: ReactiveKernels
using ReactiveKernelsPPL
using Test

const _PSP_MODEL = :(begin
    w[axes(y, 1)] .~ restricted.(StudentT.(nu, 0, 1), 0, Inf)
    y .~ Normal.(0, w)
end)

function _psp_check(nu, nonallocating)
    data = Dict(:y => [0.3 * sin(i) + 0.1 * cos(3i) for i in 1:40], :nu => nu)
    plan = lower_rkppl(_PSP_MODEL, keys(data); conditioned = (:y,))
    bound = bind_data(plan, data)
    built = build_kernel(bound)
    m = built.layout.total
    u0 = [0.4 * cos(i) for i in 1:m]
    prep() = prepare_sampler(built, bound, u0; backend = AutoEnzyme(; mode = Enzyme.Reverse),
                             nonallocating, warm = false)
    q = prep()
    literal = ReactiveKernels._with_simplify(prep, false)
    code = string(ReactiveKernels.code_expr(q.kernel))
    @test occursin("_rk_at_exp_std(", code)
    @test !occursin("_student_log_kernel(", code)
    tails = copy(u0)
    tails[1], tails[end] = -20.0, 20.0
    for u in (u0, zeros(m), tails)
        g, gl = similar(u), similar(u)
        v = first(sampler_value_and_gradient!(q, g, u))
        vl = first(sampler_value_and_gradient!(literal, gl, u))
        @test v ≈ vl rtol = 1e-13
        @test g ≈ gl rtol = 1e-12 atol = 1e-12
    end
    g = similar(u0)
    _psp_bytes(q, g, u0)
    nonallocating && @test _psp_bytes(q, g, u0) == 0
end
# Measured behind a function barrier: the sampler's type is the argument's.
_psp_bytes(q, g, u) = @allocated sampler_value_and_gradient!(q, g, u)

@testset "a half-Student prior on positive scales reads their logarithms" begin
    for nu in (1.0, 4.0), nonallocating in (true, false)
        _psp_check(nu, nonallocating)
    end
end
