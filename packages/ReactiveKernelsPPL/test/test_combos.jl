# Computed coefficients composed with a mixture likelihood lower, build, and
# differentiate (Enzyme vs finite differences). `_check_gradient` comes from test_generator.jl,
# included earlier by runtests.jl.)
using ReactiveKernelsPPL
using Test

function _cb_query(prog::Expr, data)
    cols = Dict{Symbol,AbstractVector}(k => collect(v) for (k, v) in pairs(data))
    bound = bind_data(lower_rkppl(prog, keys(cols); conditioned = keys(cols)), cols)
    built = build_kernel(bound)
    return bound, built, prepare_query(built, bound, :sampler), built.layout
end

_cb_probe(n) = [0.2 * sin(1.3i) + 0.02i for i in 1:n]

const _CB_N = 12
const _CB_X = [0.1i - 0.6 for i in 1:_CB_N]
const _CB_X2 = [cos(0.7i) for i in 1:_CB_N]
const _CB_Y = [0.4 + 0.8x + 0.2sin(3x) for x in _CB_X]
const _CB_G = [isodd(i) ? 1 : 2 for i in 1:_CB_N]

const _CB_ADMITTED = (
    ("computed coefficients x mixture", quote
        a1 ~ Normal(0, 1)
        raw1 ~ Normal(0, 1)
        lambda1 ~ HalfCauchy(1)
        tau1 ~ HalfCauchy(1)
        b1 = raw1 * lambda1 * tau1
        raw2 ~ Normal(0, 1)
        lambda2 ~ HalfCauchy(0.5)
        tau2 ~ HalfCauchy(0.25)
        b2 = raw2 * lambda2 * tau2
        mu1 = a1 .+ b1 .* x1 .+ b2 .* x2
        mu2 ~ Normal(0.0, 5.0)
        sigma ~ Exponential(1.0)
        y .~ MixtureModel.(vcat.(Normal.(mu1, sigma), Normal.(mu2, sigma)), Ref([0.3, 0.7]))
    end, (; y = _CB_Y, x1 = _CB_X, x2 = _CB_X2), 9),
)

@testset "composition matrix: admitted ($label)" for (label, prog, data, n) in _CB_ADMITTED
    bound, built, kern, lay = _cb_query(prog, data)
    @test lay.total == n
    @test isfinite(Base.invokelatest(kern, _cb_probe(n)))
    _check_gradient(built.spec, bound, _cb_probe(n))
end
