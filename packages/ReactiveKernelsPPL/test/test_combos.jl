# Residual composition matrix (matrix-b): the smooth / shrinkage /
# measurement-error / mixture atoms, each tested alone elsewhere, composed
# in one program. Admitted compositions lower, build, and differentiate
# (Enzyme vs finite differences); the compositions the surface does not
# admit fail closed with their guidance (documented boundaries, not
# silent drops). (`_check_gradient` comes from test_generator.jl,
# included earlier by runtests.jl.)
using ReactiveKernelsPPL
using Test

function _cb_query(prog::Expr, data)
    cols = Dict{Symbol,AbstractVector}(k => collect(v) for (k, v) in pairs(data))
    bound = bind_data(lower_rkppl(prog, keys(cols)), cols)
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
    ("horseshoe x mixture", quote
        a1 ~ Normal(0, 1)
        b1 ~ Horseshoe()
        b2 ~ Horseshoe(local_scale = 0.5, global_scale = 0.25)
        mu1 = a1 .+ b1 .* x1 .+ b2 .* x2
        mu2 ~ Normal(0.0, 5.0)
        sigma ~ Exponential(1.0)
        y .~ MixtureModel.(vcat.(Normal.(mu1, sigma), Normal.(mu2, sigma)), Ref([0.3, 0.7]))
    end, (; y = _CB_Y, x1 = _CB_X, x2 = _CB_X2), 9),
    ("hsgp x mixture", quote
        a1 ~ Normal(0, 1)
        hsgp_basis(:h, x; k = 6)
        mu1 = a1 .+ hsgp(:h)
        mu2 ~ Normal(0.0, 5.0)
        sigma ~ Exponential(1.0)
        y .~ MixtureModel.(vcat.(Normal.(mu1, sigma), Normal.(mu2, sigma)), Ref([0.3, 0.7]))
    end, (; y = _CB_Y, x = _CB_X), 11),
    ("hsgp x me", quote
        a ~ Normal(0, 1)
        hsgp_basis(:h, x; k = 6)
        b ~ Normal(0, 2)
        sigma ~ Exponential(1)
        mu = a .+ hsgp(:h) .+ b .* z_true
        @plate for i in eachindex(z_obs)
            z_true[i] ~ Normal(0.0, 1.0)
        end
        y .~ Normal.(mu, sigma)
        z_obs .~ Normal.(z_true, 0.5)
    end, (; y = _CB_Y, x = _CB_X, z_obs = _CB_X2), 23),
    ("grouped hsgp x student_t", quote
        a ~ Normal(0, 1)
        c0 ~ Normal(0, 1)
        hsgp_basis(:h, x; k = 6, by = grp, length_scale = 1 + (1 | grp),
            sd = (1 | grp))
        r ~ varying_effect(grp, [1])
        mu = a .+ hsgp(:h) .+ r
        ls = c0
        y .~ StudentT.(4.0, mu, exp.(ls))
    end, (; y = _CB_Y, x = _CB_X, grp = _CB_G), 24),
)

@testset "composition matrix: admitted ($label)" for (label, prog, data, n) in _CB_ADMITTED
    bound, built, kern, lay = _cb_query(prog, data)
    @test lay.total == n
    @test isfinite(Base.invokelatest(kern, _cb_probe(n)))
    _check_gradient(built.spec, bound, _cb_probe(n))
end

@testset "composition matrix: fail-closed boundaries" begin
    # capability: compose measurement-error, DAR, horseshoe and smooth
    # values (P8 1cmodra; todo `1nb43fj`).
    gap(prog, data) = @test_broken (lower_rkppl(prog, keys(data)); true)
    # Admitted: ordinary coefficient reads now compose a measurement-error
    # latent with a direct DAR summand.
    admit(prog, data) = @test (lower_rkppl(prog, keys(data)); true)
    admit(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 2)
        beta ~ truncated(Normal(0.5, 0.2), 0, 1)
        sigmad ~ HalfNormal(0.2)
        sigma ~ Exponential(1)
        mu = a .+ b .* x_true .+ dar(beta, sigmad)
        @plate for i in eachindex(x_obs)
            x_true[i] ~ Normal(0.5, 1.5)
        end
        y .~ Normal.(mu, sigma)
        x_obs .~ Normal.(x_true, 0.5)
    end, (; y = _CB_Y, x_obs = _CB_X))
    # horseshoe x hsgp: the horseshoe slice covers intercept/continuous
    # coefficients only.
    gap(quote
        a ~ Normal(0, 1)
        b1 ~ Horseshoe()
        hsgp_basis(:h, x; k = 6)
        eta = a .+ b1 .* x2 .+ hsgp(:h)
        cnt .~ Poisson.(exp.(eta))
    end, (; cnt = [round(Int, 2 + sin(i)) for i in 1:_CB_N], x = _CB_X,
        x2 = _CB_X2))
end
