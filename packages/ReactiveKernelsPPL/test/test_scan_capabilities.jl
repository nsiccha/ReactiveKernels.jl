using ReactiveKernels
using ReactiveKernelsPPL
using Distributions
using DifferentiationInterface
using Enzyme
using Test

function _scan_cap_sampler(program, data)
    plan = bind_data(lower_rkppl(program, Tuple(keys(data));
        conditioned = (:y,)), data)
    built = build_kernel(plan)
    u = [0.2sin(i) for i in 1:built.layout.total]
    sampler = prepare_sampler(built, plan, u;
        backend = AutoEnzyme(; mode = Enzyme.Reverse))
    return (; plan, built, sampler, u)
end

function _scan_cap_check(f, oracle)
    grad = similar(f.u)
    val, _ = sampler_value_and_gradient!(f.sampler, grad, f.u)
    @test val ≈ oracle(f.u) rtol=1e-11
    h = cbrt(eps(Float64))
    fd = map(eachindex(f.u)) do i
        up, dn = copy(f.u), copy(f.u)
        up[i] += h
        dn[i] -= h
        (oracle(up) - oracle(dn)) / (2h)
    end
    @test grad ≈ fd rtol=1e-5 atol=1e-7
    return grad
end

function _scan_cap_deterministic(n, sampled)
        seed = sampled ? :(h[1] ~ Normal(0, 1)) : :(h[1] = phi)
        program = quote
            phi ~ Normal(0, 1)
            @scan begin
                $seed
                for t in 2:T
                    h[t] = phi * h[t - 1]
                end
            end
            y .~ Normal.(h, 1.0)
        end
        y = [0.3sin(t) for t in 1:n]
        f = _scan_cap_sampler(program, Dict{Symbol,Any}(:y => y))
        function oracle(u)
            phi = u[1]
            h = sampled ? u[2] : phi
            lp = logpdf(Normal(), phi) + (sampled ? logpdf(Normal(), h) : 0.0)
            for t in eachindex(y)
                lp += logpdf(Normal(h, 1), y[t])
                h *= phi
            end
            return lp
        end
        return f, oracle
end

@testset "scan capabilities: deterministic and zero-step recurrences" begin
    for n in (1, 4, 8), sampled in (false, true)
        f, oracle = _scan_cap_deterministic(n, sampled)
        @test f.built.layout.total == 1 + sampled
        _scan_cap_check(f, oracle)
    end
end

# Response spelling, response data and oracle law for each location family.
_scan_cap_logistic(m) = 1 / (1 + exp(-m))
const _SCAN_CAP_RESPONSES = Dict(
    :normal => (:(y .~ Normal.(mu, 1.0)), t -> 0.3sin(t), m -> Normal(m, 1)),
    :poisson => (:(y .~ Poisson.(exp.(mu))), t -> t % 3, m -> Poisson(exp(m))),
    :bernoulli => (:(y .~ Bernoulli.(logistic.(mu))), t -> t % 2,
        m -> Bernoulli(_scan_cap_logistic(m))),
    :exponential => (:(y .~ Exponential.(exp.(mu))), t -> 0.4 + 0.2t,
        m -> Exponential(exp(m))),
    :inverse_gaussian => (:(y .~ InverseGaussian.(exp.(mu))), t -> 0.4 + 0.2t,
        m -> InverseGaussian(exp(m), 1)),
    :gamma => (:(y .~ Gamma.(2.0, exp.(mu) ./ 2.0)), t -> 0.4 + 0.2t,
        m -> Gamma(2, exp(m) / 2)),
    :negative_binomial => (:(y .~ NegativeBinomial2.(exp.(mu), 2.0)), t -> t % 3,
        m -> NegativeBinomial(2, 2 / (2 + exp(m)))),
    :beta => (:(y .~ Beta.(logistic.(mu) .* 5.0, (1 .- logistic.(mu)) .* 5.0)),
        t -> 0.1 + 0.15t,
        m -> Beta(5 * _scan_cap_logistic(m), 5 * (1 - _scan_cap_logistic(m)))))

# `steps` supplies the trajectory length `T`; otherwise it is the response's
# rows. A one-entry trajectory broadcasts over every observation.
function _scan_cap_values(n, form; response = :normal, steps = nothing)
    expression = Dict(
        :bare => :(h), :literal => :(2.0 .* h), :data => :(x .* h),
        :computed => :(c .* h), :coefficient => :(b .* h),
        :negative => :(-b .* h), :nested => :(b .* (h .+ x)),
        :product => :(h .* h), :alias => :(v))
    likelihood, ydata, law = _SCAN_CAP_RESPONSES[response]
    program = quote
        b ~ Cauchy(0, 1)
        phi ~ Normal(0, 1)
        c = b * b
        @scan begin
            h[1] ~ Normal(0, 1)
            for t in 2:T
                e ~ Normal(0, 1)
                h[t] = phi * h[t - 1] + e
            end
        end
        v = 2.0 .* h
        mu = $(expression[form])
        $likelihood
    end
    y = [ydata(t) for t in 1:n]
    x = [0.2cos(t) for t in 1:n]
    data = Dict{Symbol,Any}(:x => x, :y => y)
    steps === nothing || (data[:T] = steps)
    f = _scan_cap_sampler(program, data)
    parameter(u, name) = u[only(e for e in f.built.layout.entries
        if e.name === name).offset]
    zentry = only(e for e in f.built.layout.entries if e.kind === :scan)
    function oracle(u)
        b, phi = parameter(u, :b), parameter(u, :phi)
        z = u[zentry.offset:(zentry.offset + zentry.size - 1)]
        trajectory = accumulate((h, e) -> phi * h + e, z)
        lp = logpdf(Cauchy(), b) + logpdf(Normal(), phi) + sum(logpdf.(Normal(), z))
        for t in eachindex(y)
            h = trajectory[length(trajectory) == 1 ? 1 : t]
            mu = form === :literal || form === :alias ? 2h :
                form === :data ? x[t] * h : form === :computed ? b^2 * h :
                form === :coefficient ? b * h : form === :negative ? -b * h :
                form === :nested ? b * (h + x[t]) : form === :product ? h^2 : h
            lp += logpdf(law(mu), y[t])
        end
        return lp
    end
    return f, oracle
end

@testset "scan capabilities: ordinary trajectory values" begin
    for form in (:bare, :literal, :data, :computed, :coefficient, :negative,
            :nested, :product, :alias), n in (1, 4)
        f, oracle = _scan_cap_values(n, form)
        _scan_cap_check(f, oracle)
    end
    f, oracle = _scan_cap_values(4, :bare; response = :poisson)
    _scan_cap_check(f, oracle)
end

@testset "scan capabilities: trajectory locations of every response family" begin
    # The trajectory is the location vector itself: no linear predictor
    # carries it, under any family or link.
    for response in (:bernoulli, :exponential, :inverse_gaussian, :gamma,
            :negative_binomial, :beta)
        f, oracle = _scan_cap_values(4, :bare; response)
        _scan_cap_check(f, oracle)
    end
    # A one-entry trajectory broadcasts over four observations, as in Julia.
    for response in (:normal, :poisson, :bernoulli)
        f, oracle = _scan_cap_values(4, :bare; response, steps = 1)
        @test f.built.layout.total == 3
        _scan_cap_check(f, oracle)
    end
end
