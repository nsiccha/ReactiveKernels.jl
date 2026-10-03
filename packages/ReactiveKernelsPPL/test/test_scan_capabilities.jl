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

function _scan_cap_values(n, form; response = :normal)
    expression = Dict(
        :bare => :(h), :literal => :(2.0 .* h), :data => :(x .* h),
        :computed => :(c .* h), :coefficient => :(b .* h),
        :negative => :(-b .* h), :nested => :(b .* (h .+ x)),
        :product => :(h .* h), :alias => :(v))
    likelihood = response === :normal ? :(y .~ Normal.(mu, 1.0)) :
        :(y .~ Poisson.(exp.(mu)))
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
    y = response === :normal ? [0.3sin(t) for t in 1:n] : [t % 3 for t in 1:n]
    x = [0.2cos(t) for t in 1:n]
    f = _scan_cap_sampler(program, Dict{Symbol,Any}(:x => x, :y => y))
    parameter(u, name) = u[only(e for e in f.built.layout.entries
        if e.name === name).offset]
    zentry = only(e for e in f.built.layout.entries if e.kind === :scan)
    function oracle(u)
        b, phi = parameter(u, :b), parameter(u, :phi)
        z = u[zentry.offset:(zentry.offset + zentry.size - 1)]
        h = z[1]
        lp = logpdf(Cauchy(), b) + logpdf(Normal(), phi) + sum(logpdf.(Normal(), z))
        for t in eachindex(y)
            t == 1 || (h = phi * h + z[t])
            mu = form === :literal || form === :alias ? 2h :
                form === :data ? x[t] * h : form === :computed ? b^2 * h :
                form === :coefficient ? b * h : form === :negative ? -b * h :
                form === :nested ? b * (h + x[t]) : form === :product ? h^2 : h
            dist = response === :normal ? Normal(mu, 1) : Poisson(exp(mu))
            lp += logpdf(dist, y[t])
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
