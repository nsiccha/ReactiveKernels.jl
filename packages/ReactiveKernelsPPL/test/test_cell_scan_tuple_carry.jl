# A cell `@scan` carrying two arrays (or one array at lag depth 2) steps a
# NamedTuple carry: its lags read `carry.s_lag1`. A step that passes such a
# lag to a module function (`add(s[t - 1], z[t - 1])`, or `+` bound to a module
# function as BRM's RK emission binds it) reads the step's local `carry`, as in
# Julia, not a module path. Lowering refused this before RKPPL 0.1.141 with
# "qualifies a call with `:carry`, which is not a module visible in …"
# (snag `a-brm-plate-cell-bf665a16`). Lambda arguments, lambda locals and
# generator iteration names are lexical names of their body.
module CellScanTupleCarryTests
using Test, ReactiveKernels, ReactiveKernelsPPL, Distributions
import DifferentiationInterface as DI
import Enzyme

add(a, b) = a + b

function build(program, data)
    plan = lower_rkppl(program, Tuple(keys(data)); mod = @__MODULE__, conditioned = (:y,))
    bound = bind_data(plan, data)
    return (; plan, bound, built = build_kernel(bound))
end

raw(f, u, name) = begin
    e = only(e for e in f.built.layout.entries if e.name === name)
    u[e.offset:(e.offset + e.size - 1)]
end

# Value and native Enzyme Reverse gradient against an independent oracle,
# plus central finite differences of the oracle.
function check(f, oracle)
    n = f.built.layout.total
    initial = [0.13sin(i) for i in 1:n]
    sampler = prepare_sampler(f.built, f.bound, initial;
        backend = DI.AutoEnzyme(; mode = Enzyme.Reverse))
    for shift in (0.0, 0.11)
        u = initial .+ shift
        expected = oracle(u)
        value, gradient = sampler_value_and_gradient!(sampler, similar(u), u)
        @test value ≈ expected rtol = 2e-11
        h = cbrt(eps(Float64))
        fd = map(eachindex(u)) do i
            up, dn = copy(u), copy(u)
            up[i] += h
            dn[i] -= h
            (oracle(up) - oracle(dn)) / (2h)
        end
        @test gradient ≈ fd rtol = 2e-5 atol = 2e-7
    end
end

const DATA = (; loc = [[0.0, 0.3], [0.1]], n = [3, 2], m = [0.2, -0.1],
    y = [[0.1, 0.3, 0.2], [-0.4, 0.1]])

@testset "cell scan: two carried arrays step through a module function" begin
    # BRM's reported cell: a cell latent and a two-array scan reading it.
    program = quote
        sigma ~ Exponential(1)
        @plate for i in eachindex(y)
            z[i] .~ Normal.(loc[i], 1)
            zc = z[i]
            @scan begin
                s[1] = 0.0
                r[1] = 0.0
                for t in 2:n[i]
                    s[t] = add(s[t - 1], zc[t - 1])
                    r[t] = add(r[t - 1], 0.5 * s[t])
                end
            end
            y[i] .~ Normal.(m[i] .+ s .+ r, sigma)
        end
    end
    f = build(program, DATA)
    @test coordinate_names(f.built.layout) ==
        [:sigma, Symbol("z.1.1"), Symbol("z.1.2"), Symbol("z.2.1")]
    function oracle(u)
        log_sigma = only(raw(f, u, :sigma))
        sigma = exp(log_sigma)
        z = (u[2:3], u[4:4])
        lp = logpdf(Exponential(1), sigma) + log_sigma
        for i in 1:2
            lp += sum(logpdf.(Normal.(DATA.loc[i], 1), z[i]))
            s, r = [0.0], [0.0]
            for t in 2:DATA.n[i]
                push!(s, s[t - 1] + z[i][t - 1])
                push!(r, r[t - 1] + 0.5 * s[t])
            end
            lp += sum(logpdf.(Normal.(DATA.m[i] .+ s .+ r, sigma), DATA.y[i]))
        end
        return lp
    end
    check(f, oracle)

    # The same cell without the latent, reading a parameter.
    shared = quote
        sigma ~ Exponential(1)
        b ~ Normal(0, 1)
        @plate for i in eachindex(y)
            @scan begin
                s[1] = 0.0
                r[1] = 0.0
                for t in 2:n[i]
                    s[t] = add(s[t - 1], b)
                    r[t] = add(r[t - 1], 0.5 * s[t])
                end
            end
            y[i] .~ Normal.(m[i] .+ s .+ r, sigma)
        end
    end
    g = build(shared, DATA)
    function oracle_shared(u)
        log_sigma, b = only(raw(g, u, :sigma)), only(raw(g, u, :b))
        sigma = exp(log_sigma)
        lp = logpdf(Exponential(1), sigma) + log_sigma + logpdf(Normal(), b)
        for i in 1:2
            s, r = [0.0], [0.0]
            for t in 2:DATA.n[i]
                push!(s, s[t - 1] + b)
                push!(r, r[t - 1] + 0.5 * s[t])
            end
            lp += sum(logpdf.(Normal.(DATA.m[i] .+ s .+ r, sigma), DATA.y[i]))
        end
        return lp
    end
    check(g, oracle_shared)
end

@testset "cell scan: a lag-2 carry steps through a module function" begin
    data = (; n = [4, 3], y = [[0.1, 0.3, 0.2, 0.5], [-0.4, 0.1, 0.0]])
    program = quote
        sigma ~ Exponential(1)
        b ~ Normal(0, 1)
        @plate for i in eachindex(y)
            @scan begin
                s[1] = 0.0
                s[2] = b
                for t in 3:n[i]
                    s[t] = add(s[t - 2], b * s[t - 1])
                end
            end
            y[i] .~ Normal.(s, sigma)
        end
    end
    f = build(program, data)
    function oracle(u)
        log_sigma, b = only(raw(f, u, :sigma)), only(raw(f, u, :b))
        sigma = exp(log_sigma)
        lp = logpdf(Exponential(1), sigma) + log_sigma + logpdf(Normal(), b)
        for i in 1:2
            s = [0.0, b]
            for t in 3:data.n[i]
                push!(s, s[t - 2] + b * s[t - 1])
            end
            lp += sum(logpdf.(Normal.(s, sigma), data.y[i]))
        end
        return lp
    end
    check(f, oracle)
end

@testset "lambda and generator names are lexical in call resolution" begin
    resolve(ex) = ReactiveKernelsPPL._resolve_module_calls(ex, @__MODULE__,
        Set([:xs]), "test")
    # A lambda argument's property passed to a module function stays a read
    # of that argument; the function resolves in the module.
    lambda = resolve(:(map(c -> add(c.a, 1), xs)))
    body = last(lambda.args[2].args[2].args)
    @test body.args[1] == GlobalRef(@__MODULE__, :add)
    @test body.args[2] == :(c.a)
    # So does a generator's iteration name; its range reads the outer names.
    gen = resolve(:(sum(add(c.a, 1) for c in xs)))
    @test gen.args[2].args[1].args[2] == :(c.a)
    @test gen.args[2].args[2] == :(c = xs)
    # A name no lexical scope binds is still checked as a module path.
    @test_throws "qualifies a call with" resolve(:(map(c -> add(q.a, 1), xs)))
end

end # module
