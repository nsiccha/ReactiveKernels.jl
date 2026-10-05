using Reactant

function _missing_save_ir(name, kind, text)
    haskey(ENV, "RK_PPL_MISSING_IR_DIR") || return
    dir = ENV["RK_PPL_MISSING_IR_DIR"]
    mkpath(dir)
    write(joinpath(dir, "$name-$kind"), text)
end

function _missing_compiled_check(fx, name; structure=false)
    println("MISSING_COMPILED ", name)
    saved = deepcopy(fx.data)
    kernel = fx.sampler.kernel
    ru = Reactant.to_rarray(fx.u)
    primal = Reactant.@compile kernel(ru)
    reverse = compile_ad_value_and_gradient(fx.sampler.ad, ru)
    for u in (fx.u, fx.u .+ 0.15)
        input = Reactant.to_rarray(u)
        value, grad = sampler_value_and_gradient!(fx.sampler, similar(u), u)
        cv, cg = reverse(input)
        @test Float64(primal(input)) ≈ fx.oracle(u) rtol=1e-9
        @test Float64(cv) ≈ value rtol=1e-9
        @test Array(cg) ≈ grad rtol=1e-8 atol=1e-9
        @test Array(input) == u
    end
    query = prepare_query(fx.built, fx.bound, :pointwise)
    compiled_query = Reactant.@compile query(ru)
    native = Base.invokelatest(query, fx.u)
    actual = compiled_query(ru)
    @test Array(actual.y) ≈ native.y
    @test size(Array(actual.y)) == size(fx.data.y)
    @test isequal(fx.data, saved)
    if structure
        texts = ("primal.mlir" => repr(Reactant.@code_hlo kernel(ru)),
            "reverse.mlir" => repr(Reactant.@code_hlo reverse.f(ru)),
            "primal.hlo" => repr(only(Reactant.XLA.get_hlo_modules(primal.exec))),
            "reverse.hlo" => repr(only(Reactant.XLA.get_hlo_modules(reverse.exec))))
        for (kind, text) in texts
            _missing_save_ir(name, kind, text)
            ops = Dict{String,Int}()
            regex = endswith(kind, "mlir") ? r"stablehlo\.[a-z_]+" :
                r"(?m)^\s*(?:ROOT )?%[\w.\-]+ = .*? ([a-z][a-z0-9-]*)\("
            for m in eachmatch(regex, text)
                op = endswith(kind, "mlir") ? m.match : m.captures[1]
                ops[op] = get(ops, op, 0) + 1
            end
            _missing_save_ir(name, "$kind.inventory", repr(sort!(collect(ops))))
            println("MISSING_STRUCTURE ", name, " ", kind, " ", sort!(collect(ops)))
        end
    end
end

function _missing_compiled_cell_fixture(kind, n)
    y = Union{Missing,Float64}[isodd(i) ? 0.03*i : missing for i in 1:n]
    x = [isodd(i) ? 0.1*i : -1.0 for i in 1:n]
    data = (; y, x)
    response = kind === :local ? :(y[i] ~ Normal(a + b*sqrt(x[i]), 0.7)) :
        :(y[i] ~ Normal(a + b*x[i], sqrt(x[i])))
    ast = quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        @plate for i in eachindex(y)
            $response
        end
    end
    bound = bind_data(lower_rkppl(ast, data; conditioned=keys(data)), data)
    built = build_kernel(bound)
    u = unconstrain(built.layout, (; a=0.2, b=-0.3))
    sampler = prepare_sampler(built, bound, u; backend=AutoEnzyme(; mode=Enzyme.Reverse))
    oracle(v) = logpdf(Normal(), v[1]) + logpdf(Normal(), v[2]) +
        sum(logpdf(Normal(v[1] + v[2]*(kind === :local ? sqrt(x[i]) : x[i]),
            kind === :local ? 0.7 : sqrt(x[i])), y[i]) for i in eachindex(y) if !ismissing(y[i]))
    (; data, bound, built, u, sampler, oracle)
end

function _missing_compiled_glm_fixture(head, n)
    y = head === :NormalIDGLM ?
        Union{Missing,Float64}[isodd(i) ? 0.03*i : missing for i in 1:n] :
        Union{Missing,Int}[isodd(i) ? i%3 == 0 : missing for i in 1:n]
    x1 = [isodd(i) ? 0.1*i : 10000.0 for i in 1:n]
    x2 = fill(0.2, n)
    data = (; y, x1, x2)
    rhs = Expr(:call, head, :X, :alpha, :beta)
    head === :NormalIDGLM && push!(rhs.args, 0.7)
    ast = quote
        X = hcat(x1, x2)
        alpha ~ Normal(0, 1)
        beta[axes(X, 2)] .~ Normal.(0, 1)
        y ~ $rhs
    end
    bound = bind_data(lower_rkppl(ast, data; conditioned=keys(data)), data)
    built = build_kernel(bound)
    u = unconstrain(built.layout, (; alpha=0.2, beta=[0.3, -0.1]))
    sampler = prepare_sampler(built, bound, u; backend=AutoEnzyme(; mode=Enzyme.Reverse))
    function oracle(v)
        p = constrain(built.layout, v)
        eta = p.alpha .+ hcat(x1, x2) * p.beta
        out = logpdf(Normal(), p.alpha) + sum(logpdf.(Normal(), p.beta))
        for i in eachindex(y)
            ismissing(y[i]) && continue
            family = head === :NormalIDGLM ? Normal(eta[i], 0.7) :
                head === :BernoulliLogitGLM ? Bernoulli(1/(1 + exp(-eta[i]))) : Poisson(exp(eta[i]))
            out += logpdf(family, y[i])
        end
        out
    end
    (; data, bound, built, u, sampler, oracle)
end

@testset "Reactant: automatic missing observations, ordinary AD and retained batching" begin
    for kind in (:vector, :computed_plate, :matrix), n in (5, 9)
        y = Union{Missing,Float64}[isodd(i) ? 0.03*i : missing for i in 1:n]
        kind === :matrix && (y = hcat(y, reverse(y)))
        fx = _missing_fixture(y; computed=kind === :computed_plate)
        _missing_compiled_check(fx, "$kind-$n"; structure=true)
    end
    for kind in (:local, :scale), n in (5, 9)
        _missing_compiled_check(_missing_compiled_cell_fixture(kind, n), "$kind-$n"; structure=true)
    end
    for head in (:NormalIDGLM, :BernoulliLogitGLM, :PoissonLogGLM), n in (5, 9)
        _missing_compiled_check(_missing_compiled_glm_fixture(head, n), "$head-$n"; structure=true)
    end
    for y in (Union{Missing,Float64}[], fill(missing, 5))
        _missing_compiled_check(_missing_fixture(y), "empty-or-all-$(length(y))")
    end
    for kind in (:binomial, :bernoulli, :beta, :stopping, :invalid_missing_scale)
        _missing_compiled_check(_missing_family_fixture(kind), "family-$kind")
    end
end
