using DifferentiationInterface, Distributions, Enzyme
using LinearAlgebra
using ReactiveKernels, ReactiveKernelsPPL, Test

module SharedArrayLocationModels
const calls = Ref(0)
signal(grid, group, first, second) = first[group] .* grid .+ second[group]
function counted_signal(args...)
    calls[] += 1
    return signal(args...)
end
end

function _sal_model(kind, route, dependent, censor; counted = true)
    ast = quote
        a ~ Normal(0, 1)
        d ~ Normal(0, 1)
        add1 ~ Exponential(1)
        add2 ~ Exponential(1)
        prop1 ~ Exponential(1)
        prop2 ~ Exponential(1)
    end
    declaration, first, second = if kind === :vector
        quote
            z[levels(group)] .~ Normal.(0, 1)
            w[levels(group)] .~ Normal.(0, 1)
        end, :z, :w
    elseif kind === :column
        :(B[levels(group), 1:2] .~ Normal.(0, 1)), :(B[:, 1]), :(B[:, 2])
    else
        quote
            L ~ LKJCholesky(2, 2)
            scale[1:2] .~ Exponential.(1)
            eachrow(B[levels(group), 1:2]) .~
                MvNormalCholesky(zeros(2), scale .* L)
        end, :(B[:, 1]), :(B[:, 2])
    end
    append!(ast.args, declaration.head === :block ? declaration.args : [declaration])
    push!(ast.args, :(first = a .+ $first .+ d .* group_x))
    push!(ast.args, :(second = a .+ $second .+ d .* group_x))
    fn = counted ? :counted_signal : :signal
    call = Expr(:call, fn, :grid, :grid_group, :first, :second)
    if route === :inline
        push!(ast.args, :(mu = $call[obs_index]))
    else
        push!(ast.args, :(reads = $call))
        route === :alias && push!(ast.args, :(alias = reads))
        value = route === :alias ? :alias : :reads
        push!(ast.args, :(mu = $value[obs_index]))
    end
    push!(ast.args, :(add = [add1, add2][assay]))
    push!(ast.args, :(prop = [prop1, prop2][assay]))
    push!(ast.args, dependent ? :(sd = hypot.(add, mu .* prop)) : :(sd = add))
    push!(ast.args, censor ? :(y .~ censored.(Normal.(mu, sd), lower, Inf)) :
        :(y .~ Normal.(mu, sd)))
    return ast
end

function _sal_build(kind, route, dependent, censor; counted = true, K = 2, n = 3)
    m = n + 1
    data = Dict(:grid => [0.3 * i for i in 1:m],
        :grid_group => [mod1(i, K) for i in 1:m],
        :group_x => [0.2 * i for i in 1:K],
        :group => [mod1(i, K) for i in 1:n],
        :obs_index => collect(1:n), :assay => [mod1(i, 2) for i in 1:n],
        :y => [i == 1 ? 0.1 : 0.3 * i for i in 1:n], :lower => fill(0.1, n))
    plan = lower_rkppl(_sal_model(kind, route, dependent, censor; counted),
        keys(data); mod = SharedArrayLocationModels)
    bound = bind_data(plan, data)
    built = build_kernel(bound)
    u = [0.2 * cos(i) for i in 1:built.layout.total]
    return (; plan, bound, built, data, u)
end

function _sal_reference(kind, dependent, censor, fx, u)
    th = constrain(fx.built.layout, u)
    first, second = kind === :vector ? (th.z, th.w) : (th.B[:, 1], th.B[:, 2])
    first = th.a .+ first .+ th.d .* fx.data[:group_x]
    second = th.a .+ second .+ th.d .* fx.data[:group_x]
    reads = SharedArrayLocationModels.signal(fx.data[:grid], fx.data[:grid_group],
        first, second)
    mu = reads[fx.data[:obs_index]]
    add = [th.add1, th.add2][fx.data[:assay]]
    prop = [th.prop1, th.prop2][fx.data[:assay]]
    sd = dependent ? hypot.(add, mu .* prop) : add
    ds = Normal.(mu, sd)
    censor && (ds = censored.(ds, fx.data[:lower], Inf))
    likelihood = sum(logpdf.(ds, fx.data[:y]))
    prior = logpdf(Normal(), th.a) + logpdf(Normal(), th.d) +
        sum(logpdf.(Exponential(), [th.add1, th.add2, th.prop1, th.prop2]))
    if kind === :centered
        F = th.scale .* th.L
        prior += logpdf(LKJCholesky(2, 2), Cholesky(LowerTriangular(th.L))) +
            sum(logpdf.(Exponential(), th.scale)) +
            sum(logpdf(MvNormal(zeros(2), F * F'), row) for row in eachrow(th.B))
    elseif kind === :column
        prior += sum(logpdf.(Normal(), th.B))
    else
        prior += sum(logpdf.(Normal(), th.z)) + sum(logpdf.(Normal(), th.w))
    end
    return likelihood + prior + logjac(fx.built.layout, u)
end

function _sal_findiff(f, u; h = cbrt(eps(Float64)))
    map(eachindex(u)) do i
        up, dn = copy(u), copy(u)
        up[i] += h
        dn[i] -= h
        (f(up) - f(dn)) / (2h)
    end
end

@testset "array-reading location is evaluated once" begin
    for kind in (:vector, :column, :centered), route in (:named, :inline, :alias),
            dependent in (false, true)
        fx = _sal_build(kind, route, dependent, false)
        kern = prepare_query(fx.built, fx.bound, :sampler)
        before = deepcopy(fx.data)
        for u in (fx.u, fx.u .+ 0.1)
            SharedArrayLocationModels.calls[] = 0
            value = Base.invokelatest(kern, u)
            @test SharedArrayLocationModels.calls[] == 1
            @test value ≈ _sal_reference(kind, dependent, false, fx, u) rtol = 1e-12
        end
        @test fx.data == before
        src = string(kernel_expr(fx.bound, fx.built.layout))
        @test count("counted_signal", src) == 1
        if dependent
            scale = only(p for p in fx.plan.predictors if p.name === :sd)
            @test :mu in only(scale.terms).options.subs
            @test occursin("_ppl_lp_mu", string(only(
                s for s in ReactiveKernelsPPL._predictor_statements(fx.bound)
                if s.args[1] === :_ppl_lp_sd)))
        end
    end
    for dependent in (false, true)
        fx = _sal_build(:centered, :named, dependent, true)
        kern = prepare_query(fx.built, fx.bound, :sampler)
        SharedArrayLocationModels.calls[] = 0
        @test Base.invokelatest(kern, fx.u) ≈
            _sal_reference(:centered, dependent, true, fx, fx.u) rtol = 1e-12
        @test SharedArrayLocationModels.calls[] == 1
    end
end

@testset "shared array-reading location native gradients" begin
    # Differentiate the pure leaf; the call-count instrumentation belongs
    # only to the primal evaluation tests above.
    for kind in (:vector, :column, :centered), censor in (false, true)
        fx = _sal_build(kind, :named, true, censor; counted = false)
        q = prepare_sampler(fx.built, fx.bound, fx.u;
            backend = AutoEnzyme(; mode = Enzyme.Reverse))
        before = deepcopy(fx.data)
        value, grad = sampler_value_and_gradient!(q, similar(fx.u), fx.u)
        ref(u) = _sal_reference(kind, true, censor, fx, u)
        @test value ≈ ref(fx.u) rtol = 1e-12
        @test grad ≈ _sal_findiff(ref, fx.u) rtol = 1e-5 atol = 1e-7
        @test fx.data == before
    end
end
