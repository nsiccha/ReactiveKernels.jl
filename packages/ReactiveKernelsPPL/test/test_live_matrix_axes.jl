module PPLLiveMatrixAxesTests
using ReactiveKernels, ReactiveKernelsPPL, DifferentiationInterface, Enzyme,
    Distributions, Test

const BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)

@rkppl axis_component(X) = begin
    beta[axes(X, 2)] .~ Normal.(0, 1)
    return 0.0
end
@rkppl fixed_component(X, ncoef) = begin
    beta[1:ncoef] .~ Normal.(0, 1)
    return 0.0
end

# Binding must infer the broadcast's axes without calling its scalar body.
unavailable_value(a) = error("live scalar body was evaluated")

function source(kind)
    prefix = quote
        a ~ Normal(0, 1)
        X = hcat(ones(length(x)), a .* x)
    end
    if kind === :nested
        return quote $(prefix.args...); pop ~ axis_component(X); y .~ Normal.(0, 1) end
    elseif kind === :unused
        return quote $(prefix.args...); beta[axes(X, 2)] .~ Normal.(0, 1); y .~ Normal.(0, 1) end
    elseif kind === :unavailable
        return quote
            a ~ Normal(0, 1)
            X = hcat(ones(length(x)), unavailable_value.(a .* x))
            beta[axes(X, 2)] .~ Normal.(0, 1)
            y .~ Normal.(0, 1)
        end
    elseif kind === :alias
        return quote
            $(prefix.args...)
            Alias = X
            beta[axes(Alias, 2)] .~ Normal.(0, 1)
            y .~ Normal.(Alias * beta, 1)
        end
    elseif kind === :named
        return quote
            a ~ Normal(0, 1)
            v = exp.(a .* x)
            X = hcat(ones(length(x)), v)
            beta[axes(X, 2)] .~ Normal.(0, 1)
            y .~ Normal.(X * beta, 1)
        end
    elseif kind === :vector
        return quote
            z[axes(D, 1)] .~ Normal.(0, 1)
            X = hcat(ones(length(x)), z .* x)
            beta[axes(X, 2)] .~ Normal.(0, 1)
            y .~ Normal.(X * beta, 1)
        end
    elseif kind === :matrix
        return quote
            Z[axes(D, 1), axes(D, 2)] .~ Normal.(0, 1)
            X = hcat(ones(length(x)), Z .* D)
            beta[axes(X, 2)] .~ Normal.(0, 1)
            y .~ Normal.(X * beta, 1)
        end
    elseif kind === :product
        return quote
            z[axes(D, 2)] .~ Normal.(0, 1)
            v = D * z
            X = hcat(ones(length(x)), v)
            beta[axes(X, 2)] .~ Normal.(0, 1)
            y .~ Normal.(X * beta, 1)
        end
    end
    error("unknown fixture $kind")
end

function fixed_axes(ex, ncoef)
    ex isa Expr || return ex
    ex == :(axes(X, 2)) && return :(1:$ncoef)
    ex == :(axes(Alias, 2)) && return :(1:$ncoef)
    ex == :(axis_component(X)) && return :(fixed_component(X, $ncoef))
    return Expr(ex.head, map(a -> fixed_axes(a, ncoef), ex.args)...)
end

function build_case(kind, n; fixed = false, names_only = fixed)
    x = [0.1 + 0.04i for i in 1:n]
    data = (; x, D = hcat(x, x .+ 0.2), y = [0.2sin(i) for i in 1:n])
    ast = source(kind)
    fixed && (ast = fixed_axes(ast, kind === :matrix ? 3 : 2))
    # Exercise both public lowering doors: values and names followed by bind.
    input = names_only ? Set(keys(data)) : data
    bound = bind_data(lower_rkppl(ast, input; conditioned = (:y,), mod = @__MODULE__), data)
    built = build_kernel(bound)
    return bound, built, data
end

function reference(kind, data, names, u)
    q = Dict(zip(names, u))
    prior = sum(v -> logpdf(Normal(), v), u; init = 0.0)
    mean = if kind in (:unused, :nested, :unavailable)
        zeros(length(data.x))
    else
        b = [q[Symbol("beta.$i")] for i in 1:(kind === :matrix ? 3 : 2)]
        column = kind === :named ? exp.(q[:a] .* data.x) :
            kind === :vector ? [q[Symbol("z.$i")] * data.x[i] for i in eachindex(data.x)] :
            kind === :product ? data.D * [q[Symbol("z.$i")] for i in 1:2] :
            kind === :matrix ? [sum(q[Symbol("Z.$i.$j")] * data.D[i,j] * b[j+1]
                for j in 1:2) for i in eachindex(data.x)] : q[:a] .* data.x
        kind === :matrix ? b[1] .+ column : b[1] .+ b[2] .* column
    end
    return prior + sum(logpdf.(Normal.(mean, 1), data.y); init = 0.0)
end

function gradient(f, u)
    h = cbrt(eps(Float64))
    return [(f(u .+ h .* (eachindex(u) .== i)) -
        f(u .- h .* (eachindex(u) .== i))) / (2h) for i in eachindex(u)]
end

const KINDS = (:unused, :nested, :unavailable, :alias, :named, :vector, :matrix, :product)

@testset "live matrix axes retain priors and infer shapes without values" begin
    for kind in KINDS, n in (0, 1, 7)
        @testset "$kind / $n" begin
            bound, built, data = build_case(kind, n)
            control, fixed, _ = build_case(kind, n; fixed = true)
            named_bound, named_built, _ = build_case(kind, n; names_only = true)
            original = deepcopy(data)
            names = coordinate_names(built.layout)
            @test names == coordinate_names(fixed.layout)
            @test names == coordinate_names(named_built.layout)
            ncoef = kind === :matrix ? 3 : 2
            stem = kind === :nested ? "pop.beta" : "beta"
            @test all(Symbol("$stem.$i") in names for i in 1:ncoef)
            @test built.layout.total == ncoef +
                (kind === :vector ? n : kind === :matrix ? 2n : kind === :product ? 2 : 1)
            u = [0.2sin(i) for i in eachindex(names)]
            kernel = prepare_query(built, bound, :sampler)
            other = prepare_query(fixed, control, :sampler)
            # Diff the emitted density against the explicit-count workaround,
            # rather than attributing a density or performance cost to it.
            @test ReactiveKernels._canonical_locals(code_expr(kernel)) ==
                ReactiveKernels._canonical_locals(code_expr(other))
            @test ReactiveKernels._canonical_locals(code_expr(kernel)) ==
                ReactiveKernels._canonical_locals(code_expr(
                    prepare_query(named_built, named_bound, :sampler)))
            ad = prepare_ad(kernel, BACKEND, u; active = :unconstrained)
            for w in (u, u .+ 0.13)
                oracle = v -> reference(kind, data, names, v)
                value, grad = ad_value_and_gradient(ad, w)
                @test value ≈ oracle(w) rtol = 1e-12
                @test value ≈ other(w) rtol = 1e-12
                @test grad ≈ gradient(oracle, w) rtol = 1e-5 atol = 1e-7
                @test data == original
                kind in (:unused, :nested, :unavailable) && (@test grad ≈ -w)
            end
        end
    end
end

end
