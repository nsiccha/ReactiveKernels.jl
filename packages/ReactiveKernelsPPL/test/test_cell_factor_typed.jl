using ReactiveKernels, ReactiveKernelsPPL, Distributions, Enzyme, Test, LinearAlgebra
using DifferentiationInterface: AutoEnzyme, SecondOrder

# NB2's guarded mean is declared AbstractVector between the synthesized sum
# plate and the density plate. Each spelling must reach the same factor path.
const _CFT_MODELS = (
    plain = @rkppl(begin
        z[levels(g)] .~ Normal.(0, 1)
        eta = z[g]
        @plate for k in eachindex(y)
            y[k] ~ NegativeBinomial2(sum(exp.(eta[rows[k]])), 2.0)
        end
    end),
    weighted = @rkppl(begin
        z[levels(g)] .~ Normal.(0, 1)
        eta = z[g]
        @plate for k in eachindex(y)
            y[k] ~ NegativeBinomial2(sum(exp.(eta[rows[k]]) .* w[rows[k]]), 1.0 / cv[k]^2)
        end
    end),
    localtwo = @rkppl(begin
        z[levels(g)] .~ Normal.(0, 1)
        phi ~ Exponential(1)
        eta = z[g]
        @plate for k in eachindex(y)
            mu = exp.(eta[rows[k]])
            y[k] ~ NegativeBinomial2(sum(mu), phi * sum(mu)^2 / sum(mu .^ 2))
        end
    end))

_cft_nodes(x) = x isa Expr ? 1 + sum(_cft_nodes, x.args; init=0) : 1
struct _CFTPoint <: AbstractVector{Float64}
    values::Vector{Float64}
    reads::Vector{Int}
end
Base.size(u::_CFTPoint) = size(u.values)
Base.IndexStyle(::Type{_CFTPoint}) = IndexLinear()
Base.getindex(u::_CFTPoint, i::Int) = (push!(u.reads, i); u.values[i])
_cft_support(q, args...) = Base.invokelatest(partition_footprint_kernel(q.kernel), args...).unconstrained

@testset "gathered NB2 sums pass through typed identity assignments" begin
    for (name, model) in pairs(_CFT_MODELS)
        sizes = Vector{Int}[]
        for n in (6, 120)
            rows = [vcat([1, 2, 2], collect(3:n)), [3, 1]]
            name === :localtwo || push!(rows, Int[])
            w = collect(range(0.4, 1.2; length=n))
            y = name === :localtwo ? [3, 4] : [3, 4, 0]
            cv = [0.5, 0.8, 1.1][eachindex(y)]
            bound = model(; g=collect(1:n), rows, w, cv) | (; y)
            built = build_kernel(bound)
            u = collect(range(-0.2, 0.3; length=built.layout.total))
            parameters = constrain(built.layout, u)
            factors = prepare_cell_factors(built, bound, :y; locality=:error)
            # sum(mu) occurs twice in the nonlinear spelling. Graph order,
            # including repeated authored reductions, is the public boundary.
            count = name === :localtwo ? 3 : 1
            @test length(cell_reductions(factors)) == count
            terms = [prepare_cell_factor_query(factors, j) for j in 1:count]
            outer = prepare_cell_factor_query(factors, :outer)
            push!(sizes, [_cft_nodes(code_expr(q.kernel)) for q in [terms; outer]])
            whole = prepare_query(built, bound, :pointwise)(u).y
            names = string.(coordinate_names(built.layout))
            zi = [findfirst(==("z.$i"), names) for i in 1:n]
            oi = name === :localtwo ? [findfirst(==("phi"), names)] : Int[]
            for c in eachindex(rows)
                sums = zeros(count)
                for j in 1:count
                    @test reduction_rows(factors, j, c) == rows[c]
                    for (t, row) in enumerate(rows[c])
                        sums[j] += terms[j](u, c, t)
                        point = _CFTPoint(u, Int[])
                        @test terms[j](point, c, t) == terms[j](u, c, t)
                        @test sort!(unique!(point.reads)) == [zi[row]]
                        @test _cft_support(terms[j], u, c, t) == [zi[row]]
                    end
                end
                mu = exp.(parameters.z[rows[c]])
                mean = name === :weighted ? sum(mu .* w[rows[c]]) : sum(mu)
                shape = name === :localtwo ? parameters.phi * mean^2 / sum(abs2, mu) :
                        name === :weighted ? 1.0 / cv[c]^2 : 2.0
                expected = logpdf(NegativeBinomial(shape, shape / (shape + mean)), y[c])
                point = _CFTPoint(u, Int[])
                @test outer(point, sums, c) ≈ whole[c] ≈ expected
                @test sort!(unique!(point.reads)) == oi
                @test _cft_support(outer, u, sums, c) == oi
            end
            # Invalid supplied means retain the authored guard; no eager log
            # or gamma calculation may turn this path into an exception.
            @test outer(u, fill(-1.0, count), 1) == -Inf
            if name !== :localtwo
                @test_throws BoundsError terms[1](u, 3, 1)
            end
        end
        @test sizes[1] == sizes[2]
    end
end

@testset "nonlinear NB2 factors reconstruct native gradient and Hessian" begin
    rows, y = [[1, 2, 2], [3, 1]], [3, 4]
    bound = _CFT_MODELS.localtwo(; g=collect(1:3), rows) | (; y)
    built = build_kernel(bound)
    u = collect(range(-0.2, 0.3; length=built.layout.total))
    f = prepare_cell_factors(built, bound, :y; locality=:error)
    count, n = length(cell_reductions(f)), length(u)
    @test count == 3
    reverse = AutoEnzyme(; mode=Enzyme.Reverse)
    second = SecondOrder(reverse, AutoEnzyme(; mode=Enzyme.Forward))
    terms = [prepare_cell_factor_query(f, j) for j in 1:count]
    outer = prepare_cell_factor_query(f, :outer)
    oi = _cft_support(outer, u, ones(count), 1)
    @test length(oi) == 1
    unit(i, n) = Float64[k == i for k in 1:n]
    term_ops = map(1:count) do j
        indices = _cft_support(terms[j], u, 1, 1)
        (prepare_cell_factor_ad(prepare_ad, f, j, u, indices, reverse),
         prepare_cell_factor_ad(prepare_ad_hvp, f, j, u, indices, second, ([1.0],)))
    end
    outer_units = ntuple(count + 1) do i
        e = unit(i, count + 1)
        (e[1:count], e[count+1:end])
    end
    outer_g = prepare_cell_factor_ad(prepare_ad, f, :outer, u, oi, reverse)
    outer_h = prepare_cell_factor_ad(prepare_ad_hvp, f, :outer, u, oi, second, outer_units)
    allindices = collect(1:n)
    units = ntuple(i -> unit(i, n), n)
    whole_g = prepare_compact_cell_ad(prepare_ad, built, bound, :y, u, allindices, reverse)
    whole_h = prepare_compact_cell_ad(prepare_ad_hvp, built, bound, :y, u, allindices, second, units)
    for c in eachindex(rows)
        sums, J = zeros(count), zeros(count + 1, n)
        J[end, only(oi)] = 1
        Hs = [zeros(n, n) for j in 1:count]
        for j in 1:count, t in eachindex(rows[c])
            indices = _cft_support(terms[j], u, c, t)
            point = (u, u[indices], indices, c, t)
            value, gradient = ad_value_and_gradient(term_ops[j][1], point...)
            _, (hvp,) = ad_gradient_and_hvp(term_ops[j][2], ([1.0],), point...)
            sums[j] += value
            J[j, indices] .+= gradient
            Hs[j][indices, indices] .+= reshape(hvp, 1, 1)
        end
        point = (u, sums, u[oi], oi, c)
        value, gradient = ad_value_and_gradient(outer_g, point...)
        _, columns = ad_gradient_and_hvp(outer_h, outer_units, point...)
        gout, Hout = vcat(gradient...), hcat((vcat(column...) for column in columns)...)
        expected, eg = ad_value_and_gradient(whole_g, u, u, allindices, c)
        _, eh = ad_gradient_and_hvp(whole_h, units, u, u, allindices, c)
        @test value ≈ expected
        @test J' * gout ≈ eg atol=2e-10
        @test J' * Hout * J + sum(gout[j] .* Hs[j] for j in 1:count) ≈ hcat(eh...) atol=2e-9
    end
end

@testset "weighted summand native derivatives retain the selected row" begin
    bound = _CFT_MODELS.weighted(; g=collect(1:3), rows=[[1, 2, 2], [3, 1]],
                                w=[0.4, 0.8, 1.2], cv=[0.5, 0.8]) | (; y=[3, 4])
    built = build_kernel(bound)
    u = collect(range(-0.2, 0.3; length=built.layout.total))
    f = prepare_cell_factors(built, bound, :y; locality=:error)
    q = prepare_cell_factor_query(f, 1)
    reverse = AutoEnzyme(; mode=Enzyme.Reverse)
    second = SecondOrder(reverse, AutoEnzyme(; mode=Enzyme.Forward))
    indices = _cft_support(q, u, 1, 1)
    g = prepare_cell_factor_ad(prepare_ad, f, 1, u, indices, reverse)
    h = prepare_cell_factor_ad(prepare_ad_hvp, f, 1, u, indices, second, ([1.0],))
    for (c, t) in ((1, 1), (1, 3), (2, 1))
        indices = _cft_support(q, u, c, t)
        point = (u, u[indices], indices, c, t)
        value, gradient = ad_value_and_gradient(g, point...)
        _, (hv,) = ad_gradient_and_hvp(h, ([1.0],), point...)
        @test value ≈ q(u, c, t)
        @test only(gradient) ≈ value
        @test only(hv) ≈ value
    end
end

module _CFTConversions
using ReactiveKernels
struct Wrapped
    values::Vector{Float64}
end
Base.convert(::Type{AbstractVector}, x::Wrapped) = x.values .+ 1
@kernel convert_container(x::Wrapped) = begin
    y::AbstractVector = x
    values = plate(y) do v
        v
    end
    result = sum(values)
end
@kernel convert_elements(x::Vector{Float64}) = begin
    y::Vector{Float32} = x
    values = plate(y) do v
        v
    end
    result = sum(values)
end
container(x::AbstractVector) = Wrapped(collect(x))
container(x::Vector) = x
@kernel type_dependent(x::AbstractVector{Float64}) = begin
    raw = container(x)
    y::AbstractVector = raw
    values = plate(y) do v
        v
    end
    result = sum(values)
end
end

@testset "factor discovery preserves real typed conversions" begin
    for (spec, x) in ((_CFTConversions.convert_container, _CFTConversions.Wrapped([1.0, 2.0])),
                      (_CFTConversions.convert_elements, [1.0, eps(Float64)]),
                      (_CFTConversions.type_dependent, _CFTPoint([1.0, 2.0], Int[])))
        original = prepare(spec; want=:result)
        factors = ReactiveKernels._cell_factor_graph(spec, :result)
        factored = prepare(factors.spec; want=:result)
        @test factored(x) === original(x)
    end
    @test prepare(_CFTConversions.convert_container; want=:result)(_CFTConversions.Wrapped([1.0, 2.0])) == 5.0
    @test prepare(_CFTConversions.convert_elements; want=:result)([1.0, eps(Float64)]) === 1.0f0
    @test prepare(_CFTConversions.type_dependent; want=:result)(_CFTPoint([1.0, 2.0], Int[])) == 5.0
end
