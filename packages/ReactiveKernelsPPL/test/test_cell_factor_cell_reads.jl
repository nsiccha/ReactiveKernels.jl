module CellFactorCellReadTests
using ReactiveKernels, ReactiveKernelsPPL, Distributions, Enzyme, Test
using DifferentiationInterface: AutoEnzyme

# An observation holding one array per index whose arguments are that index's
# entries of earlier plates (`loc[i]`) gathers no rows of its own: its group
# sum stays in the outer factor. A shared-prefix cut (`partitioned`) must not
# change that when only the cut proves the type of a shared value reaching
# the cell (`scale = exp.(log_scale)`). Nor may either path take a cell's own
# position as a sum's rows when a shared helper leaves the array untyped.
const SHARED = @rkppl begin
    log_scale[1:2] .~ Normal.(0, 1)
    scale = exp.(log_scale)
    z[1:n] .~ Normal.(0, 1)
    effect = scale[1] .* z
    @plate for i in eachindex(paths)
        loc[i] = effect[g[i]] .+ paths[i]
    end
    @plate for i in eachindex(y)
        y[i] .~ Normal.(loc[i], scale[2])
    end
end

untyped_scales(x) = Any[exp(x[1]), exp(x[2])]
const HELPER = @rkppl begin
    log_scale[1:2] .~ Normal.(0, 1)
    scale = untyped_scales(log_scale)
    z[1:n] .~ Normal.(0, 1)
    effect = scale[1] .* z
    @plate for i in eachindex(paths)
        loc[i] = effect[g[i]] .+ paths[i]
    end
    @plate for i in eachindex(y)
        y[i] .~ Normal.(loc[i], scale[2])
    end
end

# The same shared scale with a genuine gather of rows (`hours[i]`) keeps its
# summand factor under both paths.
const GATHERED = @rkppl begin
    log_scale[1:2] .~ Normal.(0, 1)
    scale = exp.(log_scale)
    z[1:n] .~ Normal.(0, 1)
    effect = scale[1] .* z
    @plate for p in eachindex(paths)
        loc[p] = effect[p] .+ paths[p]
    end
    @plate for i in eachindex(y)
        y[i] .~ Normal.(loc[g[i]][hours[i]], scale[2])
    end
end

const REVERSE = AutoEnzyme(; mode = Enzyme.Reverse)
options(n, u) = (("uncached", (;)),
                 ("partitioned", (; partitioned = (:z => collect(1:n),), u0 = copy(u))))
positions(built) = (names = string.(coordinate_names(built.layout));
    (s1 = findfirst(==("log_scale.1"), names), s2 = findfirst(==("log_scale.2"), names),
     z = [findfirst(==("z.$j"), names) for j in 1:count(startswith("z."), names)]))

@testset "cell reads keep the group sum in the outer factor ($name)" for (name, model) in (("shared scale", SHARED), ("untyped helper", HELPER))
    n = 3
    data = (; n, g = [1, 3, 2, 3, 1],
        paths = [[0.1, 0.2], [0.3], [-0.1, 0.4, 0.5], [0.0], Float64[]])
    y = [[0.2, -0.1], [0.5], [0.0, 0.3, 0.1], [-0.2], Float64[]]
    bound = model(; data...) | (; y)
    built = build_kernel(bound)
    u = collect(range(-0.3, 0.4; length = built.layout.total))
    at = positions(built)
    whole = prepare_query(built, bound, :pointwise)(u).y
    # Strict cell queries need the helper's result type; the factors do not.
    cached = model === SHARED ?
        prepare_cell_query(built, bound, :y; options(n, u)[2][2]..., locality = :error) : nothing
    for (label, kw) in options(n, u)
        f = prepare_cell_factors(built, bound, :y; locality = :error, kw...)
        @test isempty(cell_reductions(f))
        q = prepare_cell_factor_query(f, :outer)
        fp = partition_footprint_kernel(q.kernel)
        ad = prepare_cell_factor_ad(prepare_ad, f, :outer, u, [at.z[1]], REVERSE)
        for c in eachindex(y)
            k = at.z[data.g[c]]
            selected = [k]
            scale, sd = exp(u[at.s1]), exp(u[at.s2])
            mu = scale * u[k] .+ data.paths[c]
            expected = sum(logpdf.(Normal.(mu, sd), y[c]); init = 0.0)
            @test q(u, Float64[], c) ≈ expected
            cached === nothing || @test cached(u, c) ≈ expected
            @test sum(whole[c]; init = 0.0) ≈ expected
            reads = Base.invokelatest(fp, u, Float64[], c).unconstrained
            # An untyped helper result is read whole, so its footprint is
            # conservative; it still covers the cell's reads.
            model === SHARED ? (@test reads == sort([at.s1, at.s2, k])) :
                (@test issubset([at.s1, at.s2, k], reads))
            primal, (ds, du) = ad_value_and_gradient(ad, u, Float64[], u[selected], selected, c)
            @test primal ≈ expected
            @test isempty(ds)
            @test du ≈ [sum((y[c] .- mu) .* scale ./ sd^2; init = 0.0)]
        end
    end
end

@testset "a gathered group sum keeps its summand factor under a shared cut" begin
    n = 3
    data = (; n, g = [2, 1, 3, 2], hours = [[2, 1, 2], [3], Int[], [1]],
        paths = [[0.2, 0.7, -0.4], [1.3, -0.1], [0.5]])
    y = [[0.0, 0.2, -0.1], [0.1], Float64[], [0.4]]
    bound = GATHERED(; data...) | (; y)
    built = build_kernel(bound)
    u = collect(range(-0.3, 0.4; length = built.layout.total))
    at = positions(built)
    for (label, kw) in options(n, u)
        f = prepare_cell_factors(built, bound, :y; locality = :error, kw...)
        @test length(cell_reductions(f)) == 1
        term = prepare_cell_factor_query(f, 1)
        outer = prepare_cell_factor_query(f, :outer)
        scale, sd = exp(u[at.s1]), exp(u[at.s2])
        for c in eachindex(y)
            p = data.g[c]
            @test reduction_rows(f, 1, c) == data.hours[c]
            mu = scale * u[at.z[p]] .+ data.paths[p][data.hours[c]]
            terms = [term(u, c, t) for t in eachindex(data.hours[c])]
            @test terms ≈ logpdf.(Normal.(mu, sd), y[c])
            @test outer(u, [sum(terms; init = 0.0)], c) ≈ sum(terms; init = 0.0)
        end
    end
end
end
