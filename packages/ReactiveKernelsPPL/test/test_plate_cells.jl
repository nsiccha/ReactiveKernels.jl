using Distributions
using LinearAlgebra
using ReactiveKernels
using ReactiveKernelsPPL
using Test

include("fixtures/level_plate_values.jl")

# `@plate` cells beyond scalars (user decision 01gqbsq, stratified prong):
# a cell means one iteration of the Julia loop, whatever the shape of its
# values. Per-level cells (`@plate for k in levels(g)`) lower to the
# whole-array declarations they mean. Helper: `_canon` (test_corpus.jl).

_pc_canon(ex, data) = sprint(_canon, lower_rkppl(ex, data; mod = @__MODULE__, conditioned = data))

@testset "per-level plate cells are their whole-array declarations" begin
    data = (:y, :x, :g, :s)
    # One scalar draw per level: `u[k] ~ Normal(0, su)` is `u[levels(s)] .~ Normal.(0, su)`.
    # One row per level: `b[j, :] ~ D` is the row statement.
    plated = quote
        a ~ Normal(0, 5); sigma ~ Exponential(1); su ~ HalfNormal(1)
        sd[1:2] .~ HalfNormal.(1); L ~ LKJCholesky(2, 1.0); F = sd .* L
        @plate for k in levels(s)
            u[k] ~ Normal(0, su)
        end
        @plate for j in levels(g)
            b[j, :] ~ MvNormalCholesky(zeros(2), F)
        end
        mu = a .+ b[g, 1] .+ x .* b[g, 2] .+ u[s]
        y .~ Normal.(mu, sigma)
    end
    written = quote
        a ~ Normal(0, 5); sigma ~ Exponential(1); su ~ HalfNormal(1)
        sd[1:2] .~ HalfNormal.(1); L ~ LKJCholesky(2, 1.0); F = sd .* L
        @plate for k in levels(s)
            u[k] ~ Normal(0, su)
        end
        @plate for j in levels(g)
            b[j, 1:2] ~ MvNormalCholesky(zeros(2), F)
        end
        mu = a .+ b[g, 1] .+ x .* b[g, 2] .+ u[s]
        y .~ Normal.(mu, sigma)
    end
    whole = quote
        a ~ Normal(0, 5); sigma ~ Exponential(1); su ~ HalfNormal(1)
        sd[1:2] .~ HalfNormal.(1); L ~ LKJCholesky(2, 1.0); F = sd .* L
        u[levels(s)] .~ Normal.(0, su)
        eachrow(b[levels(g), 1:2]) .~ MvNormalCholesky(zeros(2), F)
        mu = a .+ b[g, 1] .+ x .* b[g, 2] .+ u[s]
        y .~ Normal.(mu, sigma)
    end
    @test _pc_canon(plated, data) == _pc_canon(whole, data)
    @test _pc_canon(written, data) == _pc_canon(whole, data)
end

@testset "per-level LKJ factors and per-index cells that hold arrays" begin
    data = (:y, :x, :g, :s)
    cols = Dict{Symbol,AbstractVector}(:s => repeat([1, 2], 12),
        :g => repeat([1, 2, 3, 4], 6), :x => collect(range(-1.3, 1.7; length = 24)),
        :y => [0.8 * sin(0.7 * i) for i in 1:24])
    # Columns per component and a row per index lower to one model.
    cols_prog = quote
        a ~ Normal(0, 5); sigma ~ Exponential(1)
        sd[levels(s), 1:2] .~ HalfNormal.(1)
        @plate for k in levels(s)
            L[k] ~ LKJCholesky(2, 1.0)
        end
        z[levels(g), 1:2] .~ Normal.(0, 1)
        @plate for i in eachindex(g)
            F = sd[s[i], :] .* L[s[i]]
            row = F * z[g[i], :]
            r1[i] = row[1]
            r2[i] = row[2]
        end
        mu = a .+ r1 .+ x .* r2
        y .~ Normal.(mu, sigma)
    end
    rows_prog = quote
        a ~ Normal(0, 5); sigma ~ Exponential(1)
        sd[levels(s), 1:2] .~ HalfNormal.(1)
        @plate for k in levels(s)
            L[k] ~ LKJCholesky(2, 1.0)
        end
        z[levels(g), 1:2] .~ Normal.(0, 1)
        @plate for i in eachindex(g)
            b[i, 1:2] = (sd[s[i], :] .* L[s[i]]) * z[g[i], :]
        end
        mu = a .+ b[:, 1] .+ x .* b[:, 2]
        y .~ Normal.(mu, sigma)
    end
    for prog in (cols_prog, rows_prog)
        plan = bind_data(lower_rkppl(prog, data; mod = @__MODULE__, conditioned = data), cols)
        built = build_kernel(plan)
        u = [0.3 * sin(1.1 * i + 0.3) for i in 1:built.layout.total]
        nt = constrain(built.layout, u)
        @test size(nt.L) == (2, 2, 2)
        @test unconstrain(built.layout, nt) ≈ u
        mu = map(1:24) do i
            si, gi = cols[:s][i], cols[:g][i]
            row = (Diagonal(nt.sd[si, :]) * nt.L[:, :, si]) * nt.z[gi, :]
            nt.a + row[1] + cols[:x][i] * row[2]
        end
        hn(v) = logpdf(truncated(Normal(0, 1), 0, Inf), v)
        oracle = logpdf(Normal(0, 5), nt.a) +
            logpdf(Exponential(1), nt.sigma) + sum(hn, nt.sd) +
            sum(logpdf(LKJCholesky(2, 1.0),
                Cholesky(LowerTriangular(nt.L[:, :, k]))) for k in 1:2) +
            sum(logpdf.(Normal(0, 1), nt.z)) +
            sum(logpdf.(Normal.(mu, nt.sigma), cols[:y])) +
            logjac(built.layout, u)
        @test prepare_query(built, plan, :sampler)(u) ≈ oracle
        _check_gradient(built.spec, plan, u)
    end
end

@testset "per-level arguments and deterministic values" begin
    for kind in (:varying, :latent, :deterministic, :constant), S in (1, 3)
        built, bound, cols, u = _plv_build(kind, 12, S)
        saved = deepcopy(cols)
        value, c = _plv_oracle(built, cols, kind, u)
        @test prepare_query(built, bound, :sampler)(u) ≈ value
        @test _query(built.spec, bound, :c, u) ≈ c
        @test unconstrain(built.layout, constrain(built.layout, u)) ≈ u
        grad = _check_gradient(built.spec, bound, u)
        @test grad ≈ _findiff_grad(w -> first(_plv_oracle(built, cols, kind, w)), u)
        @test cols == saved
    end
    # Symbolic levels index declared arrays by level, without exposing
    # ordinal positions as the loop's authored values.
    built, bound, cols, u = _plv_build(:latent, 12, 3; labels = ["c", "a", "b"])
    @test prepare_query(built, bound, :sampler)(u) ≈
        first(_plv_oracle(built, cols, :latent, u))
    _check_gradient(built.spec, bound, u)

    data = (:y, :g, :m)
    lower(cell) = lower_rkppl(Expr(:block, :(a ~ Normal(0, 5)),
        Expr(:macrocall, Symbol("@plate"), LineNumberNode(0),
            Expr(:for, :(k = levels(g)), Expr(:block, cell))),
        :(mu = a .+ c[g]), :(y .~ Normal.(mu, 1.0))), data; conditioned = data)
    @test lower(:(c[k] ~ Normal(m[k], 1))) isa StructuralPlan
    @test lower(:(c[k] = 2.0)) isa StructuralPlan
end

@testset "per-level cells align selected and different level axes" begin
    for kind in (:selected, :reordered, :other, :other_selected, :varying,
            :matrix_selected, :matrix_other, :lazy, :live, :live_oob), S in (3, 5)
        built, bound, cols, u, oracle = _plv_axis_build(kind, 2S + 1, S)
        saved = deepcopy(cols)
        expected, c = oracle(u)
        @test prepare_query(built, bound, :sampler)(u) ≈ expected
        @test _query(built.spec, bound, :c, u) ≈ c
        @test _check_gradient(built.spec, bound, u) ≈
            _findiff_grad(w -> first(oracle(w)), u) rtol = 1e-5 atol = 1e-7
        @test cols == saved
    end
    for kind in (:selected, :other)
        labels = ["c", "a", "b", "unused"]
        built, bound, cols, u, oracle = _plv_axis_build(kind, 9, 4;
            labels, pooled = true)
        @test prepare_query(built, bound, :sampler)(u) ≈ first(oracle(u))
        @test _query(built.spec, bound, :c, u) ≈ last(oracle(u))
        @test _check_gradient(built.spec, bound, u) ≈
            _findiff_grad(w -> first(oracle(w)), u) rtol = 1e-5 atol = 1e-7
        @test levels(cols[:g]) == labels
    end
    for kind in (:selected, :other, :other_selected)
        built, bound, cols, u, oracle = _plv_axis_build(kind, 9, 3;
            labels = ["c", "a", "b"])
        @test prepare_query(built, bound, :sampler)(u) ≈ first(oracle(u))
        @test _check_gradient(built.spec, bound, u) ≈
            _findiff_grad(w -> first(oracle(w)), u) rtol = 1e-5 atol = 1e-7
    end
    built, bound, cols, u, oracle = _plv_stack_build(9, 3)
    @test prepare_query(built, bound, :sampler)(u) ≈ first(oracle(u))
    @test _query(built.spec, bound, :c, u) ≈ last(oracle(u))
    @test _check_gradient(built.spec, bound, u) ≈
        _findiff_grad(w -> first(oracle(w)), u) rtol = 1e-5 atol = 1e-7
    # A different declared axis still needs coverage of every loop label.
    prog = quote
        z[levels(h)] .~ Normal.(0, 1)
        @plate for k in levels(g)
            c[k] = z[k]
        end
        y .~ Normal.(c[g], 1)
    end
    plan = lower_rkppl(prog, (:y, :g, :h); conditioned = (:y,))
    @test_throws ContractValidationError bind_data(plan,
        Dict(:y => zeros(3), :g => [2, 4, 6], :h => [0, 2, 4]))
    @test_throws ContractValidationError bind_data(plan,
        Dict(:y => zeros(3),
            :g => categorical([2, 4, 2]; levels = [2, 4, 6]),
            :h => categorical([0, 2, 4]; levels = [0, 2, 4])))
end

@testset "per-level assignments keep inactive branches inactive" begin
    built, bound, u, oracle = _plv_lazy_build()
    @test prepare_query(built, bound, :sampler)(u) ≈ oracle(u)
    @test _check_gradient(built.spec, bound, u) ≈ _findiff_grad(oracle, u)
end

# A per-index cell's label index may be data or a data-only definition
# (`g = glab[pos]`): along a `levels(g)` axis either one is looked up among
# the axis labels (rkppl-use §3 "Labels or positions", §13). The labels are
# strings and integers other than 1:G, where a positional read would not
# reproduce the oracle (snag rkppl-submodel-l-00ff0f27: BRM's stratified
# group effects over a grouping computed from data).
@rkppl _pc_stratified(g, s) = begin
    tau[levels(s), 1:1] .~ restricted.(Normal(0, 1), 0.0, Inf)
    z[levels(g), 1:1] .~ Normal.(0, 1)
    @plate for i = eachindex(g)
        b[i, 1:1] = tau[s[i], :] .* z[g[i], :]
    end
    return b
end

@testset "per-index cells look labels up through data-only definitions" begin
    pos = [1, 2, 3, 4, 1, 3, 2, 4, 2]
    y = [0.4 * sin(1.3t) for t in 1:9]
    code(v, pool) = indexin(v, sort(unique(pool)))
    function check(definition, twin, cols, twin_cols, oracle)
        bound = bind_data(lower_rkppl(definition, keys(cols);
            mod = @__MODULE__, conditioned = (:y,)), cols)
        built = build_kernel(bound)
        u = [0.3 * sin(1.7k) - 0.1 for k in 1:built.layout.total]
        _check_model_math(built, bound, u, w -> oracle(constrain(built.layout, w)) +
            logjac(built.layout, w))
        # Binding the definition's value as data is the same program.
        tbound = bind_data(lower_rkppl(twin, keys(twin_cols);
            mod = @__MODULE__, conditioned = (:y,)), twin_cols)
        tbuilt = build_kernel(tbound)
        @test coordinate_names(tbuilt.layout) == coordinate_names(built.layout)
        @test isequal(prepare_query(tbuilt, tbound, :sampler)(u),
            prepare_query(built, bound, :sampler)(u))
    end
    @testset "stratified submodel, $what labels" for (what, glab, slab) in (
            ("string", ["b2", "a1", "b1", "a2"], ["b", "a", "b", "a"]),
            ("integer", [104, 101, 103, 102], [9, 7, 9, 7]))
        g, s = glab[pos], slab[pos]
        cols = (; glab, slab, pos, y)
        saved = deepcopy(cols)
        oracle(p) = sum(logpdf.(Normal(0, 1), p.b.tau)) + sum(logpdf.(Normal(0, 1), p.b.z)) +
            sum(logpdf.(Normal.(p.b.tau[code(s, s), 1] .* p.b.z[code(g, g), 1], 1.0), y))
        twin = quote
            b ~ _pc_stratified(g, s)
            y .~ Normal.(b[:, 1], 1.0)
        end
        check(quote
            g = glab[pos]
            s = slab[pos]
            b ~ _pc_stratified(g, s)
            y .~ Normal.(b[:, 1], 1.0)
        end, twin, cols, (; g, s, y), oracle)
        # One definition beside a data column.
        check(quote
            g = glab[pos]
            b ~ _pc_stratified(g, s)
            y .~ Normal.(b[:, 1], 1.0)
        end, twin, (; glab, pos, s, y), (; g, s, y), oracle)
        @test cols == saved
    end
    @testset "per-level factors read at a definition's labels" begin
        glab, slab = ["d", "a", "c", "b"], ["y", "x", "y", "x"]
        g, s = glab[pos], slab[pos]
        x = collect(range(-1.3, 1.7; length = 9))
        cell = quote
            a ~ Normal(0, 5); sigma ~ Exponential(1)
            sd[levels(s), 1:2] .~ HalfNormal.(1)
            @plate for k in levels(s)
                L[k] ~ LKJCholesky(2, 1.0)
            end
            z[levels(g), 1:2] .~ Normal.(0, 1)
            @plate for i in eachindex(g)
                b[i, 1:2] = (sd[s[i], :] .* L[s[i]]) * z[g[i], :]
            end
            mu = a .+ b[:, 1] .+ x .* b[:, 2]
            y .~ Normal.(mu, sigma)
        end
        function oracle(p)
            mu = map(1:9) do i
                si, gi = code(s, s)[i], code(g, g)[i]
                row = (Diagonal(p.sd[si, :]) * p.L[:, :, si]) * p.z[gi, :]
                p.a + row[1] + x[i] * row[2]
            end
            hn(v) = logpdf(truncated(Normal(0, 1), 0, Inf), v)
            return logpdf(Normal(0, 5), p.a) + logpdf(Exponential(1), p.sigma) +
                sum(hn, p.sd) + sum(logpdf(LKJCholesky(2, 1.0),
                    Cholesky(LowerTriangular(p.L[:, :, k]))) for k in 1:2) +
                sum(logpdf.(Normal(0, 1), p.z)) + sum(logpdf.(Normal.(mu, p.sigma), y))
        end
        check(Expr(:block, :(g = glab[pos]), :(s = slab[pos]), cell.args...), cell,
            (; glab, slab, pos, x, y), (; g, s, x, y), oracle)
    end
    @testset "an index definition on another column's levels" begin
        h = ["b2", "a1", "b1", "a2"]
        gg = ["a1", "a2", "b1", "b2"][pos]
        cell = quote
            z[levels(h), 1:1] .~ Normal.(0, 1)
            @plate for i = eachindex(gg)
                b[i, 1:1] = 2.0 .* z[gg[i], :]
            end
            y .~ Normal.(b[:, 1], 1.0)
        end
        oracle(p) = sum(logpdf.(Normal(0, 1), p.z)) +
            sum(logpdf.(Normal.(2.0 .* p.z[code(gg, h), 1], 1.0), y))
        check(Expr(:block, :(gg = lab[pos]), cell.args...), cell,
            (; lab = ["a1", "a2", "b1", "b2"], pos, h, y), (; gg, h, y), oracle)
    end
end
