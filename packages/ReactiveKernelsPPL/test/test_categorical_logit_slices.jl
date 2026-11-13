# A categorical-logit response over the slices of one logit matrix,
# `y .~ CategoricalLogit.(eachrow(E))`: each row holds observation i's K−1
# non-reference logits (class 1 the zero reference), exactly as the
# positional `CategoricalLogit.(eta_2, …, eta_K)` reads its arguments, and
# K is the row length + 1 when the model runs. One lowered program, query
# type and sampler type therefore serve every class count (CONTRIBUTING.md
# "Runtime values and sizes"; snag `rkppl-categorica-6f3d4ba0`).
using Test, ReactiveKernels, ReactiveKernelsPPL, Distributions
using DifferentiationInterface, Enzyme
using LogExpFunctions: logsumexp

const _CLS_BACKEND = DifferentiationInterface.AutoEnzyme(; mode = Enzyme.Reverse)

_cls_q(built, bound, query, u) =
    Base.invokelatest(prepare_query(built, bound, query), u)

# Every class present, in an irregular order.
_cls_levels(n, K) = [mod1(i, K) for i in 1:n][sortperm([sin(2.3i) for i in 1:n])]
_cls_data(n, K) = (; y = _cls_levels(n, K),
    X = hcat(ones(n), [sin(1.3i) for i in 1:n]), Km1 = K - 1)

# Distributions.jl's categorical over the softmax of `[0; row]`.
_cls_oracle(row, y) =
    logpdf(Categorical(exp.(vcat(0.0, row) .- logsumexp(vcat(0.0, row)))), y)

_cls_bind(ast, data; conditioned = (:y,)) =
    bind_data(lower_rkppl(ast, data; conditioned), data)

function _cls_fd(f, u; h = 1e-6)
    map(eachindex(u)) do j
        plus, minus = copy(u), copy(u)
        plus[j] += h
        minus[j] -= h
        (f(plus) - f(minus)) / 2h
    end
end

const _CLS_ROWS = quote
    B[1:2, 1:Km1] .~ Normal.(0, 1)
    E = X * B
    y .~ CategoricalLogit.(eachrow(E))
end

@testset "categorical logit slices: one program for every class count" begin
    bounds = [_cls_bind(_CLS_ROWS, _cls_data(12, K)) for K in (3, 4, 6)]
    builds = build_kernel.(bounds)
    programs = [Base.remove_linenums!(kernel_expr(b, l.layout))
        for (b, l) in zip(bounds, builds)]
    @test allequal(programs)
    src = string(first(programs))
    @test occursin("eachrow(E)", src)
    @test occursin("categorical_logit_ref", src)
    @test !occursin(r"ifelse\(\w+ == \d", src)
    @test allequal(typeof(prepare_query(l, b, :sampler))
        for (b, l) in zip(bounds, builds))
    samplers = [prepare_sampler(l, b, fill(0.1, l.layout.total);
        backend = _CLS_BACKEND) for (b, l) in zip(bounds, builds)]
    @test allequal(typeof.(samplers))
    # The `@plate` cell over one row is one program for every K too.
    cell = quote
        B[1:2, 1:Km1] .~ Normal.(0, 1)
        E = X * B
        @plate for i in eachindex(y)
            y[i] ~ CategoricalLogit(E[i, :])
        end
    end
    cell_bounds = [_cls_bind(cell, _cls_data(12, K)) for K in (3, 5)]
    cell_builds = build_kernel.(cell_bounds)
    @test allequal(Base.remove_linenums!(kernel_expr(b, l.layout))
        for (b, l) in zip(cell_bounds, cell_builds))
    @test allequal(typeof(prepare_query(l, b, :sampler))
        for (b, l) in zip(cell_bounds, cell_builds))
    # One build serves another class count's binding.
    u = [0.1 * j for j in 1:builds[3].layout.total]
    nt = constrain(builds[3].layout, u)
    d = _cls_data(12, 6)
    E = d.X * nt.B
    @test _cls_q(builds[1], bounds[3], :likelihood, u) ≈
        sum(_cls_oracle(E[i, :], d.y[i]) for i in eachindex(d.y)) rtol = 1e-12
end

@testset "categorical logit slices: density and native Reverse gradient" begin
    for K in (2, 3, 5)
        d = _cls_data(15, K)
        bound = _cls_bind(_CLS_ROWS, d)
        built = build_kernel(bound)
        u = [0.3 * sin(2j) for j in 1:built.layout.total]
        nt = constrain(built.layout, u)
        E = d.X * nt.B
        expected = [_cls_oracle(E[i, :], d.y[i]) for i in eachindex(d.y)]
        @test _cls_q(built, bound, :likelihood, u) ≈ sum(expected) rtol = 1e-12
        @test collect(_cls_q(built, bound, :pointwise, u).y) ≈ expected rtol = 1e-12
        sampler = prepare_sampler(built, bound, u; backend = _CLS_BACKEND)
        value, gradient = sampler_value_and_gradient!(sampler, similar(u), u)
        @test value ≈ _cls_q(built, bound, :sampler, u) rtol = 1e-12
        @test gradient ≈ _cls_fd(v -> _cls_q(built, bound, :sampler, v), u) rtol = 1e-6 atol = 1e-7
    end
end

@testset "categorical logit slices: spellings of one model agree" begin
    d = _cls_data(10, 3)
    spellings = (
        positional = quote
            B[1:2, 1:2] .~ Normal.(0, 1)
            E = X * B
            y .~ CategoricalLogit.(E[:, 1], E[:, 2])
        end,
        rows = _CLS_ROWS,
        inline = quote
            B[1:2, 1:Km1] .~ Normal.(0, 1)
            y .~ CategoricalLogit.(eachrow(X * B))
        end,
        cols = quote
            B[1:2, 1:Km1] .~ Normal.(0, 1)
            Et = transpose(B) * transpose(X)
            y .~ CategoricalLogit.(eachcol(Et))
        end,
        cell = quote
            B[1:2, 1:Km1] .~ Normal.(0, 1)
            E = X * B
            @plate for i in eachindex(y)
                y[i] ~ CategoricalLogit(E[i, :])
            end
        end,
        cell_local = quote
            B[1:2, 1:Km1] .~ Normal.(0, 1)
            E = X * B
            @plate for i in eachindex(y)
                logits = E[i, :]
                y[i] ~ CategoricalLogit(logits)
            end
        end,
    )
    u = [0.2, -0.4, 0.5, 0.1]
    values = map(spellings) do ast
        bound = _cls_bind(ast, d)
        built = build_kernel(bound)
        sampler = prepare_sampler(built, bound, u; backend = _CLS_BACKEND)
        value, gradient = sampler_value_and_gradient!(sampler, similar(u), u)
        (collect(_cls_q(built, bound, :pointwise, u).y), value, gradient)
    end
    for name in (:rows, :inline, :cols, :cell, :cell_local)
        @test values[name][1] ≈ values.positional[1] rtol = 1e-12
        @test values[name][2] ≈ values.positional[2] rtol = 1e-12
        @test values[name][3] ≈ values.positional[3] rtol = 1e-10 atol = 1e-12
    end
end

@testset "categorical logit slices: declared and bound logit matrices" begin
    d = _cls_data(8, 4)
    declared = _cls_bind(quote
        L[axes(y, 1), 1:Km1] .~ Normal.(0, 1)
        y .~ CategoricalLogit.(eachrow(L))
    end, d)
    built = build_kernel(declared)
    u = [0.05 * j - 0.4 for j in 1:built.layout.total]
    L = constrain(built.layout, u).L
    @test size(L) == (8, 3)
    @test _cls_q(built, declared, :likelihood, u) ≈
        sum(_cls_oracle(L[i, :], d.y[i]) for i in eachindex(d.y)) rtol = 1e-12
    # Bound logits beside a sampled shift: an inline value over data.
    data = (; y = d.y, Ld = [0.3 * sin(i + 2j) for i in 1:8, j in 1:3])
    shifted = _cls_bind(quote
        a ~ Normal(0, 1)
        y .~ CategoricalLogit.(eachrow(Ld .+ a))
    end, data)
    built = build_kernel(shifted)
    @test _cls_q(built, shifted, :likelihood, [0.7]) ≈
        sum(_cls_oracle(data.Ld[i, :] .+ 0.7, data.y[i]) for i in 1:8) rtol = 1e-12
end

@testset "categorical logit slices: far tail, support and empty data" begin
    # Class 2 sits 800 logits below the others: the log-softmax stays finite.
    tail = (; y = [1, 2, 3], X = [1.0 0.0; 1.0 800.0; 1.0 0.0], Km1 = 2)
    bound = _cls_bind(_CLS_ROWS, tail)
    built = build_kernel(bound)
    pw = collect(_cls_q(built, bound, :pointwise, [0.0, -1.0, 0.0, 0.0]).y)
    @test pw[2] ≈ -800 - log(2) rtol = 1e-14
    @test pw[1] ≈ -log(3) rtol = 1e-14
    @test pw[3] ≈ -log(3) rtol = 1e-14
    # An observation beyond its slice is outside the law's support.
    beyond = (; y = [1, 2, 4], X = tail.X, Km1 = 2)
    bound = _cls_bind(_CLS_ROWS, beyond)
    built = build_kernel(bound)
    pw = collect(_cls_q(built, bound, :pointwise, [0.1, 0.2, 0.3, 0.4]).y)
    @test isfinite(pw[1]) && isfinite(pw[2]) && pw[3] == -Inf
    @test_throws ContractValidationError _cls_bind(_CLS_ROWS,
        (; y = [1, 0, 2], X = tail.X, Km1 = 2))
    # No observations: the plate runs no cells and the likelihood is zero.
    empty = (; y = Int[], X = zeros(0, 2), Km1 = 2)
    bound = _cls_bind(_CLS_ROWS, empty)
    built = build_kernel(bound)
    @test _cls_q(built, bound, :likelihood, [0.1, 0.2, 0.3, 0.4]) == 0.0
end

@testset "categorical logit slices: weights, censoring and missing responses" begin
    d = (; _cls_data(9, 3)..., w = [0.5 + mod(i, 3) / 2 for i in 1:9])
    u = [0.3, -0.2, 0.1, 0.4]
    weighted_rows = _cls_bind(quote
        B[1:2, 1:Km1] .~ Normal.(0, 1)
        E = X * B
        y .~ weighted.(CategoricalLogit.(eachrow(E)), w)
    end, d)
    built = build_kernel(weighted_rows)
    E = d.X * constrain(built.layout, u).B
    @test _cls_q(built, weighted_rows, :likelihood, u) ≈
        sum(d.w[i] * _cls_oracle(E[i, :], d.y[i]) for i in 1:9) rtol = 1e-12
    # Censoring reads the same class tails as the positional form: class 2
    # observations are left-censored at 2 and class 3 ones right-censored at 3.
    c = (; d..., y = [2 + mod(i, 2) for i in 1:9], lo = fill(2.0, 9),
        hi = fill(3.0, 9))
    censored_rows = _cls_bind(quote
        B[1:2, 1:Km1] .~ Normal.(0, 1)
        E = X * B
        y .~ censored.(CategoricalLogit.(eachrow(E)), lo, hi)
    end, c)
    censored_positional = _cls_bind(quote
        B[1:2, 1:2] .~ Normal.(0, 1)
        E = X * B
        y .~ censored.(CategoricalLogit.(E[:, 1], E[:, 2]), lo, hi)
    end, c)
    rows_built, positional_built = build_kernel.((censored_rows, censored_positional))
    @test collect(_cls_q(rows_built, censored_rows, :pointwise, u).y) ≈
        collect(_cls_q(positional_built, censored_positional, :pointwise, u).y) rtol = 1e-12
    # A missing response is omitted, as for every other family.
    m = (; y = Union{Int,Missing}[d.y[1:3]..., missing, d.y[5:9]...], X = d.X, Km1 = 2)
    bound = _cls_bind(_CLS_ROWS, m)
    built = build_kernel(bound)
    E = d.X * constrain(built.layout, u).B
    @test _cls_q(built, bound, :likelihood, u) ≈
        sum(_cls_oracle(E[i, :], d.y[i]) for i in (1:3..., 5:9...)) rtol = 1e-12
end

# JAX parity uses the helpers test_jax_source.jl loads (it runs earlier).
@testset "categorical logit slices: JAX module log density and gradient" begin
    if !(@isdefined(rk_jax_available) && @isdefined(jax_raw_data) && rk_jax_available())
        @info "RK_JAX_PYTHON or the JAX test helpers are missing: categorical logit slice JAX parity skipped"
        @test_skip false
    else
        d = (; _cls_data(9, 4)..., lo = fill(2.0, 9), hi = fill(3.0, 9))
        d = (; d..., y = [2 + mod(i, 2) for i in 1:9])
        models = (
            rows = _CLS_ROWS,
            cols = quote
                B[1:2, 1:Km1] .~ Normal.(0, 1)
                Et = transpose(B) * transpose(X)
                y .~ CategoricalLogit.(eachcol(Et))
            end,
            cell = quote
                B[1:2, 1:Km1] .~ Normal.(0, 1)
                E = X * B
                @plate for i in eachindex(y)
                    y[i] ~ CategoricalLogit(E[i, :])
                end
            end,
            censored = quote
                B[1:2, 1:Km1] .~ Normal.(0, 1)
                E = X * B
                y .~ censored.(CategoricalLogit.(eachrow(E)), lo, hi)
            end,
        )
        for (name, ast) in pairs(models)
            @testset "$name" begin
                bound = _cls_bind(ast, d)
                built = build_kernel(bound)
                src = jax_source(built, bound)
                u = [0.25 * sin(3j) for j in 1:built.layout.total]
                native = _cls_q(built, bound, :sampler, u)
                r = rk_jax_evaluate(src, "log_density", (u, jax_raw_data(bound, src)...);
                    grad = true, prepare = true)
                @test r.value ≈ native rtol = 1e-10
                reference = _cls_fd(v -> _cls_q(built, bound, :sampler, v), u)
                @test maximum(abs.(r.grad .- reference) ./ max.(1, abs.(reference))) < 1e-5
            end
        end
    end
end

@testset "categorical logit slices: refusals name the form" begin
    d = _cls_data(6, 3)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        B[1:2, 1:2] .~ Normal.(0, 1)
        E = X * B
        y .~ CategoricalLogit.(eachrow(E), E[:, 1])
    end, d; conditioned = (:y,))
    @test_throws SurfaceLoweringError lower_rkppl(quote
        y .~ CategoricalLogit.(eachrow(Q))
    end, d; conditioned = (:y,))
end
