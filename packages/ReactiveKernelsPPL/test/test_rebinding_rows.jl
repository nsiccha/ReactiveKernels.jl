using DifferentiationInterface, Distributions, Enzyme, ReactiveKernels, ReactiveKernelsPPL, Test

# A built graph carries no bound row count: `build_kernel` on one binding,
# prepared with another binding of the same program that has other row
# counts, evaluates exactly as that binding's own build (snag
# rkppl-opaque-loc-37de7b80). Row counts the lowering resolves from data are
# read from the bound data inside the graph.

ReactiveKernels.@kernel _rr_scaled_cells(cells, a) = begin
    values = ReactiveKernels.plate(cells) do c
        a .* c
    end
    flat = reduce(vcat, values)
    return flat
end
_rr_shift(x, a) = x .+ a

const _RR_BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)

_rr_x(n) = collect(range(-1.0, 1.0; length = n))
_rr_y(n) = 0.3 .* _rr_x(n) .+ 0.05 .* sin.(1:n)
_rr_cells(lengths) = [[0.1k for k in 1:n] for n in lengths]
# Cell lengths with `n` entries in total.
_rr_lengths(n) = n == 9 ? [3, 2, 4] : [5, 1]

# Each case: the model, its binding at `n` rows, and the log likelihood at
# constrained values `nt` of that binding (an independent oracle).
function _rr_case(kind)
    if kind === :kernel_location
        m = @rkppl begin
            a ~ Normal(0, 1)
            s ~ Exponential(1)
            loc = _rr_scaled_cells(cells, a)
            y .~ Normal.(loc, s)
        end
        bind = n -> (c = _rr_cells(_rr_lengths(n));
            m(; cells = c) | (; y = 0.3 .* reduce(vcat, c) .+ 0.01))
        lik = (nt, n) -> (c = reduce(vcat, _rr_cells(_rr_lengths(n)));
            sum(logpdf.(Normal.(nt.a .* c, nt.s), 0.3 .* c .+ 0.01)))
    elseif kind === :function_location
        m = @rkppl begin
            a ~ Normal(0, 1)
            s ~ Exponential(1)
            loc = _rr_shift(x, a)
            y .~ Normal.(loc, s)
        end
        bind = n -> m(; x = _rr_x(n)) | (; y = _rr_y(n))
        lik = (nt, n) -> sum(logpdf.(Normal.(_rr_x(n) .+ nt.a, nt.s), _rr_y(n)))
    elseif kind === :scalar_location
        m = @rkppl begin
            a ~ Normal(0, 1)
            s ~ Exponential(1)
            y .~ Normal.(a, s)
        end
        bind = n -> m() | (; y = _rr_y(n))
        lik = (nt, n) -> sum(logpdf.(Normal(nt.a, nt.s), _rr_y(n)))
    elseif kind === :expression_location
        m = @rkppl begin
            a ~ Normal(0, 1)
            s ~ Exponential(1)
            l = exp(a)
            y .~ Normal.(l, s)
        end
        bind = n -> m() | (; y = _rr_y(n))
        lik = (nt, n) -> sum(logpdf.(Normal(exp(nt.a), nt.s), _rr_y(n)))
    elseif kind === :affine
        m = @rkppl begin
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            s ~ Exponential(1)
            mu = a .+ b .* x
            y .~ Normal.(mu, s)
        end
        bind = n -> m(; x = _rr_x(n)) | (; y = _rr_y(n))
        lik = (nt, n) -> sum(logpdf.(Normal.(nt.a .+ nt.b .* _rr_x(n), nt.s), _rr_y(n)))
    elseif kind === :scalar_offset
        m = @rkppl begin
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            s ~ Exponential(1)
            mu = a .+ off .+ b .* x
            y .~ Normal.(mu, s)
        end
        bind = n -> m(; x = _rr_x(n), off = 0.25) | (; y = _rr_y(n))
        lik = (nt, n) -> sum(logpdf.(Normal.(nt.a + 0.25 .+ nt.b .* _rr_x(n), nt.s), _rr_y(n)))
    elseif kind === :design
        m = @rkppl begin
            a ~ Normal(0, 1)
            s ~ Exponential(1)
            X = hcat(x1, x2)
            b[axes(X, 2)] .~ Normal.(0, 1)
            mu = a .+ X * b
            y .~ Normal.(mu, s)
        end
        bind = n -> m(; x1 = _rr_x(n), x2 = _rr_x(n) .^ 2) | (; y = _rr_y(n))
        lik = (nt, n) -> sum(logpdf.(Normal.(nt.a .+ hcat(_rr_x(n), _rr_x(n) .^ 2) * nt.b,
            nt.s), _rr_y(n)))
    elseif kind === :design_intercept_column
        m = @rkppl begin
            s ~ Exponential(1)
            X = hcat(ones(length(x1)), x1)
            b[axes(X, 2)] .~ Normal.(0, 1)
            mu = X * b
            y .~ Normal.(mu, s)
        end
        bind = n -> m(; x1 = _rr_x(n)) | (; y = _rr_y(n))
        lik = (nt, n) -> sum(logpdf.(Normal.(nt.b[1] .+ nt.b[2] .* _rr_x(n), nt.s), _rr_y(n)))
    elseif kind === :factor
        m = @rkppl begin
            a ~ Normal(0, 1)
            s ~ Exponential(1)
            c[levels(g)] .~ Normal.(0, 1)
            mu = a .+ c[g]
            y .~ Normal.(mu, s)
        end
        g = n -> [mod1(i, 3) for i in 1:n]
        bind = n -> m(; g = g(n)) | (; y = _rr_y(n))
        lik = (nt, n) -> sum(logpdf.(Normal.(nt.a .+ nt.c[g(n)], nt.s), _rr_y(n)))
    elseif kind === :log_rate
        m = @rkppl begin
            a ~ Normal(0, 1)
            y .~ Poisson.(exp.(a))
        end
        counts = n -> [mod(3i, 4) + (n == 6) for i in 1:n]
        bind = n -> m() | (; y = counts(n))
        lik = (nt, n) -> sum(logpdf.(Poisson(exp(nt.a)), counts(n)))
    elseif kind === :glm
        m = @rkppl begin
            X = hcat(x1, x2)
            alpha ~ Normal(0, 1)
            beta[axes(X, 2)] .~ Normal.(0, 1)
            y ~ NormalIDGLM(X, alpha, beta, 0.7)
        end
        bind = n -> m(; x1 = _rr_x(n), x2 = _rr_x(n) .^ 2) | (; y = _rr_y(n))
        lik = (nt, n) -> sum(logpdf.(Normal.(nt.alpha .+ hcat(_rr_x(n), _rr_x(n) .^ 2) * nt.beta,
            0.7), _rr_y(n)))
    elseif kind === :ranged
        m = @rkppl begin
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            s ~ Exponential(1)
            mu = a .+ b .* x
            y[eachindex(x)] .~ Normal.(mu, s)
        end
        bind = n -> m(; x = _rr_x(n)) | (; y = _rr_y(n))
        lik = (nt, n) -> sum(logpdf.(Normal.(nt.a .+ nt.b .* _rr_x(n), nt.s), _rr_y(n)))
    elseif kind === :missing
        m = @rkppl begin
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            s ~ Exponential(1)
            mu = a .+ b .* x
            y .~ Normal.(mu, s)
        end
        ym = n -> (v = Vector{Union{Missing,Float64}}(_rr_y(n)); v[2] = missing; v)
        bind = n -> m(; x = _rr_x(n)) | (; y = ym(n))
        lik = (nt, n) -> sum(logpdf(Normal(nt.a + nt.b * x, nt.s), y)
            for (x, y) in zip(_rr_x(n), ym(n)) if !ismissing(y))
    elseif kind === :kernel_location_and_scale
        # BRM's in-cell observation shape: both distribution arguments are
        # function-shaped kernel results with one entry per observation.
        m = @rkppl begin
            a ~ Normal(0, 1)
            s ~ Exponential(1)
            loc = _rr_scaled_cells(cells, a)
            sd = _rr_scaled_cells(cells, s)
            y .~ Normal.(loc, sd)
        end
        bind = n -> (c = _rr_cells(_rr_lengths(n));
            m(; cells = c) | (; y = 0.3 .* reduce(vcat, c) .+ 0.01))
        lik = (nt, n) -> (c = reduce(vcat, _rr_cells(_rr_lengths(n)));
            sum(logpdf.(Normal.(nt.a .* c, nt.s .* c), 0.3 .* c .+ 0.01)))
    elseif kind === :plate
        m = @rkppl begin
            a ~ Normal(0, 1)
            s ~ Exponential(1)
            @plate for i in eachindex(y)
                y[i] ~ Normal(a + x[i], s)
            end
        end
        bind = n -> m(; x = _rr_x(n)) | (; y = _rr_y(n))
        lik = (nt, n) -> sum(logpdf.(Normal.(nt.a .+ _rr_x(n), nt.s), _rr_y(n)))
    elseif kind === :ordinal
        m = @rkppl begin
            a ~ Normal(0, 1)
            thresholds[1:2] .~ Normal.(0, 1)
            y .~ Ordinal.(StoppingRatio(), LogitLink(), a .+ x, Ref(thresholds))
        end
        ys = n -> [mod1(i, 3) for i in 1:n]
        bind = n -> m(; x = _rr_x(n)) | (; y = ys(n))
        lik = (nt, n) -> sum(zip(_rr_x(n), ys(n)); init = 0.0) do (x, y)
            stop = cdf.(Logistic(), nt.thresholds .- nt.a .- x)
            log(prod(1 .- stop[1:y-1]; init = 1.0) * (y == 3 ? 1.0 : stop[y]))
        end
    elseif kind === :ragged
        m = @rkppl begin
            a ~ Normal(0, 1)
            s ~ Exponential(1)
            @plate for i in eachindex(y)
                y[i] .~ Normal.(a .+ x[i], s)
            end
        end
        xs = n -> [[0.2k * j for j in 1:mod(k, 3)] for k in 1:n]
        ys = n -> [0.3 .* x .+ 0.01 for x in xs(n)]
        bind = n -> m(; x = xs(n)) | (; y = ys(n))
        lik = (nt, n) -> sum((sum(logpdf.(Normal.(nt.a .+ x, nt.s), y))
            for (x, y) in zip(xs(n), ys(n))); init = 0.0)
    end
    return (; bind, lik)
end

# The generated program, private generated names aside.
_rr_source(plan) =
    ReactiveKernelsPPL._program_identity(kernel_expr(plan, assign_layout(plan)))

function _rr_check(kind; fit = 9, new = 6)
    case = _rr_case(kind)
    fitted, other = case.bind(fit), case.bind(new)
    built = build_kernel(fitted)
    # The generated program is the same for both bindings: no row count.
    @test ReactiveKernelsPPL._same_program(_rr_source(fitted), _rr_source(other))
    own = build_kernel(other)
    u = [0.15 - 0.1k for k in 1:built.layout.total]
    reused = prepare_sampler(built, other, u; backend = _RR_BACKEND)
    direct = prepare_sampler(own, other, u; backend = _RR_BACKEND)
    @test reused(u) == direct(u)
    gr, gd = similar(u), similar(u)
    vr, _ = sampler_value_and_gradient!(reused, gr, u)
    vd, _ = sampler_value_and_gradient!(direct, gd, u)
    @test vr == vd
    @test gr ≈ gd rtol = 1e-13
    nt = constrain(built.layout, u)
    lik = Base.invokelatest(prepare_query(built, other, :likelihood), u)
    @test lik ≈ case.lik(nt, new) rtol = 1e-12
    if kind === :ordinal
        points = Base.invokelatest(prepare_query(built, other, :pointwise), u).y
        @test length(points) == new
        @test sum(points) ≈ lik
    end
    # The original binding still evaluates on the same graph.
    @test Base.invokelatest(prepare_query(built, fitted, :likelihood), u) ≈
        case.lik(nt, fit) rtol = 1e-12
end

@testset "a built graph evaluates other row counts: response locations" begin
    for kind in (:kernel_location, :function_location, :scalar_location,
            :expression_location)
        @testset "$kind" begin
            _rr_check(kind)
            _rr_check(kind; fit = 6, new = 9)
        end
    end
end

@testset "a built graph evaluates other row counts: designs, offsets and families" begin
    for kind in (:affine, :scalar_offset, :design, :design_intercept_column,
            :factor, :log_rate, :glm, :ranged, :missing,
            :kernel_location_and_scale, :plate)
        @testset "$kind" begin
            _rr_check(kind)
            _rr_check(kind; fit = 6, new = 9)
        end
    end
end

@testset "a built graph rebinds empty observation domains in either direction" begin
    for kind in (:function_location, :scalar_location, :expression_location,
            :affine, :scalar_offset, :design, :design_intercept_column,
            :log_rate, :glm, :ranged, :plate, :ragged, :ordinal)
        @testset "$kind" begin
            _rr_check(kind; fit = 6, new = 0)
            _rr_check(kind; fit = 0, new = 6)
        end
    end
end

@testset "a build serves a binding with other level labels" begin
    u3 = [0.1, -0.2, 0.3, 0.2, -0.1]
    m = @rkppl begin
        a ~ Normal(0, 1)
        s ~ Exponential(1)
        c[levels(g)] .~ Normal.(0, 1)
        mu = a .+ c[g]
        y .~ Normal.(mu, s)
    end
    y = _rr_y(6)
    built = build_kernel(m(; g = [1, 2, 3, 1, 2, 3]) | (; y))
    relabeled = m(; g = [2, 3, 4, 2, 3, 4]) | (; y)
    # The graph reads the binding's labels, so on the labels 2, 3, 4 it evaluates
    # this model, not the build's (the wrong-model read of snag
    # rkppl-opaque-loc-37de7b80 needed the labels in the graph).
    own = build_kernel(relabeled)
    nt = constrain(own.layout, u3)
    expected = sum(logpdf.(Normal.(nt.a .+ nt.c[[1, 2, 3, 1, 2, 3]], nt.s), y))
    @test Base.invokelatest(prepare_query(own, relabeled, :likelihood), u3) ≈ expected rtol = 1e-12
    @test Base.invokelatest(prepare_query(built, relabeled, :likelihood), u3) ≈ expected rtol = 1e-12
    # Same labels in another order and count: the same program.
    regrouped = m(; g = [3, 1, 2, 2]) | (; y = y[1:4])
    @test Base.invokelatest(prepare_query(built, regrouped, :likelihood), u3) ≈
        sum(logpdf.(Normal.(nt.a .+ nt.c[[3, 1, 2, 2]], nt.s), y[1:4])) rtol = 1e-12

    scalar = @rkppl begin
        a ~ Normal(0, 1)
        s ~ Exponential(1)
        y .~ Normal.(a, s)
    end
    # refused: the graph's data arguments are typed by the build's data
    # (`Vector{Float64}`), so integer data generate another program
    @test_throws "in the data arguments" prepare_query(
        build_kernel(scalar() | (; y)), scalar() | (; y = [1, 2, 3]), :likelihood)
end

# A censored response with a bound scale per row partitions its rows by the
# guards it is written with (`sig > 0`, `y < lim`, `y <= lim`). With
# ReactiveKernels' `branch_arms = :all` (its `[ReactiveKernels]` preference;
# here the scope the preference sets) every arm is a plate, also one no row
# takes: one program and one sampler type serve rows that are all observed,
# all censored, a mix, or none (RK 0.1.118, snag `sampler-type-dif-ea945700`:
# ODL's soil rows all detected, or all below their detection limit, compiled
# the sampler again). By default a binding whose rows all take one arm
# prepares a program of its own.
@testset "censored rows: one sampler type whichever rows are censored" begin
    ReactiveKernels._with_branch_arms(:all) do
        m = @rkppl begin
            mu ~ Normal(0, 1)
            b ~ Normal(0, 1)
            y .~ censored.(Normal.(mu .+ b .* x, sig), lim, Inf)
        end
        x = [0.1, -0.3, 0.4, 0.2]
        sig = [0.7, 1.1, 0.9, 1.3]
        y = [0.5, -1.0, 1.2, 0.3]
        cases = (; mixed = [-5.0, -1.0, 0.0, -2.0], observed = fill(-5.0, 4),
                 censored = copy(y))
        u = [0.2, -0.1]
        density(nt, x, sig, y, lim) = logpdf(Normal(0, 1), nt.mu) + logpdf(Normal(0, 1), nt.b) +
            sum((yi <= li ? logcdf : logpdf)(Normal(nt.mu + nt.b * xi, si), yi)
                for (xi, si, yi, li) in zip(x, sig, y, lim); init = 0.0)
        gradient(nt, x, sig, y, lim) = sum(
            (yi <= li ? -exp(logpdf(Normal(), z) - logcdf(Normal(), z)) / si : z / si) .*
                [1.0, xi]
            for (xi, si, yi, li) in zip(x, sig, y, lim)
            for z in ((yi - nt.mu - nt.b * xi) / si,); init = [0.0, 0.0]) .-
            [nt.mu, nt.b]
        first_bound = m(; x, sig, lim = cases.mixed) | (; y)
        built = build_kernel(first_bound)
        sampler_type = typeof(prepare_query(built, first_bound, :sampler))
        bindings = [(m(; x, sig, lim) | (; y), (x, sig, y, lim)) for lim in values(cases)]
        push!(bindings, (m(; x = Float64[], sig = Float64[], lim = Float64[]) | (; y = Float64[]),
                         (Float64[], Float64[], Float64[], Float64[])))
        for (bound, data) in bindings
            own = build_kernel(bound)
            nt = constrain(own.layout, u)
            for kernel in (built, own)
                query = prepare_query(kernel, bound, :sampler)
                @test typeof(query) == sampler_type
                @test Base.invokelatest(query, u) ≈ density(nt, data...) rtol = 1e-12
                sampler = prepare_sampler(kernel, bound, u; backend = _RR_BACKEND)
                g = similar(u)
                value, _ = sampler_value_and_gradient!(sampler, g, u)
                @test value ≈ density(nt, data...) rtol = 1e-12
                @test g ≈ gradient(nt, data...) rtol = 1e-10
            end
        end
    end
    # By default a binding whose rows are all observed prepares its own program.
    m = @rkppl begin
        mu ~ Normal(0, 1)
        b ~ Normal(0, 1)
        y .~ censored.(Normal.(mu .+ b .* x, sig), lim, Inf)
    end
    x = [0.1, -0.3, 0.4, 0.2]; sig = [0.7, 1.1, 0.9, 1.3]; y = [0.5, -1.0, 1.2, 0.3]
    mixed = m(; x, sig, lim = [-5.0, -1.0, 0.0, -2.0]) | (; y)
    observed = m(; x, sig, lim = fill(-5.0, 4)) | (; y)
    built = build_kernel(mixed)
    @test typeof(prepare_query(built, observed, :sampler)) !=
          typeof(prepare_query(built, mixed, :sampler))
end
