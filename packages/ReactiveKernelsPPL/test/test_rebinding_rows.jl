using DifferentiationInterface, Distributions, Enzyme, ReactiveKernels, ReactiveKernelsPPL, Test

# A built graph carries no bound row count: `build_kernel` on one binding,
# prepared with another binding of the same program that has other row
# counts, evaluates exactly as that binding's own build (snag
# rkppl-opaque-loc-37de7b80). Row counts the lowering resolves from data are
# read from the bound data inside the graph.

ReactiveKernels.@kernel _rr_scaled_cells(cells, a) = begin
    values = ReactiveKernels.plate(cells, Ref(a)) do c, a
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
    @test _rr_source(fitted) == _rr_source(other)
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

@testset "a build refuses a binding that generates another program" begin
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
    # refused: the graph holds the build's level labels (1, 2, 3); on the
    # labels 2, 3, 4 it would evaluate another model without an error
    # (snag rkppl-opaque-loc-37de7b80)
    @test_throws ContractValidationError prepare_query(built, relabeled, :likelihood)
    @test_throws "needs its own `build_kernel(plan)`" prepare_query(built, relabeled, :likelihood)
    own = build_kernel(relabeled)
    nt = constrain(own.layout, u3)
    @test Base.invokelatest(prepare_query(own, relabeled, :likelihood), u3) ≈
        sum(logpdf.(Normal.(nt.a .+ nt.c[[1, 2, 3, 1, 2, 3]], nt.s), y)) rtol = 1e-12
    # Same labels in another order and count: the same program.
    regrouped = m(; g = [3, 1, 2, 2]) | (; y = y[1:4])
    @test Base.invokelatest(prepare_query(built, regrouped, :likelihood), u3) ≈
        sum(logpdf.(Normal.(nt.a .+ nt.c[[3, 1, 2, 2]], nt.s), y[1:4])) rtol = 1e-12

    scalar = @rkppl begin
        a ~ Normal(0, 1)
        s ~ Exponential(1)
        y .~ Normal.(a, s)
    end
    empty = build_kernel(scalar() | (; y = Float64[]))
    # refused: an empty response's likelihood is the constant zero, so the
    # empty build evaluates no observation for a binding with rows (snag
    # rkppl-opaque-loc-37de7b80)
    @test_throws ContractValidationError prepare_query(empty, scalar() | (; y), :likelihood)
    # refused: the graph's data arguments are typed by the build's data
    # (`Vector{Float64}`), so integer data generate another program
    @test_throws "in the data arguments" prepare_query(
        build_kernel(scalar() | (; y)), scalar() | (; y = [1, 2, 3]), :likelihood)
end
