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
    end
    return (; bind, lik)
end

_rr_source(plan) = kernel_expr(plan, assign_layout(plan))

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
