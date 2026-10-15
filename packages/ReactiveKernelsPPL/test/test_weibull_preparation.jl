module WeibullPreparationTests
using ReactiveKernels, ReactiveKernelsPPL, DifferentiationInterface, Enzyme, Test
using Distributions: Normal, Weibull, logpdf
const RK = ReactiveKernels

const _sampled_shape = quote
    a ~ Normal(0, 1)
    b ~ Normal(0, 1)
    k ~ LogNormal(0.0, 0.3)
    eta = a .+ b .* x
    y .~ Weibull.(k, exp.(eta))
end
const _log_shape = quote
    a ~ Normal(0, 1)
    b ~ Normal(0, 1)
    log_k ~ Normal(0.0, 0.3)
    k = exp(log_k)
    eta = a .+ b .* x
    y .~ Weibull.(k, exp.(eta))
end

function _query(program, columns)
    bound = bind_data(lower_rkppl(program, keys(columns);
                                 conditioned = keys(columns)), columns)
    built = build_kernel(bound)
    kernel = prepare_query(built, bound, :sampler)
    (; bound, built, kernel)
end

# An equal source from an earlier preparation must not own the live plan's
# branch graphs. Let the earlier plan die before collecting between emissions.
@noinline function _discard_query(program, columns)
    _query(program, columns)
    nothing
end
_collect_pass(ast) = (GC.gc(); ast)

function _emission_shape(ast)
    heads = Dict{Symbol,Int}()
    function visit(ex)
        ex isa Expr || return
        heads[ex.head] = get(heads, ex.head, 0) + 1
        foreach(visit, ex.args)
    end
    visit(ast)
    heads
end

@testset "Weibull preparation survives collection of equal branch sources" begin
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    a, b, k = 0.5, -0.25, 1.8
    for (program, q) in ((_sampled_shape, (; a, b, k)),
                          (_log_shape, (; a, b, log_k = log(k))))
        shapes = Any[]
        for copies in (1, 3)
            x = repeat([0.5, -1.0, 1.5, 0.0, -0.5, 1.0], copies)
            y = repeat([0.7, 1.4, 2.6, 0.5, 1.0, 3.0], copies)
            columns = Dict{Symbol,AbstractVector}(:x => x, :y => y)
            enabled = GC.enable(false)
            state, lowered, native = try
                _discard_query(program, columns)
                state = _query(program, columns)
                lowered = RK._lowering_plan(state.kernel.plan)
                native = RK._lower_with_ops(lowered)
                (state, lowered, native)
            finally
                GC.enable(enabled)
            end
            GC.gc()
            tensorized = RK._lower_with_ops(lowered; tensorized = true)
            aligned = native[2] == tensorized[2]
            @test aligned
            @test RK._same_readable_recipes(native[3], tensorized[3])
            # Exercise the public preparation boundary with collection exactly
            # between its two emissions, including its strict alignment checks.
            kernel = Base.invokelatest(prepare, state.kernel.plan;
                                      passes = (_collect_pass,))
            push!(shapes, (_emission_shape(code_expr(kernel)), length(kernel.ops)))
            u = unconstrain(state.built.layout, q)
            pointwise = [logpdf(Weibull(k, exp(a + b * xi)), yi)
                         for (xi, yi) in zip(x, y)]
            expected = sum(pointwise) + logpdf(Normal(), a) + logpdf(Normal(), b) +
                       logpdf(Normal(0.0, 0.3), log(k))
            @test Base.invokelatest(kernel, u) ≈ expected rtol = 1e-12

            # Independent derivatives in unconstrained coordinates. The
            # LogNormal prior's Jacobian cancels its -log(k) term, so both
            # natural spellings describe this same mathematical target.
            log_ratio = log.(y) .- (a .+ b .* x)
            power = exp.(k .* log_ratio)
            eta_gradient = k .* (power .- 1)
            expected_gradient = [sum(eta_gradient) - a,
                sum(eta_gradient .* x) - b,
                sum(1 .+ k .* log_ratio .* (1 .- power)) - log(k) / 0.3^2]
            sampler = prepare_sampler(state.built, state.bound, u; backend)
            value, gradient = sampler_value_and_gradient!(sampler, similar(u), u)
            @test value ≈ expected rtol = 1e-12
            @test gradient ≈ expected_gradient rtol = 1e-10 atol = 1e-10
            retaining = prepare_sampler(state.built, state.bound, u;
                                        backend, retain = (:pointwise,))
            value, gradient, retained = sampler_value_gradient_and_retained!(
                retaining, similar(u), u)
            @test value ≈ expected rtol = 1e-12
            @test gradient ≈ expected_gradient rtol = 1e-10 atol = 1e-10
            @test retained.pointwise.y ≈ pointwise rtol = 1e-12
        end
        @test shapes[1] == shapes[2]
    end
end
end
