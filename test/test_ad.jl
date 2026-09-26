using DifferentiationInterface
import Enzyme
using InteractiveUtils: code_llvm
using ReactiveKernelsDistributionKernels.DistributionKernelSources: bernoulli

include("test_authored_plate_chains_ad.jl")

const TEST_AD_BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)

# Wide-splat regression closures, evaluated once at file scope (world-safe:
# no eval happens inside the testset body). Forty distinct one-input lanes
# plus one 40-input fused sum, mirroring the memo joint's 43-argument
# log-Jacobian (snag `joint-decl-memo-9642bb45`).
const _WIDE_SPLAT_LANES = [
    Core.eval(@__MODULE__, :(v -> sum(v) + $(Float64(i)))) for i in 1:40
]
const _WIDE_SPLAT_SUM = let names = [Symbol(:w_, i) for i in 1:40]
    Core.eval(@__MODULE__, Expr(:->, Expr(:tuple, names...),
        Expr(:call, :+, names...)))
end

_test_ad_value_gradient_allocated(prepared, gradient, q, data) =
    @allocated ad_value_and_gradient!(prepared, gradient, q, data)

_test_ad_backend_value_gradient_allocated(prepared, gradient, q, data) =
    @allocated DifferentiationInterface.value_and_gradient!(
        prepared.call, gradient, prepared.preparation, prepared.backend,
        q, DifferentiationInterface.Constant(data))

@testset "Prepared AD kernels" begin
    @testset "authored boundary, defaults, keywords, and fresh Constants" begin
        @kernel objective(q::Vector{Float64}, scale::Float64 = 1.25;
                          data::Vector{Float64}, offset::Float64 = 0.0) = begin
            density::Float64 =
                sum(q .* data) - scale * sum(abs2, q) + offset
        end

        data = [2.0, -1.0, 0.5]
        parameters = [0.3, -0.4, 0.2]
        expected = data .- 2(1.25) .* parameters

        @test ad_gradient(
            objective, TEST_AD_BACKEND, parameters;
            data, active = :q, want = :density,
        ) ≈ expected

        prepared = prepare_ad(
            objective, TEST_AD_BACKEND, parameters;
            data, active = objective.q, want = objective.density,
        )
        @test prepared isa PreparedADKernel
        @test inputs(prepared) == inputs(objective)
        @test outputs(prepared) == (objective.density,)
        @test code_expr(prepared) == code_expr(plan(objective; want = :density))
        @test occursin("active=:q", sprint(show, prepared))
        @test occursin("want=:density", sprint(show, prepared))

        # Preparation values are exemplars. Every call reconstructs Constants
        # from the current data/default/keyword values instead of retaining the
        # originals in the DI closure or preparation.
        for (newdata, newparameters, newscale, newoffset) in (
                ([1.0, 3.0, -2.0], [0.1, 0.2, -0.3], 0.75, 4.0),
                ([-0.5, 0.25, 4.0], [-0.2, 0.6, 0.4], 2.0, -3.0))
            @test ad_gradient(
                prepared, newparameters, newscale;
                data = newdata, offset = newoffset,
            ) ≈ newdata .- 2newscale .* newparameters
        end

        # The combined prepared boundary reuses the same active ordering and
        # freshly rebound Constant data, while writing into caller-owned
        # gradient storage.
        changed_data = [1.5, -2.0, 0.25]
        changed_parameters = [-0.1, 0.4, 0.8]
        changed_scale = 0.6
        changed_offset = -1.25
        destination = fill(NaN, length(changed_parameters))
        value, returned_gradient = ad_value_and_gradient!(
            prepared, destination, changed_parameters, changed_scale;
            data = changed_data, offset = changed_offset,
        )
        @test value ≈ sum(changed_parameters .* changed_data) -
              changed_scale * sum(abs2, changed_parameters) + changed_offset
        @test returned_gradient === destination
        @test destination ≈
              changed_data .- 2changed_scale .* changed_parameters

        @test_throws UndefKeywordError prepare_ad(
            objective, TEST_AD_BACKEND, parameters;
            data, active = :q,
        )
        @test_throws ArgumentError prepare_ad(
            objective, TEST_AD_BACKEND, parameters;
            data, active = :missing, want = :density,
        )
    end

    @testset "low-level positional boundary" begin
        graph = Graph()
        observed = value!(graph, :observed, Bool)
        logit = value!(graph, :logit, Float64)
        density = value!(graph, :density, Float64)
        add!(graph, (observed, logit) => density,
             (y, z) -> (y ? z : zero(z)) - log1p(exp(z)))
        kernel = prepare(graph; have = (observed, logit), want = density)

        gradient = ad_gradient(
            kernel, TEST_AD_BACKEND, true, 0.4; active = :logit)
        @test gradient ≈ 1 - inv(1 + exp(-0.4))

        prepared = prepare_ad(
            kernel, TEST_AD_BACKEND, true, 0.4; active = :logit)
        @test ad_gradient(prepared, false, -0.7) ≈
              -inv(1 + exp(0.7))
        @test_throws ArgumentError ad_gradient(
            kernel, TEST_AD_BACKEND, true, 0.4;
            active = :logit, unsupported = 1,
        )
    end

    @testset "wide fused closure differentiates past the 1.12 splat cliff" begin
        # Julia 1.12 inference refuses to unsplat a forwarded tuple of more
        # than 32 elements into a fixed-arity callee (snag
        # `joint-decl-memo-9642bb45`): the memo joint's 43-argument
        # log-Jacobian devolved to dynamic `jl_apply_generic` dispatch and
        # Enzyme aborted the gradient. `_kernel_source_call` forwards
        # positionally, so a 40-input fused closure differentiates on every
        # version. One vector HAVE keeps the top-level boundary narrow (the
        # memo shape); the width lives only in the fused op's inputs.
        graph = Graph()
        q = value!(graph, :q, Vector{Float64})
        lanes = [value!(graph, Symbol(:w_, i), Float64) for i in 1:40]
        for (lane, shift) in zip(lanes, _WIDE_SPLAT_LANES)
            add!(graph, (q,) => lane, shift)
        end
        total = value!(graph, :total, Float64)
        add!(graph, Tuple(lanes) => total,
            ReactiveKernels._KernelSourceOp(
                Val(:wide_splat_regression), Val(:fused), _WIDE_SPLAT_SUM))
        kernel = prepare(graph; have = (q,), want = total)
        # 40 lanes plus the wide op, all surviving preparation: a future
        # pass must not silently shrink the fused call below the cliff.
        @test length(kernel.ops) == 41
        point = fill(0.25, 40)
        @test kernel(point) ≈ 0.25 * 40^2 + 40 * 41 / 2
        # Static-dispatch pin (fast; the memo e2e covers Enzyme end to
        # end): pre-fix, 1.12 emitted `jl_apply_generic` for this call and
        # Enzyme aborted the gradient.
        op = only(
            o for o in kernel.ops if o isa ReactiveKernels._KernelSourceOp)
        llvm = sprint(code_llvm, ReactiveKernels._kernel_source_call,
            Tuple{Val{:native},typeof(op),ntuple(_ -> Float64, 40)...})
        @test !occursin("jl_apply_generic", llvm)
    end

    @testset "multiple active ports share one structured reverse pass" begin
        @kernel multi_active_objective(
                alpha::Vector{Float64}, beta::Vector{Float64},
                data::Vector{Float64}; offset::Float64 = 0.0) = begin
            objective::Float64 =
                sum(abs2, alpha) + sum(beta .* data) + offset
        end

        alpha = [0.3, -0.4, 0.2]
        beta = [-0.1, 0.7, 0.5]
        data = [2.0, -1.0, 0.5]
        expected = (copy(data), 2 .* alpha)

        # The selector order, not authored HAVE order, defines the structured
        # point and returned cotangent order.
        one_shot = ad_gradient(
            multi_active_objective, TEST_AD_BACKEND, alpha, beta, data;
            active = (:beta, :alpha), want = :objective,
        )
        @test one_shot isa Tuple
        @test one_shot[1] ≈ expected[1]
        @test one_shot[2] ≈ expected[2]

        prepared = prepare_ad(
            multi_active_objective, TEST_AD_BACKEND, alpha, beta, data;
            active = (multi_active_objective.beta,
                      multi_active_objective.alpha),
            want = :objective,
        )
        @test typeof(prepared).parameters[1] == (2, 1)
        @test occursin("active=(:beta, :alpha)", sprint(show, prepared))

        changed_alpha = [-0.2, 0.6, 0.4]
        changed_beta = [0.8, -0.3, 0.1]
        changed_data = [-0.5, 0.25, 3.0]
        value, gradient = ad_value_and_gradient(
            prepared, changed_alpha, changed_beta, changed_data; offset = 1.5)
        @test value ≈ sum(abs2, changed_alpha) +
              sum(changed_beta .* changed_data) + 1.5
        @test gradient[1] ≈ changed_data
        @test gradient[2] ≈ 2 .* changed_alpha

        destinations = (similar(changed_beta), similar(changed_alpha))
        inplace_value, returned = ad_value_and_gradient!(
            prepared, destinations,
            changed_alpha, changed_beta, changed_data; offset = 1.5)
        @test inplace_value ≈ value
        @test returned === destinations
        @test destinations[1] ≈ gradient[1]
        @test destinations[2] ≈ gradient[2]

        # An explicit one-element tuple deliberately preserves a one-tuple
        # result, while the historical scalar selector stays unwrapped.
        singleton = ad_gradient(
            multi_active_objective, TEST_AD_BACKEND, alpha, beta, data;
            active = (:alpha,), want = :objective)
        @test singleton isa Tuple
        @test only(singleton) ≈ 2 .* alpha
        @test ad_gradient(
            multi_active_objective, TEST_AD_BACKEND, alpha, beta, data;
            active = :alpha, want = :objective) ≈ 2 .* alpha

        @test_throws ArgumentError prepare_ad(
            multi_active_objective, TEST_AD_BACKEND, alpha, beta, data;
            active = (), want = :objective)
        @test_throws ArgumentError prepare_ad(
            multi_active_objective, TEST_AD_BACKEND, alpha, beta, data;
            active = (:alpha, :alpha), want = :objective)

        # Heterogeneous active storage (the GLM beta+dispersion shape) is
        # normalized internally without changing the public scalar cotangent.
        @kernel mixed_active_objective(
                coefficients::Vector{Float64}, dispersion::Float64) = begin
            objective::Float64 =
                sum(abs2, coefficients) + dispersion^2
        end
        coefficients = [0.2, -0.5]
        dispersion = 1.7
        mixed = prepare_ad(
            mixed_active_objective, TEST_AD_BACKEND,
            coefficients, dispersion;
            active = (:coefficients, :dispersion), want = :objective)
        mixed_value, mixed_gradient = ad_value_and_gradient(
            mixed, coefficients, dispersion)
        @test mixed_value ≈ sum(abs2, coefficients) + dispersion^2
        @test mixed_gradient[1] ≈ 2 .* coefficients
        @test mixed_gradient[2] ≈ 2dispersion
        mixed_destination = (similar(coefficients), Ref(NaN))
        _, mixed_returned = ad_value_and_gradient!(
            mixed, mixed_destination, coefficients, dispersion)
        @test mixed_returned === mixed_destination
        @test mixed_destination[1] ≈ mixed_gradient[1]
        @test mixed_destination[2][] ≈ mixed_gradient[2]

        # Pullback preparation shares the same ordered multi-active point and
        # restores scalar cotangents at the public boundary.
        mixed_pullback = prepare_ad_pullback(
            mixed_active_objective, TEST_AD_BACKEND, 1.0,
            coefficients, dispersion;
            active = (:coefficients, :dispersion), want = :objective)
        pullback_value, mixed_cotangent = ad_value_and_pullback(
            mixed_pullback, 1.0, coefficients, dispersion)
        @test pullback_value ≈ mixed_value
        @test mixed_cotangent[1] ≈ mixed_gradient[1]
        @test mixed_cotangent[2] ≈ mixed_gradient[2]
        cotangent_destination = (similar(coefficients), Ref(NaN))
        _, returned_cotangent = ad_value_and_pullback!(
            mixed_pullback, cotangent_destination, 1.0,
            coefficients, dispersion)
        @test returned_cotangent === cotangent_destination
        @test cotangent_destination[1] ≈ mixed_gradient[1]
        @test cotangent_destination[2][] ≈ mixed_gradient[2]
    end

    @testset "active dependency boundary" begin
        function boundary_kernel(kind)
            graph = Graph()
            q = value!(graph, :unconstrained, Vector{Float64})
            middle = value!(graph, :middle, Vector{Float64})
            buffer = value!(graph, :derived_buffer, Vector{Float64})
            density = value!(graph, :density, Float64)
            add!(graph, q => middle, copy)
            if kind === :direct
                add!(graph, q => buffer, copy)
            else
                add!(graph, middle => buffer, copy)
            end
            add!(graph, buffer => density, sum)
            prepare(graph; have = (q, buffer), want = density)
        end

        for kind in (:direct, :transitive)
            error = try
                prepare_ad(
                    boundary_kernel(kind), TEST_AD_BACKEND,
                    [1.0, 2.0], [1.0, 2.0]; active = :unconstrained,
                )
                nothing
            catch exception
                exception
            end
            @test error isa ArgumentError
            @test occursin(":derived_buffer", sprint(showerror, error))
            @test occursin("transitively downstream", sprint(showerror, error))
            @test occursin("Constant", sprint(showerror, error))
        end

        # A selected active port may itself be graph-derived from an inactive
        # root: the inactive root is upstream, so treating its current boundary
        # value as Constant severs no active dependency.
        graph = Graph()
        root = value!(graph, :root, Float64)
        active = value!(graph, :active_derived, Float64)
        density = value!(graph, :density, Float64)
        add!(graph, root => active, exp)
        add!(graph, (root, active) => density, (r, q) -> r + q^2)
        kernel = prepare(graph; have = (root, active), want = density)
        @test ad_gradient(
            kernel, TEST_AD_BACKEND, 3.0, 2.0;
            active = :active_derived,
        ) ≈ 4.0
    end

    @testset "structured active storage and prepared reverse pullbacks" begin
        parameter_type = NamedTuple{
            (:location, :effects),Tuple{Float64,Vector{Float64}}}
        graph = Graph()
        parameters = value!(graph, :parameters, parameter_type)
        scale = value!(graph, :scale, Float64)
        objective = value!(graph, :objective, Float64)
        add!(graph, (parameters, scale) => objective,
             (p, s) -> p.location^2 + s * sum(abs2, p.effects))
        structured = prepare(
            graph; have = (parameters, scale), want = objective)

        point = (; location = 1.5, effects = [0.25, -0.5, 0.75])
        gradient = ad_gradient(
            structured, TEST_AD_BACKEND, point, 2.0;
            active = :parameters)
        @test gradient isa NamedTuple
        @test keys(gradient) == keys(point)
        @test gradient.location ≈ 3.0
        @test gradient.effects ≈ 4 .* point.effects

        prepared_gradient = prepare_ad(
            structured, TEST_AD_BACKEND, point, 2.0;
            active = :parameters)
        changed = (; location = -0.75, effects = [1.0, -2.0, 0.5])
        changed_gradient = ad_gradient(prepared_gradient, changed, 0.25)
        @test changed_gradient.location ≈ -1.5
        @test changed_gradient.effects ≈ 0.5 .* changed.effects
        changed_value, combined_gradient = ad_value_and_gradient(
            prepared_gradient, changed, 0.25)
        @test changed_value ≈
            changed.location^2 + 0.25 * sum(abs2, changed.effects)
        @test combined_gradient == changed_gradient
        @test code_expr(prepared_gradient) === code_expr(structured)

        vector_graph = Graph()
        x = value!(vector_graph, :x, Vector{Float64})
        data = value!(vector_graph, :data, Vector{Float64})
        pointwise = value!(vector_graph, :pointwise, Vector{Float64})
        add!(vector_graph, (x, data) => pointwise,
             (q, d) -> q .^ 2 .+ q .* d)
        vector_kernel = prepare(
            vector_graph; have = (x, data), want = pointwise)
        q = [0.2, -0.4, 0.7]
        observed = [1.5, -0.5, 0.25]
        seed = [1.0, -2.0, 0.5]
        expected = seed .* (2 .* q .+ observed)

        @test_throws ArgumentError prepare_ad(
            vector_kernel, TEST_AD_BACKEND, q, observed; active = :x)
        @test ad_pullback(
            vector_kernel, TEST_AD_BACKEND, seed, q, observed;
            active = :x) ≈ expected

        prepared_pullback = prepare_ad_pullback(
            vector_kernel, TEST_AD_BACKEND, seed, q, observed;
            active = :x)
        @test prepared_pullback isa PreparedADPullback
        @test inputs(prepared_pullback) == inputs(vector_kernel)
        @test outputs(prepared_pullback) == outputs(vector_kernel)
        @test code_expr(prepared_pullback) === code_expr(vector_kernel)
        @test occursin("active=:x", sprint(show, prepared_pullback))
        @test occursin("want=:pointwise", sprint(show, prepared_pullback))
        @test ad_pullback(prepared_pullback, seed, q, observed) ≈ expected

        q2 = [-0.1, 0.8, 0.4]
        observed2 = [0.3, -1.0, 2.0]
        seed2 = [-0.5, 1.25, 2.0]
        value2, pullback2 = ad_value_and_pullback(
            prepared_pullback, seed2, q2, observed2)
        @test value2 ≈ q2 .^ 2 .+ q2 .* observed2
        @test pullback2 ≈ seed2 .* (2 .* q2 .+ observed2)

        destination = fill(NaN, length(q2))
        value3, returned = ad_value_and_pullback!(
            prepared_pullback, destination, seed2, q2, observed2)
        @test value3 ≈ value2
        @test returned === destination
        @test destination ≈ pullback2
    end

    @testset "invalid and aliased boundaries fail loudly" begin
        graph = Graph()
        x = value!(graph, :x, Float64)
        y = value!(graph, :y, Float64)
        z = value!(graph, :z, Float64)
        add!(graph, x => (y, z), t -> (t, t^2))
        multi = prepare(graph; have = x, want = (y, z))
        @test_throws ArgumentError prepare_ad(
            multi, TEST_AD_BACKEND, 1.0; active = :x)

        int_graph = Graph()
        count = value!(int_graph, :count, Int)
        integer_objective = value!(int_graph, :objective, Float64)
        add!(int_graph, count => integer_objective, Float64)
        @test_throws ArgumentError prepare_ad(
            prepare(int_graph; have = count, want = integer_objective),
            TEST_AD_BACKEND, 2; active = :count,
        )

        alias_graph = Graph()
        q = value!(alias_graph, :q, Float64)
        q_alias = value!(alias_graph, :q_alias, Float64)
        density = value!(alias_graph, :density, Float64)
        alias_graph.aliases[q_alias.id] = q.id
        add!(alias_graph, q => density, abs2)
        alias_spec = KernelSpec(
            alias_graph,
            Dict{Symbol,Value}(
                :q => q, :q_alias => q_alias, :density => density,
            ),
            [:q, :q_alias, :density], [:q, :q_alias], [:density],
        )
        alias_error = try
            prepare_ad(
                alias_spec, TEST_AD_BACKEND, 1.0, 1.0;
                active = :q, want = :density,
            )
            nothing
        catch exception
            exception
        end
        @test alias_error isa ArgumentError
        @test occursin(":q aliases :q_alias", sprint(showerror, alias_error))

        @kernel multiple_wants(q::Float64) = begin
            first::Float64 = q^2
            second::Float64 = q^3
        end
        @test ad_gradient(
            multiple_wants, TEST_AD_BACKEND, 2.0;
            active = :q, want = :second,
        ) ≈ 12.0
        @test_throws ArgumentError prepare_ad(
            multiple_wants, TEST_AD_BACKEND, 2.0;
            active = :q, want = (:first, :second),
        )

        # Type annotations are optional authoring metadata. An abstractly typed
        # output is validated against the exemplar result at preparation.
        @kernel untyped_square(q) = q^2
        @test ad_gradient(
            untyped_square, TEST_AD_BACKEND, 3.0;
            active = :q, want = :untyped_square,
        ) ≈ 6.0
    end

    @testset "authored plate AD reuses the inspectable primal body" begin
        @kernel plate_objective(q::Vector{Float64},
                                data::Vector{Float64}) = begin
            pointwise = plate(q, data) do qi, di
                score::Float64 = qi * di - 0.5 * qi^2
                return score
            end
            objective::Float64 = sum(pointwise)
        end

        q = [0.2, -0.4, 0.7, 0.1]
        data = [1.5, -0.5, 0.25, 2.0]
        kernel = prepare(plate_objective;
            have = (:q, :data), want = :objective)
        primal_ast = code_expr(kernel)
        @test occursin("Base.Broadcast.preprocess", string(primal_ast))
        @test !occursin("_plate_dependency_changed", string(primal_ast))

        prepared = prepare_ad(
            kernel, TEST_AD_BACKEND, q, data; active = :q)
        ad_text = string(code_expr(prepared))
        @test prepared.call isa ReactiveKernels._ADNativeKernelCall
        @test prepared.call.native === kernel.f.native
        @test prepared.call.ops === kernel.ops
        @test code_expr(prepared) === primal_ast
        @test occursin("Base.Broadcast.preprocess", ad_text)
        @test !occursin("_plate_dependency_changed", ad_text)

        gradient = similar(q)
        value, returned = ad_value_and_gradient!(
            prepared, gradient, q, data)
        @test value ≈ sum(q .* data .- 0.5 .* q .^ 2)
        @test returned === gradient
        @test gradient ≈ data .- q
        _test_ad_value_gradient_allocated(prepared, gradient, q, data)
        _test_ad_backend_value_gradient_allocated(
            prepared, gradient, q, data)
        rk_allocated = _test_ad_value_gradient_allocated(
            prepared, gradient, q, data)
        backend_allocated = _test_ad_backend_value_gradient_allocated(
            prepared, gradient, q, data)
        # Julia 1.12's DI/Enzyme path currently performs bounded backend heap
        # work. The reusable RK wrapper must not add allocations beyond that
        # direct backend call; on runtimes where the backend is allocation-free,
        # this remains the original exact zero-allocation sentinel.
        @test rk_allocated <= backend_allocated
    end

    @testset "authored plate AD: Int data-axis in a capture-recapture plate" begin
        # A likelihood plate whose axis is a data-only `Vector{Int}` sitting
        # beside a `Vector{Float64}` data array and several parameter-derived
        # scalars is the posteriordb capture-recapture / GLMM shape (M0, Mh,
        # GLMM_Poisson, seeds*). The pointwise buffer's marker used to be selected
        # by a runtime `findfirst` over the whole plate argument tuple; with the
        # Int axis inactive and the Float array a differentiation `Constant`, that
        # search stays CONDITIONALLY active for a plate wide enough that it does
        # not fold to a constant index, so plain reverse-mode Enzyme (no
        # `set_runtime_activity`, the standing config above) rejected the whole
        # gradient with an `EnzymeRuntimeActivityError`. The lowering now binds the
        # axis statically, so the derivative lowers. This width matters: a narrow
        # two- or three-argument plate folds the search away and does NOT
        # reproduce the bug, so the test mirrors the real M0 argument count.
        @kernel capture_recapture_like(u::Vector{Float64}, s::Vector{Int},
                                       lchoose::Vector{Float64}, T::Int) = begin
            u_omega::Float64 = sum(view(u, 1:1))
            u_p::Float64 = sum(view(u, 2:2))
            log_omega::Float64 = -log1p(exp(-u_omega))
            log1m_omega::Float64 = -log1p(exp(u_omega))
            logp::Float64 = -log1p(exp(-u_p))
            log1mp::Float64 = -log1p(exp(u_p))
            pointwise = plate(s, lchoose, log_omega, log1m_omega, logp, log1mp,
                              T) do si, lc, lo, l1o, lp, l1p, TT
                observed = lo + lc + si * lp + (TT - si) * l1p
                unobserved = l1o + lo
                ifelse(si > 0, observed, unobserved)
            end
            objective::Float64 = sum(pointwise)
        end

        s = [1, 0, 2, 0, 1]
        lchoose = [0.0, 0.0, 0.7, 0.0, 0.0]
        T = 3
        u = [0.1, -0.2]

        # Plain-Julia reference of the same density, differentiated by the same
        # backend on a closure that never routes through the authored-plate marker
        # path — so it also confirms the recovered gradient is correct, not merely
        # non-throwing.
        reference = let s = s, lchoose = lchoose, T = T
            function (u)
                u_omega, u_p = u[1], u[2]
                log_omega = -log1p(exp(-u_omega))
                log1m_omega = -log1p(exp(u_omega))
                logp = -log1p(exp(-u_p))
                log1mp = -log1p(exp(u_p))
                acc = zero(eltype(u))
                for i in eachindex(s)
                    observed = log_omega + lchoose[i] + s[i] * logp +
                               (T - s[i]) * log1mp
                    unobserved = log1m_omega + log_omega
                    acc += ifelse(s[i] > 0, observed, unobserved)
                end
                acc
            end
        end
        ref_value, ref_grad = DifferentiationInterface.value_and_gradient(
            reference, TEST_AD_BACKEND, u)

        kernel = prepare(capture_recapture_like;
                         have = (:u, :s, :lchoose, :T), want = :objective,
                         bound = (; s = s, lchoose = lchoose, T = T))
        prepared = prepare_ad(kernel, TEST_AD_BACKEND, u; active = :u)

        gradient = similar(u)
        value, returned = ad_value_and_gradient!(prepared, gradient, u)
        @test value ≈ ref_value
        @test returned === gradient
        @test gradient ≈ ref_grad
    end

    @testset "bound matrix views externalize as owning copies" begin
        # Likelihood plates over bind-time `view(y, :, j)` columns are the
        # posteriordb lotka-volterra shape: observation columns beside an
        # active trajectory. A prebuilt `SubArray` hidden operand defeats plain
        # reverse-mode Enzyme static activity analysis (the derivative unboxes
        # the parent pointer into an active slot), so preparation materializes
        # each bound view as an owning copy with identical contents; the
        # derivative then lowers exactly like the runtime-view form.
        @kernel twin_column_like(x::Vector{Float64}, y::Matrix{Float64}) = begin
            c1 = view(y, :, 1)
            c2 = view(y, :, 2)
            m::Vector{Float64} = exp.(x)
            first_pointwise = plate(c1, m) do observation, mean
                observation * mean
            end
            second_pointwise = plate(c2, m) do observation, mean
                observation * mean
            end
            objective::Float64 = sum(first_pointwise) + sum(second_pointwise)
        end

        x = [0.3, -0.4, 0.2, 0.1, -0.2]
        y = [1.0 6.0; 2.0 7.0; 3.0 8.0; 4.0 9.0; 5.0 10.0]
        # Explicit `Constant` activity: a `let`-captured matrix defeats
        # Enzyme's readonly analysis on Julia 1.12
        # (`EnzymeMutabilityException`) while passing on 1.10; the annotated
        # form differentiates the identical math on both.
        reference(x, y) = (m = exp.(x);
            sum(view(y, :, 1) .* m) + sum(view(y, :, 2) .* m))
        ref_value, ref_grad = DifferentiationInterface.value_and_gradient(
            reference, TEST_AD_BACKEND, x, DifferentiationInterface.Constant(y))

        kernel = prepare(twin_column_like;
                         have = (:x, :y), want = :objective,
                         bound = (; y = y))
        prepared = prepare_ad(kernel, TEST_AD_BACKEND, x; active = :x)
        @test all(value -> value isa Array, prepared.external_values)
        @test prepared.external_values == (y[:, 1], y[:, 2])

        gradient = similar(x)
        value, returned = ad_value_and_gradient!(prepared, gradient, x)
        @test value ≈ ref_value
        @test returned === gradient
        @test gradient ≈ ref_grad
    end
end

if !isdefined(@__MODULE__, :AuthoredScanFixtures)
    include(joinpath(@__DIR__, "fixtures", "authored_scan.jl"))
end

@testset "authored scan plain reverse AD" begin
    spec = AuthoredScanFixtures.authored_scan_arma
    reference(q, series) = -0.5 * sum(abs2,
        AuthoredScanFixtures._authored_scan_reference(q, series))
    for series in ([0.5], sin.(1:20)), bound in ((;), (; series)),
            want in (:total, :joint)
        q = [0.2, 0.7, -0.3]
        # :joint has a second scan consumer, exercising the materialized path.
        sign = want === :total ? 1.0 : -1.0
        k = prepare(spec; bound, want)
        args = isempty(bound) ? (q, series) : (q,)
        prepared = prepare_ad(k, TEST_AD_BACKEND, args...; active = :q)
        gradient = zeros(3)
        value, returned = ad_value_and_gradient!(prepared, gradient, args...)
        @test value ≈ sign * reference(q, series)
        @test returned === gradient
        expected = map(eachindex(q)) do i
            left, right = copy(q), copy(q)
            left[i] -= 1e-5
            right[i] += 1e-5
            (reference(right, series) - reference(left, series)) / 2e-5
        end
        @test gradient ≈ sign .* expected rtol = 1e-8 atol = 1e-8
    end
end

@testset "authored scan lockstep reverse AD" begin
    spec = AuthoredScanFixtures.authored_scan_lockstep
    reference(a, b) = -0.5 * sum(abs2,
        AuthoredScanFixtures._authored_scan_lockstep_reference(a, b))
    a0, b0 = [0.9, 0.8, 0.5, -0.2], [1.0, -0.5, 0.2, 0.7]
    fd(f, x0) = map(eachindex(x0)) do i
        l, r = copy(x0), copy(x0)
        l[i] -= 1e-5
        r[i] += 1e-5
        (f(r) - f(l)) / 2e-5
    end
    # Reverse AD runs through the native inlined lockstep loop; differentiate w.r.t.
    # each of the two co-varying sequences, with the other bound.
    for active in (:a, :b)
        bound = active === :a ? (; b = b0) : (; a = a0)
        x0 = active === :a ? a0 : b0
        objective = active === :a ? (x -> reference(x, b0)) : (x -> reference(a0, x))
        k = prepare(spec; bound, want = :total)
        prepared = prepare_ad(k, TEST_AD_BACKEND, x0; active)
        gradient = zeros(length(x0))
        value, returned = ad_value_and_gradient!(prepared, gradient, x0)
        @test value ≈ reference(a0, b0)
        @test returned === gradient
        @test gradient ≈ fd(objective, x0) rtol = 1e-6 atol = 1e-8
    end
end

@testset "scan AD preserves operation-table source transforms" begin
    spec = AuthoredScanFixtures.authored_scan_arma
    q, series = [0.2, 0.7, -0.3], [0.5]
    original = prepare(spec; bound = (; series), want = :total)
    slot = findfirst(op -> op isa ReactiveKernels._AuthoredScanOp, original.ops)
    # A literal slot and an arbitrary table use must both retain the scan
    # metadata when a caller's source transform actually consumes it.
    for table in (:__ops__, :(identity(__ops__)))
        function require_scan_metadata(ast)
            insert!(ast.args[2].args, 1,
                :(@assert $table[$slot] isa ReactiveKernels._AuthoredScanOp))
            ast
        end
        kernel = prepare(spec; bound = (; series), want = :total,
                         passes = (require_scan_metadata,))
        # Editing display metadata cannot change what the compiled body uses.
        empty!(code_expr(kernel).args[2].args)
        prepared = prepare_ad(kernel, TEST_AD_BACKEND, q; active = :q)
        @test kernel(q) == original(q)
        @test prepared.call(q, prepared.external_values...) == kernel(q)
    end
end

# A direct-p `bernoulli` HAVE at an exact boundary (p = 1 & observed = true,
# p = 0 & observed = false) previously yielded a NaN reverse gradient: the p HAVE
# was routed through `logit = log(p) - log1p(-p)` whose derivative is ±Inf there,
# so `0 · ±Inf = NaN` even though the primal is finite. The boundary-safe route
# computes the log-probability directly from `p`; the logit HAVE keeps its stable
# log-sum-exp gradient. (The Reactant analog lives in `test/test_ad_reactant.jl`.)
@testset "bernoulli direct-p boundary gradients are finite" begin
    p_kernel = prepare(bernoulli.logpdf; have = (:observed, :p), want = :logpdf)
    logit_kernel = prepare(bernoulli.logpdf; have = (:observed, :logit), want = :logpdf)
    grad_p(observed, p) = ad_gradient(p_kernel, TEST_AD_BACKEND, observed, p; active = :p)
    grad_logit(observed, logit) =
        ad_gradient(logit_kernel, TEST_AD_BACKEND, observed, logit; active = :logit)

    # Exact boundary: finite (was NaN) and equal to d/dp of the selected log-prob.
    @test grad_p(true, 1.0) == 1.0     # d/dp log(p)      at p = 1
    @test grad_p(false, 0.0) == -1.0   # d/dp log1p(-p)   at p = 0

    # Interior direct-p gradients.
    for p in (0.1, 0.37, 0.6, 0.92)
        @test grad_p(true, p) ≈ 1 / p
        @test grad_p(false, p) ≈ -1 / (1 - p)
        @test isfinite(grad_p(true, p))
        @test isfinite(grad_p(false, p))
    end

    # logit HAVE keeps the stable log-sum-exp gradient (sigmoid), finite at
    # saturating tails.
    for logit in (-30.0, -20.0, 0.7, 20.0, 30.0)
        @test grad_logit(true, logit) ≈ 1 / (1 + exp(logit))     # sigmoid(-logit)
        @test grad_logit(false, logit) ≈ -1 / (1 + exp(-logit))  # -sigmoid(logit)
        @test isfinite(grad_logit(true, logit))
        @test isfinite(grad_logit(false, logit))
    end
end

# A backend value alone does not load DifferentiationInterface's backend
# extension: without `using Enzyme`, `prepare_ad` died inside
# DifferentiationInterface with a bare `MethodError` on an internal
# `_prepare_pullback_aux` symbol. Every entry point that hands a caller
# backend to DifferentiationInterface now names the missing `using` instead.
struct AutoNotLoadedTestPackage <: DifferentiationInterface.AbstractADType end

@testset "unloaded backend packages fail loudly" begin
    @test ReactiveKernels._ad_required_packages(AutoEnzyme()) == ["Enzyme"]
    @test ReactiveKernels._ad_required_packages(AutoForwardDiff()) ==
        ["ForwardDiff"]
    @test ReactiveKernels._ad_required_packages(AutoNotLoadedTestPackage()) ==
        ["NotLoadedTestPackage"]

    @kernel unloaded_backend_objective(q::Float64) = q^2
    backend = AutoNotLoadedTestPackage()
    failure = try
        prepare_ad(unloaded_backend_objective, backend, 2.0;
            active = :q, want = :unloaded_backend_objective)
        nothing
    catch exception
        exception
    end
    @test failure isa ArgumentError
    @test occursin("using NotLoadedTestPackage", sprint(showerror, failure))

    @test_throws ArgumentError ad_gradient(
        unloaded_backend_objective, backend, 2.0;
        active = :q, want = :unloaded_backend_objective)
    @test_throws ArgumentError prepare_ad_pullback(
        unloaded_backend_objective, backend, 1.0, 2.0;
        active = :q, want = :unloaded_backend_objective)
    @test_throws ArgumentError ad_pullback(
        unloaded_backend_objective, backend, 1.0, 2.0;
        active = :q, want = :unloaded_backend_objective)
end
