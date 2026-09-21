using ReactiveKernels
using Reactant
using Test
import Enzyme
using DifferentiationInterface: AutoEnzyme
using ReactiveKernelsPPLExamples.EightSchoolsExample:
    build_eight_schools_graph, EIGHT_SCHOOLS_Y, EIGHT_SCHOOLS_SIGMA
using ReactiveKernelsDistributionKernels.DistributionKernelSources: bernoulli

# Reactant-compiled AD is the AD analog of the primal Reactant path: it takes a
# native `PreparedADKernel` (which owns the scalar-WANT / active-port
# validation and the authored-order reorder) and compiles a
# DifferentiationInterface gradient / value-and-gradient through Reactant. The
# differentiation engine stays the caller's DI backend — `AutoEnzyme` here —
# exactly as on the native reverse pass. Analytic derivative checks and model
# comparisons exercise the AD boundary under ordinary compiler optimization.

const AD_REACTANT_BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)
_trace(x) = Reactant.to_rarray(x)

include("test_ad_fused_reactant.jl")

@testset "Reactant-compiled AD" begin
    @testset "standalone objective: authored defaults, keywords, parity" begin
        @kernel objective(q::Vector{Float64}, scale::Float64 = 1.25;
                          data::Vector{Float64}, offset::Float64 = 0.0) = begin
            density::Float64 = sum(q .* data) - scale * sum(abs2, q) + offset
        end

        q = [0.3, -0.4, 0.2]
        data = [2.0, -1.0, 0.5]
        prepared = prepare_ad(
            objective, AD_REACTANT_BACKEND, q; data, active = :q, want = :density)

        gref = similar(q)
        vref, gref = ad_value_and_gradient!(prepared, gref, q; data)

        # The selected HAVE boundary in authored order — supply each traced.
        names = Tuple(input.name for input in inputs(prepared))
        host = map(names) do name
            name === :q ? q :
            name === :scale ? 1.25 :
            name === :data ? data : 0.0
        end
        traced = map(_trace, host)

        compiled_gradient = compile_ad_gradient(prepared, traced...)
        gradient = Array(compiled_gradient(traced...))
        @test gradient ≈ gref
        @test gradient ≈ data .- 2(1.25) .* q

        compiled_both = compile_ad_value_and_gradient(prepared, traced...)
        value, both_gradient = compiled_both(traced...)
        @test Float64(value) ≈ vref
        @test Array(both_gradient) ≈ gref

        # A host (non-traced) active argument, or an arity mismatch, is a clear
        # ArgumentError rather than a bare MethodError.
        @test_throws ArgumentError compile_ad_gradient(prepared, q)
        @test_throws ArgumentError compile_ad_gradient(prepared, _trace(q), _trace(data))
    end


    @testset "multiple active ports compile as one structured point" begin
        @kernel multi_active_reactant(
                alpha::Vector{Float64}, beta::Vector{Float64},
                data::Vector{Float64}) = begin
            objective::Float64 = sum(abs2, alpha) + sum(beta .* data)
        end

        alpha = [0.3, -0.4, 0.2]
        beta = [-0.1, 0.7, 0.5]
        data = [2.0, -1.0, 0.5]
        prepared = prepare_ad(
            multi_active_reactant, AD_REACTANT_BACKEND, alpha, beta, data;
            active = (:beta, :alpha), want = :objective)

        traced = map(_trace, (alpha, beta, data))
        compiled = compile_ad_value_and_gradient(prepared, traced...)
        value, gradient = compiled(traced...)
        @test Float64(value) ≈ sum(abs2, alpha) + sum(beta .* data)
        @test Array(gradient[1]) ≈ data
        @test Array(gradient[2]) ≈ 2 .* alpha

        staged = let prepared = prepared
            (a, b, d) -> ad_value_and_gradient(prepared, a, b, d)
        end
        staged_compiled = Reactant.compile(staged, traced)
        staged_value, staged_gradient = staged_compiled(traced...)
        @test Float64(staged_value) ≈ Float64(value)
        @test Array(staged_gradient[1]) ≈ data
        @test Array(staged_gradient[2]) ≈ 2 .* alpha

        @kernel mixed_active_reactant(
                coefficients::Vector{Float64}, dispersion::Float64) = begin
            objective::Float64 =
                sum(abs2, coefficients) + dispersion^2
        end
        coefficients = [0.2, -0.5]
        dispersion = 1.7
        mixed = prepare_ad(
            mixed_active_reactant, AD_REACTANT_BACKEND,
            coefficients, dispersion;
            active = (:coefficients, :dispersion), want = :objective)
        mixed_traced = (
            _trace(coefficients),
            Reactant.to_rarray(dispersion; track_numbers = true),
        )
        mixed_compiled = compile_ad_value_and_gradient(
            mixed, mixed_traced...)
        mixed_value, mixed_gradient = mixed_compiled(mixed_traced...)
        @test Float64(mixed_value) ≈
              sum(abs2, coefficients) + dispersion^2
        @test Array(mixed_gradient[1]) ≈ 2 .* coefficients
        @test Float64(mixed_gradient[2]) ≈ 2dispersion
    end

    @testset "partially-evaluated AD kernels compile over the remaining ports" begin
        @kernel bound_objective(q::Vector{Float64}, data::Vector{Float64}) = begin
            shifted = data .- 1.0
            density::Float64 = sum(q .* shifted) - 0.5 * sum(abs2, q)
        end

        q = [0.3, -0.4, 0.2]
        data = [2.0, -1.0, 0.5]
        prepared = prepare_ad(
            bound_objective, AD_REACTANT_BACKEND, q;
            active = :q, want = :density, bound = (; data))
        @test Tuple(input.name for input in inputs(prepared.kernel)) == (:q,)

        gref = similar(q)
        vref, gref = ad_value_and_gradient!(prepared, gref, q)
        @test gref ≈ (data .- 1.0) .- q

        traced_q = _trace(q)
        compiled_gradient = compile_ad_gradient(prepared, traced_q)
        @test Array(compiled_gradient(traced_q)) ≈ gref

        compiled_both = compile_ad_value_and_gradient(prepared, traced_q)
        value, both_gradient = compiled_both(traced_q)
        @test Float64(value) ≈ vref
        @test Array(both_gradient) ≈ gref

        # The benchmark-facing compiler ABI exposes the same bound array as an
        # explicitly transferred inactive operand, avoiding a compiler literal
        # while retaining the q-only public PreparedADKernel above.
        _, bound_arrays =
            ReactiveKernels._externalize_bound_arrays(prepared.kernel)
        traced_bound = map(_trace, bound_arrays)
        compiled_externalized =
            ReactiveKernels._reactant_compile_ad_externalized(
                Val(:value_and_gradient), prepared,
                (traced_q,), traced_bound; sync = true)
        external_value, external_gradient =
            compiled_externalized(traced_q, traced_bound...)
        @test Float64(external_value) ≈ vref
        @test Array(external_gradient) ≈ gref
    end

    @testset "small bound arrays stay compiler literals on the automatic path" begin
        ext = Base.get_extension(ReactiveKernels, :ReactiveKernelsReactantExt)
        @kernel small_bound_objective(
                q::Vector{Float64}, data::Vector{Float64}) = begin
            density::Float64 = sum(q .* data)
        end

        q = [0.3, -0.4, 0.2]
        data = [2.0, -1.0, 0.5]
        prepared = prepare_ad(
            small_bound_objective, AD_REACTANT_BACKEND, q;
            active = :q, want = :density, bound = (; data))
        traced_q = _trace(q)

        # A small dataset is embedded so the compiler can fold data-only
        # terms; the compiled callable takes no hidden operands.
        embedded = compile_ad_gradient(prepared, traced_q)
        @test !(embedded isa ext._ExternalizedADExecutable)
        @test Array(embedded(traced_q)) ≈ data

        # Above the embedding limit the same kernel externalizes the array.
        limit = ext._REACTANT_EMBEDDED_BOUND_ARRAY_ELEMENTS
        previous_limit = limit[]
        try
            limit[] = length(data) - 1
            externalized = compile_ad_gradient(prepared, traced_q)
            @test externalized isa ext._ExternalizedADExecutable
            @test Array(externalized(traced_q)) ≈ data
        finally
            limit[] = previous_limit
        end
    end

    @testset "Eight Schools supported boundaries match native reverse pass" begin
        model = build_eight_schools_graph()
        observations = Float64.(EIGHT_SCHOOLS_Y)
        observation_scales = Float64.(EIGHT_SCHOOLS_SIGMA)
        theta = 0.25 .* collect(1.0:8.0)
        unconstrained = [1.5, log(2.0), theta...]

        boundaries = (
            (name = "packed_unconstrained/joint",
             have = (:unconstrained, :observations, :observation_scales),
             want = :posterior, active = :unconstrained,
             args = (unconstrained, observations, observation_scales)),
            (name = "minimal_likelihood/likelihood",
             have = (:θ, :observations, :observation_scales),
             want = :likelihood, active = :θ,
             args = (theta, observations, observation_scales)),
            (name = "packed_unconstrained/likelihood",
             have = (:unconstrained, :observations, :observation_scales),
             want = :likelihood, active = :unconstrained,
             args = (unconstrained, observations, observation_scales)),
        )

        for boundary in boundaries
            kernel = prepare(model; have = boundary.have, want = boundary.want)
            prepared = prepare_ad(
                kernel, AD_REACTANT_BACKEND, boundary.args...;
                active = boundary.active)

            # Host preparation deliberately freezes the fast native-only AD
            # body. Reactant compilation must reconstruct the full kernel
            # selector instead of tracing this call, so the tensorized body is
            # selected once the arguments below become Reactant values.
            @test prepared.call isa ReactiveKernels._ADNativeKernelCall

            reference = similar(boundary.args[1])
            value, reference =
                ad_value_and_gradient!(prepared, reference, boundary.args...)

            traced = map(_trace, boundary.args)
            gradient = Array(compile_ad_gradient(prepared, traced...)(traced...))
            @test gradient ≈ reference

            compiled_value, compiled_gradient =
                compile_ad_value_and_gradient(prepared, traced...)(traced...)
            @test Float64(compiled_value) ≈ value
            @test Array(compiled_gradient) ≈ reference
        end
    end

    # dogs_hierarchical shape: a bernoulli likelihood plate where `p = a^coef` (a
    # in (0,1)); the first cell has coef = 0 so `p = 1` with an observed success —
    # the exact boundary that made the direct-p HAVE NaN through the p -> logit
    # round trip. The boundary-safe route keeps the compiled gradient finite and
    # equal to the native reverse pass.
    @testset "bernoulli direct-p boundary plate gradient (compiled)" begin
        @kernel bern_plate(u::Vector{Float64}, coef::Vector{Float64},
                           y::Vector{Bool}) = begin
            param::Float64 = sum(view(u, 1:1))
            a::Float64 = 1 / (1 + exp(-param))
            log_a::Float64 = log(a)
            pointwise = plate(y, coef, log_a) do yi, c, la
                bernoulli(exp(c * la)).logpdf(yi)
            end
            total::Float64 = sum(pointwise)
            return total
        end
        u = [0.3]
        coef = [0.0, 1.0, 2.0, 3.0]   # coef[1] = 0 -> p = 1 (boundary), y[1] = true
        y = Bool[1, 0, 1, 1]
        kernel = prepare(bern_plate; have = (:u, :coef, :y), want = :total)
        prepared = prepare_ad(kernel, AD_REACTANT_BACKEND, u, coef, y; active = :u)

        reference = similar(u)
        value, reference = ad_value_and_gradient!(prepared, reference, u, coef, y)
        @test all(isfinite, reference)   # was NaN before the boundary-safe route

        traced = map(_trace, (u, coef, y))
        gradient = Array(compile_ad_gradient(prepared, traced...)(traced...))
        @test all(isfinite, gradient)
        @test gradient ≈ reference

        compiled_value, compiled_gradient =
            compile_ad_value_and_gradient(prepared, traced...)(traced...)
        @test Float64(compiled_value) ≈ value
        @test Array(compiled_gradient) ≈ reference
    end

    # reactant-full-pr-f9f453e4: a 2-deep chain over a strided slice
    # miscompiles under Reactant's default pipeline (`slice_slice` fuses the
    # nested slices into a shape that breaks `slice_elementwise` with an
    # invalid slice, or Enzyme's reverse with a mismatched add, SIGABRTing
    # the compile). The `:no_slice_slice` pipeline compiles it with
    # native-identical results. (The default-pipeline crash is fatal, so only
    # the workaround path is asserted here.)
    @testset "chained strided-slice consumers (:no_slice_slice)" begin
        @kernel chain_plate(u::Vector{Float64}, tmap, off) = begin
            s = u[1]
            reads = exp.(s .+ off)
            r = reads[tmap]
            m1 = min(r[1], r[2])
            m2 = min(m1, r[3])
            total::Float64 = m1 + m2
        end
        u = [2.0]
        tmap = [2, 4, 6]
        off = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        kernel = prepare(chain_plate; have = (:u, :tmap, :off),
                         want = :total, bound = (; tmap, off))
        prepared = prepare_ad(kernel, AD_REACTANT_BACKEND, u; active = :u)

        reference = similar(u)
        value, reference = ad_value_and_gradient!(prepared, reference, u)

        traced_u = _trace(u)
        compiled_value, compiled_gradient = compile_ad_value_and_gradient(
            prepared, traced_u; optimize = :no_slice_slice)(traced_u)
        @test Float64(compiled_value) ≈ value
        @test Array(compiled_gradient) ≈ reference

        gradient_only = Array(compile_ad_gradient(
            prepared, traced_u; optimize = :no_slice_slice)(traced_u))
        @test gradient_only ≈ reference
    end

    # An explicit `:all` pipeline matches the default path exactly; this
    # guards the `optimize` keyword's verbatim forwarding to Reactant.
    @testset "explicit :all pipeline matches the default" begin
        @kernel fill_plate(u::Vector{Float64}, tmap) = begin
            s = u[1]
            reads = fill(s, 6)
            r = reads[tmap]
            m1 = min(r[1], r[2])
            m2 = min(m1, r[3])
            total::Float64 = m1 + m2
        end
        u = [2.0]
        tmap = [2, 4, 6]
        kernel = prepare(fill_plate; have = (:u, :tmap), want = :total,
                         bound = (; tmap))
        prepared = prepare_ad(kernel, AD_REACTANT_BACKEND, u; active = :u)

        reference = similar(u)
        value, reference = ad_value_and_gradient!(prepared, reference, u)

        traced_u = _trace(u)
        default_value, default_gradient =
            compile_ad_value_and_gradient(prepared, traced_u)(traced_u)
        @test Float64(default_value) ≈ value
        @test Array(default_gradient) ≈ reference
        explicit_value, explicit_gradient = compile_ad_value_and_gradient(
            prepared, traced_u; optimize = :all)(traced_u)
        @test Float64(explicit_value) ≈ value
        @test Array(explicit_gradient) ≈ reference
    end

    @testset "surgical pipeline builder strips only slice_slice" begin
        ext = Base.get_extension(ReactiveKernels, :ReactiveKernelsReactantExt)
        pipe = ext._rk_reactant_pipeline_no_slice_slice()
        @test pipe isa String
        @test occursin("enzyme{", pipe)
        @test !occursin(r"slice_slice<\d+>;", pipe)
        @test occursin("slice_elementwise", pipe)
    end
end

@testset "staged prepared AD over bound views (§7a)" begin
    @kernel staged_view_objective(x::Vector{Float64}, y::Matrix{Float64}) = begin
        c1 = view(y, :, 1)
        m::Vector{Float64} = exp.(x)
        p1 = plate(c1, m) do o, mm
            o * mm
        end
        s::Float64 = sum(p1)
        return s
    end

    x = [0.3, -0.4, 0.2, 0.1, -0.2]
    y = [1.0 6.0; 2.0 7.0; 3.0 8.0; 4.0 9.0; 5.0 10.0]
    kernel = prepare(
        staged_view_objective; have = (:x, :y), want = :s, bound = (; y))
    prepared = prepare_ad(kernel, AD_REACTANT_BACKEND, x; active = :x)
    expected = y[:, 1] .* exp.(x)

    # The §7a contract: traced compile inputs stage the derivative inside the
    # enclosing compiled function; bound data stays bound.
    staged = let prepared = prepared
        (tq) -> ad_value_and_gradient(prepared, tq)
    end
    compiled = Reactant.compile(staged, (_trace(x),))
    value, gradient = compiled(_trace(x))
    @test Float64(value) ≈ sum(expected)
    @test Array(gradient) ≈ expected

    # Native compile inputs pass through untraced, and Reactant's autodiff
    # overlay then intercepts the native Enzyme call with all-native arguments
    # — a correct value with a silent zero gradient. That shape must fail
    # loudly at compile time instead of baking the corruption in.
    native_staged = let prepared = prepared
        (tq) -> ad_value_and_gradient(prepared, tq)
    end
    @test_throws ArgumentError Reactant.compile(native_staged, (x,))

    native_staged_inplace = let prepared = prepared
        (tq, tg) -> ad_value_and_gradient!(prepared, tg, tq)
    end
    @test_throws ArgumentError Reactant.compile(
        native_staged_inplace, (x, similar(x)))
end
