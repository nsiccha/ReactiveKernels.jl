# Focused optional-extension regression; also run by the NA integration runner.
using ReactiveKernels, MutatingFunctions, DifferentiationInterface, Test
import Enzyme

repeated_na_spec = @kernel repeated_nonallocating(q, d) = begin
    z = q .* d
    y = sum(z .* z)
    return y
end

function repeated_na_source_snapshot(spec)
    g = spec.graph
    (; values = copy(g.values), recipes = copy(g.recipes),
       producers = deepcopy(g.producers), aliases = copy(g.aliases),
       version = g.version, preparations = g.preparations,
       source = map(r -> repr(r.source), g.recipes),
       ports = copy(spec.ports), port_order = copy(spec.port_order),
       have = copy(spec.have_names), want = copy(spec.want_names))
end

@testset "independent non-allocating preparations" begin
    @test Base.get_extension(ReactiveKernels,
                             :ReactiveKernelsMutatingFunctionsExt) !== nothing
    @test Base.get_extension(ReactiveKernels,
                             :ReactiveKernelsEnzymeExt) !== nothing
    spec = repeated_na_spec
    source_before = repeated_na_source_snapshot(spec)
    d = collect(range(0.2, 1.1; length = 16))
    q0 = collect(range(-0.5, 0.4; length = 16))
    q1 = q0 .+ 0.13
    inputs_before = map(copy, (d, q0, q1))

    @testset "primal buffers belong to each callable" begin
        # Both bodies are compiled while every cache slot still holds nothing.
        k1 = prepare_nonallocating(spec; have = (:q, :d), want = :z)
        k2 = prepare_nonallocating(spec; have = (:q, :d), want = :z)
        asts = map(k -> deepcopy(code_expr(k)), (k1, k2))
        z1 = k1(q0, d)
        saved_z1 = copy(z1)
        z2 = k2(q1, d)
        @test z1 ≈ q0 .* d
        @test z2 ≈ q1 .* d
        @test z1 !== z2
        @test z1 == saved_z1
        @test any(slot -> slot isa Base.RefValue && slot[] === z1, k1.caches)
        @test any(slot -> slot isa Base.RefValue && slot[] === z2, k2.caches)
        @test k1(q1, d) === z1
        saved_z2 = copy(z2)
        @test k1(q0, d) ≈ q0 .* d
        @test z2 == saved_z2
        @test map(code_expr, (k1, k2)) == asts
    end

    for (label, backend) in (
            ("ordinary Reverse", AutoEnzyme(mode = Enzyme.Reverse)),
            ("explicit Const", AutoEnzyme(mode = Enzyme.Reverse,
                                          function_annotation = Enzyme.Const)))
        @testset "$label" begin
            # Keep the established Const recipe separate from default Reverse.
            # Compile two unseeded kernels together, then a third after warming.
            kernels = [prepare_nonallocating(spec; have = (:q, :d),
                         want = :y, bound = (; d)) for _ in 1:2]
            asts = map(k -> deepcopy(code_expr(k)), kernels)
            prepared = [prepare_ad(k, backend, q0; active = :q) for k in kernels]
            push!(kernels, prepare_nonallocating(spec; have = (:q, :d),
                                                want = :y, bound = (; d)))
            push!(asts, deepcopy(code_expr(last(kernels))))
            push!(prepared, prepare_ad(last(kernels), backend, q1; active = :q))
            for (i, k) in enumerate(kernels), other in kernels[1:i-1]
                for (slot, other_slot) in zip(k.caches, other.caches)
                    slot isa Base.RefValue || continue
                    @test slot !== other_slot
                    slot[] isa AbstractArray && @test slot[] !== other_slot[]
                end
            end
            for (round, q) in enumerate((q0, q1, q0)), i in eachindex(kernels)
                k, ad = kernels[i], prepared[i]
                # Interleave changed points in the other primal and AD objects.
                other_q = isodd(round) ? q1 : q0
                other = mod1(i + 1, length(kernels))
                @test kernels[other](other_q) ≈ sum((other_q .* d).^2)
                @test ad_value_and_gradient(prepared[other], other_q)[2] ≈
                      2 .* other_q .* d.^2
                value, gradient = ad_value_and_gradient(ad, q)
                @test value ≈ sum((q .* d).^2)
                @test gradient ≈ 2 .* q .* d.^2
                @test k(q) ≈ value
                @test map(code_expr, kernels) == asts
                @test repeated_na_source_snapshot(spec) == source_before
                @test (d, q0, q1) == inputs_before
            end
        end
    end
    @test repeated_na_source_snapshot(spec) == source_before
    @test (d, q0, q1) == inputs_before
end
