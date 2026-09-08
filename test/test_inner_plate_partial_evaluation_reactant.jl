using ReactiveKernels, Reactant, Test
using DifferentiationInterface: AutoEnzyme
import Enzyme

isdefined(@__MODULE__, :InnerPlatePartialEvaluation) ||
    include("fixtures/inner_plate_partial_evaluation.jl")

@testset "Inner partial evaluation on loaded Reactant" begin
    @test Base.get_extension(ReactiveKernels, :ReactiveKernelsReactantExt) !== nothing
    C = InnerPlatePartialEvaluation
    backend = AutoEnzyme(; mode=Enzyme.Reverse)
    for n in (3, 32), (spec, data) in (
            (C.pure, collect(range(1.0, 4.0; length=n))),
            (C.boolean, [i - 2 for i in 1:n]))
        q = [0.3, -0.1]
        for bound in (false, true)
            args = bound ? (q,) : (q, data)
            kernel = prepare(spec; bound=bound ? (; data) : NamedTuple())
            prepared = prepare_ad(kernel, backend, args...; active=:q)
            value, gradient = ad_value_and_gradient(prepared, args...)
            traced = map(Reactant.to_rarray, args)
            primal = Reactant.compile(kernel, traced; sync=true)
            both = compile_ad_value_and_gradient(prepared, traced...; sync=true)
            @test Float64(primal(traced...)) ≈ value
            result, derivative = both(traced...)
            @test Float64(result) ≈ value
            @test Array(derivative) ≈ gradient
        end
    end

    @testset "cached prefix with live atomic vector, N=$n" for n in (0, 1, 32)
        data = exp.(collect(range(0.1, 1.0; length=max(n, 2)))[1:n])
        q = [0.3, -0.1]
        kernel = prepare(C.ref_live; bound=(; data))
        caches = filter(r -> r.op isa ReactiveKernels._BoundConstant &&
            startswith(String(only(r.outputs).name), "bound_plate_"), kernel.plan.recipes)
        @test length(caches) == (n == 0 ? 0 : 1)
        if n > 0
            @test only(caches).op.value ≈ log.(data)
            @test !occursin("log(", string(code_expr(kernel)))
        end
        prepared = prepare_ad(kernel, backend, q; active=:q)
        rq = Reactant.to_rarray(q)
        primal = Reactant.compile(kernel, (rq,); sync=true)
        both = compile_ad_value_and_gradient(prepared, rq; sync=true)
        plain = prepare(C.ref_live)
        for parameter in (q, [-0.2, 0.5])
            traced = Reactant.to_rarray(parameter)
            expected = sum(log, data) * sum(parameter)
            expected_gradient = fill(sum(log, data), length(parameter))
            value, gradient = ad_value_and_gradient(prepared, parameter)
            @test kernel(parameter) ≈ plain(parameter, data) ≈ expected
            @test value ≈ expected
            @test gradient ≈ expected_gradient
            @test Float64(primal(traced)) ≈ expected
            result, derivative = both(traced)
            @test Float64(result) ≈ expected
            @test Array(derivative) ≈ expected_gradient
        end
    end
end
