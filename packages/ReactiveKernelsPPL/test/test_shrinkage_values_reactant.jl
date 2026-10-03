using Reactant

function _sh_backend_modules(sampler, ru)
    post, ad = sampler.kernel, sampler.ad
    both(v) = ad_value_and_gradient(ad, v)
    return (
        ("primal.", repr(Reactant.@code_hlo optimize=false post(ru))),
        ("primal.optimized.", repr(Reactant.@code_hlo post(ru))),
        ("reverse.", repr(Reactant.@code_hlo optimize=false both(ru))),
        ("reverse.optimized.", repr(Reactant.@code_hlo both(ru))),
    )
end

function _sh_backend_operations(sampler, ru)
    return [prefix * m.match for (prefix, hlo) in _sh_backend_modules(sampler, ru) for m in
        eachmatch(r"\b(?:stablehlo|chlo|enzyme|func|arith)\.\w+", hlo)]
end

@testset "explicit shrinkage values: compiled math and retained structure" begin
    raw_operations = Dict{Symbol,Vector{String}}()
    optimized_bodies = Dict{Symbol,Vector{String}}()
    for (n, groups) in ((7, 3), (9, 4), (13, 6)), kind in (:indexed, :monotonic, :r2)
        @testset "$kind / n=$n / groups=$groups" begin
            case = _sh_case(kind, n, groups)
            original = deepcopy(case.data)
            bound, built = _sh_model(case)
            u = [0.2 * sin(i) for i in 1:built.layout.total]
            oracle = w -> _sh_oracle_parts(case, built.layout, w).posterior
            sampler = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
            grad = similar(u)
            native, _ = sampler_value_and_gradient!(sampler, grad, u)
            ru = Reactant.to_rarray(u)
            primal = Reactant.@compile sampler.kernel(ru)
            reverse = compile_ad_value_and_gradient(sampler.ad, ru)
            for shift in (0.0, 0.27)
                v = u .+ shift
                rv = Reactant.to_rarray(v)
                @test Float64(primal(rv)) ≈ oracle(v) rtol=1e-12
                value, gradient = reverse(rv)
                @test Float64(value) ≈ oracle(v) rtol=1e-12
                @test Array(gradient) ≈ _findiff_grad(oracle, v) rtol=1e-5 atol=1e-7
                if shift == 0
                    @test Float64(value) ≈ native
                    @test Array(gradient) ≈ grad
                end
            end
            ops = Base.invokelatest(_sh_backend_operations, sampler, ru)
            @test !isempty(ops)
            raw = filter(op -> !occursin(".optimized.", op), ops)
            if groups == 3
                raw_operations[kind] = raw
            else
                @test raw == raw_operations[kind]
            end
            # At groups=3 the simplex has one packed coordinate: optimizing
            # its reductions removes scalar special cases. Compare ordinary
            # optimized bodies at non-singleton widths. The inspected reverse
            # IR also replaces a slice/concatenate pair with pad as its packed
            # output grows. These three shape operations contain no bodies;
            # retain every arithmetic and control-flow operation in the check.
            shape_ops = ("stablehlo.slice", "stablehlo.concatenate", "stablehlo.pad")
            body = filter(op -> occursin(".optimized.", op) &&
                !any(s -> endswith(op, s), shape_ops), ops)
            if groups == 4
                optimized_bodies[kind] = body
            elseif groups == 6
                @test body == optimized_bodies[kind]
            end
            @test case.data == original
        end
    end
end
