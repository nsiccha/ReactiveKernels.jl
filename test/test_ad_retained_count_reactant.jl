module RetainedCountADReactantTests
using ReactiveKernels, Reactant, Test
import Enzyme
using DifferentiationInterface: AutoEnzyme

@kernel count_objective(u::Vector{Float64}, counts::Vector{Int}) = begin
    terms = plate(counts, Ref(u)) do n, point
        term = begin
            total = zero(point[1])
            i = 0
            while i < n
                total += exp(i * point[1] - point[2])
                i += 1
            end
            log(total)
        end
        return term
    end
    density = sum(terms)
end

function operation_counts(hlo)
    counts = Dict{String,Int}()
    for op in eachmatch(r"stablehlo\.[a-z_]+", hlo)
        # Constants may deduplicate differently as bound values change.
        op.match == "stablehlo.constant" && continue
        counts[op.match] = get(counts, op.match, 0) + 1
    end
    counts
end

@testset "bound batched counts retain primal and reverse loops" begin
    backend = AutoEnzyme(; mode=Enzyme.Reverse)
    u = [0.2, 0.3]
    ru = Reactant.to_rarray(u)
    ext = Base.get_extension(ReactiveKernels, :ReactiveKernelsReactantExt)
    pipe = ext._rk_reactant_pipeline_no_slice_slice()
    primal_structures, reverse_structures = Dict{String,Int}[], Dict{String,Int}[]
    for n in (3, 7), rows in (2, 12)
        counts = fill(n, rows)
        kernel = prepare(count_objective; want=:density, bound=(; counts))
        prepared = prepare_ad(kernel, backend, u; active=:u)
        weights = exp.((0:n-1) .* u[1] .- u[2])
        expected_value = rows * log(sum(weights))
        expected_gradient = [rows * sum((0:n-1) .* weights) / sum(weights), -rows]
        native_value, native_gradient = ad_value_and_gradient(prepared, u)
        @test native_value ≈ expected_value
        @test native_gradient ≈ expected_gradient
        compiled = compile_ad_value_and_gradient(prepared, ru; optimize=:no_slice_slice)
        value, gradient = compiled(ru)
        @test Float64(value) ≈ expected_value
        @test Array(gradient) ≈ expected_gradient
        primal = Reactant.compile(kernel, (ru,); optimize=pipe)
        @test Float64(primal(ru)) ≈ expected_value
        primal_hlo = repr(Reactant.@code_hlo optimize=pipe kernel(ru))
        reverse(u) = ad_value_and_gradient(prepared, u)
        reverse_hlo = repr(Reactant.@code_hlo optimize=pipe reverse(ru))
        primal_ops, reverse_ops = operation_counts(primal_hlo), operation_counts(reverse_hlo)
        @test get(primal_ops, "stablehlo.while", 0) > 0
        @test get(reverse_ops, "stablehlo.while", 0) > 0
        push!(primal_structures, primal_ops)
        push!(reverse_structures, reverse_ops)
    end
    @test all(==(first(primal_structures)), primal_structures)
    @test all(==(first(reverse_structures)), reverse_structures)
end
end
