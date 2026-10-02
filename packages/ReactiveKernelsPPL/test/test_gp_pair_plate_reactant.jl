using ReactiveKernels, Reactant, DifferentiationInterface, Test
import Enzyme
isdefined(@__MODULE__, :GPPairPlateFixtures) || include("test_gp_pair_plate.jl")

function _gp_hlo_counts(hlo)
    counts = Dict{String,Int}()
    for m in eachmatch(r"(?:stablehlo|enzyme)\.[a-z_]+", hlo)
        counts[m.match] = get(counts, m.match, 0) + 1
    end
    counts
end

@testset "GP pair plates: compiled covariance cuts and retained reverse" begin
    F = GPPairPlateFixtures
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    for periodic in (false, true), bound_locations in (false, true)
        structure = []
        for n in (8, 32)
            x = collect(range(-0.7, 1.1; length = n))
            factors = periodic ? [1.7^2, 0.8^2, 1.3, 1e-4] : [1.7^2, 2 * 0.8^2, 1e-4]
            q = bound_locations ? factors : vcat(factors, x)
            matrix_oracle = periodic ?
                w -> F.periodic(bound_locations ? x : w[5:end], sqrt(w[1]), sqrt(w[2]), w[3], w[4]) :
                w -> F.exp_quad(bound_locations ? x : w[4:end], sqrt(w[1]), sqrt(w[2]/2), w[3])
            oracle = w -> sum(matrix_oracle(w))
            spec = bound_locations ? (periodic ? F.periodic_data_cut_loss : F.exp_data_cut_loss) :
                (periodic ? F.periodic_cut_loss : F.exp_cut_loss)
            data = bound_locations ? (; x) : (;)
            kernel = prepare(spec; bound = data)
            rq = Reactant.to_rarray(q)
            matrix_kernel = prepare(spec; bound = data, want = :covariance)
            compiled_matrix = Reactant.@compile matrix_kernel(rq)
            @test Array(compiled_matrix(rq)) ≈ matrix_oracle(q) rtol = 1e-10
            compiled = Reactant.@compile kernel(rq)
            @test Float64(compiled(rq)) ≈ oracle(q) rtol = 1e-10
            ad = prepare_ad(kernel, backend, q; active = :q)
            native_value, native_gradient = ad_value_and_gradient(ad, q)
            @test native_value ≈ oracle(q)
            @test native_gradient ≈ F.findiff(oracle, q) rtol = 1e-5 atol = 1e-6
            compiled_ad = compile_ad_value_and_gradient(ad, rq)
            value, gradient = compiled_ad(rq)
            @test Float64(value) ≈ oracle(q) rtol = 1e-10
            @test Array(gradient) ≈ native_gradient rtol = 1e-8 atol = 1e-8
            q2 = q .+ 0.01
            value2, gradient2 = compiled_ad(Reactant.to_rarray(q2))
            @test Float64(value2) ≈ oracle(q2) rtol = 1e-10
            @test Array(gradient2) ≈ F.findiff(oracle, q2) rtol = 1e-5 atol = 1e-6
            @test Array(rq) == q
            both = w -> ad_value_and_gradient(ad, w)
            primal_hlo = repr(Reactant.@code_hlo optimize = false kernel(rq))
            reverse_hlo = repr(Reactant.@code_hlo optimize = false both(rq))
            primal = _gp_hlo_counts(primal_hlo)
            reverse = _gp_hlo_counts(reverse_hlo)
            @test get(primal, "enzyme.batch", 0) > 0
            @info "GP pair structure" periodic bound_locations n primal_ops = sum(values(primal)) reverse_ops = sum(values(reverse)) primal_batches = get(primal, "enzyme.batch", 0) reverse_batches = get(reverse, "enzyme.batch", 0)
            push!(structure, (primal, reverse))
        end
        # Shapes specialize, while the scalar covariance and derivative cell
        # regions remain one body rather than one copy per location pair.
        @test structure[1] == structure[2]
    end
end
