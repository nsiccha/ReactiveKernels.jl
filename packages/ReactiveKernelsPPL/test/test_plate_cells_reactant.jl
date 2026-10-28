using DifferentiationInterface
using Distributions
using Enzyme
using ReactiveKernels
using ReactiveKernelsPPL
using Reactant
using Test

# Observation and stratum counts are bound data, not structural constants.
# Check backend operations as well as primal and reverse-mode parity.
function _pcr_build(n, S)
    model = @rkppl begin
        a ~ Normal(0, 5)
        sigma ~ Exponential(1)
        weights[levels(s), 1:2] .~ Normal.(0, 1)
        z[levels(g), 1:2] .~ Normal.(0, 1)
        @plate for i in eachindex(y)
            r[i, 1:2] = weights[s[i], :] .* z[g[i], :]
        end
        mu = a .+ r[:, 1] .+ x .* r[:, 2]
        y .~ Normal.(mu, sigma)
    end
    bound = (model(; x = collect(range(-1.3, 1.7; length = n)), g = [mod1(i, 4) for i in 1:n], s = [mod1(i, S) for i in 1:n]) | (; y = [0.8 * sin(0.7i) for i in 1:n]))
    built = build_kernel(bound)
    u = [0.3 * sin(1.1i + 0.3) for i in 1:built.layout.total]
    return Base.invokelatest(_pcr_measure, built, bound, u; structure_body = true)
end

function _pcr_measure(built, bound, u; expected = nothing, reference = nothing,
        structure_ad = false, structure_body = false)
    kern = prepare_query(built, bound, :sampler)
    ru = Reactant.to_rarray(u)
    hlo = repr(Reactant.@code_hlo optimize = false kern(ru))
    ops = Dict{String,Int}()
    for m in eachmatch(r"(?:stablehlo|enzyme)\.[a-z_]+", hlo)
        ops[m.match] = get(ops, m.match, 0) + 1
    end
    if structure_body
        inventory = _ppl_backend_operation_inventory(hlo)
        ops = inventory.body_ops
        println("array plate complete inventory: ", sort!(collect(inventory.all_ops)))
    end
    native = kern(u)
    compiled = Reactant.@compile kern(ru)
    @test Float64(compiled(ru)) ≈ native rtol = 1e-9
    q = prepare_sampler(built, bound, u;
        backend = AutoEnzyme(; mode = Enzyme.Reverse))
    value, grad = sampler_value_and_gradient!(q, similar(u), u)
    cad = compile_ad_value_and_gradient(q.ad, ru)
    rvalue, rgrad = cad(ru)
    @test value ≈ native rtol = 1e-12
    @test Float64(rvalue) ≈ value rtol = 1e-9
    @test Array(rgrad) ≈ grad rtol = 1e-8 atol = 1e-9
    if expected !== nothing
        @test Float64(rvalue) ≈ expected[1] rtol = 1e-9
        @test Array(rgrad) ≈ expected[2] rtol = 1e-5 atol = 1e-7
    end
    if reference !== nothing
        nextu = u .+ 0.03
        nextvalue, nextgrad = cad(Reactant.to_rarray(nextu))
        @test Float64(nextvalue) ≈ reference(nextu) rtol = 1e-9
        @test Array(nextgrad) ≈ _findiff_grad(reference, nextu) rtol = 1e-5 atol = 1e-7
    end
    if structure_ad
        # SamplerQuery keeps host-side layout metadata. Trace its documented
        # AD field directly, as compile_ad_value_and_gradient does above.
        ad = q.ad
        both(w) = ad_value_and_gradient(ad, w)
        adhlo = repr(Reactant.@code_hlo optimize = false both(ru))
        for m in eachmatch(r"(?:stablehlo|enzyme)\.[a-z_]+", adhlo)
            key = "ad." * m.match
            ops[key] = get(ops, key, 0) + 1
        end
    end
    return ops
end

@testset "Reactant: array plate cells retain data-dependent iteration" begin
    small = _pcr_build(24, 3)
    large = _pcr_build(40, 5)
    # Before the batching pass, enzyme.batch retains one cell region. The
    # ordinary compile above lowers it with the default optimizer.
    @test get(small, "enzyme.batch", 0) > 0
    @test small == large
end

@testset "Reactant: level values and varying priors retain one cell body" begin
    for kind in (:varying, :latent, :deterministic, :constant)
        histograms = map(((6, 2), (20, 5))) do (n, S)
            built, bound, cols, u = _plv_build(kind, n, S)
            oracle(w) = first(_plv_oracle(built, cols, kind, w))
            expected = (oracle(u), _findiff_grad(oracle, u))
            Base.invokelatest(_pcr_measure, built, bound, u; expected,
                reference = oracle, structure_ad = true)
        end
        @test get(histograms[1], "enzyme.batch", 0) > 0
        @test histograms[1] == histograms[2]
    end
    built, bound, u, oracle = _plv_lazy_build()
    expected = (oracle(u), _findiff_grad(oracle, u))
    Base.invokelatest(_pcr_measure, built, bound, u; expected,
        reference = oracle, structure_ad = true)
    built, bound, cols, u = _plv_build(:latent, 6, 3; labels = ["c", "a", "b"])
    string_oracle(w) = first(_plv_oracle(built, cols, :latent, w))
    expected = (string_oracle(u), _findiff_grad(string_oracle, u))
    Base.invokelatest(_pcr_measure, built, bound, u; expected,
        reference = string_oracle, structure_ad = true)
end

@testset "Reactant: level-axis gathers precede the cell" begin
    for kind in (:selected, :reordered, :other, :other_selected, :varying,
            :matrix_selected, :matrix_other, :lazy, :live)
        histograms = map(((7, 3), (21, 5))) do (n, S)
            built, bound, cols, u, oracle = _plv_axis_build(kind, n, S)
            reference(w) = first(oracle(w))
            expected = (reference(u), _findiff_grad(reference, u))
            Base.invokelatest(_pcr_measure, built, bound, u; expected,
                reference, structure_ad = true)
        end
        @test get(histograms[1], "enzyme.batch", 0) > 0
        @test histograms[1] == histograms[2]
    end
    histograms = map(((7, 3), (21, 5))) do (n, S)
        built, bound, cols, u, oracle = _plv_stack_build(n, S)
        reference(w) = first(oracle(w))
        expected = (reference(u), _findiff_grad(reference, u))
        Base.invokelatest(_pcr_measure, built, bound, u; expected,
            reference, structure_ad = true)
    end
    @test get(histograms[1], "enzyme.batch", 0) > 0
    @test histograms[1] == histograms[2]
    for kind in (:selected, :other)
        built, bound, cols, u, oracle = _plv_axis_build(kind, 9, 3;
            labels = ["c", "a", "b"])
        reference(w) = first(oracle(w))
        expected = (reference(u), _findiff_grad(reference, u))
        Base.invokelatest(_pcr_measure, built, bound, u; expected,
            reference, structure_ad = true)
    end
    for kind in (:selected, :other)
        histograms = map(((9, ["c", "a", "b", "unused"]),
                (21, ["c", "a", "b", "unused", "extra", "more"]))) do (n, labels)
            built, bound, cols, u, oracle = _plv_axis_build(kind, n, length(labels);
                labels, pooled = true)
            reference(w) = first(oracle(w))
            expected = (reference(u), _findiff_grad(reference, u))
            Base.invokelatest(_pcr_measure, built, bound, u; expected,
                reference, structure_ad = true)
        end
        @test get(histograms[1], "enzyme.batch", 0) > 0
        @test histograms[1] == histograms[2]
    end
end

@testset "Reactant limitation: inactive invalid constant read in a live branch" begin
    # Native primal and AD pass above. The backend-only reproducer is
    # benchmark/repro_reactant_inactive_constant_index.jl.
    built, bound, cols, u, oracle = _plv_axis_build(:live_oob, 7, 3)
    kernel = prepare_query(built, bound, :sampler)
    ru = Reactant.to_rarray(u)
    err = try
        Reactant.@compile kernel(ru)
        nothing
    catch e
        e
    end
    if err !== nothing
        @test err isa BoundsError
        @test err.i == (100,)
    end
    @test_broken err === nothing
end
