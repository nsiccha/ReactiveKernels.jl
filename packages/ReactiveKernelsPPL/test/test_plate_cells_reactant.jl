using DifferentiationInterface
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
        r ~ varying_stratified_correlated(g, s, 2)
        mu = a .+ r[:, 1] .+ x .* r[:, 2]
        y .~ Normal.(mu, sigma)
    end
    bound = model(; y = [0.8 * sin(0.7i) for i in 1:n],
        x = collect(range(-1.3, 1.7; length = n)),
        g = [mod1(i, 4) for i in 1:n], s = [mod1(i, S) for i in 1:n])
    built = build_kernel(bound)
    u = [0.3 * sin(1.1i + 0.3) for i in 1:built.layout.total]
    return Base.invokelatest(_pcr_measure, built, bound, u)
end

function _pcr_measure(built, bound, u)
    kern = prepare_query(built, bound, :sampler)
    ru = Reactant.to_rarray(u)
    hlo = repr(Reactant.@code_hlo optimize = false kern(ru))
    ops = Dict{String,Int}()
    for m in eachmatch(r"(?:stablehlo|enzyme)\.[a-z_]+", hlo)
        ops[m.match] = get(ops, m.match, 0) + 1
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
