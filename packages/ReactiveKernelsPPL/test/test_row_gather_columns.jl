using DifferentiationInterface
using Distributions
using Enzyme
using ReactiveKernels
using ReactiveKernelsPPL
using Test

@rkppl _rgc_joint(c) = begin
    eachcol(S[1:3, 1:3]) .~ Dirichlet([1.0, 1.0, 1.0])
    beta[1:3] .~ Normal.(0.0, 1.0)
    return (cumsum(vcat(zeros(1, 3), S); dims=1) .* beta')[c, :]
end

const _RGC_JOINT = quote
    P ~ _rgc_joint(c)
    y1 .~ Normal.(P[:, 1], 1.0)
    y2 .~ Normal.(P[:, 2], 1.0)
    y3 .~ Normal.(P[:, 3], 1.0)
end

const _RGC_DIRECT = quote
    eachcol(S[1:3, 1:3]) .~ Dirichlet([1.0, 1.0, 1.0])
    beta[1:3] .~ Normal.(0.0, 1.0)
    scaled = cumsum(vcat(zeros(1, 3), S); dims=1) .* beta'
    y1 .~ Normal.(scaled[c, 1], 1.0)
    y2 .~ Normal.(scaled[c, 2], 1.0)
    y3 .~ Normal.(scaled[c, 3], 1.0)
end

_rgc_canonical(name) = replace(string(name), r"^P\." => "")

function _rgc_prepared(ast, data)
    plan = lower_rkppl(ast, (:c, :y1, :y2, :y3);
        conditioned=(:y1, :y2, :y3), mod=@__MODULE__)
    bound = bind_data(plan, data)
    built = build_kernel(bound)
    names = _rgc_canonical.(coordinate_names(built.layout))
    u = [0.2sin(sum(codeunits(name))) for name in names]
    q = prepare_sampler(built, bound, u;
        backend=AutoEnzyme(; mode=Enzyme.Reverse))
    value, grad = sampler_value_and_gradient!(q, similar(u), u)
    return (; plan, bound, built, u, value, names, grad=Dict(zip(names, grad)))
end

@testset "row-gathered matrix columns read the shared result in place" begin
    for n in (1, 7)
        data = Dict{Symbol,Any}(:c => [mod1(i, 4) for i in 1:n],
            :y1 => [0.2sin(i) for i in 1:n],
            :y2 => [0.3cos(i) for i in 1:n],
            :y3 => [0.1sin(2i) for i in 1:n])
        original = deepcopy(data)
        joint = _rgc_prepared(_RGC_JOINT, data)
        direct = _rgc_prepared(_RGC_DIRECT, data)
        @test Set(joint.names) == Set(direct.names)
        @test joint.value ≈ direct.value atol=1e-12
        @test all(isapprox(joint.grad[name], direct.grad[name]; atol=1e-9, rtol=1e-9)
            for name in joint.names)
        code = string(kernel_expr(joint.bound, joint.built.layout))
        @test occursin("_tensorized_view(P", code)
        observations = prepare_query(joint.built, joint.bound, :observations)
        locations = Base.invokelatest(observations, joint.u)
        @test all(getproperty(locations, y).location isa SubArray
            for y in (:y1, :y2, :y3))
        @test isequal(data, original)
    end
end
