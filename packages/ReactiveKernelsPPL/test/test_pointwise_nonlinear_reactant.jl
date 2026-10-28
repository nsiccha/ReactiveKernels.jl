using Test, ReactiveKernelsPPL, ReactiveKernels, DifferentiationInterface, Enzyme, Reactant
using Distributions: Normal, LogNormal

@testset "nonlinear pointwise likelihood keeps one compiled loop" begin
    model = @rkppl begin
        effects[1:3] .~ Normal.(0, 1)
        scale ~ LogNormal(0, 0.5)
        power ~ LogNormal(0, 0.5)
        mu = effects[idx] .* exp.(-((x ./ scale) .^ power))
        y .~ Normal.(mu, 1.2)
    end
    fixtures = map((3, 7)) do n
        idx = mod1.(1:n, 3)
        x = [0.2 + 0.03i for i in 1:n]
        y = [-0.4 + 0.02i for i in 1:n]
        plan = model(; idx, x) | (; y)
        built = build_kernel(plan)
        u = collect(unconstrain(built.layout,
            (; effects = [-0.2, 0.3, 0.1], scale = 1.1, power = 1.4)))
        query = prepare_query(built, plan, :sampler)
        code = sprint(show, Base.remove_linenums!(deepcopy(code_expr(query))))
        (; plan, built, u, query,
            loops = length(collect(eachmatch(r"for ", code))))
    end
    @test fixtures[1].loops == fixtures[2].loops > 0

    f = first(fixtures)
    native = f.query(f.u)
    traced = Reactant.to_rarray(f.u)
    compiled = Reactant.@compile f.query(traced)
    @test Float64(compiled(traced)) ≈ native rtol = 1e-9

    sampler = prepare_sampler(f.built, f.plan, f.u;
        backend = AutoEnzyme(; mode = Enzyme.Reverse))
    gradient = similar(f.u)
    native_value, _ = sampler_value_and_gradient!(sampler, gradient, f.u)
    compiled_ad = compile_ad_value_and_gradient(sampler.ad, traced)
    compiled_value, compiled_gradient = compiled_ad(traced)
    @test Float64(compiled_value) ≈ native_value rtol = 1e-9
    @test Array(compiled_gradient) ≈ gradient rtol = 1e-7 atol = 1e-9
end
