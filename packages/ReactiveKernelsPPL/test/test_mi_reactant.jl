# Compiled (Reactant) checks for the cases in test_mi.jl.
using Reactant

@testset "mi gaussian under Reactant" begin
    plan = _mi_gaussian_plan()
    built = build_kernel(plan)
    u = [0.5, -0.25, 0.1]
    post_q = Base.invokelatest(prepare_query, built, plan, :sampler)
    native = Base.invokelatest(post_q, u)
    compiled = Reactant.@compile post_q(Reactant.to_rarray(u))
    @test Float64(compiled(Reactant.to_rarray(u))) ≈ native
    q = prepare_sampler(built, plan, u; backend = _MI_BACKEND)
    g = similar(u)
    val, _ = sampler_value_and_gradient!(q, g, u)
    cad = compile_ad_value_and_gradient(q.ad, Reactant.to_rarray(u))
    rval, rgrad = cad(Reactant.to_rarray(u))
    @test Float64(rval) ≈ val
    @test Array(rgrad) ≈ g
end
