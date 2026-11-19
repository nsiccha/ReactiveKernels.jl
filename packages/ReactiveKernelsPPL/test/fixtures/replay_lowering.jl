using LoweringWorkload, ReactiveKernelsPPL, DifferentiationInterface, Enzyme, Test
const W = LoweringWorkload
@test W.IN_IMAGE
@test !ReactiveKernelsPPL._lowering_named_module(W.DEFINITIONS)
entry = only(values(W.CACHE.entries))
@test isempty(entry.plan.frozen.values) && isempty(entry.plan.frozen.inputs)
@test isempty(entry.plan.columns) && entry.plan.data_values === nothing
@test isempty(entry.plan.submodel_scopes)
fit = W.fitted([9., 10., 11., 12., 13.])
@test fit.values.center == 11.0
programs = Expr[]
for design in ((; x=[2., -1., 3., 0.], y=[1., 0., -0.2, 0.4]),
               (; x=[4., 2.], y=[0.2, 0.3]))
    # Mark the image entry in this test process. Only an actual hit can copy
    # the marker. Remove it before evaluating the independently owned result.
    marker = :from_package_image => :from_package_image
    push!(entry.plan.named_values, marker)
    plan = W.lower(design, fit)
    @test pop!(entry.plan.named_values) == marker
    @test pop!(plan.named_values) == marker
    @test plan.frozen === fit
    @test !isempty(plan.submodel_scopes)
    bound = bind_data(plan, design)
    built = build_kernel(bound)
    fresh = bind_data(W.lower(design, fit; cache=nothing), design)
    @test built.program == build_kernel(fresh).program
    push!(programs, built.program)
    a, b = 0.1, 0.8
    z = design.x .- 11
    u = [a, b]
    locations = Base.invokelatest(prepare_query(built, bound, :observations), u).y.location
    @test locations ≈ a .+ b .* z
    r = design.y .- locations
    sampler = prepare_sampler(built, bound, u; backend=AutoEnzyme(; mode=Enzyme.Reverse))
    value, grad = sampler_value_and_gradient!(sampler, similar(u), u)
    @test value ≈ -0.5*((2+length(r))*log(2pi) + a^2 + b^2 + sum(abs2, r))
    @test grad ≈ [sum(r)-a, sum(r .* z)-b]
    plans = fetch.([Threads.@spawn W.lower(design, fit) for _ in 1:4])
    @test all(p -> p.responses !== plan.responses, plans)
    empty!(first(plans).responses)
    @test all(p -> !isempty(p.responses), plans[2:end])
end
@test allequal(programs)
