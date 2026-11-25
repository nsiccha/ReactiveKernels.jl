using LoweringWorkload, ReactiveKernels, ReactiveKernelsPPL, DifferentiationInterface, Enzyme, Test
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
submodel_entry = W.SUBMODEL_BUILDS.entry
@test submodel_entry !== nothing
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
    built = build_kernel(bound; cache=W.SUBMODEL_BUILDS)
    @test W.SUBMODEL_BUILDS.entry === submodel_entry
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

@testset "package-image constructor binds each fitted child's current graph" begin
    entry = W.BUILDS.entry
    @test entry !== nothing
    data = (; x=[3., 1., 2.], y=[2., 4., 1.])
    plan = W.child_plan([7., 9.], data)
    original = build_kernel(plan; cache=W.BUILDS)
    @test W.BUILDS.entry === entry
    child = W.clipped
    saved = kernel_revision(child)
    @test saved == W.CHILD_REVISION
    old = child.ports[:z]
    try
        child.ports[:z] = child.ports[:x]
        @test kernel_revision(child) != saved
        edited = build_kernel(plan; cache=W.BUILDS)
        @test W.BUILDS.entry === entry
        @test original.spec.graph !== edited.spec.graph
        for (built, increment) in ((original, 1.), (edited, 0.))
            u = [0.5]
            sampler = prepare_sampler(built, plan, u; backend=AutoEnzyme(; mode=Enzyme.Reverse))
            value, gradient = sampler_value_and_gradient!(sampler, similar(u), u)
            residual = data.y .- (data.x .+ 2*(only(u)+increment))
            @test value ≈ -0.5 * (4log(2pi) + (only(u)-8)^2 + sum(abs2, residual))
            @test only(gradient) ≈ 8 - only(u) + 2sum(residual)
        end
    finally
        child.ports[:z] = old
    end
    @test kernel_revision(child) == saved
end
