using DifferentiationInterface, Distributions, Enzyme, ReactiveKernels, ReactiveKernelsPPL, Test
import LogDensityProblems

# A sampler prepared on another binding of one build takes everything but
# that binding's data from the build's earlier bindings: RK's bound template
# (the plan, its data-only prefix and compiled residual), the non-allocating
# program of that residual, and the differentiated program of its gradient.
# Each binding still evaluates its own data prefix, seeds its own buffers and
# owns them. Before, every binding prepared all three afresh: about 0.2 s of
# each batch of RKPPLBench's batched SMC (`update_smc` on a target prepared
# from the cumulative rows), against 5 ms now (snag rkppl-smc-with-d-80655b87).

const _RP_BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)
const _RP_MODEL = @rkppl begin
    beta[1:P] .~ Normal.(0, s0)
    y .~ Normal.(X * beta, 1.0)
end

function _rp_data(n)
    X = hcat(ones(n), sin.(1:n), cos.(0.5 .* (1:n)))
    (; X, y = X * [0.5, -1.0, 2.0] .+ 0.1 .* sin.(3 .* (1:n)))
end
_rp_plan(n) = (d = _rp_data(n); _RP_MODEL(; P = 3, s0 = 2.0, X = d.X) | (; y = d.y))

# Bypass the model memo for the independent construction control.
function _rp_fresh_build(plan)
    layout = assign_layout(plan)
    def = kernel_expr(plan, layout)
    spec = ReactiveKernelsPPL._fresh_build_spec(def, layout)
    (; spec, layout, program = ReactiveKernelsPPL._program_identity(def))
end

_rp_nonallocating(built, plan, u) = Base.invokelatest(ReactiveKernels.prepare_nonallocating,
    ReactiveKernels._kernel_latest_prepared(
        prepare_query(built, plan, :sampler; exemplars = (u,))), u)

@testset "a sampler on another binding of one build reuses its preparation" begin
    u = zeros(3)
    sizes = (40, 100, 7, 250)
    built = build_kernel(_rp_plan(first(sizes)))
    points = ([0.3, -0.2, 1.1], [-1.0, 0.4, 0.0], [2.0, 2.0, -3.0])
    for n in sizes
        plan = _rp_plan(n)
        rebound = prepare_sampler(built, plan, u; backend = _RP_BACKEND)
        own = prepare_sampler(_rp_fresh_build(plan), plan, u; backend = _RP_BACKEND)
        for x in points
            v1, g1 = LogDensityProblems.logdensity_and_gradient(rebound, x)
            v2, g2 = LogDensityProblems.logdensity_and_gradient(own, x)
            @test v1 === v2
            @test g1 == g2
        end
    end

    # The bindings share one compiled residual, one non-allocating program and
    # one differentiated program; each owns its step buffers.
    kernels = [ReactiveKernels._kernel_latest_prepared(
        prepare_query(built, _rp_plan(n), :sampler; exemplars = (u,))) for n in sizes]
    @test all(k -> k.f === first(kernels).f, kernels)
    nonallocating = [_rp_nonallocating(built, _rp_plan(n), u) for n in sizes]
    first_na = first(nonallocating)
    for k in nonallocating[2:end]
        @test getfield(k, :f) === getfield(first_na, :f)
        @test getfield(k, :ast) === getfield(first_na, :ast)
        @test all(i -> getfield(k, :caches)[i] === nothing ||
                       getfield(k, :caches)[i] !== getfield(first_na, :caches)[i],
                  eachindex(getfield(k, :caches)))
    end
    # Interleaved calls keep each binding's values: the buffers are not shared.
    for (k, n) in zip(nonallocating, sizes), x in points
        plan = _rp_plan(n)
        @test Base.invokelatest(k, x) ===
              Base.invokelatest(_rp_nonallocating(_rp_fresh_build(plan), plan, u), x)
    end
    operators = [Base.invokelatest(prepare_ad, k, _RP_BACKEND, u; active = :unconstrained)
                 for k in nonallocating]
    @test all(op -> typeof(op.call) === typeof(first(operators).call), operators)
end

@testset "a retained regression build refreshes bound response scales" begin
    model = @rkppl begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        y .~ Normal.(a .+ b .* x, s)
    end
    function data(n, shift)
        x = cos.(0.3 .* (1:n))
        y = sin.(0.2 .* (1:n))
        s = 0.6 .+ abs.(x .+ shift)
        (; x, y, s)
    end
    # The first pair changes only the scale at the same length. The remaining
    # bindings exercise a different row count and an empty observation.
    bindings = [data(17, 0.0), data(17, 0.8), data(39, -0.3), data(0, 0.0)]
    plans = [model(; d.x, d.s) | (; d.y) for d in bindings]
    u = zeros(2)
    built = build_kernel(first(plans))
    samplers = [prepare_sampler(built, p, u; backend = _RP_BACKEND) for p in plans]
    fresh = [prepare_sampler(_rp_fresh_build(p), p, u; backend = _RP_BACKEND) for p in plans]
    # Hold every sampler live and revisit earlier bindings after later calls.
    for point in ([0.2, -1.2], [-0.7, 0.4]), i in (1, 3, 2, 4, 1)
        d = bindings[i]
        residual = (d.y .- point[1] .- point[2] .* d.x) ./ d.s
        value = -(length(d.y) + 2) * log(2pi) / 2 - sum(log, d.s; init=0.0) -
            (sum(abs2, residual) + sum(abs2, point)) / 2
        grad = [sum(residual ./ d.s), sum(residual .* d.x ./ d.s)] - point
        v, g = LogDensityProblems.logdensity_and_gradient(samplers[i], point)
        vf, gf = LogDensityProblems.logdensity_and_gradient(fresh[i], point)
        @test v === vf
        @test g == gf
        @test v ≈ value
        @test g ≈ grad
    end
    kernels = [ReactiveKernels._kernel_latest_prepared(
        prepare_query(built, p, :sampler; exemplars = (u,))) for p in plans[1:3]]
    @test first(kernels).rebinding !== nothing
    @test all(k -> k.rebinding === first(kernels).rebinding, kernels)
    steps = [_rp_nonallocating(built, p, u) for p in plans[1:3]]
    for k in steps[2:end]
        @test getfield(k, :ast) === getfield(first(steps), :ast)
        @test all(zip(getfield(k, :caches), getfield(first(steps), :caches))) do (a, b)
            a === nothing || a !== b
        end
    end
end

@noinline function _rp_transient_owner()
    model = deepcopy(_RP_MODEL)
    d = _rp_data(11)
    plan = model(; P=3, s0=2.0, X=d.X) | (; y=d.y)
    build_kernel(plan)
    return plan, WeakRef(model.builds)
end

_rp_extra_check(u) = length(u) == 1 || throw(ArgumentError("this build requires one coordinate"))

@testset "repeated model lowerings share sampler work and own their public graphs" begin
    PPL = ReactiveKernelsPPL
    model = deepcopy(_RP_MODEL)
    binding(m, n) = (d = _rp_data(n); m(; P=3, s0=2.0, X=d.X) | (; y=d.y))
    plans = [binding(model, n) for n in (27, 41, 27)]
    # First callers may arrive together. Each still receives its own ports.
    jobs = [Threads.@spawn build_kernel(p) for p in plans]
    builds = fetch.(jobs)
    shared = PPL._sampler_build(first(builds), first(plans)).spec
    @test all(PPL._sampler_build(b, p).spec === shared for (b,p) in zip(builds,plans))
    @test all(b -> b.spec !== shared, builds)
    @test allunique([b.spec.unconstrained.id for b in builds])
    @test builds[1].layout !== builds[2].layout
    u = [0.1, -0.6, 0.3]
    samplers = [prepare_sampler(b,p,u; backend=_RP_BACKEND) for (b,p) in zip(builds,plans)]
    for i in (1,2,3,1)
        d = _rp_data((27,41,27)[i])
        r = d.y - d.X * u
        value, gradient = LogDensityProblems.logdensity_and_gradient(samplers[i],u)
        @test value ≈ -(length(r)+3)*log(2pi)/2 - 3*log(2) - sum(abs2,u)/8 - sum(abs2,r)/2
        @test gradient ≈ d.X' * r - u/4
    end

    # A shape check added by one caller changes that sampler alone.
    ReactiveKernels.input_check!(builds[1].spec, :unconstrained, _rp_extra_check, :unconstrained)
    @test PPL._sampler_build(builds[1],plans[1]).spec === builds[1].spec
    @test_throws "this build requires one coordinate" begin
        edited = prepare_sampler(builds[1],plans[1],u; backend=_RP_BACKEND)
        edited(u)
    end
    @test prepare_sampler(builds[2],plans[2],u; backend=_RP_BACKEND)(u) == samplers[2](u)
    plan_without_model, owner = _rp_transient_owner()
    GC.gc(true)
    @test owner.value === nothing
    @test build_kernel(plan_without_model).layout.total == 3
    other = deepcopy(model)
    @test other.builds.entry === nothing
    otherplan = binding(other, 41)
    otherbuild = build_kernel(otherplan)
    @test PPL._sampler_build(otherbuild,otherplan).spec !== shared
    @test PPL._sampler_build(builds[2],otherplan).spec === builds[2].spec
    # Capture compatibility must use the actual lexical owner. Quoted data
    # and user functions have their normal evaluation and lifetime.
    @test !PPL._build_source!(Dict{GlobalRef,Any}(), QuoteNode([true,false]))
    @test !PPL._build_source!(Dict{GlobalRef,Any}(), QuoteNode(let b=1; x -> b+x end))
    @test !PPL._build_source!(Dict{GlobalRef,Any}(), Expr(:call,GlobalRef(Module(gensym()),:f),1))
end

@testset "a rebound caller-owned kernel is evaluated in its defining module" begin
    scope = Module(gensym(:Rebinding))
    Core.eval(scope, :(using Distributions, ReactiveKernels, ReactiveKernelsPPL))
    model = Core.eval(scope, quote
        @kernel twice(x) = begin
            result = 2 .* x
        end
        @kernel thrice(x) = begin
            result = 3 .* x
        end
        active = twice
        @rkppl begin
            a ~ Normal(0.0, 1.0)
            mu = active(a .+ x)
            y .~ Normal.(mu, 1.0)
        end
    end)
    x, y = [0.3,-0.2,0.7], [0.1,0.6,-0.3]
    function bind_sampler()
        p = model(; x) | (; y)
        prepare_sampler(build_kernel(p), p, [0.0]; backend = _RP_BACKEND)
    end
    first_query = bind_sampler()
    Core.eval(scope, :(active = thrice))
    second_query = bind_sampler()
    for (q,k) in ((first_query,2), (second_query,3), (first_query,2))
        a = 0.7
        r = y .- k .* (a .+ x)
        v, g = LogDensityProblems.logdensity_and_gradient(q,[a])
        @test v ≈ logpdf(Normal(),a) + sum(logpdf.(Normal(),r))
        @test only(g) ≈ k*sum(r) - a
    end
end

@testset "pins supplied at binding time keep the model owner" begin
    model = @rkppl begin
        a ~ Normal(0, 1)
        offset ~ Normal(0, 1)
        y .~ Normal.(a + offset, 1.0)
    end
    y = [-0.4, 0.9]
    builds = map((0.2, -1.3)) do offset
        plan = model(; offset) | (; y)
        @test ReactiveKernelsPPL._structural_cache(plan).model_builds.value === model.builds
        built = build_kernel(plan)
        q = prepare_sampler(built, plan, [0.3]; backend = _RP_BACKEND)
        v, g = LogDensityProblems.logdensity_and_gradient(q, [0.3])
        @test v ≈ logpdf(Normal(), 0.3) + sum(logpdf.(Normal(0.3 + offset), y))
        @test only(g) ≈ sum(y .- (0.3 + offset)) - 0.3
        ReactiveKernelsPPL._sampler_build(built, plan).spec
    end
    @test builds[1] === builds[2]
end
