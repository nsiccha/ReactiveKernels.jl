using Test, ReactiveKernelsPPL, ReactiveKernels, DifferentiationInterface, Enzyme

@testset "one-axis arithmetic producers and an internal indexed read" begin
    model = @rkppl begin
        a ~ Normal(0, 2)
        raw[1:J] .~ Normal.(0, 1)
        b[1:2] .~ Normal.(0, 1)
        alpha = a .+ raw
        mu = alpha[idx] .+ b[1] .* x .+ b[2] .* z
        y .~ Normal.(mu, 1)
    end
    for idx in (Int[], [2], [1, 3, 1, 2, 3, 1])
        n = length(idx)
        x = [0.2 + 0.1i for i in 1:n]
        z = [-0.3 + 0.15i for i in 1:n]
        y = [0.1 + 0.12i for i in 1:n]
        saved = deepcopy((idx, x, z, y))
        plan = model(; J=3, idx, x, z) | (; y)
        built = build_kernel(plan)
        code = sprint(show, Base.remove_linenums!(deepcopy(kernel_expr(plan, built.layout))))
        @test occursin("alpha = plate(", code)
        @test occursin(r"_rkppl_synth_\d+ = ReactiveKernels\._readonly_gather\(alpha, idx\)", code)
        @test occursin("ReactiveKernels._readonly_gather(alpha, idx)", code)
        @test occursin("_ppl_offset_mu = plate(", code)

        a, raw, b = 0.25, [-0.2, 0.4, 0.1], [0.3, -0.15]
        u = unconstrain(built.layout, (; a, raw, b))
        q = prepare_query(built, plan, :sampler)
        sampler = prepare_sampler(built, plan, u;
            backend=AutoEnzyme(; mode=Enzyme.Reverse))
        gradient = similar(u)
        value, _ = sampler_value_and_gradient!(sampler, gradient, u)
        location = (a .+ raw)[idx] .+ b[1] .* x .+ b[2] .* z
        logunit(v) = -0.5v^2 - 0.5log(2pi)
        expected = logunit(a/2) - log(2) + sum(logunit, raw) +
            sum(logunit, b) + sum(logunit.(y .- location))
        @test value ≈ expected rtol=1e-12
        @test q(u) ≈ expected rtol=1e-12
        for k in eachindex(u)
            h = 1e-6
            up, down = copy(u), copy(u)
            up[k] += h
            down[k] -= h
            @test gradient[k] ≈ (q(up) - q(down))/(2h) atol=2e-6
        end
        @test (idx, x, z, y) == saved
    end
end

@testset "two generated county gathers fuse into the likelihood" begin
    model = @rkppl begin
        alpha_raw[1:J] .~ Normal.(0, 1)
        beta_raw[1:J] .~ Normal.(0, 1)
        mu_alpha ~ Normal(0, 10)
        mu_beta ~ Normal(0, 10)
        sigma ~ HalfNormal(1)
        alpha = mu_alpha .+ alpha_raw
        beta = mu_beta .+ beta_raw
        mu = alpha[idx] .+ x .* beta[idx]
        y .~ Normal.(mu, sigma)
    end
    named_gathers = @rkppl begin
        alpha_raw[1:J] .~ Normal.(0, 1)
        beta_raw[1:J] .~ Normal.(0, 1)
        mu_alpha ~ Normal(0, 10)
        mu_beta ~ Normal(0, 10)
        sigma ~ HalfNormal(1)
        alpha = mu_alpha .+ alpha_raw
        beta = mu_beta .+ beta_raw
        selected_alpha = alpha[idx]
        selected_beta = beta[idx]
        mu = selected_alpha .+ x .* selected_beta
        y .~ Normal.(mu, sigma)
    end
    for idx in (Int[], [2], [2, 1, 3, 2, 3, 1])
        x = [0.2 - 0.07i for i in eachindex(idx)]
        y = [-0.1 + 0.09i for i in eachindex(idx)]
        saved = deepcopy((idx, x, y))
        plan = model(; J=3, idx, x) | (; y)
        reference = named_gathers(; J=3, idx, x) | (; y)
        built = build_kernel(plan)
        reference_built = build_kernel(reference)
        @test coordinate_names(built.layout) == coordinate_names(reference_built.layout)
        params = (; alpha_raw=[-0.2, 0.4, 0.1], beta_raw=[0.3, -0.15, 0.25],
            mu_alpha=0.25, mu_beta=-0.35, sigma=1.1)
        u = collect(Float64, unconstrain(built.layout, params))
        unchanged_u = copy(u)
        backend = AutoEnzyme(; mode=Enzyme.Reverse)
        sampler = prepare_sampler(built, plan, u; backend)
        control = prepare_sampler(reference_built, reference, u; backend)
        gradient, control_gradient = similar(u), similar(u)
        value, _ = sampler_value_and_gradient!(sampler, gradient, u)
        control_value, _ = sampler_value_and_gradient!(control, control_gradient, u)
        @test value ≈ control_value rtol=1e-12
        @test gradient ≈ control_gradient rtol=1e-12 atol=1e-12
        @test (idx, x, y) == saved
        @test u == unchanged_u

        source = sprint(show, Base.remove_linenums!(deepcopy(kernel_expr(plan, built.layout))))
        @test occursin(r"_rkppl_synth_\d+ = plate\(", source)
        @test occursin("ReactiveKernels._readonly_gather(beta, idx)", source)
        emitted = sprint(show, Base.remove_linenums!(deepcopy(code_expr(sampler.kernel))))
        if !isempty(idx)
            @test length(collect(eachmatch(r"_plate_similar_output", emitted))) == 2
        end
    end
end

@testset "an authored indexed array may borrow read-only storage" begin
    model = @rkppl begin
        a ~ Normal(0, 1)
        raw[1:3] .~ Normal.(0, 1)
        alpha = a .+ raw
        selected = alpha[idx]
        mu = selected .+ x
        y .~ Normal.(mu, 1)
    end
    plan = model(; idx=[1, 3, 1], x=[0.2, 0.4, 0.6]) | (; y=[0.1, 0.2, 0.3])
    built = build_kernel(plan)
    code = sprint(show, Base.remove_linenums!(deepcopy(kernel_expr(plan, built.layout))))
    @test occursin("selected = ReactiveKernels._readonly_gather(alpha, idx)", code)
end

_pointwise_primal_bytes(q, u) = @allocated q(u)
_pointwise_reverse_bytes(q, g, u) = @allocated sampler_value_and_gradient!(q, g, u)

@testset "direct parameter gathers share read-only views with queries" begin
    model = @rkppl begin
        center ~ Normal(0, 2)
        effects[1:J] .~ Normal.(center, 1)
        slope[1:2] .~ Normal.(0, 1)
        mu = effects[idx] .+ slope[1] .* x .+ slope[2] .* z
        y .~ Normal.(mu, 1)
    end
    sizes = Int[]
    loops = Int[]
    for idx in (Int[], [2], [3, 1, 3, 2, 1, 3], mod1.(1:128, 3))
        n = length(idx)
        x, z, y = [0.1i for i in 1:n], [-0.05i for i in 1:n], [0.2i for i in 1:n]
        saved = deepcopy((idx, x, z, y))
        plan = model(; J=3, idx, x, z) | (; y)
        built = build_kernel(plan)
        source = sprint(show, Base.remove_linenums!(deepcopy(kernel_expr(plan, built.layout))))
        @test occursin("ReactiveKernels._readonly_gather(effects, idx)", source)
        gather = only(d for d in plan.derived if d.expr == :(effects[idx]))
        @test occursin("$(gather.name) = ReactiveKernels._readonly_gather(effects, idx)", source)
        @test only(e for e in built.layout.entries if e.name === :effects).offset > 1
        query = ReactiveKernelsPPL._prepare_wants(built, plan, (gather.name, :posterior))
        q = prepare_query(built, plan, :sampler)
        u = collect(unconstrain(built.layout,
            (; center=0.3, effects=[-0.2, 0.4, 0.1], slope=[0.15, -0.25])))
        sampler = prepare_sampler(built, plan, u;
            backend=AutoEnzyme(; mode=Enzyme.Reverse))
        gradient = similar(u)
        for shift in (0.0, 0.2)
            point = u .+ shift
            unchanged = copy(point)
            center, effects, slope = 0.3 + shift, [-0.2, 0.4, 0.1] .+ shift,
                [0.15, -0.25] .+ shift
            location = effects[idx] .+ slope[1] .* x .+ slope[2] .* z
            logunit(v) = -0.5v^2 - 0.5log(2pi)
            expected = logunit(center/2) - log(2) + sum(logunit, effects .- center) +
                sum(logunit, slope) + sum(logunit, y .- location; init=0.0)
            value, _ = sampler_value_and_gradient!(sampler, gradient, point)
            @test value ≈ expected rtol=1e-12
            selected, queried_value = Base.invokelatest(query, point)
            @test selected isa SubArray
            @test selected == effects[idx]
            @test isempty(idx) || Base.mightalias(selected, point)
            @test queried_value ≈ value rtol=1e-12
            @test point == unchanged
            @test (idx, x, z, y) == saved
            for k in eachindex(point)
                up, down = copy(point), copy(point)
                up[k] += 1e-6
                down[k] -= 1e-6
                @test gradient[k] ≈ (q(up) - q(down))/2e-6 rtol=2e-6 atol=2e-6
            end
        end
        if n > 1
            emitted = sprint(show, Base.remove_linenums!(deepcopy(code_expr(sampler.kernel))))
            push!(sizes, length(collect(eachmatch(r"_plate_similar_output", emitted))))
            push!(loops, length(collect(eachmatch(r"for ", emitted))))
            @test _pointwise_primal_bytes(sampler.kernel, u) == 0
            @test _pointwise_reverse_bytes(sampler, gradient, u) == 0
        end
    end
    @test sizes == [0, 0]
    @test loops[1] == loops[2] > 0
    for idx in ([0], [4])
        # Invalid positional gathers are rejected at binding, before evaluation.
        @test_throws ContractValidationError model(; J=3, idx, x=[0.2], z=[0.3]) | (; y=[0.1])
    end
end

@testset "gather substitution commits only with a recognized consumer" begin
    model = @rkppl begin
        effects[1:3] .~ Normal.(0, 1)
        mu = effects[idx] .+ x
        y .~ Normal.(mu, 1)
    end
    plan = model(; idx=[3, 1, 3], x=[0.2, 0.4, 0.6]) | (; y=[0.1, 0.2, 0.3])
    layout = assign_layout(plan)
    raw = ReactiveKernelsPPL._kernel_expr_latest(plan, layout)
    gather = only(d for d in plan.derived if d.expr == :(effects[idx]))
    name = gather.name
    for use in (:(identity($name)), :($name .+ unknown), :($name .+ x))
        def = deepcopy(raw)
        consumer = only(st for st in def.args[2].args if Meta.isexpr(st, :(=), 2) &&
            ReactiveKernelsPPL._pointwise_occurrences(st.args[2], name) > 0)
        consumer.args[2] = use
        if use == :($name .+ x)
            # An array recovered from a query container still has a concrete
            # representation. The arithmetic use may independently borrow.
            append!(def.args[2].args, [:(grouped = ($name,)),
                :(extracted = grouped[1]), :(extra = typeof(extracted))])
        end
        actual = ReactiveKernelsPPL._pointwise_arithmetic_plates(plan, def)
        text = sprint(show, actual)
        @test occursin("$name = effects[idx]", text)
        @test occursin("_readonly_gather", text) == (use == :($name .+ x))
    end
end

@testset "authored names do not restrict read-only gathers" begin
    model = @rkppl begin
        effects[1:3] .~ Normal.(0, 1)
        _rkppl_synth_7 = effects[idx]
        mu = _rkppl_synth_7 .+ x
        y .~ Normal.(mu, 1)
    end
    plan = model(; idx=[3, 1, 3], x=[0.2, 0.4, 0.6]) | (; y=[0.1, 0.2, 0.3])
    built = build_kernel(plan)
    source = sprint(show, kernel_expr(plan, built.layout))
    @test occursin("_rkppl_synth_7 = ReactiveKernels._readonly_gather(effects, idx)", source)
    query = ReactiveKernelsPPL._prepare_wants(built, plan, :_rkppl_synth_7)
    u = [-0.1, 0.2, 0.3]
    selected = Base.invokelatest(query, u)
    @test selected == u[[3, 1, 3]]
    @test selected isa SubArray
    @test Base.mightalias(selected, u)
end

@testset "read-only contiguous slice gather keeps bounds and offset" begin
    packed = [99.0, 98.0, 1.0, 2.0, 3.0, 97.0]
    block = view(packed, 3:5)
    for idx in (Int[], [2], [3, 1, 3])
        saved = copy(idx)
        selected = ReactiveKernels._readonly_gather(block, idx)
        @test selected == block[idx]
        @test parent(selected) === block
        @test idx == saved
    end
    # Invalid reads must retain Julia's BoundsError; in particular, index 0
    # must not be offset into the preceding packed parameter.
    for idx in ([0], [-1], [4], [1, 4, 2])
        @test_throws BoundsError ReactiveKernels._readonly_gather(block, idx)
    end
    @test packed == [99.0, 98.0, 1.0, 2.0, 3.0, 97.0]
end

_gather_vector_only(x::Vector{Float64}) = 2 .* x

@testset "named gathers with several consumers and representation boundaries" begin
    model = @rkppl begin
        effects[1:3] .~ Normal.(0, 1)
        selected = effects[idx]
        left = selected .+ x
        right = selected .* z
        mu = left .+ right
        y .~ Normal.(mu, 1)
    end
    concrete = @rkppl begin
        effects[1:3] .~ Normal.(0, 1)
        selected = effects[idx]
        alias = selected
        transformed = _gather_vector_only(alias)
        mu = transformed .+ x
        y .~ Normal.(mu, 1)
    end
    idx, x, z, y = [3, 1, 3], [0.2, 0.4, 0.6], [-0.1, 0.2, -0.3], [0.1, 0.2, 0.3]
    plan = model(; idx, x, z) | (; y)
    built = build_kernel(plan)
    source = sprint(show, kernel_expr(plan, built.layout))
    @test occursin("selected = ReactiveKernels._readonly_gather(effects, idx)", source)
    query = ReactiveKernelsPPL._prepare_wants(built, plan, (:selected, :left, :right, :posterior))
    u = [-0.1, 0.2, 0.3]
    selected, left, right, value = Base.invokelatest(query, u)
    @test selected isa SubArray
    @test selected == u[idx]
    @test left ≈ u[idx] .+ x
    @test right ≈ u[idx] .* z
    changed = u .+ 0.7
    selected2, left2, right2, value2 = Base.invokelatest(query, changed)
    @test selected2 == changed[idx]
    @test selected == u[idx] # a later call does not overwrite an earlier view
    @test left2 ≈ changed[idx] .+ x
    @test right2 ≈ changed[idx] .* z
    @test value2 != value
    plan2 = concrete(; idx, x) | (; y)
    built2 = build_kernel(plan2)
    source2 = sprint(show, kernel_expr(plan2, built2.layout))
    @test occursin("selected = effects[idx]", source2)
    query2 = ReactiveKernelsPPL._prepare_wants(built2, plan2, (:selected, :transformed))
    selected_copy, transformed = Base.invokelatest(query2, u)
    @test selected_copy isa Vector{Float64}
    @test !Base.mightalias(selected_copy, u)
    @test transformed == 2 .* u[idx]
end
