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
        @test occursin(r"_rkppl_synth_\d+ = ReactiveKernels\._tensorized_view\(alpha, idx\)", code)
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
    copied_gathers = @rkppl begin
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
        reference = copied_gathers(; J=3, idx, x) | (; y)
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
        @test occursin("ReactiveKernels._tensorized_view(beta, idx)", source)
        emitted = sprint(show, Base.remove_linenums!(deepcopy(code_expr(sampler.kernel))))
        if !isempty(idx)
            @test length(collect(eachmatch(r"_plate_similar_output", emitted))) == 2
        end
    end
end

@testset "an authored indexed array retains its copy" begin
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
    @test occursin("selected = alpha[idx]", code)
end
