using DifferentiationInterface
using Enzyme
using ReactiveKernels
using ReactiveKernelsPPL
using Test

const _DERIVED_AFFINE_MODEL = @rkppl begin
    a ~ Normal(0, 2)
    b ~ Normal(0, 3)
    lambda ~ Uniform(0.5, 1)
    mu = a .- b .* (lambda .^ x)
    y .~ Normal.(mu, s)
end

const _AFFINE_GATHER_OFFSET = @rkppl begin
    b ~ Normal(0, 1)
    z[1:J] .~ Normal.(0, 1)
    mu = b .* x .+ z[group]
    y .~ Normal.(mu, 0.9)
end

const _AFFINE_TRANSFORMED_OFFSET = @rkppl begin
    b ~ Normal(0, 1)
    a ~ Normal(0, 1)
    z[1:J] .~ Normal.(0, 1)
    level = a .+ z
    mu = level[group] .+ b .* x
    y .~ Normal.(mu, 0.9)
end

_offset_expr_nodes(x) = x isa Expr ?
    1 + sum(_offset_expr_nodes, x.args; init=0) : 0

@testset "scalar-column predictors fuse vector offsets" begin
    for (model, transformed) in ((_AFFINE_GATHER_OFFSET, false),
            (_AFFINE_TRANSFORMED_OFFSET, true))
        structures = Int[]
        for (n, J) in ((0, 3), (1, 3), (48, 3), (384, 7))
            group = [mod1(div(i-1, 16)+1, J) for i in 1:n]
            x = [0.2 + sin(i/11) for i in 1:n]
            y = [cos(i/9)/2 for i in 1:n]
            pristine = deepcopy((group, x, y))
            plan = model(; J, group, x) | (; y)
            built = build_kernel(plan)
            source = kernel_expr(plan, built.layout)
            @test !occursin("_ppl_design_mu", string(source))
            @test occursin("mu = plate(", string(source))
            n >= 48 && push!(structures, _offset_expr_nodes(source))
            for shift in (0.0, 0.15)
                z = [0.2sin(i) + shift for i in 1:J]
                b, a = 0.3-shift, transformed ? -0.2+shift : 0.0
                params = transformed ? (; b, a, z) : (; b, z)
                u = collect(Float64, unconstrain(built.layout, params))
                saved_u = copy(u)
                q = prepare_sampler(built, plan, u;
                    backend=AutoEnzyme(; mode=Enzyme.Reverse),
                    native_forward_batch=false)
                @test q.ad.backend.mode === Enzyme.Reverse
                mu = b .* x .+ a .+ z[group]
                residual = (y .- mu) ./ 0.9^2
                gz = -copy(z)
                for i in eachindex(group)
                    gz[group[i]] += residual[i]
                end
                gb = -b + sum(residual .* x; init=0.0)
                ga = -a + sum(residual; init=0.0)
                gradients = transformed ? (; b=gb, a=ga, z=gz) : (; b=gb, z=gz)
                expected_g = collect(Float64, unconstrain(built.layout, gradients))
                logunit(v) = -0.5v^2 - 0.5log(2pi)
                expected = logunit(b) + sum(logunit, z; init=0.0) +
                    (transformed ? logunit(a) : 0.0) +
                    sum(v -> logunit(v/0.9)-log(0.9), y .- mu; init=0.0)
                g = similar(u)
                value, _ = sampler_value_and_gradient!(q, g, u)
                @test value ≈ expected rtol=1e-12 atol=1e-12
                @test q(u) ≈ expected rtol=1e-12 atol=1e-12
                @test g ≈ expected_g rtol=1e-11 atol=1e-11
                @test u == saved_u
                @test (group, x, y) == pristine
                if n == 48 && shift == 0.0
                    observations = prepare_query(built, plan, :observations)
                    @test Base.invokelatest(observations, u).y.location ≈ mu
                end
            end
        end
        @test all(==(first(structures)), structures)
        @test_throws ContractValidationError model(;
            J=3, group=fill(0, 16), x=zeros(16)) | (; y=zeros(16))
    end
end

function _derived_affine_findiff(f, u)
    h = cbrt(eps(Float64))
    [begin
        up, down = copy(u), copy(u)
        up[i] += h
        down[i] -= h
        (f(up) - f(down)) / (2h)
    end for i in eachindex(u)]
end

@testset "derived numeric column in affine response" begin
    for n in (4, 17)
        x = collect(0.5:0.25:(0.5 + 0.25 * (n - 1)))
        y = 1 .+ sin.(x)
        s = 0.7
        plan = _DERIVED_AFFINE_MODEL(; x, s) | (; y)
        built = build_kernel(plan)
        u = [1.5, 0.4, -0.3]
        values = constrain(built.layout, u)
        expected_mu = values.a .- values.b .* (values.lambda .^ x)
        expected_pw = @. -0.5 * log(2π) - log(s) -
            0.5 * ((y - expected_mu) / s)^2

        obs = prepare_query(built, plan, :observations)
        pw = prepare_query(built, plan, :pointwise)
        @test Base.invokelatest(obs, u).y.location ≈ expected_mu
        @test Base.invokelatest(pw, u).y ≈ expected_pw

        named = prepare(built.spec; have = (:unconstrained, :s, :x, :y),
            want = :mu, bound = (; s, x, y), exemplars = (u,))
        @test Base.invokelatest(named, u) ≈ expected_mu

        q = prepare_sampler(built, plan, u;
            backend = AutoEnzyme(; mode = Enzyme.Reverse))
        g = similar(u)
        value, _ = sampler_value_and_gradient!(q, g, u)
        @test value ≈ q(u)
        @test all(isfinite, g)
        @test isapprox(g, _derived_affine_findiff(q, u);
            rtol = 1e-5, atol = 1e-7)
    end
end
