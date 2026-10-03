using Distributions: MvNormal, logpdf
using Enzyme
using LinearAlgebra
using ReactiveKernels, ReactiveKernelsPPL, Reactant, Test
using ReactiveKernelsPPL: _lower_solve_rows_logpdf
using ReactiveKernelsPPL: _mvnormal_cholesky_slices_logpdf, _mvnormal_slices_logpdf, _SliceRows
using DifferentiationInterface: AutoEnzyme

@kernel _mv_checked_cholesky_test(X, mu, F) =
    _mvnormal_cholesky_slices_logpdf(_SliceRows(), X, mu, F)
@kernel _mv_checked_covariance_test(X, mu, Sigma) =
    _mvnormal_slices_logpdf(_SliceRows(), X, mu, Sigma)

# The ordinary mathematical core has already validated factors at its
# callers. Test it directly so these checks also cover data-sized margins,
# independently of the model's factor transform or invalid-domain policy.
@kernel _mv_solve_test(X, F) = begin
    logdet = sum(log.(diag(F)))
    lp = _lower_solve_rows_logpdf(X, F, logdet)
    return lp
end

function _mv_solve_ops(hlo)
    ops = Dict{String,Int}()
    for m in eachmatch(r"\b(?:stablehlo|chlo|func|arith|enzyme)\.\w+", hlo)
        ops[m.match] = get(ops, m.match, 0) + 1
    end
    return ops
end

function _mv_generated_measure(built, bound, u)
    post = prepare_query(built, bound, :sampler)
    native = post(u)
    ru = Reactant.to_rarray(u)
    hlo = repr(Reactant.@code_hlo optimize = false post(ru))
    compiled = Reactant.@compile post(ru)
    @test Float64(compiled(ru)) ≈ native rtol = 1e-10
    sampler = prepare_sampler(built, bound, u;
        backend = AutoEnzyme(; mode = Enzyme.Reverse))
    g = similar(u)
    value, _ = sampler_value_and_gradient!(sampler, g, u)
    @test value ≈ native rtol = 1e-12
    grad(p) = Enzyme.gradient(Enzyme.Reverse, post, p)
    reverse = repr(Reactant.@code_hlo grad(ru))
    cad = compile_ad_value_and_gradient(sampler.ad, ru)
    rvalue, rgrad = cad(ru)
    @test Float64(rvalue) ≈ native rtol = 1e-10
    @test Array(rgrad) ≈ g rtol = 1e-8 atol = 1e-9
    @test Array(ru) == u
    return _mv_solve_ops(hlo), _mv_solve_ops(reverse)
end

@testset "generated centered multivariate queries and samplers" begin
    for covariance in (false, true)
        prior = covariance ? :(MvNormal(zeros(2), Sigma)) :
                             :(MvNormalCholesky(zeros(2), F))
        plan = lower_rkppl(quote
            a ~ Normal(0., 1.)
            sd[1:2] .~ Exponential.(2.)
            L ~ LKJCholesky(2, 1.5)
            F = sd .* L
            Sigma = F * F'
            eachrow(B[levels(group), 1:2]) .~ $prior
            mu = a .+ B[group, 1] .+ x .* B[group, 2]
            y .~ Normal.(mu, 1.5)
        end, (:y, :group, :x); conditioned = (:y, :group, :x))
        maps, reverse_maps = [], []
        for (G, N) in ((5, 10), (8, 24))
            groups = [mod1(i, G) for i in 1:N]
            x = [.2sin(i) for i in 1:N]
            columns = Dict{Symbol,AbstractVector}(:group => groups, :x => x,
                :y => [.3cos(i) for i in 1:N])
            bound = bind_data(plan, columns)
            built = build_kernel(bound)
            u = [.2sin(i + .7) for i in 1:built.layout.total]
            primal, reverse = Base.invokelatest(_mv_generated_measure, built, bound, u)
            push!(maps, primal)
            push!(reverse_maps, reverse)
        end
        @test maps[1] == maps[2]
        @test reverse_maps[1] == reverse_maps[2]
    end
end

@testset "compiled factor and covariance errors remain runtime checks" begin
    X = [.2 -.3; .1 .4; -.5 .3]
    mu = [.1, -.2]
    F = [.9 0.; .15 1.1]
    for (spec, covariance) in ((_mv_checked_cholesky_test, false),
                               (_mv_checked_covariance_test, true))
        factor = covariance ? F * F' : F
        kernel = prepare(spec)
        ignored = prepare(spec; on_error = :ignore)
        rx, rm, rf = Reactant.to_rarray(X), Reactant.to_rarray(mu), Reactant.to_rarray(factor)
        hlo = repr(Reactant.@code_hlo optimize = false kernel(rx, rm, rf))
        ignored_hlo = repr(Reactant.@code_hlo optimize = false ignored(rx, rm, rf))
        @test occursin("reactant_julia_callback", hlo)
        @test !occursin("reactant_julia_callback", ignored_hlo)
        compiled = Reactant.@compile kernel(rx, rm, rf)
        grad(X, mu, factor) = Enzyme.gradient(Enzyme.Reverse, kernel, X, mu, factor)
        compiled_grad = Reactant.@compile grad(rx, rm, rf)
        reference = kernel(X, mu, factor)
        @test ignored(X, mu, factor) == reference
        bad_diagonal = copy(factor)
        bad_diagonal[1, 1] = 0.0
        bad_triangle = copy(factor)
        bad_triangle[1, 2] += 0.02
        bad_domain = covariance ? [1. 2.; 2. 1.] : [.9 0.; .15 NaN]
        for bad in (bad_diagonal, bad_triangle, bad_domain)
            # refused: factors must have a positive diagonal and zero upper
            # triangle; covariances must be symmetric positive definite.
            @test_throws ArgumentError kernel(X, mu, bad)
            rbad = Reactant.to_rarray(bad)
            # Reactant propagates a callback's exception as its runtime error.
            @test_throws Reactant.XLA.ReactantInternalError compiled(rx, rm, rbad)
            @test_throws Reactant.XLA.ReactantInternalError compiled_grad(rx, rm, rbad)
            @test Array(rbad) == bad || isequal(Array(rbad), bad)
            @test Float64(compiled(rx, rm, rf)) ≈ reference rtol = 1e-10
        end
        @test Array(rx) == X && Array(rm) == mu && Array(rf) == factor
        @test_throws DimensionMismatch kernel(X, mu, zeros(3, 3))
        @test_throws DimensionMismatch kernel(X, zeros(3), factor)
    end
end

_mv_solve_gradient(k, X, F) = Enzyme.gradient(Enzyme.Reverse, k, X, F)

@testset "multivariate validation with default compiled execution" begin
    for spec in (_mv_checked_cholesky_test, _mv_checked_covariance_test)
        kernel = prepare(spec)
        maps, reverse_maps = [], []
        for (K, G) in ((2, 3), (5, 8), (8, 13))
            X = [.2sin(i + 3j) for i in 1:G, j in 1:K]
            mu = [.05j for j in 1:K]
            F = [i == j ? .8 + .05i : i > j ? .07cos(i-j) : 0. for i in 1:K, j in 1:K]
            Sigma = F * F'
            factor = spec === _mv_checked_cholesky_test ? F : Sigma
            reference = sum(logpdf(MvNormal(mu, Sigma), X[g, :]) for g in 1:G)
            @test kernel(X, mu, factor) ≈ reference rtol = 1e-12
            rx, rm, rf = Reactant.to_rarray(X), Reactant.to_rarray(mu), Reactant.to_rarray(factor)
            hlo = repr(Reactant.@code_hlo optimize = false kernel(rx, rm, rf))
            push!(maps, _mv_solve_ops(hlo))
            compiled = Reactant.@compile kernel(rx, rm, rf)
            @test Float64(compiled(rx, rm, rf)) ≈ reference rtol = 1e-10
            grad(X, mu, factor) = Enzyme.gradient(Enzyme.Reverse, kernel, X, mu, factor)
            native = grad(X, mu, factor)
            # Inspect the working, default reverse pipeline above its
            # small-loop unrolling threshold as both data dimensions grow.
            if K > 4
                reverse = repr(Reactant.@code_hlo grad(rx, rm, rf))
                @test occursin("stablehlo.while", reverse)
                push!(reverse_maps, _mv_solve_ops(reverse))
            end
            compiled_grad = Reactant.@compile grad(rx, rm, rf)
            result = compiled_grad(rx, rm, rf)
            @test Array(result[1]) ≈ native[1] rtol = 1e-9 atol = 1e-11
            @test Array(result[2]) ≈ native[2] rtol = 1e-9 atol = 1e-11
            @test Array(result[3]) ≈ native[3] rtol = 1e-9 atol = 1e-11
            @test Array(rx) == X && Array(rm) == mu && Array(rf) == factor
        end
        @test maps[1] == maps[2] == maps[3]
        @test reverse_maps[1] == reverse_maps[2]
    end
end

@testset "multivariate forward substitution: retained primal and reverse" begin
    k = prepare(_mv_solve_test)
    primal_ops, reverse_ops = Dict{String,Int}[], Dict{String,Int}[]
    for (margins, slices) in ((2, 3), (5, 8))
        X = [.2sin(i + 3j) for i in 1:slices, j in 1:margins]
        F = [i == j ? .8 + .05i : i > j ? .07cos(i - j) : 0.
            for i in 1:margins, j in 1:margins]
        old_X, old_F = copy(X), copy(F)
        reference = sum(logpdf(MvNormal(F * F'), X[g, :]) for g in 1:slices)
        @test k(X, F) ≈ reference rtol = 4e-14
        rx, rf = Reactant.to_rarray(X), Reactant.to_rarray(F)
        hlo = repr(Reactant.@code_hlo optimize = false k(rx, rf))
        @test count("stablehlo.while", hlo) == 1
        push!(primal_ops, _mv_solve_ops(hlo))
        compiled = Reactant.@compile k(rx, rf)
        @test Float64(compiled(rx, rf)) ≈ reference rtol = 1e-10
        native_grad = _mv_solve_gradient(k, X, F)
        grad(X, F) = _mv_solve_gradient(k, X, F)
        reverse = repr(Reactant.@code_hlo optimize = :only_enzyme grad(rx, rf))
        @test occursin("stablehlo.while", reverse)
        push!(reverse_ops, _mv_solve_ops(reverse))
        compiled_grad = Reactant.@compile grad(rx, rf)
        result = compiled_grad(rx, rf)
        @test Array(result[1]) ≈ native_grad[1] rtol = 1e-9 atol = 1e-11
        @test Array(result[2]) ≈ native_grad[2] rtol = 1e-9 atol = 1e-11
        @test X == old_X && F == old_F
        @test Array(rx) == old_X && Array(rf) == old_F
    end
    # Shapes specialize; neither axis replicates the primal or AD body.
    @test primal_ops[1] == primal_ops[2]
    @test reverse_ops[1] == reverse_ops[2]

    # Empty batches are the empty sum and do not index a zero-width solve
    # buffer. Native reverse returns empty/zero gradients; Reactant 0.2.290
    # cannot export the empty gradient (benchmark/repro_reactant_empty_gradient.jl).
    X, F = zeros(0, 2), Matrix{Float64}(I, 2, 2)
    rx, rf = Reactant.to_rarray(X), Reactant.to_rarray(F)
    @test k(X, F) == 0.0
    compiled = Reactant.@compile k(rx, rf)
    @test Float64(compiled(rx, rf)) == 0.0
    empty_grad(X, F) = _mv_solve_gradient(k, X, F)
    native_grad = empty_grad(X, F)
    @test native_grad[1] == X
    @test native_grad[2] == zeros(2, 2)
    @test_throws r"'tensor.empty' op unsupported op for export to XLA" Reactant.@compile empty_grad(rx, rf)
end
