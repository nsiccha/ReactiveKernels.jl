using DifferentiationInterface
using Distributions
using Enzyme
using LinearAlgebra
using ReactiveKernels
using ReactiveKernelsPPL
using Test

# Declared array parameters (`ArrayParameter`): `L ~ LKJCholesky(K, eta)`,
# sized `z[1:K] .~ ...` / `z[levels(g)] .~ ...` / two-axis
# `z[levels(g), 1:K] .~ ...`, and bound data matrices, all read as plain
# values (`L[2, 1]`, `z[1]`, `z[g]`, `B * w`); a free simplex (a
# `VectorParameter` read as a model-level value) is read by position too.
# Densities are checked against Distributions.jl, gradients against
# central differences. Helpers `_query` / `_check_gradient` come from
# test_generator.jl (included earlier).

_av_y() = [0.3, -1.2, 2.1, 0.7, -0.4, 1.5, 0.2, -0.8]
_av_x() = [0.5, -1.0, 1.5, 0.0, -0.5, 1.0, 2.0, -1.5]
_av_x2() = [1.0, 0.25, -0.75, 2.0, -1.0, 0.5, 0.0, 1.25]
_av_g() = ["b", "a", "c", "a", "b", "c", "a", "b"]
_av_B() = hcat(_av_x(), _av_x2(), ones(8))

# Fixed off-origin unconstrained points (no randomness in the oracles).
_av_point(n) = [0.37 * sin(1.3 * i) - 0.2 for i in 1:n]

_av_node(built, bound, node, u) = _query(built.spec, bound, node, u)

@testset "array values: positional scalar offsets" begin
    for kind in (:vector, :matrix), named in (false, true), sign in (1, -1)
        declaration, read = kind === :vector ?
            (:(z[1:3] .~ Normal.(0, 1)), :(z[2])) :
            (:(L ~ LKJCholesky(2, 1.0)), :(L[2, 1]))
        definition = named ? :(r = $read) : nothing
        value = named ? :r : read
        location = sign > 0 ? :(a .+ $value .+ x) : :(a .- $value .+ x)
        ast = quote
            a ~ Normal(0, 5)
            sigma ~ Exponential(1)
            $declaration
            $definition
            mu = $location
            y .~ Normal.(mu, sigma)
        end
        filter!(!isnothing, ast.args)
        plan = lower_rkppl(ast, (:y, :x); conditioned = (:y, :x))
        @test isempty(plan.derived)
        @test length(plan.assignments) == 1
        @test only(plan.predictors).terms[2].kind === OffsetTerm
        for n in (3, 8)
            y, x = _av_y()[1:n], _av_x()[1:n]
            bound = bind_data(plan, Dict(:y => y, :x => x))
            built = build_kernel(bound)
            @test built.layout.total == (kind === :vector ? 5 : 3)
            u = _av_point(built.layout.total)
            nt = constrain(built.layout, u)
            offset = kind === :vector ? nt.z[2] : nt.L[2, 1]
            array_prior = kind === :vector ? sum(logpdf.(Normal(), nt.z)) :
                logpdf(LKJCholesky(2, 1.0), Cholesky(LowerTriangular(nt.L)))
            prior = array_prior + logpdf(Normal(0, 5), nt.a) +
                logpdf(Exponential(1), nt.sigma)
            likelihood = sum(logpdf.(Normal.(nt.a .+ sign * offset .+ x,
                nt.sigma), y))
            @test _av_node(built, bound, :prior, u) ≈ prior
            @test _av_node(built, bound, :likelihood, u) ≈ likelihood
            @test _av_node(built, bound, :posterior, u) ≈
                prior + likelihood + logjac(built.layout, u)
            _check_gradient(built.spec, bound, u)
        end
    end
end

@testset "array values: composed positional scalar offsets" begin
    # Multiple reads, an array-valued definition and no coefficient.
    ast = quote
        z[1:3] .~ Normal.(0, 1)
        M = (2 .* z)'
        mu = x .+ M[1, 2] .- z[1] .+ z[2]
        nu = x2
        y .~ Normal.(mu, 0.7)
        y2 .~ Normal.(nu, 0.7)
    end
    y, x = _av_y(), _av_x()
    y2, x2 = _av_y(), _av_x2()
    bound = bind_data(lower_rkppl(ast, (:y, :x, :y2, :x2); conditioned = (:y, :x, :y2, :x2)),
        Dict(:y => y, :x => x, :y2 => y2, :x2 => x2))
    built = build_kernel(bound)
    @test built.layout.total == 3
    u = _av_point(3)
    z = constrain(built.layout, u).z
    want = sum(logpdf.(Normal(), z)) +
        sum(logpdf.(Normal.(x .+ 3z[2] .- z[1], 0.7), y)) +
        sum(logpdf.(Normal.(x2, 0.7), y2))
    @test _av_node(built, bound, :posterior, u) ≈ want
    _check_gradient(built.spec, bound, u)
end

@testset "array values: LKJCholesky factor read as a matrix" begin
    m = @rkppl begin
        L ~ LKJCholesky(3, 2.0)
        a ~ Normal(0, 1)
        sigma ~ Exponential(1)
        w = L[2, 1] .* x .+ L[3, 2] .* x2
        mu = a .+ w
        y .~ Normal.(mu, sigma)
    end
    y, x, x2 = _av_y(), _av_x(), _av_x2()
    bound = (m(; x, x2) | (; y))
    @test only(bound.array_parameters).family === :lkj_cholesky
    built = build_kernel(bound)
    @test built.layout.total == 3 + 2
    u = _av_point(built.layout.total)
    nt = constrain(built.layout, u)
    L = nt.L
    @test L isa Matrix{Float64} && size(L) == (3, 3)
    @test istril(L)
    @test all(i -> norm(L[i, :]) ≈ 1, 1:3)
    @test unconstrain(built.layout, nt) ≈ u
    a = nt.a  # the declared scalar intercept
    prior = logpdf(LKJCholesky(3, 2.0), Cholesky(LowerTriangular(L))) +
        logpdf(Normal(0, 1), a) + logpdf(Exponential(1), nt.sigma)
    @test _av_node(built, bound, :prior, u) ≈ prior
    lik = sum(logpdf.(Normal.(a .+ L[2, 1] .* x .+ L[3, 2] .* x2,
        nt.sigma), y))
    @test _av_node(built, bound, :likelihood, u) ≈ lik
    @test _av_node(built, bound, :log_jacobian, u) ≈ logjac(built.layout, u)
    _check_gradient(built.spec, bound, u)
end

@testset "array values: LKJCholesky density matches Distributions" begin
    # The prior node alone (no other parameters) over several K and eta,
    # including eta == 1 (the uniform-correlation branch) and K == 1.
    for (K, eta) in ((1, 2.0), (2, 1.0), (2, 0.7), (3, 1.0), (4, 3.5))
        ast = :(begin
            L ~ LKJCholesky($K, $eta)
            a ~ Normal(0, 1)
            sigma ~ Exponential(1)
            w = L[$K, 1] .* x
            mu = a .+ w
            y .~ Normal.(mu, sigma)
        end)
        bound = bind_data(lower_rkppl(ast, (:y, :x); conditioned = (:y, :x)),
            Dict{Symbol,ColumnData}(:y => _av_y(), :x => _av_x()))
        built = build_kernel(bound)
        u = _av_point(built.layout.total)
        nt = constrain(built.layout, u)
        want = logpdf(LKJCholesky(K, eta), Cholesky(LowerTriangular(nt.L))) +
            logpdf(Normal(0, 1), nt.a) +
            logpdf(Exponential(1), nt.sigma)
        @test _av_node(built, bound, :prior, u) ≈ want
    end
end

@testset "array values: sized vector with per-element scales and B * w" begin
    m = @rkppl begin
        tau ~ HalfNormal(1)
        lambda[1:3] .~ HalfCauchy.(1)
        w[axes(B, 2)] .~ Normal.(0, lambda .* tau)
        s[1:3] .~ Normal.([0.0, 1.0, -1.0], [1.0, 2.0, 0.5])
        a ~ Normal(0, 1)
        sigma ~ Exponential(1)
        v = s[2] .* x
        mu = a .+ B * w .+ v
        y .~ Normal.(mu, sigma)
    end
    y, x, B = _av_y(), _av_x(), _av_B()
    bound = (m(; x, B) | (; y))
    @test Set(p.name for p in bound.array_parameters) == Set([:lambda, :w, :s])
    built = build_kernel(bound)
    @test built.layout.total == 1 + 3 + 3 + 3 + 2
    u = _av_point(built.layout.total)
    nt = constrain(built.layout, u)
    @test nt.lambda isa Vector{Float64} && length(nt.lambda) == 3
    @test all(>(0), nt.lambda)
    @test unconstrain(built.layout, nt) ≈ u
    prior = logpdf(truncated(Normal(0, 1), 0, Inf), nt.tau) +
        sum(logpdf.(truncated.(Cauchy.(0, 1), 0, Inf), nt.lambda)) +
        sum(logpdf.(Normal.(0, nt.lambda .* nt.tau), nt.w)) +
        sum(logpdf.(Normal.([0.0, 1.0, -1.0], [1.0, 2.0, 0.5]), nt.s)) +
        logpdf(Normal(0, 1), nt.a) + logpdf(Exponential(1), nt.sigma)
    @test _av_node(built, bound, :prior, u) ≈ prior
    lik = sum(logpdf.(Normal.(nt.a .+ B * nt.w .+ nt.s[2] .* x,
        nt.sigma), y))
    @test _av_node(built, bound, :likelihood, u) ≈ lik
    @test _av_node(built, bound, :log_jacobian, u) ≈ logjac(built.layout, u)
    _check_gradient(built.spec, bound, u)
end

@testset "array values: free simplex read by position" begin
    # The simplex stays a VectorParameter (a model-level value, as for
    # `cumsum(phi)`); reading it by position needs no array parameter.
    m = @rkppl begin
        phi ~ Dirichlet([2.0, 1.0, 3.0])
        a ~ Normal(0, 1)
        b ~ Normal(0, 2)
        sigma ~ Exponential(1)
        w = phi[1] .* x .+ phi[2] .* x2 .+ phi[3]
        mu = a .+ b .* w
        y .~ Normal.(mu, sigma)
    end
    y, x, x2 = _av_y(), _av_x(), _av_x2()
    bound = (m(; x, x2) | (; y))
    @test isempty(bound.array_parameters)
    @test only(bound.vector_parameters).name === :phi
    built = build_kernel(bound)
    u = _av_point(built.layout.total)
    nt = constrain(built.layout, u)
    @test sum(nt.phi) ≈ 1 && all(>(0), nt.phi)
    prior = logpdf(Dirichlet([2.0, 1.0, 3.0]), nt.phi) +
        logpdf(Normal(0, 1), nt.a) + logpdf(Normal(0, 2), nt.b) +
        logpdf(Exponential(1), nt.sigma)
    @test _av_node(built, bound, :prior, u) ≈ prior
    w = nt.phi[1] .* x .+ nt.phi[2] .* x2 .+ nt.phi[3]
    lik = sum(logpdf.(Normal.(nt.a .+ nt.b .* w, nt.sigma), y))
    @test _av_node(built, bound, :likelihood, u) ≈ lik
    @test _av_node(built, bound, :log_jacobian, u) ≈ logjac(built.layout, u)
    _check_gradient(built.spec, bound, u)
end

@testset "array values: level gather and a two-axis array" begin
    # Non-centered varying intercept and a correlated two-margin effect:
    # `z[g]` / `Z[g, j]` look each observation's level up on the
    # `levels(g)` axis (sorted: "a", "b", "c").
    m = @rkppl begin
        tau ~ HalfNormal(1)
        z[levels(g)] .~ Normal.(0, 1)
        L ~ LKJCholesky(2, 2.0)
        sd[1:2] .~ HalfNormal.(1)
        Z[levels(g), 1:2] .~ Normal.(0, 1)
        M = (sd .* L)'
        a ~ Normal(0, 1)
        sigma ~ Exponential(1)
        r = tau .* z[g] .+ Z[g, :] * M[:, 1] .+ (Z[g, :] * M[:, 2]) .* x
        mu = a .+ r
        y .~ Normal.(mu, sigma)
    end
    y, x, g = _av_y(), _av_x(), _av_g()
    bound = (m(; x, g) | (; y))
    built = build_kernel(bound)
    u = _av_point(built.layout.total)
    nt = constrain(built.layout, u)
    @test size(nt.Z) == (3, 2)
    @test unconstrain(built.layout, nt) ≈ u
    codes = [findfirst(==(v), ["a", "b", "c"]) for v in g]
    M = (nt.sd .* nt.L)'
    @test M ≈ (Diagonal(nt.sd) * nt.L)'
    r = nt.tau .* nt.z[codes] .+ nt.Z[codes, :] * M[:, 1] .+
        (nt.Z[codes, :] * M[:, 2]) .* x
    lik = sum(logpdf.(Normal.(nt.a .+ r, nt.sigma), y))
    @test _av_node(built, bound, :likelihood, u) ≈ lik
    prior = logpdf(truncated(Normal(0, 1), 0, Inf), nt.tau) +
        sum(logpdf.(Normal(0, 1), nt.z)) +
        logpdf(LKJCholesky(2, 2.0), Cholesky(LowerTriangular(nt.L))) +
        sum(logpdf.(truncated(Normal(0, 1), 0, Inf), nt.sd)) +
        sum(logpdf.(Normal(0, 1), nt.Z)) +
        logpdf(Normal(0, 1), nt.a) + logpdf(Exponential(1), nt.sigma)
    @test _av_node(built, bound, :prior, u) ≈ prior
    @test _av_node(built, bound, :log_jacobian, u) ≈ logjac(built.layout, u)
    _check_gradient(built.spec, bound, u)
    # Coordinates follow Stan's matrix naming (`Z.i.j`, column-major).
    names = coordinate_names(built.layout)
    @test Symbol("Z.2.1") in names && Symbol("Z.1.2") in names
    @test findfirst(==(Symbol("Z.2.1")), names) <
        findfirst(==(Symbol("Z.1.2")), names)
end

# Model-module helpers for whole-value calls over arrays (`group_sums`
# needs one value per group: a model-level call over the whole array).
module ArrayRowsModels
group_sums(B) = vec(sum(B; dims = 2))
addvec(a, b) = a .+ b
end

@testset "array values: column reads passed to a module call" begin
    # `Z[:, j]` is the j-th column (one value per level); passed to a
    # model-level call whose result is gathered per observation.
    ast = :(begin
        s ~ Exponential(1)
        Z[levels(k), 1:2] .~ Normal.(0, 1)
        v = addvec(Z[:, 1], Z[:, 2])
        y .~ Normal.(v[k], s)
    end)
    y = _av_y()
    k = [1, 2, 3, 1, 2, 3, 1, 2]
    bound = bind_data(lower_rkppl(ast, (:y, :k); mod = ArrayRowsModels, conditioned = (:y, :k)),
        Dict{Symbol,ColumnData}(:y => y, :k => k))
    built = build_kernel(bound)
    u = _av_point(built.layout.total)
    nt = constrain(built.layout, u)
    v = nt.Z[:, 1] .+ nt.Z[:, 2]
    @test _av_node(built, bound, :likelihood, u) ≈
        sum(logpdf.(Normal.(v[k], nt.s), y))
    _check_gradient(built.spec, bound, u)
end

@testset "array values: row-wise MvNormalCholesky (centered correlated)" begin
    # Every row of `B` (one per level of `g`) is one draw of
    # `MvNormal(mu, F * F')`; the entries of `B` are the coordinates.
    m = @rkppl begin
        L ~ LKJCholesky(2, 2.0)
        sd[1:2] .~ HalfNormal.(1)
        F = sd .* L
        eachrow(B[levels(g), 1:2]) .~ MvNormalCholesky([0.5, -0.25], F)
        a ~ Normal(0, 1)
        sigma ~ Exponential(1)
        r = B[g, 1] .+ B[g, 2] .* x
        mu = a .+ r
        y .~ Normal.(mu, sigma)
    end
    y, x, g = _av_y(), _av_x(), _av_g()
    bound = (m(; x, g) | (; y))
    p = only(q for q in bound.array_parameters if q.name === :B)
    @test p.family === :mvnormal_cholesky_rows
    built = build_kernel(bound)
    @test built.layout.total == 1 + 2 + 6 + 1 + 1
    u = _av_point(built.layout.total)
    nt = constrain(built.layout, u)
    @test nt.B isa Matrix{Float64} && size(nt.B) == (3, 2)
    @test unconstrain(built.layout, nt) ≈ u
    # Centered: the entries of `B` are its coordinates (column-major).
    names = coordinate_names(built.layout)
    iB = findfirst(==(Symbol("B.1.1")), names)
    @test names[iB:iB + 5] == Symbol.(["B.1.1", "B.2.1", "B.3.1", "B.1.2",
        "B.2.2", "B.3.2"])
    @test u[iB:iB + 5] == vec(nt.B)
    F = nt.sd .* nt.L
    @test F ≈ Diagonal(nt.sd) * nt.L
    rowprior = sum(logpdf(MvNormal([0.5, -0.25], F * F'), nt.B[j, :])
        for j in 1:3)
    prior = logpdf(LKJCholesky(2, 2.0), Cholesky(LowerTriangular(nt.L))) +
        sum(logpdf.(truncated(Normal(0, 1), 0, Inf), nt.sd)) + rowprior +
        logpdf(Normal(0, 1), nt.a) + logpdf(Exponential(1), nt.sigma)
    @test _av_node(built, bound, :prior, u) ≈ prior
    codes = [findfirst(==(v), ["a", "b", "c"]) for v in g]
    lik = sum(logpdf.(Normal.(nt.a .+ nt.B[codes, 1] .+
        nt.B[codes, 2] .* x, nt.sigma), y))
    @test _av_node(built, bound, :likelihood, u) ≈ lik
    @test _av_node(built, bound, :log_jacobian, u) ≈ logjac(built.layout, u)
    _check_gradient(built.spec, bound, u)
end

@testset "array values: row-wise array read whole by a module call" begin
    # The whole levels × K matrix reaches a model-level call (one value per
    # group), whose result is gathered per observation.
    ast = :(begin
        s ~ Exponential(1)
        L ~ LKJCholesky(3, 2.0)
        sd[1:3] .~ Exponential.(1)
        F = sd .* L
        eachrow(B[levels(k), 1:3]) .~ MvNormalCholesky(zeros(3), F)
        v = group_sums(B)
        y .~ Normal.(v[k], s)
    end)
    y = _av_y()
    k = [1, 2, 3, 1, 2, 3, 1, 2]
    bound = bind_data(lower_rkppl(ast, (:y, :k); mod = ArrayRowsModels, conditioned = (:y, :k)),
        Dict{Symbol,ColumnData}(:y => y, :k => k))
    built = build_kernel(bound)
    @test built.layout.total == 1 + 3 + 3 + 9
    u = _av_point(built.layout.total)
    nt = constrain(built.layout, u)
    F = nt.sd .* nt.L
    prior = logpdf(Exponential(1), nt.s) +
        logpdf(LKJCholesky(3, 2.0), Cholesky(LowerTriangular(nt.L))) +
        sum(logpdf.(Exponential(1), nt.sd)) +
        sum(logpdf(MvNormal(zeros(3), F * F'), nt.B[j, :]) for j in 1:3)
    @test _av_node(built, bound, :prior, u) ≈ prior
    v = vec(sum(nt.B; dims = 2))
    @test _av_node(built, bound, :likelihood, u) ≈
        sum(logpdf.(Normal.(v[k], nt.s), y))
    _check_gradient(built.spec, bound, u)
end

@testset "array values: row-wise prior equals the centered conditional chain" begin
    # The same centered model spelled two ways: rows of `B` drawn jointly,
    # and margin by margin (`b2 | b1`). Both pack the levels × 2 values
    # column-major after the same scalars, so the posteriors agree at
    # every point.
    rows = :(begin
        s ~ Exponential(1)
        L ~ LKJCholesky(2, 2.0)
        sd[1:2] .~ HalfNormal.(1)
        F = sd .* L
        eachrow(B[levels(k), 1:2]) .~ MvNormalCholesky(zeros(2), F)
        v = addvec(B[:, 1], B[:, 2])
        y .~ Normal.(v[k], s)
    end)
    chain = :(begin
        s ~ Exponential(1)
        L ~ LKJCholesky(2, 2.0)
        sd[1:2] .~ HalfNormal.(1)
        b1[levels(k)] .~ Normal.(0, sd[1])
        b2[levels(k)] .~ Normal.(L[2, 1] .* sd[2] ./ sd[1] .* b1,
            sd[2] .* L[2, 2])
        v = addvec(b1, b2)
        y .~ Normal.(v[k], s)
    end)
    data = Dict{Symbol,ColumnData}(:y => _av_y(),
        :k => [1, 2, 3, 1, 2, 3, 1, 2])
    br, bc = (bind_data(lower_rkppl(ast, (:y, :k); mod = ArrayRowsModels, conditioned = (:y, :k)),
        data) for ast in (rows, chain))
    kr, kc = build_kernel(br), build_kernel(bc)
    @test kr.layout.total == kc.layout.total == 10
    u = _av_point(10)
    @test _av_node(kr, br, :posterior, u) ≈ _av_node(kc, bc, :posterior, u)
    @test _av_node(kr, br, :prior, u) ≈ _av_node(kc, bc, :prior, u)
end

@testset "array values: row-wise MvNormalCholesky fail closed" begin
    bindm(ast, data) = bind_data(lower_rkppl(ast, Tuple(keys(data)); conditioned = Tuple(keys(data))),
        Dict{Symbol,ColumnData}(pairs(data)))
    y, x, g = _av_y(), _av_x(), _av_g()
    prog(decl) = :(begin
        L ~ LKJCholesky(2, 2.0)
        sd[1:2] .~ HalfNormal.(1)
        F = sd .* L
        $decl
        a ~ Normal(0, 1)
        r = B[g, 1] .* x
        mu = a .+ r
        y .~ Normal.(mu, 1.0)
    end)
    # refused: `eachrow` makes the rows the broadcast elements, so the
    # statement broadcasts (`.~`); `~` would claim one draw for the whole
    # matrix (standard-Julia semantics, @rkppl principle 3).
    @test_throws SurfaceLoweringError lower_rkppl(prog(:(eachrow(B[levels(g),
        1:2]) ~ MvNormalCholesky(zeros(2), F))), (:y, :x, :g); conditioned = (:y, :x, :g))
    # refused: a row is a vector, so a univariate family cannot draw it
    # (standard-Julia semantics, principle 3); elementwise priors are
    # `B[a, b] .~ Fam.(...)`.
    @test_throws SurfaceLoweringError lower_rkppl(prog(:(eachrow(B[levels(g),
        1:2]) .~ Normal(0, 1))), (:y, :x, :g); conditioned = (:y, :x, :g))
    # refused: eachrow of a vector produces length-1 rows, which disagree with the length-2 multivariate prior (P3 dimension contract)
    @test_throws SurfaceLoweringError lower_rkppl(prog(:(eachrow(B[1:2]) .~
        MvNormalCholesky(zeros(2), F))), (:y, :x, :g); conditioned = (:y, :x, :g))
    # refused: the mean of a multivariate normal is a vector, as in
    # Distributions' `MvNormal` (principle 3); a zero mean is `zeros(K)`.
    @test_throws SurfaceLoweringError lower_rkppl(prog(:(eachrow(B[levels(g),
        1:2]) .~ MvNormalCholesky(0, F))), (:y, :x, :g); conditioned = (:y, :x, :g))
    # A mean that is not a K-vector: here a column the predictor reads
    # per observation (8 values for rows of length 2).
    # refused: row, mean and covariance-factor dimensions must agree (distribution domain, P3)
    @test_throws ContractValidationError bindm(prog(:(eachrow(B[levels(g),
        1:2]) .~ MvNormalCholesky(x, F))), (; y, x, g))
    # A literal mean whose length differs from the rows.
    # refused: row, mean and covariance-factor dimensions must agree (distribution domain, P3)
    @test_throws ContractValidationError bindm(prog(:(eachrow(B[levels(g),
        1:2]) .~ MvNormalCholesky([0.0, 0.0, 0.0], F))), (; y, x, g))
    # A declared factor whose size differs from the rows.
    # refused: row, mean and covariance-factor dimensions must agree (distribution domain, P3)
    @test_throws ContractValidationError bindm(:(begin
        L ~ LKJCholesky(3, 2.0)
        eachrow(B[levels(g), 1:2]) .~ MvNormalCholesky(zeros(2), L)
        a ~ Normal(0, 1)
        r = B[g, 1] .* x
        mu = a .+ r
        y .~ Normal.(mu, 1.0)
    end), (; y, x, g))
end

@testset "array values: an integer axis gathers by position" begin
    m = @rkppl begin
        z[1:3] .~ Normal.(0, 1)
        a ~ Normal(0, 1)
        sigma ~ Exponential(1)
        mu = a .+ z[k]
        y .~ Normal.(mu, sigma)
    end
    k = [1, 2, 3, 1, 2, 3, 3, 2]
    bound = (m(; k) | (; y = _av_y()))
    built = build_kernel(bound)
    u = _av_point(built.layout.total)
    nt = constrain(built.layout, u)
    lik = sum(logpdf.(Normal.(nt.a .+ nt.z[k], nt.sigma), _av_y()))
    @test _av_node(built, bound, :likelihood, u) ≈ lik
    _check_gradient(built.spec, bound, u)
end

@testset "array values: fail closed" begin
    bindm(ast, data) = bind_data(lower_rkppl(ast, Tuple(keys(data)); conditioned = Tuple(keys(data))),
        Dict{Symbol,ColumnData}(pairs(data)))
    y, x = _av_y(), _av_x()
    # `~` on a sized declaration.
    # refused: scalar ~ on a sized array declaration; ~ is scalar, .~ broadcasts (P3)
    @test_throws SurfaceLoweringError lower_rkppl(:(begin
        z[1:3] ~ Normal(0, 1)
        y .~ Normal.(z[1], 1.0)
    end), (:y,); conditioned = (:y,))
    # A bare array combined with per-observation data.
    # capability: array parameters broadcast with observation data using
    # ordinary Julia dimensions (todo `15lq8iu`).
    @test (lower_rkppl(:(begin
        z[1:3] .~ Normal.(0, 1)
        a ~ Normal(0, 1)
        w = z .+ x
        mu = a .+ w
        y .~ Normal.(mu, 1.0)
    end), (:y, :x); conditioned = (:y, :x)); true)
    # Undotted elementwise math over an array is a Julia error.
    # refused: undotted scalar + vector is a Julia MethodError (P3)
    @test_throws SurfaceLoweringError lower_rkppl(:(begin
        z[1:3] .~ Normal.(0, 1)
        s ~ Exponential(1)
        v = s + z
        a ~ Normal(0, 1)
        w = v[1] .* x
        mu = a .+ w
        y .~ Normal.(mu, 1.0)
    end), (:y, :x); conditioned = (:y, :x))
    # LKJCholesky: non-literal eta, upper factor.
    # A sampled scalar is an ordinary LKJCholesky prior argument (P8).
    @test (lower_rkppl(:(begin
        e ~ Exponential(1)
        L ~ LKJCholesky(2, e)
        a ~ Normal(0, 1)
        w = L[2, 1] .* x
        mu = a .+ w
        y .~ Normal.(mu, 1.0)
    end), (:y, :x); conditioned = (:y, :x)); true)
    # LKJCholesky's stated orientation is retained in the matrix value.
    @test (lower_rkppl(:(begin
        L ~ LKJCholesky(2, 2.0, 'U')
        a ~ Normal(0, 1)
        w = L[1, 2] .* x
        mu = a .+ w
        y .~ Normal.(mu, 1.0)
    end), (:y, :x); conditioned = (:y, :x)); true)
    # A data matrix whose columns do not match the array.
    # refused: B * w column/length mismatch is a Julia DimensionMismatch (wrong data)
    @test_throws ContractValidationError bindm(:(begin
        w[1:2] .~ Normal.(0, 1)
        a ~ Normal(0, 1)
        mu = a .+ B * w
        y .~ Normal.(mu, 1.0)
    end), (; y, B = _av_B()))
    # A vector where `B * w` needs a matrix.
    # refused: vector * vector for B * w is a Julia MethodError (wrong data, P3)
    @test_throws ContractValidationError bindm(:(begin
        w[1:3] .~ Normal.(0, 1)
        a ~ Normal(0, 1)
        mu = a .+ B * w
        y .~ Normal.(mu, 1.0)
    end), (; y, B = x))
    # Out-of-bounds literal position.
    # refused: out-of-bounds literal index z[4] (BoundsError)
    @test_throws ContractValidationError bindm(:(begin
        z[1:3] .~ Normal.(0, 1)
        a ~ Normal(0, 1)
        w = z[4] .* x
        mu = a .+ w
        y .~ Normal.(mu, 1.0)
    end), (; y, x))
    # A gather by values not on the levels axis.
    # refused: gather by values not on the levels axis (invalid index)
    @test_throws ContractValidationError bindm(:(begin
        z[levels(g)] .~ Normal.(0, 1)
        a ~ Normal(0, 1)
        w = z[h] .* x
        mu = a .+ w
        y .~ Normal.(mu, 1.0)
    end), (; y, x, g = _av_g(), h = ["a", "b", "z", "a", "b", "c", "a", "b"]))
    # An integer axis gathered by non-integers.
    # refused: integer axis gathered by non-integers (non-integer index)
    @test_throws ContractValidationError bindm(:(begin
        z[1:3] .~ Normal.(0, 1)
        a ~ Normal(0, 1)
        mu = a .+ z[g]
        y .~ Normal.(mu, 1.0)
    end), (; y, g = _av_g()))
    # Per-element literal arguments of the wrong length.
    # refused: per-element prior args of the wrong length (DimensionMismatch)
    @test_throws ContractValidationError bindm(:(begin
        z[1:3] .~ Normal.(0, [1.0, 2.0])
        a ~ Normal(0, 1)
        w = z[1] .* x
        mu = a .+ w
        y .~ Normal.(mu, 1.0)
    end), (; y, x))
end
