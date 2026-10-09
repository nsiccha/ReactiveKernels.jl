using DifferentiationInterface
using Distributions
using Enzyme
using LinearAlgebra
using ReactiveKernels
using ReactiveKernelsPPL
using Test

# Multivariate slice priors on declared arrays: `eachrow(B[a, b]) .~ D`,
# `eachcol(B[a, b]) .~ D` and `b[1:K] ~ D`, for `D` one of
# `MvNormalCholesky`, `MvNormal`, `Dirichlet` and `Ordered(Normal(…), K)`,
# with shared (undotted, `Ref`) and per-slice (`eachrow(M)` / `eachcol(M)`)
# arguments. Densities are checked against Distributions.jl, gradients
# against central differences. Helpers come from test_generator.jl and
# test_array_values.jl (included earlier); the row-wise
# `MvNormalCholesky` basics live in test_array_values.jl.

module SliceModels
rowsums(B) = vec(sum(B; dims = 2))
colsums(B) = vec(sum(B; dims = 1))
end

const _SL_K = [1, 2, 3, 1, 2, 3, 1, 2]

function _sl_check(ast, refprior)
    data = Dict{Symbol,ColumnData}(:y => _av_y(), :k => _SL_K)
    bound = bind_data(lower_rkppl(ast, (:y, :k); mod = SliceModels, conditioned = (:y, :k)), data)
    built = build_kernel(bound)
    u = _av_point(built.layout.total)
    nt = constrain(built.layout, u)
    @test unconstrain(built.layout, nt) ≈ u
    @test _av_node(built, bound, :prior, u) ≈ refprior(nt)
    @test _av_node(built, bound, :log_jacobian, u) ≈ logjac(built.layout, u)
    _check_gradient(built.spec, bound, u)
    return (; bound, built, nt, u)
end

_sl_cov(nt) = (nt.sd .* nt.L) * (nt.sd .* nt.L)'
_sl_base(nt) = logpdf(Exponential(1), nt.s) +
    logpdf(LKJCholesky(3, 2.0), Cholesky(LowerTriangular(nt.L))) +
    sum(logpdf.(Exponential(1), nt.sd))

@testset "array slices: eachcol MvNormalCholesky" begin
    r = _sl_check(:(begin
        s ~ Exponential(1)
        L ~ LKJCholesky(3, 2.0)
        sd[1:3] .~ Exponential.(1)
        F = sd .* L
        eachcol(B[1:3, levels(k)]) .~ MvNormalCholesky(zeros(3), F)
        v = colsums(B)
        y .~ Normal.(v[k], s)
    end), nt -> _sl_base(nt) +
        sum(logpdf(MvNormal(zeros(3), _sl_cov(nt)), nt.B[:, j]) for j in 1:3))
    p = only(q for q in r.bound.array_parameters if q.name === :B)
    @test p.family === :mvnormal_cholesky_cols
    @test size(r.nt.B) == (3, 3)
    # Centered: the entries are the coordinates (column-major).
    names = coordinate_names(r.built.layout)
    i = findfirst(==(Symbol("B.1.1")), names)
    @test r.u[i:i + 8] == vec(r.nt.B)
end

@testset "array slices: MvNormal with a full covariance" begin
    _sl_check(:(begin
        s ~ Exponential(1)
        L ~ LKJCholesky(3, 2.0)
        sd[1:3] .~ Exponential.(1)
        F = sd .* L
        S = F * F'
        eachrow(B[levels(k), 1:3]) .~ MvNormal([0.5, 0.0, -0.5], S)
        v = rowsums(B)
        y .~ Normal.(v[k], s)
    end), nt -> _sl_base(nt) + sum(logpdf(MvNormal([0.5, 0.0, -0.5],
        Symmetric(_sl_cov(nt))), nt.B[j, :]) for j in 1:3))
end

@testset "array slices: per-slice mean" begin
    # `eachrow(M)` pairs row g of B with row g of M; `Ref(F)` shares F.
    _sl_check(:(begin
        s ~ Exponential(1)
        L ~ LKJCholesky(3, 2.0)
        sd[1:3] .~ Exponential.(1)
        F = sd .* L
        M[levels(k), 1:3] .~ Normal.(0, 1)
        eachrow(B[levels(k), 1:3]) .~ MvNormalCholesky.(eachrow(M), Ref(F))
        v = rowsums(B)
        y .~ Normal.(v[k], s)
    end), nt -> _sl_base(nt) + sum(logpdf.(Normal(0, 1), nt.M)) +
        sum(logpdf(MvNormal(nt.M[j, :], _sl_cov(nt)), nt.B[j, :])
            for j in 1:3))
    # Columns of a K × G matrix pair with rows of a G × K one.
    _sl_check(:(begin
        s ~ Exponential(1)
        L ~ LKJCholesky(3, 2.0)
        sd[1:3] .~ Exponential.(1)
        F = sd .* L
        M[1:3, levels(k)] .~ Normal.(0, 1)
        eachrow(B[levels(k), 1:3]) .~ MvNormalCholesky.(eachcol(M), Ref(F))
        v = rowsums(B)
        y .~ Normal.(v[k], s)
    end), nt -> _sl_base(nt) + sum(logpdf.(Normal(0, 1), nt.M)) +
        sum(logpdf(MvNormal(nt.M[:, j], _sl_cov(nt)), nt.B[j, :])
            for j in 1:3))
end

@testset "array slices: one multivariate normal vector" begin
    r = _sl_check(:(begin
        s ~ Exponential(1)
        L ~ LKJCholesky(3, 2.0)
        sd[1:3] .~ Exponential.(1)
        F = sd .* L
        b[1:3] ~ MvNormalCholesky(zeros(3), F)
        y .~ Normal.(b[k], s)
    end), nt -> _sl_base(nt) + logpdf(MvNormal(zeros(3), _sl_cov(nt)), nt.b))
    @test r.nt.b isa Vector{Float64} && length(r.nt.b) == 3
    @test only(q for q in r.bound.array_parameters
        if q.name === :b).family === :mvnormal_cholesky_vector
    # One value per level, drawn jointly (a full covariance), gathered by
    # level.
    r = _sl_check(:(begin
        s ~ Exponential(1)
        L ~ LKJCholesky(3, 2.0)
        sd[1:3] .~ Exponential.(1)
        F = sd .* L
        S = F * F'
        b[levels(k)] ~ MvNormal([0.5, 0.0, -0.5], S)
        y .~ Normal.(b[k], s)
    end), nt -> _sl_base(nt) +
        logpdf(MvNormal([0.5, 0.0, -0.5], Symmetric(_sl_cov(nt))), nt.b))
    @test _av_node(r.built, r.bound, :likelihood, r.u) ≈
        sum(logpdf.(Normal.(r.nt.b[_SL_K], r.nt.s), _av_y()))
end

@testset "array slices: Dirichlet rows and columns" begin
    r = _sl_check(:(begin
        s ~ Exponential(1)
        eachrow(P[levels(k), 1:3]) .~ Dirichlet([1.0, 2.0, 3.0])
        v = rowsums(P)
        y .~ Normal.(P[k, 1] .+ v[k], s)
    end), nt -> logpdf(Exponential(1), nt.s) +
        sum(logpdf(Dirichlet([1.0, 2.0, 3.0]), nt.P[j, :]) for j in 1:3))
    @test r.built.layout.total == 1 + 3 * 2
    @test all(j -> sum(r.nt.P[j, :]) ≈ 1 && all(>(0), r.nt.P[j, :]), 1:3)
    # Each row is the vector simplex transform of its coordinates.
    U = reshape(r.u[2:end], 3, 2)
    @test r.nt.P ≈ permutedims(reduce(hcat,
        [simplex_constrain(U[j, :]) for j in 1:3]))
    # Symmetric `Dirichlet(K, a)` over columns.
    r = _sl_check(:(begin
        s ~ Exponential(1)
        eachcol(P[1:4, levels(k)]) .~ Dirichlet(4, 2.0)
        v = colsums(P .* P)
        y .~ Normal.(v[k], s)
    end), nt -> logpdf(Exponential(1), nt.s) +
        sum(logpdf(Dirichlet(4, 2.0), nt.P[:, j]) for j in 1:3))
    @test all(j -> sum(r.nt.P[:, j]) ≈ 1, 1:3)
end

@testset "array slices: Dirichlet with parameter concentrations" begin
    _sl_check(:(begin
        s ~ Exponential(1)
        alpha[1:3] .~ Exponential.(1)
        eachrow(P[levels(k), 1:3]) .~ Dirichlet(alpha)
        y .~ Normal.(P[k, 2], s)
    end), nt -> logpdf(Exponential(1), nt.s) +
        sum(logpdf.(Exponential(1), nt.alpha)) +
        sum(logpdf(Dirichlet(nt.alpha), nt.P[j, :]) for j in 1:3))
    _sl_check(:(begin
        s ~ Exponential(1)
        A[levels(k), 1:3] .~ Exponential.(1)
        eachrow(P[levels(k), 1:3]) .~ Dirichlet.(eachrow(A))
        y .~ Normal.(P[k, 3], s)
    end), nt -> logpdf(Exponential(1), nt.s) +
        sum(logpdf.(Exponential(1), nt.A)) +
        sum(logpdf(Dirichlet(nt.A[j, :]), nt.P[j, :]) for j in 1:3))
end

@testset "array slices: per-slice literal concentrations" begin
    # Column g of the 2 × 3 literal is slice g's concentration.
    A = [1.0 2.0 1.0; 1.0 1.0 3.0]
    ref(nt) = logpdf(Exponential(1), nt.s) +
        sum(logpdf(Dirichlet(A[:, j]), nt.S[:, j]) for j in 1:3)
    # `D.([v1, v2, v3])` iterates its elements, as Julia broadcasting does,
    # and pairs slice g with v_g, exactly as `eachcol` of their matrix.
    for spelling in (:(Dirichlet.(eachcol([1.0 2.0 1.0; 1.0 1.0 3.0]))),
            :(Dirichlet.(eachcol([1 2 1; 1 1 3]))),
            :(Dirichlet.([[1.0, 1.0], [2.0, 1.0], [1.0, 3.0]])))
        r = _sl_check(:(begin
            s ~ Exponential(1)
            eachcol(S[1:2, levels(k)]) .~ $spelling
            y .~ Normal.(S[1, k], s)
        end), ref)
        @test r.built.layout.total == 1 + 3
    end
    _sl_check(:(begin
        s ~ Exponential(1)
        eachrow(P[levels(k), 1:2]) .~ Dirichlet.(eachrow([1.0 1.0; 2.0 1.0; 1.0 3.0]))
        y .~ Normal.(P[k, 1], s)
    end), nt -> logpdf(Exponential(1), nt.s) +
        sum(logpdf(Dirichlet(A[:, j]), nt.P[j, :]) for j in 1:3))
    # Per-slice means listed as vectors, with a shared factor.
    means = [[0.1, -0.2], [0.3, 0.0], [-0.1, 0.4]]
    _sl_check(:(begin
        s ~ Exponential(1)
        L ~ LKJCholesky(2, 2.0)
        eachrow(B[levels(k), 1:2]) .~ MvNormalCholesky.(
            [[0.1, -0.2], [0.3, 0.0], [-0.1, 0.4]], Ref(L))
        y .~ Normal.(B[k, 1], s)
    end), nt -> logpdf(Exponential(1), nt.s) +
        logpdf(LKJCholesky(2, 2.0), Cholesky(LowerTriangular(nt.L))) +
        sum(logpdf(MvNormal(means[j], nt.L * nt.L'), nt.B[j, :]) for j in 1:3))
end

@testset "array slices: ordered rows and columns" begin
    r = _sl_check(:(begin
        s ~ Exponential(1)
        eachrow(C[levels(k), 1:3]) .~ Ordered(Normal(0, 2), 3)
        y .~ Normal.(C[k, 1] .+ C[k, 3], s)
    end), nt -> logpdf(Exponential(1), nt.s) +
        sum(logpdf.(Normal(0, 2), nt.C)))
    @test all(j -> issorted(r.nt.C[j, :]), 1:3)
    U = reshape(r.u[2:end], 3, 3)
    @test r.nt.C ≈ permutedims(reduce(hcat,
        [ordered_constrain(U[j, :]) for j in 1:3]))
    # A parameter location, over columns; `Ordered.(…)` is `Ordered(…)`.
    r = _sl_check(:(begin
        s ~ Exponential(1)
        m ~ Normal(0, 1)
        eachcol(C[1:3, levels(k)]) .~ Ordered.(Normal(m, 2), 3)
        v = colsums(C)
        y .~ Normal.(v[k], s)
    end), nt -> logpdf(Exponential(1), nt.s) + logpdf(Normal(0, 1), nt.m) +
        sum(logpdf.(Normal(nt.m, 2), nt.C)))
    @test all(j -> issorted(r.nt.C[:, j]), 1:3)
end

@testset "array slices: multivariate normal slice densities" begin
    o = ReactiveKernelsPPL._SliceRows()
    B = [0.1 0.2; 0.3 -0.4]
    F = [1.0 0.0; 0.5 0.8]
    @test ReactiveKernelsPPL._mvnormal_cholesky_slices_logpdf(o, B, zeros(2),
        F) ≈ sum(logpdf(MvNormal(zeros(2), F * F'), B[j, :]) for j in 1:2)
    @test ReactiveKernelsPPL._mvnormal_slices_logpdf(o, B, zeros(2),
        F * F') ≈ sum(logpdf(MvNormal(zeros(2), F * F'), B[j, :]) for j in 1:2)
    # A non-symmetric or indefinite covariance is not a covariance.
    # refused: a covariance must be symmetric positive definite (distribution domain, P3)
    @test_throws ArgumentError ReactiveKernelsPPL._mvnormal_slices_logpdf(o,
        B, zeros(2), [1.0 0.5; 0.0 1.0])
    # refused: a covariance must be symmetric positive definite (distribution domain, P3)
    @test_throws ArgumentError ReactiveKernelsPPL._mvnormal_slices_logpdf(o,
        B, zeros(2), [1.0 2.0; 2.0 1.0])
end

# The call throws a `T` whose message names `needle` (so a refusal test
# cannot pass on an unrelated failure).
function _sl_refuses(T, f, needle)
    err = try
        f()
        nothing
    catch e
        e
    end
    # refused: each caller below identifies its invalid index, shape or declaration contract (P3/P6).
    @test err isa T
    @test err !== nothing && occursin(needle, sprint(showerror, err))
end

@testset "array slices: fail closed" begin
    lowerm(ast) = lower_rkppl(ast, (:y, :k); mod = SliceModels, conditioned = (:y, :k))
    bindm(ast) = bind_data(lowerm(ast), Dict{Symbol,ColumnData}(
        :y => _av_y(), :k => _SL_K))
    prog(decl) = :(begin
        s ~ Exponential(1)
        L ~ LKJCholesky(2, 2.0)
        sd[1:2] .~ Exponential.(1)
        F = sd .* L
        M[levels(k), 1:2] .~ Normal.(0, 1)
        $decl
        v = rowsums(B)
        y .~ Normal.(v[k], s)
    end)
    # The base program lowers and binds.
    @test bindm(prog(:(eachrow(B[levels(k), 1:2]) .~
        MvNormalCholesky.(eachrow(M), Ref(F))))) isa StructuralPlan
    # refused: in a broadcast a bare vector iterates its ELEMENTS (Julia
    # broadcasting, principle 3); shared arguments are `Ref(mu)`.
    _sl_refuses(SurfaceLoweringError, () -> lowerm(prog(:(eachrow(B[levels(k),
        1:2]) .~ MvNormalCholesky.(zeros(2), Ref(F))))),
        "every argument is shared (`Ref(x)`)")
    # refused: a literal vector of numbers iterates its NUMBERS in a
    # broadcast (principle 3); one vector per slice is a vector of vectors.
    _sl_refuses(SurfaceLoweringError, () -> lowerm(prog(:(eachrow(B[levels(k),
        1:2]) .~ MvNormalCholesky.([0.0, 0.0], Ref(F))))),
        "every argument is shared (`Ref(x)`)")
    # refused: `eachrow(M)` gives one value per slice, which only a
    # broadcast pairs with the slices (principle 3).
    _sl_refuses(SurfaceLoweringError, () -> lowerm(prog(:(eachrow(B[levels(k),
        1:2]) .~ MvNormalCholesky(eachrow(M), F)))),
        "pairs slices only in a broadcast")
    # refused: `Ref` marks a shared broadcast argument; an undotted call
    # has no broadcast to share across (principle 3).
    _sl_refuses(SurfaceLoweringError, () -> lowerm(prog(:(eachrow(B[levels(k),
        1:2]) .~ MvNormalCholesky(Ref(zeros(2)), F)))),
        "marks a shared argument")
    # refused: rows of a matrix are vectors, never Cholesky factors
    # (principle 3).
    _sl_refuses(SurfaceLoweringError, () -> lowerm(prog(:(eachrow(B[levels(k),
        1:2]) .~ MvNormalCholesky.(eachrow(M), eachrow(F))))),
        "is one matrix shared by every slice")
    # refused: a slice is a vector, which a univariate family cannot draw
    # (principle 3); elementwise priors are `B[a, b] .~ Normal.(…)`.
    _sl_refuses(SurfaceLoweringError, () -> lowerm(prog(:(eachrow(B[levels(k),
        1:2]) .~ Normal(0, 1)))),
        "each row is a vector, drawn by a multivariate distribution")
    # refused: a covariance is a matrix (principle 3).
    _sl_refuses(SurfaceLoweringError, () -> lowerm(prog(:(eachrow(B[levels(k),
        1:2]) .~ MvNormal(zeros(2), 1.0)))),
        "covariance is a matrix, got the scalar")
    # refused: a matrix has no single orientation for its draws —
    # Distributions treats columns as samples, `eachrow(B)` names rows; the
    # slices are stated (principle 2, no unstated conventions).
    _sl_refuses(SurfaceLoweringError, () -> lowerm(prog(:(B[levels(k), 1:2] ~
        MvNormalCholesky(zeros(2), F)))),
        "a two-axis array draws its slices")
    # refused: `.~` draws every element of a vector separately, so one
    # multivariate draw is `~` (principle 3).
    _sl_refuses(SurfaceLoweringError, () -> lowerm(:(begin
        s ~ Exponential(1)
        L ~ LKJCholesky(2, 2.0)
        b[1:2] .~ MvNormalCholesky(zeros(2), L)
        y .~ Normal.(b[k], s)
    end)), "is one draw of the multivariate `MvNormalCholesky`")
    # A concentration must be positive.
    _sl_refuses(ContractValidationError, () -> bindm(prog(:(eachrow(B[levels(k),
        1:2]) .~ Dirichlet([1.0, -1.0])))),
        "Dirichlet concentrations are positive")
    # `Ordered` declares the slice length.
    _sl_refuses(ContractValidationError, () -> bindm(prog(:(eachrow(B[levels(k),
        1:2]) .~ Ordered(Normal(0, 1), 3)))), "`Ordered` declares length 3")
    # Per-slice means of the wrong orientation (2 × 3 for 3 rows of 2).
    _sl_refuses(ContractValidationError, () -> bindm(prog(:(eachrow(B[levels(k),
        1:2]) .~ MvNormalCholesky.(eachcol(M), Ref(F))))),
        "has 3 slice(s) of length 2")
    # A shared mean that is not a K-vector.
    _sl_refuses(ContractValidationError, () -> bindm(prog(:(eachrow(B[levels(k),
        1:2]) .~ MvNormalCholesky([0.0, 0.0, 0.0], F)))),
        "has 3 slice(s) of length 2")
end
