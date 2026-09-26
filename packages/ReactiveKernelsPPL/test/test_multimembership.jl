# Multi-membership `mm(...)` + stratified `gr(g; by=b)` varying draws
# (SB `multi_membership_*` / `ranef_correlated_by` mirrors): surface
# admission, bind-time union/strata fitting + validation, layout, value
# parity vs hand oracles (default / normalized / raw weights, K=1
# vacuous routes), Enzyme-vs-findiff gradients, Reactant/XLA value+grad,
# and SB-parity probes (M1/M2/M3/S1/S2 — peer BRM data, appended when
# the BRM half delivers). (`_query` / `_check_gradient` /
# `_findiff_grad` / `_GEN_BACKEND` come from test_generator.jl,
# included first.)
using Distributions: Normal, logpdf
using LinearAlgebra: Diagonal, dot
using ReactiveKernels
using ReactiveKernelsPPL
using Reactant
using Test
import ReactiveKernelsPPL: lkj_chol_constrain, lkj_logconst

# Lower + bind + build + query a program; return
# `(bound, built, kern, layout)`.
function _mm_query(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    kern = Base.invokelatest(prepare_query, built, bound, :sampler)
    return bound, built, kern, built.layout
end

# Layout segment for one entry (coefficients match by predictor).
function _mm_seg(lay, nm::Symbol)
    e = only(e for e in lay.entries
        if e.name === nm || (e.kind === :coefficient && e.predictor === nm))
    return e.offset:(e.offset + e.size - 1)
end

# Shared mm gather oracle: SB `multi_membership_*` association
# transcribed literally (`r[i] += w*b`). `gs`/`ws` are the per-slot
# group-index / weight vectors, `b` the G×K draws, `Z` the n×K design.
function _mm_ref_r(gs::Vector, ws::Vector, b::Matrix, Z::Matrix)
    M = length(gs)
    n = size(Z, 1)
    r = zeros(n)
    for i in 1:n, m in 1:M
        r[i] += ws[m][i] * dot(Z[i, :], b[gs[m][i], :])
    end
    return r
end

# Shared stratified oracle: SB `ranef_correlated_by` association.
# `s_of_g` maps group levels to strata; `bs[k]` is the G×K frame.
function _mm_ref_strat_r(g::AbstractVector, s_of_g::AbstractVector,
        bs::Vector, Z::Matrix)
    n = size(Z, 1)
    return [dot(Z[i, :], bs[s_of_g[g[i]]][g[i], :]) for i in 1:n]
end

const _MM_G1 = [1, 2, 1, 3]
const _MM_G2 = [2, 3, 1, 2]
const _MM_W1 = [0.7, 0.2, 0.5, 0.9]
const _MM_W2 = [0.3, 0.8, 0.5, 0.1]
const _MM_X = [0.5, -1.0, 1.5, 0.0]
const _MM_Y = [1.0, 2.0, 1.5, 2.5]
const _MM_B = [1, 1, 1, 2]
_mm_cols() = Dict{Symbol,AbstractVector}(:y => copy(_MM_Y), :x => copy(_MM_X),
    :g1 => copy(_MM_G1), :g2 => copy(_MM_G2), :w1 => copy(_MM_W1),
    :w2 => copy(_MM_W2), :b => copy(_MM_B))

@testset "mm surface admission" begin
    @testset "default weights intercept" begin
        plan = lower_rkppl(quote
                a ~ Normal(0, 5)
                d ~ varying_draws(mm(g1, g2), [1])
                r ~ varying_slice(d, 1)
                mu = a .+ r
                y .~ Normal.(mu, 1.0)
            end, (:y, :g1, :g2))
        d = only(plan.varying_draws)
        @test d.kind === :intercept1
        @test isnan(d.lkj_eta)
        @test d.group === :mm__g1__g2
        @test d.suffix == "mm__g1__g2"
        @test d.mm.groups == [:g1, :g2]
        @test d.mm.weights === nothing
        @test d.mm.normalize === true
        @test d.strata === nothing
    end
    @testset "weighted normalized correlated" begin
        plan = lower_rkppl(quote
                a ~ Normal(0, 5)
                d ~ varying_draws(mm(g1, g2; weights = (w1, w2)), [1, x])
                r ~ varying_slice(d, 1:2)
                mu = a .+ r
                y .~ Normal.(mu, 1.0)
            end, (:y, :x, :g1, :g2, :w1, :w2))
        d = only(plan.varying_draws)
        @test d.kind === :correlated
        @test d.lkj_eta == 1.0
        @test d.group === :mm__g1__g2__w__w1__w2
        @test d.mm.weights == [:w1, :w2]
        @test d.mm.normalize === true
    end
    @testset "raw weights suffix" begin
        plan = lower_rkppl(quote
                a ~ Normal(0, 5)
                d ~ varying_draws(mm(g1, g2; weights = (w1, w2),
                    normalize = false), [1, x])
                r ~ varying_slice(d, 1:2)
                mu = a .+ r
                y .~ Normal.(mu, 1.0)
            end, (:y, :x, :g1, :g2, :w1, :w2))
        d = only(plan.varying_draws)
        @test d.group === :mm__g1__g2__w__w1__w2__raw
        @test d.mm.normalize === false
    end
    @testset "explicit eta 1.0 accepted" begin
        plan = lower_rkppl(quote
                a ~ Normal(0, 5)
                d ~ varying_draws(mm(g1, g2), [1, x]; eta = 1.0)
                r ~ varying_slice(d, 1:2)
                mu = a .+ r
                y .~ Normal.(mu, 1.0)
            end, (:y, :x, :g1, :g2))
        d = only(plan.varying_draws)
        @test (d.kind, d.lkj_eta) === (:correlated, 1.0)
    end
    @testset "single slope is correlated" begin
        plan = lower_rkppl(quote
                a ~ Normal(0, 5)
                d ~ varying_draws(mm(g1, g2), [x])
                r ~ varying_slice(d, 1)
                mu = a .+ r
                y .~ Normal.(mu, 1.0)
            end, (:y, :x, :g1, :g2))
        d = only(plan.varying_draws)
        @test (d.kind, d.lkj_eta) === (:correlated, 1.0)
    end
    @testset "three groups" begin
        plan = lower_rkppl(quote
                a ~ Normal(0, 5)
                d ~ varying_draws(mm(g1, g2, g3), [1])
                r ~ varying_slice(d, 1)
                mu = a .+ r
                y .~ Normal.(mu, 1.0)
            end, (:y, :g1, :g2, :g3))
        d = only(plan.varying_draws)
        @test d.mm.groups == [:g1, :g2, :g3]
        @test d.group === :mm__g1__g2__g3
    end
    @testset "fused effect spelling" begin
        plan = lower_rkppl(quote
                a ~ Normal(0, 5)
                r ~ varying_effect(mm(g1, g2), [1])
                mu = a .+ r
                y .~ Normal.(mu, 1.0)
            end, (:y, :g1, :g2))
        d = only(plan.varying_draws)
        @test d.kind === :intercept1
        @test d.mm.groups == [:g1, :g2]
    end
    @testset "error spellings" begin
        # One group.
        @test_throws SurfaceLoweringError lower_rkppl(quote
                a ~ Normal(0, 5)
                d ~ varying_draws(mm(g1), [1])
                r ~ varying_slice(d, 1)
                mu = a .+ r
                y .~ Normal.(mu, 1.0)
            end, (:y, :g1))
        # Weights as a vector, not a tuple.
        @test_throws SurfaceLoweringError lower_rkppl(quote
                a ~ Normal(0, 5)
                d ~ varying_draws(mm(g1, g2; weights = [w1, w2]), [1])
                r ~ varying_slice(d, 1)
                mu = a .+ r
                y .~ Normal.(mu, 1.0)
            end, (:y, :g1, :g2, :w1, :w2))
        # Weight arity mismatch.
        @test_throws SurfaceLoweringError lower_rkppl(quote
                a ~ Normal(0, 5)
                d ~ varying_draws(mm(g1, g2; weights = (w1,)), [1])
                r ~ varying_slice(d, 1)
                mu = a .+ r
                y .~ Normal.(mu, 1.0)
            end, (:y, :g1, :g2, :w1))
        # Non-data group.
        @test_throws SurfaceLoweringError lower_rkppl(quote
                a ~ Normal(0, 5)
                d ~ varying_draws(mm(g1, ghost), [1])
                r ~ varying_slice(d, 1)
                mu = a .+ r
                y .~ Normal.(mu, 1.0)
            end, (:y, :g1))
        # Non-Bool normalize.
        @test_throws SurfaceLoweringError lower_rkppl(quote
                a ~ Normal(0, 5)
                d ~ varying_draws(mm(g1, g2; normalize = 1), [1])
                r ~ varying_slice(d, 1)
                mu = a .+ r
                y .~ Normal.(mu, 1.0)
            end, (:y, :g1, :g2))
        # Unknown keyword.
        @test_throws SurfaceLoweringError lower_rkppl(quote
                a ~ Normal(0, 5)
                d ~ varying_draws(mm(g1, g2; id = 1), [1])
                r ~ varying_slice(d, 1)
                mu = a .+ r
                y .~ Normal.(mu, 1.0)
            end, (:y, :g1, :g2))
        # Missing semicolon.
        @test_throws SurfaceLoweringError lower_rkppl(quote
                a ~ Normal(0, 5)
                d ~ varying_draws(mm(g1, g2, normalize = false), [1])
                r ~ varying_slice(d, 1)
                mu = a .+ r
                y .~ Normal.(mu, 1.0)
            end, (:y, :g1, :g2))
        # eta != 1.0 on correlated.
        @test_throws SurfaceLoweringError lower_rkppl(quote
                a ~ Normal(0, 5)
                d ~ varying_draws(mm(g1, g2), [1, x]; eta = 2.0)
                r ~ varying_slice(d, 1:2)
                mu = a .+ r
                y .~ Normal.(mu, 1.0)
            end, (:y, :x, :g1, :g2))
        # eta on intercept-only.
        @test_throws SurfaceLoweringError lower_rkppl(quote
                a ~ Normal(0, 5)
                d ~ varying_draws(mm(g1, g2), [1]; eta = 1.0)
                r ~ varying_slice(d, 1)
                mu = a .+ r
                y .~ Normal.(mu, 1.0)
            end, (:y, :g1, :g2))
    end
end

@testset "stratified surface admission" begin
    @testset "K=2 correlated" begin
        plan = lower_rkppl(quote
                a ~ Normal(0, 5)
                d ~ varying_draws(gr(g; by = b), [1, x])
                r ~ varying_slice(d, 1:2)
                mu = a .+ r
                y .~ Normal.(mu, 1.0)
            end, (:y, :x, :g, :b))
        d = only(plan.varying_draws)
        @test d.kind === :correlated
        @test d.lkj_eta == 1.0
        @test d.group === :g
        @test d.strata.by === :b
        @test d.strata.levels === nothing
        @test d.mm === nothing
    end
    @testset "K=1 is correlated (no special-case)" begin
        plan = lower_rkppl(quote
                a ~ Normal(0, 5)
                d ~ varying_draws(gr(g; by = b), [1])
                r ~ varying_slice(d, 1)
                mu = a .+ r
                y .~ Normal.(mu, 1.0)
            end, (:y, :g, :b))
        d = only(plan.varying_draws)
        @test (d.kind, d.lkj_eta) === (:correlated, 1.0)
    end
    @testset "explicit eta 1.0 accepted" begin
        plan = lower_rkppl(quote
                a ~ Normal(0, 5)
                d ~ varying_draws(gr(g; by = b), [1, x]; eta = 1.0)
                r ~ varying_slice(d, 1:2)
                mu = a .+ r
                y .~ Normal.(mu, 1.0)
            end, (:y, :x, :g, :b))
        @test only(plan.varying_draws).lkj_eta == 1.0
    end
    @testset "fused effect spelling" begin
        plan = lower_rkppl(quote
                a ~ Normal(0, 5)
                r ~ varying_effect(gr(g; by = b), [1, x])
                mu = a .+ r
                y .~ Normal.(mu, 1.0)
            end, (:y, :x, :g, :b))
        d = only(plan.varying_draws)
        @test d.kind === :correlated
        @test d.strata.by === :b
    end
    @testset "error spellings" begin
        # Bare gr(g).
        @test_throws SurfaceLoweringError lower_rkppl(quote
                a ~ Normal(0, 5)
                d ~ varying_draws(gr(g), [1])
                r ~ varying_slice(d, 1)
                mu = a .+ r
                y .~ Normal.(mu, 1.0)
            end, (:y, :g))
        # Two groups.
        @test_throws SurfaceLoweringError lower_rkppl(quote
                a ~ Normal(0, 5)
                d ~ varying_draws(gr(g, h; by = b), [1])
                r ~ varying_slice(d, 1)
                mu = a .+ r
                y .~ Normal.(mu, 1.0)
            end, (:y, :g, :h, :b))
        # Non-data by.
        @test_throws SurfaceLoweringError lower_rkppl(quote
                a ~ Normal(0, 5)
                d ~ varying_draws(gr(g; by = ghost), [1])
                r ~ varying_slice(d, 1)
                mu = a .+ r
                y .~ Normal.(mu, 1.0)
            end, (:y, :g))
        # by === group.
        @test_throws SurfaceLoweringError lower_rkppl(quote
                a ~ Normal(0, 5)
                d ~ varying_draws(gr(g; by = g), [1])
                r ~ varying_slice(d, 1)
                mu = a .+ r
                y .~ Normal.(mu, 1.0)
            end, (:y, :g))
        # id keyword (BRM-side spelling).
        @test_throws SurfaceLoweringError lower_rkppl(quote
                a ~ Normal(0, 5)
                d ~ varying_draws(gr(g; by = b, id = 1), [1])
                r ~ varying_slice(d, 1)
                mu = a .+ r
                y .~ Normal.(mu, 1.0)
            end, (:y, :g, :b))
        # eta != 1.0.
        @test_throws SurfaceLoweringError lower_rkppl(quote
                a ~ Normal(0, 5)
                d ~ varying_draws(gr(g; by = b), [1, x]; eta = 2.0)
                r ~ varying_slice(d, 1:2)
                mu = a .+ r
                y .~ Normal.(mu, 1.0)
            end, (:y, :x, :g, :b))
        # Missing semicolon.
        @test_throws SurfaceLoweringError lower_rkppl(quote
                a ~ Normal(0, 5)
                d ~ varying_draws(gr(g, by = b), [1])
                r ~ varying_slice(d, 1)
                mu = a .+ r
                y .~ Normal.(mu, 1.0)
            end, (:y, :g, :b))
    end
    @testset "reserved grouping names" begin
        @test_throws SurfaceLoweringError lower_rkppl(quote
                mm = 1.0
                a ~ Normal(0, 5)
                mu = a
                y .~ Normal.(mu, 1.0)
            end, (:y,))
        @test_throws SurfaceLoweringError lower_rkppl(quote
                gr = 1.0
                a ~ Normal(0, 5)
                mu = a
                y .~ Normal.(mu, 1.0)
            end, (:y,))
    end
end

# Hand-built draws for contract-validation tests (surface-independent).
# Label stays `:draws_g` so the base plan's slice/term linkage holds
# and the grouping check under test is what fires.
function _mm_draws(; group = :mm__g1__g2, kind = :intercept1,
        margins = VaryingMargin[VaryingMargin(:Intercept,
            VaryingZRecipe(:ones, :none, nothing))],
        lkj_eta = NaN, label = :draws_g, suffix = "mm__g1__g2",
        levels = nothing, sd_priors = VaryingSdPrior[],
        mm = VaryingMultiMembership([:g1, :g2], nothing, true),
        strata = nothing)
    return VaryingDraws(group, kind, margins, lkj_eta, label, suffix,
        levels, sd_priors, mm, strata)
end

function _mm_validate(d::VaryingDraws)
    plan = lower_rkppl(quote
            a ~ Normal(0, 5)
            r ~ varying_effect(g, [1])
            mu = a .+ r
            y .~ Normal.(mu, 1.0)
        end, (:y, :g))
    plan.varying_draws[1] = d
    K = length(d.margins)
    s = plan.varying_slices[1]
    plan.varying_slices[1] = VaryingSlice(d.label, 1:K, s.target)
    return validate_structure(plan)
end

@testset "mm/strata contract validation" begin
    ones1() = VaryingMargin[VaryingMargin(:Intercept,
        VaryingZRecipe(:ones, :none, nothing))]
    slope1() = VaryingMargin[VaryingMargin(:x,
        VaryingZRecipe(:column, :x, nothing))]
    @testset "mm one group rejected" begin
        d = _mm_draws(mm = VaryingMultiMembership([:g1], nothing, true))
        @test_throws ContractValidationError _mm_validate(d)
    end
    @testset "mm weight arity rejected" begin
        d = _mm_draws(
            mm = VaryingMultiMembership([:g1, :g2], [:w1], true))
        @test_throws ContractValidationError _mm_validate(d)
    end
    @testset "mm slope1 kind rejected" begin
        d = _mm_draws(kind = :slope1, margins = slope1())
        @test_throws ContractValidationError _mm_validate(d)
    end
    @testset "mm intercept with eta rejected" begin
        d = _mm_draws(kind = :correlated, lkj_eta = 1.0)
        @test_throws ContractValidationError _mm_validate(d)
    end
    @testset "mm eta != 1.0 rejected" begin
        d = _mm_draws(kind = :correlated, lkj_eta = 2.0,
            margins = vcat(ones1(), slope1()))
        @test_throws ContractValidationError _mm_validate(d)
    end
    @testset "mm sd priors rejected" begin
        d = _mm_draws(kind = :correlated, lkj_eta = 1.0,
            margins = vcat(ones1(), slope1()),
            sd_priors = [VaryingSdPrior(:std_normal, 1.0),
                VaryingSdPrior(:std_normal, 1.0)])
        @test_throws ContractValidationError _mm_validate(d)
    end
    @testset "mm x strata rejected" begin
        d = _mm_draws(kind = :correlated, lkj_eta = 1.0,
            margins = vcat(ones1(), slope1()),
            strata = VaryingStrata(:b, nothing))
        @test_throws ContractValidationError _mm_validate(d)
    end
    @testset "stratified non-correlated rejected" begin
        d = _mm_draws(group = :g, kind = :intercept1, mm = nothing,
            label = :draws_g, suffix = "g",
            strata = VaryingStrata(:b, nothing))
        @test_throws ContractValidationError _mm_validate(d)
    end
    @testset "stratified eta != 1.0 rejected" begin
        d = _mm_draws(group = :g, kind = :correlated, lkj_eta = 2.0,
            margins = vcat(ones1(), slope1()), mm = nothing,
            label = :draws_g, suffix = "g",
            strata = VaryingStrata(:b, nothing))
        @test_throws ContractValidationError _mm_validate(d)
    end
    @testset "stratified sd priors rejected" begin
        d = _mm_draws(group = :g, kind = :correlated, lkj_eta = 1.0,
            margins = vcat(ones1(), slope1()), mm = nothing,
            label = :draws_g, suffix = "g",
            sd_priors = [VaryingSdPrior(:std_normal, 1.0),
                VaryingSdPrior(:std_normal, 1.0)],
            strata = VaryingStrata(:b, nothing))
        @test_throws ContractValidationError _mm_validate(d)
    end
    @testset "stratified by === group rejected" begin
        d = _mm_draws(group = :g, kind = :correlated, lkj_eta = 1.0,
            margins = vcat(ones1(), slope1()), mm = nothing,
            label = :draws_g, suffix = "g",
            strata = VaryingStrata(:g, nothing))
        @test_throws ContractValidationError _mm_validate(d)
    end
    @testset "mm multi-slice rejected (ID defense)" begin
        # Lowering validates structure, so the rejection fires here.
        @test_throws ContractValidationError lower_rkppl(quote
                a ~ Normal(0, 5)
                c ~ Normal(0, 5)
                d ~ varying_draws(mm(g1, g2), [1, x])
                r1 ~ varying_slice(d, 1:1)
                r2 ~ varying_slice(d, 2:2)
                mu = a .+ r1
                nu = c .+ r2
                y .~ Normal.(mu, 1.0)
                z .~ Normal.(nu, 1.0)
            end, (:y, :z, :x, :g1, :g2))
    end
    @testset "stratified multi-slice allowed (ID+gr path)" begin
        plan = lower_rkppl(quote
                a ~ Normal(0, 5)
                c ~ Normal(0, 5)
                d ~ varying_draws(gr(g; by = b), [1, x])
                r1 ~ varying_slice(d, 1:1)
                r2 ~ varying_slice(d, 2:2)
                mu = a .+ r1
                nu = c .+ r2
                y .~ Normal.(mu, 1.0)
                z .~ Normal.(nu, 1.0)
            end, (:y, :z, :x, :g, :b))
        @test validate_structure(plan) === nothing
    end
end

@testset "mm bind" begin
    prog = quote
        a ~ Normal(0, 5)
        d ~ varying_draws(mm(g1, g2; weights = (w1, w2)), [1])
        r ~ varying_slice(d, 1)
        mu = a .+ r
        y .~ Normal.(mu, 1.0)
    end
    @testset "union fit" begin
        plan = lower_rkppl(prog, (:y, :g1, :g2, :w1, :w2))
        bound = bind_data(plan, _mm_cols())
        @test only(bound.varying_draws).levels == [1, 2, 3]
    end
    @testset "declared union passthrough" begin
        plan = lower_rkppl(quote
                a ~ Normal(0, 5)
                d ~ varying_draws(mm(g1, g2), [1]; levels = [1, 2, 3, 4])
                r ~ varying_slice(d, 1)
                mu = a .+ r
                y .~ Normal.(mu, 1.0)
            end, (:y, :g1, :g2))
        bound = bind_data(plan, _mm_cols())
        @test only(bound.varying_draws).levels == [1, 2, 3, 4]
    end
    @testset "union coverage failure" begin
        plan = lower_rkppl(quote
                a ~ Normal(0, 5)
                d ~ varying_draws(mm(g1, g2), [1]; levels = [1, 2])
                r ~ varying_slice(d, 1)
                mu = a .+ r
                y .~ Normal.(mu, 1.0)
            end, (:y, :g1, :g2))
        @test_throws ContractValidationError bind_data(plan, _mm_cols())
    end
    @testset "unbound membership" begin
        plan = lower_rkppl(prog, (:y, :g1, :g2, :w1, :w2))
        cols = _mm_cols()
        delete!(cols, :g2)
        @test_throws ContractValidationError bind_data(plan, cols)
    end
    @testset "weight failures" begin
        plan = lower_rkppl(prog, (:y, :g1, :g2, :w1, :w2))
        # Non-finite.
        cols = _mm_cols()
        cols[:w1] = [0.7, Inf, 0.5, 0.9]
        @test_throws ContractValidationError bind_data(plan, cols)
        # Negative.
        cols = _mm_cols()
        cols[:w2] = [0.3, -0.8, 0.5, 0.1]
        @test_throws ContractValidationError bind_data(plan, cols)
        # Zero row total.
        cols = _mm_cols()
        cols[:w1] = [0.0, 0.2, 0.5, 0.9]
        cols[:w2] = [0.0, 0.8, 0.5, 0.1]
        @test_throws ContractValidationError bind_data(plan, cols)
        # Wrong length.
        cols = _mm_cols()
        cols[:w1] = [0.7, 0.2, 0.5]
        @test_throws ContractValidationError bind_data(plan, cols)
        # Non-real eltype.
        cols = _mm_cols()
        cols[:w1] = ["a", "b", "c", "d"]
        @test_throws ContractValidationError bind_data(plan, cols)
    end
    @testset "unorderable union" begin
        plan = lower_rkppl(quote
                a ~ Normal(0, 5)
                d ~ varying_draws(mm(g1, g2), [1])
                r ~ varying_slice(d, 1)
                mu = a .+ r
                y .~ Normal.(mu, 1.0)
            end, (:y, :g1, :g2))
        cols = _mm_cols()
        cols[:g2] = ["a", "b", "c", "d"]
        @test_throws ContractValidationError bind_data(plan, cols)
    end
end

@testset "stratified bind" begin
    prog = quote
        a ~ Normal(0, 5)
        d ~ varying_draws(gr(g; by = b), [1, x])
        r ~ varying_slice(d, 1:2)
        mu = a .+ r
        y .~ Normal.(mu, 1.0)
    end
    @testset "strata fit" begin
        plan = lower_rkppl(prog, (:y, :x, :g, :b))
        cols = Dict{Symbol,AbstractVector}(:y => copy(_MM_Y),
            :x => copy(_MM_X), :g => [1, 2, 1, 3], :b => copy(_MM_B))
        bound = bind_data(plan, cols)
        d = only(bound.varying_draws)
        @test d.levels == [1, 2, 3]
        @test d.strata.levels == [1, 2]
    end
    @testset "straddling group rejected" begin
        plan = lower_rkppl(prog, (:y, :x, :g, :b))
        cols = Dict{Symbol,AbstractVector}(:y => copy(_MM_Y),
            :x => copy(_MM_X), :g => [1, 2, 1, 3], :b => [1, 1, 2, 2])
        @test_throws ContractValidationError bind_data(plan, cols)
    end
    @testset "unbound by" begin
        plan = lower_rkppl(prog, (:y, :x, :g, :b))
        cols = Dict{Symbol,AbstractVector}(:y => copy(_MM_Y),
            :x => copy(_MM_X), :g => [1, 2, 1, 3])
        @test_throws ContractValidationError bind_data(plan, cols)
    end
    @testset "by wrong length" begin
        plan = lower_rkppl(prog, (:y, :x, :g, :b))
        cols = Dict{Symbol,AbstractVector}(:y => copy(_MM_Y),
            :x => copy(_MM_X), :g => [1, 2, 1, 3], :b => [1, 1, 1])
        @test_throws ContractValidationError bind_data(plan, cols)
    end
end

@testset "mm/strata layout" begin
    @testset "mm totals mirror plain geometries" begin
        bound, built, _, lay = _mm_query(quote
                a ~ Normal(0, 5)
                d ~ varying_draws(mm(g1, g2), [1])
                r ~ varying_slice(d, 1)
                mu = a .+ r
                y .~ Normal.(mu, 1.0)
            end, _mm_cols())
        # a + log_scale + xi(3).
        @test lay.total == 5
        _, _, _, lay2 = _mm_query(quote
                a ~ Normal(0, 5)
                d ~ varying_draws(mm(g1, g2), [x])
                r ~ varying_slice(d, 1)
                mu = a .+ r
                y .~ Normal.(mu, 1.0)
            end, _mm_cols())
        # a + 0 thetas + tau(1) + z(3): the vacuous route packs no L.
        @test lay2.total == 5
        @test count(e -> e.kind === :varying_corr, lay2.entries) == 1
        @test only(e for e in lay2.entries if e.kind === :varying_corr).size == 0
    end
    @testset "stratified entries and order" begin
        _, _, _, lay = _mm_query(quote
                a ~ Normal(0, 5)
                d ~ varying_draws(gr(g; by = b), [1, x])
                r ~ varying_slice(d, 1:2)
                mu = a .+ r
                y .~ Normal.(mu, 1.0)
            end, Dict{Symbol,AbstractVector}(:y => copy(_MM_Y),
            :x => copy(_MM_X), :g => [1, 2, 1, 3], :b => copy(_MM_B)))
        # SB declaration order: all L, all tau, then z.
        names = [e.name for e in lay.entries]
        @test names ==
            [:mu_coef, :L_g_s1, :L_g_s2, :tau_g_s1, :tau_g_s2, :z_flat_g]
        kinds = [e.kind for e in lay.entries]
        @test kinds ==
            [:coefficient, :varying_corr, :varying_corr, :varying, :varying, :varying]
        @test [e.size for e in lay.entries] == [1, 1, 1, 2, 2, 6]
        @test lay.total == 13
        @test length(coordinate_names(lay)) == 13
    end
    @testset "constrain fail-closed on stratified" begin
        _, _, _, lay = _mm_query(quote
                a ~ Normal(0, 5)
                d ~ varying_draws(gr(g; by = b), [1, x])
                r ~ varying_slice(d, 1:2)
                mu = a .+ r
                y .~ Normal.(mu, 1.0)
            end, Dict{Symbol,AbstractVector}(:y => copy(_MM_Y),
            :x => copy(_MM_X), :g => [1, 2, 1, 3], :b => copy(_MM_B)))
        u = collect(range(-0.4, 0.4; length = lay.total))
        err = try
            constrain(lay, u)
            nothing
        catch e
            e
        end
        @test err isa ContractValidationError
        @test occursin("gr(g, by=b)", sprint(showerror, err))
    end
end

@testset "stratified name tables" begin
    @testset "unbound per-stratum names don't shadow user locals" begin
        plan = lower_rkppl(quote
                L_g = 1.0
                a ~ Normal(0, 5)
                d ~ varying_draws(gr(g; by = b), [1, x])
                r ~ varying_slice(d, 1:2)
                mu = a .+ r
                y .~ Normal.(mu, 1.0)
            end, (:y, :x, :g, :b))
        @test validate_structure(plan) === nothing
    end
    @testset "bound per-stratum names proven unique" begin
        plan = lower_rkppl(quote
                a ~ Normal(0, 5)
                c ~ Normal(0, 5)
                d ~ varying_draws(gr(g; by = b), [1, x])
                r ~ varying_slice(d, 1:2)
                d2 ~ varying_draws(g_s1, [1, x])
                r2 ~ varying_slice(d2, 1:2)
                mu = a .+ r
                nu = c .+ r2
                y .~ Normal.(mu, 1.0)
                z .~ Normal.(nu, 1.0)
            end, (:y, :z, :x, :g, :b, :g_s1))
        # Unbound: only the shared z is tabled — no false positive.
        @test validate_structure(plan) === nothing
        cols = Dict{Symbol,AbstractVector}(:y => copy(_MM_Y),
            :z => copy(_MM_Y), :x => copy(_MM_X), :g => [1, 2, 1, 3],
            :b => copy(_MM_B), :g_s1 => [1, 1, 2, 2])
        bound = bind_data(plan, cols)
        # Bound: per-stratum `L_g_s1` vs plain `L_g_s1` collide loudly.
        @test_throws ContractValidationError build_kernel(bound)
    end
end

@testset "mm intercept e2e values and gradient" begin
    @testset "default weights" begin
        bound, built, _, lay = _mm_query(quote
                a ~ Normal(0, 5)
                d ~ varying_draws(mm(g1, g2), [1])
                r ~ varying_slice(d, 1)
                mu = a .+ r
                y .~ Normal.(mu, 1.0)
            end, _mm_cols())
        u = [0.2, 0.1, -0.3, 0.4, 0.0]
        nt = constrain(lay, u)
        b = reshape(exp(nt.log_scale_mm__g1__g2) .* nt.xi_mm__g1__g2, 3, 1)
        r = _mm_ref_r([_MM_G1, _MM_G2],
            [fill(0.5, 4), fill(0.5, 4)], b, ones(4, 1))
        ll = sum(logpdf.(Normal.(nt.mu[1] .+ r, 1.0), _MM_Y))
        pr = logpdf(Normal(0, 5), nt.mu[1]) +
            logpdf(Normal(0, 1), nt.log_scale_mm__g1__g2) +
            sum(logpdf.(Normal(0, 1), nt.xi_mm__g1__g2))
        @test _query(built.spec, bound, :likelihood, u) ≈ ll
        @test _query(built.spec, bound, :prior, u) ≈ pr
        @test _query(built.spec, bound, :posterior, u) ≈ ll + pr
        _check_gradient(built.spec, bound, u)
    end
    @testset "normalized weights" begin
        bound, built, _, lay = _mm_query(quote
                a ~ Normal(0, 5)
                d ~ varying_draws(mm(g1, g2; weights = (w1, w2)), [1])
                r ~ varying_slice(d, 1)
                mu = a .+ r
                y .~ Normal.(mu, 1.0)
            end, _mm_cols())
        u = [0.2, 0.1, -0.3, 0.4, 0.0]
        nt = constrain(lay, u)
        sfx = "mm__g1__g2__w__w1__w2"
        b = reshape(exp(getfield(nt, Symbol("log_scale_", sfx))) .*
            getfield(nt, Symbol("xi_", sfx)), 3, 1)
        tot = _MM_W1 .+ _MM_W2
        r = _mm_ref_r([_MM_G1, _MM_G2], [_MM_W1 ./ tot, _MM_W2 ./ tot],
            b, ones(4, 1))
        ll = sum(logpdf.(Normal.(nt.mu[1] .+ r, 1.0), _MM_Y))
        pr = logpdf(Normal(0, 5), nt.mu[1]) +
            logpdf(Normal(0, 1), getfield(nt, Symbol("log_scale_", sfx))) +
            sum(logpdf.(Normal(0, 1), getfield(nt, Symbol("xi_", sfx))))
        @test _query(built.spec, bound, :likelihood, u) ≈ ll
        @test _query(built.spec, bound, :prior, u) ≈ pr
        @test _query(built.spec, bound, :posterior, u) ≈ ll + pr
        _check_gradient(built.spec, bound, u)
    end
    @testset "raw weights" begin
        bound, built, _, lay = _mm_query(quote
                a ~ Normal(0, 5)
                d ~ varying_draws(mm(g1, g2; weights = (w1, w2),
                    normalize = false), [1])
                r ~ varying_slice(d, 1)
                mu = a .+ r
                y .~ Normal.(mu, 1.0)
            end, _mm_cols())
        u = [0.2, 0.1, -0.3, 0.4, 0.0]
        nt = constrain(lay, u)
        sfx = "mm__g1__g2__w__w1__w2__raw"
        b = reshape(exp(getfield(nt, Symbol("log_scale_", sfx))) .*
            getfield(nt, Symbol("xi_", sfx)), 3, 1)
        r = _mm_ref_r([_MM_G1, _MM_G2], [_MM_W1, _MM_W2], b, ones(4, 1))
        ll = sum(logpdf.(Normal.(nt.mu[1] .+ r, 1.0), _MM_Y))
        pr = logpdf(Normal(0, 5), nt.mu[1]) +
            logpdf(Normal(0, 1), getfield(nt, Symbol("log_scale_", sfx))) +
            sum(logpdf.(Normal(0, 1), getfield(nt, Symbol("xi_", sfx))))
        @test _query(built.spec, bound, :likelihood, u) ≈ ll
        @test _query(built.spec, bound, :prior, u) ≈ pr
        @test _query(built.spec, bound, :posterior, u) ≈ ll + pr
        _check_gradient(built.spec, bound, u)
    end
end

@testset "mm correlated e2e values and gradient" begin
    @testset "normalized weights" begin
        bound, built, _, lay = _mm_query(quote
                a ~ Normal(0, 5)
                d ~ varying_draws(mm(g1, g2; weights = (w1, w2)), [1, x])
                r ~ varying_slice(d, 1:2)
                mu = a .+ r
                y .~ Normal.(mu, 1.0)
            end, _mm_cols())
        # a + theta + 2 tau + 6 z.
        @test lay.total == 10
        u = collect(range(-0.4, 0.4; length = lay.total))
        nt = constrain(lay, u)
        sfx = "mm__g1__g2__w__w1__w2"
        L = getfield(nt, Symbol("L_", sfx))
        tau = getfield(nt, Symbol("tau_", sfx))
        zf = getfield(nt, Symbol("z_flat_", sfx))
        b = Matrix((Diagonal(tau) * L * reshape(zf, 2, 3))')
        Z = hcat(ones(4), _MM_X)
        tot = _MM_W1 .+ _MM_W2
        r = _mm_ref_r([_MM_G1, _MM_G2], [_MM_W1 ./ tot, _MM_W2 ./ tot],
            b, Z)
        ll = sum(logpdf.(Normal.(nt.mu[1] .+ r, 1.0), _MM_Y))
        pr = logpdf(Normal(0, 5), nt.mu[1]) + lkj_logconst(2, 1.0) +
            sum(logpdf.(Normal(0, 1), tau)) +
            sum(logpdf.(Normal(0, 1), zf))
        @test _query(built.spec, bound, :likelihood, u) ≈ ll
        @test _query(built.spec, bound, :prior, u) ≈ pr
        # Jacobian: tau exps + the one K=2 vine term.
        th = only(u[_mm_seg(lay, Symbol("L_", sfx))])
        jac = sum(u[_mm_seg(lay, Symbol("tau_", sfx))]) +
            log(1 - tanh(th)^2)
        @test _query(built.spec, bound, :posterior, u) ≈ ll + pr + jac
        _check_gradient(built.spec, bound, u)
    end
    @testset "raw weights" begin
        bound, built, _, lay = _mm_query(quote
                a ~ Normal(0, 5)
                d ~ varying_draws(mm(g1, g2; weights = (w1, w2),
                    normalize = false), [1, x])
                r ~ varying_slice(d, 1:2)
                mu = a .+ r
                y .~ Normal.(mu, 1.0)
            end, _mm_cols())
        u = collect(range(-0.4, 0.4; length = lay.total))
        nt = constrain(lay, u)
        sfx = "mm__g1__g2__w__w1__w2__raw"
        L = getfield(nt, Symbol("L_", sfx))
        tau = getfield(nt, Symbol("tau_", sfx))
        zf = getfield(nt, Symbol("z_flat_", sfx))
        b = Matrix((Diagonal(tau) * L * reshape(zf, 2, 3))')
        Z = hcat(ones(4), _MM_X)
        r = _mm_ref_r([_MM_G1, _MM_G2], [_MM_W1, _MM_W2], b, Z)
        ll = sum(logpdf.(Normal.(nt.mu[1] .+ r, 1.0), _MM_Y))
        pr = logpdf(Normal(0, 5), nt.mu[1]) + lkj_logconst(2, 1.0) +
            sum(logpdf.(Normal(0, 1), tau)) +
            sum(logpdf.(Normal(0, 1), zf))
        @test _query(built.spec, bound, :likelihood, u) ≈ ll
        @test _query(built.spec, bound, :prior, u) ≈ pr
        th = only(u[_mm_seg(lay, Symbol("L_", sfx))])
        jac = sum(u[_mm_seg(lay, Symbol("tau_", sfx))]) +
            log(1 - tanh(th)^2)
        @test _query(built.spec, bound, :posterior, u) ≈ ll + pr + jac
        _check_gradient(built.spec, bound, u)
    end
end

@testset "mm slope vacuous e2e values and gradient" begin
    bound, built, _, lay = _mm_query(quote
            a ~ Normal(0, 5)
            d ~ varying_draws(mm(g1, g2), [x])
            r ~ varying_slice(d, 1)
            mu = a .+ r
            y .~ Normal.(mu, 1.0)
        end, _mm_cols())
    # a + 0 thetas + tau(1) + z(3).
    @test lay.total == 5
    u = collect(range(-0.4, 0.4; length = lay.total))
    nt = constrain(lay, u)
    sfx = "mm__g1__g2"
    tau = getfield(nt, Symbol("tau_", sfx))
    zf = getfield(nt, Symbol("z_flat_", sfx))
    # Vacuous 1x1 LKJ: L = [1.0], b[g] = tau[1]*z[g].
    b = reshape(tau[1] .* zf, 3, 1)
    Z = reshape(_MM_X, 4, 1)
    r = _mm_ref_r([_MM_G1, _MM_G2], [fill(0.5, 4), fill(0.5, 4)], b, Z)
    ll = sum(logpdf.(Normal.(nt.mu[1] .+ r, 1.0), _MM_Y))
    pr = logpdf(Normal(0, 5), nt.mu[1]) + 0.0 +
        sum(logpdf.(Normal(0, 1), tau)) + sum(logpdf.(Normal(0, 1), zf))
    @test _query(built.spec, bound, :likelihood, u) ≈ ll
    @test _query(built.spec, bound, :prior, u) ≈ pr
    jac = only(u[_mm_seg(lay, Symbol("tau_", sfx))])
    @test _query(built.spec, bound, :posterior, u) ≈ ll + pr + jac
    _check_gradient(built.spec, bound, u)
end

@testset "stratified e2e values and gradient" begin
    @testset "K=2" begin
        cols = Dict{Symbol,AbstractVector}(:y => copy(_MM_Y),
            :x => copy(_MM_X), :g => [1, 2, 1, 3], :b => copy(_MM_B))
        bound, built, _, lay = _mm_query(quote
                a ~ Normal(0, 5)
                d ~ varying_draws(gr(g; by = b), [1, x])
                r ~ varying_slice(d, 1:2)
                mu = a .+ r
                y .~ Normal.(mu, 1.0)
            end, cols)
        u = collect(range(-0.4, 0.4; length = lay.total))
        # constrain is fail-closed on stratified draws — slice u manually.
        a = only(u[_mm_seg(lay, :mu)])
        L1 = lkj_chol_constrain(Vector{Float64}(u[_mm_seg(lay, :L_g_s1)]), 2)
        L2 = lkj_chol_constrain(Vector{Float64}(u[_mm_seg(lay, :L_g_s2)]), 2)
        t1 = exp.(u[_mm_seg(lay, :tau_g_s1)])
        t2 = exp.(u[_mm_seg(lay, :tau_g_s2)])
        zmat = reshape(u[_mm_seg(lay, :z_flat_g)], 2, 3)
        b1 = Matrix((Diagonal(t1) * L1 * zmat)')
        b2 = Matrix((Diagonal(t2) * L2 * zmat)')
        Z = hcat(ones(4), _MM_X)
        r = _mm_ref_strat_r([1, 2, 1, 3], [1, 1, 2], [b1, b2], Z)
        ll = sum(logpdf.(Normal.(a .+ r, 1.0), _MM_Y))
        pr = logpdf(Normal(0, 5), a) + lkj_logconst(2, 1.0) +
            lkj_logconst(2, 1.0) +
            sum(logpdf.(Normal(0, 1), t1)) +
            sum(logpdf.(Normal(0, 1), t2)) +
            sum(logpdf.(Normal(0, 1), zmat))
        @test _query(built.spec, bound, :likelihood, u) ≈ ll
        @test _query(built.spec, bound, :prior, u) ≈ pr
        th1 = only(u[_mm_seg(lay, :L_g_s1)])
        th2 = only(u[_mm_seg(lay, :L_g_s2)])
        jac = sum(u[_mm_seg(lay, :tau_g_s1)]) +
            sum(u[_mm_seg(lay, :tau_g_s2)]) +
            log(1 - tanh(th1)^2) + log(1 - tanh(th2)^2)
        @test _query(built.spec, bound, :posterior, u) ≈ ll + pr + jac
        _check_gradient(built.spec, bound, u)
    end
    @testset "K=1 vacuous" begin
        cols = Dict{Symbol,AbstractVector}(:y => copy(_MM_Y),
            :g => [1, 2, 1, 3], :b => copy(_MM_B))
        bound, built, _, lay = _mm_query(quote
                a ~ Normal(0, 5)
                d ~ varying_draws(gr(g; by = b), [1])
                r ~ varying_slice(d, 1)
                mu = a .+ r
                y .~ Normal.(mu, 1.0)
            end, cols)
        # a + 2 empty thetas + 2 taus + 3 z.
        @test lay.total == 6
        u = collect(range(-0.4, 0.4; length = lay.total))
        a = only(u[_mm_seg(lay, :mu)])
        t1 = exp.(u[_mm_seg(lay, :tau_g_s1)])
        t2 = exp.(u[_mm_seg(lay, :tau_g_s2)])
        zf = u[_mm_seg(lay, :z_flat_g)]
        b1 = reshape(t1[1] .* zf, 3, 1)
        b2 = reshape(t2[1] .* zf, 3, 1)
        r = _mm_ref_strat_r([1, 2, 1, 3], [1, 1, 2], [b1, b2], ones(4, 1))
        ll = sum(logpdf.(Normal.(a .+ r, 1.0), _MM_Y))
        pr = logpdf(Normal(0, 5), a) + 0.0 + 0.0 +
            sum(logpdf.(Normal(0, 1), t1)) +
            sum(logpdf.(Normal(0, 1), t2)) +
            sum(logpdf.(Normal(0, 1), zf))
        @test _query(built.spec, bound, :likelihood, u) ≈ ll
        @test _query(built.spec, bound, :prior, u) ≈ pr
        jac = sum(u[_mm_seg(lay, :tau_g_s1)]) +
            sum(u[_mm_seg(lay, :tau_g_s2)])
        @test _query(built.spec, bound, :posterior, u) ≈ ll + pr + jac
        _check_gradient(built.spec, bound, u)
    end
end

# Statement-head histogram of the generated kernel (the joint-parity
# O(1) pattern): mm/stratified plates must not unroll over observations.
function _mm_statement_heads(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols))
    bound = bind_data(plan, cols)
    def = ReactiveKernelsPPL.kernel_expr(bound, assign_layout(bound))
    heads = Dict{String,Int}()
    for st in def.args[2].args
        st isa Expr || continue
        heads[string(st.head)] = get(heads, string(st.head), 0) + 1
    end
    return heads
end

function _mm_double(cols::Dict{Symbol,AbstractVector})
    return Dict{Symbol,AbstractVector}(k => vcat(v, v) for (k, v) in cols)
end

@testset "mm/stratified emission is O(1) in n_obs" begin
    mm_prog = quote
        a ~ Normal(0, 5)
        d ~ varying_draws(mm(g1, g2; weights = (w1, w2)), [1, x])
        r ~ varying_slice(d, 1:2)
        mu = a .+ r
        y .~ Normal.(mu, 1.0)
    end
    @test _mm_statement_heads(mm_prog, _mm_cols()) ==
        _mm_statement_heads(mm_prog, _mm_double(_mm_cols()))
    st_cols = Dict{Symbol,AbstractVector}(:y => copy(_MM_Y),
        :x => copy(_MM_X), :g => [1, 2, 1, 3], :b => copy(_MM_B))
    st_prog = quote
        a ~ Normal(0, 5)
        d ~ varying_draws(gr(g; by = b), [1, x])
        r ~ varying_slice(d, 1:2)
        mu = a .+ r
        y .~ Normal.(mu, 1.0)
    end
    @test _mm_statement_heads(st_prog, st_cols) ==
        _mm_statement_heads(st_prog, _mm_double(st_cols))
end

# Reactant/XLA value+grad parity at an unconstrained probe (no oracle —
# native vs compiled), plus the traced program size.
function _mm_reactant(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    post_q = prepare_query(built, bound, :sampler)
    u = [0.3 * sin(1.7i) for i in 1:built.layout.total]
    return Base.invokelatest(_mm_reactant_measure, built, bound, post_q, u)
end

function _mm_reactant_measure(built, bound, post_q, u)
    hlo = repr(Reactant.@code_hlo optimize = false post_q(Reactant.to_rarray(u)))
    native = post_q(u)
    compiled = Reactant.@compile post_q(Reactant.to_rarray(u))
    primal = Float64(compiled(Reactant.to_rarray(u)))
    q = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
    g = similar(u)
    val, _ = sampler_value_and_gradient!(q, g, u)
    cad = compile_ad_value_and_gradient(q.ad, Reactant.to_rarray(u))
    rval, rgrad = cad(Reactant.to_rarray(u))
    return (; lines = count(==('\n'), hlo), native, primal, val, g,
        rval = Float64(rval), rgrad = Array(rgrad))
end

@testset "mm/stratified under Reactant" begin
    progs = [
        ("mm correlated", quote
            a ~ Normal(0, 5)
            d ~ varying_draws(mm(g1, g2; weights = (w1, w2)), [1, x])
            r ~ varying_slice(d, 1:2)
            mu = a .+ r
            y .~ Normal.(mu, 1.0)
        end, _mm_cols()),
        ("stratified K=2", quote
            a ~ Normal(0, 5)
            d ~ varying_draws(gr(g; by = b), [1, x])
            r ~ varying_slice(d, 1:2)
            mu = a .+ r
            y .~ Normal.(mu, 1.0)
        end, Dict{Symbol,AbstractVector}(:y => copy(_MM_Y),
            :x => copy(_MM_X), :g => [1, 2, 1, 3], :b => copy(_MM_B))),
    ]
    for (name, prog, cols) in progs
        @testset "$name" begin
            fx = _mm_reactant(prog, cols)
            @test fx.primal ≈ fx.native rtol = 1e-9
            @test fx.val ≈ fx.native rtol = 1e-12
            @test fx.rval ≈ fx.native rtol = 1e-9
            @test fx.rgrad ≈ fx.g rtol = 1e-8
        end
    end
    @testset "traced program is O(1) in n_obs" begin
        _, prog, cols = progs[1]
        small = _mm_reactant(prog, cols)
        large = _mm_reactant(prog, _mm_double(cols))
        @test small.lines == large.lines
    end
end

# M1-M5/S1-S2 SB-parity probes (peer BRM literals, adopted verbatim;
# N=6 hand-written rows with a zero-weight edge row; models
# `loc ~ 1 + <ranef>`, `y ~ Normal(loc, sigma)`,
# `effect(loc, Intercept) ~ Normal(0,5)`, `sigma ~ Exponential(1)`).
# Stable byte hash over the canonical serialization (see peer brief
# `158zoc1`):
# bytes2hex(sha256(...))
# = 404ec49cad496438662c2b73cd8e44d0893874b71b5db18be2cf76edf5a83e10.
const _SB_MM_G1 = ["a", "a", "b", "c", "b", "a"]
const _SB_MM_G2 = ["b", "c", "c", "a", "a", "b"]
const _SB_MM_W1 = [2.0, 1.0, 0.0, 1.0, 3.0, 1.0]
const _SB_MM_W2 = [1.0, 1.0, 3.0, 2.0, 1.0, 1.0]
const _SB_X = [0.2, -0.1, 0.4, 0.3, -0.5, 0.1]
const _SB_Y = [0.1, 0.2, 0.3, -0.2, 0.15, 0.05]
const _SB_GR_G = ["s1", "s1", "s2", "s3", "s3", "s4"]
const _SB_GR_B = ["A", "A", "A", "B", "B", "B"]
_sb_mm_cols() = Dict{Symbol,AbstractVector}(:y => copy(_SB_Y),
    :x => copy(_SB_X), :g1 => copy(_SB_MM_G1), :g2 => copy(_SB_MM_G2),
    :w1 => copy(_SB_MM_W1), :w2 => copy(_SB_MM_W2))
_sb_gr_cols() = Dict{Symbol,AbstractVector}(:y => copy(_SB_Y),
    :x => copy(_SB_X), :g => copy(_SB_GR_G), :b => copy(_SB_GR_B))

# u_sb (SB unc order) -> u (RK layout order) via
# `(entry-name => sb-range)` pairs; inverse for gradients.
function _mm_remap_u(lay, u_sb::AbstractVector, pairs::Vector)
    u = similar(u_sb)
    for (nm, rng) in pairs
        u[_mm_seg(lay, nm)] .= u_sb[rng]
    end
    return u
end

function _mm_remap_g(g_rk::AbstractVector, lay, pairs::Vector, n_sb::Int)
    g_sb = similar(g_rk, n_sb)
    for (nm, rng) in pairs
        g_sb[rng] .= g_rk[_mm_seg(lay, nm)]
    end
    return g_sb
end

# One SB probe: RK posterior + Enzyme gradient at the remapped SB u
# against the brief's jacT value + BridgeStan-AD gradient.
function _mm_sb_check(prog::Expr, cols::Dict{Symbol,AbstractVector},
        pairs::Vector, sb_val::Float64, sb_grad::Vector{Float64})
    bound, built, _, lay = _mm_query(prog, cols)
    u_sb = collect(range(-0.4, 0.4; length = length(sb_grad)))
    u = _mm_remap_u(lay, u_sb, pairs)
    got = _query(built.spec, bound, :posterior, u)
    @test got ≈ sb_val rtol = 1e-12
    q = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
    g = similar(u)
    val, _ = sampler_value_and_gradient!(q, g, u)
    @test val ≈ sb_val rtol = 1e-12
    @test _mm_remap_g(g, lay, pairs, length(sb_grad)) ≈ sb_grad rtol = 1e-8
    return got
end

@testset "SB parity M1 mm-equal" begin
    sfx = "mm__g1__g2"
    pairs = [:mu => 1:1, Symbol("log_scale_", sfx) => 2:2,
        Symbol("xi_", sfx) => 3:5, :sigma => 6:6]
    _mm_sb_check(quote
            a ~ Normal(0, 5)
            sigma ~ Exponential(1)
            d ~ varying_draws(mm(g1, g2), [1])
            r ~ varying_slice(d, 1)
            mu = a .+ r
            y .~ Normal.(mu, sigma)
        end, _sb_mm_cols(), pairs, -15.58948984045714,
        [1.250881394124855, 0.29153139168734266, 0.4642299002163417,
            0.28655716611840265, -0.01939495660342591, -5.86641796738053])
end

@testset "SB parity M5 mm-weighted" begin
    sfx = "mm__g1__g2__w__w1__w2"
    pairs = [:mu => 1:1, Symbol("log_scale_", sfx) => 2:2,
        Symbol("xi_", sfx) => 3:5, :sigma => 6:6]
    _mm_sb_check(quote
            a ~ Normal(0, 5)
            sigma ~ Exponential(1)
            d ~ varying_draws(mm(g1, g2; weights = (w1, w2)), [1])
            r ~ varying_slice(d, 1)
            mu = a .+ r
            y .~ Normal.(mu, sigma)
        end, _sb_mm_cols(), pairs, -15.574382310331188,
        [1.236743206846504, 0.3026598991134787, 0.4652073800224629,
            0.19836679706561827, 0.05669644062510948, -5.896633027632439])
end

@testset "SB parity M3 mm-raw" begin
    sfx = "mm__g1__g2__w__w1__w2__raw"
    pairs = [:mu => 1:1, Symbol("log_scale_", sfx) => 2:2,
        Symbol("xi_", sfx) => 3:5, :sigma => 6:6]
    _mm_sb_check(quote
            a ~ Normal(0, 5)
            sigma ~ Exponential(1)
            d ~ varying_draws(mm(g1, g2; weights = (w1, w2),
                normalize = false), [1])
            r ~ varying_slice(d, 1)
            mu = a .+ r
            y .~ Normal.(mu, sigma)
        end, _sb_mm_cols(), pairs, -15.492630101384526,
        [0.9963940231145335, 0.31276498503668976, 1.0513921097313184,
            0.7277672529115509, 0.11772905659279653, -6.06013744552576])
end

@testset "SB parity M2 mm-corr" begin
    sfx = "mm__g1__g2__w__w1__w2"
    pairs = [:mu => 1:1, Symbol("L_", sfx) => 2:2,
        Symbol("tau_", sfx) => 3:4, Symbol("z_flat_", sfx) => 5:10,
        :sigma => 11:11]
    _mm_sb_check(quote
            a ~ Normal(0, 5)
            sigma ~ Exponential(1)
            d ~ varying_draws(mm(g1, g2; weights = (w1, w2)), [1, x])
            r ~ varying_slice(d, 1:2)
            mu = a .+ r
            y .~ Normal.(mu, sigma)
        end, _sb_mm_cols(), pairs, -21.14276794489944,
        [1.2077376837444953, 0.630506832103128, 0.4380159250117601,
            0.28074249933394135, 0.4590194499479547, 0.008632321192932425,
            0.22583253802326175, -0.2162400184627247, 0.008988672674119819,
            -0.2612908509632625, -5.914394132783399])
end

@testset "SB parity M4 mm-slope" begin
    sfx = "mm__g1__g2__w__w1__w2"
    pairs = [:mu => 1:1, Symbol("tau_", sfx) => 2:2,
        Symbol("z_flat_", sfx) => 3:5, :sigma => 6:6]
    _mm_sb_check(quote
            a ~ Normal(0, 5)
            sigma ~ Exponential(1)
            d ~ varying_draws(mm(g1, g2; weights = (w1, w2)), [x])
            r ~ varying_slice(d, 1)
            mu = a .+ r
            y .~ Normal.(mu, sigma)
        end, _sb_mm_cols(), pairs, -16.150519243927082,
        [1.3390094281599112, 0.3962900564469555, 0.08969746080452912,
            -0.1351518971245377, -0.15557751296907585, -5.785542552246796])
end

@testset "SB parity S2 gr-int" begin
    pairs = [:mu => 1:1, :tau_g_s1 => 2:2, :tau_g_s2 => 3:3,
        :z_flat_g => 4:7, :sigma => 8:8]
    _mm_sb_check(quote
            a ~ Normal(0, 5)
            sigma ~ Exponential(1)
            d ~ varying_draws(gr(g; by = b), [1])
            r ~ varying_slice(d, 1)
            mu = a .+ r
            y .~ Normal.(mu, sigma)
        end, _sb_gr_cols(), pairs, -18.507393826337786,
        [1.145341170105935, 0.42507826583901354, 0.34282266028297126,
            0.45756876030978466, 0.16471983334259416, 0.003138445644975174,
            -0.2064866000560754, -5.884658580828696])
end

@testset "SB parity S1 gr-corr" begin
    pairs = [:mu => 1:1, :L_g_s1 => 2:2, :L_g_s2 => 3:3,
        :tau_g_s1 => 4:5, :tau_g_s2 => 6:7, :z_flat_g => 8:15,
        :sigma => 16:16]
    _mm_sb_check(quote
            a ~ Normal(0, 5)
            sigma ~ Exponential(1)
            d ~ varying_draws(gr(g; by = b), [1, x])
            r ~ varying_slice(d, 1:2)
            mu = a .+ r
            y .~ Normal.(mu, sigma)
        end, _sb_gr_cols(), pairs, -26.70296217977992,
        [1.0807184231225622, 0.6767830395897837, 0.5495863505829115,
            0.38756444882142826, 0.32100344408274084, 0.2871856402541647,
            0.13217327990018723, 0.4242400424375435, -0.012366222688957774,
            0.10351650824855292, -0.04838614643697972, 0.020748011648512643,
            -0.3345450858779599, -0.22836648276224303, -0.33989497827367926,
            -5.922002826385627])
end
