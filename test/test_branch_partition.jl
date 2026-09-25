using ReactiveKernels
using DifferentiationInterface
using Enzyme
using Test

isdefined(@__MODULE__, :BranchPartition) ||
    include("fixtures/branch_partition.jl")

# A plate cell whose lazy branch reads bound data only is split at preparation
# into one branch-free plate per taken arm (docs/src/constraints.md).
@testset "data-bound branch partitioning of authored plates" begin
    C = BranchPartition
    RK = ReactiveKernels
    plates(p) = filter(r -> r.op isa RK._AuthoredPlateOp, p.recipes)
    has_branch(p) = any(r -> r.op isa RK._KernelSourceOp &&
                             r.op.f isa RK._KernelBranch, p.recipes)

    x = [0.5, 1.5, 2.0, 0.25, 3.0, 1.25, 0.75]
    y = [1.0, -2.0, 0.5, -1.0, 2.0, -0.5, 1.5]
    reference_guarded(x, y) = sum(yi > 0 ? log(xi * yi) : xi - 1.0
                                  for (xi, yi) in zip(x, y))

    @testset "two arms, each over its own lanes" begin
        C.arm_calls[] = 0
        plain = prepare(C.guarded; have = (:x, :y), want = :total)
        @test plain(x, y) ≈ reference_guarded(x, y)
        @test C.arm_calls[] == count(>(0), y) # lazy per lane when unbound
        bound = prepare(C.guarded; have = (:x, :y), want = :total, bound = (; y))
        @test length(plates(bound.plan)) == 2
        @test !any(r -> has_branch(plate_body(r)), plates(bound.plan))
        C.arm_calls[] = 0
        @test bound(x) ≈ reference_guarded(x, y)
        @test C.arm_calls[] == count(>(0), y)
        # A one-sided condition selects its arm statically: one plate, no branch.
        ypos = abs.(y) .+ 0.5
        one = prepare(C.guarded; have = (:x, :y), want = :total, bound = (; y = ypos))
        @test length(plates(one.plan)) == 1
        @test !has_branch(plate_body(only(plates(one.plan))))
        @test one(x) ≈ reference_guarded(x, ypos)
    end

    @testset "pointwise output is reassembled in lane order" begin
        bound = prepare(C.guarded; have = (:x, :y), want = :pointwise, bound = (; y))
        @test length(plates(bound.plan)) == 2
        @test bound(x) ≈ [yi > 0 ? log(xi * yi) : xi - 1.0 for (xi, yi) in zip(x, y)]
    end

    @testset "nested arms and in-bounds gathers" begin
        cuts = [-1.0, 0.0, 1.5]
        level = [1, 2, 4, 3, 1, 4, 2, 3]
        eta = [0.3, -0.2, 0.9, 0.1, -0.4, 0.6, 0.2, -0.7]
        w = [1.0, 0.5, 2.0, 1.5, 1.0, 0.25, 1.0, 2.0]
        K = length(cuts) + 1
        ref(eta, level, cuts, w) = sum(wi * (l == 1 ? cuts[1] - e :
            (l == K ? e - cuts[end] : cuts[l] * e - cuts[l - 1]))
            for (l, e, wi) in zip(level, eta, w))
        have = (:eta, :level, :cuts, :w, :nlev)
        plain = prepare(C.leveled; have, want = :total)
        @test plain(eta, level, cuts, w, K) ≈ ref(eta, level, cuts, w)
        bound = prepare(C.leveled; have, want = :total,
                        bound = (; level, w, nlev = K))
        @test length(plates(bound.plan)) == 3
        @test !any(r -> has_branch(plate_body(r)), plates(bound.plan))
        @test bound(eta, cuts) ≈ ref(eta, level, cuts, w)
        # The same program for any threshold count: arms depend on the cell,
        # never on the number of levels.
        cuts6 = [-2.0, -1.0, 0.0, 0.5, 1.5, 3.0]
        level6 = [1, 7, 3, 5, 2, 7, 4, 6]
        K6 = length(cuts6) + 1
        bound6 = prepare(C.leveled; have, want = :total,
                         bound = (; level = level6, w, nlev = K6))
        @test length(plates(bound6.plan)) == 3
        @test bound6(eta, cuts6) ≈ sum(wi * (l == 1 ? cuts6[1] - e :
            (l == K6 ? e - cuts6[end] : cuts6[l] * e - cuts6[l - 1]))
            for (l, e, wi) in zip(level6, eta, w))
        # Reverse through the partitioned plates: the analytic score.
        backend = AutoEnzyme(; mode = Enzyme.Reverse)
        live_eta = prepare(C.leveled; have, want = :total,
                           bound = (; level, w, nlev = K, cuts))
        @test length(plates(live_eta.plan)) == 3
        pb = prepare_ad(live_eta, backend, eta; active = :eta)
        value, grad = ReactiveKernels.ad_value_and_gradient!(pb, similar(eta), eta)
        @test value ≈ ref(eta, level, cuts, w)
        @test grad ≈ [wi * (l == 1 ? -1.0 : l == K ? 1.0 : cuts[l])
                      for (l, wi) in zip(level, w)]
    end

    @testset "a bound-only cell condition partitions after caching" begin
        yv = [2.0, 0.5, 1.0, 3.0, 0.1]
        xv = [1.5, 2.5, 3.5, 0.5, 1.0]
        ref = sum(yi >= 1.0 ? log(xi) * yi : 2xi for (xi, yi) in zip(xv, yv))
        bound = prepare(C.derived_condition; have = (:x, :y), want = :total,
                        bound = (; y = yv))
        @test length(plates(bound.plan)) == 2
        C.arm_calls[] = 0
        @test bound(xv) ≈ ref
        @test C.arm_calls[] == count(>=(1.0), yv)
    end

    @testset "a constant arm contributes once per lane" begin
        ref(x, y) = sum(yi > 0 ? xi * yi : 1.0 for (xi, yi) in zip(x, y))
        bound = prepare(C.constant_fallback; have = (:x, :y), want = :total,
                        bound = (; y))
        @test length(plates(bound.plan)) == 2
        @test bound(x) ≈ ref(x, y)
        pointwise = prepare(C.constant_fallback; have = (:x, :y),
                            want = :pointwise, bound = (; y))
        @test pointwise(x) ≈ [yi > 0 ? xi * yi : 1.0 for (xi, yi) in zip(x, y)]
        # All lanes take the constant arm: selected statically, still per lane.
        yneg = -abs.(y)
        allconst = prepare(C.constant_fallback; have = (:x, :y), want = :total,
                           bound = (; y = yneg))
        @test length(plates(allconst.plan)) == 1
        @test allconst(x) ≈ length(x)
    end

    @testset "a live condition stays a lazy branch" begin
        bound = prepare(C.live_condition; have = (:x, :s), want = :total,
                        bound = (; s = 1.0))
        # `s` is bound but broadcast: the condition still reads live `x`.
        @test length(plates(bound.plan)) == 1
        @test has_branch(plate_body(only(plates(bound.plan))))
        @test bound(x) ≈ sum(abs(xi - 1.0) for xi in x)
    end

    @testset "partitioned recipes keep the readable view named" begin
        # The docs Generated-kernel pane refuses an `operation(` callee
        # (docs/kernel_examples.jl): the partition's synthesized lane ops must
        # render under their own names, not as an opaque operation slot.
        for want in (:total, :pointwise)
            bound = prepare(C.guarded; have = (:x, :y), want, bound = (; y))
            readable = string(RK._readable_expr(RK.code_expr(bound), bound))
            @test !occursin(r"__ops__\[\d+\]", readable)
            @test !occursin(r"\boperation\(", readable)
        end
        pointwise = prepare(C.guarded; have = (:x, :y), want = :pointwise,
                            bound = (; y))
        readable = string(RK._readable_expr(RK.code_expr(pointwise), pointwise))
        @test occursin("lane_gather", readable)
        @test occursin("lane_assemble", readable)
    end
end

# A constructed endpoint under a lazy branch arm lowers in its arm: each taken
# arm carries its own endpoint evaluation, the arm stays lazy (an endpoint
# that is invalid off-arm never evaluates there), and the inlined math keeps
# the backend's ordinary reverse mode.
@testset "constructed endpoints under branch arms" begin
    C = BranchPartition
    RK = ReactiveKernels
    plates(p) = filter(r -> r.op isa RK._AuthoredPlateOp, p.recipes)
    has_branch(p) = any(r -> r.op isa RK._KernelSourceOp &&
                             r.op.f isa RK._KernelBranch, p.recipes)
    nlogpdf(x, m, s) = -0.5 * log(2π) - log(s) - 0.5 * ((x - m) / s)^2

    y_full = [0.5, -1.0, 2.0, 0.25]
    lp = [0.0, 1.0, -0.5, 0.75]
    mis_pos = [0, 1, 0, 1]
    ref_guarded = sum(mp == 0 ? nlogpdf(yf, lpi, 1.5) : 0.0 * yf
                      for (yf, lpi, mp) in zip(y_full, lp, mis_pos))

    @testset "missingness guard splits per arm" begin
        plain = prepare(C.guarded_endpoint; have = (:y_full, :lp, :mis_pos),
                        want = :total)
        @test plain(y_full, lp, mis_pos) ≈ ref_guarded
        bound = prepare(C.guarded_endpoint; have = (:y_full, :lp, :mis_pos),
                        want = :total, bound = (; y_full, mis_pos))
        @test length(plates(bound.plan)) == 2
        @test !any(r -> has_branch(plate_body(r)), plates(bound.plan))
        @test bound(lp) ≈ ref_guarded
        pointwise = prepare(C.guarded_endpoint; have = (:y_full, :lp, :mis_pos),
                            want = :pointwise, bound = (; y_full, mis_pos))
        @test pointwise(lp) ≈
            [mp == 0 ? nlogpdf(yf, lpi, 1.5) : 0.0 * yf
             for (yf, lpi, mp) in zip(y_full, lp, mis_pos)]
    end

    @testset "an invalid-off-arm endpoint stays lazy" begin
        # `scale = yf` is nonpositive on else-lanes: `log` there throws iff
        # the endpoint evaluates off-arm, through either path.
        yy = [1.5, -1.0, 2.0, -0.5]
        mm = [0.1, 0.2, 0.3, 0.4]
        gg = [0, 1, 0, 1]
        ref = nlogpdf(0.1, 0.0, 1.5) + nlogpdf(0.3, 0.0, 2.0)
        plain = prepare(C.lazy_endpoint_arm; have = (:y, :m, :g), want = :total)
        @test plain(yy, mm, gg) ≈ ref
        bound = prepare(C.lazy_endpoint_arm; have = (:y, :m, :g), want = :total,
                        bound = (; y = yy, g = gg))
        @test length(plates(bound.plan)) == 2
        @test bound(mm) ≈ ref
        nested = prepare(C.nested_endpoint_arm; have = (:y, :m, :g),
                         want = :total)
        @test nested(yy, mm, gg) ≈ 2ref
    end

    @testset "else arm, named and computed bindings, && guard" begin
        ref_else = sum(mp == 0 ? 0.0 : nlogpdf(yf, lpi, 1.5)
                       for (yf, lpi, mp) in zip(y_full, lp, mis_pos))
        bound_else = prepare(C.else_endpoint_arm; have = (:y, :m, :g),
                             want = :total, bound = (; y = y_full, g = mis_pos))
        @test bound_else(lp) ≈ ref_else
        sg = [1.0, 2.0, 1.5, 0.5]
        ref_named = sum(gf == 0 ? nlogpdf(yf, mf, sf) : 0.0
                        for (yf, mf, sf, gf) in zip(y_full, lp, sg, mis_pos))
        named = prepare(C.named_endpoint_arm; have = (:y, :m, :s, :g),
                        want = :total, bound = (; y = y_full, g = mis_pos))
        @test named(lp, sg) ≈ ref_named
        ref_computed =
            sum(gf == 0 ? nlogpdf(log(abs(yf) + 1), mf + 0.25, exp(mf)) : 0.0
                for (yf, mf, gf) in zip(y_full, lp, mis_pos))
        computed = prepare(C.computed_endpoint_arm; have = (:y, :m, :g),
                           want = :total)
        @test computed(y_full, lp, mis_pos) ≈ ref_computed
        ref_and = sum((gf == 0 && yf > 0) ? nlogpdf(yf, mf, 1.5) : 0.0
                      for (yf, mf, gf) in zip(y_full, lp, mis_pos))
        anded = prepare(C.and_endpoint_arm; have = (:y, :m, :g), want = :total,
                        bound = (; y = y_full, g = mis_pos))
        @test anded(lp) ≈ ref_and
    end

    @testset "condition position and top level" begin
        ref_cond = sum(nlogpdf(yf, mf, 1.5) > -2.0 ? 1.0 : -1.0
                       for (yf, mf) in zip(y_full, lp))
        conded = prepare(C.condition_endpoint; have = (:y, :m), want = :total)
        @test conded(y_full, lp) ≈ ref_cond
        top = prepare(C.toplevel_endpoint_arm)
        @test top(0.5, 0.0, 0) ≈ nlogpdf(0.5, 0.0, 1.5)
        @test top(0.5, 0.0, 1) ≈ 0.0
    end

    @testset "reverse through the split arm plates" begin
        bound = prepare(C.guarded_endpoint; have = (:y_full, :lp, :mis_pos),
                        want = :total, bound = (; y_full, mis_pos))
        backend = AutoEnzyme(; mode = Enzyme.Reverse)
        pb = prepare_ad(bound, backend, lp; active = :lp)
        value, grad =
            ReactiveKernels.ad_value_and_gradient!(pb, similar(lp), lp)
        @test value ≈ ref_guarded
        @test grad ≈ [(mp == 0 ? (yf - lpi) / 1.5^2 : 0.0)
                      for (yf, lpi, mp) in zip(y_full, lp, mis_pos)]
    end

    @testset "a non-Base global is rejected loudly" begin
        ex = :(@kernel reject_endpoint_arm(y::Vector{Float64},
                                           m::Vector{Float64},
                                           g::Vector{Int}) = begin
            pointwise = plate(y, m, g) do yf, mf, gf
                cell::Float64 = gf == 0 ?
                    helper_normal(mf, 1.5).logpdf(yf) : 0.0
                cell
            end
            total::Float64 = sum(pointwise)
        end)
        err = try
            macroexpand(C, ex)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("cannot be evaluated under a branch arm", sprint(showerror, err))
        @test occursin("hoist the endpoint call above the branch", sprint(showerror, err))
    end
end
