using DifferentiationInterface
using Distributions
using Enzyme
using ReactiveKernels
using ReactiveKernelsPPL
using Test

# Panel-kernel (KernelPlate) reader tests: P2 mirrors against the BRM SB
# oracles (bit-exact values + constrained gradients), the committed
# pred-kernel fixture vs hand-computed Gaussian math, and the fail-closed
# battery (surface / structure / bind). Oracle provenance: BRM parent
# corpus todo 14bv4nq / brief 1imaflv, verified on c13f41c.

const _KERNEL_BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)

_normal_logpdf(y, mu, s) =
    -0.5 * log(2pi) - log(s) - 0.5 * ((y - mu) / s)^2

function _kernel_findiff(f, u; h = cbrt(eps(Float64)))
    g = Vector{Float64}(undef, length(u))
    for i in eachindex(u)
        up = Vector{Float64}(u)
        up[i] += h
        dn = Vector{Float64}(u)
        dn[i] -= h
        g[i] = (f(up) - f(dn)) / (2h)
    end
    return g
end

# Evaluate a resolved plate's canonical cell assignments against flat data
# (observes intermediates the kernel never returns): slice params bind
# their flat refs, globals bind constrained values, assignments run in
# order, and the collected name is returned.
function _eval_kernel_cell(kp::KernelPlate, columns::AbstractDict{Symbol},
        globals::Dict{Symbol,<:Real})
    binds = Expr(:block)
    for (col, param, kind) in kp.slices
        flat = (kind === :vector || kp.timepoints === nothing) ? columns[col] :
            columns[ReactiveKernelsPPL._kexp_name(kp.result, col)]
        push!(binds.args, Expr(:(=), param, flat))
    end
    for (nm, v) in globals
        push!(binds.args, Expr(:(=), nm, Float64(v)))
    end
    stmts = Expr(:block)
    for (nm, ex) in kp.assignments
        push!(stmts.args, Expr(:(=), nm, ex))
    end
    push!(stmts.args, kp.collected)
    return Core.eval(Module(), Expr(:let, binds, stmts))
end

function _pk1cmt_ast()
    return quote
        sigma ~ Exponential(1.0)
        pred ~ plate(t, dose, dv, CL, Vc, Ka; subjects = kernel_nsub_pred) do ts, d, yy, CLi, Vci, Kai
            ke = CLi / Vci
            mu = d * Kai / (Vci * (Kai - ke)) .* (exp.(-ke .* ts) .- exp.(-Kai .* ts))
            yy .~ Normal.(mu, sigma)
            mu
        end
    end
end

function _pk1cmt_columns()
    t1 = [0.5, 1.0, 2.0, 4.0]
    dv1 = [1.0, 2.0, 1.5, 0.8]
    return Dict{Symbol,AbstractVector}(
        :t => vcat(t1, t1, t1),
        :dose => [100.0, 100.0, 100.0],
        :dv => vcat(dv1, dv1, dv1),
        :CL => [5.0, 6.0, 4.5],
        :Vc => [50.0, 55.0, 48.0],
        :Ka => [1.0, 1.2, 0.9],
    )
end

@testset "P2 Ex1: 1-cmt oral PK panel vs SB oracle" begin
    data_names = (:CL, :Ka, :Vc, :dose, :dv, :t)
    unbound = lower_rkppl(_pk1cmt_ast(), data_names; conditioned = data_names)
    @test isempty(unbound.responses)
    kp0 = only(unbound.kernel_plates)
    @test kp0.result === :pred
    @test kp0.subjects === :kernel_nsub_pred
    @test kp0.timepoints === nothing
    @test [p for (_, p, _) in kp0.slices] == [:ts, :d, :yy, :CLi, :Vci, :Kai]
    @test all(s -> s[3] === :unknown, kp0.slices)
    # The emitter passes scalar-context user code verbatim (undotted).
    @test kp0.assignments[1] == (:ke => :(CLi / Vci))
    @test only(kp0.obs).response === :yy
    @test only(kp0.obs).family === GaussianFam
    @test only(kp0.obs).location === :mu
    @test only(kp0.obs).scale === :sigma
    @test kp0.collected === :mu

    dims = Dict{Symbol,Int}(:kernel_nsub_pred => 3, :kernel_T_pred => 4)
    bound = bind_data(unbound, _pk1cmt_columns(); dims)
    kp = only(bound.kernel_plates)
    @test kp.subjects == 3
    @test kp.timepoints == 4
    @test [k for (_, _, k) in kp.slices] ==
        [:vector, :scalar, :vector, :scalar, :scalar, :scalar]
    @test bound.n_obs == 12
    @test bound.roles[:dv] === :response
    # Bind canonicalizes with slice-kind provenance (dotify Ex1's undotted
    # scalar-context ops over flat vectors).
    @test kp.assignments[1] == (:ke => :(CLi ./ Vci))
    @test haskey(bound.columns, :pred_kexp_dose)
    @test bound.columns[:pred_kexp_CL] == repeat([5.0, 6.0, 4.5]; inner = 4)

    # Per-subject mu (flat subject-order blocks) vs the oracle series.
    mu1 = [0.7659972550846236, 1.193239948587816, 1.518656599647487, 1.448898682548678]
    mu2 = [0.7962076603469246, 1.190909376688747, 1.4265225940838275, 1.2763057758286027]
    mu3 = [0.7362291031335232, 1.1719551200917093, 1.5435586743228227, 1.534803619403906]
    mu_flat = _eval_kernel_cell(kp, bound.columns, Dict{Symbol,Real}(:sigma => 1.0))
    @test mu_flat ≈ vcat(mu1, mu2, mu3) atol = 1e-12

    # Execution parity at constrained sigma = 1.0 (u = [0.0]): posterior
    # == ll + prior (logjac(0) = 0); the query gradient is unconstrained
    # d/dq == oracle constrained d/dσ + dlogjac/dq (exp-Jacobian: +1).
    built = build_kernel(bound)
    u0 = [0.0]
    @test prepare_query(built, bound, :sampler)(u0) ≈ -13.703526816545866 atol = 1e-9
    @test prepare_query(built, bound, :likelihood)(u0) ≈ -12.703526816545866 atol = 1e-9
    @test prepare_query(built, bound, :prior)(u0) ≈ -1.0 atol = 1e-12
    @test prepare_query(built, bound, :log_jacobian)(u0) ≈ 0.0 atol = 1e-12
    q = prepare_sampler(built, bound, u0; backend = _KERNEL_BACKEND)
    g = Vector{Float64}(undef, 1)
    val, _ = sampler_value_and_gradient!(q, g, u0)
    @test val ≈ -13.703526816545866 atol = 1e-9
    @test g[1] ≈ -9.647471163820416 + 1.0 atol = 1e-8
    @test _kernel_findiff(u -> prepare_query(built, bound, :sampler)(u), u0)[1] ≈
        g[1] atol = 1e-6
end

function _doseplate_ast()
    return quote
        sigma ~ Exponential(1.0)
        pred ~ plate(dose, dv, ls; subjects = kernel_nsub_pred) do dd, yy, lsi
            mu = (dd ./ 10.0) .* exp.(lsi)
            yy .~ Normal.(mu, sigma)
            mu
        end
    end
end

@testset "P2 Ex2: scalar dose-response plate vs SB oracle" begin
    data_names = (:dose, :dv, :ls)
    unbound = lower_rkppl(_doseplate_ast(), data_names; conditioned = data_names)
    columns = Dict{Symbol,AbstractVector}(
        :dose => [100.0, 100.0, 100.0, 100.0],
        :dv => [0.5, 1.2, 2.1, 3.3],
        :ls => [0.1, 0.2, 0.15, 0.25],
    )
    # All-scalar: no T key at all (not T == 1).
    bound = bind_data(unbound, columns; dims = Dict{Symbol,Int}(:kernel_nsub_pred => 4))
    kp = only(bound.kernel_plates)
    @test kp.subjects == 4
    @test kp.timepoints === nothing
    @test all(s -> s[3] === :scalar, kp.slices)
    @test bound.n_obs == 4
    @test !any(n -> startswith(string(n), "pred_kexp_"), keys(bound.columns))

    mu_oracle = [11.051709180756477, 12.214027581601698, 11.61834242728283, 12.840254166877415]
    mu_flat = _eval_kernel_cell(kp, bound.columns, Dict{Symbol,Real}(:sigma => 1.0))
    @test mu_flat ≈ mu_oracle atol = 1e-12

    built = build_kernel(bound)
    u0 = [0.0]
    @test prepare_query(built, bound, :sampler)(u0) ≈ -211.80708530040758 atol = 1e-9
    @test prepare_query(built, bound, :likelihood)(u0) ≈ -210.80708530040758 atol = 1e-9
    @test prepare_query(built, bound, :prior)(u0) ≈ -1.0 atol = 1e-12
    q = prepare_sampler(built, bound, u0; backend = _KERNEL_BACKEND)
    g = Vector{Float64}(undef, 1)
    val, _ = sampler_value_and_gradient!(q, g, u0)
    @test val ≈ -211.80708530040758 atol = 1e-9
    @test g[1] ≈ 409.2626623351777 + 1.0 atol = 1e-7
end

@testset "pred fixture (BRM rk_emitter e2e) vs hand math" begin
    # BRM `rk_emitter.jl` "kernel(...) end-to-end via _brm_rk_plan": n_sub
    # = 2, T = 3, slices [t vector, dose scalar, obs vector], globals
    # sigma/b0. The emitted AST is quoted verbatim (dotted obs).
    ast = quote
        sigma ~ Exponential(1.0)
        b0 ~ Normal(0.0, 1.0)
        pred ~ plate(t, dose, obs; subjects = kernel_nsub_pred) do ts, d, yy
            mu = (b0 .* d) .* ts
            yy .~ Normal.(mu, sigma)
            mu
        end
    end
    columns = Dict{Symbol,AbstractVector}(
        :t => [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        :dose => [10.0, 20.0],
        :obs => [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    )
    unbound = lower_rkppl(ast, (:dose, :obs, :t); conditioned = (:dose, :obs, :t))
    bound = bind_data(unbound, columns;
        dims = Dict{Symbol,Int}(:kernel_nsub_pred => 2, :kernel_T_pred => 3))
    kp = only(bound.kernel_plates)
    @test [k for (_, _, k) in kp.slices] == [:vector, :scalar, :vector]
    @test bound.columns[:pred_kexp_dose] == [10.0, 10.0, 10.0, 20.0, 20.0, 20.0]

    built = build_kernel(bound)
    names = coordinate_names(built.layout)
    @test Set(names) == Set([:sigma, :b0])
    # Constrained (sigma, b0) = (2.0, 0.5); sigma rides exp.
    u = [n === :sigma ? log(2.0) : 0.5 for n in names]
    dex = [10.0, 10.0, 10.0, 20.0, 20.0, 20.0]
    mu = (0.5 .* dex) .* columns[:t]
    ll = sum(_normal_logpdf(y, m, 2.0) for (y, m) in zip(columns[:obs], mu))
    prior = -2.0 + _normal_logpdf(0.5, 0.0, 1.0)
    want = ll + prior + log(2.0)
    @test prepare_query(built, bound, :sampler)(u) ≈ want atol = 1e-10
    @test prepare_query(built, bound, :likelihood)(u) ≈ ll atol = 1e-10
    q = prepare_sampler(built, bound, u; backend = _KERNEL_BACKEND)
    g = Vector{Float64}(undef, 2)
    val, _ = sampler_value_and_gradient!(q, g, u)
    @test val ≈ want atol = 1e-10
    fd = _kernel_findiff(x -> prepare_query(built, bound, :sampler)(x), u)
    @test g ≈ fd atol = 1e-6
end

@testset "T == 1 recovers scalar (unobservable kinds)" begin
    ast = quote
        sigma ~ Exponential(1.0)
        pred ~ plate(t, dose, obs; subjects = 2) do ts, d, yy
            mu = (b0 .* d) .* ts
            yy .~ Normal.(mu, sigma)
            mu
        end
        b0 ~ Normal(0.0, 1.0)
    end
    columns = Dict{Symbol,AbstractVector}(
        :t => [1.0, 2.0],
        :dose => [10.0, 20.0],
        :obs => [0.1, 0.4],
    )
    unbound = lower_rkppl(ast, (:dose, :obs, :t); conditioned = (:dose, :obs, :t))
    bound = bind_data(unbound, columns; dims = Dict{Symbol,Int}(:kernel_T_pred => 1))
    kp = only(bound.kernel_plates)
    @test kp.subjects == 2
    @test kp.timepoints == 1
    @test all(s -> s[3] === :scalar, kp.slices)
    @test bound.n_obs == 2
    built = build_kernel(bound)
    names = coordinate_names(built.layout)
    u = [n === :sigma ? 0.0 : 0.5 for n in names]
    mu = (0.5 .* [10.0, 20.0]) .* [1.0, 2.0]
    ll = sum(_normal_logpdf(y, m, 1.0) for (y, m) in zip([0.1, 0.4], mu))
    @test prepare_query(built, bound, :likelihood)(u) ≈ ll atol = 1e-12
end

@testset "kernel surface fail-closed battery" begin
    data = (:t, :dose, :obs)
    good_cell = [
        :(mu = (b0 .* d) .* ts),
        :(yy .~ Normal.(mu, sigma)),
        :mu,
    ]
    function plate_ast(cell, kw; lhs = :pred, cols = [:t, :dose, :obs],
            params = [:ts, :d, :yy], tilde = :~)
        call = Expr(:call, :plate, Expr(:parameters, kw...), cols...)
        lam = Expr(:->, Expr(:tuple, params...), Expr(:block, cell...))
        return Expr(:block,
            Expr(:call, :~, :sigma, Expr(:call, :Exponential, 1.0)),
            Expr(:call, :~, :b0, Expr(:call, :Normal, 0.0, 1.0)),
            Expr(:call, tilde, lhs, Expr(:do, call, lam)))
    end
    subj = Expr(:kw, :subjects, :kernel_nsub_pred)
    # refused: `.~` on the plate statement; the plate yields one collected result, `~` is scalar, `.~` broadcasts (P3)
    @test_throws "carries a scalar `~`" lower_rkppl(
        plate_ast(good_cell, [subj]; tilde = :.~), data; conditioned = data)
    # refused: scalar `~` over vector slices; `.~` broadcasts (P3, explicit-dots ruling)
    @test_throws "scalar `~` over vectors" lower_rkppl(
        plate_ast([good_cell[1], :(yy ~ Normal(mu, sigma)), :mu], [subj]), data; conditioned = data)
    # refused: the cell observes `yy` twice (single assignment)
    @test_throws "more than one `.~`" lower_rkppl(
        plate_ast([good_cell[1], good_cell[2], good_cell[2], :mu], [subj]), data; conditioned = data)
    # capability: deterministic panel cells collect their value with zero likelihood (todo `1qlbn5b`).
    free = lower_rkppl(plate_ast([good_cell[1], :mu], [subj]), data; conditioned = data)
    @test isempty(only(free.kernel_plates).obs)
    # refused: plate cell has no trailing collected name; a `.~` statement has no value to collect (P2, P3)
    @test_throws "collected result name" lower_rkppl(
        plate_ast([good_cell[1], good_cell[2]], [subj]), data; conditioned = data)
    # refused: bare `mu` read before its definition (Julia UndefVarError, P3)
    @test_throws "only the trailing statement" lower_rkppl(
        plate_ast([:mu, good_cell[1], good_cell[2], :mu], [subj]), data; conditioned = data)
    # v2 (ordered contract change): panel admits the scalar
    # response-space set — Poisson lowers (1-arg, scaleless node).
    pois = lower_rkppl(
        plate_ast([good_cell[1], :(yy .~ Poisson.(mu)), :mu], [subj]), data; conditioned = data)
    @test only(only(pois.kernel_plates).obs).family === PoissonLogFam
    @test only(only(pois.kernel_plates).obs).scale === nothing
    # Grouped-only joint heads fail closed in panel cells.
    # refused: TgiResponse is a BRM-specific joint head; domain functions use ordinary explicit parameters (P10, 1cmodra domain)
    @test_throws SurfaceLoweringError lower_rkppl(
        plate_ast([good_cell[1],
            :(yy .~ TgiResponse.(mu, sigma, a, b, c, d)), :mu], [subj]), data; conditioned = data)
    # Fused link-space heads fail closed naming the pre-assignment fix.
    # capability: fused link-space obs heads in cells (Distributions.BernoulliLogit) (todo `05fuzch`)
    @test_broken (lower_rkppl(
        plate_ast([good_cell[1], :(yy .~ BernoulliLogit.(mu)), :mu],
            [subj]), data; conditioned = data); true)
    # Response-space Binomial threads integer trials and probability values.
    # capability: Binomial in-cell observations with explicit trials and probabilities (todo `1qlbn5b`)
    binom = lower_rkppl(plate_ast([good_cell[1], :(yy .~ Binomial.(2, logistic.(mu))), :mu],
        [subj]), data; conditioned = data)
    @test only(only(binom.kernel_plates).obs).family === BinomialProbFam
    # Unknown heads fail with the admitted list.
    # capability: arbitrary Distributions families as in-cell observations (Cauchy) (todo `1qlbn5b`)
    cauchy_plan = lower_rkppl(plate_ast([good_cell[1], :(yy .~ Cauchy.(mu, sigma)), :mu],
        [subj]), data; conditioned = data)
    @test only(only(cauchy_plan.kernel_plates).obs).family === CauchyFam
    # 1-arg arity pins.
    # refused: malformed distribution, Poisson takes one argument (Julia MethodError, P3)
    @test_throws "exactly 1 arguments" lower_rkppl(
        plate_ast([good_cell[1], :(yy .~ Poisson.(mu, sigma)), :mu], [subj]), data; conditioned = data)
    # 3-arg StudentT arity pin (nu, mu, sigma — Distributions order).
    # refused: malformed distribution, StudentT takes (nu, mu, sigma)
    @test_throws "exactly 3 arguments" lower_rkppl(
        plate_ast([good_cell[1], :(yy .~ StudentT.(mu, sigma)), :mu], [subj]), data; conditioned = data)
    # refused: undotted `Normal(mu, sigma)` over vector `mu` is a Julia MethodError (P3)
    @test_throws "obs broadcasts" lower_rkppl(
        plate_ast([good_cell[1], :(yy .~ Normal(mu, sigma)), :mu], [subj]), data; conditioned = data)
    # Arity message generalized to digits when the joint families joined
    # the obs table (same fail-closed behavior).
    # admitted: 1-arg `Normal.(mu)` (Distributions default sigma = 1) in cells (todo `139j2uo`)
    @test (lower_rkppl(
        plate_ast([good_cell[1], :(yy .~ Normal.(mu)), :mu], [subj]), data; conditioned = data); true)
    # capability: inline expression arguments in in-cell observations (`Normal.(mu .+ 1.0, sigma)`) (todo `0fkd9yk`)
    inline = lower_rkppl(plate_ast([good_cell[1], :(yy .~ Normal.(mu .+ 1.0, sigma)), :mu],
        [subj]), data; conditioned = data)
    @test :yy_arg1 in first.(only(inline.kernel_plates).assignments)
    # refused: undeclared name `zz` observed (P6, 05oe96l)
    @test_throws "not a slice param" lower_rkppl(
        plate_ast([good_cell[1], :(zz .~ Normal.(mu, sigma)), :mu], [subj]), data; conditioned = data)
    # refused: panel plate missing its required `subjects=` keyword (subject count underdetermined; P2, P3)
    @test_throws "needs `subjects=N`" lower_rkppl(
        plate_ast(good_cell, []), data; conditioned = data)
    # refused: non-positive subject count (mathematically invalid input)
    @test_throws "must be positive" lower_rkppl(
        plate_ast(good_cell, [Expr(:kw, :subjects, 0)]), data; conditioned = data)
    # capability: computed `subjects=` expression (`1 + 1`) (todo `0fkd9yk`)
    @test_broken (lower_rkppl(
        plate_ast(good_cell, [Expr(:kw, :subjects, :(1 + 1))]), data; conditioned = data); true)
    # refused: unknown keyword `group` (Julia MethodError, P3)
    @test_throws "exactly one keyword" lower_rkppl(
        plate_ast(good_cell, [subj, Expr(:kw, :group, :g)]), data; conditioned = data)
    # refused: slice column `zz` is not bound data (undeclared name, P6)
    @test_throws "not bound data" lower_rkppl(
        plate_ast(good_cell, [subj]; cols = [:t, :dose, :zz]), data; conditioned = data)
    # refused: do-block arity mismatch, 2 params for 3 columns (Julia MethodError, P3)
    @test_throws "one param per column" lower_rkppl(
        plate_ast(good_cell, [subj]; params = [:ts, :d]), data; conditioned = data)
    # refused: collected name `zz` is undeclared (P6, 05oe96l)
    @test_throws "not a cell name" lower_rkppl(
        plate_ast([good_cell[1], good_cell[2], :zz], [subj]), data; conditioned = data)
    # A plate may carry a likelihood beside a top-level response.
    resp = LikelihoodSpec(GaussianFam, IdentityLink, :y, :mu, :s, nothing,
        ResponseEvidence(:none, nothing, nothing), :y_resp)
    pred_spec = PredictorSpec(:mu, IdentityLink,
        TermSpec[TermSpec(InterceptTerm, ColumnRef[], NamedTuple(),
            :Intercept, :intercept)], :mu)
    sparam = SampledParameter(:s, :exponential, (arg1 = 1.0,), nothing, :s)
    kp_hand = KernelPlate(:pred, 1, nothing, [(:t, :ts, :scalar)],
        Pair{Symbol,Any}[],
        (response = :ts, family = GaussianFam, location = :ts, scale = :s,
            params = ()),
        :ts, :pred)
    both_plan = StructuralPlan([resp], [pred_spec],
        PopulationPrior[PopulationPrior(:mu, :Intercept, 0.0, 1.0)],
        [sparam], AssignmentSpec[], Dict{Symbol,AbstractVector}(), 0;
        kernel_plates = [kp_hand])
    @test validate_structure(both_plan) === nothing
    # Two panel plates in one model (v2: panels compose freely —
    # distinct cell names; shared names trip single-assignment first).
    # A shared subjects dims key consumes once.
    cell2 = [
        :(mu2 = (b0 .* d2) .* ts2),
        :(yy2 .~ Normal.(mu2, sigma)),
        :mu2,
    ]
    two = Expr(:block,
        plate_ast(good_cell, [subj]).args...,
        plate_ast(cell2, [subj]; lhs = :pred2,
            params = [:ts2, :d2, :yy2]).args[3])
    two_plan = lower_rkppl(two, data; conditioned = data)
    @test [kp.result for kp in two_plan.kernel_plates] == [:pred, :pred2]
    # Duplicate plate results trip single-assignment at the surface.
    dupe = Expr(:block,
        plate_ast(good_cell, [subj]).args...,
        plate_ast(cell2, [subj]; params = [:ts2, :d2, :yy2]).args[3])
    # refused: plate result `pred` defined twice (single assignment)
    @test_throws "defined twice" lower_rkppl(dupe, data; conditioned = data)
    # Prior-only declarations are valid model statements.
    nolhs = Expr(:block, Expr(:call, :~, :sigma, Expr(:call, :Exponential, 1.0)))
    # admitted: prior-only model (already supported on the canonical base).
    @test (lower_rkppl(nolhs, (:y,); conditioned = (:y,)); true)
end

@testset "kernel cell + bind fail-closed battery" begin
    data = (:t, :dose, :obs)
    subj = Expr(:kw, :subjects, :kernel_nsub_pred)
    function plate_ast(cell, kw = [subj])
        call = Expr(:call, :plate, Expr(:parameters, kw...), :t, :dose, :obs)
        lam = Expr(:->, Expr(:tuple, :ts, :d, :yy), Expr(:block, cell...))
        return Expr(:block,
            Expr(:call, :~, :sigma, Expr(:call, :Exponential, 1.0)),
            Expr(:call, :~, :b0, Expr(:call, :Normal, 0.0, 1.0)),
            Expr(:call, :~, :pred, Expr(:do, call, lam)))
    end
    cols6 = Dict{Symbol,AbstractVector}(
        :t => [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        :dose => [10.0, 20.0],
        :obs => [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    )
    dims = Dict{Symbol,Int}(:kernel_nsub_pred => 2, :kernel_T_pred => 3)
    mu_stmt = :(mu = (b0 .* d) .* ts)
    obs_stmt = :(yy .~ Normal.(mu, sigma))
    # Series reductions return one value per subject.
    # capability: per-subject series reductions (`mean(ts)`) in a panel cell (panel v1) (todo `1qlbn5b`)
    reduced = lower_rkppl(plate_ast([:(m = mean(ts)), mu_stmt, obs_stmt, :mu]), data;
        conditioned = data)
    @test :m in first.(only(reduced.kernel_plates).assignments)
    # Supported: module functions consume whole values in ordinary
    # plate cells (P3; todo `0bfiemp`), with explicit covariance arguments.
    @test validate_structure(lower_rkppl(quote
        @plate for i in eachindex(y)
            cs = cumsum(grid)
            K = gp_exp_quad_cov(grid,1.0,1.0,1e-9)
            y[i] ~ Normal(sum(K) + sum(cs),1.0)
        end
    end,(:y,:grid);conditioned=(:y,:grid))) === nothing
    # Cross-cell refs fail closed.
    # refused: undeclared name `zz` (P6, 05oe96l)
    @test_throws "unknown name" lower_rkppl(
        plate_ast([:(mu = (b0 .* d) .* zz), obs_stmt, :mu]), data; conditioned = data)
    # Scalar-obs literal domains (v2 — response-space roles).
    # refused: Bernoulli probability 2.0 outside [0, 1] (mathematically invalid input)
    @test_throws "not a probability" lower_rkppl(
        plate_ast([mu_stmt, :(yy .~ Bernoulli.(2.0)), :mu]), data; conditioned = data)
    # refused: negative Poisson mean (mathematically invalid input)
    @test_throws "not a nonnegative mean" lower_rkppl(
        plate_ast([mu_stmt, :(yy .~ Poisson.(-1.0)), :mu]), data; conditioned = data)
    # refused: NegativeBinomial2 phi = 0 (mathematically invalid input)
    @test_throws "response-space phi" lower_rkppl(
        plate_ast([mu_stmt, :(yy .~ NegativeBinomial2.(mu, 0.0)), :mu]), data; conditioned = data)
    # refused: Gamma alpha = 0 (mathematically invalid input)
    @test_throws "response-space alpha" lower_rkppl(
        plate_ast([mu_stmt, :(yy .~ Gamma.(0.0, sigma)), :mu]), data; conditioned = data)
    # refused: negative Beta b (mathematically invalid input)
    @test_throws "response-space b" lower_rkppl(
        plate_ast([mu_stmt, :(yy .~ Beta.(mu, -1.0)), :mu]), data; conditioned = data)
    # refused: StudentT nu = 0 (mathematically invalid input)
    @test_throws "positive degrees of freedom" lower_rkppl(
        plate_ast([mu_stmt, :(yy .~ StudentT.(0.0, mu, sigma)), :mu]), data; conditioned = data)
    # refused: StudentT sigma = 0 (mathematically invalid input)
    @test_throws "response-space sigma" lower_rkppl(
        plate_ast([mu_stmt, :(yy .~ StudentT.(4.0, mu, 0.0)), :mu]), data; conditioned = data)
    # v1 Gaussian scale message byte-preserved through the scalar path.
    # refused: Normal sigma = 0 (mathematically invalid input)
    @test_throws "must be finite positive" lower_rkppl(
        plate_ast([mu_stmt, :(yy .~ Normal.(mu, 0.0)), :mu]), data; conditioned = data)
    # Undotted over 2+ genuinely-vector operands fails at bind (kinds
    # resolve there): ts and yy are both vector slices here.
    two_vec_unbound = lower_rkppl(
        plate_ast([:(mu = ts * yy), obs_stmt, :mu]), data; conditioned = data)
    # refused: undotted vector*vector `ts * yy` is a Julia MethodError (P3)
    @test_throws "write the dotted form" bind_data(two_vec_unbound, cols6; dims)
    # Scalar-context undotted canonicalizes instead (Ex1 shape).
    ok = bind_data(lower_rkppl(
        plate_ast([:(ke = 2.0 * 3.0), mu_stmt, obs_stmt, :mu]), data; conditioned = data), cols6; dims)
    @test only(ok.kernel_plates).assignments[1] == (:ke => :(2.0 * 3.0))
    # Unbound subjects dims key.
    unbound = lower_rkppl(plate_ast([mu_stmt, obs_stmt, :mu]), data; conditioned = data)
    # refused: subjects dims key not bound (missing data)
    @test_throws "not bound" bind_data(unbound, cols6;
        dims = Dict{Symbol,Int}(:kernel_T_pred => 3))
    # Leftover dims keys fail closed (typo'd key). Surface nodes leave
    # timepoints symbolic (leftover-key T rule), so leftovers past T need
    # an explicit timepoints key (hand-built shape).
    kp0 = only(unbound.kernel_plates)
    kp_exp = KernelPlate(kp0.result, kp0.subjects, :kernel_T_pred, kp0.slices,
        kp0.assignments, kp0.obs, kp0.collected, kp0.label)
    unbound_exp = StructuralPlan(unbound.responses, unbound.predictors,
        unbound.population_priors, unbound.parameters, unbound.assignments,
        Dict{Symbol,AbstractVector}(), 0; kernel_plates = [kp_exp])
    # refused: typo'd dims key not consumed (wrong data)
    @test_throws "not consumed" bind_data(unbound_exp, cols6;
        dims = Dict{Symbol,Int}(:kernel_nsub_pred => 2, :kernel_T_pred => 3,
            :kernel_T_preed => 3))
    # No key other than `kernel_T_<result>` is taken as T: a typo'd or
    # stray key is never inferred to be the timepoint count (it used to
    # bind T = 3 silently when it was the only leftover key).
    # refused: no T dims key bound, stray key not inferred as T (wrong data; P2)
    @test_throws "no T dims key is bound" bind_data(unbound, cols6;
        dims = Dict{Symbol,Int}(:kernel_nsub_pred => 2, :whatever_typo => 3))
    # refused: no T dims key bound, stray keys not inferred as T (wrong data; P2)
    @test_throws "no T dims key is bound" bind_data(unbound, cols6;
        dims = Dict{Symbol,Int}(:kernel_nsub_pred => 2, :T1 => 3, :T2 => 3))
    # All-scalar slices need no T: a stray key is unconsumed, not T.
    scalar6 = Dict{Symbol,AbstractVector}(
        :t => [1.0, 2.0], :dose => [10.0, 20.0], :obs => [0.1, 0.4])
    # refused: stray dims key not consumed by any plate (wrong data)
    @test_throws "not consumed by any kernel plate" bind_data(unbound,
        scalar6; dims = Dict{Symbol,Int}(:kernel_nsub_pred => 2, :T => 3))
    # The `kernel_T_<result>` convention names the plate's T key; a
    # second key beside it is a stray, not an ambiguity.
    # refused: stray dims key not consumed by any plate (wrong data)
    @test_throws "not consumed by any kernel plate" bind_data(unbound, cols6;
        dims = Dict{Symbol,Int}(:kernel_nsub_pred => 2, :kernel_T_pred => 3,
            :kernel_T_pred2 => 3))
    # T bound but unused (T > 1, all scalar).
    scalar_cols = Dict{Symbol,AbstractVector}(
        :t => [1.0, 2.0], :dose => [10.0, 20.0], :obs => [0.1, 0.4])
    # refused: T bound but no vector slice uses it (inconsistent dims/data)
    @test_throws "no vector slice uses it" bind_data(unbound, scalar_cols; dims)
    # Vector-shaped column without a T key.
    # refused: vector-length column with no T dims key (missing data)
    @test_throws "no T dims key is bound" bind_data(unbound, cols6;
        dims = Dict{Symbol,Int}(:kernel_nsub_pred => 2))
    # Lengths matching neither n_sub nor n_sub * T.
    bad_cols = Dict{Symbol,AbstractVector}(
        :t => [0.0, 1.0, 2.0, 0.0], :dose => [10.0, 20.0],
        :obs => [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    # refused: column lengths match neither n_sub nor n_sub*T (length mismatch)
    @test_throws "neither n_sub" bind_data(unbound, bad_cols; dims)
    # Non-numeric scalar slice with T bound: the numeric gate fires ahead
    # of the T-block Float64 conversion (not a raw MethodError).
    bad_str = merge(cols6, Dict{Symbol,AbstractVector}(:dose => ["a", "b"]))
    # refused: non-numeric (String) slice column (wrong eltype)
    @test_throws "must be numeric" bind_data(unbound, bad_str; dims)
    bad_sym = merge(cols6, Dict{Symbol,AbstractVector}(:dose => [:a, :b]))
    # refused: non-numeric (Symbol) slice column (wrong eltype)
    @test_throws "must be numeric" bind_data(unbound, bad_sym; dims)
    # Count responses reject non-count columns at bind.
    float_cols = Dict{Symbol,AbstractVector}(
        :t => [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        :dose => [10.0, 20.0],
        :obs => [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    )
    pois_unbound = lower_rkppl(
        plate_ast([mu_stmt, :(yy .~ Poisson.(mu)), :mu]), data; conditioned = data)
    # refused: Poisson response not non-negative integers (wrong eltype)
    @test_throws "non-negative integers" bind_data(pois_unbound, float_cols; dims)
    bern_unbound = lower_rkppl(
        plate_ast([mu_stmt, :(yy .~ Bernoulli.(mu)), :mu]), data; conditioned = data)
    # refused: Bernoulli response not Bool/0-1 (wrong eltype)
    @test_throws "Bool or 0/1" bind_data(bern_unbound, float_cols; dims)
    # Scalar count slices keep their integer type when repeated over T.
    scalar_count_cols = Dict{Symbol,AbstractVector}(
        :t => [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        :dose => [10.0, 20.0],
        :obs => [1, 2],
    )
    # capability: count (Poisson/NB2) response on a scalar slice in a T-model (Int expansion; Bernoulli twin already admits it) (todo `1308iv0`)
    scalar_count_bound = bind_data(pois_unbound, scalar_count_cols; dims)
    @test scalar_count_bound.columns[:pred_kexp_obs] == [1,1,1,2,2,2]
    @test eltype(scalar_count_bound.columns[:pred_kexp_obs]) === Int
    # Gamma/Beta response domains.
    gamma_unbound = lower_rkppl(
        plate_ast([mu_stmt, :(yy .~ Gamma.(sigma, mu)), :mu]), data; conditioned = data)
    zero_cols = Dict{Symbol,AbstractVector}(
        :t => [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        :dose => [10.0, 20.0],
        :obs => [0.0, 0.2, 0.3, 0.4, 0.5, 0.6],
    )
    # refused: Gamma response 0.0 outside support (invalid data)
    @test_throws "strictly positive" bind_data(gamma_unbound, zero_cols; dims)
    beta_unbound = lower_rkppl(
        plate_ast([mu_stmt, :(yy .~ Beta.(sigma, mu)), :mu]), data; conditioned = data)
    oob_cols = Dict{Symbol,AbstractVector}(
        :t => [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        :dose => [10.0, 20.0],
        :obs => [0.1, 0.2, 1.5, 0.4, 0.5, 0.6],
    )
    # refused: Beta response 1.5 outside (0, 1) (invalid data)
    @test_throws "strictly inside (0, 1)" bind_data(beta_unbound, oob_cols; dims)
    # Nonpositive dims values.
    # refused: non-positive subjects dims value (invalid input)
    @test_throws "positive integer" bind_data(unbound, cols6;
        dims = Dict{Symbol,Int}(:kernel_nsub_pred => 0, :kernel_T_pred => 3))
    # Caller collision with a materialized expansion name.
    collide = merge(cols6, Dict{Symbol,AbstractVector}(
        :pred_kexp_dose => [10.0, 10.0, 10.0, 20.0, 20.0, 20.0]))
    # refused: caller column collides with reserved kernel expansion name
    @test_throws "reserved for kernel" bind_data(unbound, collide; dims)
    # Subjects literal + T-only dims resolves (leftover-key T rule).
    lit_ast = Expr(:block,
        Expr(:call, :~, :sigma, Expr(:call, :Exponential, 1.0)),
        Expr(:call, :~, :b0, Expr(:call, :Normal, 0.0, 1.0)),
        Expr(:call, :~, :pred, Expr(:do,
            Expr(:call, :plate, Expr(:parameters, Expr(:kw, :subjects, 2)),
                :t, :dose, :obs),
            Expr(:->, Expr(:tuple, :ts, :d, :yy),
                Expr(:block, mu_stmt, obs_stmt, :mu)))))
    lit = bind_data(lower_rkppl(lit_ast, data; conditioned = data), cols6;
        dims = Dict{Symbol,Int}(:kernel_T_pred => 3))
    kp_lit = only(lit.kernel_plates)
    @test kp_lit.subjects == 2
    @test kp_lit.timepoints == 3
end

# --- Axis 2: non-gaussian panel obs (values + Enzyme vs findiff) ------------

const _AXIS2_T = [0.5, 1.0, 2.0, 0.5, 1.0, 2.0]
const _AXIS2_DOSE = [1.0, 2.0]
const _AXIS2_DEX = [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
const _AXIS2_DIMS = Dict{Symbol,Int}(:kernel_nsub_pred => 2, :kernel_T_pred => 3)

function _axis2_build(cell, obsy; params = [:(b0 ~ Normal(0.0, 1.0))])
    ast = Expr(:block, params...,
        Expr(:call, :~, :pred, Expr(:do,
            Expr(:call, :plate,
                Expr(:parameters, Expr(:kw, :subjects, :kernel_nsub_pred)),
                :t, :dose, :obs),
            Expr(:->, Expr(:tuple, :ts, :d, :yy),
                Expr(:block, cell...)))))
    columns = Dict{Symbol,AbstractVector}(
        :t => _AXIS2_T, :dose => _AXIS2_DOSE, :obs => obsy)
    bound = bind_data(lower_rkppl(ast, (:dose, :obs, :t); conditioned = (:dose, :obs, :t)), columns;
        dims = _AXIS2_DIMS)
    return build_kernel(bound), bound
end

function _axis2_value_grad(built, bound, u, ll, prior, jac)
    @test prepare_query(built, bound, :likelihood)(u) ≈ ll atol = 1e-10
    @test prepare_query(built, bound, :prior)(u) ≈ prior atol = 1e-10
    @test prepare_query(built, bound, :sampler)(u) ≈ ll + prior + jac atol = 1e-10
    q = prepare_sampler(built, bound, u; backend = _KERNEL_BACKEND)
    g = Vector{Float64}(undef, length(u))
    val, _ = sampler_value_and_gradient!(q, g, u)
    @test val ≈ ll + prior + jac atol = 1e-10
    @test g ≈ _kernel_findiff(
        x -> prepare_query(built, bound, :sampler)(x), u) atol = 1e-6
end

@testset "axis2 Poisson plate vs Distributions oracle" begin
    built, bound = _axis2_build(
        [:(mu = exp.(b0 .* d .* ts)), :(yy .~ Poisson.(mu)), :mu],
        [1, 0, 2, 3, 1, 0])
    @test only(only(bound.kernel_plates).obs).family === PoissonLogFam
    names = coordinate_names(built.layout)
    @test names == [:b0]
    u = [0.25]
    mu = exp.(0.25 .* _AXIS2_DEX .* _AXIS2_T)
    ll = sum(logpdf.(Poisson.(mu), [1, 0, 2, 3, 1, 0]))
    _axis2_value_grad(built, bound, u, ll, logpdf(Normal(0.0, 1.0), 0.25), 0.0)
end

@testset "axis2 Bernoulli plate (Int lanes) vs oracle" begin
    built, bound = _axis2_build(
        [:(eta = b0 .* d .* ts), :(p = 1 ./ (1 .+ exp.(-eta))),
            :(yy .~ Bernoulli.(p)), :p],
        [1, 0, 1, 1, 0, 0])
    @test only(only(bound.kernel_plates).obs).family === BernoulliLogitFam
    # Int lanes read their bind-materialized Bool twin (exact 0/1;
    # the `!=` comparison misdifferentiates under native Enzyme —
    # snag `bernoulli-int-la-78487520`). Dense Vector{Bool}: BitArrays
    # are overlay-hostile (reactant ladder-1b).
    twin = bound.columns[ReactiveKernelsPPL._kbool_name(:pred, :yy)]
    @test twin isa Vector{Bool}
    @test twin == Bool[1, 0, 1, 1, 0, 0]
    # Hand-bound plans verify the twin (verified, not trusted).
    notwin = deepcopy(bound)
    delete!(notwin.columns, ReactiveKernelsPPL._kbool_name(:pred, :yy))
    # refused: a hand-bound plan deleted the required Bernoulli Bool twin;
    # this is missing generated metadata, rather than a Bool numeric value.
    @test_throws ContractValidationError ReactiveKernelsPPL._validate_kernels_data(notwin)
    u = [0.5]
    eta = 0.5 .* _AXIS2_DEX .* _AXIS2_T
    p = 1 ./ (1 .+ exp.(-eta))
    ll = sum(logpdf.(Bernoulli.(p), Bool[1, 0, 1, 1, 0, 0]))
    _axis2_value_grad(built, bound, u, ll, logpdf(Normal(0.0, 1.0), 0.5), 0.0)
end

@testset "axis2 Bernoulli scalar response twins its expansion" begin
    # Mechanical shape (one y per subject over T lanes — the
    # likelihood counts each 3x): pins expansion→twin→lanes wiring.
    ast = Expr(:block, :(b0 ~ Normal(0.0, 1.0)),
        Expr(:call, :~, :pred, Expr(:do,
            Expr(:call, :plate,
                Expr(:parameters, Expr(:kw, :subjects, :kernel_nsub_pred)),
                :t, :dose, :obs),
            Expr(:->, Expr(:tuple, :ts, :d, :yy), Expr(:block,
                :(eta = b0 .* d .* ts),
                :(p = 1 ./ (1 .+ exp.(-eta))),
                :(yy .~ Bernoulli.(p)), :p)))))
    columns = Dict{Symbol,AbstractVector}(
        :t => _AXIS2_T, :dose => _AXIS2_DOSE, :obs => Bool[1, 0])
    bound = bind_data(lower_rkppl(ast, (:dose, :obs, :t); conditioned = (:dose, :obs, :t)), columns;
        dims = _AXIS2_DIMS)
    # Scalar Bool responses stay Bool through their flat expansion.
    expanded = bound.columns[:pred_kexp_obs]
    @test expanded isa Vector{Bool}
    @test expanded == Bool[1, 1, 1, 0, 0, 0]
    built = build_kernel(bound)
    u = [0.5]
    eta = 0.5 .* _AXIS2_DEX .* _AXIS2_T
    p = 1 ./ (1 .+ exp.(-eta))
    ll = sum(logpdf.(Bernoulli.(p), Bool[1, 1, 1, 0, 0, 0]))
    _axis2_value_grad(built, bound, u, ll, logpdf(Normal(0.0, 1.0), 0.5), 0.0)
end

@testset "axis2 Bernoulli plate (Bool lanes) vs oracle" begin
    built, bound = _axis2_build(
        [:(eta = b0 .* d .* ts), :(p = 1 ./ (1 .+ exp.(-eta))),
            :(yy .~ Bernoulli.(p)), :p],
        Bool[1, 0, 1, 1, 0, 0])
    u = [0.5]
    eta = 0.5 .* _AXIS2_DEX .* _AXIS2_T
    p = 1 ./ (1 .+ exp.(-eta))
    ll = sum(logpdf.(Bernoulli.(p), Bool[1, 0, 1, 1, 0, 0]))
    _axis2_value_grad(built, bound, u, ll, logpdf(Normal(0.0, 1.0), 0.5), 0.0)
end

@testset "axis2 NB2 plate vs Distributions oracle" begin
    built, bound = _axis2_build(
        [:(mu = exp.(b0 .* d .* ts)), :(yy .~ NegativeBinomial2.(mu, phi)), :mu],
        [1, 0, 2, 3, 1, 0];
        params = [:(b0 ~ Normal(0.0, 1.0)), :(phi ~ Exponential(1.0))])
    names = coordinate_names(built.layout)
    fixed = Dict(:b0 => 0.25, :phi => 2.0)
    u = [n === :phi ? log(fixed[n]) : fixed[n] for n in names]
    mu = exp.(0.25 .* _AXIS2_DEX .* _AXIS2_T)
    # Stan NB2(mu, phi) == NegativeBinomial(phi, phi/(mu+phi)).
    nb = NegativeBinomial.(2, 2.0 ./ (mu .+ 2.0))
    ll = sum(logpdf.(nb, [1, 0, 2, 3, 1, 0]))
    prior = logpdf(Normal(0.0, 1.0), 0.25) + logpdf(Exponential(1.0), 2.0)
    _axis2_value_grad(built, bound, u, ll, prior, log(2.0))
end

@testset "axis2 Gamma plate (symbolic scale) vs oracle" begin
    built, bound = _axis2_build(
        [:(mu = exp.(b0 .* d .* ts)), :(sc = mu ./ alpha),
            :(yy .~ Gamma.(alpha, sc)), :mu],
        [0.5, 1.2, 2.1, 0.8, 1.5, 2.5];
        params = [:(b0 ~ Normal(0.0, 1.0)), :(alpha ~ Exponential(1.0))])
    names = coordinate_names(built.layout)
    fixed = Dict(:b0 => 0.25, :alpha => 2.0)
    u = [n === :alpha ? log(fixed[n]) : fixed[n] for n in names]
    mu = exp.(0.25 .* _AXIS2_DEX .* _AXIS2_T)
    sc = mu ./ 2.0
    ll = sum(logpdf.(Gamma.(2.0, sc), [0.5, 1.2, 2.1, 0.8, 1.5, 2.5]))
    prior = logpdf(Normal(0.0, 1.0), 0.25) + logpdf(Exponential(1.0), 2.0)
    _axis2_value_grad(built, bound, u, ll, prior, log(2.0))
end

@testset "axis2 Beta plate vs Distributions oracle" begin
    built, bound = _axis2_build(
        [:(mu = 1 ./ (1 .+ exp.(-b0 .* d .* ts))), :(a = mu .* kappa),
            :(b = (1 .- mu) .* kappa), :(yy .~ Beta.(a, b)), :mu],
        [0.2, 0.7, 0.5, 0.3, 0.8, 0.4];
        params = [:(b0 ~ Normal(0.0, 1.0)), :(kappa ~ Exponential(1.0))])
    names = coordinate_names(built.layout)
    fixed = Dict(:b0 => 0.5, :kappa => 3.0)
    u = [n === :kappa ? log(fixed[n]) : fixed[n] for n in names]
    mu = 1 ./ (1 .+ exp.(-0.5 .* _AXIS2_DEX .* _AXIS2_T))
    ll = sum(logpdf.(Beta.(mu .* 3.0, (1 .- mu) .* 3.0),
        [0.2, 0.7, 0.5, 0.3, 0.8, 0.4]))
    prior = logpdf(Normal(0.0, 1.0), 0.5) + logpdf(Exponential(1.0), 3.0)
    _axis2_value_grad(built, bound, u, ll, prior, log(3.0))
end

@testset "axis2 StudentT plate vs Distributions oracle" begin
    built, bound = _axis2_build(
        [:(mu = (b0 .* d) .* ts), :(yy .~ StudentT.(nu, mu, sigma)), :mu],
        [0.5, 1.2, 2.1, 0.8, 1.5, 2.5];
        params = [:(b0 ~ Normal(0.0, 1.0)), :(sigma ~ Exponential(1.0)),
            :(nu ~ Gamma(2.0, 0.1))])
    names = coordinate_names(built.layout)
    fixed = Dict(:b0 => 0.5, :sigma => 1.5, :nu => 4.0)
    u = [n === :b0 ? fixed[n] : log(fixed[n]) for n in names]
    mu = (0.5 .* _AXIS2_DEX) .* _AXIS2_T
    td = LocationScale.(mu, 1.5, Ref(TDist(4.0)))
    ll = sum(logpdf.(td, [0.5, 1.2, 2.1, 0.8, 1.5, 2.5]))
    prior = logpdf(Normal(0.0, 1.0), 0.5) + logpdf(Exponential(1.0), 1.5) +
        logpdf(Gamma(2.0, 0.1), 4.0)
    _axis2_value_grad(built, bound, u, ll, prior, log(1.5) + log(4.0))
end

@testset "axis2 literal obs args (folded, not threaded)" begin
    built, bound = _axis2_build(
        [:(mu = exp.(b0 .* d .* ts)), :(sc = mu ./ 2.0),
            :(yy .~ Gamma.(2.0, sc)), :mu],
        [0.5, 1.2, 2.1, 0.8, 1.5, 2.5])
    names = coordinate_names(built.layout)
    @test names == [:b0]
    u = [0.25]
    mu = exp.(0.25 .* _AXIS2_DEX .* _AXIS2_T)
    ll = sum(logpdf.(Gamma.(2.0, mu ./ 2.0), [0.5, 1.2, 2.1, 0.8, 1.5, 2.5]))
    _axis2_value_grad(built, bound, u, ll, logpdf(Normal(0.0, 1.0), 0.25), 0.0)
    # StudentT with literal nu + sigma (params slot literal).
    built2, bound2 = _axis2_build(
        [:(mu = (b0 .* d) .* ts), :(yy .~ StudentT.(4.0, mu, 1.5)), :mu],
        [0.5, 1.2, 2.1, 0.8, 1.5, 2.5])
    mu2 = (0.5 .* _AXIS2_DEX) .* _AXIS2_T
    ll2 = sum(logpdf.(LocationScale.(mu2, 1.5, Ref(TDist(4.0))),
        [0.5, 1.2, 2.1, 0.8, 1.5, 2.5]))
    _axis2_value_grad(built2, bound2, [0.5], ll2,
        logpdf(Normal(0.0, 1.0), 0.5), 0.0)
end

# The generated program's statement count is O(1) in the data (flat
# codegen — constraints.md acceptance): tiling subjects must not
# change the emitted program's shape.
function _kernel_statement_heads(bound)
    def = ReactiveKernelsPPL.kernel_expr(bound, assign_layout(bound))
    heads = Dict{String,Int}()
    for st in def.args[2].args
        st isa Expr || continue
        h = string(st.head)
        if h == "=" && st.args[2] isa Expr && st.args[2].head === :call &&
                st.args[2].args[1] isa Symbol
            h = "=call:" * string(st.args[2].args[1])
        end
        heads[h] = get(heads, h, 0) + 1
    end
    return heads
end

@testset "panel emission is O(1) in the subject count" begin
    cell = [:(mu = exp.(b0 .* d .* ts)), :(yy .~ Poisson.(mu)), :mu]
    small = Dict{Symbol,AbstractVector}(
        :t => _AXIS2_T, :dose => _AXIS2_DOSE, :obs => [1, 0, 2, 3, 1, 0])
    big = Dict{Symbol,AbstractVector}(
        :t => vcat(_AXIS2_T, _AXIS2_T), :dose => vcat(_AXIS2_DOSE, _AXIS2_DOSE),
        :obs => [1, 0, 2, 3, 1, 0, 1, 0, 2, 3, 1, 0])
    mkast() = Expr(:block, :(b0 ~ Normal(0.0, 1.0)),
        Expr(:call, :~, :pred, Expr(:do,
            Expr(:call, :plate,
                Expr(:parameters, Expr(:kw, :subjects, :kernel_nsub_pred)),
                :t, :dose, :obs),
            Expr(:->, Expr(:tuple, :ts, :d, :yy), Expr(:block, cell...)))))
    b2 = bind_data(lower_rkppl(mkast(), (:dose, :obs, :t); conditioned = (:dose, :obs, :t)), small;
        dims = Dict{Symbol,Int}(:kernel_nsub_pred => 2, :kernel_T_pred => 3))
    b4 = bind_data(lower_rkppl(mkast(), (:dose, :obs, :t); conditioned = (:dose, :obs, :t)), big;
        dims = Dict{Symbol,Int}(:kernel_nsub_pred => 4, :kernel_T_pred => 3))
    @test _kernel_statement_heads(b2) == _kernel_statement_heads(b4)
end

# Joint-3 fixture shared by the value test and the battery: two
# all-scalar panel plates, unequal n=4/n=3, shared b0, sigma on plate 1
# only. n_obs is the total lanes (4+3).
function _axis3_joint3()
    ast = quote
        b0 ~ Normal(0.0, 1.0)
        sigma ~ Exponential(1.0)
        pred1 ~ plate(x1, y1; subjects = kernel_nsub_pred1) do xx1, yy1
            mu1 = b0 .* xx1
            yy1 .~ Normal.(mu1, sigma)
            mu1
        end
        pred2 ~ plate(x2, y2; subjects = kernel_nsub_pred2) do xx2, yy2
            mu2 = exp.(b0 .* xx2)
            yy2 .~ Poisson.(mu2)
            mu2
        end
    end
    columns = Dict{Symbol,AbstractVector}(
        :x1 => [0.5, 1.0, 1.5, 2.0], :y1 => [0.4, 1.1, 1.4, 2.2],
        :x2 => [0.5, 1.0, 1.5], :y2 => [1, 2, 3])
    dims = Dict{Symbol,Int}(:kernel_nsub_pred1 => 4, :kernel_nsub_pred2 => 3)
    return ast, columns, dims
end

@testset "axis3 joint-3 two-plate gaussian+poisson vs SB" begin
    # Joint primary (SB mirror: partner brief
    # `BayesianRegressionModels:rk:kernel:plate/briefs/2026-09-27T17-16-01-132-1u4svoe`).
    ast, columns, dims = _axis3_joint3()
    x1, y1 = columns[:x1], columns[:y1]
    x2, y2 = columns[:x2], columns[:y2]
    bound =
        bind_data(lower_rkppl(ast, (:x1, :y1, :x2, :y2); conditioned = (:x1, :y1, :x2, :y2)), columns; dims = dims)
    @test bound.n_obs == 7
    @test [kp.result for kp in bound.kernel_plates] == [:pred1, :pred2]
    @test [kp.timepoints for kp in bound.kernel_plates] == [nothing, nothing]
    built = build_kernel(bound)
    names = coordinate_names(built.layout)
    u = [n === :b0 ? 0.5 : log(1.5) for n in names]
    # SB value (propto=false + Jacobian, u = [b0, sigma-unc]).
    @test prepare_query(built, bound, :sampler)(u) ≈ -11.969630233025292 atol = 1e-9
    # Independent Distributions oracle (lane-exact + sigma Jacobian).
    mu1 = 0.5 .* x1
    ll1 = sum(logpdf.(Normal.(mu1, 1.5), y1))
    mu2 = exp.(0.5 .* x2)
    ll2 = sum(logpdf.(Poisson.(mu2), y2))
    want = ll1 + ll2 + logpdf(Normal(0.0, 1.0), 0.5) +
        logpdf(Exponential(1.0), 1.5) + log(1.5)
    @test prepare_query(built, bound, :sampler)(u) ≈ want atol = 1e-10
    q = prepare_sampler(built, bound, u; backend = _KERNEL_BACKEND)
    g = Vector{Float64}(undef, length(u))
    val, _ = sampler_value_and_gradient!(q, g, u)
    @test val ≈ want atol = 1e-10
    # SB grad in SB u-order [b0, sigma-unc]; map by name.
    sb = Dict(:b0 => 2.833765996036988, :sigma => -3.5022222222222226)
    for (i, n) in enumerate(names)
        @test g[i] ≈ sb[n] atol = 1e-8
    end
    @test g ≈ _kernel_findiff(
        x -> prepare_query(built, bound, :sampler)(x), u) atol = 1e-6
end

@testset "axis3 multi-plate fail-closed battery" begin
    ast, columns, dims = _axis3_joint3()
    unbound = lower_rkppl(ast, (:x1, :y1, :x2, :y2); conditioned = (:x1, :y1, :x2, :y2))
    # Stray dims keys fail once, globally (a key for no plate at all).
    stray = merge(dims, Dict{Symbol,Int}(:kernel_T_preed => 3))
    # refused: stray dims key not consumed by any plate (wrong data)
    @test_throws "not consumed by any kernel plate" bind_data(unbound, columns; dims = stray)
    # Several plates may contribute beside a top-level response.
    resp = LikelihoodSpec(GaussianFam, IdentityLink, :y, :mu, :s, nothing,
        ResponseEvidence(:none, nothing, nothing), :y_resp)
    pred_spec = PredictorSpec(:mu, IdentityLink,
        TermSpec[TermSpec(InterceptTerm, ColumnRef[], NamedTuple(),
            :Intercept, :intercept)], :mu)
    sparam = SampledParameter(:s, :exponential, (arg1 = 1.0,), nothing, :s)
    mkplate(result, col, param) = KernelPlate(result, 1, nothing,
        [(col, param, :scalar)], Pair{Symbol,Any}[],
        (response = param, family = GaussianFam, location = param, scale = :s,
            params = ()),
        param, result)
    both2 = StructuralPlan([resp], [pred_spec],
        PopulationPrior[PopulationPrior(:mu, :Intercept, 0.0, 1.0)],
        [sparam], AssignmentSpec[], Dict{Symbol,AbstractVector}(), 0;
        kernel_plates = [mkplate(:pred1, :t, :ts),
            mkplate(:pred2, :t2, :ts2)])
    @test validate_structure(both2) === nothing
    # Hand-bound n_obs must be the lanes sum (positive control first).
    bound = bind_data(unbound, columns; dims = dims)
    mkhand(n) = StructuralPlan(bound.responses, bound.predictors,
        bound.population_priors, bound.parameters, bound.assignments,
        bound.columns, n; kernel_plates = bound.kernel_plates)
    @test validate_data(mkhand(7)) === nothing
    # refused: hand-bound n_obs differs from the total likelihood lanes.
    @test_throws "total likelihood lanes" validate_data(mkhand(6))
    # A shared subjects key consumes once (all-scalar pair).
    shr = quote
        b0 ~ Normal(0.0, 1.0)
        sigma ~ Exponential(1.0)
        pa ~ plate(x, y; subjects = kernel_nsub) do xx, yy
            ma = b0 .* xx
            yy .~ Normal.(ma, sigma)
            ma
        end
        pb ~ plate(x, y; subjects = kernel_nsub) do xx2, yy2
            mb = b0 .* xx2
            yy2 .~ Normal.(mb, sigma)
            mb
        end
    end
    shr_cols = Dict{Symbol,AbstractVector}(:x => [1.0, 2.0], :y => [0.5, 1.5])
    shr_bound = bind_data(lower_rkppl(shr, (:x, :y); conditioned = (:x, :y)), shr_cols;
        dims = Dict{Symbol,Int}(:kernel_nsub => 2))
    @test shr_bound.n_obs == 4
    # Per-plate timepoints via the convention (bind-level, no values).
    t_ast = quote
        b0 ~ Normal(0.0, 1.0)
        p1 ~ plate(t, dose, obs; subjects = kernel_nsub_p1) do ts, d, yy
            m1 = exp.(b0 .* d .* ts)
            yy .~ Poisson.(m1)
            m1
        end
        p2 ~ plate(t, dose, obs; subjects = kernel_nsub_p2) do ts2, d2, yy2
            m2 = exp.(b0 .* d2 .* ts2)
            yy2 .~ Poisson.(m2)
            m2
        end
    end
    t_cols = Dict{Symbol,AbstractVector}(
        :t => [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        :dose => [10.0, 20.0],
        :obs => [1, 0, 2, 3, 1, 0])
    t_bound = bind_data(lower_rkppl(t_ast, (:t, :dose, :obs); conditioned = (:t, :dose, :obs)), t_cols;
        dims = Dict{Symbol,Int}(:kernel_nsub_p1 => 2, :kernel_T_p1 => 3,
            :kernel_nsub_p2 => 2, :kernel_T_p2 => 3))
    @test t_bound.n_obs == 12
    @test [kp.timepoints for kp in t_bound.kernel_plates] == [3, 3]
    # A non-conventional T key with N plates: the T-needing plate names
    # its conventional key.
    t_unbound = lower_rkppl(t_ast, (:t, :dose, :obs); conditioned = (:t, :dose, :obs))
    # refused: non-conventional T key, plate's `kernel_T_p1` unbound (wrong data)
    @test_throws "bind `kernel_T_p1`" bind_data(t_unbound, t_cols;
        dims = Dict{Symbol,Int}(:kernel_nsub_p1 => 2, :kernel_nsub_p2 => 2,
            :Tee => 3))
end
