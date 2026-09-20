using DifferentiationInterface
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
    unbound = lower_rkppl(_pk1cmt_ast(), data_names)
    @test isempty(unbound.responses)
    kp0 = only(unbound.kernel_plates)
    @test kp0.result === :pred
    @test kp0.subjects === :kernel_nsub_pred
    @test kp0.timepoints === nothing
    @test [p for (_, p, _) in kp0.slices] == [:ts, :d, :yy, :CLi, :Vci, :Kai]
    @test all(s -> s[3] === :unknown, kp0.slices)
    # The emitter passes scalar-context user code verbatim (undotted).
    @test kp0.assignments[1] == (:ke => :(CLi / Vci))
    @test kp0.obs.response === :yy
    @test kp0.obs.family === GaussianFam
    @test kp0.obs.location === :mu
    @test kp0.obs.scale === :sigma
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
    unbound = lower_rkppl(_doseplate_ast(), data_names)
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
    unbound = lower_rkppl(ast, (:dose, :obs, :t))
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
    unbound = lower_rkppl(ast, (:dose, :obs, :t))
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
    @test_throws "carries a scalar `~`" lower_rkppl(
        plate_ast(good_cell, [subj]; tilde = :.~), data)
    @test_throws "scalar `~` over vectors" lower_rkppl(
        plate_ast([good_cell[1], :(yy ~ Normal(mu, sigma)), :mu], [subj]), data)
    @test_throws "more than one `.~`" lower_rkppl(
        plate_ast([good_cell[1], good_cell[2], good_cell[2], :mu], [subj]), data)
    @test_throws "no `.~` observation" lower_rkppl(
        plate_ast([good_cell[1], :mu], [subj]), data)
    @test_throws "collected result name" lower_rkppl(
        plate_ast([good_cell[1], good_cell[2]], [subj]), data)
    @test_throws "only the trailing statement" lower_rkppl(
        plate_ast([:mu, good_cell[1], good_cell[2], :mu], [subj]), data)
    @test_throws "Gaussian in-cell observation only" lower_rkppl(
        plate_ast([good_cell[1], :(yy .~ Poisson.(mu)), :mu], [subj]), data)
    @test_throws "obs broadcasts" lower_rkppl(
        plate_ast([good_cell[1], :(yy .~ Normal(mu, sigma)), :mu], [subj]), data)
    @test_throws "exactly two arguments" lower_rkppl(
        plate_ast([good_cell[1], :(yy .~ Normal.(mu)), :mu], [subj]), data)
    @test_throws "name or a numeric literal" lower_rkppl(
        plate_ast([good_cell[1], :(yy .~ Normal.(mu .+ 1.0, sigma)), :mu], [subj]), data)
    @test_throws "not a slice param" lower_rkppl(
        plate_ast([good_cell[1], :(zz .~ Normal.(mu, sigma)), :mu], [subj]), data)
    @test_throws "needs `subjects=N`" lower_rkppl(
        plate_ast(good_cell, []), data)
    @test_throws "must be positive" lower_rkppl(
        plate_ast(good_cell, [Expr(:kw, :subjects, 0)]), data)
    @test_throws "integer literal or a dims-key" lower_rkppl(
        plate_ast(good_cell, [Expr(:kw, :subjects, :(1 + 1))]), data)
    @test_throws "exactly one keyword" lower_rkppl(
        plate_ast(good_cell, [subj, Expr(:kw, :group, :g)]), data)
    @test_throws "not bound data" lower_rkppl(
        plate_ast(good_cell, [subj]; cols = [:t, :dose, :zz]), data)
    @test_throws "one param per column" lower_rkppl(
        plate_ast(good_cell, [subj]; params = [:ts, :d]), data)
    @test_throws "not a cell name" lower_rkppl(
        plate_ast([good_cell[1], good_cell[2], :zz], [subj]), data)
    # Plate alongside a top-level response: the plate no longer carries
    # the only likelihood (hand-built plan — the surface would trip on
    # the response's predictor first).
    resp = LikelihoodSpec(GaussianFam, IdentityLink, :y, :mu, :s, nothing,
        ResponseEvidence(:none, nothing, nothing), :y_resp)
    pred_spec = PredictorSpec(:mu, IdentityLink,
        TermSpec[TermSpec(InterceptTerm, ColumnRef[], NamedTuple(),
            :Intercept, :intercept)], :mu)
    sparam = SampledParameter(:s, :exponential, (arg1 = 1.0,), nothing, :s)
    kp_hand = KernelPlate(:pred, 1, nothing, [(:t, :ts, :scalar)],
        Pair{Symbol,Any}[],
        (response = :ts, family = GaussianFam, location = :ts, scale = :s),
        :ts, :pred)
    both_plan = StructuralPlan([resp], [pred_spec],
        PopulationPrior[PopulationPrior(:mu, :Intercept, 0.0, 1.0)],
        [sparam], AssignmentSpec[], Dict{Symbol,AbstractVector}(), 0;
        kernel_plates = [kp_hand])
    @test_throws "only likelihood" validate_structure(both_plan)
    # Two plates in one model (distinct cell names — shared names would
    # trip single-assignment first).
    cell2 = [
        :(mu2 = (b0 .* d2) .* ts2),
        :(yy2 .~ Normal.(mu2, sigma)),
        :mu2,
    ]
    two = Expr(:block,
        plate_ast(good_cell, [subj]).args...,
        plate_ast(cell2, [subj]; lhs = :pred2,
            params = [:ts2, :d2, :yy2]).args[3])
    @test_throws "at most one kernel plate" lower_rkppl(two, data)
    # Responseless GLM plans (no plate) still fail as before.
    nolhs = Expr(:block, Expr(:call, :~, :sigma, Expr(:call, :Exponential, 1.0)))
    @test_throws "plan has no responses" lower_rkppl(nolhs, (:y,))
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
    # Series reductions are cross-timepoint (P3).
    @test_throws "does not lower in a cell" lower_rkppl(
        plate_ast([:(m = mean(ts)), mu_stmt, obs_stmt, :mu]), data)
    # Whole-column GP functions are whole-model constructs.
    @test_throws "does not lower in a cell" lower_rkppl(
        plate_ast([:(m = gp_exp_quad_cov(ts)), mu_stmt, obs_stmt, :mu]), data)
    # Cross-cell refs fail closed.
    @test_throws "unknown name" lower_rkppl(
        plate_ast([:(mu = (b0 .* d) .* zz), obs_stmt, :mu]), data)
    # Undotted over 2+ genuinely-vector operands fails at bind (kinds
    # resolve there): ts and yy are both vector slices here.
    two_vec_unbound = lower_rkppl(
        plate_ast([:(mu = ts * yy), obs_stmt, :mu]), data)
    @test_throws "write the dotted form" bind_data(two_vec_unbound, cols6; dims)
    # Scalar-context undotted canonicalizes instead (Ex1 shape).
    ok = bind_data(lower_rkppl(
        plate_ast([:(ke = 2.0 * 3.0), mu_stmt, obs_stmt, :mu]), data), cols6; dims)
    @test only(ok.kernel_plates).assignments[1] == (:ke => :(2.0 * 3.0))
    # Unbound subjects dims key.
    unbound = lower_rkppl(plate_ast([mu_stmt, obs_stmt, :mu]), data)
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
    @test_throws "not consumed" bind_data(unbound_exp, cols6;
        dims = Dict{Symbol,Int}(:kernel_nsub_pred => 2, :kernel_T_pred => 3,
            :kernel_T_preed => 3))
    # Ambiguous T keys fail closed.
    @test_throws "ambiguous timepoints" bind_data(unbound, cols6;
        dims = Dict{Symbol,Int}(:kernel_nsub_pred => 2, :kernel_T_pred => 3,
            :kernel_T_pred2 => 3))
    # T bound but unused (T > 1, all scalar).
    scalar_cols = Dict{Symbol,AbstractVector}(
        :t => [1.0, 2.0], :dose => [10.0, 20.0], :obs => [0.1, 0.4])
    @test_throws "no vector slice uses it" bind_data(unbound, scalar_cols; dims)
    # Vector-shaped column without a T key.
    @test_throws "no T dims key is bound" bind_data(unbound, cols6;
        dims = Dict{Symbol,Int}(:kernel_nsub_pred => 2))
    # Lengths matching neither n_sub nor n_sub * T.
    bad_cols = Dict{Symbol,AbstractVector}(
        :t => [0.0, 1.0, 2.0, 0.0], :dose => [10.0, 20.0],
        :obs => [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    @test_throws "neither n_sub" bind_data(unbound, bad_cols; dims)
    # Nonpositive dims values.
    @test_throws "positive integer" bind_data(unbound, cols6;
        dims = Dict{Symbol,Int}(:kernel_nsub_pred => 0, :kernel_T_pred => 3))
    # Caller collision with a materialized expansion name.
    collide = merge(cols6, Dict{Symbol,AbstractVector}(
        :pred_kexp_dose => [10.0, 10.0, 10.0, 20.0, 20.0, 20.0]))
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
    lit = bind_data(lower_rkppl(lit_ast, data), cols6;
        dims = Dict{Symbol,Int}(:kernel_T_pred => 3))
    kp_lit = only(lit.kernel_plates)
    @test kp_lit.subjects == 2
    @test kp_lit.timepoints == 3
end

