# ReactiveHMC.jl-FAITHFUL `@kernel` NUTS AUTHORING FIXTURE.
#
# Re-authored (correct-then-rebase) directly against the ACTUAL ReactiveHMC.jl source — NOT the
# in-repo oracle. Reference provenance:
#   ReactiveHMC v0.1.0, installed at ~/.julia/packages/ReactiveHMC/781sB/src, pinned in
#   ReactiveKernels' src/hmc.jl to main@ca9ea4ca41924bb0e1fadc01c717e1333916aba6
#   (github.com/nsiccha/ReactiveHMC.jl/blob/ca9ea4ca.../src/nuts.jl). Files transcribed:
#   phasepoints.jl (euclidean_phasepoint), integrators.jl (leapfrog!), nuts.jl (nuts_state +
#   tree helpers), adaptation.jl (dual_averaging_state, welford_var). The ONLY change from the
#   reference is @reactive -> the unified sole @kernel with an explicit `self` first method
#   parameter; field NAMES, object COMPOSITION, the pot_f/grad_f phase-point shape, the direct
#   in-object state mutation, the DA (m/H/mu + fit!) and Welford (n/mean/var + step!(x;dn))
#   recurrences, and the restore!/rcopy! reset semantics are the reference's, verbatim.
#
# This REPLACES the earlier fixture, which diverged from the reference (invented value_gradient
# bundle, velocity/kinetic/hamiltonian names, an external _NUTSScratch argument + a no-self-mutation
# "segment/orchestration" contract, and warmup!/adapt_metric! that live OUTSIDE nuts_state in the
# reference). 5e8773b is retained only as algorithm/oracle history; it is NOT the production target.
#
# STAGE: SOURCE-SHAPE + MACRO-CONSTRUCTION only (construction-only substrate). If the graph
# analysis / MethodIR rejects any faithful shape here (in-object tree state mutation, fwd/bwd
# derived from gofwd, callable pot_f/grad_f, restore!/rcopy!), that is a COMPILER REQUIREMENT to
# lower the reference shape — NOT a fixture defect to be worked around. NO execution/parity/0-B/perf
# claim. One `@kernel` macro; no Graph/add!/applier/binding plumbing; no compiler-limitation
# compromises.
using ReactiveKernels
using LinearAlgebra, LogExpFunctions, Random

# ---- nuts.jl / adaptation.jl module helpers (verbatim from the reference) ---------------------------
fillf(f::Function, value, n::Int) = [f(value) for _ in 1:n]
finiteorneginf(x) = isfinite(x) ? x : typeof(x)(-Inf)
min1exp(x) = x >= 0 ? one(x) : exp(x)
badd(args...) = Base.broadcasted(+, args...)
randbernoullilog(rng, logprob) = logprob > 0 ? true : -randexp(rng) < logprob
logswapprob(tree) = tree.log_weight[1] - tree.log_weight[2]
compute_criterion(mom, bwd_dham_dmom, fwd_dham_dmom) =
    (dot(mom, bwd_dham_dmom) > 0 && dot(mom, fwd_dham_dmom) > 0)
smooth(prev, new, new_weight) = (1 - new_weight) * prev + new_weight * new

trajectory(d::Int) = trajectory(zeros(d), zeros(d))
trajectory(bwd, fwd) = (; bwd, fwd)
mv(mom, dham_dmom) = (; mom, dham_dmom)
mv(d::Int) = mv(zeros(d), zeros(d))
tree(d::Int) = (;
    log_weight = fill(-Inf, 2),
    bwd = mv(d),
    bwd_fwd = mv(d),
    summed_mom = trajectory(d),
)
tree(phasepoint) = tree(length(phasepoint.pos))

# ---- phasepoints.jl: euclidean_phasepoint (methodless => stateless; derived-only) ------------------
# grad_f(pos) returns (pot, dpot_dpos); pot_f(pos) the potential alone — the reference's TWO-function
# gradient shape (NOT a value_gradient bundle). Reactive fields: pot, dpot_dpos, chol_metric,
# dkin_dmom, kin, ham, dham_dpos, dham_dmom.
@kernel euclidean_phasepoint(pot_f, grad_f, metric, pos, mom) = begin
    pot = pot_f(pos)
    pot, dpot_dpos = grad_f(pos)

    chol_metric = cholesky(metric)
    dkin_dmom = chol_metric \ mom
    # (reference wraps the logdet term in @node as a caching hint; @kernel's analysis owns caching.)
    kin = .5 * (logdet(chol_metric) + dot(mom, dkin_dmom))

    ham = pot + kin
    dham_dpos = dpot_dpos
    dham_dmom = dkin_dmom
end

# ---- integrators.jl: leapfrog! is a FREE function taking the phase point; stepsize is a keyword ----
leapfrog!(phasepoint; stepsize) = begin
    @. phasepoint.mom -= .5 * stepsize * phasepoint.dham_dpos
    @. phasepoint.pos +=      stepsize * phasepoint.dham_dmom
    @. phasepoint.mom -= .5 * stepsize * phasepoint.dham_dpos
end

# ---- nuts.jl: nuts_state — ONE in-object multinomial NUTS sampler --------------------------------
# Tree scratch (gofwd/may_sample/may_continue/fwdbwd/fwd/bwd/trees/proposals/dham/diverged) are the
# object's OWN composed fields; fwd/bwd are DERIVED from gofwd; step!/flip!/finish_tree!/start_tree!
# are methods that mutate those fields DIRECTLY (self.gofwd = !self.gofwd; self.may_sample = false;
# self.dham = …). This is the reference's composition and mutation model — no external scratch, no
# segment/orchestration split. reset is restore!(self;force) + bwd.mom.*=-1 + rcopy!(init, proposals[end]).
@kernel nuts_state(init; rng, max_depth = 10, min_dham = -1000.,
                   step_f = nothing, stats_f = nothing) = begin
    gofwd = true
    may_sample = true
    may_continue = true
    fwdbwd = fillf(deepcopy, init, 2)
    fwd = fwdbwd[gofwd ? 1 : 2]                 # derived from gofwd (reference wraps in Ref)
    bwd = fwdbwd[gofwd ? 2 : 1]
    trees = fillf(tree, init, max_depth + 1)
    proposals = fillf(deepcopy, init, max_depth + 2)
    dham = 0.
    diverged = !(dham >= min_dham)

    stepfwd!(self) = self.step_f(self.fwd)
    collectstats!(self) = isnothing(self.stats_f) || self.stats_f(self)
    logadvanceprob(self, depth) =
        self.trees[depth - 1].log_weight[1] - self.trees[depth].log_weight[1]
    swapproposal!(self, i, j = length(self.proposals)) = begin
        self.proposals[i], self.proposals[j] = self.proposals[j], self.proposals[i]
    end
    step!(self; force = true) = begin
        restore!(self; force)
        self.bwd.mom .*= -1
        self.trees[1].log_weight[1] = 0.
        for depth in 1:self.max_depth
            rand(self.rng, Bool) && flip!(self, depth)
            finish_tree!(self, depth)
            self.may_sample || break
            randbernoullilog(self.rng, logswapprob(self.trees[depth])) && swapproposal!(self, depth)
            self.may_continue || break
        end
        rcopy!(self.init, self.proposals[end])
    end
    flip!(self, depth) = if depth > 1
        self.gofwd = !self.gofwd
        tree = self.trees[depth]
        @. tree.bwd.mom = -self.bwd.mom
        @. tree.bwd.dham_dmom = -self.bwd.dham_dmom
        @. tree.summed_mom.fwd *= -1
    end
    finish_tree!(self, depth) = begin
        tree = self.trees[depth]
        suptree = self.trees[depth + 1]
        tree.log_weight[2] = tree.log_weight[1]
        if depth == 1
            rcopy!(suptree.bwd, (; self.fwd.mom, self.fwd.dham_dmom))
        else
            rcopy!(suptree.bwd, tree.bwd)
            rcopy!(tree.bwd_fwd, (; self.fwd.mom, self.fwd.dham_dmom))
            tree.summed_mom.bwd .= tree.summed_mom.fwd
        end
        start_tree!(self, depth)
        self.may_continue || return self.may_sample = false
        suptree.log_weight[1] = logaddexp(tree.log_weight[1], tree.log_weight[2])
        self.may_continue = if depth == 1
            suptree.summed_mom.fwd .= suptree.bwd.mom .+ self.fwd.mom
            compute_criterion(suptree.summed_mom.fwd, suptree.bwd.dham_dmom, self.fwd.dham_dmom)
        else
            suptree.summed_mom.fwd .= tree.summed_mom.bwd .+ tree.summed_mom.fwd
            (
                compute_criterion(suptree.summed_mom.fwd, suptree.bwd.dham_dmom, self.fwd.dham_dmom) &&
                compute_criterion(badd(tree.summed_mom.bwd, tree.bwd.mom),
                                  suptree.bwd.dham_dmom, tree.bwd.dham_dmom) &&
                compute_criterion(badd(tree.bwd_fwd.mom, tree.summed_mom.fwd),
                                  tree.bwd_fwd.dham_dmom, self.fwd.dham_dmom)
            )
        end
    end
    start_tree!(self, depth) = if depth == 1
        stepfwd!(self)
        self.dham = finiteorneginf(self.init.ham - self.fwd.ham)
        collectstats!(self)
        self.diverged && return self.may_continue = false
        self.trees[1].log_weight[1] = self.dham
        rcopy!(self.proposals[1], self.fwd)
    else
        start_tree!(self, depth - 1)
        self.may_continue || return self.may_sample = false
        swapproposal!(self, depth - 1, depth)
        finish_tree!(self, depth - 1)
        if self.may_sample && randbernoullilog(self.rng, logadvanceprob(self, depth))
            swapproposal!(self, depth - 1, depth)
        end
    end
end

# ---- adaptation.jl: dual_averaging_state (m/H/mu + fit!(x)) ----------------------------------------
@kernel dual_averaging_state(init; target = .8, regularization_scale = .05,
                             relaxation_exponent = .75, offset = 10) = begin
    m = one(init)
    H = zero(init)
    mu = log(10) + log(init)
    log_current = mu - sqrt(m) / regularization_scale * H
    log_final = zero(init)
    current = exp(log_current)
    final = exp(log_final)
    fit!(self, x) = begin
        self.m += 1
        self.H += (self.target - x - self.H) / (self.m + self.offset)
        self.log_final += self.m^(-self.relaxation_exponent) * (self.log_current - self.log_final)
    end
end

# ---- adaptation.jl: welford_var (n/mean/var + step!(x; dn)) ----------------------------------------
@kernel welford_var(dim) = begin
    n = 0.
    mean = zeros(dim)
    var = zeros(dim)
    step!(self, x::AbstractVector; dn = 1.) = begin
        self.n += dn
        w = dn / self.n
        @. self.var = smooth(self.var, (x - smooth(self.mean, x, w)) * (x - self.mean), w)
        @. self.mean = smooth(self.mean, x, w)
    end
    # reference forwards `; kwargs...`; @kernel construction currently rejects a kwargs-splat in a
    # method signature (FLAGGED to poc as a required capability). `dn` is the ONLY keyword the vector
    # method accepts, so forwarding it explicitly is behaviorally identical to the reference splat.
    step!(self, x::AbstractMatrix; dn = 1.) = for xi in eachcol(x)
        step!(self, xi; dn = dn)
    end
end

# NOTE: trajectory_stats / sampling_stats (statistics.jl) are faithful recorders built on
# ElasticArrays, which is not a current ReactiveKernels dependency; they are deferred to a follow-up
# (adding the dep is a cross-cutting change) and are NOT part of the reference nuts_state surface —
# metric/step-size adaptation and stats live OUTSIDE nuts_state in ReactiveHMC (user composes
# welford_var + dual_averaging_state + Diagonal(max.(1e-6, wv.var))), so nuts_state stays the pure
# transition, exactly as the reference.

# ==== reference-surface source-shape gate (construction only; NO execution/parity/perf) =============
if abspath(PROGRAM_FILE) == @__FILE__
    RKS = ReactiveKernels
    spec_of(k) = k isa RKS.KernelSpec ? k : getfield(k, :spec)
    fields_of(k) = Set(spec_of(k).port_order)
    methods_of(k) = Set(m.name for m in RKS.kernel_methods(k))
    body_str(k) = join([string(m.body) for m in RKS.kernel_methods(k)], "\n")

    # ---- AST util: does a method body directly mutate a `self.<field>` (assignment or .= broadcast)?
    _walk(f, x) = (f(x); x isa Expr && foreach(a -> _walk(f, a), x.args); nothing)
    _is_self_lhs(x, self) =
        x isa Expr && ((x.head === :. && (x.args[1] === self || _is_self_lhs(x.args[1], self))) ||
                       (x.head === :ref && _is_self_lhs(x.args[1], self)))
    const _CMP = Set([:(==), :(!=), :(<=), :(>=), :(===), :(!==), :(.==), :(.!=), :(.<=), :(.>=)])
    _is_assign(h) = h isa Symbol && (s = String(h); endswith(s, "=") && !(h in _CMP))
    function self_mutates(method)
        found = Ref(false)
        _walk(method.body) do x
            x isa Expr && _is_assign(x.head) && length(x.args) >= 1 &&
                _is_self_lhs(x.args[1], method.self) && (found[] = true)
        end
        found[]
    end
    method_named(k, name) = (ms = filter(m -> m.name === name, RKS.kernel_methods(k)); ms)

    # (a) euclidean_phasepoint: methodless (stateless), REFERENCE derived-field names, pot_f/grad_f ---
    @assert euclidean_phasepoint isa RKS.KernelSpec "euclidean_phasepoint must be methodless (stateless)"
    pp = fields_of(euclidean_phasepoint)
    for f in (:pot_f, :grad_f, :metric, :pos, :mom,           # sources (pot_f AND grad_f — two funcs)
              :pot, :dpot_dpos, :chol_metric, :dkin_dmom, :kin, :ham, :dham_dpos, :dham_dmom)  # derived
        @assert f in pp "euclidean_phasepoint missing reference field $f"
    end
    @assert Set(spec_of(euclidean_phasepoint).want_names) == Set([:ham, :dham_dpos, :dham_dmom]) "phasepoint exposes reference outputs"
    # invented (non-reference) names must be ABSENT from the phase-point surface
    for bad in (:value_gradient, :velocity, :kinetic, :potential, :hamiltonian, :gradient, :chol)
        @assert !(bad in pp) "non-reference name $bad present in phase point — surface diverged"
    end

    # (b) leapfrog! is a free function on dham_dpos/dham_dmom with a `stepsize` keyword --------------
    lf = first(methods(leapfrog!))
    @assert :stepsize in Base.kwarg_decl(lf) "leapfrog! must take a stepsize keyword"
    lfsrc = read(@__FILE__, String)
    @assert occursin("phasepoint.dham_dpos", lfsrc) && occursin("phasepoint.dham_dmom", lfsrc) "leapfrog! must act on dham_dpos/dham_dmom"

    # (c) nuts_state: in-object composed tree fields + reference method inventory --------------------
    ns = fields_of(nuts_state)
    for f in (:init, :rng, :max_depth, :min_dham, :step_f, :stats_f,          # sources
              :gofwd, :may_sample, :may_continue, :fwdbwd, :fwd, :bwd,        # composed control/endpoints
              :trees, :proposals, :dham, :diverged)                          # composed scratch + diagnostics
        @assert f in ns "nuts_state missing reference field $f — tree state must be composed IN the object"
    end
    @assert methods_of(nuts_state) == Set([:stepfwd!, :collectstats!, :logadvanceprob, :swapproposal!,
                                           :step!, :flip!, :finish_tree!, :start_tree!]) "nuts_state method inventory: $(methods_of(nuts_state))"

    # (d) DIRECT in-object mutation (the reference model) — these methods MUST mutate self fields ----
    @assert self_mutates(only(method_named(nuts_state, :flip!))) "flip! must mutate self (gofwd/tree) directly"
    @assert self_mutates(only(method_named(nuts_state, :finish_tree!))) "finish_tree! must mutate self.may_sample/may_continue directly"
    @assert self_mutates(only(method_named(nuts_state, :start_tree!))) "start_tree! must mutate self.dham/may_* directly"
    @assert self_mutates(only(method_named(nuts_state, :step!))) "step! must mutate self.trees directly"
    nb = body_str(nuts_state)
    @assert occursin("self.gofwd = !", nb) "flip! must toggle self.gofwd"
    @assert occursin("self.may_sample = false", nb) "must set self.may_sample=false on stop"
    @assert occursin("self.dham = finiteorneginf", nb) "start_tree! must set self.dham = finiteorneginf(init.ham - fwd.ham)"

    # (e) reference multinomial machinery present (log weights, tournament, RNG, criterion, reset) ---
    for tok in ("log_weight", "swapproposal!", "randbernoullilog", "logswapprob", "logadvanceprob",
                "compute_criterion", "logaddexp", "summed_mom", "bwd_fwd",
                "restore!", "rcopy!", "proposals[end]")
        @assert occursin(tok, nb) "nuts_state must use reference construct `$tok`"
    end

    # (f) dual_averaging_state: reference m/H/mu accumulators + fit! -------------------------------
    da = fields_of(dual_averaging_state)
    for f in (:m, :H, :mu, :log_current, :log_final, :current, :final)
        @assert f in da "dual_averaging_state missing reference field $f"
    end
    @assert methods_of(dual_averaging_state) == Set([:fit!]) "DA inventory"
    db = body_str(dual_averaging_state)
    @assert occursin("self.m += 1", db) && occursin("self.H +=", db) && occursin("self.log_final +=", db) "DA fit! reference recurrence (m/H/log_final)"
    for bad in (:iteration, :error, :center, :regularization)   # earlier-invented DA names must be gone
        @assert !(bad in da) "non-reference DA name $bad present"
    end

    # (g) welford_var: reference n/mean/var + step!(x;dn) (two dispatch methods) --------------------
    wf = fields_of(welford_var)
    for f in (:n, :mean, :var)
        @assert f in wf "welford_var missing reference field $f"
    end
    @assert methods_of(welford_var) == Set([:step!]) "welford inventory (step!)"
    @assert length(method_named(welford_var, :step!)) == 2 "welford must have vector + matrix step! methods"
    wb = body_str(welford_var)
    @assert occursin("self.n += dn", wb) && occursin("smooth(", wb) "welford step! reference recurrence"

    # (h) sole @kernel; no @reactive / no Graph/add!/applier/binding plumbing in any authored body --
    allbodies = join([body_str(k) for k in (nuts_state, dual_averaging_state, welford_var)], "\n") *
                "\n" * string(spec_of(euclidean_phasepoint).port_order)
    for bad in ("@reactive", "Graph(", "add!(", "output_binding", "_RecipeApplier", "compile_update",
                "bind_schedule", "_NUTSScratch", "reset_scratch!", "mirror!", "recompute_init!",
                "value_gradient", "adapt_metric!", "warmup!")
        @assert !occursin(bad, allbodies) "plumbing/non-reference/invented token present: $bad"
    end

    println("Reference-surface source-shape gate PASS: transcribed from ACTUAL ReactiveHMC.jl v0.1.0")
    println("(781sB @ ca9ea4ca). euclidean_phasepoint (methodless, pot_f/grad_f, reference derived")
    println("names) + free leapfrog!(phasepoint;stepsize) + nuts_state (in-object composed tree state,")
    println("fwd/bwd derived from gofwd, DIRECT self mutation in step!/flip!/finish_tree!/start_tree!,")
    println("restore!/rcopy! reset, multinomial tournament/log-weights/RNG/criterion) + DA (m/H/mu +")
    println("fit!) + welford (n/mean/var + step!(x;dn), 2 methods). Sole @kernel; no plumbing; invented")
    println("names (value_gradient/velocity/kinetic/hamiltonian/external-scratch/mirror!/warmup!) ABSENT.")
    println("Construction-only substrate — NO execution/parity/0-B/perf claim; MethodIR lowering of the")
    println("faithful shape is a compiler requirement, not a fixture defect.")
end
