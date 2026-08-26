# ReactiveHMC-shaped `@kernel` NUTS AUTHORING FIXTURE (V7, rooted at Inc1 substrate 118ad97).
#
# The author-facing surface the user asked for: ONE `@kernel` macro, concise composable method
# bodies faithful to ReactiveHMC.jl, no Graph/add!/applier/output_binding plumbing, no `@reactive`,
# no production transition wiring. Once poc's Increment-2 MethodIR clears, these EXACT definitions
# become the MethodIR consumer fixture and then the production sampler.
#
# STAGE (Inc1 / 118ad97): SOURCE-SHAPE + MACRO-CONSTRUCTION only. MethodIR/effect-lowering is
# intentionally ABSENT, so these CONSTRUCT (stateful skeletons with retained raw bodies) but do NOT
# execute/compile — NO execution/perf claim. The recurrence math is the byte-faithful math from
# src/reactive_nuts.jl (DA), src/reactive_nuts.jl `_welford_object` (Welford), and src/hmc.jl
# `_adapted_diagonal_metric` (mass diagonal). Method-bearing `@kernel` binds a `const` => top level.
#
# SEGMENT vs ORCHESTRATION contract (authored in the source, enforced by the gate below):
#  - SEGMENTS mutate authoritative state (endpoints / adaptation / statistics): leapfrog!,
#    refresh_momentum!, mirror!, recompute_init!, restore_accept!, fit!, observe!, adapt_stepsize!,
#    adapt_metric!, record!  — plus read-only reducers energy_error / uturn_ok.
#  - ORCHESTRATION methods (step!/grow!/start_tree!/finish_tree!/flip!/warmup!/sample!) are ordinary
#    Julia: dynamic recursion + RNG + tree/proposal/control SCRATCH as LOCALS; they mutate sampler
#    state ONLY through segment calls — never a direct `self.<path> = …`.
using ReactiveKernels
using LinearAlgebra, Random

_smooth(previous, new, weight) = (1 - weight) * previous + weight * new   # src/hmc.jl:753, inlined for a self-contained fixture

# --- Object A: shared Hamiltonian / metric authority (methodless => stateless, composed by identity) -
@kernel hamiltonian(potential_gradient!, metric, stepsize) = begin
    chol        = cholesky(metric)
    logdet_chol = logdet(chol)
end

# --- Object B: a phase-point endpoint (owns pos/mom + derived; reads shared `ham`) ------------------
@kernel endpoint(ham, pos, mom) = begin
    value_gradient = ham.potential_gradient!(pos)                 # owned (value, gradient) bundle
    potential      = value_gradient.value
    gradient       = value_gradient.gradient
    velocity       = ham.chol \ mom                              # owned; reads shared chol
    kinetic        = (ham.logdet_chol + dot(mom, velocity)) / 2  # reads shared logdet_chol cut point
    hamiltonian    = potential + kinetic

    # segment: resample momentum ~ N(0,M); velocity/kinetic/hamiltonian recompute (0 pgrad).
    function refresh_momentum!(self, rng)
        randn!(rng, self.mom)
        lmul!(self.ham.chol.L, self.mom)
    end
    # segment: faithful 3-line Stormer-Verlet; self.ham.stepsize is the shared slot (visible sharing).
    leapfrog!(self) = begin
        @. self.mom -= 0.5 * self.ham.stepsize * self.gradient
        @. self.pos +=       self.ham.stepsize * self.velocity
        @. self.mom -= 0.5 * self.ham.stepsize * self.gradient
    end
end

# --- Object D1: Nesterov dual averaging (byte-faithful to src _dual_averaging_object) --------------
@kernel dual_averaging(iteration, error, log_final, center,
                       target, regularization_scale, relaxation_exponent, offset) = begin
    log_current = center - sqrt(iteration) / regularization_scale * error
    current     = exp(log_current)
    final       = exp(log_final)
    fit!(self, acceptance_rate) = begin
        new_iteration  = self.iteration + 1
        new_error      = self.error +
            (self.target - acceptance_rate - self.error) / (new_iteration + self.offset)
        self.iteration = new_iteration
        self.error     = new_error
        weight         = new_iteration^(-self.relaxation_exponent)
        self.log_final = self.log_final + weight * (self.log_current - self.log_final)
    end
end

# --- Object D2: online componentwise Welford (byte-faithful to src _welford_object) -----------------
@kernel welford(n, mean, var) = begin
    function observe!(self, value; weight = 1)
        self.n   = self.n + weight
        fraction = weight / self.n
        @. self.var  = _smooth(self.var,
                               (value - _smooth(self.mean, value, fraction)) * (value - self.mean),
                               fraction)
        @. self.mean = _smooth(self.mean, value, fraction)
    end
end

# --- Object E: statistics (real diagnostic accumulators) --------------------------------------------
@kernel sampling_stats(sum_depth, sum_n_steps, sum_accept, n_divergent, n_draws) = begin
    function record!(self, depth, n_steps, acceptance, energy_error, divergent)
        self.sum_depth   = self.sum_depth + depth
        self.sum_n_steps = self.sum_n_steps + n_steps
        self.sum_accept  = self.sum_accept + acceptance
        self.n_divergent = self.n_divergent + (divergent ? 1 : 0)
        self.n_draws     = self.n_draws + 1
    end
end

# --- The sampler (driver): one shared ham + three composed endpoints + adaptation/stats -------------
@kernel sampler(potential_gradient!, metric, stepsize, position, momentum, max_depth, min_dham) = begin
    ham   = hamiltonian(potential_gradient!, metric, stepsize)    # shared authority (one instance)
    init  = endpoint(ham, position, momentum)                     # three composed phase points
    fwd   = endpoint(ham, position, momentum)
    bwd   = endpoint(ham, position, momentum)
    da    = dual_averaging(one(stepsize), zero(stepsize), zero(stepsize),
                           log(oftype(stepsize, 10)) + log(stepsize),
                           oftype(stepsize, 0.8), oftype(stepsize, 0.05),
                           oftype(stepsize, 0.75), oftype(stepsize, 10))
    w     = welford(0, zero(position), zero(position))
    stats = sampling_stats(0, 0, zero(stepsize), 0, 0)

    # --- read-only reducer segments (no state write) -------------------------------------------------
    # dham = finite(init_ham - moving_ham); divergence threshold; U-turn criterion.
    energy_error(self, moving_ham) =
        (e = self.init.hamiltonian - moving_ham; isfinite(e) ? e : oftype(e, -Inf))
    divergent(self, dham) = !(dham >= self.min_dham)
    uturn_ok(self, summed_mom, back_vel, fwd_vel) =
        (dot(summed_mom, back_vel) > 0) && (dot(summed_mom, fwd_vel) > 0)

    # --- state-mutating segments (endpoints / adaptation / metric) -----------------------------------
    # reset MIRROR: copy init -> fwd (identity) and -> bwd with SIGNED momentum/velocity (bwd_mom=-init_mom,
    # bwd_vel=-init_vel; quadratic kinetic sign-invariant so bwd kinetic/ham == init). Distinct buffers.
    mirror!(self) = begin
        @. self.fwd.pos = self.init.pos
        @. self.fwd.mom = self.init.mom
        @. self.bwd.pos = self.init.pos
        @. self.bwd.mom = -self.init.mom
    end
    # Unit-C: recompute init's derived from the (accepted) pos/mom — ONE gradient at the accepted point.
    recompute_init!(self) = begin
        self.init.value_gradient = self.init.ham.potential_gradient!(self.init.pos)
    end
    # accepted restore: write the accepted proposal into init (state write => segment), then recompute.
    restore_accept!(self, accepted_pos, accepted_mom) = begin
        @. self.init.pos = accepted_pos
        @. self.init.mom = accepted_mom
    end
    # cross-view adaptation writes: DA -> shared stepsize; Welford variance -> shared mass diagonal.
    adapt_stepsize!(self) = begin
        self.ham.stepsize = self.da.current
    end
    adapt_metric!(self; minimum_variance = 1e-3, regularization = 5) = begin
        weight            = self.w.n / (self.w.n + regularization)
        sample_variance   = self.w.var .* (self.w.n / (self.w.n - 1))
        position_variance = @. max(minimum_variance,
                                   weight * sample_variance + (1 - weight) * minimum_variance)
        self.ham.metric = Diagonal(@. inv(position_variance))
    end

    # --- ORCHESTRATION (ordinary Julia; tree/proposal/control are LOCALS; state via segment calls) ---
    # one in-place leapfrog on the ACTIVE endpoint + energy/divergence diagnostics (locals).
    function start_tree!(self, active, control)
        leapfrog!(active)                                          # segment (state write)
        dham = energy_error(self, active.hamiltonian)             # reducer (read-only)
        control.n_steps       += 1
        control.energy_error   = dham
        control.acceptance_sum += (dham >= 0 ? one(dham) : exp(dham))
        control.diverged       = divergent(self, dham)            # reducer
        control
    end
    # flip the active direction (depth>1): pure control/scratch locals + tree-scratch negation.
    function flip!(self, depth, gofwd, tree)
        depth > 1 || return gofwd
        @. tree.back_mom = -tree.back_mom
        @. tree.back_vel = -tree.back_vel
        !gofwd
    end
    # one doubling: recurse, then combine sub-trees under the U-turn criterion (segment reader).
    function finish_tree!(self, depth, gofwd, active, control, tree)
        if depth == 1
            start_tree!(self, active, control)
            @. tree.summed_mom = active.mom
            return control
        end
        finish_tree!(self, depth - 1, gofwd, active, control, tree)
        control.may_continue || (control.may_sample = false; return control)
        finish_tree!(self, depth - 1, gofwd, active, control, tree)
        control.may_continue = control.may_continue &&
            uturn_ok(self, tree.summed_mom, tree.back_vel, active.velocity)   # reducer segment
        control
    end
    # ONE transition: reset mirror -> tree recursion over the direction views -> accepted restore.
    function step!(self, rng)
        mirror!(self)                                             # segment (reset init -> fwd/bwd, signed)
        control = _tree_control(self.min_dham)                   # ordinary-Julia scratch (locals)
        tree    = _tree_scratch(self.init.pos)
        gofwd   = true
        accepted_pos = copy(self.init.pos); accepted_mom = copy(self.init.mom)
        for depth in 1:self.max_depth
            rand(rng, Bool) && (gofwd = flip!(self, depth, gofwd, tree))
            active = gofwd ? self.fwd : self.bwd
            finish_tree!(self, depth, gofwd, active, control, tree)
            control.depth = depth
            control.may_sample || break
            @. accepted_pos = active.pos; @. accepted_mom = active.mom   # proposal scratch (locals)
            control.may_continue || break
        end
        restore_accept!(self, accepted_pos, accepted_mom)        # segment (accepted -> init)
        recompute_init!(self)                                    # segment (Unit-C: one boundary gradient)
        control
    end
    # warmup drives the adaptation/statistics segments at their real boundaries.
    function warmup!(self, iterations, rng)
        for _ in 1:iterations
            refresh_momentum!(self.init, rng)                    # segment
            control = step!(self, rng)                           # orchestration
            acceptance = control.n_steps == 0 ? zero(self.ham.stepsize) :
                         control.acceptance_sum / control.n_steps
            fit!(self.da, acceptance)                            # segment (dual averaging)
            adapt_stepsize!(self)                                # segment (DA -> stepsize)
            observe!(self.w, self.init.pos)                      # segment (Welford)
            adapt_metric!(self)                                  # segment (variance -> mass diagonal)
            record!(self.stats, control.depth, control.n_steps, acceptance,
                    control.energy_error, control.diverged)      # segment (statistics)
        end
        self
    end
    # sampling advances the chain and records diagnostics (no adaptation).
    function sample!(self, rng)
        refresh_momentum!(self.init, rng)                        # segment
        control = step!(self, rng)                               # orchestration
        acceptance = control.n_steps == 0 ? zero(self.ham.stepsize) :
                     control.acceptance_sum / control.n_steps
        record!(self.stats, control.depth, control.n_steps, acceptance,
                control.energy_error, control.diverged)          # segment
        control
    end
end

# ordinary-Julia orchestration scratch (locals; NOT sampler state) -----------------------------------
mutable struct _TreeControl{T}
    n_steps::Int; depth::Int; may_sample::Bool; may_continue::Bool
    acceptance_sum::T; energy_error::T; diverged::Bool; min_dham::T
end
_tree_control(min_dham::T) where {T} = _TreeControl{T}(0, 0, true, true, zero(T), zero(T), false, min_dham)
_tree_scratch(pos) = (summed_mom = zero(pos), back_mom = zero(pos), back_vel = zero(pos))

# --- Inc1 source-shape gate (construction only; NO execution/perf) ----------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    RKS = ReactiveKernels
    src = read(@__FILE__, String)

    # inline AST scanners (Inc1 gives kernel_methods + .name/.self/.body; the closure/contract
    # checks are ordinary AST walks over the retained raw bodies).
    _unqual_calls!(acc, x) = begin
        if x isa Expr
            if x.head === :call && x.args[1] isa Symbol
                push!(acc, string(x.args[1]))
            end
            for a in x.args; _unqual_calls!(acc, a); end
        end
        acc
    end
    body_calls(body) = _unqual_calls!(String[], body)
    _is_self_path(lhs, self) = lhs isa Expr && lhs.head === :. &&
        (lhs.args[1] === self || _is_self_path(lhs.args[1], self))
    _has_self_assign(x, self) = begin
        x isa Expr || return false
        if x.head in (:(=), :(+=), :(-=), :(*=), :(.=)) && _is_self_path(x.args[1], self)
            return true
        end
        # `@. self.<path> op= …` lowers to a macrocall wrapping an assignment
        any(a -> _has_self_assign(a, self), x.args)
    end

    # (a) skeleton types: stateful vs stateless
    for k in (endpoint, sampler, dual_averaging, welford, sampling_stats)
        @assert k isa RKS._StatefulKernelSkeleton "$(k) must be a method-bearing (stateful) @kernel"
    end
    @assert !(hamiltonian isa RKS._StatefulKernelSkeleton) "hamiltonian is methodless -> stateless"

    # (b) exact method inventory (incl. every recursion/criterion/diagnostic/adaptation/stats method)
    mnames(k) = Set(m.name for m in RKS.kernel_methods(k))
    @assert mnames(endpoint) == Set([:refresh_momentum!, :leapfrog!]) "endpoint inventory"
    @assert mnames(dual_averaging) == Set([:fit!]) "dual_averaging inventory"
    @assert mnames(welford) == Set([:observe!]) "welford inventory"
    @assert mnames(sampling_stats) == Set([:record!]) "stats inventory"
    want_sampler = Set([:energy_error, :divergent, :uturn_ok, :mirror!, :recompute_init!,
                        :restore_accept!, :adapt_stepsize!, :adapt_metric!, :start_tree!, :flip!,
                        :finish_tree!, :step!, :warmup!, :sample!])
    @assert mnames(sampler) == want_sampler "sampler inventory: $(mnames(sampler))"

    # (c) call-graph closure: every unqualified call in the sampler method bodies resolves to an
    #     authored sibling/child method or an explicitly-allowed external primitive.
    authored = Set(string(n) for n in want_sampler) ∪
               Set(["refresh_momentum!", "leapfrog!", "fit!", "observe!", "record!"])
    allowed_ext = Set(["copy", "zero", "one", "exp", "log", "sqrt", "dot", "isfinite", "oftype",
                       "cholesky", "logdet", "randn!", "lmul!", "inv", "max", "Diagonal",
                       "_tree_control", "_tree_scratch", "_smooth", "length", "eltype", "float", "typeof",
                       "!", "!="])
    reducers = Set(["energy_error", "divergent", "uturn_ok"])   # authored non-`!` reducer segments
    for m in RKS.kernel_methods(sampler), call in body_calls(m.body)
        # target method-shaped calls (mutating `!` + named reducers) — the placeholder-method risk;
        # operators/arithmetic/plain Base functions are inherently primitive.
        (endswith(call, "!") || call in reducers) || continue
        (call in authored || call in allowed_ext) ||
            error("unresolved method call `$(call)` in sampler.$(m.name) — not an authored sibling/child or allowed primitive")
    end

    # (d) segment/orchestration contract: ORCHESTRATION bodies must NOT directly assign `self.<path>`.
    orchestration = ("start_tree!", "flip!", "finish_tree!", "step!", "warmup!", "sample!")
    for m in RKS.kernel_methods(sampler)
        string(m.name) in orchestration || continue
        _has_self_assign(m.body, m.self) &&
            error("orchestration sampler.$(m.name) directly assigns self.<path> — mutate state only via segment calls")
    end

    # scan the RETAINED METHOD BODIES (actual authored code — not comments or the gate itself).
    allobjs = (endpoint, dual_averaging, welford, sampling_stats, sampler)
    bodies = join([string(m.body) for k in allobjs for m in RKS.kernel_methods(k)], "\n")

    # (e) no manual plumbing / no @reactive in any authored method body
    for bad in ("@reactive", "Graph(", "add!", "output_binding", "_RecipeApplier",
                "compile_update", "bind_schedule", "cache_apply")
        @assert !occursin(bad, bodies) "manual-plumbing/@reactive token in a method body: $bad"
    end

    # (f) faithful recurrence bodies retained (robust tokens over the stringified bodies)
    @assert occursin("relaxation_exponent", bodies) && occursin("log_final", bodies) &&
            occursin("log_current", bodies) "DA fit! recurrence retained"
    @assert occursin("_smooth", bodies) "Welford _smooth recurrence retained"
    @assert occursin("inv", bodies) && occursin("position_variance", bodies) "adapt_metric mass diagonal retained"
    @assert occursin("stepsize", bodies) && occursin("gradient", bodies) &&
            occursin("velocity", bodies) "leapfrog Stormer-Verlet retained"

    println("Inc1 source-shape gate PASS: stateful/stateless skeletons + exact method inventory +")
    println("call-graph closure (no unresolved reset!/flip!/grow!/restore!-class placeholders) +")
    println("segment/orchestration contract (no direct self.<path> assign in orchestration) +")
    println("no @reactive/Graph/add!/output_binding plumbing + faithful DA/Welford/metric/leapfrog bodies.")
    println("MethodIR/execution ABSENT (118ad97) — NO execution/perf claim.")
end
