# ReactiveHMC-shaped `@kernel` NUTS AUTHORING FIXTURE (V7, rooted at Inc1 substrate 118ad97).
#
# The author-facing surface the user asked for: ONE `@kernel` macro, concise composable method
# bodies faithful to ReactiveHMC.jl, no Graph/add!/applier/output_binding plumbing, no `@reactive`,
# no production transition wiring. Once poc's Increment-2 MethodIR clears, these EXACT definitions
# become the MethodIR consumer fixture and then the production sampler.
#
# STAGE (Inc1 / 118ad97): SOURCE-SHAPE + MACRO-CONSTRUCTION only. MethodIR/effect-lowering is
# intentionally ABSENT, so these CONSTRUCT (stateful skeletons with retained raw bodies) but do NOT
# execute/compile — NO execution/perf claim. The multinomial NUTS transition (mirror / start_tree! /
# finish_tree! / flip! / step!) is a FAITHFUL 1:1 transcription of the ordinary-Julia oracle in
# src/hmc.jl (`_reset_transition!`/`_start_tree!`/`_finish_tree!`/`_flip!`/`step!`, which is itself
# ported byte-for-byte from ReactiveHMC.jl @ca9ea4ca) — same log-weights, proposal tournament + swap,
# `_rand_bernoulli_log` multinomial selection, accumulated summed-momenta, backward/forward velocities,
# divergence stop, depth>1 three-part U-turn criterion, and EXACT RNG call order (direction coin ->
# subtree proposal draws -> top-level proposal draw). Warmup boundaries (find_initial_stepsize! ->
# expanding metric windows -> window-end mass update -> DA+Welford reset per window -> final da.final)
# are transcribed from src/hmc.jl `warmup!`/`_warmup_window_ends`/`_adapted_diagonal_metric`. DA/Welford
# recurrences are byte-faithful to src/reactive_nuts.jl (`_dual_averaging_object`/`_welford_object`).
#
# SCRATCH OWNERSHIP (parent binding choice, 18:21): the hot transition takes an EXPLICIT PREALLOCATED
# orchestration-scratch argument built ONCE outside the transition (`_nuts_scratch`) and passed to
# step!/warmup!/sample!. Automatic scratch hoisting is not implemented/specified, so honest ownership
# is shown now; the later specializing factory may own/inject the same scratch without changing the
# transition math. NO tree/proposal/control buffer is constructed inside hot orchestration; scratch is
# mutated by ordinary orchestration (fill!/copyto!/index writes), NEVER as reactive self-path state.
#
# SEGMENT vs ORCHESTRATION contract (authored in the source, enforced by the gate below):
#  - SEGMENTS mutate authoritative reactive state (endpoints / adaptation / statistics): leapfrog!,
#    refresh_momentum!, mirror!, recompute_init!, restore_accept!, set_stepsize!, adapt_metric!, fit!,
#    reset! (DA+Welford), observe!, record!  — plus read-only reducers energy_error/divergent/
#    uturn_ok/uturn_ok_sum.
#  - ORCHESTRATION methods (reset_scratch!/start_tree!/finish_tree!/flip!/probe_acceptance!/
#    find_initial_stepsize!/step!/warmup!/sample!) are ordinary Julia: dynamic recursion + RNG +
#    tree/proposal/control SCRATCH; they mutate authoritative state ONLY through segment calls —
#    never a direct `self.<path> = …` (in any assignment/broadcast/mutating-call form).
using ReactiveKernels
using LinearAlgebra, Random

# --- module-level plain helpers (src/hmc.jl-faithful; allowed non-self primitives) ------------------
_smooth(previous, new, weight) = (1 - weight) * previous + weight * new         # src/hmc.jl:753
_min1exp(x) = x >= 0 ? one(x) : exp(x)                                          # src/hmc.jl:376
_finite_or_neginf(x) = isfinite(x) ? x : oftype(x, -Inf)                        # src/hmc.jl:375
_rand_bernoulli_log(rng, lp) = lp > 0 ? true : -randexp(rng) < lp              # src/hmc.jl:377-378
function _logaddexp(a, b)                                                       # LogExpFunctions-faithful
    a == b == -Inf && return oftype(a, -Inf)
    m = max(a, b)
    m + log1p(exp(min(a, b) - m))
end
# src/hmc.jl:839-858 verbatim (pure; expanding Stan-style slow windows).
function _warmup_window_ends(iterations::Int, initial_buffer::Int,
                             terminal_buffer::Int, first_window::Int)
    slow_start = initial_buffer + 1
    slow_stop = iterations - terminal_buffer
    slow_start > slow_stop && return Int[]
    ends = Int[]
    start = slow_start
    window = first_window
    while start <= slow_stop
        remaining = slow_stop - start + 1
        if 2window > remaining
            push!(ends, slow_stop)
            break
        end
        push!(ends, start + window - 1)
        start += window
        window *= 2
    end
    ends
end

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

# --- Object D1: Nesterov dual averaging (byte-faithful to src _dual_averaging_object + reset!) ------
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
    # segment: reinitialize accumulators in place to a fresh dual_averaging_state(initial) — src
    # reset!(::DualAveragingState, initial) (reactive_nuts.jl:917-931). Reused at each metric window.
    reset!(self, initial; target = 0.8, regularization_scale = 0.05,
           relaxation_exponent = 0.75, offset = 10) = begin
        self.iteration            = one(self.center)
        self.error                = zero(self.center)
        self.log_final            = zero(self.center)
        self.center               = log(oftype(self.center, 10)) + log(oftype(self.center, initial))
        self.target               = oftype(self.center, target)
        self.regularization_scale = oftype(self.center, regularization_scale)
        self.relaxation_exponent  = oftype(self.center, relaxation_exponent)
        self.offset               = oftype(self.center, offset)
    end
end

# --- Object D2: online componentwise Welford (byte-faithful to src _welford_object + reset!) --------
@kernel welford(n, mean, var) = begin
    function observe!(self, value; weight = 1)
        self.n   = self.n + weight
        fraction = weight / self.n
        @. self.var  = _smooth(self.var,
                               (value - _smooth(self.mean, value, fraction)) * (value - self.mean),
                               fraction)
        @. self.mean = _smooth(self.mean, value, fraction)
    end
    # segment: zero n/mean/var in place — src reset!(::WelfordVariance) (reactive_nuts.jl:1002-1007).
    reset!(self) = begin
        self.n = zero(self.n)
        @. self.mean = zero(self.mean)
        @. self.var  = zero(self.var)
    end
end

# --- Object E: statistics (real diagnostic accumulators incl. energy error) -------------------------
@kernel sampling_stats(sum_depth, sum_n_steps, sum_accept, sum_energy_error, n_divergent, n_draws) = begin
    function record!(self, depth, n_steps, acceptance, energy_error, divergent)
        self.sum_depth        = self.sum_depth + depth
        self.sum_n_steps      = self.sum_n_steps + n_steps
        self.sum_accept       = self.sum_accept + acceptance
        self.sum_energy_error = self.sum_energy_error + energy_error
        self.n_divergent      = self.n_divergent + (divergent ? 1 : 0)
        self.n_draws          = self.n_draws + 1
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
    stats = sampling_stats(0, 0, zero(stepsize), zero(stepsize), 0, 0)

    # --- read-only reducer segments (no state write) -------------------------------------------------
    # dham = finite(init_ham - moving_ham); divergence threshold; U-turn criteria (endpoint-momentum).
    energy_error(self, moving_ham) = _finite_or_neginf(self.init.hamiltonian - moving_ham)
    divergent(self, dham) = !(dham >= self.min_dham)
    uturn_ok(self, summed_mom, back_vel, fwd_vel) =                          # src _compute_criterion
        (dot(summed_mom, back_vel) > 0) && (dot(summed_mom, fwd_vel) > 0)
    uturn_ok_sum(self, left_mom, right_mom, back_vel, fwd_vel) =            # src _compute_criterion_sum
        uturn_ok(self, left_mom .+ right_mom, back_vel, fwd_vel)

    # --- state-mutating segments (endpoints / adaptation / metric) -----------------------------------
    # AUTHORITATIVE reset MIRROR (src _reset_transition! + step!'s backward-momentum negation, fused):
    # fwd = FULL identity copy of init; bwd = init with SIGNED momentum+velocity and copied pot/grad/
    # kin/ham (kinetic quadratic => sign-invariant). Distinct owned fwd/bwd buffers => the first leaf
    # reads the mirrored init gradient WITHOUT a pgrad recompute (fused == n_steps + 1).
    mirror!(self) = begin
        @. self.fwd.pos      = self.init.pos
        @. self.fwd.mom      = self.init.mom
        @. self.fwd.velocity = self.init.velocity
        @. self.fwd.gradient = self.init.gradient
        self.fwd.potential   = self.init.potential
        self.fwd.kinetic     = self.init.kinetic
        self.fwd.hamiltonian = self.init.hamiltonian
        @. self.bwd.pos      = self.init.pos
        @. self.bwd.mom      = -self.init.mom
        @. self.bwd.velocity = -self.init.velocity
        @. self.bwd.gradient = self.init.gradient
        self.bwd.potential   = self.init.potential
        self.bwd.kinetic     = self.init.kinetic
        self.bwd.hamiltonian = self.init.hamiltonian
    end
    # Unit-C: make init's FULL phase point current after accepted pos/mom with EXACTLY one pgrad — the
    # boundary gradient that also seeds the next transition's mirror (amortized fused == n_steps + 1).
    recompute_init!(self) = begin
        self.init.value_gradient = self.init.ham.potential_gradient!(self.init.pos)   # the one pgrad
        self.init.potential      = self.init.value_gradient.value
        self.init.gradient       = self.init.value_gradient.gradient
        self.init.velocity       = self.init.ham.chol \ self.init.mom
        self.init.kinetic        = (self.init.ham.logdet_chol +
                                    dot(self.init.mom, self.init.velocity)) / 2
        self.init.hamiltonian    = self.init.potential + self.init.kinetic
    end
    # accepted restore: write the multinomially-selected proposal pos/mom into init (state write).
    restore_accept!(self, accepted_pos, accepted_mom) = begin
        @. self.init.pos = accepted_pos
        @. self.init.mom = accepted_mom
    end
    # cross-view adaptation writes: DA -> shared stepsize slot; Welford variance -> shared mass diagonal.
    set_stepsize!(self, stepsize) = begin
        self.ham.stepsize = stepsize
    end
    # src _adapted_diagonal_metric (hmc.jl:860-881): n>1 guard/fallback + metric TYPE preservation.
    adapt_metric!(self; minimum_variance = 1e-3, regularization = 5) = begin
        weight = self.w.n / (self.w.n + regularization)
        sample_variance = self.w.n > 1 ?
            self.w.var .* (self.w.n / (self.w.n - 1)) :
            fill(one(eltype(self.w.var)), length(self.w.var))
        position_variance = @. max(minimum_variance,
                                   weight * sample_variance + (1 - weight) * minimum_variance)
        mass_diagonal = @. inv(position_variance)
        current = self.ham.metric
        self.ham.metric = current isa Diagonal ?
            Diagonal(convert(typeof(current.diag), mass_diagonal)) :
            convert(typeof(current), Matrix(Diagonal(mass_diagonal)))
    end

    # --- ORCHESTRATION (ordinary Julia; tree/proposal/control come from the PASSED scratch) ----------
    # per-transition scratch reset (src _reset_transition!'s proposal/tree/flag clears): fill!/copyto!
    # into the PREALLOCATED scratch; reads self.init but writes ONLY scratch (never self-path state).
    function reset_scratch!(self, scratch)
        c = scratch.control
        c.go_forward = true
        c.may_sample = true
        c.may_continue = true
        c.energy_error = zero(c.energy_error)
        c.diverged = false
        c.depth = 0
        c.n_steps = 0
        c.acceptance_sum = zero(c.acceptance_sum)
        for p in scratch.proposals
            copyto!(p.pos, self.init.pos)
            copyto!(p.mom, self.init.mom)
        end
        for t in scratch.trees
            fill!(t.log_weight, -Inf)
            fill!(t.backward.momentum, 0)
            fill!(t.backward.velocity, 0)
            fill!(t.backward_forward.momentum, 0)
            fill!(t.backward_forward.velocity, 0)
            fill!(t.summed_momentum.backward, 0)
            fill!(t.summed_momentum.forward, 0)
        end
        scratch
    end
    # src _flip! (hmc.jl:553-562): toggle direction, negate the (now) backward endpoint's momentum/
    # velocity into the tree, and flip the accumulated forward momentum. depth==1 is a no-op.
    function flip!(self, depth, scratch)
        depth > 1 || return scratch
        scratch.control.go_forward = !scratch.control.go_forward
        tree = scratch.trees[depth]
        back = scratch.control.go_forward ? self.bwd : self.fwd
        @. tree.backward.momentum = -back.mom
        @. tree.backward.velocity = -back.velocity
        @. tree.summed_momentum.forward *= -1
        scratch
    end
    # src _start_tree! (hmc.jl:569-601): base leaf = one leapfrog + energy/divergence + log-weight +
    # proposal copy; recursive step swaps proposals then draws the SUBTREE Bernoulli-log selection.
    function start_tree!(self, depth, scratch, rng)
        if depth == 1
            active = scratch.control.go_forward ? self.fwd : self.bwd
            leapfrog!(active)                                        # segment (the fused-gradient leaf)
            scratch.control.n_steps += 1
            e = energy_error(self, active.hamiltonian)              # reducer (finite(init_ham - ham))
            scratch.control.energy_error = e
            scratch.control.acceptance_sum += _min1exp(e)
            scratch.control.diverged = divergent(self, e)          # reducer
            if scratch.control.diverged
                scratch.control.may_continue = false               # divergence stop
                return scratch
            end
            scratch.trees[1].log_weight[1] = e
            copyto!(scratch.proposals[1].pos, active.pos)
            copyto!(scratch.proposals[1].mom, active.mom)
            return scratch
        end
        start_tree!(self, depth - 1, scratch, rng)
        if !scratch.control.may_continue
            scratch.control.may_sample = false
            return scratch
        end
        scratch.proposals[depth - 1], scratch.proposals[depth] =
            scratch.proposals[depth], scratch.proposals[depth - 1]           # _swap_proposal!(d-1,d)
        finish_tree!(self, depth - 1, scratch, rng)
        if scratch.control.may_sample && _rand_bernoulli_log(rng,            # SUBTREE proposal draw
                scratch.trees[depth - 1].log_weight[1] - scratch.trees[depth].log_weight[1])
            scratch.proposals[depth - 1], scratch.proposals[depth] =
                scratch.proposals[depth], scratch.proposals[depth - 1]
        end
        scratch
    end
    # src _finish_tree! (hmc.jl:603-660): combine sub-trees under logaddexp weights + accumulated
    # summed-momenta + the depth>1 three-part endpoint-momentum U-turn criterion.
    function finish_tree!(self, depth, scratch, rng)
        tree = scratch.trees[depth]
        supertree = scratch.trees[depth + 1]
        tree.log_weight[2] = tree.log_weight[1]
        active = scratch.control.go_forward ? self.fwd : self.bwd
        if depth == 1
            copyto!(supertree.backward.momentum, active.mom)
            copyto!(supertree.backward.velocity, active.velocity)
        else
            copyto!(supertree.backward.momentum, tree.backward.momentum)
            copyto!(supertree.backward.velocity, tree.backward.velocity)
            copyto!(tree.backward_forward.momentum, active.mom)
            copyto!(tree.backward_forward.velocity, active.velocity)
            copyto!(tree.summed_momentum.backward, tree.summed_momentum.forward)
        end
        start_tree!(self, depth, scratch, rng)
        if !scratch.control.may_continue
            scratch.control.may_sample = false
            return scratch
        end
        supertree.log_weight[1] = _logaddexp(tree.log_weight[1], tree.log_weight[2])
        if depth == 1
            @. supertree.summed_momentum.forward = supertree.backward.momentum + active.mom
            scratch.control.may_continue = uturn_ok(self,
                supertree.summed_momentum.forward, supertree.backward.velocity, active.velocity)
        else
            @. supertree.summed_momentum.forward =
                tree.summed_momentum.backward + tree.summed_momentum.forward
            scratch.control.may_continue =
                uturn_ok(self, supertree.summed_momentum.forward,
                         supertree.backward.velocity, active.velocity) &&
                uturn_ok_sum(self, tree.summed_momentum.backward, tree.backward.momentum,
                             supertree.backward.velocity, tree.backward.velocity) &&
                uturn_ok_sum(self, tree.backward_forward.momentum, tree.summed_momentum.forward,
                             tree.backward_forward.velocity, active.velocity)
        end
        scratch
    end
    # src step! (hmc.jl:663-685): reset mirror -> per-depth doubling with EXACT RNG order (direction
    # coin -> [subtree draws inside finish_tree!] -> top-level proposal draw) -> accepted restore.
    function step!(self, scratch, rng)
        mirror!(self)                                              # segment (init -> fwd/bwd, bwd signed)
        reset_scratch!(self, scratch)                             # orchestration (scratch clears)
        scratch.trees[1].log_weight[1] = 0
        for depth in 1:self.max_depth
            rand(rng, Bool) && flip!(self, depth, scratch)        # RNG #1: direction coin
            finish_tree!(self, depth, scratch, rng)               # RNG #2..k: subtree proposal draws
            scratch.control.depth = depth
            scratch.control.may_sample || break
            if _rand_bernoulli_log(rng,                            # RNG last-of-depth: top-level draw
                    scratch.trees[depth].log_weight[1] - scratch.trees[depth].log_weight[2])
                scratch.proposals[depth], scratch.proposals[end] =
                    scratch.proposals[end], scratch.proposals[depth]         # _swap_proposal!(depth)
            end
            scratch.control.may_continue || break
        end
        restore_accept!(self, scratch.proposals[end].pos, scratch.proposals[end].mom)   # segment
        recompute_init!(self)                                     # segment (Unit-C: one boundary pgrad)
        scratch
    end
    # src _probe_acceptance (hmc.jl:783-790): fresh momentum, one leapfrog on a mirrored endpoint,
    # min1exp(energy error). Uses fwd as probe scratch (overwritten by the next transition's mirror).
    function probe_acceptance!(self, rng, stepsize)
        refresh_momentum!(self.init, rng)                         # segment
        set_stepsize!(self, stepsize)                             # segment (write shared stepsize)
        mirror!(self)                                             # segment (init -> fwd/bwd)
        leapfrog!(self.fwd)                                       # segment (one step on fwd)
        _min1exp(energy_error(self, self.fwd.hamiltonian))        # reducer
    end
    # src find_initial_stepsize! (hmc.jl:803-837): double/halve until a one-step proposal crosses target.
    function find_initial_stepsize!(self, rng; initial = 1, target = 0.8,
                                    min_stepsize = eps(Float64), max_stepsize = 1e3,
                                    max_iterations = 32)
        stepsize = clamp(oftype(self.ham.stepsize, initial),
                         oftype(self.ham.stepsize, min_stepsize),
                         oftype(self.ham.stepsize, max_stepsize))
        acceptance = probe_acceptance!(self, rng, stepsize)
        increase = acceptance > target
        for _ in 1:max_iterations
            crossed = increase ? acceptance <= target : acceptance >= target
            crossed && break
            next_stepsize = increase ? 2stepsize : stepsize / 2
            next_stepsize = clamp(next_stepsize,
                                  oftype(stepsize, min_stepsize), oftype(stepsize, max_stepsize))
            next_stepsize == stepsize && break
            stepsize = next_stepsize
            acceptance = probe_acceptance!(self, rng, stepsize)
        end
        set_stepsize!(self, stepsize)                             # segment
        stepsize
    end
    # src warmup! (hmc.jl:898-988): initial step-size search, dual averaging every iteration, Stan-style
    # expanding metric windows with window-end mass update + DA/Welford reset, final da.final.
    function warmup!(self, iterations, scratch, rng; target_accept = 0.8, minimum_variance = 1e-3)
        n = iterations
        initial_stepsize = find_initial_stepsize!(self, rng; target = target_accept)   # orchestration
        reset!(self.da, initial_stepsize; target = target_accept)  # segment (start DA at initial)
        reset!(self.w)                                             # segment (start Welford clean)
        initial_count = 75
        terminal_count = 50
        window_size = 25
        if initial_count + window_size + terminal_count > n
            initial_count = floor(Int, 0.15n)
            terminal_count = floor(Int, 0.10n)
            window_size = n - initial_count - terminal_count
        end
        window_ends = n >= 20 ?
            _warmup_window_ends(n, initial_count, terminal_count, window_size) : Int[]
        next_window = 1
        adaptation_updates = 0
        for iteration in 1:n
            refresh_momentum!(self.init, rng)                     # segment
            step!(self, scratch, rng)                             # orchestration (one transition)
            acceptance = scratch.control.n_steps == 0 ? zero(self.ham.stepsize) :
                         scratch.control.acceptance_sum / scratch.control.n_steps
            fit!(self.da, acceptance)                             # segment (dual averaging)
            adaptation_updates += 1
            set_stepsize!(self, self.da.current)                  # segment (DA current -> stepsize)
            inside_slow_window = initial_count < iteration <= n - terminal_count
            inside_slow_window && observe!(self.w, self.init.pos) # segment (Welford, slow window only)
            if next_window <= length(window_ends) && iteration == window_ends[next_window]
                adapt_metric!(self; minimum_variance = minimum_variance)   # segment (window-end mass)
                restart = find_initial_stepsize!(self, rng; initial = self.ham.stepsize,
                                                 target = target_accept)   # orchestration
                reset!(self.da, restart; target = target_accept) # segment (restart DA per window)
                adaptation_updates = 0
                reset!(self.w)                                    # segment (restart Welford per window)
                next_window += 1
            end
            record!(self.stats, scratch.control.depth, scratch.control.n_steps, acceptance,
                    scratch.control.energy_error, scratch.control.diverged)   # segment (statistics)
        end
        adaptation_updates == 0 || set_stepsize!(self, self.da.final)   # segment (final da.final)
        self
    end
    # src sample! (hmc.jl:711-715): refresh momentum, one transition, record diagnostics (no adaptation).
    function sample!(self, scratch, rng)
        refresh_momentum!(self.init, rng)                         # segment
        step!(self, scratch, rng)                                 # orchestration
        acceptance = scratch.control.n_steps == 0 ? zero(self.ham.stepsize) :
                     scratch.control.acceptance_sum / scratch.control.n_steps
        record!(self.stats, scratch.control.depth, scratch.control.n_steps, acceptance,
                scratch.control.energy_error, scratch.control.diverged)   # segment
        scratch
    end
end

# --- PREALLOCATED orchestration scratch (built ONCE outside the transition; parent binding choice) --
# src _OracleNUTSState's trees (max_depth+1) / proposals (max_depth+2) / control flags, as ordinary
# non-authoritative buffers. NOT sampler state; two instances are conceptually independent.
mutable struct _NUTSControl{T}
    go_forward::Bool; may_sample::Bool; may_continue::Bool
    energy_error::T; diverged::Bool; depth::Int; n_steps::Int; acceptance_sum::T
end
struct _NUTSTree{L,V}
    log_weight::L
    backward::@NamedTuple{momentum::V, velocity::V}
    backward_forward::@NamedTuple{momentum::V, velocity::V}
    summed_momentum::@NamedTuple{backward::V, forward::V}
end
struct _NUTSProposal{V}
    pos::V; mom::V
end
struct _NUTSScratch{T,L,V}
    control::_NUTSControl{T}
    trees::Vector{_NUTSTree{L,V}}
    proposals::Vector{_NUTSProposal{V}}
end
function _nuts_scratch(position, momentum, max_depth::Integer)
    T = eltype(momentum)
    mv() = (momentum = zero(momentum), velocity = zero(momentum))
    tree() = _NUTSTree(fill(T(-Inf), 2), mv(), mv(),
                       (backward = zero(momentum), forward = zero(momentum)))
    control = _NUTSControl{T}(true, true, true, zero(T), false, 0, 0, zero(T))
    trees = _NUTSTree[tree() for _ in 1:(max_depth + 1)]
    proposals = _NUTSProposal[_NUTSProposal(copy(position), copy(momentum))
                              for _ in 1:(max_depth + 2)]
    _NUTSScratch(control, [t for t in trees], [p for p in proposals])
end

# --- Inc1 source-shape gate (construction only; NO execution/perf) — NON-VACUOUS structural checks --
if abspath(PROGRAM_FILE) == @__FILE__
    RKS = ReactiveKernels
    method_of(k, name) = (ms = filter(m -> m.name === name, RKS.kernel_methods(k));
                          @assert length(ms) == 1 "expected exactly one $(name) method"; ms[1])

    # ---- AST utilities over the RETAINED raw method bodies (Inc1 gives .name/.self/.body) -----------
    _walk(f, x) = (f(x); x isa Expr && foreach(a -> _walk(f, a), x.args); nothing)
    # unqualified call heads (for the call-graph closure)
    function body_calls(body)
        acc = String[]
        _walk(body) do x
            x isa Expr && x.head === :call && x.args[1] isa Symbol && push!(acc, string(x.args[1]))
        end
        acc
    end
    # ordered call heads in source/traversal order (qualified `a.b(...)` head => "b"); for RNG order.
    function ordered_call_heads(body)
        acc = String[]
        _walk(body) do x
            if x isa Expr && x.head === :call
                h = x.args[1]
                h isa Symbol && push!(acc, string(h))
                h isa Expr && h.head === :. && h.args[2] isa QuoteNode && push!(acc, string(h.args[2].value))
            end
        end
        acc
    end
    count_call(body, name) = count(==(name), body_calls(body))
    # dotted self-path of an lvalue (self.a.b / self.a.b[i]) => "a.b" (or nothing)
    function self_path(x, self)
        x isa Expr || return nothing
        if x.head === :ref
            return self_path(x.args[1], self)
        elseif x.head === :.
            base = x.args[1]
            field = x.args[2] isa QuoteNode ? string(x.args[2].value) : nothing
            field === nothing && return nothing
            base === self && return field
            inner = self_path(base, self)
            return inner === nothing ? nothing : string(inner, ".", field)
        end
        nothing
    end
    # FULL assignment grammar: any head ending in `=` that is not a comparison, plus mutating calls
    # (copyto!/fill!/lmul!/ldiv!/mul!/randn!) whose DESTINATION (first arg) is a self-path.
    const _CMP = Set([:(==), :(!=), :(<=), :(>=), :(===), :(!==), :(.==), :(.!=), :(.<=), :(.>=)])
    is_assign_head(h::Symbol) = (s = String(h); endswith(s, "=") && !(h in _CMP))
    is_assign_head(::Any) = false
    const _MUT = Set(["copyto!", "fill!", "lmul!", "ldiv!", "mul!", "randn!", "rmul!"])
    # collect every self-path WRITE target (assignment LHS + mutating-call destination), recursing
    # through @. macrocalls (their inner assignment is a normal Expr under :macrocall args).
    function self_writes(body, self)
        acc = String[]
        _walk(body) do x
            x isa Expr || return
            if is_assign_head(x.head) && length(x.args) >= 1
                p = self_path(x.args[1], self); p === nothing || push!(acc, p)
            elseif x.head === :call && x.args[1] isa Symbol && string(x.args[1]) in _MUT &&
                   length(x.args) >= 2
                p = self_path(x.args[2], self); p === nothing || push!(acc, p)
            end
        end
        acc
    end
    subseq(hay, needles) = begin      # are `needles` a subsequence of `hay` (order preserved)?
        i = 1
        for h in hay
            i > length(needles) && break
            h == needles[i] && (i += 1)
        end
        i > length(needles)
    end

    # (a) skeleton types: stateful vs stateless ------------------------------------------------------
    for k in (endpoint, sampler, dual_averaging, welford, sampling_stats)
        @assert k isa RKS._StatefulKernelSkeleton "$(k) must be a method-bearing (stateful) @kernel"
    end
    @assert !(hamiltonian isa RKS._StatefulKernelSkeleton) "hamiltonian is methodless -> stateless"

    # (b) exact method inventory (incl. reset!, probe/stepsize search, full recursion/adaptation) -----
    mnames(k) = Set(m.name for m in RKS.kernel_methods(k))
    @assert mnames(endpoint) == Set([:refresh_momentum!, :leapfrog!]) "endpoint inventory"
    @assert mnames(dual_averaging) == Set([:fit!, :reset!]) "dual_averaging inventory"
    @assert mnames(welford) == Set([:observe!, :reset!]) "welford inventory"
    @assert mnames(sampling_stats) == Set([:record!]) "stats inventory"
    want_sampler = Set([:energy_error, :divergent, :uturn_ok, :uturn_ok_sum, :mirror!,
                        :recompute_init!, :restore_accept!, :set_stepsize!, :adapt_metric!,
                        :reset_scratch!, :flip!, :start_tree!, :finish_tree!, :step!,
                        :probe_acceptance!, :find_initial_stepsize!, :warmup!, :sample!])
    @assert mnames(sampler) == want_sampler "sampler inventory: $(mnames(sampler))"

    # (c) call-graph closure: every method-shaped call (`!`-suffixed or a named reducer) in the sampler
    #     method bodies resolves to an authored sibling/child method or an allowed external primitive.
    reducers = Set(["energy_error", "divergent", "uturn_ok", "uturn_ok_sum"])
    authored = Set(string(n) for n in want_sampler) ∪
               Set(["refresh_momentum!", "leapfrog!", "fit!", "reset!", "observe!", "record!"])
    allowed_ext = Set(["copy", "zero", "one", "exp", "log", "log1p", "sqrt", "dot", "isfinite",
                       "oftype", "cholesky", "logdet", "randn!", "lmul!", "ldiv!", "inv", "max",
                       "min", "clamp", "convert", "Matrix", "Diagonal", "fill", "fill!", "copyto!",
                       "floor", "length", "eltype", "float", "typeof", "rand", "push!",
                       "_min1exp", "_finite_or_neginf", "_logaddexp", "_rand_bernoulli_log",
                       "_smooth", "_warmup_window_ends", "_nuts_scratch", "!", "!="])
    for m in RKS.kernel_methods(sampler), call in body_calls(m.body)
        (endswith(call, "!") || call in reducers) || continue
        (call in authored || call in allowed_ext) ||
            error("unresolved method call `$(call)` in sampler.$(m.name) — not authored/allowed")
    end

    # (d) segment/orchestration contract: ORCHESTRATION bodies have ZERO self-path WRITES (full
    #     assignment grammar + mutating-call destinations), so they mutate state only via segments.
    orchestration = Set([:reset_scratch!, :flip!, :start_tree!, :finish_tree!, :step!,
                         :probe_acceptance!, :find_initial_stepsize!, :warmup!, :sample!])
    for m in RKS.kernel_methods(sampler)
        m.name in orchestration || continue
        w = self_writes(m.body, m.self)
        isempty(w) || error("orchestration sampler.$(m.name) writes self-path(s) $(w) — use segments")
    end

    # (e) FULL authoritative mirror: exact fwd+bwd fields; bwd momentum/velocity SIGNED (unary minus) --
    mirror = method_of(sampler, :mirror!)
    mw = Set(self_writes(mirror.body, mirror.self))
    for f in ("pos", "mom", "velocity", "gradient", "potential", "kinetic", "hamiltonian")
        @assert "fwd.$f" in mw "mirror! must copy init.$f -> fwd.$f"
        @assert "bwd.$f" in mw "mirror! must set bwd.$f"
    end
    # bwd.mom and bwd.velocity RHS must be a NEGATION of the init source (signed backward half).
    function mirror_bwd_signed(body, self, field)
        signed = Ref(false)
        _walk(body) do x
            x isa Expr && x.head === :(=) || return
            self_path(x.args[1], self) == "bwd.$field" || return
            rhs = x.args[2]
            rhs isa Expr && rhs.head === :call && rhs.args[1] === :- && length(rhs.args) == 2 &&
                (signed[] = true)
        end
        signed[]
    end
    # the `@. bwd.mom = -init.mom` lowers the RHS negation inside a macrocall; scan the stringified body.
    mbody = string(mirror.body)
    @assert occursin("bwd.mom = -", mbody) || mirror_bwd_signed(mirror.body, mirror.self, "mom") "bwd momentum must be signed"
    @assert occursin("bwd.velocity = -", mbody) || mirror_bwd_signed(mirror.body, mirror.self, "velocity") "bwd velocity must be signed"

    # (f) Unit-C: recompute_init! makes the FULL phase point current with EXACTLY one pgrad ------------
    rec = method_of(sampler, :recompute_init!)
    rw = Set(self_writes(rec.body, rec.self))
    for f in ("value_gradient", "potential", "gradient", "velocity", "kinetic", "hamiltonian")
        @assert "init.$f" in rw "recompute_init! must make init.$f current"
    end
    @assert count(occursin("potential_gradient!", string(s)) for s in
                  [a for a in rec.body.args]) >= 0   # (presence; count-of-one asserted via string)
    @assert length(collect(eachmatch(r"potential_gradient!", string(rec.body)))) == 1 "recompute_init! must do EXACTLY one pgrad"

    # (g) metric adaptation: n>1 guard + type preservation (src _adapted_diagonal_metric) -------------
    am = string(method_of(sampler, :adapt_metric!).body)
    @assert occursin("> 1", am) "adapt_metric! must guard n>1 (no n==1 NaN)"
    @assert occursin("isa Diagonal", am) && occursin("convert", am) "adapt_metric! must preserve metric type"
    @assert occursin("inv", am) && occursin("position_variance", am) "adapt_metric! mass diagonal retained"

    # (h) warmup boundaries: initial search + expanding windows + per-window DA/Welford reset + final --
    wu = method_of(sampler, :warmup!)
    wuc = body_calls(wu.body)
    @assert "find_initial_stepsize!" in wuc "warmup! must run initial step-size search"
    @assert occursin("_warmup_window_ends", string(wu.body)) "warmup! must compute expanding windows"
    @assert count_call(wu.body, "reset!") >= 3 "warmup! must reset DA+Welford (start + per window)"
    @assert count_call(wu.body, "adapt_metric!") >= 1 "warmup! must update mass at window ends"
    @assert occursin("final", string(wu.body)) "warmup! must apply final da.final"
    @assert occursin("observe!", string(wu.body)) "warmup! must accumulate Welford in slow windows"

    # (i) multinomial tree: log weights + proposal swap + Bernoulli selection + accumulation + criterion
    stb = start_tree = method_of(sampler, :start_tree!).body
    @assert occursin("log_weight", string(stb)) "start_tree! must set per-depth log weights"
    @assert count_call(stb, "_rand_bernoulli_log") == 1 "start_tree! must draw exactly one subtree selection"
    ftb = method_of(sampler, :finish_tree!).body
    @assert occursin("_logaddexp", string(ftb)) "finish_tree! must combine weights via logaddexp"
    @assert occursin("summed_momentum", string(ftb)) "finish_tree! must accumulate summed momenta"
    @assert count_call(ftb, "uturn_ok") + count_call(ftb, "uturn_ok_sum") >= 4 "finish_tree! must apply the depth>1 three-part criterion"
    @assert occursin("backward.velocity", string(ftb)) && occursin("backward_forward", string(ftb)) "finish_tree! must use backward/forward velocities"

    # (j) EXACT RNG call order in step!: direction coin -> subtree draws (in finish_tree!) -> top draw --
    stepb = method_of(sampler, :step!).body
    heads = ordered_call_heads(stepb)
    @assert subseq(heads, ["rand", "flip!", "finish_tree!", "_rand_bernoulli_log"]) "step! RNG/site order: coin -> flip! -> finish_tree! -> top-level proposal draw"
    @assert count_call(stepb, "rand") == 1 "step! must draw exactly one direction coin per depth site"
    @assert count_call(stepb, "_rand_bernoulli_log") == 1 "step! must have exactly one top-level proposal draw site"
    @assert occursin("may_continue", string(stepb)) "step! must honor the U-turn/divergence stop"
    @assert occursin("proposals[end]", string(stepb)) "step! must restore the tournament-selected proposal"

    # (k) divergence stop is wired: start_tree! sets may_continue=false on divergence -------------------
    @assert occursin("may_continue = false", string(stb)) "start_tree! must stop on divergence"

    # (l) stats records energy error (real diagnostic) ------------------------------------------------
    rc = method_of(sampling_stats, :record!)
    @assert "sum_energy_error" in Set(self_writes(rc.body, rc.self)) "record! must accumulate energy error"

    # (m) scratch: NO tree/proposal/control construction inside hot orchestration; buffers from scratch -
    hot = Set([:reset_scratch!, :flip!, :start_tree!, :finish_tree!, :step!,
               :probe_acceptance!, :find_initial_stepsize!, :warmup!, :sample!])
    for m in RKS.kernel_methods(sampler)
        m.name in hot || continue
        s = string(m.body)
        # forbid per-transition BUFFER allocation (arrays) + scratch construction; scalar control
        # resets like `zero(c.energy_error)` are legitimate in-place scratch mutation and allowed.
        for bad in ("_nuts_scratch", "_NUTSTree", "_NUTSProposal", "_NUTSControl",
                    "zeros(", "similar(", "copy(",
                    "zero(self.init.pos", "zero(self.init.mom", "zero(pos", "zero(position", "zero(momentum")
            @assert !occursin(bad, s) "hot orchestration sampler.$(m.name) constructs scratch/buffers ($bad) — must come from the passed scratch"
        end
    end
    # scratch is genuinely preallocated + two instances independent (distinct buffers).
    let s1 = _nuts_scratch(zeros(3), zeros(3), 4), s2 = _nuts_scratch(zeros(3), zeros(3), 4)
        @assert length(s1.trees) == 5 && length(s1.proposals) == 6 "scratch sizing (max_depth+1 / +2)"
        @assert s1.proposals[1].pos !== s2.proposals[1].pos "two scratch instances must be independent"
        @assert s1.trees[1].log_weight !== s2.trees[1].log_weight "two scratch instances must be independent"
    end

    # (n) no manual plumbing / no @reactive in any authored method body ------------------------------
    allobjs = (endpoint, dual_averaging, welford, sampling_stats, sampler)
    bodies = join([string(m.body) for k in allobjs for m in RKS.kernel_methods(k)], "\n")
    for bad in ("@reactive", "Graph(", "add!", "output_binding", "_RecipeApplier",
                "compile_update", "bind_schedule", "cache_apply")
        @assert !occursin(bad, bodies) "manual-plumbing/@reactive token in a method body: $bad"
    end

    # (o) faithful recurrence bodies retained (DA/Welford/leapfrog) -----------------------------------
    @assert occursin("relaxation_exponent", bodies) && occursin("log_final", bodies) &&
            occursin("log_current", bodies) "DA fit! recurrence retained"
    @assert occursin("_smooth", bodies) "Welford _smooth recurrence retained"
    @assert occursin("stepsize", bodies) && occursin("gradient", bodies) &&
            occursin("velocity", bodies) "leapfrog Stormer-Verlet retained"

    println("Inc1 source-shape gate PASS (NON-VACUOUS): stateful/stateless skeletons + exact method")
    println("inventory + call-graph closure + segment/orchestration contract (full assignment grammar,")
    println("zero self-writes in orchestration) + FULL authoritative mirror (14 fields, bwd mom+vel")
    println("signed) + Unit-C full phase point w/ exactly-one pgrad + n>1 metric guard + type")
    println("preservation + warmup windows/DA+Welford reset/final + multinomial tree (log weights,")
    println("proposal swap, subtree+top-level Bernoulli draws, accumulated momenta, 3-part criterion,")
    println("divergence stop) + EXACT RNG order + energy-error stats + preallocated independent scratch.")
    println("MethodIR/execution ABSENT (118ad97) — NO execution/perf claim.")
end
