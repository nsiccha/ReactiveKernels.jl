# ReactiveHMC-shaped `@kernel` NUTS AUTHORING FIXTURE (V7, rooted at Inc1 substrate 118ad97).
#
# This is the author-facing surface the user asked for: ONE `@kernel` macro, concise composable
# method bodies faithful to ReactiveHMC.jl, no Graph/add!/applier/output_binding plumbing and no
# production transition wiring. Once poc's Increment-2 MethodIR clears, these EXACT definitions
# become the MethodIR consumer fixture and then the production sampler.
#
# STAGE (Inc1): gated on MACRO CONSTRUCTION + explicit-self / nested-path / source-shape only.
# MethodIR / effect-lowering is intentionally ABSENT at 118ad97 — these objects CONSTRUCT (stateful
# skeletons with retained raw method bodies) but do NOT execute/compile yet. No execution/perf claim.
#
# Method-bearing `@kernel` binds a `const`, so every definition is top level.
using ReactiveKernels
using LinearAlgebra, Random

# --- Object A: shared Hamiltonian / metric authority ------------------------
# Cut points depend only on the metric (chol/logdet_chol) — composed BY IDENTITY into each endpoint,
# so init/fwd/bwd share ONE slot each; `potential_gradient!` and `stepsize` are shared sources too.
@kernel hamiltonian(potential_gradient!, metric, stepsize) = begin
    chol        = cholesky(metric)
    logdet_chol = logdet(chol)
end

# --- Object B: a phase-point endpoint ---------------------------------------
# OWNS pos/mom + its derived value-gradient/velocity/kinetic/hamiltonian; READS the shared `ham`.
# `leapfrog!` and `refresh_momentum!` are the faithful ReactiveHMC method bodies (segments).
@kernel endpoint(ham, pos, mom) = begin
    value_gradient = ham.potential_gradient!(pos)                 # owned (value, gradient) bundle
    potential      = value_gradient.value
    gradient       = value_gradient.gradient
    velocity       = ham.chol \ mom                              # owned; reads shared chol
    kinetic        = (ham.logdet_chol + dot(mom, velocity)) / 2  # reads shared logdet_chol cut point
    hamiltonian    = potential + kinetic

    # segment: resample momentum ~ N(0, M); velocity/kinetic/hamiltonian recompute (0 pgrad).
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

# --- Object D: adaptation (dual averaging + Welford) -------------------------
@kernel dual_averaging(mu, gamma, t0, kappa) = begin
    log_step_bar = zero(mu)
    h_bar        = zero(mu)
    t            = 0
    current      = exp(mu)          # current stepsize (derived over the accumulators)

    fit!(self, accept_stat; target = 0.8) = begin
        self.t     = self.t + 1
        self.h_bar = (1 - 1 / (self.t + self.t0)) * self.h_bar + (target - accept_stat) / (self.t + self.t0)
    end
    # cross-view write, authored on the sampler (see below); here the DA exposes `current`.
end

@kernel welford(dim) = begin
    n    = 0
    mean = zeros(dim)
    m2   = zeros(dim)

    function step!(self, position)
        self.n = self.n + 1
        @. self.mean = self.mean + (position - self.mean) / self.n
    end
end

# --- Object E: statistics ---------------------------------------------------
@kernel sampling_stats(dim) = begin
    n_draws = 0
    function record!(self, endpoint)
        self.n_draws = self.n_draws + 1
    end
end

# --- The sampler (driver): one shared ham + three composed endpoints + adaptation/stats ----
# SEGMENTS: leapfrog!/refresh_momentum! (endpoint), fit!/adapt_stepsize!/adapt_metric!, uturn_ok,
#           the Unit-C recompute. ORCHESTRATION: step!/grow!/flip!/reset!/restore! (ordinary Julia,
#           dynamic tree recursion + RNG; mutate state ONLY via segment calls; scratch = plain arrays).
@kernel sampler(potential_gradient!, metric, stepsize, position, momentum, max_depth) = begin
    ham  = hamiltonian(potential_gradient!, metric, stepsize)     # shared authority (one instance)
    init = endpoint(ham, position, momentum)                      # three composed phase points
    fwd  = endpoint(ham, position, momentum)
    bwd  = endpoint(ham, position, momentum)
    da   = dual_averaging(zero(stepsize), 0.05, 10.0, 0.75)
    w    = welford(length(position))
    stats = sampling_stats(length(position))

    may_continue = true
    may_sample   = true

    # segment: U-turn criterion — pure dot-product reducer over endpoint momentum/velocity.
    uturn_ok(self, summed_mom, back_vel, fwd_vel) =
        (dot(summed_mom, back_vel) > 0) && (dot(summed_mom, fwd_vel) > 0)

    # cross-view segments: adaptation authority -> endpoint sources (owner-path dispatch, one state).
    adapt_stepsize!(self) = begin
        self.ham.stepsize = self.da.current
    end
    adapt_metric!(self) = begin
        self.ham.metric = self.w.mean
    end

    # orchestration (ordinary Julia; drives segments; the compiler-owned coarse epoch is scoped here):
    function step!(self, rng)
        reset!(self)                                    # mirror init -> fwd/bwd (compiled)
        gofwd = true
        for depth in 1:self.max_depth
            rand(rng, Bool) && (gofwd = flip!(self, depth))
            grow!(self, depth, gofwd, rng)              # calls leapfrog!(self.fwd/bwd) + uturn_ok
            self.may_continue || break
        end
        restore!(self)                                  # accepted -> init (Unit-C recompute)
    end
    function sample!(self, rng)
        refresh_momentum!(self.init, rng)               # segment (rng RuntimeArg)
        step!(self, rng)
    end
end

# --- Inc1 construction / source-shape assertions (NO execution/perf) --------
# These gate the fixture on the substrate available at 118ad97: macro constructs, method-presence
# discrimination, explicit-self, nested self paths, retained raw bodies. MethodIR is absent, so we do
# NOT prepare/execute. Full raw method bodies are preserved above for poc's Increment-2 emission.
if abspath(PROGRAM_FILE) == @__FILE__
    RKS = ReactiveKernels
    @assert endpoint isa RKS._StatefulKernelSkeleton "endpoint is a method-bearing (stateful) @kernel"
    @assert sampler  isa RKS._StatefulKernelSkeleton "sampler is a stateful @kernel"
    @assert dual_averaging isa RKS._StatefulKernelSkeleton
    @assert welford isa RKS._StatefulKernelSkeleton
    @assert !(hamiltonian isa RKS._StatefulKernelSkeleton) "hamiltonian is methodless -> stateless"
    println("Inc1 construction OK: endpoint/sampler/dual_averaging/welford are stateful @kernel skeletons;")
    println("hamiltonian is stateless; faithful @kernel method bodies (leapfrog!/refresh_momentum!/step!/")
    println("fit!/adapt_stepsize!/uturn_ok) retained. MethodIR/execution ABSENT until poc Increment 2.")
end
