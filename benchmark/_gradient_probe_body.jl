# Inner body of benchmark/gradient_probe.jl (temp env: DI+Enzyme + ReactiveKernels).
#
# ATTRIBUTION probe: run_reactive's @timed block includes per-chain construction and
# FIRST-USE compilation (nuts_state, adaptation/stats specialized programs, RGF +
# Enzyme codegen). This profiles each stage COLD (first use of a concrete type,
# includes compilation) vs WARM (same concrete type already compiled), so the
# comparator's "after-compilation" sampling can be separated from setup/compilation.

const RK = ReactiveKernels

const BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)
# log density of a standard normal (matches the comparator's _gaussian_logp): the
# wrapper negates this into the POTENTIAL U(q)=+½‖q‖² with gradient +q — a valid
# Gaussian, not the inverted one a +½ log density would produce.
_logp(q) = -0.5 * sum(abs2, q)

mutable struct Target{P}
    gradient_calls::Int
    preparation::P
end
Target(D) = Target(0, prepare_gradient(_logp, BACKEND, zeros(D)))

_pos(D) = [sin(1.0i) for i in 1:D]

# Build a fresh potential_gradient! + group of the SAME concrete types each call.
function make_group(D)
    target = Target(D)
    function potential_gradient!(gradient, position)
        target.gradient_calls += 1
        value = first(value_and_gradient!(_logp, gradient, target.preparation, BACKEND, position))
        gradient .*= -1
        -value
    end
    RK.reactive_nuts_group(potential_gradient!, Matrix{Float64}(I, D, D), copy(_pos(D)), zeros(D))
end
_sf() = RK.partial(RK.leapfrog!; stepsize = 0.1)

const _SINK = Ref(0.0)

# One prepared-DI vs bundle comparison at fixed N with allocation slopes.
_probe(f, n) = (s = 0.0; @inbounds for _ in 1:n; s += f(); end; _SINK[] += s; nothing)
function _alloc_slope(f::F; n = 2000) where {F}
    _probe(f, 2n); a1 = @allocated _probe(f, n); a2 = @allocated _probe(f, 2n)
    (percall = max(0, round(Int, (a2 - a1) / n)), totals = (a1, a2))
end
function _warm_ns(f::F; batch = 500, reps = 100) where {F}
    _probe(f, 3batch); best = Inf
    for _ in 1:reps
        t0 = time_ns(); _probe(f, batch); t1 = time_ns(); best = min(best, (t1 - t0) / batch)
    end
    best
end

function run(; D = 4)
    println((; probe = :gradient_stages, D, julia = string(VERSION)))

    # Warm ALL concrete-type code paths once (this is the compilation the comparator's
    # compile_paths is meant to cover). Timed as the cold pass.
    coldgroup = make_group(D)
    cold = @timed begin
        s = RK.nuts_state(coldgroup; rng = Xoshiro(1), step_f = _sf(), max_depth = 6)
        RK.warmup!(s, 50; target_accept = 0.8)
        RK.sample!(s, 20)
    end
    println((; stage = :COLD_first_use_incl_compilation,
               seconds = cold.time, bytes = cold.bytes))

    # WARM stages: same concrete types, already compiled.
    g2 = make_group(D)
    tconstruct = @timed RK.nuts_state(g2; rng = Xoshiro(2), step_f = _sf(), max_depth = 6)
    sampler = tconstruct.value
    twarmup = @timed RK.warmup!(sampler, 200; target_accept = 0.8)
    tsample = @timed RK.sample!(sampler, 1000)
    chain = tsample.value
    mean_steps = sum(d -> d.n_steps, chain.diagnostics) / length(chain.diagnostics)
    divergences = count(d -> d.diverged, chain.diagnostics)
    println((; stage = :WARM_nuts_state_construct, seconds = tconstruct.time, bytes = tconstruct.bytes))
    println((; stage = :WARM_warmup_200, seconds = twarmup.time, bytes = twarmup.bytes,
               per_transition_us = twarmup.time / 200 * 1e6))
    println((; stage = :WARM_sample_1000, seconds = tsample.time, bytes = tsample.bytes,
               per_draw_us = tsample.time / 1000 * 1e6,
               bytes_per_draw = tsample.bytes / 1000,
               mean_steps, divergences))    # sanity: valid Gaussian => modest steps, ~0 divergences
    # copy(group) cost (per chain in the comparator).
    tcopy = @timed copy(g2)
    println((; stage = :WARM_copy_group, seconds = tcopy.time, bytes = tcopy.bytes))

    # Direct prepared DI vs the reactive bundle recompute, fixed N, allocation slopes.
    target = Target(D); prep = target.preparation
    x = _pos(D); grad = similar(x)
    pg!(gr, p) = (v = first(value_and_gradient!(_logp, gr, target.preparation, BACKEND, p)); gr .*= -1; -v)
    vg = RK._grad_bundle(pg!, copy(x))
    direct() = (value_and_gradient!(_logp, grad, prep, BACKEND, x); grad[1])
    bundle() = RK._nuts_cache_apply(vg, RK._grad_bundle, pg!, x).value
    println((; barrier = :direct_prepared_DI, ns = round(_warm_ns(direct); digits = 2),
               alloc = _alloc_slope(direct)))
    println((; barrier = :reactive_bundle, ns = round(_warm_ns(bundle); digits = 2),
               alloc = _alloc_slope(bundle)))
    println((; probe = :ok))
    nothing
end

run()
