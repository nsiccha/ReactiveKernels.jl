# JAX translation of the native ODE bridges `rk_ode_rk45_tol` and
# `rk_ode_bdf_tol` (`native_ode.jl`): one adaptive diffrax solve with the
# caller's tolerances and per-output step budget, outputs from the solver's
# dense interpolant, and reverse-mode derivatives of the discrete solve
# (diffrax's `RecursiveCheckpointAdjoint`), so a gradient's accuracy follows
# the caller's tolerances as the native forward sensitivities' does. A module
# imports diffrax only when it solves an ODE.

# `_RKOutputBudget` counts accepted steps toward the first requested output
# after the time a step starts at, and fails a solve whose count for one
# output exceeds `limit`: the native bridges' `OutputBudget`, which lets
# `while t < output` take at most `maxsteps` accepted steps per output.
# diffrax's own `max_steps` is a static bound on attempted steps that sizes
# no buffer here (`checkpoints` does, so a gradient's memory does not grow
# with it): it is the largest int32, as the native solve sets no such bound.
# The solve also fails (status 3) where the step size falls below 16 ulps of
# the latest time, or a step starts from a non-finite state or leaves a
# non-finite step size, as the native solve stops as `DtLessThanMin` or
# `Unstable`. diffrax would otherwise reject every later step until its
# `max_steps`: a finite-time blow-up keeps proposing a step of a few ulps,
# and its PID controller keeps a NaN step size.
# `_rk_ode_solve` returns the times-by-states outputs and a status: 0 solved,
# 1 invalid inputs, 2 a per-output budget exceeded, 3 the solver failed. Only
# the first `n` output times are requested; later entries (a padded row's)
# repeat the last requested time, so their rows repeat its state.
const _JAX_ODE_SOLVE = """
import diffrax
import equinox as eqx


class _RKFiniteTrial:
    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        step = super().step
        inputs = (terms, t0, t1, y0, args, solver_state, made_jump)
        candidate, error, *_ = step(*inputs)
        finite = jnp.all(jnp.isfinite(candidate)) & jnp.all(jnp.isfinite(error))

        # The accepted solution has no dependence on a discarded trial. Mask
        # its inputs in AD: a zero cotangent at its output can still multiply
        # an infinite intermediate in the Runge-Kutta stages. For values alone
        # stop_gradient is the identity and both calls have the same inputs.
        def step_input(x):
            return jnp.where(finite, x, jax.lax.stop_gradient(x)) if eqx.is_inexact_array(x) else x

        candidate, error, dense, state, result = step(*jax.tree.map(step_input, inputs))
        # PID scales error by the size of the candidate. Inf / Inf gives NaN
        # and hence a NaN step size. A finite scale and infinite error ask it
        # to retry at a smaller step. Diffrax discards the solver result on a
        # rejection; a controller that accepts this trial must report failure.
        return (jnp.where(finite, candidate, y0), jnp.where(finite, error, jnp.inf),
                dense, state, diffrax.RESULTS.where(finite, result, diffrax.RESULTS.nonfinite))


class _RKDopri5(_RKFiniteTrial, diffrax.Dopri5):
    pass


class _RKKvaerno5(_RKFiniteTrial, diffrax.Kvaerno5):
    pass


class _RKOutputBudget(diffrax.AbstractAdaptiveStepSizeController):
    # ReactiveKernels' per-output step budget: at most `limit` accepted steps
    # between consecutive requested outputs.
    inner: diffrax.PIDController
    ts: Any
    limit: Any

    @property
    def rtol(self):
        return self.inner.rtol

    @property
    def atol(self):
        return self.inner.atol

    @property
    def norm(self):
        return self.inner.norm

    def wrap(self, direction):
        return _RKOutputBudget(self.inner.wrap(direction), self.ts, self.limit)

    def init(self, terms, t0, t1, y0, dt0, args, func, error_order):
        t, state = self.inner.init(terms, t0, t1, y0, dt0, args, func, error_order)
        target = jnp.searchsorted(self.ts, t0, side="right")
        return t, (state, jnp.zeros((), dtype=target.dtype), target)

    def adapt_step_size(self, t0, t1, y0, y1_candidate, args, y_error, error_order,
                        controller_state):
        state, count, target = controller_state
        keep, next_t0, next_t1, jump, state, result = self.inner.adapt_step_size(
            t0, t1, y0, y1_candidate, args, y_error, error_order, state)
        now = jnp.searchsorted(self.ts, t0, side="right")
        count = jnp.where(now == target, count, 0)
        count = jnp.where(keep, count + 1, count)
        result = diffrax.RESULTS.where(count > self.limit, diffrax.RESULTS.max_steps_reached,
                                       result)
        # A non-finite starting state or unusable next step cannot advance.
        unstable = ~jnp.all(jnp.isfinite(y0)) | ~jnp.isfinite(next_t1 - next_t0)
        result = diffrax.RESULTS.where(unstable, diffrax.RESULTS.nonfinite, result)
        return keep, next_t0, next_t1, jump, (state, count, now), result


def _rk_ode_solve(solver, rhs, y0, t0, ts, n, reltol, abstol, maxsteps, args):
    # ReactiveKernels' rk_ode_rk45_tol / rk_ode_bdf_tol: integrate
    # dy/dt = rhs(t, y, args) from t0, returning the state at each of the
    # first n output times (rows) and a status.
    dtype = jnp.result_type(float)
    cap = 2 ** 31 - 1
    if isinstance(maxsteps, int):
        maxsteps = min(maxsteps, cap)
    y0 = jnp.asarray(y0, dtype=dtype)
    ts = jnp.asarray(ts, dtype=dtype)
    t0 = jnp.asarray(t0, dtype=dtype)
    width = ts.shape[0]
    requested = jnp.arange(width) < n
    previous = jnp.concatenate([t0[None], ts[:-1]])
    valid = (jnp.all(jnp.isfinite(y0)) & jnp.isfinite(t0)
             & jnp.all(~requested | (jnp.isfinite(ts) & (ts > previous)))
             & jnp.isfinite(reltol) & (reltol >= 0) & jnp.isfinite(abstol) & (abstol >= 0)
             & ((reltol > 0) | (abstol > 0)) & (maxsteps > 0))
    # A row with no requested time (a padded lane) solves a unit interval
    # whose rows are padding.
    last = jnp.where(n > 0, ts[jnp.maximum(n, 1) - 1], t0 + 1)
    ok = valid & (n > 0)
    last = jnp.where(ok, last, t0 + 1)
    ts = jnp.where(requested & ok, ts, last)
    # A step this short no longer advances time controllably: diffrax steps
    # to rounded times, and a few ulps of t are as short as a step gets.
    dtmin = jax.lax.stop_gradient(16 * jnp.finfo(dtype).eps * jnp.maximum(jnp.abs(t0), jnp.abs(last)))
    # Include traced values captured by the RHS among the trial's AD inputs.
    term = diffrax.ODETerm(eqx.filter_closure_convert(rhs, t0, y0, args))
    sol = diffrax.diffeqsolve(
        term, solver, t0, last, None, y0, args=args,
        saveat=diffrax.SaveAt(ts=ts),
        stepsize_controller=_RKOutputBudget(
            diffrax.PIDController(rtol=jnp.where(ok, reltol, 1.0), atol=jnp.where(ok, abstol, 1.0),
                                  dtmin=dtmin, force_dtmin=False),
            ts, jnp.where(ok, maxsteps, cap)),
        max_steps=cap, adjoint=diffrax.RecursiveCheckpointAdjoint(checkpoints=64), throw=False)
    status = jnp.where(sol.result == diffrax.RESULTS.successful, 0,
                       jnp.where(sol.result == diffrax.RESULTS.max_steps_reached, 2, 3))
    status = jnp.where(valid, status, 1)
    return sol.ys, jnp.where(n > 0, status, 0)
"""

_jax_ode_solver(::typeof(rk_ode_rk45_tol)) = "Dopri5"
_jax_ode_solver(::typeof(rk_ode_bdf_tol)) = "Kvaerno5"

for f in (rk_ode_rk45_tol, rk_ode_bdf_tol)
    @eval _jax_function_call(ctx::_JaxContext, f::typeof($f), args::Vector{_PyVal}) =
        _jax_ode_call(ctx, f, args)
end

# `rk_ode_*_tol(f, y0, t0, ts, reltol, abstol, maxsteps, args...)`: the
# right-hand side `f(t, y, args...)` becomes a nested Python function of
# `(t, y, args)`, diffrax's vector-field signature, translated like any
# helper; `args` carries the extra arguments that are runtime values, and the
# others (functions, static values) are read where `f` was called.
function _jax_ode_call(ctx::_JaxContext, f, args::Vector{_PyVal})
    name = nameof(f)
    ctx.host === nothing ||
        _jax_error("`", name, "` has no host (prepare_data) translation", _jax_where(ctx))
    length(args) >= 7 || _jax_error("`", name, "` takes f, y0, t0, ts, reltol, abstol, ",
                                    "maxsteps and the arguments of f", _jax_where(ctx))
    rhs, y0, t0, ts, rt, at, mx = args[1:7]
    extra = args[8:end]
    rhs.T isa _JaxCallable || rhs.T isa Type && rhs.T <: Function ||
        _jax_error("`", name, "`'s right-hand side is not a function", _jax_where(ctx))
    _jax_rank(y0.T) == 1 && isempty(y0.extent) ||
        _jax_error("`", name, "`'s initial state is a `", y0.T, "`, not a vector of ",
                   "one length", _jax_where(ctx))
    _jax_rank(ts.T) == 1 && !_jax_ragged(ts.T) ||
        _jax_error("`", name, "`'s output times are a `", ts.T, "`", _jax_where(ctx))
    for (what, v) in (("initial time", t0), ("relative tolerance", rt), ("absolute tolerance", at))
        v.T isa Type && v.T <: Real ||
            _jax_error("`", name, "`'s ", what, " is a `", v.T, "`", _jax_where(ctx))
    end
    mx.T isa Type && mx.T <: Integer ||
        _jax_error("`", name, "`'s maxsteps is a `", mx.T, "`", _jax_where(ctx))
    helper = _jax_helper!(ctx, "_rk_ode_solve", _JAX_ODE_SOLVE)
    ts = _jax_bind!(ctx, :ts, ts)
    n = isempty(ts.extent) ? ts.code * ".shape[0]" : ts.extent
    fun = _jax_ode_rhs!(ctx, _jax_function_of(rhs), extra)
    passed = [a for a in extra if !_jax_ode_closed(a)]
    solution = _jax_fresh(ctx, :solution)
    status = _jax_fresh(ctx, :status)
    _jax_emit!(ctx, solution * ", " * status * " = " * helper * "(" *
               join(["_RK" * _jax_ode_solver(f) * "()", fun, y0.code, t0.code, ts.code, n, rt.code, at.code,
                     mx.code, _jax_tuple_code(passed)], ", ") * ")")
    label = string(name, " (", _jax_ode_solver(f), ")")
    # Invalid inputs raise, as natively (`ArgumentError`); an exceeded budget
    # or a failed solve is a numerical failure (`ODESolveFailure` natively),
    # where a density function answers -inf.
    for (code, message) in ((1, " has an invalid initial state, output times, tolerances or maxsteps"),
                            (2, " exceeded maxsteps between requested outputs"),
                            (3, " failed"))
        _jax_masked_check(ctx, _PyVal(status * " != " * string(code), Bool), repr(label * message);
                          reject = code != 1)
    end
    _PyVal(solution, Matrix{Float64}, false, ts.extent, "", isempty(ts.extent) ? 0 : 1)
end

# An extra argument the right-hand side reads where it was called instead of
# through diffrax's `args`: a function or type, an aggregate of translated
# values, or a static value (a shape or literal the body may branch on).
_jax_ode_closed(v::_PyVal) = _jax_translation_only(v) || v.static ||
    (v.T isa Type && v.T <: Function)

# The nested Python function `def <f>_rhs(t, y, args)` computing `f(t, y,
# extra...)`, emitted in the current scope before its call.
function _jax_ode_rhs!(ctx::_JaxContext, f, extra::Vector{_PyVal})
    outer = ctx.scope
    fname = _jax_fresh(ctx, Symbol(nameof_or(f), :_rhs))
    inner = _JaxScope(outer)
    inner.indent = outer.indent + 1
    t = _jax_identifier(inner.used, :t)
    y = _jax_identifier(inner.used, :y)
    a = _jax_identifier(inner.used, :args)
    inputs = copy(extra)
    params = String[]
    for (k, v) in enumerate(extra)
        _jax_ode_closed(v) && continue
        p = _jax_identifier(inner.used, Symbol(:arg, k))
        push!(params, p)
        inputs[k] = _PyVal(p, v.T, false, v.extent, v.limit, v.padded, v.origin, v.field_extents)
    end
    saved = copy(ctx.locals)
    ctx.scope = inner
    dy = try
        isempty(params) || _jax_emit!(ctx, join(params, ", ") * (length(params) == 1 ? "," : "") *
                                           " = " * a)
        _jax_call(ctx, f, vcat(_PyVal[_PyVal(t, Float64), _PyVal(y, Vector{Float64})], inputs))
    finally
        ctx.scope = outer
        empty!(ctx.locals); merge!(ctx.locals, saved)
    end
    _jax_rank(dy.T) == 1 && isempty(dy.extent) ||
        _jax_error("the right-hand side `", nameof_or(f), "` returns a `", dy.T,
                   "`, not a state vector", _jax_where(ctx))
    push!(outer.lines, "    "^outer.indent * "def " * fname * "(" * t * ", " * y * ", " * a * "):")
    append!(outer.lines, inner.lines)
    push!(outer.lines, "    "^inner.indent * "return " * dy.code)
    union!(outer.used, inner.used)
    fname
end
