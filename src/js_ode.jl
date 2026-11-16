# JavaScript translation of the native ODE bridge `rk_ode_rk45_tol`
# (`native_ode.jl`): one call of the vendored `@reside-ic/dopri` 1.0.1
# Dormand–Prince 5(4) integrator (`js_vendor/dopri.js`, `_rk_dopri` in
# `_JS_RUNTIME`; decision `1r7ufgm`, prong `solver`), the method native Julia
# runs as OrdinaryDiffEq's `DP5()`.
#
# The adapter `_rk_ode_rk45` (in `_JS_RUNTIME`, `js.jl`) does what RK's native
# adapter (`ReactiveKernelsSciMLODEExt._native_ode_tol`) does around its solver:
# - the same argument checks, with the same exception types;
# - one continuous solve on the normalized time s = (t - t0) / (T - t0) in
#   [0, 1], the right-hand side scaled by T - t0, the caller's tolerances
#   unchanged, steps no longer than the span, and a last step that lands on T;
# - outputs from the method's dense interpolant;
# - at most `maxsteps` accepted steps between consecutive outputs, counted
#   from t0;
# - `ReactiveKernels.ODESolveFailure` as a `JuliaError` of that name carrying
#   `retcode` and `t`: `MaxIters` for the budget, and `Unstable` where native's
#   step size shrinks below eps(t) (native has no step floor), which is also
#   where a NaN in the right-hand side ends natively.
#
# The library computes its own step sequence (its initial step and its rule of
# no growth right after a rejected step differ from OrdinaryDiffEq's), so
# values agree with native within the solvers' accuracy, not to rounding. The
# library members the adapter uses beyond its documented interface are listed
# in `js_vendor/README.md`.



# A runtime value read by a function the module defines: bound to a name first.
_js_closed!(ir, v::_JsVal, hint) = v.static ? v : _js_simple!(ir, v, hint)
_js_closed!(ir, v::_JsTuple, hint) = _JsTuple(Any[_js_closed!(ir, e, hint) for e in v.items], v.T)
_js_closed!(ir, v, hint) = v

function _js_function_call(ir::_JsIR, f::typeof(rk_ode_rk45_tol), args, T)
    mod = ir.fn.mod
    length(args) >= 7 ||
        _js_error(mod, "`rk_ode_rk45_tol` takes (f, y0, t0, ts, reltol, abstol, maxsteps, args...)")
    rhs, y0, t0, ts, rt, at, mx = args[1:7]
    numvec(v) = _js_isvector(v) && eltype(_js_type(v)) <: _JsNumber
    numvec(y0) || _js_error(mod, "the initial state of `rk_ode_rk45_tol` is a `", _js_type(y0),
                            "`; JavaScript needs a vector of numbers")
    numvec(ts) || _js_error(mod, "the output times of `rk_ode_rk45_tol` are a `", _js_type(ts),
                            "`; JavaScript needs a vector of numbers")
    for (what, v) in (("initial time", t0), ("relative tolerance", rt), ("absolute tolerance", at))
        _js_isnumber(v) || _js_error(mod, "the ", what, " of `rk_ode_rk45_tol` is a `", _js_type(v), "`")
    end
    _js_isint(mx) || _js_error(mod, "the step limit of `rk_ode_rk45_tol` is a `", _js_type(mx), "`")
    T === Matrix{Float64} || _js_error(mod, "internal: `rk_ode_rk45_tol` typed as `", T, "`")
    g = rhs isa _JsVal && rhs.static ? rhs.value : rhs isa _JsStruct ? rhs :
        _js_error(mod, "the right-hand side of `rk_ode_rk45_tol` is a runtime function value")
    _js_flush!(ir)
    # The right-hand side as a JavaScript function of (t, y), reading the extra
    # arguments by name, translated in a scope of its own.
    fn = ir.fn
    extra = Any[_js_closed!(ir, a, "arg") for a in args[8:end]]
    t, y = _js_local!(fn, "t"), _js_local!(fn, "y")
    inner = _JsFn(mod, String[], fn.indent + 1, copy(fn.taken))
    kir = _js_ir(inner, only(Base.code_lowered(_js_noop, Tuple{})), 0)
    rargs = Any[_js_rt(t, Float64; ref = true), _js_rt(y, Vector{Float64}; ref = true), extra...]
    RT = _js_return_type(g, Any[_js_argtype(a) for a in rargs])
    RT isa Type && RT <: Vector && eltype(RT) <: _JsNumber ||
        _js_error(mod, "the right-hand side `", g isa _JsStruct ? g.T : g, "` of `rk_ode_rk45_tol` returns a `", RT,
                  "` for (Float64, Vector{Float64}, ...); JavaScript needs a vector of numbers")
    push!(mod.stack, "the right-hand side of `rk_ode_rk45_tol`")
    dy = _js_call!(kir, g, rargs, RT)
    code = _js_runtime(mod, dy).code
    _js_flush!(kir)
    pop!(mod.stack)
    union!(fn.taken, inner.taken)
    name = _js_local!(fn, "rhs")
    _js_line!(fn, "const " * name * " = (" * t * ", " * y * ") => {")
    append!(fn.lines, inner.lines)
    push!(fn.lines, "  "^(fn.indent + 1) * "return " * code * ";")
    _js_line!(fn, "};")
    _js_fcall(ir, _js_use!(mod, :_rk_ode_rk45), Any[_js_rt(name, Nothing; ref = true), y0, t0, ts, rt, at, mx], T;
              pure = false)
end

_js_function_call(ir::_JsIR, ::typeof(rk_ode_bdf_tol), args, T) =
    _js_error(ir.fn.mod, "`rk_ode_bdf_tol` (FBDF) has no JavaScript translation yet")
