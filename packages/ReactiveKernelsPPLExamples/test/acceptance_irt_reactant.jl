# Included only by acceptance_irt_four_axis.jl after `import Reactant`. Keeping
# these macro uses in a separate file ensures the native process never lowers a
# Reactant `@compile` expression, even behind a runtime conditional.
function _reactant_probe(kernel, prepared, stan_model, q, label)
    traced_q = Reactant.to_rarray(q)

    compiled_primal = Reactant.@compile sync = true kernel(traced_q)
    reactant_value = Float64(compiled_primal(traced_q))
    native_value = kernel(q)
    reactant_value_error = _relative_scalar(reactant_value, native_value)
    @assert isfinite(reactant_value)
    @assert reactant_value_error < REACTANT_VALUE_TOL
    println("  Reactant primal [", label, "] max_rel=",
            round(reactant_value_error; sigdigits = 4))

    traced_gradient = Reactant.to_rarray(similar(q))
    compiled_gradient = Reactant.@compile sync = true ReactiveKernels.ad_value_and_gradient!(
        prepared, traced_gradient, traced_q,
    )
    _, reactant_gradient = compiled_gradient(prepared, traced_gradient, traced_q)
    host_gradient = Array{Float64}(reactant_gradient)
    stan_gradient = _stan_gradient(stan_model, q)
    reactant_gradient_error = _relative_vector(host_gradient, stan_gradient)
    @assert all(isfinite, host_gradient)
    @assert reactant_gradient_error < REACTANT_GRAD_TOL
    println("  Reactant gradient [", label, "] max_rel=",
            round(reactant_gradient_error; sigdigits = 4))
    nothing
end

function _reactant_axes(case, kernel, prepared, stan_model, q, stress_q)
    _reactant_probe(kernel, prepared, stan_model, q, "ordinary probe")
    if stress_q !== nothing
        _reactant_probe(
            kernel, prepared, stan_model, stress_q, "probability-saturation stress",
        )
    end
    nothing
end
