module ManualDerivativeRuleExample

using LinearAlgebra
using ReactiveKernels

export EXAMPLE_INPUTS
export PRIMAL_HAVE, PRIMAL_WANT, FORWARD_HAVE, FORWARD_WANT
export REVERSE_HAVE, REVERSE_WANT, X_REVERSE_HAVE, X_REVERSE_WANT
export matvec_rule, matvec_primal, matvec_forward, matvec_reverse
export matvec_reverse_x, MatvecPullback, MatvecXPullback
export value_and_pullback, value_and_x_pullback, run

# One graph owns the primal, JVP, and VJP mathematics. Numerical directions and
# covectors are ordinary graph values; no AD package's protocol types appear.
# -- BEGIN DOCS: pure mathematical derivative rule --
@kernel matvec_rule(
        A::Matrix{Float64}, x::Vector{Float64},
        A_dot::Matrix{Float64}, x_dot::Vector{Float64},
        y_bar::Vector{Float64}) = begin
    y::Vector{Float64} = A * x

    A_direction::Vector{Float64} = A_dot * x
    x_direction::Vector{Float64} = A * x_dot
    y_dot::Vector{Float64} = A_direction + x_direction

    A_bar::Matrix{Float64} = y_bar * transpose(x)
    x_bar::Vector{Float64} = transpose(A) * y_bar

    return y, y_dot, A_bar, x_bar
end
# -- END DOCS: pure mathematical derivative rule --

const PRIMAL_HAVE = (:A, :x)
const PRIMAL_WANT = :y
const FORWARD_HAVE = (:A, :x, :A_dot, :x_dot)
const FORWARD_WANT = (:y, :y_dot)
const REVERSE_HAVE = (:A, :x, :y_bar)
const REVERSE_WANT = (:A_bar, :x_bar)
const X_REVERSE_HAVE = (:A, :y_bar)
const X_REVERSE_WANT = :x_bar

const matvec_primal = prepare(
    matvec_rule; have = PRIMAL_HAVE, want = PRIMAL_WANT)
const matvec_forward = prepare(
    matvec_rule; have = FORWARD_HAVE, want = FORWARD_WANT)
const matvec_reverse = prepare(
    matvec_rule; have = REVERSE_HAVE, want = REVERSE_WANT)
const matvec_reverse_x = prepare(
    matvec_rule; have = X_REVERSE_HAVE, want = X_REVERSE_WANT)

# These backend-neutral structs show the staging shape that generated rule
# adapters would own. They are deliberately example code, not a package API.
# -- BEGIN DOCS: generated-style pullback staging --
struct MatvecPullback{K,TA,TX}
    vjp::K
    A::TA
    x::TX
end

function (pullback::MatvecPullback)(y_bar)
    pullback.vjp(pullback.A, pullback.x, y_bar)
end

struct MatvecXPullback{K,TA}
    vjp::K
    A::TA
end

function (pullback::MatvecXPullback)(y_bar)
    pullback.vjp(pullback.A, y_bar)
end

function value_and_pullback(A, x)
    y = matvec_primal(A, x)
    y, MatvecPullback(matvec_reverse, A, x)
end

function value_and_x_pullback(A, x)
    y = matvec_primal(A, x)
    y, MatvecXPullback(matvec_reverse_x, A)
end
# -- END DOCS: generated-style pullback staging --

const EXAMPLE_INPUTS = (
    A = [1.0 2.0; 3.0 4.0],
    x = [0.5, -1.0],
    A_dot = [0.1 -0.2; 0.3 0.4],
    x_dot = [-0.7, 0.2],
    y_bar = [1.2, -0.4],
)

"""Execute every documented cut and return its inspectable acceptance evidence."""
function run(; inputs = EXAMPLE_INPUTS)
    (; A, x, A_dot, x_dot, y_bar) = inputs

    y = matvec_primal(A, x)
    y_forward, y_dot = matvec_forward(A, x, A_dot, x_dot)
    A_bar, x_bar = matvec_reverse(A, x, y_bar)

    y_reverse, pullback = value_and_pullback(A, x)
    pullback_A_bar, pullback_x_bar = pullback(y_bar)
    y_x_only, x_pullback = value_and_x_pullback(A, x)
    x_only_bar = x_pullback(y_bar)

    (; y, y_forward, y_reverse, y_x_only, y_dot, A_bar, x_bar,
       pullback_A_bar, pullback_x_bar, x_only_bar,
       adjoint_pair = (
           dot(y_bar, y_dot),
           dot(A_bar, A_dot) + dot(x_bar, x_dot),
       ),
       recipe_ids = (
           primal = Tuple(recipe.id for recipe in matvec_primal.plan.recipes),
           forward = Tuple(recipe.id for recipe in matvec_forward.plan.recipes),
           reverse = Tuple(recipe.id for recipe in matvec_reverse.plan.recipes),
           x_reverse = Tuple(recipe.id for recipe in matvec_reverse_x.plan.recipes),
       ),
       captured_fields = (
           both = fieldnames(typeof(pullback)),
           x_only = fieldnames(typeof(x_pullback)),
       ))
end

end # module ManualDerivativeRuleExample
