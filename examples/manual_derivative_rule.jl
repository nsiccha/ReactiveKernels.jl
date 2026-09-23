module ManualDerivativeRuleExample

using LinearAlgebra
using ReactiveKernels

export EXAMPLE_INPUTS
export PRIMAL_HAVE, PRIMAL_WANT, FORWARD_HAVE, FORWARD_WANT
export REVERSE_HAVE, REVERSE_WANT, X_REVERSE_HAVE, X_REVERSE_WANT
export matvec_rule, matvec
export matvec_primal, matvec_forward, matvec_reverse, matvec_reverse_x, run

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

# The graph's ports are named into roles once; the generator returns an
# RK-owned callable whose cuts and AD adapters all come from this one graph.
# -- BEGIN DOCS: generated rule --
const matvec = derivative_rule(matvec_rule; primal = :y,
    directions = (A = :A_dot, x = :x_dot), tangent = :y_dot,
    covector = :y_bar, cotangents = (A = :A_bar, x = :x_bar),
    name = :matvec)
# -- END DOCS: generated rule --

# The same HAVE→WANT boundaries, prepared explicitly so the documentation can
# render each cut's generated kernel and compute DAG.
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

const EXAMPLE_INPUTS = (
    A = [1.0 2.0; 3.0 4.0],
    x = [0.5, -1.0],
    A_dot = [0.1 -0.2; 0.3 0.4],
    x_dot = [-0.7, 0.2],
    y_bar = [1.2, -0.4],
)

"""Execute the generated rule's cuts and return inspectable acceptance evidence."""
function run(; inputs = EXAMPLE_INPUTS)
    (; A, x, A_dot, x_dot, y_bar) = inputs

    y = matvec(A, x)
    y_forward, y_dot = forward_cut(matvec, A, x, A_dot, x_dot)
    # Activity mask: bit `i` set when input `i` (here `A`, `x`) is active.
    A_bar, x_bar = reverse_cut(matvec, Val(3), A, x, y_bar)
    (x_only_bar,) = reverse_cut(matvec, Val(2), A, nothing, y_bar)

    (; y, y_forward, y_dot, A_bar, x_bar, x_only_bar,
       prepared = (
           y = matvec_primal(A, x),
           forward = matvec_forward(A, x, A_dot, x_dot),
           reverse = matvec_reverse(A, x, y_bar),
           x_reverse = matvec_reverse_x(A, y_bar),
       ),
       adjoint_pair = (
           dot(y_bar, y_dot),
           dot(A_bar, A_dot) + dot(x_bar, x_dot),
       ),
       residuals = (
           A_only = reverse_residuals(matvec, Val(1)),
           x_only = reverse_residuals(matvec, Val(2)),
           both = reverse_residuals(matvec, Val(3)),
       ),
       recipe_ids = (
           primal = Tuple(recipe.id for recipe in matvec_primal.plan.recipes),
           forward = Tuple(recipe.id for recipe in matvec_forward.plan.recipes),
           reverse = Tuple(recipe.id for recipe in matvec_reverse.plan.recipes),
           x_reverse = Tuple(recipe.id for recipe in matvec_reverse_x.plan.recipes),
       ))
end

end # module ManualDerivativeRuleExample
