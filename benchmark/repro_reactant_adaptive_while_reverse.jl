# Standalone CPU reproducer: only Reactant and Enzyme are required.
#
# Enzyme reverse THROUGH a `@trace while` loop whose exit is data dependent
# does not lower. This is the loop shape of an adaptive ODE solver
# (`packages/ReactiveKernelsReactantODESolvers`): integrate until the span
# is covered, bounded by an iteration cap. The primal compiles and runs; the
# reverse pass fails inside Reactant's Enzyme lowering because the `while`
# op has no statically known iteration count. Recorded failure (strato2,
# Reactant 0.2.284 / Enzyme 0.13, 2026-09-22):
#   CompilationError: MLIR pass pipeline "all" failed
#   error: WhileOp does not have known iteration count for cache removal
#   (raised from Reactant/src/ControlFlow.jl while_loop)
using Reactant, Enzyme

function integrate(p)
    z = sum(p .* 0.0)
    t = z + 0.0
    n = z + 0.0
    x = p .* 0.0 .+ 1.0
    Reactant.@trace track_numbers = false while (n < 1000.0) & (t < 1.0)
        dt = min(z + 0.3, 1.0 - t)
        x = x .+ dt .* (p .* x)
        t = t + dt
        n = n + 1.0
    end
    sum(x)
end

p = [0.5, 1.5]
rp = Reactant.to_rarray(p)
primal = Reactant.@compile sync = true integrate(rp)
@assert Float64(primal(rp)) ≈ integrate(p)
# Closed form of the four steps (0.3, 0.3, 0.3, 0.1): x_i = (1 + 0.3 p_i)^3 (1 + 0.1 p_i).
expected = [0.9 * (1 + 0.3pi)^2 * (1 + 0.1pi) + 0.1 * (1 + 0.3pi)^3 for pi in p]
gradient(p) = only(Enzyme.gradient(Enzyme.Reverse, integrate, p))
compiled_gradient = Reactant.@compile sync = true gradient(rp)   # fails here
@assert Array(compiled_gradient(rp)) ≈ expected
