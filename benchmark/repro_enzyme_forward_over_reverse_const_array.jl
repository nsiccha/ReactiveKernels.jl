# Standalone CPU reproducer: only Enzyme is required.
#
# Native Enzyme forward mode over a native reverse-mode gradient (static
# activity analysis, the default) fails when the differentiated function
# multiplies a constant matrix argument by the active vector, as a design
# matrix product `A * β` does:
#   EnzymeRuntimeActivityError: Detected potential need for runtime activity.
#   Constant memory is stored (or returned) to a differentiable variable
# It fails for `A * view(u, 1:3)` and `A * u[1:3]` alike. Every value is valid
# and the second derivative is defined: reverse mode over forward mode
# differentiates both, and so does forward mode with `set_runtime_activity`.
# A function that reads a constant vector only elementwise into a fused
# reduction (`sum(c .* exp.(u))`) passes forward over reverse.
#
# ReactiveKernels' Hessian-vector products (`prepare_ad_hvp`) inherit this:
# an RKPPL program whose linear predictor lowers to a bound design matrix
# times the coefficient view fails under forward over reverse and passes
# under reverse over forward (`test/test_ad_tangent_operators.jl`).
# Recorded on gordito, Enzyme 0.13.210, Enzyme_jll 0.0.301,
# DifferentiationInterface 0.7.21, Julia 1.10.12, 2026-10-09.
using Enzyme

const A = [1.0 0.5 0.0; 1.0 0.0 0.5; 1.0 1.0 1.0; 1.0 -1.0 2.0]

blas_view(u, A) = sum(abs2, A * view(u, 1:3))
blas_copy(u, A) = sum(abs2, A * u[1:3])
elementwise(u, c) = sum(c .* exp.(u))

function gradient!(du, f, u, A)
    fill!(du, 0.0)
    autodiff(Reverse, f, Active, Duplicated(u, du), Const(A))
    nothing
end

function forward_over_reverse(f, u, v, A; mode = Forward)
    du, hv = zeros(length(u)), zeros(length(u))
    autodiff(mode, gradient!, Const, Duplicated(du, hv), Const(f),
             Duplicated(u, v), Const(A))
    hv
end

directional(u, v, f, A) = autodiff(Forward, f, Duplicated(u, v), Const(A))[1]

function reverse_over_forward(f, u, v, A)
    hv = zeros(length(u))
    autodiff(Reverse, directional, Active, Duplicated(u, hv), Const(v),
             Const(f), Const(A))
    hv
end

function run(label, g)
    result = try
        g()
        "ok"
    catch err
        "$(nameof(typeof(err)))"
    end
    println(rpad(label, 60), result)
end

u, v = [0.1, 0.2, -0.3, 0.4], ones(4)
c = [2.0, -1.0, 0.5, 1.5]
println("Julia $VERSION; expected ok everywhere, failures are the boundary")
for (name, f) in (("A * view(u, 1:3)", blas_view), ("A * u[1:3]", blas_copy))
    run("forward over reverse: $name", () -> forward_over_reverse(f, u, v, A))
    run("forward (runtime activity) over reverse: $name",
        () -> forward_over_reverse(f, u, v, A;
                                   mode = set_runtime_activity(Forward)))
    run("reverse over forward: $name", () -> reverse_over_forward(f, u, v, A))
end
run("forward over reverse: sum(c .* exp.(u))",
    () -> forward_over_reverse(elementwise, u, v, c))
