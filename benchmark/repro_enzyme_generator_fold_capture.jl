# Standalone native-backend reproducer; requires only Enzyme and stdlib Test.
# Base's `sum` over a generator whose closure holds a constant array beside an
# active one. The fold (`mapfoldl` -> `foldl_impl`) takes the generator as an
# argument, and registered Enzyme rejects the store of the constant array into
# its closure with `EnzymeRuntimeActivityError`, as for `collect` in
# `repro_enzyme_generator_const_array_capture.jl` (issue #39, upstream
# EnzymeAD/Enzyme.jl#2386).
#
# Measured outcomes (the three objectives below):
# - Enzyme 0.13.216 / Enzyme_jll 0.0.303: `direct` fails on Julia 1.12.7 and
#   1.13.1 and passes on 1.10.12, whose inliner removes the closure;
#   `recurrence` and `captures` fail on 1.10.12, 1.12.7 and 1.13.1.
# - Enzyme dogfood 2690dbd8 / Enzyme_jll 0.0.301: all three pass on 1.10.12,
#   1.12.7 and 1.13.1.
# An `@rkppl` `@scan` step reading `sum(X[m, t, j] * alpha[j] for j in 1:P)` is
# this shape. The same fold written with `init` runs natively as RK's explicit
# loop, which has no generator closure.
module NativeGeneratorFoldRepro
using Enzyme, Test

direct(alpha, X, m, t) = sum(X[m, t, j] * alpha[j] for j in 1:size(X, 3))

# A recurrence whose step closure is called from a non-inlined driver.
@noinline function drive(step, init, n)
    carry = init
    for t in 2:n
        carry = step(carry, t)
    end
    carry
end
recurrence(alpha, X, mu, m, n) = drive(1.0, n) do carry, t
    carry * mu[m] * exp(-sum(X[m, t, j] * alpha[j] for j in 1:size(X, 3)))
end

# The step beside other captures and a tuple carry, as in a renewal equation.
captures(alpha, X, mu, pop, kernel, y, m, n) = drive((y[m], 0.0), n) do carry, t
    rt = mu[m] * exp(-sum(X[m, t, j] * alpha[j] for j in 1:size(X, 3)))
    convolution = sum(kernel[k] * carry[1] for k in 1:min(t, length(kernel)))
    ((pop[m] - carry[2]) / pop[m] * rt * convolution, carry[2] + carry[1])
end[1]

# Reverse gradient of `f(args...)` with respect to the arguments in `active`,
# or `nothing` when Enzyme raises the runtime-activity error.
function reverse_outcome(f, args, active)
    shadows = map(a -> a isa AbstractArray ? zero(a) : nothing, args)
    annotated = map(i -> i in active ? Duplicated(args[i], shadows[i]) : Const(args[i]),
                    Tuple(eachindex(args)))
    try
        autodiff(Reverse, f, Active, annotated...)
    catch err
        err isa Enzyme.Compiler.EnzymeRuntimeActivityError || rethrow()
        println("UNSUPPORTED ", f, ": ", nameof(typeof(err)))
        return nothing
    end
    Tuple(shadows[i] for i in active)
end

function central_differences(f, args, i)
    h = cbrt(eps(Float64))
    map(eachindex(args[i])) do k
        up, dn = map(copy, args), map(copy, args)
        up[i][k] += h
        dn[i][k] -= h
        (f(up...) - f(dn...)) / (2h)
    end
end

function run_reproducer()
    println("Julia=", VERSION, " Enzyme=", pkgversion(Enzyme))
    X = reshape([0.1 * sin(i) for i in 1:(3 * 20 * 6)], 3, 20, 6)
    alpha = [0.2, 0.1, 0.3, 0.05, 0.15, 0.25]
    mu = [1.1, 0.9, 1.3]
    pop = [1e4, 2e4, 3e4]
    kernel = [exp(-k / 3) for k in 1:20]
    y = [0.5, 0.7, 0.9]
    # `direct` differentiates on Julia 1.10, where the fold is inlined.
    cases = ((direct, (alpha, X, 2, 5), (1,), VERSION >= v"1.11-"),
             (recurrence, (alpha, X, mu, 2, 20), (1, 3), true),
             (captures, (alpha, X, mu, pop, kernel, y, 2, 20), (1, 3, 6), true))
    @testset "native generator fold constant-array capture" begin
        for (f, args, active, gapped) in cases
            snapshot = deepcopy(args)
            outcome = reverse_outcome(f, args, active)
            @test isequal(args, snapshot)
            expected = Tuple(central_differences(f, args, i) for i in active)
            exact = outcome !== nothing &&
                all(isapprox.(outcome, expected; rtol = 1e-6, atol = 1e-9))
            # Capability gap, not a refusal: an upstream lift is an
            # Unexpected Pass here.
            gapped ? (@test_broken exact) : (@test exact)
        end
    end
end
end

if abspath(PROGRAM_FILE) == @__FILE__
    NativeGeneratorFoldRepro.run_reproducer()
end
