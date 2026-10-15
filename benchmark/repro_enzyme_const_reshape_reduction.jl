# Standalone native reverse reproducer; only Enzyme and Base are required.
# Run with both the default bounds mode and --check-bounds=yes.
#
# A Const vector is reshaped and read by both a constant row reduction and
# an active broadcast. Adding the constant reduction to the active one makes
# Enzyme 0.13.216 / Julia 1.13.1 / --check-bounds=yes report mixed activity
# at the pointer store in Base.reshape. Neither input is mutated. The
# control without the constant reduction differentiates correctly.
# This is the shape of a shared concentration in a Dirichlet slice density;
# no distribution implementation or ReactiveKernels code is needed here.
using Enzyme

function weighted_rows(X, a)
    A = reshape(a, 1, length(a))
    sum(sum((A .- 1) .* log.(X); dims=2))
end

function shifted_weighted_rows(X, a)
    A = reshape(a, 1, length(a))
    offset = sum(A; dims=2)
    sum(offset .+ sum((A .- 1) .* log.(X); dims=2))
end

function check_const_reshape(f, groups, width)
    X = reshape(collect(range(0.2, 0.8; length=groups * width)), groups, width)
    a = collect(range(1.5, 2.5; length=width))
    saved_X, saved_a = copy(X), copy(a)
    expected_gradient = (reshape(a, 1, width) .- 1) ./ X
    dX = zero(X)
    println("case=", f, " shape=", size(X), " primal=", f(X, a))
    flush(stdout)
    result = try
        autodiff(ReverseWithPrimal, f, Active, Duplicated(X, dX), Const(a))
    catch err
        err isa Enzyme.Compiler.EnzymeRuntimeActivityError || rethrow()
        @assert X == saved_X && a == saved_a
        println("UNSUPPORTED ", typeof(err))
        showerror(stdout, err)
        println()
        return false
    end
    @assert last(result) ≈ f(X, a)
    @assert dX ≈ expected_gradient
    @assert X == saved_X && a == saved_a
    println("PASS analytic gradient and unchanged inputs")
    true
end

println("Julia ", VERSION, "; Enzyme ", pkgversion(Enzyme),
    "; check_bounds=", Base.JLOptions().check_bounds)
success = true
for groups in (2, 5), width in (2, 3), f in (weighted_rows, shifted_weighted_rows)
    global success &= check_const_reshape(f, groups, width)
end
success || exit(1)
