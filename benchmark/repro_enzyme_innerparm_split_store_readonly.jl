# Standalone CPU reproducer: only Enzyme (and the LinearAlgebra stdlib) is
# required.
#
# Enzyme treats an unannotated function argument as `Const`. When the
# argument's type can hold derivative data, Enzyme first tries to prove the
# function never writes it (`err_if_func_written` in Enzyme's compiler).
# That proof rejects a read-only callable holding a matrix beside a
# `UniformScaling{Float64}` whose body adds the two with Base's
# `Matrix + UniformScaling`:
#   EnzymeMutabilityException: Function argument passed to autodiff cannot
#   be proven readonly. ... The potentially writing call is
#   store double %.fca.0.1.0.extract, ptr %.fca.0.1.0.gep, ..., using
#   %.fca.0.1.0.gep = getelementptr inbounds nuw i8,
#   ptr %"c::Shifted.innerparm", i64 8
# The flagged store does not come from the callable's body. `innerparm` is
# the stack slot Enzyme's calling-convention wrapper creates for a function
# argument passed by value; the wrapper stores the incoming value there and
# passes its address on. LLVM splits that aggregate store into one store per
# field (the `.fca.<path>.extract` / `.fca.<path>.gep` names), and the
# readonly proof reports the store of the `UniformScaling`'s `Float64` field
# into Enzyme's own copy as a write to the argument.
# `Const(f)` (Enzyme's own remedy for a function argument that holds no
# derivative data) and passing the same data as a `Const` argument
# differentiate exactly. The same captured fields pass when the body builds no
# matrix (`sum(A) * x^2 + n * s.λ`), so holding them is not enough by itself.
# A body that does `x^2 .* A` and reads `s.λ` itself hits the separate
# loop-store readonly gap instead (`repro_enzyme_closure_float_store_readonly.jl`),
# which Enzyme dogfood also repairs.
# ReactiveKernels reaches this when partial evaluation folds a bound-only
# `UniformScaling`, such as a covariance jitter `K = C + 1e-10 * I`, into the
# operation table of the callable it hands DifferentiationInterface: plain
# `AutoEnzyme(; mode = Enzyme.Reverse)` then raises this exception on Julia
# 1.12.7 and 1.13.1 (a latent Gaussian process differentiates exactly on
# 1.10.12), while
# `AutoEnzyme(; mode = Enzyme.Reverse, function_annotation = Enzyme.Const)`
# differentiates the same model exactly.
# Recorded on gordito, Julia 1.10.12, 1.12.7 and 1.13.1 with Enzyme 0.13.216,
# and Julia 1.13.1 with Enzyme dogfood `3ede25b` (0.13.215), 2026-10-14: the
# first row raises. Enzyme dogfood from `4c73c40c` (0.13.215) repairs the
# proof: every row is exact on Julia 1.13.1, 2026-10-15.
using Enzyme, LinearAlgebra

struct Shifted{T}
    data::T
end
function (c::Shifted)(x)
    A, s = c.data
    sum(x^2 .* A + s)
end

struct ShiftedField{T}
    data::T
end
function (c::ShiftedField)(x)
    A, s = c.data
    sum(A) * x^2 + size(A, 1) * s.λ
end

A = [1.0 0.5; 0.5 1.0]
x = 1.5
expected = 2x * sum(A)

row(label, f) = println(rpad(label, 52), try
    derivatives, value = f()
    d = first(derivatives)
    string("d/dx = ", d, ", value = ", value,
           isapprox(d, expected; rtol = 1e-12) ? "" : "   WRONG")
catch err
    s = sprint(showerror, err)
    i = findfirst("The potentially writing call is", s)
    i === nothing ? string(nameof(typeof(err))) :
        string(nameof(typeof(err)), ": ", first(split(s[first(i):end], "\n")))
end)

println(VERSION, ", Enzyme ", pkgversion(Enzyme))
println("expected d/dx = ", expected)
row("captured, Base Matrix + UniformScaling",
    () -> autodiff(ReverseWithPrimal, Shifted((A, 1e-10I)), Active, Active(x)))
row("Const(f), Base Matrix + UniformScaling",
    () -> autodiff(ReverseWithPrimal, Const(Shifted((A, 1e-10I))), Active, Active(x)))
row("Const data argument, Base Matrix + UniformScaling",
    () -> autodiff(ReverseWithPrimal, (x, d) -> Shifted(d)(x), Active, Active(x),
                   Const((A, 1e-10I))))
row("captured, same fields, no matrix result",
    () -> autodiff(ReverseWithPrimal, ShiftedField((A, 1e-10I)), Active, Active(x)))
