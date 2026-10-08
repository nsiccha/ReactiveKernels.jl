# Standalone CPU reproducer: only Enzyme is required.
#
# Enzyme treats an unannotated function argument as `Const`. When the
# argument holds mutable data, such as a closure capturing an array, Enzyme
# first tries to prove the function never writes it (`err_if_func_written`
# in Enzyme's compiler). On Julia 1.12 and 1.13 that proof rejects ordinary
# loops that only READ the captured array:
#   EnzymeMutabilityException: Function argument passed to autodiff cannot
#   be proven readonly. ... The potentially writing call is
#   store double %9, ptr addrspace(13) %memoryref_data54, ..., using
#   %9 = fadd contract double %value_phi29, %8
# The proof walks every value derived from the argument. A Float64 read from
# the captured array at a loop-dependent offset (`load double` through a
# `getelementptr i8` off the array's data pointer) gets no Julia type, so the
# walk follows it through the arithmetic. It then calls the store of that
# Float64 into the fresh result a capture of the argument. A floating-point
# value cannot hold a reference, so the store cannot write the argument.
# Enzyme emits the error at the store, so it is raised only when the store
# runs: a loop that runs no iteration, or a first element stored outside the
# loop, passes. In another shape (`fixed_buffer_inbounds`) no error is
# raised, but the primal and both derivatives come back as 0.0.
# On Julia 1.12 and 1.13, `Const(f)` (Enzyme's own remedy for a function
# argument that holds no derivative data) and passing the array as an explicit
# `Const` argument differentiate every case exactly. Julia 1.10.12 passes
# every captured case except `mapped`, which fails there in all three forms:
# that is the separate Julia 1.10 generator-capture boundary
# (`repro_enzyme_generator_const_array_capture.jl`).
# An Enzyme whose readonly proof does not treat the store of a floating-point
# value as a capture passes every case; a store into the argument, such as
# `xs[i] = xs[i - 1] * g`, still raises.
# ReactiveKernels' `prepare_ad` passes data as `Constant` contexts and is
# unaffected; a closure capturing data around an RK scan op or a prepared
# scan kernel fails in the same way.
# Recorded on gordito, Enzyme 0.13.210 and 0.13.213, Julia 1.12.7 and 1.13.1
# (Julia 1.10.12 with Enzyme 0.13.210), 2026-10-08.
using Enzyme

# A peeled recurrence, the shape of an authored scan.
@inline function peeled(xs, g)
    r = similar(xs)
    isempty(xs) && return r
    c = xs[1] * g
    r[1] = c
    for i in 2:length(xs)
        c = muladd(xs[i], g, c)
        r[i] = c
    end
    r
end
@inline function pointwise(xs, g)
    r = similar(xs)
    for i in eachindex(xs)
        r[i] = xs[i] * g
    end
    r
end
broadcasted(xs, g) = xs .* g
mapped(xs, g) = map(x -> x * g, xs)
@inline function fixed_buffer_inbounds(xs, g)
    r = Vector{Float64}(undef, 8)
    fill!(r, 0.0)
    @inbounds for i in eachindex(xs)
        r[i] = xs[i] * g
    end
    r
end

# How the array reaches Enzyme: captured by an unannotated closure, captured
# by a closure annotated `Const`, or as a `Const` argument.
differentiate(kernel, xs, ::Val{:captured}) =
    autodiff(ReverseWithPrimal, g -> sum(kernel(xs, g)), Active, Active(0.7))
differentiate(kernel, xs, ::Val{:const_closure}) =
    autodiff(ReverseWithPrimal, Const(g -> sum(kernel(xs, g))), Active, Active(0.7))
differentiate(kernel, xs, ::Val{:const_argument}) =
    autodiff(ReverseWithPrimal, (g, data) -> sum(kernel(data, g)), Active, Active(0.7),
             Const(xs))

function row(label, kernel, xs, expected, form)
    result = try
        derivatives, value = differentiate(kernel, xs, Val(form))
        d = first(derivatives)
        exact = isapprox(d, expected; atol = 1e-12) &&
                isapprox(value, sum(kernel(xs, 0.7)); atol = 1e-12)
        string("d/dg = ", d, ", value = ", value, exact ? "" : "   WRONG")
    catch err
        string(nameof(typeof(err)))
    end
    println(rpad("$label, $form", 40), rpad("n = $(length(xs))", 8), result)
end

println(VERSION, ", Enzyme ", pkgversion(Enzyme))
println("expected d/dg: sum(cumsum(xs)) for `peeled`, sum(xs) otherwise")
for xs in (sin.(1:1), sin.(1:5))
    for (label, kernel, expected) in (
            ("peeled", peeled, sum(cumsum(xs))),
            ("pointwise", pointwise, sum(xs)),
            ("broadcasted", broadcasted, sum(xs)),
            ("mapped", mapped, sum(xs)),
            ("fixed_buffer_inbounds", fixed_buffer_inbounds, sum(xs)))
        for form in (:captured, :const_closure, :const_argument)
            row(label, kernel, xs, expected, form)
        end
    end
end
