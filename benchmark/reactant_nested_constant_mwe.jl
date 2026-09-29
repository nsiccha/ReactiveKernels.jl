# Backend-only reproducer (no ReactiveKernels import): Reactant's MLIR constant
# fallback recurses without termination on arrays whose `collect` never
# becomes a dense Number array.
#
# Observed on Reactant 0.2.289 (git tree f32f004d, ReactantCore 0.1.23,
# Julia 1.10.11, x86_64-linux and ARM64): promoting a nested host vector
# inside a traced region overflows the stack with ~40k identical
# `Reactant.Ops.constant` frames and no usable signal:
#
#   julia --project=<env-with-Reactant> benchmark/reactant_nested_constant_mwe.jl
#
# Expected output before the guard:  `MWE REPRODUCED: StackOverflowError (...)`.
# Expected output after the guard below: `MWE GUARDED: <loud error>`.
#
# Root cause (`Reactant/src/Ops.jl`, `constant(x::AbstractArray)`):
# `constant(collect(x))` assumes `collect` densifies. For
# `Vector{Vector{Int}}`, `collect` is a same-type copy, so the fallback calls
# itself forever. `collect` preserves the element type in general, so any
# non-Number element type diverges the same way.
#
# Local guard under test (NOT published; see the snag decision for the draft
# PR — do not file upstream from this reproducer alone):
#
#   @noinline function constant(
#       x::AbstractArray{T,N}; location=mlir_stacktrace("constant", @__FILE__, @__LINE__)
#   ) where {T,N}
#       @assert !(x isa TracedRArray)
#       collected = collect(x)
#       collected isa DenseArray{<:Number} && return constant(collected; location)
#       error(
#           "Cannot lower a constant of type $(typeof(x)) to MLIR: only dense " *
#           "arrays of numbers are supported " *
#           "(collected element type $(eltype(collected))).",
#       )
#   end
#
# The guard recurses exactly when the current code terminates (a dense Number
# collection reaches the dense method) and errors exactly when the current
# code diverges, so it is behavior-preserving apart from replacing the silent
# overflow with a loud error.
using Reactant

function main()
    r = Reactant.to_rarray([1.0, 2.0])
    nested = [[1, 2], [3, 4]]
    f = x -> (Reactant.promote_to(Reactant.TracedRArray, nested); x .+ 1.0)
    try
        hlo = repr(Reactant.@code_hlo f(r))
        println("MWE UNEXPECTED PASS: traced $(ncodeunits(hlo)) bytes")
        return 2
    catch e
        if e isa StackOverflowError
            frames = try
                length(stacktrace(catch_backtrace()))
            catch
                -1
            end
            println("MWE REPRODUCED: StackOverflowError ($frames frames)")
            return 0
        elseif e isa ErrorException &&
               occursin("only dense arrays of numbers", sprint(showerror, e))
            println("MWE GUARDED: $(sprint(showerror, e))")
            return 0
        else
            println("MWE UNEXPECTED ERROR: $(typeof(e)): $(sprint(showerror, e))")
            return 1
        end
    end
end

exit(main())
