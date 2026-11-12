# Standalone CPU reproducer: only Enzyme is required.
#
# Native Enzyme reverse mode (static activity analysis, the default) rejects a
# lazy column concatenation of an active vector and a constant vector when the
# concatenation is passed by reference to a function Julia does not inline,
# which reads every entry into a fresh table:
#   EnzymeRuntimeActivityError: Detected potential need for runtime activity.
#   Constant memory is stored (or returned) to a differentiable variable
#   Mismatched activity for: store ptr addrspace(10) %"absent::Array", ...
# The flagged store writes the constant vector into the concatenation's stack
# slot beside the active one. Every entry is read, never written, and the
# derivative is defined; static analysis cannot prove the constant field
# inactive once the slot escapes into the call.
# `set_runtime_activity(Reverse)` differentiates it.
#
# This is the shape ReactiveKernels 0.1.203's indexed-invariant tables take
# for a whole-event mixture whose component matrix is `hcat(p, absent)` with
# `p` a parameter and `absent` bound data (PosteriorDB `Mt_data-Mt_model`):
# `_indexed_map(Val(formula), _cell_array_hcat(p, absent))` builds the
# `log(p[j,k])` and `log1p(-p[j,k])` tables before the event loop. The same
# concatenation read inline inside the event loop (the lowering before 0.1.203)
# differentiates.
#
# Enzyme dogfood repairs this class (mixed-activity rule 4 and the
# constant-field proof, `enzyme-use`); registered releases do not. Upstream
# report: Enzyme todo `0dfpaar` (a `Const` array in a struct passed by reference
# beside active data).
#
# Recorded on gordito, Julia 1.13.1, registered Enzyme 0.13.219 (Enzyme_jll
# 0.0.307) and Enzyme dogfood 1ed42fb (libEnzyme eea6a012), 2026-11-12.
using Enzyme

struct LazyHcat{A}
    arguments::A
    rows::Int
    columns::Int
end
# The dense operand load stays inside its own call, as in ReactiveKernels.
@noinline entry(a::Vector{Float64}, i::Int) = a[i]
@inline hcat_entry(args::Tuple, i, j) =
    j <= 1 ? entry(first(args), i) : hcat_entry(Base.tail(args), i, j - 1)
@inline hcat_entry(::Tuple{}, i, j) = throw(BoundsError())
@inline getentry(h::LazyHcat, k::Int) = hcat_entry(h.arguments, mod1(k, h.rows), div(k - 1, h.rows) + 1)

function table(f::F, h::LazyHcat) where {F}
    out = Vector{Float64}(undef, h.rows * h.columns)
    @inbounds for k in eachindex(out)
        out[k] = f(getentry(h, k))
    end
    out
end

function tables(p, absent)
    h = LazyHcat((p, absent), length(p), 2)
    present = table(log, h)        # log(p[j,k]); -Inf for the absent column
    missed = table(x -> log1p(-x), h)
    s = 0.0
    for k in eachindex(present)
        s += (isfinite(present[k]) ? present[k] : 0.0) + 2 * missed[k]
    end
    s
end

Enzyme.Compiler.VERBOSE_ERRORS[] = true
p = [0.3, 0.6, 0.45]
absent = zeros(3)
expected = 1 ./ p .- 2 ./ (1 .- p)
println("Enzyme ", pkgversion(Enzyme))
for (label, mode) in (("static Reverse", Reverse), ("runtime activity", set_runtime_activity(Reverse)))
    dp = zero(p)
    try
        autodiff(mode, Const(tables), Active, Duplicated(p, dp), Const(absent))
        println(label, ": max |gradient - closed form| = ", maximum(abs.(dp .- expected)))
    catch e
        lines = split(sprint(showerror, e), '\n')
        println(label, ": ", typeof(e).name.name, "\n    ",
            something(findfirst(l -> occursin("Mismatched activity", l), lines) |>
                k -> k === nothing ? nothing : lines[k], first(lines)))
    end
end
