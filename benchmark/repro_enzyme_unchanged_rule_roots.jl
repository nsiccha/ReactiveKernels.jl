# Standalone CPU reproducer: only Enzyme is required.
#
# A custom reverse rule reads an unchanged `Const` tuple that holds array
# references, here `(Vector{Int}, Vector{Int}, Int, Bool)`, from its arguments
# in the reverse pass. `EnzymeRules.overwritten(config)` reports every operand
# unchanged, so the rule keeps nothing on its tape. On Julia 1.13.1, Enzyme
# before dogfood repair `unchanged_custom_rule_roots` (dogfood `c036eb1`) hands
# the rule the caller's root slots with conflicting alias metadata, LLVM
# forwards their initial nulls, and the first read of the tuple's first vector
# in the reverse rule segfaults the process. Enzyme's regression keeps the two
# rule calls behind a branch, noting that one straight-line call did not expose
# the conflict; this script keeps that shape.
#
# This is the shape of a generated plate reverse scheduled by index runs
# (src/plate_reverse_runs.jl): its run metadata is such a tuple, and a plate
# that takes two passes reads it in the reverse pass. The Enzyme extension
# therefore leaves such a plate to ordinary reverse unless
# `Enzyme.has_dogfood_repair(:unchanged_custom_rule_roots)` (Enzyme snag
# `readonly-rule-tu-14b643ff`). Enzyme's own regression for the repair is
# `test/rules/unchanged_roots.jl` on its dogfood line, from which this script
# is reduced. Recorded on a Linux fleet host, 2026-11-14 (fleet job
# `d8fbf5570bfe4a1b`): on Julia 1.13.1, dogfood `f025967` dies with signal 11 in
# `metadata_reverse` (its buffered output is lost with it) and dogfood `c036eb1`
# prints the expected values; on Julia 1.10.12, dogfood `a8f7301` (no repair,
# and no inline roots before Julia 1.12) prints the expected values.
using Enzyme
using Enzyme.EnzymeCore: EnzymeRules

struct MetadataRule{Large} end

@noinline function (::MetadataRule{Large})(x, y, a, ax, metadata, b) where {Large}
    return (Large ? 2 : 1) * a * sum(metadata[1]) + sum(abs2, x) + sum(y) + b
end

@noinline function metadata_reverse(x, y, a, ax, metadata, b, seed, shadow)
    shadow .+= (2seed) .* x
    return seed * sum(metadata[1])
end

@generated argument_values(args::Tuple) =
    Expr(:tuple, [:(getfield(args, $i).val) for i in 1:length(args.parameters)]...)

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig, fn::Const{MetadataRule{Large}},
        ::Type{RT}, args::Vararg{Annotation,N}) where {Large,RT,N}
    primal = fn.val(argument_values(args)...)
    # Nothing is retained: the tuple and its arrays are unchanged.
    return EnzymeRules.AugmentedReturn(
        EnzymeRules.needs_primal(config) ? primal : nothing, nothing, nothing)
end

function EnzymeRules.reverse(
        ::EnzymeRules.RevConfig, ::Const{MetadataRule{Large}},
        dret::Active, tape, args::Vararg{Annotation,N}) where {Large,N}
    partial = metadata_reverse(argument_values(args)..., dret.val, args[1].dval)
    return (nothing, nothing, (Large ? 2 : 1) * partial, nothing, nothing, nothing)
end

@noinline function objective(x, y, a, v, w, n)
    vx = view(x, v)
    metadata = (v, w, n, true)
    if n > 100
        return MetadataRule{true}()(vx, y, a, axes(y), metadata, 0.0)
    else
        return MetadataRule{false}()(vx, y, a, axes(y), metadata, 0.0)
    end
end

println("Julia ", VERSION, ", Enzyme ", pkgdir(Enzyme))
println("repair reported: ", isdefined(Enzyme, :has_dogfood_repair) &&
    Enzyme.has_dogfood_repair(:unchanged_custom_rule_roots))
println("expected: d/da = 9.0 then 18.0, dx = [2.0, 0.0, 6.0, 0.0, 10.0, 0.0]")
for n in (3, 101)
    x = Float64[1, 2, 3, 4, 5, 6]
    dx = zeros(6)
    result = autodiff(Reverse, objective, Active, Duplicated(x, dx), Const(Float64[2, 3, 4]),
                      Active(0.7), Const([1, 3, 5]), Const([2, 4, 6]), Const(n))
    println("n = ", n, ": d/da = ", result[1][3], ", dx = ", dx)
end
