# Standalone CPU reproducer: only Enzyme and Test are required.
#
#     julia --project=<environment with Enzyme> benchmark/repro_enzyme_nested_overwrite.jl
#
# A custom reverse rule that reloads input storage in its reverse pass needs to
# know whether the caller wrote to that storage after the rule's call. The rule
# for `row` below asks `EnzymeRules.overwritten(config)`: it saves the weight on
# its tape when the flag of its operand is set, and reloads it from the operand
# otherwise. The caller `write_alias` overwrites the weight after the call,
# through a second argument that is the same array.
#
# With the array passed directly the flag is set, and the rule returns the
# weight the primal used (3). With the same array inside the immutable
# `Weights` the flag stays unset, so the rule reloads the overwritten weight and
# returns 42. Ordinary Enzyme through `native`, without the rule, returns 3 in
# both cases, and so does the rule when nothing is written.
#
# Recorded on a Linux fleet host, 2026-11-20: Julia 1.10.12, Enzyme dogfood
# `83fc0bfc`, libEnzyme `09013d21` (LLVM 15). 15 of 16 checks pass and
# "Weights, write_alias, row" fails with `42.0 == 3.0`, so the script exits
# nonzero. Enzyme snag `used-the-public-c34388a0` tracks whether the flag is
# meant to cover storage reached through a composite operand.
#
# This is why the Enzyme adapter of derivative rules
# (`_retain_residuals`, ext/ReactiveKernelsEnzymeExt.jl) borrows only arrays
# passed directly and copies every array held inside a composite input on each
# reverse call (snag `native-reverse-r-b8de62b9`). The module
# `CompositeInputChecks` in test/test_derivative_rule_snapshots.jl holds the
# same writes at a declared rule.
module NestedOverwrite
using Enzyme, Test
import Enzyme.EnzymeRules: augmented_primal, reverse, RevConfig, AugmentedReturn,
    needs_primal, overwritten

struct Weights
    values::Vector{Float64}
end
weight(g::Weights) = first(g.values)
weight(a::Vector) = first(a)
native(g, x) = weight(g) * x
row(g, x) = native(g, x)

function augmented_primal(config::RevConfig, ::Const{typeof(row)}, ::Type{<:Active},
        g::Const, x::Active)
    saved = overwritten(config)[2] ? weight(g.val) : nothing
    AugmentedReturn(needs_primal(config) ? row(g.val, x.val) : nothing, nothing, saved)
end
function reverse(config::RevConfig, ::Const{typeof(row)}, dy::Active, saved,
        g::Const, x::Active)
    w = overwritten(config)[2] ? saved : weight(g.val)
    (nothing, dy.val * w)
end

unchanged(f, g, alias, x) = f(g, x)
function write_alias(f, g, alias, x)
    y = f(g, x)
    alias[1] = 42.0
    y
end

@testset "reverse reads the weight used for the primal" begin
    for wrap in (identity, Weights), caller in (unchanged, write_alias), f in (native, row)
        @testset "$wrap, $caller, $f" begin
            a = [3.0]
            grads = only(autodiff(Reverse, Const(caller), Active,
                Const(f), Const(wrap(a)), Const(a), Active(2.0)))
            @test a[1] == (caller === write_alias ? 42.0 : 3.0)
            @test last(grads) == 3.0
        end
    end
end
end
