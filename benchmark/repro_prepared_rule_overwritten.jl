# Prepared-program reproducer: ReactiveKernels and dogfood Enzyme.
#
#     julia --project=<environment with ReactiveKernels and Enzyme dogfood> \
#         benchmark/repro_prepared_rule_overwritten.jl
#
# A derivative rule reads two numbers from `Geometry`, a struct that holds a
# vector of `n` numbers, and returns a two-element vector. With Enzyme's
# dogfood repair `overwritten_reachable_data`, the rule's Enzyme adapter keeps
# the struct's vector in place when `EnzymeRules.overwritten` proves it
# unchanged, and copies it otherwise (snag `native-reverse-r-b8de62b9`).
#
# The script prints the bytes per call at two sizes of the vector, and checks
# the derivatives, for four callers:
# - `direct`: Enzyme on a function that sums two rule calls;
# - `prepared`: a prepared RK program whose plate cells hold `sum(row(...))`;
# - `stored`: the same program with cells that hold the rule's vector itself;
# - `pullback`: a prepared pullback (`prepare_ad_pullback`) of the rule's
#   vector, one `ad_pullback` per call.
#
# Recorded on a Linux fleet host, 2026-11-20: Julia 1.10.12, ReactiveKernels
# 0.1.351, Enzyme dogfood `b862f996` with libEnzyme `400c1bdc` (repairs
# `overwritten_reachable_data` and `unexposed_storage_barriers` present).
# Bytes per call at n = 1,024 and 65,536:
#
#     direct   320 / 320
#     prepared 1,376 / 1,376
#     stored   1,568 / 1,568
#     pullback 8,560 / 524,576
#
# A probe of the rule's Enzyme config shows why `pullback` grows with `n`.
# `overwritten` is `(false, false, false, false)` for the first three callers
# and `(false, true, false, false)` for `pullback`: Enzyme marks the struct
# (the rule's first argument) as possibly overwritten, so the adapter copies
# all `n` numbers on every call. With Enzyme's reverse mode a prepared
# pullback keeps the port's value for `ad_value_and_pullback`:
# `_ADSeededCall{true}` stores it into a `Ref` the preparation passes in
# (`box[] = values`, src/ad.jl). `Enzyme.API.printperf!(true)` names that store
# as the write that follows the rule call. With `ad_retains_primal_sweep`
# overridden to `false` for this backend (a diagnostic only, so no value is
# kept), the flag is `false` and a pullback takes 736 / 1,040 B. A BRM GMRF
# coefficient row read through `prepare_ad_pullback` has this shape: 15.5 MB
# per row Jacobian at 4,096 states. Snag `native-reverse-r-b8de62b9` reported
# this case to Enzyme after `rule-overwritten-cc607ba7` landed.
#
# Before libEnzyme `400c1bdc` (dogfood `8a381501`, libEnzyme `05ce259c`),
# `stored` copied too: 18,208 / 1,050,240 B. The program's store of the rule's
# vector into plate storage it had allocated counted as a write the rule cannot
# see (Enzyme snag `rule-overwritten-cc607ba7`).
module PreparedRuleOverwritten
using ReactiveKernels, Test
using ReactiveKernels.DifferentiationInterface: AutoEnzyme
import Enzyme

struct Geometry
    values::Vector{Float64}
end
row_value(g, i, x) = [g.values[i] * x, g.values[i + 1] * x]
row_slope(g, i, bar) = g.values[i] * bar[1] + g.values[i + 1] * bar[2]
@kernel function row_graph(g, i::Int, x::Float64, bar::Vector{Float64})
    y::Vector{Float64} = row_value(g, i, x)
    dx::Float64 = row_slope(g, i, bar)
    return y, dx
end
const row = derivative_rule(row_graph; primal = :y, covector = :bar,
    cotangents = (g = nothing, i = nothing, x = :dx), name = :geometry_row)

direct(g, x) = sum(row(g, 1, x)) + sum(row(g, 3, x))
function direct_bytes(g)
    Enzyme.autodiff(Enzyme.Reverse, Enzyme.Const(direct), Enzyme.Active,
        Enzyme.Const(g), Enzyme.Active(2.0))
    @allocated Enzyme.autodiff(Enzyme.Reverse, Enzyme.Const(direct), Enzyme.Active,
        Enzyme.Const(g), Enzyme.Active(2.0))
end

const spec = @kernel prepared_rows(theta, g, rows) = begin
    @plate for k in eachindex(rows)
        term[k] = sum(row(g, rows[k], theta[1]))
    end
    total = sum(term)
end
# The plate's cells hold the rule's vector, which the program stores.
const stored_spec = @kernel prepared_stored_rows(theta, g, rows) = begin
    @plate for k in eachindex(rows)
        cell[k] = row(g, rows[k], theta[1])
    end
    total = sum(sum, cell)
end
function prepared_bytes(g, spec = spec)
    theta, rows = [2.0], [1, 3]
    k = prepare(spec; have = (:theta, :g, :rows), want = :total)
    ad = prepare_ad(k, AutoEnzyme(; mode = Enzyme.Reverse), theta, g, rows; active = :theta)
    value, gradient = ad_value_and_gradient(ad, theta, g, rows)
    expected = sum(g.values[1:4])
    @test value ≈ 2expected
    @test only(gradient) ≈ expected
    ad_value_and_gradient(ad, theta, g, rows)
    @allocated ad_value_and_gradient(ad, theta, g, rows)
end

# The kernel's WANT is the rule's vector; a pullback seeds it with `seed`.
const pullback_spec = @kernel prepared_row(theta, g, i) = begin
    y = row(g, i, theta[1])
end
function pullback_bytes(g)
    theta, i, seed = [2.0], 1, [1.0, 0.5]
    k = prepare(pullback_spec; have = (:theta, :g, :i), want = :y)
    p = prepare_ad_pullback(k, AutoEnzyme(; mode = Enzyme.Reverse), seed, theta, g, i;
        active = :theta)
    @test only(ad_pullback(p, seed, theta, g, i)) ≈ g.values[1] + 0.5 * g.values[2]
    ad_pullback(p, seed, theta, g, i)
    @allocated ad_pullback(p, seed, theta, g, i)
end

repair(name) = isdefined(Enzyme, :has_dogfood_repair) && Enzyme.has_dogfood_repair(name)

@testset "a derivative rule over a struct input, direct and in a prepared program" begin
    println("REPAIRS ", (; overwritten_reachable_data = repair(:overwritten_reachable_data),
        unexposed_storage_barriers = repair(:unexposed_storage_barriers)))
    @test Enzyme.autodiff(Enzyme.Reverse, Enzyme.Const(direct), Enzyme.Active,
        Enzyme.Const(Geometry([1.0, 2.0, 3.0, 4.0])), Enzyme.Active(2.0))[1][2] == 10.0
    for n in (1_024, 65_536)
        g = Geometry(collect(1.0:n))
        println("RULE_BYTES ", (; n, direct = direct_bytes(g), prepared = prepared_bytes(g),
            stored = prepared_bytes(g, stored_spec), pullback = pullback_bytes(g)))
    end
end
end
