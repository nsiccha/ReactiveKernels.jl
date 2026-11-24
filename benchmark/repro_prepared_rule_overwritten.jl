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
# the derivatives, for five callers:
# - `direct`: Enzyme on a function that sums two rule calls;
# - `prepared`: a prepared RK program whose plate cells hold `sum(row(...))`;
# - `stored`: the same program with cells that hold the rule's vector itself;
# - `pullback`: a prepared pullback (`prepare_ad_pullback`) of the rule's
#   vector, one `ad_pullback` per call;
# - `pullback_indexed`: the same pullback over `IndexedGeometry`, the same
#   numbers beside a `Dict`.
#
# Recorded on a Linux fleet host, 2026-11-22: ReactiveKernels 0.1.369, Enzyme
# dogfood `0997a2c7` with libEnzyme `24b44443` (repairs
# `overwritten_reachable_data`, `unexposed_storage_barriers`,
# `later_pointer_stores` and `later_store_targets` present). Bytes per call at
# n = 1,024 and 65,536:
#
#                      Julia 1.10.12    Julia 1.12.7 and 1.13.1
#     direct           320 / 320        320 / 320
#     prepared         1,376 / 1,376    17,680 / 1,049,872
#     stored           1,568 / 1,568    17,872 / 1,050,064
#     pullback         240 / 240        224 / 224
#     pullback_indexed 256 / 256        224 / 224
#
# A caller's bytes grow with `n` when Enzyme marks the struct (the rule's
# first argument) as possibly overwritten: the adapter then copies all `n`
# numbers on every call. A probe of the rule's Enzyme config
# (`EnzymeRules.overwritten`) reads `(false, false, false, false)` in every
# compile of the rule on Julia 1.10.12. On Julia 1.12.7 and 1.13.1 it also
# reads `(false, true, false, false)` in compiles over `Geometry`, and
# `prepared` and `stored` grow with `n`.
#
# A prepared program writes after the rule's call, where the rule cannot see.
# Each repair lets Enzyme show that one such write misses the struct's data:
# - A plate stores each result into cell storage the program allocated.
#   `unexposed_storage_barriers` (libEnzyme `400c1bdc`, Enzyme snag
#   `rule-overwritten-cc607ba7`) covers that store on Julia 1.10 only. From
#   Julia 1.11 an array keeps its elements in a `Memory`, and libEnzyme does
#   not trace a store into it back to the allocation, so `prepared` and
#   `stored` still copy there.
# - With Enzyme's reverse mode a prepared pullback keeps the port's value for
#   `ad_value_and_pullback`: `_ADSeededCall{true}` stores it into a `Ref` the
#   preparation passes in (`box[] = values`, src/ad.jl). That store is the
#   only write after the rule call, and it stores a pointer.
#   `later_pointer_stores` (dogfood `484b27a4`, Enzyme snag
#   `rule-overwritten-53a3ec5c`) keeps the flag `false` after stores of
#   pointers alone, on Julia 1.10, unless the struct leads to mutable memory
#   that holds a pointer. A `Dict` is such memory, and from Julia 1.11 so is
#   an `Array`. `later_store_targets` (Enzyme todo `1kcy4qm`) finds the object
#   each later store targets and keeps the flag `false` when the struct
#   cannot reach that object, as it cannot reach the pullback's `Ref`.
#
# A BayesianRegressionModels GMRF coefficient row read through
# `prepare_ad_pullback` has the `pullback_indexed` shape, since its rule's
# footprint holds a `Dict`. The Enzyme lane measured its row Jacobian on Julia
# 1.10.12 at 64 and 1,024 stations: 470,816 B and 471,040 B with
# `later_store_targets`, 1,396,832 B and 15,508,400 B without it.
#
# Earlier builds, Julia 1.10.12:
# - dogfood `484b27a4` (libEnzyme `400c1bdc`, without `later_store_targets`):
#   `pullback_indexed` 9,648 / 525,664 B;
# - dogfood `b862f996` (libEnzyme `400c1bdc`, without `later_pointer_stores`):
#   `pullback` 8,560 / 524,576 B;
# - dogfood `8a381501` (libEnzyme `05ce259c`): `stored` copied too,
#   18,208 / 1,050,240 B.
module PreparedRuleOverwritten
using ReactiveKernels, Test
using ReactiveKernels.DifferentiationInterface: AutoEnzyme
import Enzyme

struct Geometry
    values::Vector{Float64}
end
# The same numbers beside a `Dict`, as BRM's GMRF footprint holds one: a
# `Dict`'s fields are pointers, so a later store of a pointer could replace them.
struct IndexedGeometry
    values::Vector{Float64}
    position::Dict{Int, Int}
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
        unexposed_storage_barriers = repair(:unexposed_storage_barriers),
        later_pointer_stores = repair(:later_pointer_stores),
        later_store_targets = repair(:later_store_targets)))
    @test Enzyme.autodiff(Enzyme.Reverse, Enzyme.Const(direct), Enzyme.Active,
        Enzyme.Const(Geometry([1.0, 2.0, 3.0, 4.0])), Enzyme.Active(2.0))[1][2] == 10.0
    for n in (1_024, 65_536)
        g = Geometry(collect(1.0:n))
        println("RULE_BYTES ", (; n, direct = direct_bytes(g), prepared = prepared_bytes(g),
            stored = prepared_bytes(g, stored_spec), pullback = pullback_bytes(g),
            pullback_indexed = pullback_bytes(IndexedGeometry(g.values, Dict(1 => 1)))))
    end
end
end
