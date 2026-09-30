using ReactiveKernels
using Test
using LinearAlgebra

# Regression for the deterministic getter-AST fix (src/stateful.jl _ensure_expr):
# pure-path recipe-result locals are named from the stable recipe index, not a
# fresh gensym, so two structurally identical programs share one RGF/getter TYPE
# instead of forcing a full recompile per construction.

_split(x) = (sum(x), x .* 2)                       # (scalar, vector) multi-output

@testset "deterministic getter AST — same-signature repeated dual/welford share type" begin
    @test typeof(dual_averaging_state(0.1)) === typeof(dual_averaging_state(0.2))
    @test typeof(welford_var(3)) === typeof(welford_var(3))
    @test typeof(dual_averaging_state(0.1f0)) !== typeof(dual_averaging_state(0.1))
    @test typeof(welford_var(3, Float32)) !== typeof(welford_var(3, Float64))
end

# Direct prepare_reactive multi-output on the pure default path: a
# multi-output recipe prepared twice from the same spec yields the SAME program type
# and computes correctly.
_det_spec = @kernel begin
    x::Vector{Float64}
    (lo::Float64, hi::Vector{Float64}) = _split(x)
    return (lo, hi)
end

@testset "deterministic getter AST — direct prepare_reactive multi-output" begin
    p1 = prepare_reactive(_det_spec)
    p2 = prepare_reactive(_det_spec)
    @test typeof(p1) === typeof(p2)                          # shared program type
    state = p1([1.0, 2.0, 3.0])
    @test get!(state, statevalue(p1, _det_spec.lo)) == 6.0
    @test get!(state, statevalue(p1, _det_spec.hi)) == [2.0, 4.0, 6.0]
end


# --- poc exact-SHA gate additions ---------------------------------------------
@testset "reactive getter codegen has no active gensym (deterministic AST)" begin
    src = read(joinpath(pkgdir(ReactiveKernels), "src", "stateful.jl"), String)
    lo = findfirst("function _ensure_expr(", src)
    hi = findnext("\nfunction _getter_ast(", src, last(lo))
    fn = src[first(lo):first(hi)]
    code_lines = filter(l -> !occursin(r"^\s*#", l), split(fn, "\n"))
    @test !any(l -> occursin("gensym(", l), code_lines)   # only comments mention it
end

@testset "mixed pure+in-place NUTS group — same signature shares one type" begin
    _mg_grad!(g, q) = (copyto!(g, q); 0.5 * sum(abs2, q))
    metric = Matrix{Float64}(I, 4, 4)
    a = reactive_nuts_group(_mg_grad!, metric, [1.0, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
    b = reactive_nuts_group(_mg_grad!, metric, [5.0, 6, 7, 8], [0.5, 0.6, 0.7, 0.8])
    @test typeof(a) === typeof(b)                          # in-place bundle AST stable too
    @test typeof(reactive_program(a)) === typeof(reactive_program(b))
    @test a.dham ≈ b.dham                                 # both compute (behavior intact)
end


# --- prepared-kernel codegen: gensym-free RGF bodies ----------------------------
# Lowering mints scratch locals with `gensym` (plate axes/indices, scan carries,
# the renamed locals of a spliced embedded kernel). `compile` alpha-renames them
# (`_canonical_locals`) so an UNCHANGED graph prepares to ONE
# `RuntimeGeneratedFunction` type and is compiled once per graph shape. Snag
# `prepare-with-bou-2b4faf57`: a request-time `prepare(...; bound = ...)` of a
# graph embedding a prepared plate child minted a fresh callable type — and paid
# tens of milliseconds of compilation — on every call.
const _RGF = ReactiveKernels.RuntimeGeneratedFunctions

@testset "_canonical_locals: gensym-invariant, hygiene-preserving" begin
    a1, b1 = gensym(:x), gensym(:x)
    a2, b2 = gensym(:x), gensym(:x)
    lowered(a, b) = Base.remove_linenums!(
        :(function (__ops__, v); $a = v + 1; $b = $a * 2; return $b; end))
    e1, e2 = lowered(a1, b1), lowered(a2, b2)
    @test e1 != e2
    c1, c2 = ReactiveKernels._canonical_locals(e1), ReactiveKernels._canonical_locals(e2)
    @test c1 == c2
    body = c1.args[2].args
    @test body[1].args[1] != body[2].args[1]           # distinct originals stay distinct
    @test body[3].args[1] == body[2].args[1]           # same original, same name
    quoted = :(f($(QuoteNode(a1)), $a1))
    cq = ReactiveKernels._canonical_locals(quoted)
    @test cq.args[2] === quoted.args[2]                # quoted data untouched
    @test cq.args[3] != a1                             # the local is renamed
    @test ReactiveKernels._canonical_locals(:(x + y)) == :(x + y)
end

_det_child_plate = @kernel _det_child_plate(obs::Vector{Float64}, units::Vector{Float64},
                                            weights::Vector{Float64}) = begin
    out::Vector{Float64} = plate(obs, Ref(units), Ref(weights)) do o, u, w
        o * sum(u .* w)
    end
    return out
end
const _DET_CHILD_PLATE = prepare(_det_child_plate)

_det_child_scan = @kernel _det_child_scan(xs::Vector{Float64}, gain::Float64) = begin
    values::Vector{Float64} = scan(xs, Ref(gain); init = 0.0) do carry, x, g
        next = carry * g + x
        (next, next)
    end
    return values
end
const _DET_CHILD_SCAN = prepare(_det_child_scan)

# An embedded plate child at top level plus an embedded scan child inside a lazy
# branch, with the live array port kept: the reporter's graph shape.
_det_parent = @kernel _det_parent(data::Vector{Float64}, scale::Float64, n::Int,
                                  amounts::Vector{Float64}) = begin
    units::Vector{Float64} = data .* scale
    weights::Vector{Float64} = if n == 0
        Float64[]
    else
        _DET_CHILD_SCAN(amounts, scale)
    end
    conc::Vector{Float64} = _DET_CHILD_PLATE(data, units, weights)
    total::Float64 = sum(conc) + sum(weights)
    return total
end

@testset "repeated prepare of an unchanged graph shares one callable type" begin
    data = collect(range(0.1, 2.0; length = 8))
    amounts = collect(range(1.0, 3.0; length = 8))
    bound_prepare() = prepare(_det_parent; have = (:data, :scale, :n, :amounts),
                              want = :total, bound = (; data, scale = 2.0, n = 8))
    k1, k2 = bound_prepare(), bound_prepare()
    @test typeof(k1) === typeof(k2)                    # one PreparedKernel type
    @test k1.f isa ReactiveKernels._EmbeddedFunctionPair
    @test typeof(k1.f.native) === typeof(k2.f.native)
    @test typeof(k1.f.tensorized) === typeof(k2.f.tensorized)
    # Same RGF id ⇒ one cached body: the compiled expressions are identical.
    @test _RGF.get_expression(k1.f.native) == _RGF.get_expression(k2.f.native)
    @test k1(amounts) == k2(amounts)
    @test k1(amounts) ≈ sum(data .* sum((data .* 2.0) .* _DET_CHILD_SCAN(amounts, 2.0))) +
                        sum(_DET_CHILD_SCAN(amounts, 2.0))
    # Unbound, and a plate-only parent, share types the same way.
    u1 = prepare(_det_parent; have = (:data, :scale, :n, :amounts), want = :total)
    u2 = prepare(_det_parent; have = (:data, :scale, :n, :amounts), want = :total)
    @test typeof(u1) === typeof(u2)
    @test u1(data, 2.0, 8, amounts) == k1(amounts)
    # Inspection keeps the lowering's own (gensym'd) locals: only the compiled
    # body is canonical.
    @test code_expr(k1) isa Expr && code_expr(k1).head === :function
    @test code_expr(k1) != _RGF.get_expression(k1.f.native)
    # A different bound VALUE of the same type is still one type; a different
    # graph shape is not.
    k3 = prepare(_det_parent; have = (:data, :scale, :n, :amounts), want = :total,
                 bound = (; data = data .+ 1.0, scale = 3.0, n = 8))
    @test typeof(k3) === typeof(k1)
    @test k3(amounts) != k1(amounts)
    plate_only = prepare(_det_child_plate)
    @test typeof(plate_only) !== typeof(k1)
end
