using ReactiveKernels
using ReactiveKernels: prepare, plate, scan, code_expr
using InteractiveUtils: code_typed
using Test

# Regression for snag `embedded-prepare-612e5944`: a prepared child called from
# a lazy arm of a parent recipe re-enters RK's generic call operators while they
# are already on the inference stack; Julia's recursion heuristic widens the
# re-entered signatures and the child's `Vector{Float64}` reaches the parent as
# an abstract `Vector`, so every consumer -- an embedded plate's per-cell call
# above all -- dispatches dynamically (2.4x read time, 3x bytes on the ShinyRK
# simulation graph). The native lowering now emits every annotated recipe
# output as a typed local, which restores the authored contract at the
# assignment and keeps the value inferable downstream.

const _DOT_RGF = ReactiveKernels.RuntimeGeneratedFunctions

@kernel _dot_child_scan(xs::Vector{Float64}, scale::Float64) = begin
    weights::Vector{Float64} = scan(xs; init = 0.0) do carry, x
        next = carry + scale * x
        (next, next)
    end
    return weights
end
const _DOT_CHILD = prepare(_dot_child_scan)

@kernel _dot_cell_plate(idx, units, weights) = begin
    values::Vector{Float64} = plate(idx) do i
        sum((units[max(i - j, 1)] * weights[j] for j in eachindex(weights)); init = 0.0)
    end
    return values
end
const _DOT_CELL = prepare(_dot_cell_plate)

@kernel _dot_parent(xs::Vector{Float64}, scale::Float64, idx, units, n::Int) = begin
    weights::Vector{Float64} = if n == 0
        Float64[]
    else
        _DOT_CHILD(xs, scale)
    end
    values = _DOT_CELL(idx, units, weights)
    total::Float64 = sum(values)
    return total
end

# A kernel with a plate is prepared as a native/tensorized pair, so its native
# body is host-only and may declare.
@kernel _dot_convert(n::Int, xs::Vector{Float64}, w::Vector{Float64}) = begin
    y::Vector{Float64} = fill(n, 3)
    z::Float64 = 2 * n
    scaled::Vector{Float64} = plate(xs) do x
        x * z + w[1]
    end
    return (y, z, scaled)
end

# A plain kernel has no tensorized product: Reactant traces its native body, so
# that body must not bind host types.
@kernel _dot_plain(n::Int, xs::Vector{Float64}) = begin
    y::Vector{Float64} = xs .* n
    return y
end

@kernel _dot_batched(position::Float64, xs::Vector{Float64}, n::Int) = begin
    weights::Vector{Float64} = if n == 0
        Float64[]
    else
        _DOT_CHILD(xs, position)
    end
    total::Float64 = sum(weights)
    return total
end

function _dot_native(k)
    f = k.f isa ReactiveKernels._ArrayFunctionPair ? k.f.native : k.f
    f isa ReactiveKernels._PrecompileWarmFunction ? f.f : f
end
# `code_typed` (not `return_types`) reaches through the generated RGF entry.
_dot_return_type(k, argtypes...) = last(only(code_typed(
    _DOT_RGF.generated_callfunc,
    Tuple{typeof(_dot_native(k)), typeof(k.ops), argtypes...})))
_dot_declarations(ast::Expr) = [s.args[1] for s in ast.args[2].args
                                if s isa Expr && s.head === :local]
# The emitted declaration carries the type OBJECT, not a `curly` expression.
_dot_decl(name::Symbol, T) = Expr(:(::), name, T)

@testset "declared output types are typed locals of the native product" begin
    xs = collect(1.0:14.0)
    idx = collect(1:2048)
    units = collect(range(0.0, 1.0; length = 2100))
    scale = 0.5
    p = prepare(_dot_parent; have = (:xs, :scale, :idx, :units, :n), want = :total,
                bound = (; idx, units))
    declarations = _dot_declarations(code_expr(p))
    @test _dot_decl(:weights, Vector{Float64}) in declarations
    @test _dot_decl(:total, Float64) in declarations
    # The prepared child's result stays concretely typed across the lazy arm, so
    # the parent infers exactly and the embedded plate loop dispatches statically.
    @test _dot_return_type(p, Vector{Float64}, Float64, Int) === Float64
    branch_index = findfirst(op -> op isa ReactiveKernels._KernelSourceOp &&
                                   op.f isa ReactiveKernels._KernelBranch, p.ops)
    # `code_typed` includes the callable type when checking a generated method;
    # `return_types` omits it on Julia 1.10 and rejects this concrete signature.
    @test last(only(code_typed(p.ops[branch_index], (Int, Vector{Float64}, Float64)))) ===
          Vector{Float64}
    # Value parity with the typed-HAVE cut, and the empty arm.
    q = prepare(_dot_parent; have = (:weights, :idx, :units), want = :total,
                bound = (; idx, units))
    w = _DOT_CHILD(xs, scale)
    @test p(xs, scale, 14) == q(w)
    @test p(xs, scale, 0) == 0.0
    # The nested read allocates like the typed cut, not like a boxed per-cell loop
    # (measured before the fix: 908 KB against 131 KB at this size).
    p(xs, scale, 14); q(w)
    nested = @allocated p(xs, scale, 14)
    control = @allocated q(w)
    @test nested <= control + 4096
end

@testset "declared types convert at the assignment; only host-only products declare" begin
    k = prepare(_dot_convert)
    y, z, scaled = k(2, [1.0, 2.0], [0.5])
    @test y isa Vector{Float64} && y == [2.0, 2.0, 2.0]
    @test z isa Float64 && z === 4.0
    @test scaled == [4.5, 8.5]
    p = plan(_dot_convert; have = (:n, :xs, :w), want = (:y, :z, :scaled))
    # `inline_embedded = true` is the product `prepare` compiles.
    native, _, _ = lower_with_ops(p; inline_embedded = true)
    tensorized, _, _ = lower_with_ops(p; tensorized = true, inline_embedded = true)
    @test Set(_dot_declarations(native)) == Set([
        _dot_decl(:y, Vector{Float64}), _dot_decl(:z, Float64),
        _dot_decl(:scaled, Vector{Float64})])
    @test isempty(_dot_declarations(tensorized))
    @test Set(_dot_declarations(code_expr(k))) == Set(_dot_declarations(native))
    # The plain kernel's single product is traced under Reactant: no declarations.
    plain = prepare(_dot_plain)
    @test isempty(_dot_declarations(code_expr(plain)))
    @test plain(2, [1.0, 2.0]) == [2.0, 4.0]
    # A bound constant never declares, so a deliberately bound view stays a view.
    shifts = [0.5, 1.5, 2.5]
    bound = prepare(_dot_convert; have = (:n, :xs, :w), want = (:y, :z, :scaled),
                    bound = (; w = view(shifts, 2:3)))
    @test !(_dot_decl(:w, Vector{Float64}) in _dot_declarations(code_expr(bound)))
    @test any(op -> op isa ReactiveKernels._BoundConstant && op.value isa SubArray, bound.ops)
    @test bound(2, [1.0, 2.0])[3] == [5.5, 9.5]
end

@testset "declared output types hold in position batching" begin
    batch = vectorize(_dot_batched; batched = :position, want = :total)
    positions = [0.5, 1.0, 2.0]
    xs = collect(1.0:5.0)
    @test batch(positions, xs, 5) == [sum(_DOT_CHILD(xs, s)) for s in positions]
    # The per-position body is spliced with renamed locals (`##embedded_weights#N`).
    @test any(_dot_declarations(code_expr(batch))) do declaration
        name, T = declaration.args
        T === Vector{Float64} &&
            (name === :weights || occursin(r"^##embedded_weights#\d+$", String(name)))
    end
end
