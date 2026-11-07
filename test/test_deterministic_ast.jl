using ReactiveKernels
using Test
using LinearAlgebra

# Regression for the deterministic getter-AST fix (src/stateful.jl _ensure_expr):
# pure-path recipe-result locals are named from the stable recipe index, not a
# fresh gensym, so two structurally identical programs share one RGF/getter TYPE
# instead of forcing a full recompile per construction.

_split(x) = (sum(x), x .* 2)                       # (scalar, vector) multi-output

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
        :(function (__constants__, __callables__, v); $a = v + 1; $b = $a * 2; return $b; end))
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
    out::Vector{Float64} = plate(obs) do o
        o * sum(units .* weights)
    end
    return out
end
const _DET_CHILD_PLATE = prepare(_det_child_plate)

_det_child_scan = @kernel _det_child_scan(xs::Vector{Float64}, gain::Float64) = begin
    values::Vector{Float64} = scan(xs; init = 0.0) do carry, x
        next = carry * gain + x
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
    @test k1.f isa ReactiveKernels._ArrayFunction
    @test typeof(k1.f.native) === typeof(k2.f.native)
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

# --- one definition built twice: one set of types --------------------------------
# A generator evaluates its definition anew per build (RKPPL's `build_kernel`), so
# what the expansion names and what a build's value ids order must not differ
# between two builds of one program: any difference is a new type, and every
# reader of it compiles again (snag `gensym-in-kernel-54164b85`).
function _det_types!(types, x, seen = IdDict{Any,Nothing}())
    x isa Union{Module,Type,Symbol,Number,AbstractString,Expr,QuoteNode,LineNumberNode,
                Task,Base.AbstractLock} && return types
    isbitstype(typeof(x)) && return types
    haskey(seen, x) && return types
    seen[x] = nothing
    push!(types, string(typeof(x)))
    if x isa AbstractArray
        isbitstype(eltype(x)) && return types
        for i in eachindex(x)
            isassigned(x, i) && _det_types!(types, x[i], seen)
        end
    elseif x isa AbstractDict
        for (k, v) in x
            _det_types!(types, k, seen)
            _det_types!(types, v, seen)
        end
    else
        for f in 1:fieldcount(typeof(x))
            isdefined(x, f) && _det_types!(types, getfield(x, f), seen)
        end
    end
    types
end
_det_types(x) = _det_types!(Set{String}(), x)

function _det_build(definition)
    mod = Module(gensym(:DetBuild))
    Core.eval(mod, :(using ReactiveKernels))
    ReactiveKernels._kernel_eval_definition(mod, definition)
end

@testset "one definition built twice spells one set of types" begin
    # `exp(log_k)` lifts out of the cell, and the rest of the recipe keeps split
    # nodes, whose type carries the lifted name.
    definition = :(f(xs::Vector{Float64}, ws::Vector{Float64}, log_k::Float64) = begin
        out::Vector{Float64} = plate(xs, ws) do x, w
            r = sin(exp(log_k)) * cos(x * w) + w
            r
        end
        return sum(out)
    end)
    spec1, spec2 = _det_build(definition), _det_build(definition)
    types1, types2 = _det_types(spec1), _det_types(spec2)
    # The comparison means something only if a generated name reaches a type.
    @test any(t -> occursin("_KernelSplit", t) && occursin("##cell_invariant#", t), types1)
    @test types1 == types2
    k1, k2 = prepare(spec1), prepare(spec2)
    @test typeof(k1) === typeof(k2)
    xs, ws = [0.5, -1.0, 2.0], [0.25, 1.5, -0.75]
    @test k1(xs, ws, 0.3) == k2(xs, ws, 0.3)
    @test k1(xs, ws, 0.3) ≈ sum(sin(exp(0.3)) * cos(x * w) + w for (x, w) in zip(xs, ws))
    # A different definition keeps its own names.
    other = _det_build(:(f(xs::Vector{Float64}, ws::Vector{Float64}, log_k::Float64) = begin
        out::Vector{Float64} = plate(xs, ws) do x, w
            r = sin(exp(log_k)) * cos(x + w) + w
            r
        end
        return sum(out)
    end))
    # By its 12-digit fingerprint and 6-digit position; the generated body's
    # canonical `##cell_invariant#2` is the same in every program.
    names(types) = Set(m.match for t in types for m in eachmatch(r"##cell_invariant#\d{18}", t))
    @test !isempty(names(types1))
    @test isdisjoint(names(types1), names(_det_types(other)))
end

# A child called in a generator term or under a lazy arm is spliced into the
# caller's source as its operations' objects (`_kernel_inline_endpoint_call`),
# and each carries its definition's token and module. A generator such as BRM
# evaluates both definitions in a fresh module per build, so these differed in
# every build, and `RuntimeGeneratedFunctions` hashes every field of an embedded
# object into the body's id: each build prepared another type (snag
# `sampler-runtimeg-13afd928`, an ODL archive sampler).
_DET_SPLICED_DEFINITIONS = (
    :(@kernel _det_spliced_child(y::Float64, mu::Float64, s::Float64) = begin
        r::Float64 = -abs2((y - mu) / s) / 2 - log(s)
        return r
    end),
    :(@kernel _det_spliced_parent(ys::Vector{Float64}, limit::Float64, mu::Float64,
                                  s::Float64) = begin
        total::Float64 = sum(_det_spliced_child(ys[i], mu, s) for i in eachindex(ys))
        tail::Float64 = limit < mu ? _det_spliced_child(limit, mu, s) : log(s)
        return total + tail
    end))

function _det_spliced_build()
    mod = Module(gensym(:DetBuild))
    Core.eval(mod, :(using ReactiveKernels))
    foreach(definition -> Core.eval(mod, definition), _DET_SPLICED_DEFINITIONS)
    Core.eval(mod, :_det_spliced_parent)
end

# The source operations a lowered body embeds as objects.
function _det_embedded_ops!(found, node)
    value = node isa QuoteNode ? node.value : node
    value isa ReactiveKernels._KernelSourceOp && return push!(found, value)
    node isa Expr && foreach(arg -> _det_embedded_ops!(found, arg), node.args)
    found
end
_det_embedded_ops(ast) = _det_embedded_ops!(Any[], ast)

@testset "a spliced child's operations do not name their build" begin
    RK = ReactiveKernels
    spec1, spec2 = _det_spliced_build(), _det_spliced_build()
    k1, k2 = prepare(spec1), prepare(spec2)
    # The comparison means something only if the bodies embed operations whose
    # token and module differ between the builds.
    ops1, ops2 = _det_embedded_ops(code_expr(k1)), _det_embedded_ops(code_expr(k2))
    @test !isempty(ops1) && length(ops1) == length(ops2)
    @test all(getfield(a, :token) !== getfield(b, :token) for (a, b) in zip(ops1, ops2))
    @test all(getfield(a, :mod) isa Module && getfield(a, :mod) !== getfield(b, :mod)
              for (a, b) in zip(ops1, ops2))
    @test typeof(k1) === typeof(k2)
    @test typeof(k1.f) === typeof(k2.f)
    child(y, mu, s) = -abs2((y - mu) / s) / 2 - log(s)
    ys = [0.5, -1.0, 2.0]
    for (limit, mu, s) in ((0.2, 1.0, 0.7), (1.5, 1.0, 0.7))
        expected = sum(child(y, mu, s) for y in ys) +
                   (limit < mu ? child(limit, mu, s) : log(s))
        @test k1(ys, limit, mu, s) == k2(ys, limit, mu, s)
        @test k1(ys, limit, mu, s) ≈ expected
    end
    # The body keeps each operation's record, form, call identity and twin, and
    # drops only the token and module.
    for op in ops1
        copy = RK._embedded_execution_copy(op)
        @test typeof(copy) === typeof(op)
        @test getfield(copy, :f) === getfield(op, :f)
        @test getfield(copy, :form) === getfield(op, :form)
        @test getfield(copy, :call_identity) === getfield(op, :call_identity)
        @test getfield(copy, :token) === RK._EMBEDDED_SOURCE_TOKEN
        @test getfield(copy, :mod) === nothing
    end
    twin = RK._kernel_with_ignored_throws(first(ops1), first(ops2))
    copied = RK._embedded_execution_copy(twin)
    @test getfield(getfield(copied, :ignored_throws), :token) === RK._EMBEDDED_SOURCE_TOKEN
    @test getfield(getfield(copied, :ignored_throws), :mod) === nothing
    @test getfield(getfield(copied, :ignored_throws), :f) === getfield(first(ops2), :f)
    # Everything else a body embeds is left as it is.
    ast = :(f($(QuoteNode(1.5)), $(QuoteNode(:x)), y))
    @test RK._canonical_embedded(ast) == ast
    @test RK._canonical_embedded(ast).args[2] === ast.args[2]
    # An embedded object that reaches state a call could write keeps its build's
    # names, so two builds never share it.
    stateful = (first(ops1), Ref(1.0))
    @test RK._canonical_embedded(:(f($(QuoteNode(stateful))))).args[2].value === stateful
    data = (first(ops1), [1.0, 2.0])
    copied_data = RK._canonical_embedded(:(f($(QuoteNode(data))))).args[2].value
    @test getfield(copied_data[1], :token) === RK._EMBEDDED_SOURCE_TOKEN
    @test copied_data[2] === data[2]
end

# A child holding a plate, called under a lazy arm, is prepared when its caller
# is defined and embedded in the caller's body whole (`_KernelPreparedChild`):
# its plan and recipes carry the value ids of its build.
_DET_ARM_CHILD_DEFINITIONS = (
    :(@kernel _det_arm_child(ys::Vector{Float64}, mu::Float64) = begin
        out::Vector{Float64} = plate(ys) do y
            -abs2(y - mu) / 2
        end
        return sum(out)
    end),
    :(@kernel _det_arm_parent(ys::Vector{Float64}, limit::Float64, mu::Float64) = begin
        tail::Float64 = limit < mu ? _det_arm_child(ys, mu) : 0.0
        return tail
    end))

function _det_arm_build()
    mod = Module(gensym(:DetBuild))
    Core.eval(mod, :(using ReactiveKernels))
    foreach(definition -> Core.eval(mod, definition), _DET_ARM_CHILD_DEFINITIONS)
    Core.eval(mod, :_det_arm_parent)
end

@testset "a prepared child under a lazy arm does not name its build" begin
    RK = ReactiveKernels
    k1, k2 = prepare(_det_arm_build()), prepare(_det_arm_build())
    children(ast) = (found = Any[];
        walk(n) = (n isa QuoteNode && n.value isa RK._KernelPreparedChild && push!(found, n.value);
                   n isa Expr && foreach(walk, n.args));
        walk(ast); found)
    c1, c2 = children(code_expr(k1)), children(code_expr(k2))
    @test length(c1) == 1 && length(c2) == 1
    @test only(c1).kernel.plan.have != only(c2).kernel.plan.have  # distinct builds' value ids
    @test typeof(k1) === typeof(k2)
    ys = [0.5, 1.0, -0.25]
    @test k1(ys, 0.2, 1.0) == k2(ys, 0.2, 1.0)
    @test k1(ys, 0.2, 1.0) ≈ sum(-abs2(y - 1.0) / 2 for y in ys)
    @test k1(ys, 2.0, 1.0) == 0.0
    # The body's copy calls the same compiled child.
    copy = RK._embedded_execution_copy(only(c1))
    @test typeof(copy) === typeof(only(c1))
    @test copy.kernel.f === only(c1).kernel.f
    @test copy(ys, 1.0) == only(c1)(ys, 1.0)
    @test isempty(copy.kernel.plan.recipes)
end

@testset "a generated name is a function of the definition and its position" begin
    RK = ReactiveKernels
    fingerprint = RK._kernel_source_fingerprint
    source = :(f(x) = begin
        y = g(x, 1.0)
        return y
    end)
    # Line numbers, a literal's value, a generator's numbering and a fresh module
    # are not part of the program's identity here.
    @test fingerprint(source) == fingerprint(Base.remove_linenums!(copy(source)))
    @test fingerprint(:(g(x, 1.0))) == fingerprint(:(g(x, 2.0)))
    a, b = gensym(:t), gensym(:t)
    @test fingerprint(:($a + $b)) == fingerprint(:($(gensym(:t)) + $(gensym(:t))))
    @test fingerprint(:($a + $a)) != fingerprint(:($a + $b))
    fresh() = Module(gensym(:Fresh))
    @test fingerprint(GlobalRef(fresh(), :g)) == fingerprint(GlobalRef(fresh(), :g))
    @test fingerprint(GlobalRef(Base, :g)) != fingerprint(GlobalRef(Core, :g))
    # Authored names and structure are.
    @test fingerprint(:(g(x))) != fingerprint(:(h(x)))
    @test fingerprint(:(g(x))) != fingerprint(:(g(y)))
    @test fingerprint(:(g(x, y))) != fingerprint(:(g(y, x)))
    @test fingerprint(:(g(x)[1])) != fingerprint(:(g(x)))

    names() = RK._with_kernel_names(() -> (RK._kernel_gensym(:v), RK._kernel_gensym(:v)), source)
    n1, n2 = names()
    @test n1 !== n2
    @test names() === (n1, n2)
    # The shape everything that recognizes a generated name reads.
    @test RK._lowering_gensym(n1)
    @test Base.isgensym(n1)
    @test RK._generated_value_name(n1)
    @test RK._gensym_base(n1) == "v"
    @test startswith(String(RK._value_local_name(n1)), "#v#")
    @test RK._canonical_locals(:($n1 + $n2)) == RK._canonical_locals(:($(gensym(:v)) + $(gensym(:v))))
    # A nested expansion keeps its own position and restores the outer one.
    outer = RK._with_kernel_names(source) do
        first = RK._kernel_gensym(:v)
        RK._with_kernel_names(() -> RK._kernel_gensym(:v), :(other_source))
        (first, RK._kernel_gensym(:v))
    end
    @test outer === (n1, n2)
    # A generator's steps continue one count, inside one another or inside an
    # expansion, so a submodel called twice is named apart.
    continued = RK._within_kernel_names(source) do
        first = RK._kernel_gensym(:v)
        (first, RK._within_kernel_names(() -> RK._kernel_gensym(:v), source))
    end
    @test continued === (n1, n2)
    # Outside an expansion it is `gensym`.
    @test RK._kernel_gensym(:v) !== RK._kernel_gensym(:v)
end

# One source with different meanings: two definitions spell the same generated
# names, and must still be two computations in one graph.
const _DET_SAME_ONE = let c = 1.0
    @kernel f(xs::Vector{Float64}, k::Float64) = begin
        out::Vector{Float64} = plate(xs) do x
            r = x * sin(exp(k) + c)
            r
        end
        return sum(out)
    end
end
const _DET_SAME_TWO = let c = 2.0
    @kernel f(xs::Vector{Float64}, k::Float64) = begin
        out::Vector{Float64} = plate(xs) do x
            r = x * sin(exp(k) + c)
            r
        end
        return sum(out)
    end
end
module _DetFirst
using ReactiveKernels
g(x) = x + 1.0
@kernel f(xs::Vector{Float64}, k::Float64) = begin
    out::Vector{Float64} = plate(xs) do x
        r = sin(g(x) * exp(k))
        r
    end
    return sum(out)
end
end
module _DetSecond
using ReactiveKernels
g(x) = x + 2.0
@kernel f(xs::Vector{Float64}, k::Float64) = begin
    out::Vector{Float64} = plate(xs) do x
        r = sin(g(x) * exp(k))
        r
    end
    return sum(out)
end
end
@kernel _det_same_source(xs::Vector{Float64}, k::Float64) = begin
    a::Float64 = _DET_SAME_ONE(xs, k)
    b::Float64 = _DET_SAME_TWO(xs, k)
    c::Float64 = _DetFirst.f(xs, k)
    d::Float64 = _DetSecond.f(xs, k)
    return (a, b, c, d)
end

@testset "two definitions of one source stay two computations" begin
    xs = [0.5, -1.0, 2.0]
    k = 0.3
    a, b, c, d = prepare(_det_same_source)(xs, k)
    @test a ≈ sum(x * sin(exp(k) + 1.0) for x in xs)
    @test b ≈ sum(x * sin(exp(k) + 2.0) for x in xs)
    @test c ≈ sum(sin((x + 1.0) * exp(k)) for x in xs)
    @test d ≈ sum(sin((x + 2.0) * exp(k)) for x in xs)
    @test a != b
    @test c != d
end

@testset "a build's value ids order nothing a type reads" begin
    RK = ReactiveKernels
    # A bound plate's axes do not depend on the order of its arguments, and are
    # what splatting them gives.
    args = Any[[1, 2, 3], Ref([1.0, 2.0]), [1.0, 2.0, 3.0], Ref(1.0), fill(0.5, 1)]
    @test RK._partial_bound_axes(args) === Base.Broadcast.combine_axes(args...)
    @test RK._partial_bound_axes(reverse(args)) === Base.Broadcast.combine_axes(args...)
    @test RK._partial_bound_axes(Any[Ref(1.0)]) === ()
    @test RK._partial_bound_axes(Any[Float64[], [1.0]]) === (Base.OneTo(0),)
    @test_throws DimensionMismatch RK._partial_bound_axes(Any[[1, 2], [1, 2, 3]])
    # A child's snapshot follows the order the graph was built in, whatever its
    # dictionaries hold, so two builds of one child have one snapshot type.
    snapshot = RK._kernel_capture_child(:child, _DET_SAME_ONE)
    ids = [value.id for value in snapshot.values]
    @test issorted(ids)
    @test length(ids) == length(_DET_SAME_ONE.graph.values)
    producers = [first(p) for p in snapshot.producers]
    @test issorted(producers)
    @test length(producers) == length(_DET_SAME_ONE.graph.producers)
    rebuilt = RK._kernel_reconstruct(snapshot)
    @test rebuilt.graph.values == _DET_SAME_ONE.graph.values
    @test rebuilt.graph.producers == _DET_SAME_ONE.graph.producers
    # Many values of two types, so a dictionary's order shows in the tuple's type.
    definition = Expr(:(=), :(f(x::Float64)), Expr(:block,
        Any[:($(Symbol(:a, i))::Float64 = x + $i) for i in 1:40]...,
        :(u = identity(a1)),
        Any[:($(Symbol(:b, i))::Float64 = u + $(Symbol(:a, i))) for i in 1:40]...,
        :(return b40)))
    tuples = [typeof(RK._kernel_capture_child(:child, _det_build(definition)).values)
              for _ in 1:4]
    @test Value{Any} in fieldtypes(first(tuples))
    @test allequal(tuples)
end
