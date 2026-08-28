using ReactiveKernels
using Test
using LinearAlgebra

# Non-HMC exercise of the public @reactive object/method facade: signature args
# are mutable HAVE sources, body assignments are compiled reactive derived
# nodes, inner defs are ordinary methods taking the object first. No HMC here.

# Count recipe evaluations to prove reactive (minimal) recomputation.
const _FACADE_CALLS = (cost = Ref(0), shifted = Ref(0))

@reactive particle(pos::Vector{Float64}, scale::Float64) = begin
    cost::Float64 = (_FACADE_CALLS.cost[] += 1; scale * sum(abs2, pos))
    shifted::Vector{Float64} = (_FACADE_CALLS.shifted[] += 1; pos .+ 1.0)
    energy() = cost + sum(shifted)
    bump!(delta) = begin
        pos = pos .+ delta
    end
    rescale!(s) = begin
        scale = s
    end
    combo!(delta, s) = begin
        bump!(delta)
        rescale!(s)
    end
end

@testset "@reactive facade — invalidation, methods, hygiene, inference" begin
    _FACADE_CALLS.cost[] = 0; _FACADE_CALLS.shifted[] = 0
    p = particle([1.0, 2.0], 2.0)
    @test p isa ReactiveObject

    # derived reactive nodes compute lazily, once
    @test p.cost == 2.0 * (1.0 + 4.0)
    @test p.shifted == [2.0, 3.0]
    @test (_FACADE_CALLS.cost[], _FACADE_CALLS.shifted[]) == (1, 1)
    @test p.cost == 10.0                      # cached, no recompute
    @test _FACADE_CALLS.cost[] == 1

    # method mutates a HAVE source -> downstream derived nodes invalidate
    bump!(p, [1.0, 1.0])
    @test p.pos == [2.0, 3.0]
    @test p.cost == 2.0 * (4.0 + 9.0)         # recomputed
    @test p.shifted == [3.0, 4.0]
    @test _FACADE_CALLS.cost[] == 2

    # mutating one source recomputes only its dependents (both depend on pos here,
    # scale-only change leaves `shifted` valid)
    s0 = _FACADE_CALLS.shifted[]
    rescale!(p, 3.0)
    @test p.cost == 3.0 * (4.0 + 9.0)
    @test _FACADE_CALLS.shifted[] == s0        # shifted independent of scale

    # a method reading derived fields
    @test energy(p) == p.cost + sum(p.shifted)

    # a method calling sibling methods (forwarded with the object)
    combo!(p, [1.0, 0.0], 1.0)
    @test p.pos == [3.0, 3.0]
    @test p.cost == 1.0 * (9.0 + 9.0)

    # type inference through the object property getter and a method
    _cost(o) = o.cost
    @test @inferred(_cost(p)) == p.cost
    @test @inferred(energy(p)) == energy(p)

    # hygiene: caller globals with field names do not leak into the object
    pos = 999.0; scale = -1.0; cost = :nope   # shadow the field names in caller scope
    q = particle([5.0], 4.0)
    @test q.cost == 4.0 * 25.0
    @test q.pos == [5.0]

    # independence / composition: two objects share no state; copy is detached
    a = particle([1.0], 1.0); b = particle([1.0], 1.0)
    bump!(a, [10.0])
    @test a.pos == [11.0] && b.pos == [1.0]
    c = copy(a)
    bump!(a, [100.0])
    @test a.pos == [111.0] && c.pos == [11.0]
    @test c.program === a.program              # program shared, state detached
end

@reactive kwparticle(pos::Vector{Float64}; scale::Float64 = 2.0, shift::Float64 = 0.0) = begin
    cost::Float64 = scale * sum(abs2, pos) + shift
    tweak!(s) = begin
        scale = s
    end
end

@testset "@reactive facade — keyword/default sources" begin
    p = kwparticle([1.0, 2.0])                    # defaults scale=2, shift=0
    @test p.cost == 2.0 * 5.0
    @test p.scale == 2.0 && p.shift == 0.0
    q = kwparticle([1.0, 2.0]; scale = 3.0, shift = 1.0)
    @test q.cost == 3.0 * 5.0 + 1.0
    tweak!(q, 10.0)                               # mutate a keyword-source field
    @test q.cost == 10.0 * 5.0 + 1.0
    kp = kwparticle([1.0]; scale = 5.0)
    _kwcost(o) = o.cost
    @test @inferred(_kwcost(kp)) == 5.0           # type-stable read behind a barrier
end

# Derived-field override (assign! lowering) + method keyword/default + sibling
# forwarding, exactly the ca9 shapes (gofwd/may_continue writes; step!(;force),
# swapproposal!(i, j=...)).
@reactive counter(base::Int) = begin
    derived::Int = base + 10
    total::Int = derived * 2
    override!(v::Int) = begin
        derived = v                       # assign a DERIVED field: temporary override
    end
    bumpbase!(; by::Int = 1) = begin      # keyword-only method
        base = base + by
    end
    combo!(v::Int, n::Int = 2) = begin    # positional default + sibling calls
        override!(v)                      # sibling call (positional)
        bumpbase!(; by = n)               # sibling call with a keyword
    end
end

@testset "@reactive facade — derived override + method kwargs/defaults" begin
    c = counter(5)
    @test c.derived == 15 && c.total == 30

    # a method overrides a DERIVED field: get! returns the override (no recompute),
    # and its downstream recomputes from the override
    override!(c, 100)
    @test c.derived == 100
    @test c.total == 200

    # a later true-upstream change invalidates the override -> recompute from recipe
    bumpbase!(c; by = 3)
    @test c.base == 8
    @test c.derived == 18                 # 8 + 10, override discarded
    @test c.total == 36

    # keyword default
    bumpbase!(c)                          # by defaults to 1
    @test c.base == 9 && c.derived == 19

    # positional default + sibling forwarding (positional and keyword)
    combo!(c, 999, 4)                     # override(999) then bumpbase!(by=4)
    @test c.base == 13                    # 9 + 4
    @test c.derived == 23                 # override 999 superseded by base change
    combo!(c, 0)                          # n defaults to 2
    @test c.base == 15                    # 13 + 2
end

# Additive @reactive `prepare=` hook: the macro owns spec/want/handles; an injected
# callable returns the ReactiveProgram, so a definition can opt into the proven
# non-allocating cache_apply/is_mutating preparation without any HMC- or
# MutatingFunctions-specific coupling. Default (no option) stays byte-for-byte.
@reactive _prep_default(pos::Vector{Float64}) = begin
    copied::Vector{Float64} = copy(pos)
end

# Hand-written, MutatingFunctions-agnostic in-place hook.
_prep_hca(cache::AbstractVector, ::typeof(copy), a) =
    (length(cache) == length(a) || resize!(cache, length(a)); copyto!(cache, a); cache)
_prep_hca(cache, op, args...) = op(args...)
_prep_inplace(spec; want) = ReactiveKernels._prepare_reactive(
    spec; want = want, cache_apply = _prep_hca,
    is_mutating = ReactiveKernels._default_is_mutating)

@reactive prepare=_prep_inplace _prep_owned(pos::Vector{Float64}) = begin
    copied::Vector{Float64} = copy(pos)
end

@testset "@reactive facade — additive prepare hook (default unchanged)" begin
    # Default expansion is byte-for-byte identical (option resolves to the same
    # prepare_reactive value at macro-expansion time).
    default_ast = @macroexpand @reactive _md(x::Vector{Float64}) = begin
        y::Vector{Float64} = copy(x)
    end
    @test !occursin("cache_apply", string(default_ast))
    @test occursin("prepare_reactive", string(default_ast))

    d = _prep_default([1.0, 2.0])
    @test d isa ReactiveObject
    @test d.copied == [1.0, 2.0]
    # Default is the allocating path: each recompute returns a fresh array.
    first_copy = d.copied
    setproperty!(d, :pos, [3.0, 4.0])
    @test d.copied == [3.0, 4.0]
    @test d.copied !== first_copy
end

@testset "@reactive facade — custom in-place preparation (0 B, isolation)" begin
    owned = _prep_owned([1.0, 2.0, 3.0])
    @test owned isa ReactiveObject
    buffer = owned.copied
    @test buffer == [1.0, 2.0, 3.0]

    # The owned slot is reused in place across invalidate→recompute (stable id).
    setproperty!(owned, :pos, [4.0, 5.0, 6.0])
    @test owned.copied === buffer
    @test owned.copied == [4.0, 5.0, 6.0]

    # Steady-state 0 bytes behind a function barrier.
    function _cycle!(object, value, n)
        for _ in 1:n
            setproperty!(object, :pos, value)
            object.copied
        end
    end
    probe_value = [4.0, 5.0, 6.0]
    _cycle!(owned, probe_value, 1)
    @test @allocated(_cycle!(owned, probe_value, 1000)) == 0

    # Per-instance cache isolation.
    other = _prep_owned([7.0, 8.0, 9.0])
    @test other.copied !== owned.copied
    setproperty!(other, :pos, [0.0, 0.0, 0.0])
    @test owned.copied == [4.0, 5.0, 6.0]

    # copy deep-copies the owned buffer (no aliasing across copies).
    clone = copy(owned)
    @test clone.copied !== owned.copied
    setproperty!(clone, :pos, [1.0, 1.0, 1.0])
    @test owned.copied == [4.0, 5.0, 6.0]

    # Inference: the reactive read is concretely typed.
    @test (@inferred((o -> o.copied)(owned)))::Vector{Float64} == [4.0, 5.0, 6.0]
end

# A prepare callable that does not return a ReactiveProgram; the validation runs in
# the const program-cache initializer at definition-eval time.
_bad_prep(spec; want) = 42
_unwrap_err(err) = err isa LoadError ? err.error : err

@testset "@reactive facade — prepare hook robustness (duplicate/validation/AST)" begin
    # Duplicate prepare= options are rejected deterministically (macro-expansion).
    dup = try
        @eval @reactive prepare=identity prepare=identity _dup(x::Vector{Float64}) = begin
            y::Vector{Float64} = copy(x)
        end
        nothing
    catch err
        err
    end
    @test _unwrap_err(dup) isa ArgumentError

    # The injected callable must return a ReactiveProgram — actionable build error.
    thrown = try
        @eval @reactive prepare=_bad_prep _bad(x::Vector{Float64}) = begin
            y::Vector{Float64} = copy(x)
        end
        nothing
    catch err
        err
    end
    @test _unwrap_err(thrown) isa ArgumentError
    @test occursin("must return a ReactiveProgram", sprint(showerror, _unwrap_err(thrown)))

    # Default expansion AST is unchanged by the additive option (no cache_apply,
    # still prepare_reactive; option-free and prepare=default produce the same
    # program-build call shape).
    ast_default = string(@macroexpand @reactive _a1(x::Vector{Float64}) = begin
        y::Vector{Float64} = copy(x)
    end)
    @test occursin("prepare_reactive", ast_default)
    @test !occursin("__cache_apply__", ast_default)
    @test !occursin("must return a ReactiveProgram", ast_default)   # no validation wrap
end


# --- Per-construction specialization (specialize=true) FOUNDATION tests. ---
# Named scaling helpers (two distinct concrete callable types). We avoid a bare
# `pos .* scale` RHS: `_kernel_operation` currently preserves a dotted operator as
# a direct callee, and `.*` is not a bound function — tracked as a separate
# authoring follow-up, not needed by these tests or the sampler's recipes.
struct _SpScale{S}; s::S; end
(op::_SpScale)(x) = broadcast(*, op.s, x)      # Float32-preserving when s::Float32
struct _SpShift{S}; s::S; end
(op::_SpShift)(x) = broadcast(+, x, op.s)

# Runtime-derived annotations (`typeof(pos)`, `eltype(pos)`) evaluated in the
# constructor against the concrete args make the DERIVED slots concrete too — no
# `where` grammar. HAVE ports are typed from `typeof(arg)`.
@reactive specialize=true _sp_obj(op, pos, metric) = begin
    applied::typeof(pos) = op(pos)
    total::eltype(pos) = sum(applied)
    kept_metric::typeof(metric) = metric
end

@reactive specialize=true _sp_kw(pos; scale = 2.0) = begin
    scaled::typeof(pos) = op_scale(pos, scale)
end
op_scale(pos, scale) = broadcast(*, scale, pos)

# Misbehaving prepare callables (same object, different failure modes).
_sp_wrong_graph(spec; want) = begin
    g = ReactiveKernels.Graph(); x = ReactiveKernels.value!(g, :x, Float64)
    ReactiveKernels.prepare_reactive(g; have = (x,), want = (x,))
end
_sp_wrong_want(spec; want) = ReactiveKernels.prepare_reactive(spec; want = ())
_sp_wrong_order(spec; want) =
    ReactiveKernels.prepare_reactive(spec;
        have = Tuple(reverse(spec.have_names)), want = want)
@reactive prepare=_sp_wrong_graph specialize=true _sp_bad_graph(pos, mom) = begin
    y::typeof(pos) = op_scale(pos, 1.0)
end
@reactive prepare=_sp_wrong_want specialize=true _sp_bad_want(pos, mom) = begin
    y::typeof(pos) = op_scale(pos, 1.0)
end
@reactive prepare=_sp_wrong_order specialize=true _sp_bad_order(pos, mom) = begin
    y::typeof(pos) = op_scale(pos, 1.0)
end

_sp_ref(object, name) = typeof(getfield(object, :state).slots[
    ReactiveKernels._slot_index(getfield(object, :handles)[name])])

@testset "@reactive specialize — concrete HAVE+derived slots, distinct programs" begin
    op_a = _SpScale(2.0); op_b = _SpShift(1.0)     # two distinct callable types
    m_dense = [2.0 0.0; 0.0 3.0]; m_diag = Diagonal([2.0, 3.0])

    a64 = _sp_obj(op_a, [1.0, 2.0], m_dense)
    b64 = _sp_obj(op_b, [1.0, 2.0], m_dense)
    a32 = _sp_obj(_SpScale(2.0f0), Float32[1, 2], Matrix{Float32}([2 0; 0 3]))
    adiag = _sp_obj(op_a, [1.0, 2.0], m_diag)

    # Distinct concrete object types across callable, precision, and storage.
    @test typeof(a64) !== typeof(b64)
    @test typeof(a64) !== typeof(a32)
    @test typeof(a64) !== typeof(adiag)

    # Exact HAVE slot Ref types (via handle indices) — concrete, no Ref{Any}.
    @test _sp_ref(a64, :op) === Base.RefValue{_SpScale{Float64}}
    @test _sp_ref(a64, :pos) === Base.RefValue{Vector{Float64}}
    @test _sp_ref(a64, :metric) === Base.RefValue{Matrix{Float64}}
    @test _sp_ref(a32, :pos) === Base.RefValue{Vector{Float32}}
    @test _sp_ref(adiag, :metric) === Base.RefValue{Diagonal{Float64,Vector{Float64}}}
    # Exact DERIVED slot Ref types — concrete via runtime annotations.
    @test _sp_ref(a64, :applied) === Base.RefValue{Vector{Float64}}
    @test _sp_ref(a64, :total) === Base.RefValue{Float64}
    @test _sp_ref(a32, :applied) === Base.RefValue{Vector{Float32}}
    @test _sp_ref(a32, :total) === Base.RefValue{Float32}
    # No hot slot is Ref{Any}.
    @test !any(t -> t === Base.RefValue{Any}, typeof(a64.state.slots).parameters)
    @test !any(t -> t === Base.RefValue{Any}, typeof(a32.state.slots).parameters)

    # Values compute correctly, precision preserved.
    @test a64.applied == [2.0, 4.0] && a64.total == 6.0
    @test b64.applied == [2.0, 3.0]
    @test a32.applied == Float32[2, 4] && a32.applied isa Vector{Float32}
    @test a32.total isa Float32

    # Function-barrier inference of derived reads is concrete.
    @test (@inferred((o -> o.applied)(a64)))::Vector{Float64} == [2.0, 4.0]
    @test (@inferred((o -> o.total)(a64)))::Float64 == 6.0
    @test (@inferred((o -> o.applied)(a32)))::Vector{Float32} == Float32[2, 4]

    # Per-instance isolation + copy detachment.
    setproperty!(a64, :pos, [5.0, 6.0])
    @test a64.applied == [10.0, 12.0]
    @test b64.applied == [2.0, 3.0]
    clone = copy(a64)
    setproperty!(clone, :pos, [0.0, 0.0])
    @test a64.applied == [10.0, 12.0]
    @test clone.program === a64.program

    # Real function-barrier 0-B source mutation/read cycle (derived-recompute 0-B
    # is a tracked follow-up once in-place derived preparation composes with this).
    function _src_cycle!(o, v, n)
        for _ in 1:n
            setproperty!(o, :pos, v)
            o.pos
        end
        nothing
    end
    v = [7.0, 8.0]
    _src_cycle!(a64, v, 2)
    @test @allocated(_src_cycle!(a64, v, 1000)) == 0
end

@testset "@reactive specialize — defaults/kwargs, validation, option robustness" begin
    k = _sp_kw([1.0, 2.0])                         # default scale=2
    @test k.scaled == [2.0, 4.0]
    @test (@inferred((o -> o.scaled)(k)))::Vector{Float64} == [2.0, 4.0]
    @test _sp_ref(k, :pos) === Base.RefValue{Vector{Float64}}
    @test _sp_ref(k, :scaled) === Base.RefValue{Vector{Float64}}
    k2 = _sp_kw([1.0, 2.0]; scale = 10.0)
    @test k2.scaled == [10.0, 20.0]

    # Validation at construction: wrong graph, missing want, wrong input order.
    eg = try; _sp_bad_graph([1.0], [1.0]); nothing; catch e; e; end
    @test eg isa ArgumentError && occursin("different", sprint(showerror, eg))
    ew = try; _sp_bad_want([1.0], [1.0]); nothing; catch e; e; end
    @test ew isa ArgumentError && occursin("missing the exposed", sprint(showerror, ew))
    eo = try; _sp_bad_order([1.0], [1.0]); nothing; catch e; e; end
    @test eo isa ArgumentError && occursin("in order", sprint(showerror, eo))

    # specialize must be a literal Bool; a runtime symbol is rejected at expansion.
    lit = try
        @eval @reactive specialize=some_runtime_flag _sp_lit(x) = begin y::typeof(x) = x end
        nothing
    catch e; e; end
    @test _unwrap_err(lit) isa ArgumentError
    @test occursin("literal Bool", sprint(showerror, _unwrap_err(lit)))
end

# Hygiene regression: a HAVE port literally named `typeof` must not shadow the
# runtime type binding used by specialize=true (fixed via Core.typeof).
@reactive specialize=true _sp_shadow(typeof, pos) = begin
    held::Base.eltype(pos) = pos[1]
end

@testset "@reactive specialize — port named typeof does not shadow (hygiene)" begin
    o = _sp_shadow(42, [10.0, 20.0])            # a port literally named `typeof`
    @test o isa ReactiveObject
    @test o.typeof == 42
    @test o.held == 10.0
    # HAVE ports concretely typed despite the shadowing name.
    @test _sp_ref(o, :typeof) === Base.RefValue{Int}
    @test _sp_ref(o, :pos) === Base.RefValue{Vector{Float64}}
    @test _sp_ref(o, :held) === Base.RefValue{Float64}
end

# Hygiene regression 2 (non-vacuous): the @reactive DEFINITION itself is inside a
# `let` that shadows `typeof`, so the generated auto HAVE annotation is expanded
# where `typeof` is bound to `identity`. Only the hygienic `Core.typeof` keeps the
# HAVE slots concrete; a caller-scope shadow alone would NOT capture the generated
# annotation (that expands in the definition scope), hence the definition-site let.
@testset "@reactive specialize — definition-site typeof shadow (hygiene)" begin
    obj = let typeof = identity
        @reactive specialize=true _local_shadow_obj(pos, weight) = begin
            scaled::Base.eltype(pos) = weight * sum(pos)
        end
        _local_shadow_obj([1.0, 2.0, 3.0], 2.0)
    end
    @test obj isa ReactiveObject
    # Auto-generated HAVE refs stay concrete (not Ref{Any}) despite the shadow.
    @test _sp_ref(obj, :pos) === Base.RefValue{Vector{Float64}}
    @test _sp_ref(obj, :weight) === Base.RefValue{Float64}
    @test obj.scaled == 12.0
end


# --- Control-flow / scoping grammar boundary (alias-soundness) ----------------
# Each rejected form is exercised NON-VACUOUSLY: the construct binds or mutates the
# reactive field `pos`, exactly the shadow/capture case the rewriter cannot track,
# so silent expansion would produce a wrong graph. Expansion must reject instead.
_facade_eval(defexpr) =
    try; Core.eval(@__MODULE__, defexpr); nothing; catch e; e; end

@testset "@reactive facade — control-flow grammar boundary (rejection)" begin
    # (name, quoted @reactive def whose METHOD BODY holds the offending form,
    #  substring of the construct the error must name)
    cases = [
        (:let, quote
            @reactive _rej_let(pos::Vector{Float64}, scale::Float64) = begin
                c::Float64 = scale
                bad() = (let pos = 0.0; pos + 1.0; end)   # let-binding shadows field pos
            end
        end, "`let` block"),
        (:try, quote
            @reactive _rej_try(pos::Vector{Float64}, scale::Float64) = begin
                c::Float64 = scale
                bad!() = (try; pos = pos .+ 1.0; catch; end)  # field mutation in try
            end
        end, "`try`/`catch` block"),
        (:comprehension, quote
            @reactive _rej_comp(pos::Vector{Float64}, scale::Float64) = begin
                c::Float64 = scale
                bad() = [pos for pos in 1:3]              # comprehension var shadows field
            end
        end, "comprehension"),
        (:generator, quote
            @reactive _rej_gen(pos::Vector{Float64}, scale::Float64) = begin
                c::Float64 = scale
                bad() = sum(pos for pos in 1:3)           # generator var shadows field
            end
        end, "generator expression"),
        (:do, quote
            @reactive _rej_do(pos::Vector{Float64}, scale::Float64) = begin
                c::Float64 = scale
                bad() = foreach(1:2) do pos; pos; end      # do-param shadows field
            end
        end, "`do` block"),
        (:closure, quote
            @reactive _rej_cls(pos::Vector{Float64}, scale::Float64) = begin
                c::Float64 = scale
                bad() = (f = pos -> pos + 1.0; f(scale))   # closure param shadows field
            end
        end, "anonymous function / closure"),
        (:nested_long, quote
            @reactive _rej_fnl(pos::Vector{Float64}, scale::Float64) = begin
                c::Float64 = scale
                bad() = (function inner(pos); pos + 1.0; end; inner(scale))
            end
        end, "nested function definition"),
        (:nested_short, quote
            @reactive _rej_fns(pos::Vector{Float64}, scale::Float64) = begin
                c::Float64 = scale
                bad() = (inner(pos) = pos + 1.0; inner(scale))
            end
        end, "nested function definition"),
    ]
    for (label, defexpr, needle) in cases
        err = _facade_eval(defexpr)
        @test _unwrap_err(err) isa ArgumentError    # deterministic expansion rejection
        msg = sprint(showerror, _unwrap_err(err))
        @test occursin(needle, msg)                 # error NAMES the construct
        @test occursin("does not support", msg)
    end
    # The actionable guidance wording is present (declared grammar boundary).
    guide = sprint(showerror, _unwrap_err(_facade_eval(cases[1][2])))
    @test occursin("sibling method declared in the same @reactive definition", guide)
    # Long- and short-form nested defs are distinct ASTs both rejected as such.
    @test _facade_eval(cases[7][2]) isa Union{Exception,LoadError}
    @test _facade_eval(cases[8][2]) isa Union{Exception,LoadError}
end

# Supported straight-line/if/for/@./compound grammar keeps working, inferred, 0-B.
@reactive _grammar_ok(vec::Vector{Float64}, k::Float64) = begin
    total::Float64 = k * sum(vec)
    readsum() = begin
        s = 0.0
        for x in vec                 # for-loop (supported)
            s += x                   # compound assignment on a LOCAL (supported)
        end
        s > 0.0 ? s + total : total  # ternary (supported)
    end
    scale!(f) = begin
        k = k * f                    # compound field write (supported)
        @. vec = vec * f             # @. in-place field mutation (supported)
    end
end

# Measure inside a named function (warm then measure the SAME call) so the barrier
# is compiled code, not a top-level @allocated over globals with boxing artifacts.
_grammar_alloc(o, f) = (scale!(o, f); @allocated scale!(o, f))

@testset "@reactive facade — supported control flow stays inferred + 0-B" begin
    o = _grammar_ok([1.0, 2.0, 3.0], 2.0)
    @test o isa ReactiveObject
    @test o.total == 12.0
    @test readsum(o) == 6.0 + 12.0             # for + compound + ternary path
    @test (@inferred readsum(o)) isa Float64   # supported grammar stays inferred

    scale!(o, 2.0)                             # one call: vec *= 2, k *= 2
    @test o.vec == [2.0, 4.0, 6.0]
    @test o.k == 4.0
    @test _grammar_alloc(o, 1.0) == 0          # field @./compound writes are 0 B
end
