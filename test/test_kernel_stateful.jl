# Increment 1 (V7 architecture GO): the stateful `@kernel` authoring SUBSTRATE
# SKELETON. Asserts the substrate — the discriminator, unique Token, explicit-self
# object/view type skeletons, short/long (incl. kwargs/typed/where/return-annotated)
# method detection, deterministic unsupported-local-scope rejection, and the FROZEN
# detached child-capture snapshot with reconstruction. NO effect-lowering yet.
#
# A method-bearing `@kernel` binds a `const` (a stable owner binding), so it MUST be
# defined at module top level — the fixtures below live outside the `@testset`
# (which is a local scope); the local-scope rejection is asserted separately.

const RKS = ReactiveKernels

# GENERIC soundness assertion: every reachable node in a frozen recipe-source value
# is either a frozen wrapper / immutable composite (recursed) or an immutable leaf —
# i.e. NO mutable object of ANY kind (`Expr`, `Vector`, `Dict`, `Set`, a mutable
# struct, …) survives. The terminal check is `!ismutable`, so it is not tied to the
# specific container types this fixture happens to use.
_frozen_all_immutable(x::RKS._FrozenExpr) = all(_frozen_all_immutable, x.args)
_frozen_all_immutable(x::RKS._FrozenQuoteNode) = _frozen_all_immutable(x.value)
_frozen_all_immutable(x::RKS._FrozenVector) = all(_frozen_all_immutable, x.elems)
_frozen_all_immutable(x::RKS._FrozenDict) =
    all(p -> _frozen_all_immutable(p.first) && _frozen_all_immutable(p.second), x.pairs)
_frozen_all_immutable(x::Tuple) = all(_frozen_all_immutable, x)
_frozen_all_immutable(x::NamedTuple) = all(_frozen_all_immutable, values(x))
_frozen_all_immutable(x::Pair) =
    _frozen_all_immutable(x.first) && _frozen_all_immutable(x.second)
_frozen_all_immutable(::Symbol) = true   # interned atom: ismutable-true but un-mutatable
_frozen_all_immutable(::String) = true   # builtin immutable string (NOT AbstractString broadly)
# generic reachable-mutability check, independent of the freezer: recurse into an
# immutable struct's fields too, so a hidden Set/Dict/mutable-AbstractString (an immutable
# wrapper of mutable state) is still caught, not only a bare Expr/Vector/Dict.
function _frozen_all_immutable(x)
    isbits(x) && return true
    ismutable(x) && return false
    return all(i -> _frozen_all_immutable(getfield(x, i)), 1:nfields(x))
end

# A mutable leaf the freezer must REJECT (never silently retain).
mutable struct _AdvMutableLeaf
    v::Int
end

# A MUTABLE custom `AbstractString` (backed by a mutable buffer) — the freezer must
# reject it, never retain it by identity (RK repro on bb3cbcd).
mutable struct _EvilString <: AbstractString
    buf::Vector{UInt8}
end

# Recursively assert `typeof` matches at every corresponding node — a `Vector{Int}` that
# thawed as `Vector{Any}`, or a `Dict{Symbol,Int}` as a base `Dict{Any,Any}`, must FAIL
# even though `==` would pass.
function _types_match(a, b)
    typeof(a) === typeof(b) || return false
    if a isa Expr
        length(a.args) == length(b.args) || return false
        return all(i -> _types_match(a.args[i], b.args[i]), eachindex(a.args))
    elseif a isa QuoteNode
        return _types_match(a.value, b.value)
    elseif a isa Vector
        length(a) == length(b) || return false
        return all(i -> _types_match(a[i], b[i]), eachindex(a))
    elseif a isa AbstractDict
        keys(a) == keys(b) || return false
        return all(k -> _types_match(a[k], b[k]), keys(a))
    elseif a isa Tuple || a isa NamedTuple
        return all(i -> _types_match(a[i], b[i]), 1:length(a))
    elseif a isa Pair
        return _types_match(a.first, b.first) && _types_match(a.second, b.second)
    else
        return true                       # leaf: top-level typeof already matched
    end
end

# --- top-level stateful fixtures (method-bearing ⇒ const ⇒ top-level only) ---
# V7 implicit-field pivot: nested methods declare NO `self` formal; bare unshadowed
# names are the owner's fields; `__self__` appears ONLY as the first actual of a
# sibling object-pass call.
@kernel StatefulObjFixture(ham, pos, mom) = begin
    derived = ham
    leapfrog!() = begin
        @. pos += mom
    end
    function refresh!(rng)
        mom = rng
    end
end

# kwargs/defaults, typed args + `where`, and a return `::` annotation.
@kernel StatefulSigFixture(a) = begin
    r = a
    fit!(x; target = 0.8) = begin
        r = x + target
    end
    m!(x::S) where {S} = begin
        r = x
    end
    function n!()::Nothing
        r = 0
        nothing
    end
end

# Keyword `; kwargs...` splat (the ReactiveHMC fidelity signature), a positional
# `xs...` vararg, and a mixed named-kwarg + keyword-splat block. Bodies are raw AST
# (Increment 1 does not lower them), so the annotations/names need not resolve.
@kernel StatefulSplatFixture(a) = begin
    r = a
    step!(x::AbstractMatrix; kwargs...) = begin
        r = x
    end
    scan!(xs...) = begin
        r = xs
    end
    tune!(y; target = 0.8, opts...) = begin
        r = y + target
    end
end

# Recipe-source provenance: conditional (`? :`), index (`[]`), and `Ref(...)` call
# structure in the recipe portion — the exact shapes poc's MethodIR must recover.
# (The `Ref` here exercises arbitrary-AST capture fidelity; it is NOT production
# NUTS storage, which is the no-Ref concrete backend.)
@kernel StatefulRecipeSrcFixture(fwdbwd, gofwd) = begin
    fwd = Ref(fwdbwd[gofwd ? 1 : 2])
    bwd = Ref(fwdbwd[gofwd ? 2 : 1])
    swap!() = begin
        fwd = bwd
    end
end

@kernel StatefulOneFixture(a) = begin
    r = a
    m!() = begin
        r = 1
    end
end
@kernel StatefulTwoFixture(a) = begin
    r = a
    m!() = begin
        r = 1
    end
end

# Implicit-receiver sibling object-pass: `drive!` invokes sibling `refresh!` with
# `__self__` as the synthetic receiver actual → a recorded receiver edge.
@kernel StatefulSiblingFixture(a) = begin
    r = a
    refresh!(x) = begin
        r = x
    end
    drive!(x) = begin
        refresh!(__self__, x)
    end
end

# Mode-2 free methods: a METHODLESS body mutating the FIRST positional subject's
# fields (`@.` broadcast + augmented assign). `leapfrog!` (bang, not `!!`) with the
# canonical sign-corrected integrator; `nuts!!` (strong same-object update).
@kernel leapfrog!(phasepoint; stepsize) = begin
    @. phasepoint.mom -= 0.5 * stepsize * phasepoint.dham_dpos
    @. phasepoint.pos +=       stepsize * phasepoint.dham_dmom
    @. phasepoint.mom -= 0.5 * stepsize * phasepoint.dham_dpos
end
@kernel nuts!!(state; rng) = begin
    state.pos = rng
    return state
end

@testset "stateful @kernel substrate (Increment 1)" begin
    @testset "(3) methodless @kernel unchanged — routes to the stateless path" begin
        @kernel plain(f, x) = begin
            y = f(x)
        end
        @test plain isa KernelSpec
        @test !(plain isa RKS._StatefulKernelSkeleton)
        @test prepare(plain)(x -> x + 1, 2) == 3
        @test !RKS._kernel_body_has_methods(:(begin
            y = f(x)
            z = g(y)
        end))
        @test !RKS._kernel_body_has_methods(:(begin
            (a, b) = f(x)
        end))
        @test !RKS._kernel_body_has_methods(:(begin
            y::Float64 = f(x)
        end))
    end

    @testset "(1)+(2) method-presence discriminator, incl. where/::/kwargs" begin
        @test RKS._kernel_stmt_method_form(:(m!(self) = self.r)) === :short
        @test RKS._kernel_stmt_method_form(:(function m!(self); self.r; end)) === :long
        @test RKS._kernel_stmt_method_form(:(m!(self::T, x::T) where {T} = x)) === :short
        @test RKS._kernel_stmt_method_form(:(function m!(self)::Nothing; nothing; end)) === :long
        @test RKS._kernel_stmt_method_form(:(m!(self; kw = 1) = self.r)) === :short
        # non-methods
        @test RKS._kernel_stmt_method_form(:(y = f(x))) === nothing
        @test RKS._kernel_stmt_method_form(:(y::Float64 = f(x))) === nothing
        @test RKS._kernel_stmt_method_form(:((a, b) = f(x))) === nothing
    end

    @testset "(1)+(2) short/long extraction with kwargs/typed/where/return-ann" begin
        ms = RKS.kernel_methods(StatefulSigFixture)
        @test length(ms) == 3
        # fit!(x; target = 0.8): positional x, keyword target (implicit receiver)
        @test (ms[1].name, ms[1].form, ms[1].argnames) ==
              (:fit!, :short, (:x, :target))
        # m!(x::S) where {S}: typed arg through `where`
        @test (ms[2].name, ms[2].form, ms[2].argnames) ==
              (:m!, :short, (:x,))
        # function n!()::Nothing: return annotation peeled, no formals
        @test (ms[3].name, ms[3].form, ms[3].argnames) ==
              (:n!, :long, ())
        # (5) argnames is an immutable Tuple, not a Vector
        @test all(m -> m.argnames isa Tuple, ms)
        # the exposed metadata hands poc the FULL authored signature + peeled call
        # + raw body (sufficient for Increment 2's MethodIR emission).
        @test all(m -> hasproperty(m, :signature) && hasproperty(m, :call) &&
                       hasproperty(m, :body), ms)
        @test occursin("where", string(ms[2].signature))   # m!(...) where {S}
        @test occursin("Nothing", string(ms[3].signature)) # n!(self)::Nothing
    end

    @testset "extraction retains BOTH the full authored signature and peeled call" begin
        # constrained `where` binder is preserved in the signature, absent from the call
        m = RKS._kernel_extract_method(:(m!(x::S) where {S<:Real} = x), :short)
        @test occursin("where", string(m.signature))
        @test occursin("S", string(m.signature)) && occursin("Real", string(m.signature))
        @test !occursin("where", string(m.call))          # peeled call has no where
        @test m.call.head === :call
        @test (m.name, m.argnames) == (:m!, (:x,))

        # return `::` annotation preserved in the signature, absent from the call
        n = RKS._kernel_extract_method(
            :(function n!()::Nothing; nothing; end), :long)
        @test occursin("Nothing", string(n.signature))    # ::Nothing retained
        @test n.call.head === :call
        @test string(n.call) == string(:(n!()))            # peeled call drops ::Nothing
        @test (n.name, n.argnames) == (:n!, ())
    end

    @testset "kwargs-splat + positional-vararg extraction (forwarding-ready)" begin
        ms = RKS.kernel_methods(StatefulSplatFixture)
        @test length(ms) == 3

        # step!(x::AbstractMatrix; kwargs...) — the ReactiveHMC fidelity sig (implicit
        # receiver): x is the one scalar positional; `kwargs` is stored as the keyword
        # splat, NOT as an argname.
        s = ms[1]
        @test (s.name, s.form, s.argnames) == (:step!, :short, (:x,))
        @test s.vararg === nothing
        @test s.kwargs_splat === :kwargs
        # forwarding AST preserved verbatim — the raw signature + peeled call both
        # still carry `; kwargs...` (and the typed positional), so lowering can
        # forward the splat to the reference source.
        @test occursin("kwargs...", string(s.signature))
        @test occursin("kwargs...", string(s.call))
        @test occursin("AbstractMatrix", string(s.signature))

        # scan!(xs...) — positional vararg recorded in `vararg`, not argnames.
        v = ms[2]
        @test (v.name, v.argnames) == (:scan!, ())
        @test v.vararg === :xs
        @test v.kwargs_splat === nothing
        @test occursin("xs...", string(v.call))

        # tune!(y; target = 0.8, opts...) — named kwarg stays in argnames,
        # the trailing `opts...` is the keyword splat.
        t = ms[3]
        @test (t.name, t.argnames) == (:tune!, (:y, :target))
        @test t.vararg === nothing
        @test t.kwargs_splat === :opts
        @test occursin("opts...", string(t.signature))

        # the exposed metadata carries the new immutable splat slots
        @test all(m -> hasproperty(m, :vararg) && hasproperty(m, :kwargs_splat), ms)
    end

    @testset "splat storage + self/__self__-formal rejection (direct extraction)" begin
        m = RKS._kernel_extract_method(
            :(step!(x::AbstractMatrix; kwargs...) = (r = x)), :short)
        @test (m.name, m.argnames, m.vararg, m.kwargs_splat) ==
              (:step!, (:x,), nothing, :kwargs)
        # both slots are Union{Symbol,Nothing} — immutable metadata
        @test m.kwargs_splat isa Symbol && m.vararg === nothing

        # a plain method (no splats) reports `nothing` for BOTH slots — the additive
        # fields don't perturb the extraction contract.
        n = RKS._kernel_extract_method(:(m!(x; k = 1) = r), :short)
        @test n.vararg === nothing && n.kwargs_splat === nothing
        @test n.argnames == (:x, :k)

        # a `self`/`__self__` formal is rejected (implicit receiver) — scalar or splat.
        @test_throws ArgumentError RKS._kernel_extract_method(:(bad!(self) = 1), :short)
        @test_throws ArgumentError RKS._kernel_extract_method(:(bad!(__self__) = 1), :short)
        @test_throws ArgumentError RKS._kernel_extract_method(:(bad!(self...) = 1), :short)
    end

    @testset "implicit-receiver `__self__` object-pass recognition + rejection" begin
        # a sibling call `refresh!(__self__, x)` records the receiver edge.
        d = RKS._kernel_extract_method(:(drive!(x) = refresh!(__self__, x)), :short)
        @test d.sibling_calls == (:refresh!,)
        # no `__self__` ⇒ no edges.
        @test RKS._kernel_extract_method(:(plain!(x) = (r = x)), :short).sibling_calls == ()
        # the fixture exposes the edge on its `drive!` method.
        sib = RKS.kernel_methods(StatefulSiblingFixture)
        drive = sib[findfirst(m -> m.name === :drive!, sib)]
        @test drive.sibling_calls == (:refresh!,)
        # `__self__.field` (field qualifier) is rejected.
        @test_throws ArgumentError RKS._kernel_extract_method(
            :(m!(x) = (__self__.r = x)), :short)
        # a bare `__self__` outside an object-pass actual is rejected.
        @test_throws ArgumentError RKS._kernel_extract_method(:(m!() = (y = __self__)), :short)
        @test_throws ArgumentError RKS._kernel_extract_method(:(m!() = sib!(x, __self__)), :short)
    end

    @testset "Mode-2 free-method recognition (subject field mutation)" begin
        # discriminator: a body mutating the first positional's field ⇒ Mode-2,
        # NOT stateless and NOT Mode-1 (object kernel).
        @test leapfrog! isa RKS._Mode2KernelSkeleton
        @test !(leapfrog! isa KernelSpec)
        @test !(leapfrog! isa RKS._StatefulKernelSkeleton)
        @test RKS.kernel_subject(leapfrog!) === :phasepoint
        # shallow-declared effect roots on the subject (top owned fields)
        @test RKS.kernel_write_roots(leapfrog!) == (:mom, :pos)
        @test RKS.kernel_read_roots(leapfrog!) == (:dham_dpos, :dham_dmom)
        @test !RKS.kernel_is_bangbang(leapfrog!)          # `!` is not `!!`
        @test RKS.kernel_token(leapfrog!) isa Symbol
        @test RKS.kernel_module(leapfrog!) === @__MODULE__
        @test RKS.kernel_recipe_ast(leapfrog!) isa Expr    # frozen body thaws fresh
        @test RKS.kernel_spec(leapfrog!) isa KernelSpec     # ports-only recipe spec

        # `!!` strong same-object update registration recognized off the name
        @test nuts!! isa RKS._Mode2KernelSkeleton
        @test RKS.kernel_subject(nuts!!) === :state
        @test RKS.kernel_write_roots(nuts!!) == (:pos,)
        @test RKS.kernel_is_bangbang(nuts!!)
        @test RKS.kernel_token(leapfrog!) !== RKS.kernel_token(nuts!!)  # def-unique Tokens

        # the discriminator: mutation of the first positional vs a bare READ / recipe
        @test !RKS._kernel_body_mutates_subject(:(begin y = f(x) end), :x)
        @test !RKS._kernel_body_mutates_subject(:(begin y = x.field end), :x)  # read ≠ mutation
        @test RKS._kernel_body_mutates_subject(:(begin x.a = 1 end), :x)
        @test RKS._kernel_body_mutates_subject(:(begin @. x.a -= 1 end), :x)
        @test RKS._kernel_body_mutates_subject(:(begin x.a.b = 1 end), :x)     # nested owner path

        # direct effect-root scan: read/write classification incl. a field both ways
        r = RKS._kernel_subject_effect_roots(:(begin
            @. p.mom -= p.g
            p.pos = p.mom
        end), :p)
        @test r.writes == (:mom, :pos)
        @test r.reads == (:g, :mom)   # `p.mom` on the RHS of `p.pos = p.mom` is a read
    end

    @testset "registered-kernel resolver + hygiene/rebind + core intrinsic" begin
        # Mode-2 free method → registered with Token + subject + effect roots + !! flag
        reg = RKS.kernel_registration(leapfrog!)
        @test reg isa RKS._KernelRegistration
        @test reg.kind === :free_method
        @test reg.subject === :phasepoint
        @test reg.token === RKS.kernel_token(leapfrog!)
        @test reg.write_roots == (:mom, :pos)
        @test reg.read_roots == (:dham_dpos, :dham_dmom)
        @test !reg.is_bang_bang
        @test RKS.kernel_registration(nuts!!).is_bang_bang            # `!!` metadata carried

        # Mode-1 object kernel → registered by Token
        obj = RKS.kernel_registration(StatefulObjFixture)
        @test obj.kind === :object_kernel
        @test obj.token === RKS.kernel_token(StatefulObjFixture)

        # a stateless kernel → recognized by value (no Token in Increment 1)
        @kernel plainreg(f, x) = begin
            y = f(x)
        end
        sreg = RKS.kernel_registration(plainreg)
        @test sreg.kind === :stateless
        @test sreg.token === nothing

        # an ordinary Julia callable / value is NOT registered ⇒ opaque
        @test RKS.kernel_registration(sin) === nothing
        @test RKS.kernel_registration(42) === nothing

        # RK-core intrinsic: the SAME resolver accommodates an intrinsic Token + `!!`
        ireg = RKS.kernel_registration(RKS.copy!!)
        @test ireg.kind === :intrinsic
        @test ireg.is_bang_bang
        @test ireg.subject === :dest
        @test ireg.token === RKS.kernel_token(RKS.copy!!)
        @test ireg.write_roots == () && ireg.read_roots == ()        # structural, no field list
        # intrinsic token and @kernel-def tokens are distinct identities
        @test ireg.token !== reg.token
        @test ireg.token !== obj.token
        @test reg.token !== obj.token

        # hygienic name→registration hop (by module binding identity, no eval)
        @test RKS.kernel_registration(@__MODULE__, :leapfrog!).token === RKS.kernel_token(leapfrog!)
        @test RKS.kernel_registration(@__MODULE__, :a_name_that_is_undefined_xyz) === nothing

        # rebind discriminator
        cap = RKS.kernel_registration(leapfrog!)
        @test !RKS.kernel_rebound(cap, leapfrog!)                     # same binding
        @test RKS.kernel_rebound(cap, nuts!!)                         # different Token
        @test RKS.kernel_rebound(cap, sin)                            # no longer a kernel
    end

    @testset "recipe-source provenance: frozen capture + kernel_recipe_ast" begin
        skel = StatefulRecipeSrcFixture
        ast = RKS.kernel_recipe_ast(skel)
        @test ast isa Expr && ast.head === :block

        # (gate 2) recipe SOURCE round-trips the conditional/index/Ref statements in
        # authored order, structure intact.
        stmts = filter(s -> !(s isa LineNumberNode), ast.args)
        @test length(stmts) == 2
        @test stmts[1] == :(fwd = Ref(fwdbwd[gofwd ? 1 : 2]))
        @test stmts[2] == :(bwd = Ref(fwdbwd[gofwd ? 2 : 1]))

        # (gate 3) each call thaws a FRESH copy; mutating one does not change a later
        # read.
        ast2 = RKS.kernel_recipe_ast(skel)
        @test ast2 == ast && ast2 !== ast
        empty!(ast.args); push!(ast.args, :corrupted)     # corrupt the first copy
        ast3 = RKS.kernel_recipe_ast(skel)
        @test ast3 != ast                                  # stored source untouched
        @test filter(s -> !(s isa LineNumberNode), ast3.args)[1] ==
              :(fwd = Ref(fwdbwd[gofwd ? 1 : 2]))

        # (gate 4) capture is a value fixed at expansion — no runtime-state input, so
        # repeated reads are stable regardless of prior mutation.
        @test RKS.kernel_recipe_ast(skel) == ast2

        # (gate 5) NO Expr / Vector / Dict lives in the stored frozen form.
        stored = getfield(skel, :recipe_source)
        @test stored isa Tuple
        @test _frozen_all_immutable(stored)
        @test all(n -> n isa RKS._FrozenExpr, stored)

        # definition module retained for hygienic resolution of the captured names.
        @test RKS.kernel_module(skel) isa Module
    end

    @testset "freeze/thaw AST round-trips; no mutable node survives freezing" begin
        for src in (:(fwd = Ref(fwdbwd[gofwd ? 1 : 2])),
                    :(bwd = Ref(fwdbwd[gofwd ? 2 : 1])),
                    :(y = f(:sym, x[i], a ? b : c)))
            fr = RKS._kernel_freeze_ast(src)
            @test fr isa RKS._FrozenExpr
            @test _frozen_all_immutable(fr)                      # (gate 5) exact structure
            thawed = RKS._kernel_thaw_ast(fr)
            @test thawed == src && thawed !== src          # (gate 2) round-trip, fresh
        end

        # a thawed copy is fully detached — mutating it can't reach the frozen node.
        fr = RKS._kernel_freeze_ast(:(fwd = Ref(fwdbwd[gofwd ? 1 : 2])))
        t1 = RKS._kernel_thaw_ast(fr)
        t1.args[1] = :zzz
        @test RKS._kernel_thaw_ast(fr) == :(fwd = Ref(fwdbwd[gofwd ? 1 : 2]))

        # a QuoteNode wrapping an Expr is frozen too — no raw Expr survives (gate 5).
        qn = Expr(:(=), :x, QuoteNode(Expr(:call, :+, :a, :b)))
        frq = RKS._kernel_freeze_ast(qn)
        @test _frozen_all_immutable(frq)
        @test RKS._kernel_thaw_ast(frq) == qn
    end

    @testset "freeze soundness: mutable containers round-trip, others reject" begin
        # (gate 5, generic) raw / QuoteNode-nested Vector+Dict and a Tuple-containing-
        # mutable all freeze so NO mutable object survives, and thaw a FRESH exact copy.
        for src in (Expr(:call, :f, [1, 2, 3]),                    # raw Vector leaf
                    Expr(:call, :g, Dict(:a => 1, :b => 2)),       # raw Dict leaf
                    Expr(:(=), :x, QuoteNode([1, 2])),             # QuoteNode-nested Vector
                    Expr(:(=), :y, QuoteNode(Dict(1 => 2))),       # QuoteNode-nested Dict
                    Expr(:call, :h, (1, [2, 3], Dict(:k => 4))),   # Tuple hiding mutables
                    Expr(:call, :k, (a = 1, b = [2, 3])),          # NamedTuple hiding a Vector
                    Expr(:call, :m, :p => Dict(1 => 2)))           # Pair hiding a Dict
            fr = RKS._kernel_freeze_ast(src)
            @test _frozen_all_immutable(fr)                        # no mutable object survives
            thawed = RKS._kernel_thaw_ast(fr)
            @test thawed == src && thawed !== src                  # exact fresh round-trip
            @test _types_match(thawed, src)                        # …incl. EXACT concrete typeof
        end

        # a thawed container is fully detached — mutating it can't reach the frozen store.
        fr = RKS._kernel_freeze_ast(Expr(:call, :f, [1, 2, 3]))
        t = RKS._kernel_thaw_ast(fr)
        push!(t.args[2], 99)                                       # mutate the thawed Vector
        @test RKS._kernel_thaw_ast(fr) == Expr(:call, :f, [1, 2, 3])

        # an UNSUPPORTED mutable leaf is REJECTED, not silently retained — bare, and
        # nested inside a QuoteNode or a Tuple.
        @test_throws ArgumentError RKS._kernel_freeze_ast(Expr(:call, :f, Set([1, 2])))
        @test_throws ArgumentError RKS._kernel_freeze_ast(Expr(:call, :f, _AdvMutableLeaf(1)))
        @test_throws ArgumentError RKS._kernel_freeze_ast(QuoteNode(Set([1])))
        @test_throws ArgumentError RKS._kernel_freeze_ast(Expr(:call, :f, (1, Set([2]))))
        # a non-`Dict` AbstractDict is not generically reconstructable → REJECT (do NOT
        # silently thaw it as a base `Dict`).
        @test_throws ArgumentError RKS._kernel_freeze_ast(Expr(:call, :f, IdDict(1 => 2)))

        # concrete container types are reconstructed EXACTLY, not widened to Any.
        @test typeof(RKS._kernel_thaw_ast(RKS._kernel_freeze_ast([1, 2, 3]))) === Vector{Int}
        @test typeof(RKS._kernel_thaw_ast(RKS._kernel_freeze_ast(Dict(:a => 1)))) ===
              Dict{Symbol,Int}
        @test typeof(RKS._kernel_thaw_ast(RKS._kernel_freeze_ast((1, [2, 3])))) ===
              Tuple{Int,Vector{Int}}

        # a MUTABLE custom AbstractString reaches mutable storage → REJECT (bare,
        # QuoteNode-nested, Expr-nested); it is NEVER retained by identity.
        @test_throws ArgumentError RKS._kernel_freeze_ast(_EvilString(UInt8[0x61]))
        @test_throws ArgumentError RKS._kernel_freeze_ast(QuoteNode(_EvilString(UInt8[0x61])))
        @test_throws ArgumentError RKS._kernel_freeze_ast(Expr(:call, :f, _EvilString(UInt8[0x61])))

        # a plain String is the exceptional safe atom; a genuinely-immutable string wrapper
        # (SubString: a String + Ints) passes the GENERIC recursive field proof and is
        # retained (safe — immutable) and round-trips.
        @test RKS._kernel_thaw_ast(RKS._kernel_freeze_ast("s")) === "s"
        let ss = SubString("hello", 1, 3)
            @test _frozen_all_immutable(RKS._kernel_freeze_ast(ss))
            @test RKS._kernel_thaw_ast(RKS._kernel_freeze_ast(ss)) === ss
        end

        # generic assertion on the REAL captured fixture: no mutable object in the store.
        @test _frozen_all_immutable(getfield(StatefulRecipeSrcFixture, :recipe_source))
    end

    @testset "(4) short/long detection on the plain fixture" begin
        obj = StatefulObjFixture
        @test obj isa RKS._StatefulKernelSkeleton
        @test RKS.kernel_spec(obj) isa KernelSpec
        ms = RKS.kernel_methods(obj)
        @test length(ms) == 2
        @test (ms[1].name, ms[1].form, ms[1].argnames) ==
              (:leapfrog!, :short, ())
        @test (ms[2].name, ms[2].form, ms[2].argnames) ==
              (:refresh!, :long, (:rng,))
    end

    @testset "(4) implicit receiver: no-self ACCEPTED; self/__self__ formal REJECTS" begin
        # a method with no receiver formal is the NORMAL form now — expands cleanly.
        @test (@macroexpand @kernel ok(a) = begin
            r = a
            noself!() = begin
                r = 1
            end
        end) isa Expr
        # a formal literally named `self` or `__self__` is rejected at expansion.
        @test_throws Exception @macroexpand @kernel bad1(a) = begin
            r = a
            m!(self) = begin
                r = 1
            end
        end
        @test_throws Exception @macroexpand @kernel bad2(a) = begin
            r = a
            m!(__self__) = begin
                r = 1
            end
        end
    end

    @testset "(4)+(6) unsupported local scope rejects with the exact diagnostic" begin
        err = try
            @eval function _stateful_in_local_scope()
                @kernel inner(a) = begin
                    r = a
                    m!(x) = begin
                        r = 1
                    end
                end
            end
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("unsupported `const` declaration on local variable",
                       sprint(showerror, err))
    end

    @testset "definition module is exposed for hygienic op GlobalRefs" begin
        @test RKS.kernel_module(StatefulOneFixture) === @__MODULE__
        @test RKS.kernel_module(StatefulObjFixture) isa Module
    end

    @testset "(2) unique per-definition Token + object/view skeletons" begin
        @test RKS.kernel_token(StatefulOneFixture) !== RKS.kernel_token(StatefulTwoFixture)
        tok = RKS.kernel_token(StatefulOneFixture)
        object = RKS.KernelObject{tok,Nothing,Nothing}(nothing, nothing)
        @test RKS.kernel_token(object) === tok
        @test RKS.kernel_token(typeof(object)) === tok
        view = RKS.KernelView{typeof(object),:fwd}(object)
        @test RKS.kernel_view_path(view) === :fwd
        @test RKS.kernel_view_parent(view) === object
        @test typeof(RKS.KernelView{typeof(object),:bwd}(object)) !== typeof(view)
    end

    @testset "(3) frozen snapshot is structurally immutable + reconstructs" begin
        @kernel child(a, b) = begin
            s = a + b
        end
        snap = RKS._kernel_capture_child(:child, child)
        # structurally immutable: fields are Tuples (not a mutable Graph/Dict/Vector)
        @test snap.recipes isa Tuple
        @test snap.ports isa Tuple
        @test snap.have_names isa Tuple
        @test snap.want_names isa Tuple
        @test !hasproperty(snap, :graph)   # no live mutable graph exposed
        # preserves ALL metadata + reconstructs a matching KernelSpec for planning
        rebuilt = RKS._kernel_reconstruct(snap)
        @test rebuilt isa KernelSpec
        @test keys(rebuilt) == keys(child)
        @test length(kernel_graph(rebuilt).recipes) == length(kernel_graph(child).recipes)
        @test Tuple(rebuilt.have_names) == Tuple(child.have_names)
        @test Tuple(rebuilt.want_names) == Tuple(child.want_names)
        @test prepare(rebuilt)(1.0, 2.0) == prepare(child)(1.0, 2.0)
    end

    @testset "(3)+(4) snapshot survives original mutation AND rebinding" begin
        @kernel child(a, b) = begin
            s = a + b
        end
        snap = RKS._kernel_capture_child(:child, child)
        recipes_before = RKS._kernel_snapshot_recipe_count(snap)
        ports_before = RKS._kernel_snapshot_port_names(snap)

        # MUTATE the original child graph after capture — snapshot unchanged.
        push!(child.graph.recipes, first(child.graph.recipes))
        @test length(child.graph.recipes) == recipes_before + 1    # original changed
        @test RKS._kernel_snapshot_recipe_count(snap) == recipes_before

        # REBIND the original `child` binding to a different spec — snapshot unchanged.
        @kernel replacement(a) = begin
            s = a
        end
        child = replacement                                        # real rebind
        @test child === replacement
        @test RKS._kernel_snapshot_recipe_count(snap) == recipes_before
        @test RKS._kernel_snapshot_port_names(snap) == ports_before
    end

    @testset "(7) methodless expansion identity (normalized macroexpand)" begin
        # The methodless macro path is `esc(Expr(:(=), name, _kernel_expand(...)))`,
        # unchanged by the stateful routing guard. Compare the macro's expansion to
        # the stateless helper's output built from the SAME parsed definition parts,
        # with line numbers stripped and per-expansion gensym counters normalized.
        norm(ex) = replace(string(Base.remove_linenums!(deepcopy(ex))), r"#\d+" => "#N")
        def = :(plain(f, x) = begin
            y = f(x)
        end)
        actual = @macroexpand1 @kernel plain(f, x) = begin
            y = f(x)
        end
        nm, inp, sig, posn, blk = RKS._kernel_definition_parts(def)
        ref = Expr(:(=), nm, RKS._kernel_expand(blk, inp, sig))
        @test norm(actual) == norm(ref)
        # and no stateful codegen leaked into the methodless expansion
        @test !occursin("_StatefulKernelSkeleton", string(actual))
        @test !occursin("KernelObject", string(actual))
    end
end
