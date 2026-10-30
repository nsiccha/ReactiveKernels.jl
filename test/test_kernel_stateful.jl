# Method-bearing `@kernel` objects and the pure parts of the authoring front
# end: the method discriminator and extraction (implicit receiver; `__self__` only
# as a sibling object-pass actual), pure endpoint straightness, `@node`
# port-lifting, deterministic unsupported-local-scope rejection, the frozen
# detached owner-spec snapshot, and the rejection of stateful free methods.
#
# A method-bearing `@kernel` binds a `const` (a stable owner binding), so it MUST be
# defined at module top level — the fixtures below live outside the `@testset`
# (which is a local scope); the local-scope rejection is asserted separately.
#
# Isolated in a MODULE so the top-level fixtures cannot shadow package exports in
# the shared `Pkg.test` Main; it still nests under runtests.jl's outer `@testset`.
module TestKernelStateful
using ReactiveKernels, Test

const RKS = ReactiveKernels

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

# Spoof module for marker identity-confirmation: same NAMES, different bindings.
module MarkerSpoof
deepcopy(x) = x                    # NOT Base.deepcopy
macro node(ex); esc(ex); end       # NOT ReactiveKernels.@node
end

@testset "method-bearing @kernel objects" begin
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

    @testset "pure-endpoint straightness: authored plate cells are RK subgraphs" begin
        straight = RKS._kernel_endpoint_has_nonstraight
        # A `plate(...) do` cell is lowered by the recipe pass, not an opaque
        # closure: it keeps the method a pure endpoint (lazy `?:` inside).
        @test !straight(:(plate(y, eta) do observed, e
            cell::Float64 = observed == 1 ? log(cuts[1] - e) : log(cuts[observed] - e)
            cell
        end))
        @test !straight(:(ReactiveKernels.plate(y, mu) do observed, m
            observed - m
        end))
        # Control flow inside the cell, or in a threaded argument, is still
        # control flow; an ordinary closure or another do-block stays opaque.
        @test straight(:(plate(y, eta) do observed, e
            for i in 1:3; e = e + i; end
            e
        end))
        @test straight(:(plate(map(x -> x + 1, y), eta) do observed, e
            observed - e
        end))
        @test straight(:(mapreduce(y, eta) do observed, e
            observed - e
        end))
        @test straight(:(map(x -> x + 1, y)))
        # A `let` only scopes its bindings; what it contains still decides.
        @test !straight(:(observed == 0 ? let a = n * x, b = a + 1
            a <= 700 ? log(b) : a
        end : 0.0))
        @test straight(:(let a = x
            if a > 0; a = 1.0; end
            a
        end))
        @test straight(:(let f = y -> y + 1
            f(x)
        end))
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
        # `__self__.field` (field qualifier) is rejected.
        @test_throws ArgumentError RKS._kernel_extract_method(
            :(m!(x) = (__self__.r = x)), :short)
        # a bare `__self__` outside an object-pass actual is rejected.
        @test_throws ArgumentError RKS._kernel_extract_method(:(m!() = (y = __self__)), :short)
        @test_throws ArgumentError RKS._kernel_extract_method(:(m!() = sib!(x, __self__)), :short)
    end

    @testset "@node transformer soundness (identity / collision / control-flow)" begin
        # HAPPY PATH: a genuine straight-line `@node` is PORT-LIFTED into a distinct
        # schedulable graph node. Non-vacuous graph gate: 2 recipes, 3 ports (:m, the
        # lifted node, :kin), the node being the port that is neither :m nor :kin.
        @kernel withnode(m) = begin
            kin = 0.5 * (@node(abs(m)) + 2.0)
        end
        @test withnode isa KernelSpec
        @test length(kernel_graph(withnode).recipes) == 2
        ks = collect(keys(withnode))
        @test length(ks) == 3 && :m in ks && :kin in ks
        @test length(filter(p -> p !== :m && p !== :kin, ks)) == 1   # exactly one lifted node
        # a @node-FREE recipe block is returned byte-identical (the SAME object)
        blk = :(begin
            y = f(x)
        end)
        @test RKS._kernel_lift_nodes(blk, @__MODULE__) === blk
        @test (@node 41 + 1) == 42                 # harmless identity outside a kernel

        # FIX (A): IDENTITY-AWARE — a foreign `@node` (name only) is NOT promoted as RK
        # semantics; it expands as its own macro, leaving NO hidden node.
        @kernel spoofednode(m) = begin
            kin = 0.5 * (MarkerSpoof.@node(abs(m)) + 2.0)
        end
        @test spoofednode isa KernelSpec
        @test length(kernel_graph(spoofednode).recipes) == 1         # only `kin`; no lifted node
        @test Set(keys(spoofednode)) == Set((:m, :kin))

        # FIX (B): COLLISION-FREE — the generated node name is a gensym, so it cannot
        # alias an authored port even one spelled like the old `#node#1`. NON-VACUOUS
        # gate: EXACTLY four ports (x, authored #node#1, a DISTINCT generated node, y),
        # and the authored vs lifted graph Values carry DIFFERENT ids (the old shared-port
        # bug had only 3 ports — the generated node collided onto authored `#node#1`).
        @kernel collidenode(x) = begin
            var"#node#1" = x + 1
            y = @node(x + 2)
        end
        @test collidenode isa KernelSpec
        kc = collect(keys(collidenode))
        @test length(kc) == 4                                        # x, #node#1, generated, y
        @test :x in kc && Symbol("#node#1") in kc && :y in kc
        gen = filter(p -> !(p in (:x, :y, Symbol("#node#1"))), kc)
        @test length(gen) == 1                                       # one DISTINCT generated node
        @test gen[1] !== Symbol("#node#1")                           # NOT aliasing the authored port
        @test collidenode.ports[Symbol("#node#1")].id != collidenode.ports[gen[1]].id
        # `y = @node(x+2)` lifts to a generated-node recipe + a bare-identity `y = <node>`; the
        # bare identity is now a CANONICAL ALIAS (RK 2026-08-27), not an identity recipe — so two
        # recipes (#node#1, generated node), and `y` shares the generated node's canonical Value.
        @test length(kernel_graph(collidenode).recipes) == 2
        @test canon_id(kernel_graph(collidenode), collidenode.ports[:y].id) ==
              canon_id(kernel_graph(collidenode), collidenode.ports[gen[1]].id)

        # FIX (C): a genuine `@node` in a NON-straight-line CONTROL context (branch/loop)
        # is REJECTED, not silently made unconditional in the static graph.
        @test_throws Exception @macroexpand @kernel condnode(c, x) = begin
            y = c ? @node(abs(x)) : x
        end
        @test_throws Exception @macroexpand @kernel andnode(c, x) = begin
            y = c && @node(abs(x))
        end
        @test_throws Exception @macroexpand @kernel loopnode(x) = begin
            y = [@node(abs(x)) for _ in 1:2]
        end
        # ...and in a DEFERRED/LEXICAL scope (lambda/let/do), where a hoisted node would
        # escape the local binding it references — also REJECTED.
        @test_throws Exception @macroexpand @kernel lamnode(g) = begin
            f = x -> @node(g(x))
        end
        @test_throws Exception @macroexpand @kernel letnode(x) = begin
            y = let z = x
                @node(abs(z))
            end
        end
        @test_throws Exception @macroexpand @kernel donode(g, x) = begin
            y = map(g) do e
                @node(abs(x))
            end
        end
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
        # Generated source retains its lexical Module as an identity-bearing
        # quoted leaf. Copy only AST nodes before stripping their line numbers.
        copy_ast(ex) = ex
        copy_ast(ex::Expr) = Expr(ex.head, map(copy_ast, ex.args)...)
        copy_ast(ex::QuoteNode) = QuoteNode(copy_ast(ex.value))
        norm(ex) = replace(string(Base.remove_linenums!(copy_ast(ex))), r"#\d+" => "#N")
        def = :(plain(f, x) = begin
            y = f(x)
        end)
        actual = @macroexpand1 @kernel plain(f, x) = begin
            y = f(x)
        end
        nm, inp, sig, posn, rawsig, blk = RKS._kernel_definition_parts(def)
        ref = Expr(:(=), nm, RKS._kernel_expand(blk, inp, sig, @__MODULE__))
        @test norm(actual) == norm(ref)
        # and no stateful codegen leaked into the methodless expansion
        @test !occursin("_StatefulKernelSkeleton", string(actual))
        @test !occursin("KernelObject", string(actual))
    end

    @testset "owner-spec detached snapshot: mutation isolation + freshness (E)" begin
        # kernel_spec returns a FRESH reconstructed planning spec each call
        s1 = RKS.kernel_spec(StatefulObjFixture)
        s2 = RKS.kernel_spec(StatefulObjFixture)
        @test s1 isa KernelSpec && s2 isa KernelSpec
        @test s1 !== s2                                       # distinct objects
        @test keys(s1) == keys(s2)                            # equal field/port sets
        # mutating a returned spec's graph cannot change a later read's field set
        before = collect(keys(RKS.kernel_spec(StatefulObjFixture)))
        empty!(kernel_graph(s1).recipes)                      # corrupt the returned graph
        @test collect(keys(RKS.kernel_spec(StatefulObjFixture))) == before
        # immutable port-name surface (a detached Tuple)
        @test RKS.kernel_port_names(StatefulObjFixture) isa Tuple
        @test :ham in RKS.kernel_port_names(StatefulObjFixture)
    end

    @testset "stateful free methods are rejected; objects keep only pure parts" begin
        @test_throws ArgumentError @macroexpand @kernel bump!(p; step) = begin
            @. p.pos += step
        end
        @test_throws ArgumentError @macroexpand @kernel advance!!(state; rng) = begin
            step!(state, rng)
            return state
        end
        @test fieldnames(RKS._StatefulKernelSkeleton) ==
              (:name, :spec_snapshot, :endpoint_templates)
        @test StatefulObjFixture isa RKS._KernelObjectSkeleton
        @test RKS._kernel_pure_skeleton(StatefulObjFixture) === StatefulObjFixture
    end
end

end # module TestKernelStateful
