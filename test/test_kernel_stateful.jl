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

# --- top-level stateful fixtures (method-bearing ⇒ const ⇒ top-level only) ---
@kernel StatefulObjFixture(ham, pos, mom) = begin
    derived = ham
    leapfrog!(self) = begin
        @. self.pos += self.mom
    end
    function refresh!(self, rng)
        self.mom = rng
    end
end

# kwargs/defaults, typed args + `where`, and a return `::` annotation.
@kernel StatefulSigFixture(a) = begin
    r = a
    fit!(self, x; target = 0.8) = begin
        self.r = x + target
    end
    m!(self::S, x::S) where {S} = begin
        self.r = x
    end
    function n!(self)::Nothing
        self.r = 0
        nothing
    end
end

@kernel StatefulOneFixture(a) = begin
    r = a
    m!(self) = begin
        self.r = 1
    end
end
@kernel StatefulTwoFixture(a) = begin
    r = a
    m!(self) = begin
        self.r = 1
    end
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
        # fit!(self, x; target = 0.8): self, positional x, keyword target
        @test (ms[1].name, ms[1].form, ms[1].self, ms[1].argnames) ==
              (:fit!, :short, :self, (:x, :target))
        # m!(self::S, x::S) where {S}: typed args, self peeled through `where`
        @test (ms[2].name, ms[2].form, ms[2].self, ms[2].argnames) ==
              (:m!, :short, :self, (:x,))
        # function n!(self)::Nothing: return annotation peeled
        @test (ms[3].name, ms[3].form, ms[3].self, ms[3].argnames) ==
              (:n!, :long, :self, ())
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
        m = RKS._kernel_extract_method(:(m!(self::S, x::S) where {S<:Real} = x), :short)
        @test occursin("where", string(m.signature))
        @test occursin("S", string(m.signature)) && occursin("Real", string(m.signature))
        @test !occursin("where", string(m.call))          # peeled call has no where
        @test m.call.head === :call
        @test (m.name, m.self, m.argnames) == (:m!, :self, (:x,))

        # return `::` annotation preserved in the signature, absent from the call
        n = RKS._kernel_extract_method(
            :(function n!(self)::Nothing; nothing; end), :long)
        @test occursin("Nothing", string(n.signature))    # ::Nothing retained
        @test n.call.head === :call
        @test string(n.call) == string(:(n!(self)))        # peeled call drops ::Nothing
        @test (n.name, n.self, n.argnames) == (:n!, :self, ())
    end

    @testset "(4) short/long detection on the plain fixture" begin
        obj = StatefulObjFixture
        @test obj isa RKS._StatefulKernelSkeleton
        @test RKS.kernel_spec(obj) isa KernelSpec
        ms = RKS.kernel_methods(obj)
        @test length(ms) == 2
        @test (ms[1].name, ms[1].form, ms[1].self, ms[1].argnames) ==
              (:leapfrog!, :short, :self, ())
        @test (ms[2].name, ms[2].form, ms[2].self, ms[2].argnames) ==
              (:refresh!, :long, :self, (:rng,))
    end

    @testset "(4) a method with no explicit self rejects at expansion" begin
        @test_throws Exception @macroexpand @kernel bad(a) = begin
            r = a
            noself!() = begin
                r = 1
            end
        end
    end

    @testset "(4)+(6) unsupported local scope rejects with the exact diagnostic" begin
        err = try
            @eval function _stateful_in_local_scope()
                @kernel inner(a) = begin
                    r = a
                    m!(self) = begin
                        self.r = 1
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
        nm, inp, sig, blk = RKS._kernel_definition_parts(def)
        ref = Expr(:(=), nm, RKS._kernel_expand(blk, inp, sig))
        @test norm(actual) == norm(ref)
        # and no stateful codegen leaked into the methodless expansion
        @test !occursin("_StatefulKernelSkeleton", string(actual))
        @test !occursin("KernelObject", string(actual))
    end
end
