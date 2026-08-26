# STRENGTHENED STRUCTURAL gate + lexical-shadowing inventory for the A/B/C @kernel NUTS fixture
# (+ 22:46 source-contract corrections). Parses the fixture as TEXT (Meta.parseall) — does NOT eval
# @kernel, so it runs while the source-capture substrate is being implemented.
using Test

const FIX = joinpath(@__DIR__, "nuts_kernel_authoring_fixture.jl")
const SRC = read(FIX, String)
const AST = Meta.parseall(SRC; filename = FIX)

_dropln(xs) = filter(x -> !(x isa LineNumberNode), xs)
_walk(f, x) = (f(x); x isa Expr && foreach(a -> _walk(f, a), x.args); nothing)
_signame(s) = s isa Symbol ? s : (s isa Expr && s.head in (:call, :where, :(::), :curly) ? _signame(s.args[1]) : nothing)
is_methoddef(st) = st isa Expr && ((st.head === :(=) && st.args[1] isa Expr && st.args[1].head === :call) || st.head === :function)
methodname(st) = _signame(st.args[1]); methodsig(st) = st.args[1]; methodbody(st) = st.args[2]
gofwd_ternary(x) = x isa Expr && x.head === :if && x.args[1] === :gofwd

function kernel_blocks()
    blocks = Dict{Symbol,Any}()
    _walk(AST) do x
        x isa Expr && x.head === :macrocall && x.args[1] === Symbol("@kernel") || return
        eq = x.args[end]; eq isa Expr && eq.head === :(=) || return
        name = _signame(eq.args[1]); sig = eq.args[1]; body = eq.args[2]
        srcs = Symbol[]
        for a in (sig isa Expr && sig.head === :call ? sig.args[2:end] : [])
            if a isa Expr && a.head === :parameters
                for p in a.args; s = _signame(p isa Expr && p.head === :kw ? p.args[1] : p); s isa Symbol && push!(srcs, s); end
            else
                s = _signame(a isa Expr && a.head === :kw ? a.args[1] : a); s isa Symbol && push!(srcs, s)
            end
        end
        fields = Symbol[]; methods = Any[]
        for st in (body isa Expr && body.head === :block ? _dropln(body.args) : Any[body])
            is_methoddef(st) ? push!(methods, st) :
                (st isa Expr && st.head === :(=) && st.args[1] isa Symbol && push!(fields, st.args[1]))
        end
        blocks[name] = (; sig, sources = srcs, fields, methods, body)
    end
    blocks
end
const CMP = Set([:(==),:(!=),:(<=),:(>=),:(===),:(!==),:(.==),:(.!=),:(.<=),:(.>=)])
is_assign(h) = h isa Symbol && (s = String(h); endswith(s, "=") && !(h in CMP))
function formals(sig)
    out = Symbol[]; call = sig
    call isa Expr && call.head === :where && (call = call.args[1])
    call isa Expr && call.head === :call || return out
    for a in call.args[2:end]
        if a isa Expr && a.head === :parameters
            for p in a.args
                p isa Expr && p.head === :(...) ? push!(out, _signame(p.args[1])) : push!(out, _signame(p isa Expr && p.head === :kw ? p.args[1] : p))
            end
        else
            push!(out, _signame(a isa Expr && a.head === :kw ? a.args[1] : a))
        end
    end
    filter(!isnothing, out)
end
srcof(ex) = string(ex)
method_named(b, n) = only(filter(m -> methodname(m) === n, b.methods))

blocks = kernel_blocks()
@testset "A/B/C + 22:46 source-contract gate" begin
    for k in (:euclidean_phasepoint, :leapfrog!, :nuts_state, Symbol("nuts!!"),
              :dual_averaging_state, :welford_var)
        @test haskey(blocks, k)
    end
    @test !haskey(blocks, Symbol("copy!!"))   # copy!! is the RK-CORE intrinsic — NOT authored in the fixture

    # ---- FORM A: leapfrog! RK @kernel, exact 3-line mom/pos/mom -----------------------------------
    lf = blocks[:leapfrog!]; @test :stepsize in formals(lf.sig)
    lstmts = _dropln(lf.body.args)
    @test length(lstmts) == 3
    heads = map(s -> (s isa Expr && s.head === :macrocall && s.args[1] === Symbol("@__dot__")) ? _dropln(s.args)[end].head : nothing, lstmts)
    @test heads == [:(-=), :(+=), :(-=)]
    lb = srcof(lf.body)
    @test occursin("phasepoint.dham_dpos", lb) && occursin("phasepoint.dham_dmom", lb)
    println("  (A) leapfrog! RK @kernel, exact 3-line Stormer-Verlet. OK")

    # ---- FORM B (+22:46): implicit-field, no-Ref, TWO-DIRECT-BRANCH direction, runtime rng ---------
    ns = blocks[:nuts_state]; nb = srcof(ns.body)
    @test !occursin("Ref(", nb) && !occursin("RefValue", nb) && !occursin("fwdbwd", nb)
    @test :fwd in ns.fields && :bwd in ns.fields && :init in ns.sources
    @test occursin("fwd = deepcopy(init)", nb) && occursin("bwd = deepcopy(init)", nb)
    # (B-rng) rng is a RUNTIME arg, NOT a source/field.
    @test !(:rng in ns.sources) && !(:rng in ns.fields)

    # (B-dir) two-direct-branch only: accumulate violations in a single walk.
    per_method = Dict{Symbol,Int}()
    bad_branch = Ref(false); sel_local = Ref(false); ternary_value = Ref(false)
    named_fwdbwd_local = Ref(false)
    for m in ns.methods
        mn = methodname(m)
        _walk(methodbody(m)) do x
            x isa Expr || return
            if gofwd_ternary(x)
                per_method[mn] = get(per_method, mn, 0) + 1
                for br in x.args[2:end]
                    (br === :fwd || br === :bwd) && (bad_branch[] = true)              # bare endpoint value
                    (br isa Expr && br.head in (:call,:macrocall,:block,:(&&),:(||),:if)) || (bad_branch[] = true)
                end
            end
            if x.head === :(=) && x.args[1] isa Symbol
                x.args[2] isa Expr && gofwd_ternary(x.args[2]) && (sel_local[] = true)  # x = gofwd ? .. : ..
                x.args[1] in (:forward, :backward) && (named_fwdbwd_local[] = true)     # banned selection names
            end
            if x.head === :.                                                            # ternary as property root
                gofwd_ternary(x.args[1]) && (ternary_value[] = true)
            elseif x.head === :call                                                     # ternary as call actual
                any(a -> gofwd_ternary(a), x.args[2:end]) && (ternary_value[] = true)
            end
        end
    end
    @test !bad_branch[]                     # every gofwd branch is an operation with concrete endpoints
    @test !sel_local[]                      # no `local = gofwd ? fwd : bwd`
    @test !named_fwdbwd_local[]             # no forward/backward selection local
    @test !ternary_value[]                  # ternary result never used as property root / call actual
    @test get(per_method, :step!, 0) >= 2   # step!: backward-negate branch + finish! branch
    @test get(per_method, :flip!, 0) == 1   # flip!: one direction branch
    @test sum(values(per_method)) >= 3
    for m in ns.methods; @test !(:forward in formals(methodsig(m))) && !(:backward in formals(methodsig(m))); end

    # (B-implicit) no self/__self__ formal; no self.X/__self__.X access.
    bad_self = Ref(false)
    for m in ns.methods
        fs = formals(methodsig(m)); (:self in fs || Symbol("__self__") in fs) && (bad_self[] = true)
        _walk(methodbody(m)) do x
            x isa Expr && x.head === :. && (x.args[1] === :self || x.args[1] === Symbol("__self__")) && (bad_self[] = true)
        end
    end
    @test !bad_self[]

    # (B-mut) SOURCE-CALL CENSUS (spelling ONLY — NOT a hygienic-registration proof; the resolver/MethodIR
    #         gate proves captured GlobalRef/Token identity + rejects rebinds). Every hot mutating
    #         (!-suffixed, non-operator, unqualified) callee must be spelled as an authored @kernel, a
    #         captured nuts_state sibling, or the RK-CORE strong-update intrinsic copy!!. A QUALIFIED
    #         primitive (Base.fill!) is a `.`-call, not a bare Symbol, so it is skipped. Any bare
    #         unqualified !-callee outside those sets (restore!/rcopy!/reset_one_tree!/helper) is rejected.
    authored_kernels = Set(string(k) for k in keys(blocks))
    sibling_methods = Set(string(methodname(m)) for m in ns.methods)
    core_intrinsics = Set(["copy!!"])
    bad_mut = String[]
    for m in ns.methods
        _walk(methodbody(m)) do x
            x isa Expr && x.head === :call && x.args[1] isa Symbol || return
            f = string(x.args[1]); endswith(f, "!") && f != "!" && f != "!=" || return
            (f in authored_kernels || f in sibling_methods || f in core_intrinsics) || push!(bad_mut, f)
        end
    end
    @test isempty(bad_mut)
    @test any(m -> methodname(m) === :reset!, ns.methods)
    @test occursin("step_f(ep)", nb)                         # step_f invoked on a concrete endpoint
    @test !occursin("reset_one_tree!", SRC)                  # opaque helper inlined away
    # copy!! is the ONLY reset/proposal-restore path: no authored fieldwise copy, no @kernel copy!!/rcopy!!.
    @test occursin("copy!!(", nb) && !occursin("@kernel copy!!", SRC) && !occursin("rcopy!!", SRC)
    @test occursin("Base.fill!", nb)                         # qualified visible primitive for buffer clears

    # (B-own) the pinned ownership policy is EXPECTED METADATA (documented, NOT hand-implemented): the
    #         shared authority + the complete owned set are named in the source.
    for f in ("pot_f", "grad_f", "metric", "chol_metric"); @test occursin(f, SRC); end       # shared
    for f in ("pos", "mom", "dpot_dpos", "dkin_dmom", "dham_dpos", "dham_dmom", "pot", "kin", "ham")
        @test occursin(f, SRC)                                                                # owned
    end

    # (B-bind) REAL parsed constructor/binding expressions (SPELLING; identity proven later by resolver):
    #          partial(leapfrog!; stepsize) + nuts_state(init; step_f=..., stats_f).
    has_step_partial = Ref(false); has_nuts_binding = Ref(false)
    _walk(AST) do x
        x isa Expr && x.head === :call || return
        if x.args[1] === :partial && any(a -> a === :leapfrog!, x.args) &&
           any(a -> a isa Expr && a.head === :parameters &&
                    any(p -> p === :stepsize || (p isa Expr && p.head === :kw && p.args[1] === :stepsize), a.args), x.args)
            has_step_partial[] = true
        end
        x.args[1] === :nuts_state && any(a -> a isa Expr && a.head === :parameters, x.args) && (has_nuts_binding[] = true)
    end
    @test has_step_partial[]
    @test has_nuts_binding[]

    # (B-noforce) `force` removed from step! (dead after registered reset).
    @test !(:force in formals(methodsig(method_named(ns, :step!))))

    # (B-derived) NO nested method writes the DERIVED field `diverged` (authored once as the recipe);
    #             reset! writes `dham` ONLY among the two.
    wrote_diverged = Ref(false)
    for m in ns.methods
        _walk(methodbody(m)) do x
            x isa Expr && is_assign(x.head) && x.args[1] === :diverged && (wrote_diverged[] = true)
        end
    end
    @test !wrote_diverged[]
    reset_writes = Set{Symbol}()
    _walk(methodbody(method_named(ns, :reset!))) do x
        x isa Expr && is_assign(x.head) && x.args[1] isa Symbol && push!(reset_writes, x.args[1])
    end
    @test :dham in reset_writes && !(:diverged in reset_writes)

    # (B-reqkw) step_f is a REQUIRED keyword (bare symbol in :parameters, no :kw default); stats_f optional.
    params = ns.sig.args[findfirst(a -> a isa Expr && a.head === :parameters, ns.sig.args)]
    @test any(p -> p === :step_f, params.args)                                           # required (no default)
    @test any(p -> p isa Expr && p.head === :kw && p.args[1] === :stats_f, params.args)   # optional
    @test !(:step_f in ns.fields)                                                         # a source, not a field

    # (B-f32f64) type-preserving scalar source forms: min_dham=oftype(init.ham,-1000); dham=zero(init.ham);
    #            no hardcoded Float64 literal for either (F32/F64 source-form discriminator).
    @test occursin("oftype(init.ham", SRC)
    @test occursin("zero(init.ham)", nb)
    @test !occursin("min_dham = -1000", SRC) && !occursin("dham = 0.", nb)

    # (B-fields) reference field spellings + concrete endpoint param `ep` threaded through recursion.
    nsnames = Set(ns.fields) ∪ Set(ns.sources)
    for f in (:gofwd,:may_sample,:may_continue,:fwd,:bwd,:trees,:proposals,:dham,:diverged,
              :init,:max_depth,:min_dham,:step_f,:stats_f); @test f in nsnames; end
    @test :ep in formals(methodsig(method_named(ns, :finish!))) && :rng in formals(methodsig(method_named(ns, :finish!)))
    @test :ep in formals(methodsig(method_named(ns, :start!)))  && :rng in formals(methodsig(method_named(ns, :start!)))
    @test Set(blocks[:dual_averaging_state].fields) ⊇ Set([:m,:H,:mu,:log_current,:log_final,:current,:final])
    @test Set(blocks[:welford_var].fields) ⊇ Set([:n,:mean,:var])
    println("  (B) no Ref/fwdbwd; explicit init/fwd/bwd; TWO-DIRECT gofwd branches (no endpoint value/")
    println("      alias/selection-local); rng runtime not state; core copy!! / visible reset!; implicit fields. OK")

    # ---- FORM C: public @kernel nuts!!(state; rng) threads rng + returns the SAME object ----------
    nb2 = blocks[Symbol("nuts!!")]
    @test :state in formals(nb2.sig) && :rng in formals(nb2.sig)
    c = srcof(nb2.body)
    @test occursin("return state", c)
    @test occursin("step!(state, rng)", c) || occursin("step!(state; rng", c)   # rng consumed, not ignored
    println("  (C) @kernel nuts!!(state; rng): threads runtime rng into step!(state, rng), returns state. OK")

    # ---- @node preserved; @reactive-as-macro absent; free @kernel leapfrog!/rcopy!!/nuts!! ----------
    @test occursin("@node(logdet(chol_metric))", SRC)
    has_reactive = Ref(false)
    _walk(AST) do x; x isa Expr && x.head === :macrocall && x.args[1] === Symbol("@reactive") && (has_reactive[] = true); end
    @test !has_reactive[]
    @test occursin("@kernel leapfrog!", SRC) && occursin("@kernel nuts!!", SRC)
    println("  (@node) preserved; @reactive absent (macro); free @kernel leapfrog!/nuts!!; copy!! is core. OK")
end

# --- NON-VACUOUS lexical-shadowing inventory (nuts_state) --------------------------------------------
println("\n=== LEXICAL-SHADOWING INVENTORY (nuts_state) ===")
let b = blocks[:nuts_state], fieldset = Set(b.fields) ∪ Set(b.sources)
    for m in b.methods
        fs = formals(methodsig(m)); locals = Symbol[]; reads = Set{Symbol}(); writes = Set{Symbol}()
        _walk(methodbody(m)) do x
            if x isa Expr && is_assign(x.head) && x.args[1] isa Symbol
                s = x.args[1]; (s in fieldset) ? push!(writes, s) : push!(locals, s)
            elseif x isa Expr && is_assign(x.head) && x.args[1] isa Expr && x.args[1].head === :. &&
                   x.args[1].args[1] isa Symbol && x.args[1].args[1] in fieldset
                push!(writes, x.args[1].args[1])
            elseif x isa Symbol && x in fieldset
                push!(reads, x)
            end
        end
        shadow = union(Set(fs), Set(locals))
        ur = sort(collect(setdiff(reads, shadow)); by=string); uw = sort(collect(writes); by=string)
        println(rpad(string(methodname(m)), 14), " formals=", isempty(fs) ? "()" : fs,
                "  locals(shadow)=", isempty(locals) ? "()" : sort(unique(locals); by=string))
        println(rpad("", 14), "   field WRITES=", isempty(uw) ? "()" : uw, "   field READS=", isempty(ur) ? "()" : ur)
    end
end
println("\nSTRENGTHENED STRUCTURAL GATE PASS (source-form / Meta.parseall — does not eval @kernel).")
println("Construction verified SEPARATELY on the c998ec3 Inc1 substrate (all six @kernel blocks define:")
println("@node, implicit fields + __self__ receiver, free-kernel leapfrog!/nuts!!, stateful nuts_state).")
println("Source-form + construction only — NO MethodIR/lowering/execution/parity/alloc/perf claim.")
