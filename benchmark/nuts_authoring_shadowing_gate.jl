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
    for k in (:euclidean_phasepoint, :leapfrog!, Symbol("refresh_momentum!!"), Symbol("nuts_stats!"),
              :nuts_state, Symbol("nuts!!"), :dual_averaging_state, :welford_var)
        @test haskey(blocks, k)
    end
    @test !haskey(blocks, Symbol("copy!!"))   # copy!! is the RK-CORE intrinsic — NOT authored in the fixture

    # ---- GRAD: euclidean_phasepoint takes ONLY grad_f (pot_f DROPPED — no required-but-unused authority) ----
    ep0 = blocks[:euclidean_phasepoint]; epf = formals(ep0.sig)
    @test :grad_f in epf && :metric in epf && :pos in epf && :mom in epf
    @test !(:pot_f in epf)                                             # pot_f absent from the signature
    # AST-based (formatting-robust): find the assignment (pot, dpot_dpos) = grad_f(pos).
    grad_recipe = Ref(false); pot_f_producer = Ref(false)
    _walk(ep0.body) do x
        x isa Expr || return
        if x.head === :(=) && x.args[1] isa Expr && x.args[1].head === :tuple && x.args[1].args == [:pot, :dpot_dpos] &&
           x.args[2] isa Expr && x.args[2].head === :call && x.args[2].args == [:grad_f, :pos]
            grad_recipe[] = true
        end
        x.head === :call && x.args[1] === :pot_f && (pot_f_producer[] = true)
    end
    @test grad_recipe[]                                              # ONE destination-bound grad recipe (pot+dpot)
    @test !pot_f_producer[]                                          # no redundant pot-only producer call
    # (grad-f32) kinetic 1/2 is typed — bare .5/0.5 would widen a Float32 phase point to Float64.
    epb = srcof(ep0.body)
    @test occursin("oftype(pot, 0.5)", epb) && !occursin("0.5 * (", epb)
    println("  (grad) euclidean_phasepoint(grad_f, metric, pos, mom): one grad recipe, pot_f absent, typed 1/2. OK")

    # ---- FORM A: leapfrog! RK @kernel, exact 3-line mom/pos/mom -----------------------------------
    lf = blocks[:leapfrog!]; @test :stepsize in formals(lf.sig)
    lstmts = _dropln(lf.body.args)
    @test length(lstmts) == 3
    heads = map(s -> (s isa Expr && s.head === :macrocall && s.args[1] === Symbol("@__dot__")) ? _dropln(s.args)[end].head : nothing, lstmts)
    @test heads == [:(-=), :(+=), :(-=)]
    lb = srcof(lf.body)
    @test occursin("phasepoint.dham_dpos", lb) && occursin("phasepoint.dham_dmom", lb)
    # (A-f32) typed half-step coefficient — bare .5/0.5 would widen a Float32 stepsize to Float64.
    @test occursin("oftype(stepsize, 0.5)", lb)                      # half-kicks use a typed 1/2
    @test !occursin("0.5 * stepsize", lb) && !occursin("0.5 * (", lb)   # no bare untyped 1/2 at a kick site
    println("  (A) leapfrog! RK @kernel, exact 3-line Stormer-Verlet, typed half-step. OK")

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
    # (B-reset-minimal) Bind the COMPLETE parsed reset! effect body, not a census of a few spellings. This
    #                   admits only the seven authored scalar resets followed by the four LIVE endpoint seeds.
    #                   Any helper call, loop, eachindex/foreach/broadcast form, alternate fill, extra write, or
    #                   reordering changes this AST and rejects — including an eager clear hidden in a sibling.
    expected_reset = Any[
        :(gofwd = true), :(may_sample = true), :(may_continue = true),
        :(dham = zero(init.ham)), :(n_steps = 0), :(reached_depth = 0),
        :(acceptance_rate = zero(init.ham)),
        :(copy!!(fwd, init)), :(copy!!(bwd, init)),
        :(copy!!(proposals[1], init)), :(copy!!(proposals[length(proposals)], init)),
    ]
    reset_effects(body) = body isa Expr && body.head === :block ? _dropln(body.args) : Any[body]
    reset_exact(body) = reset_effects(body) == expected_reset
    @test reset_exact(methodbody(method_named(ns, :reset!)))

    # Load-bearing negative controls: mutually different eager-clear spellings all fail the SAME complete-body
    # contract. These are parser-level controls only; the actual fixture assertion above is the production gate.
    for extra in (:(reset_all!()),
                  :(for i in eachindex(proposals); copy!!(proposals[i], init); end),
                  :(foreach(p -> copy!!(p, init), proposals)),
                  :(trees .= zero.(trees)),
                  :(fill!(trees, init)),
                  :(LinearAlgebra.fill!(trees, init)))
        @test !reset_exact(Expr(:block, expected_reset..., extra))
    end
    @test !reset_exact(Expr(:block, expected_reset[2:end]...))                # missing scalar reset
    reordered_reset = copy(expected_reset)
    reordered_reset[1], reordered_reset[2] = reordered_reset[2], reordered_reset[1]
    @test !reset_exact(Expr(:block, reordered_reset...))                      # reordered scalar reset
    @test !reset_exact(Expr(:block, expected_reset..., :(copy!!(fwd, init)))) # duplicate/extra live copy
    smuggled_reset = copy(expected_reset)
    smuggled_reset[4] = :(dham = (foreach(p -> copy!!(p, init), proposals); zero(init.ham)))
    @test !reset_exact(Expr(:block, smuggled_reset...))                       # work hidden in an allowed RHS

    # (B-own) the pinned ownership policy is EXPECTED METADATA (documented, NOT hand-implemented): the
    #         shared authority + the complete owned set are named in the source.
    for f in ("grad_f", "metric", "chol_metric"); @test occursin(f, SRC); end                # shared (NO pot_f)
    # pot_f absence is proven on the euclidean_phasepoint signature/body above (not a text scan — comments
    # legitimately mention "no pot_f"); assert it is not a formal of ANY authored @kernel block.
    @test all(b -> !(:pot_f in formals(b.sig)), values(blocks))
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
    # (B-Tforms) tree/DA/Welford construction is T-derived — sentinel oftype(ham,-Inf); DA defaults oftype(init,…);
    #            Welford over a template vector (zero(template)); no hardcoded Float64 constructors.
    @test occursin("oftype(phasepoint.ham, -Inf)", SRC) && !occursin("fill(-Inf", SRC)      # tree sentinel typed
    @test occursin("oftype(init, .8)", SRC) && occursin("oftype(init, .05)", SRC)           # DA defaults typed
    @test occursin("welford_var(template::AbstractVector)", SRC) && occursin("zero(template)", SRC)  # Welford T-template
    @test !occursin("welford_var(dim::Int)", SRC) && !occursin("n = 0.", SRC)               # no Int/Float64 Welford
    @test occursin("trajectory(v::AbstractVector)", SRC) && occursin("mv(v::AbstractVector)", SRC)   # buffers from template

    # (B-diag) owned diagnostics on nuts_state (compiler-owned storage), reset per transition; n_steps produced
    #          by the registered stats callback, independent of pgrad + leapfrog body marker.
    for f in (:n_steps, :reached_depth, :acceptance_rate); @test f in ns.fields; end
    @test :n_steps in reset_writes && :acceptance_rate in reset_writes                       # diagnostics reset per transition
    @test occursin("collectstats!(__self__)", nb) && occursin("stats_f(__self__)", nb)       # per-leaf stats callback

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
    # (C-refresh) public nuts!! refreshes momentum on the owned init phasepoint BEFORE step! (no hidden refresh).
    @test occursin("refresh_momentum!!(state.init; rng)", c)
    ci_refresh = findfirst("refresh_momentum!!", c); ci_step = findfirst("step!(state", c)
    @test ci_refresh !== nothing && ci_step !== nothing && first(ci_refresh) < first(ci_step)   # refresh precedes step!
    println("  (C) @kernel nuts!!(state; rng): refresh_momentum!! on state.init THEN step!(state, rng), returns state. OK")

    # ---- FORM A2: refresh_momentum!! RK @kernel — SOURCE MUTATION ONLY (randn! + lmul!), no cache writes ----
    rf = blocks[Symbol("refresh_momentum!!")]; @test :phasepoint in formals(rf.sig) && :rng in formals(rf.sig)
    rfb = srcof(rf.body)
    @test occursin("Random.randn!(rng, phasepoint.mom)", rfb) && occursin("LinearAlgebra.lmul!(phasepoint.chol_metric.L, phasepoint.mom)", rfb)
    @test !occursin("@node", rfb)                                            # no @node reference (shared logdet reused)
    for w in ("phasepoint.kin =", "phasepoint.ham =", "phasepoint.dkin_dmom =", "phasepoint.dham_dmom ="); @test !occursin(w, rfb); end  # NO author cache writes
    println("  (A2) refresh_momentum!!: source mutation only (randn!/lmul!), no cache/@node writes. OK")

    # ---- STATS IDENTITY: registered nuts_stats! increments owned n_steps; production binding uses it ----
    st = blocks[Symbol("nuts_stats!")]; @test :state in formals(st.sig)
    stb = srcof(st.body)
    @test occursin("state.n_steps += 1", stb)                                # one increment per collectstats!/leaf
    @test occursin("stats_f = nuts_stats!", SRC)                             # production binding is the registered callback (not nothing)

    # ---- @node preserved; @reactive-as-macro absent; free @kernel leapfrog!/rcopy!!/nuts!! ----------
    @test occursin("@node(logdet(chol_metric))", SRC)
    has_reactive = Ref(false)
    _walk(AST) do x; x isa Expr && x.head === :macrocall && x.args[1] === Symbol("@reactive") && (has_reactive[] = true); end
    @test !has_reactive[]
    @test occursin("@kernel leapfrog!", SRC) && occursin("@kernel nuts!!", SRC)
    @test occursin("@kernel refresh_momentum!!", SRC) && occursin("@kernel nuts_stats!", SRC)   # free refresh + stats kernels
    # public @rk_* exact-identity effect declarations (973f7f4/bf7d2ed) — SEVEN module helpers (incl. min1exp,
    # which nuts_stats! calls; the compiler is forbidden to inspect its body, so it MUST be declared).
    for d in ("@rk_pure finiteorneginf 1", "@rk_pure min1exp 1", "@rk_borrows badd 2", "@rk_rng randbernoullilog 2 1",
              "@rk_pure logswapprob 1", "@rk_pure compute_criterion 3", "@rk_pure smooth 3")
        @test occursin(d, SRC)
    end
    # every ordinary module helper the authored kernels CALL must carry an @rk_* declaration (no body inference).
    @test occursin("min1exp(state.dham)", srcof(blocks[Symbol("nuts_stats!")].body)) && occursin("@rk_pure min1exp 1", SRC)
    println("  (@node) preserved; @reactive absent; free @kernel leapfrog!/refresh_momentum!!/nuts_stats!/nuts!!;")
    println("      SEVEN public @rk_* helper effect declarations (incl. @rk_pure min1exp 1); copy!! is core. OK")
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
println("SOURCE-FORM VERIFIED ONLY. This updated eight-@kernel/effect-declaration source (euclidean_phasepoint,")
println("leapfrog!, refresh_momentum!!, nuts_stats!, nuts_state, nuts!!, dual_averaging_state, welford_var) is")
println("NOT YET constructed: construction is PENDING cherry-pick onto the current effects/factory substrate")
println("(@rk_* declarations + @kernel source-capture + nuts!! execution seam). Does NOT inherit the prior")
println("612ceee construction claim. NO MethodIR/lowering/execution/parity/alloc/perf claim.")
