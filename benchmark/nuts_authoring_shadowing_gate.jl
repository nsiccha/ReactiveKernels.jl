# STRUCTURAL gate + NON-VACUOUS lexical-shadowing inventory for the A/B/C @kernel NUTS fixture.
#
# Parses benchmark/nuts_kernel_authoring_fixture.jl as TEXT (Meta.parseall) — does NOT evaluate @kernel,
# so it runs even while the source-capture substrate is being implemented (fixture construction may be
# blocked). Verifies the locked forms A (RK-visible @kernel leapfrog!), B (implicit-field, no-Ref,
# explicit gofwd-branch nuts_state), C (@kernel nuts!! returns state), @node preserved, and prints the
# per-method formals/locals(shadow) vs unshadowed field reads/writes inventory.
using Test

const FIX = joinpath(@__DIR__, "nuts_kernel_authoring_fixture.jl")
const SRC = read(FIX, String)
const AST = Meta.parseall(SRC; filename = FIX)

_dropln(xs) = filter(x -> !(x isa LineNumberNode), xs)
_walk(f, x) = (f(x); x isa Expr && foreach(a -> _walk(f, a), x.args); nothing)
_signame(s) = s isa Symbol ? s : (s isa Expr && s.head in (:call, :where, :(::), :curly) ? _signame(s.args[1]) : nothing)
is_methoddef(st) = st isa Expr && ((st.head === :(=) && st.args[1] isa Expr && st.args[1].head === :call) || st.head === :function)
methodname(st) = _signame(st.args[1]); methodsig(st) = st.args[1]; methodbody(st) = st.args[2]

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

blocks = kernel_blocks()
@testset "A/B/C locked-form structural gate" begin
    for k in (:euclidean_phasepoint, :leapfrog!, :nuts_state, Symbol("nuts!!"), :dual_averaging_state, :welford_var)
        @test haskey(blocks, k)
    end

    # ---- FORM A: leapfrog! is an RK @kernel (free kernel) with the exact 3-line mom/pos/mom body -----
    lf = blocks[:leapfrog!]
    @test :stepsize in formals(lf.sig)                                 # stepsize keyword
    lb = srcof(lf.body)
    @test occursin("phasepoint.mom", lb) && occursin("phasepoint.pos", lb)
    @test occursin("phasepoint.dham_dpos", lb) && occursin("phasepoint.dham_dmom", lb)
    # exactly three broadcast statements: mom-=, pos+=, mom-=
    lstmts = _dropln(lf.body.args)
    @test length(lstmts) == 3
    heads = map(s -> (s isa Expr && s.head === :macrocall && s.args[1] === Symbol("@__dot__")) ? _dropln(s.args)[end].head : nothing, lstmts)
    @test heads == [:(-=), :(+=), :(-=)]
    println("  (A) leapfrog! is an RK @kernel; exact 3-line Stormer-Verlet on dham_dpos/dham_dmom. OK")

    # ---- FORM B: nuts_state implicit-field, NO Ref/fwdbwd, explicit init/fwd/bwd, gofwd branches -------
    ns = blocks[:nuts_state]; nb = srcof(ns.body)
    # (B1) NO Ref/RefValue current-view machinery, NO fwdbwd index anywhere in nuts_state source.
    @test !occursin("Ref(", nb) && !occursin("RefValue", nb) && !occursin("fwdbwd", nb)
    # (B2) explicit fixed physical owned endpoints init/fwd/bwd (fwd/bwd are deepcopy fields; init source).
    @test :fwd in ns.fields && :bwd in ns.fields && :init in ns.sources
    @test occursin("fwd = deepcopy(init)", nb) && occursin("bwd = deepcopy(init)", nb)
    # (B3) direction is resolved by EXPLICIT gofwd branches (ternary `gofwd ? .. : ..` parses to
    #      Expr(:if, :gofwd, ..)); require several across step/criterion/snapshot/reset methods.
    gofwd_branches = 0
    for m in ns.methods
        _walk(methodbody(m)) do x
            x isa Expr && x.head === :if && x.args[1] === :gofwd && (gofwd_branches += 1)
        end
    end
    @test gofwd_branches >= 4
    # (B4) implicit-field methods: NO self/__self__ formal; NO self.X/__self__.X; __self__ only as call actual.
    for m in ns.methods
        fs = formals(methodsig(m))
        @test !(:self in fs) && !(Symbol("__self__") in fs)
        _walk(methodbody(m)) do x
            if x isa Expr && x.head === :.
                @test x.args[1] !== :self && x.args[1] !== Symbol("__self__")
            end
        end
    end
    # (B5) reference field spellings present.
    nsnames = Set(ns.fields) ∪ Set(ns.sources)
    for f in (:gofwd,:may_sample,:may_continue,:fwd,:bwd,:trees,:proposals,:dham,:diverged,
              :init,:rng,:max_depth,:min_dham,:step_f,:stats_f)
        @test f in nsnames
    end
    @test Set(blocks[:dual_averaging_state].fields) ⊇ Set([:m,:H,:mu,:log_current,:log_final,:current,:final])
    @test Set(blocks[:welford_var].fields) ⊇ Set([:n,:mean,:var])
    println("  (B) nuts_state: no Ref/fwdbwd; explicit init/fwd/bwd; gofwd branches; implicit fields;")
    println("      no self/__self__ formal; no self.X/__self__.X; reference spellings. OK")

    # ---- FORM C: public @kernel nuts!!(state; rng) returns the SAME object -----------------------------
    nb2 = blocks[Symbol("nuts!!")]
    @test :state in formals(nb2.sig)
    @test occursin("return state", srcof(nb2.body))
    println("  (C) @kernel nuts!!(state; rng) mutates + explicitly `return state`. OK")

    # ---- @node preserved; @reactive absent (as a MACRO, not a comment mention); leapfrog!/nuts!! @kernel
    @test occursin("@node(logdet(chol_metric))", SRC)
    has_reactive = Ref(false)
    _walk(AST) do x
        x isa Expr && x.head === :macrocall && x.args[1] === Symbol("@reactive") && (has_reactive[] = true)
    end
    @test !has_reactive[]
    @test occursin("@kernel leapfrog!", SRC) && occursin("@kernel nuts!!", SRC)
    println("  (@node) @node(logdet(chol_metric)) preserved; @reactive absent; free @kernel leapfrog!/nuts!!. OK")
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
println("\nSTRUCTURAL GATE PASS (A/B/C). Construction may be BLOCKED on the source-capture substrate")
println("(required capabilities: implicit fields + __self__ receiver + free-kernel leapfrog!/nuts!!")
println("discrimination + no-Julia-IR capture). NO execution/parity/perf claim.")
