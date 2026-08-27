# Executable prepared-endpoint phasepoint receipt (RK 07:43 joint milestone). The interim compile_leaf /
# effect-root / copy / transition scaffold tests (which consumed fake `_OwnerState`/`_Synth` storage) were
# retired with that scaffold on the 4c6ed92 seam; the generic schedule-MODEL tests live in
# test_kernel_lowering.jl (no fake storage). This suite executes the REAL captured six-handle phasepoint.
using ReactiveKernels, Test, LinearAlgebra
const RK = ReactiveKernels

# ============================================================================================
# EXECUTABLE PREPARED-ENDPOINT phasepoint — the RK 07:43 joint receipt (six real captured handles).
# Isolated module so the endpoint @kernel globals never collide with other suites' fixtures. Canonical
# ids are resolved by FIELD NAME through the plan (the global Value-id counter is not 1..N in-suite).
# ============================================================================================
module _PPFix
    using ReactiveKernels, LinearAlgebra
    @kernel euclidean_ep(grad_f, metric, pos, mom) = begin
        pot, dpot_dpos = grad_f(pos)
        chol_metric = cholesky(metric)
        dkin_dmom = chol_metric \ mom
        kin = oftype(pot, 0.5) * (@node(logdet(chol_metric)) + dot(mom, dkin_dmom))
        ham = pot + kin
        dham_dpos = dpot_dpos
        dham_dmom = dkin_dmom
    end
    @kernel leapfrog_ep!(phasepoint; stepsize) = begin
        @. phasepoint.mom -= stepsize * phasepoint.dham_dpos
        @. phasepoint.pos +=       stepsize * phasepoint.dham_dmom
    end
    mutable struct CountPgrad; n::Int; end
    (c::CountPgrad)(dest, pos) = (c.n += 1; dest .= 2 .* pos; sum(abs2, pos))
end

using LinearAlgebra: cholesky, ldiv!, dot, logdet, Cholesky, I, PosDefException

_pp_pf() = RK._prepare_factory(_PPFix.euclidean_ep, RK.kernel_registration(_PPFix.leapfrog_ep!))
# canonical id of a plan field by NAME; @node resolved by its gensym prefix.
_pp_c(plan, name::Symbol) = only(s.canon for s in RK.kernel_plan_slots(plan) if s.path[end] === name)
_pp_node(plan) = only(unique(s.canon for s in RK.kernel_plan_slots(plan) if startswith(String(s.path[end]), "##node")))
# UNCOMPUTED typed placeholders keyed by REAL canons (RK 08:03/08:15): allocated factors Matrix + the
# Cholesky wrapper constructor (Char 'U') WITHOUT calling cholesky; zeros elsewhere. NO recipe is executed.
function _pp_values(plan, ::Type{T}, d, pgrad) where {T}
    v = Dict{Int,Any}()
    for s in RK.kernel_plan_slots(plan)
        haskey(v, s.canon) && continue
        nm = s.path[end]; ns = String(nm)
        v[s.canon] =
            nm === :grad_f      ? pgrad :
            nm === :metric      ? (Matrix{T}(I(d)) .+ T(0.5)) :
            nm === :pos         ? T[1,2,3][1:d] :
            nm === :mom         ? T[0.5,-1,2][1:d] :
            nm === :chol_metric ? Cholesky(Matrix{T}(undef, d, d), 'U', 0) :
            startswith(ns, "##node") ? zero(T) :
            (nm in (:pot, :kin, :ham)) ? zero(T) :
            zeros(T, d)
    end
    v
end
_pp_construct(pf, plan, values) = RK._kernel_construct_endpoint(
    Val(RK.kernel_prepared_token(pf)), plan, values, RK.kernel_prepared_external(pf))
_pp_rd(plan, ow, sh, c) = (fs = RK.kernel_plan_field(plan, c);
    fs[1] === :owned ? RK._canon_slot(ow, Val(fs[2])) : RK._canon_slot(sh, Val(fs[2])))
_pp_cur(plan, ow, sh, c) = (fs = RK.kernel_plan_field(plan, c);
    RK._canon_current(fs[1] === :owned ? ow : sh, Val(fs[2])))
function _pp_ref(::Type{T}, d) where {T}
    metric = Matrix{T}(I(d)) .+ T(0.5); pos = T[1,2,3][1:d]; mom = T[0.5,-1,2][1:d]
    ch = cholesky(metric); dk = ch \ mom
    (pot = sum(abs2, pos), dpot = 2 .* pos, ld = logdet(ch), dkin = dk,
     kin = T(0.5)*(logdet(ch)+dot(mom,dk)), ham = sum(abs2,pos)+T(0.5)*(logdet(ch)+dot(mom,dk)))
end

@testset "prepared-endpoint — FULL six-handle cold init: math + exact masks + counts, F32 & F64 (RK 08:04b)" begin
    pf = _pp_pf(); plan = RK.kernel_prepared_plan(pf); hs = RK.kernel_prepared_handles(pf)
    @test length(hs) == 6
    @test RK.kernel_plan_recipes(plan) == (2, 3, 1, 4, 5, 6)
    @test [RK.recipe_handle_mode(h) for h in hs] == [:destination, :assign, :assign, :ldiv, :assign, :assign]
    for T in (Float64, Float32)
        d = 3; cp = _PPFix.CountPgrad(0)
        ow, sh = _pp_construct(pf, plan, _pp_values(plan, T, d, cp))
        @test cp.n == 0                                        # construction computes NOTHING (RK 08:00 gate 1)
        @test RK.compile_prepared_initialization(pf, typeof(ow), typeof(sh))(ow, sh, hs) === ow  # returns owned
        r = _pp_ref(T, d)
        @test _pp_rd(plan, ow, sh, _pp_c(plan, :pot))       ≈ r.pot
        @test _pp_rd(plan, ow, sh, _pp_c(plan, :dpot_dpos)) ≈ r.dpot
        @test _pp_rd(plan, ow, sh, _pp_node(plan))          ≈ r.ld
        @test _pp_rd(plan, ow, sh, _pp_c(plan, :dkin_dmom)) ≈ r.dkin
        @test _pp_rd(plan, ow, sh, _pp_c(plan, :kin))       ≈ r.kin
        @test _pp_rd(plan, ow, sh, _pp_c(plan, :ham))       ≈ r.ham
        @test cp.n == 1                                        # EXACTLY one pgrad (destination handle)
        cur = Set{Int}(s.canon for s in RK.kernel_plan_slots(plan) if _pp_cur(plan, ow, sh, s.canon))
        @test cur == Set(RK.kernel_plan_entry_current(plan))   # final currentness == entry_current EXACTLY
    end
end

@testset "prepared-endpoint — transition trace from MethodIR writes: grad/velocity/kin/ham, chol+@node EXCLUDED (RK 08:06/08:10/08:13)" begin
    pf = _pp_pf(); plan = RK.kernel_prepared_plan(pf)
    tr = RK.prepared_transition_trace(plan, RK.method_irs(_PPFix.leapfrog_ep!)[1])   # PRODUCTION from leaf writes
    Rids = RK.selected_trace_recipes(tr)                       # read-only accessor (RK 08:13; no type introspection)
    @test RK.selected_trace_key(tr) === RK.kernel_plan_key(plan)
    prod = Dict(c => r for (c, r) in RK.kernel_plan_producer(plan))
    grad_r, vel_r = prod[_pp_c(plan, :pot)], prod[_pp_c(plan, :dkin_dmom)]
    kin_r, ham_r  = prod[_pp_c(plan, :kin)], prod[_pp_c(plan, :ham)]
    chol_r, node_r = prod[_pp_c(plan, :chol_metric)], prod[_pp_node(plan)]
    @test Set(Rids) == Set([grad_r, vel_r, kin_r, ham_r])      # grad/velocity/kin/ham by RECIPE IDENTITY
    @test !(chol_r in Rids) && !(node_r in Rids)               # exact chol + @node recipe ids EXCLUDED
    # PROVENANCE (RK 08:37): the PUBLIC production API takes the leaf MethodIR and derives the trace itself,
    # so a caller CANNOT inject a recipe list — a _SelectedTrace is not an accepted public argument.
    @test_throws MethodError RK.compile_prepared_schedule(pf, RK._CanonOwned1, RK._CanonShared1,
        RK._SelectedTrace{RK.kernel_plan_key(plan), (grad_r, vel_r)}())
    # A same-Key / ARBITRARY-Rids trace IS internally constructible (compiler-internal provenance, NOT a
    # security boundary) — but reachable only through the PRIVATE overload, which is test-only. It compiles
    # whatever Rids it is handed; the public path above never exposes that lever.
    forged = RK._SelectedTrace{RK.kernel_plan_key(plan), (chol_r,)}()   # wrong recipe, right Key
    @test RK.selected_trace_recipes(forged) == (chol_r,)               # constructible; no boundary
    # the private overload still rejects a WRONG-plan Key (the one real provenance check it enforces)
    @test_throws RK._LLowerReject RK._compile_prepared_schedule(pf, RK._CanonOwned1,
        RK._CanonShared1, RK._SelectedTrace{:forged_key, (2, 4)}())
end

@testset "prepared-endpoint — POST-WRITE RECOMPUTE 0-B/@inferred, chol+@node Δ0, pgrad Δ1, F32 & F64 (RK 08:04c/08:15)" begin
    for T in (Float64, Float32)
        pf = _pp_pf(); plan = RK.kernel_prepared_plan(pf); hs = RK.kernel_prepared_handles(pf)
        d = 3; cp = _PPFix.CountPgrad(0)
        ow, sh = _pp_construct(pf, plan, _pp_values(plan, T, d, cp))
        RK.compile_prepared_initialization(pf, typeof(ow), typeof(sh))(ow, sh, hs)
        @test cp.n == 1
        warm = RK.compile_prepared_schedule(pf, typeof(ow), typeof(sh), RK.method_irs(_PPFix.leapfrog_ep!)[1])
        @test warm(ow, sh, hs) === ow                          # returns the owned object (no tuple alloc)
        chol0 = _pp_rd(plan, ow, sh, _pp_c(plan, :chol_metric)); node0 = _pp_rd(plan, ow, sh, _pp_node(plan))
        n0 = cp.n
        @test (@allocated warm(ow, sh, hs)) == 0               # EXACT 0-B warmed post-write recompute
        @test cp.n == n0 + 1                                   # pgrad Δ one per post-write recompute
        Test.@inferred warm(ow, sh, hs)
        # chol/@node Δ ZERO: same object identity AND value (statically omitted, not guarded)
        warm(ow, sh, hs)
        @test _pp_rd(plan, ow, sh, _pp_c(plan, :chol_metric)) === chol0    # SAME Cholesky object
        @test _pp_rd(plan, ow, sh, _pp_c(plan, :chol_metric)).factors == chol0.factors
        @test _pp_rd(plan, ow, sh, _pp_node(plan)) == node0
    end
end

@testset "prepared-endpoint — JOINT poison gate: live-graph op/producer mutation cannot change captured exec (RK 08:15)" begin
    pf = _pp_pf(); plan = RK.kernel_prepared_plan(pf); hs = RK.kernel_prepared_handles(pf); d = 3
    cp = _PPFix.CountPgrad(0)
    ow, sh = _pp_construct(pf, plan, _pp_values(plan, Float64, d, cp))
    init = RK.compile_prepared_initialization(pf, typeof(ow), typeof(sh))
    leaf = RK.method_irs(_PPFix.leapfrog_ep!)[1]
    warm = RK.compile_prepared_schedule(pf, typeof(ow), typeof(sh), leaf)
    init(ow, sh, hs); ham_clean = _pp_rd(plan, ow, sh, _pp_c(plan, :ham))
    trace_clean = RK.selected_trace_recipes(RK.prepared_transition_trace(plan, leaf))
    g = RK.kernel_graph(_PPFix.euclidean_ep)
    saved_recipes = copy(g.recipes); saved_producers = copy(g.producers)
    try
        empty!(g.recipes); empty!(g.producers)                 # POISON the live graph after preparation
        ow2, sh2 = _pp_construct(pf, plan, _pp_values(plan, Float64, d, _PPFix.CountPgrad(0)))
        init(ow2, sh2, hs)                                     # already-captured executor — no live-graph read
        @test _pp_rd(plan, ow2, sh2, _pp_c(plan, :ham)) ≈ ham_clean            # identical math
        tr2 = RK.prepared_transition_trace(plan, RK.method_irs(_PPFix.leapfrog_ep!)[1])
        @test RK.selected_trace_recipes(tr2) == trace_clean   # identical trace (plan-derived, not graph)
    finally
        append!(g.recipes, saved_recipes); merge!(g.producers, saved_producers)
    end
end

@testset "prepared-endpoint — exception soundness: pgrad throw + genuine later-handle throw (RK 08:00/08:07)" begin
    pf = _pp_pf(); plan = RK.kernel_prepared_plan(pf); hs = RK.kernel_prepared_handles(pf); d = 3
    cpot, cdpot = _pp_c(plan, :pot), _pp_c(plan, :dpot_dpos); cchol = _pp_c(plan, :chol_metric)
    # (1) pgrad (handle 1) THROWS -> pot+dpot BOTH dirty
    throwgrad(dest, pos) = error("pgrad boom")
    ow, sh = _pp_construct(pf, plan, _pp_values(plan, Float64, d, throwgrad))
    init = RK.compile_prepared_initialization(pf, typeof(ow), typeof(sh))
    @test_throws ErrorException init(ow, sh, hs)
    @test !_pp_cur(plan, ow, sh, cpot) && !_pp_cur(plan, ow, sh, cdpot)
    # (2) GENUINE later-handle throw: cholesky (handle 2) on an INDEFINITE metric throws AFTER grad succeeds
    cp = _PPFix.CountPgrad(0); v2 = _pp_values(plan, Float64, d, cp)
    v2[_pp_c(plan, :metric)] = [1.0 2.0 3.0; 2.0 1.0 4.0; 3.0 4.0 1.0]   # symmetric INDEFINITE -> PosDefException
    ow2, sh2 = _pp_construct(pf, plan, v2)
    @test_throws PosDefException init(ow2, sh2, hs)
    @test cp.n == 1                                            # grad ran & completed
    @test _pp_cur(plan, ow2, sh2, cpot) && _pp_cur(plan, ow2, sh2, cdpot)  # grad outputs BLESSED
    @test !_pp_cur(plan, ow2, sh2, cchol)                      # chol output NOT blessed (dirty); obj discardable
end

@testset "prepared-endpoint — child copy: owned-only children, SAME shared authority, ZERO extra pgrad, F32 & F64 (RK 06:53/08:15)" begin
    for T in (Float64, Float32)
        pf = _pp_pf(); plan = RK.kernel_prepared_plan(pf); hs = RK.kernel_prepared_handles(pf); d = 3
        cham = _pp_c(plan, :ham)
        cp = _PPFix.CountPgrad(0)
        ow, sh = _pp_construct(pf, plan, _pp_values(plan, T, d, cp))
        RK.compile_prepared_initialization(pf, typeof(ow), typeof(sh))(ow, sh, hs)
        @test cp.n == 1
        tok = RK.kernel_prepared_token(pf)
        fwd = RK._kernel_construct_owned_child(Val(tok), plan, _pp_values(plan, T, d, cp))
        bwd = RK._kernel_construct_owned_child(Val(tok), plan, _pp_values(plan, T, d, cp))
        @test typeof(fwd) == typeof(ow) && typeof(bwd) == typeof(ow)   # owned layout, NO shared authority built
        RK._seed_children!(ow, fwd, bwd)
        @test cp.n == 1                                        # ZERO additional pgrad across child seeds
        @test RK._canon_slot(fwd, Val(RK.kernel_plan_field(plan, cham)[2])) ≈
              RK._canon_slot(ow,  Val(RK.kernel_plan_field(plan, cham)[2]))   # ham transferred
        # EXECUTE the child against the SAME shared authority `sh` (fixed metric) — proves shared wiring:
        warm = RK.compile_prepared_schedule(pf, typeof(fwd), typeof(sh), RK.method_irs(_PPFix.leapfrog_ep!)[1])
        @test warm(fwd, sh, hs) === fwd                       # child recompute returns the child owned obj
        warm(fwd, sh, hs)
        @test (@allocated warm(fwd, sh, hs)) == 0             # child post-write recompute 0-B over the shared chol
        @test _pp_rd(plan, fwd, sh, cham) ≈ _pp_ref(T, d).ham # child recomputes correct ham via shared chol
    end
end
