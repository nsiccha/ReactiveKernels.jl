using Test, ReactiveKernels, LinearAlgebra, Random
const RK = ReactiveKernels

module _NativeNutsFix
    include(joinpath(@__DIR__, "..", "benchmark", "nuts_kernel_authoring_fixture.jl"))
end
module _NativeNutsFix2
    include(joinpath(@__DIR__, "..", "benchmark", "nuts_kernel_authoring_fixture.jl"))
end

_native_pf() = RK._prepare_factory(_NativeNutsFix.euclidean_phasepoint,
                                    RK.kernel_registration(_NativeNutsFix.leapfrog!))
function _native_vals(pf,T;metric=T[2 0;0 2])
    P=RK.kernel_prepared_plan(pf); m=metric; d=Dict{Int,Any}()
    for s in RK.kernel_plan_slots(P)
        n=String(s.path[1]); d[s.canon] = n=="grad_f" ? ((dst,p)->(dst .= 2 .* p;sum(abs2,p))) :
          n=="metric" ? m : n=="chol_metric" ? cholesky(m) : startswith(n,"##node") ? zero(T) :
          n=="pos" ? T[1,2] : n=="mom" ? T[3,4] :
          n in ("dpot_dpos","dham_dpos","dkin_dmom","dham_dmom") ? T[0,0] : zero(T)
    end; d
end
function _native_frame(pf,T,md;stats=nothing,min_dham=-1000,metric=T[2 0;0 2])
    f=RK._construct_nuts_frame(pf,_native_vals(pf,T;metric),md;step_f=RK.partial(_NativeNutsFix.leapfrog!;stepsize=T(.1)),
                               stats_f=stats,min_dham=min_dham)
    RK.compile_prepared_initialization(pf,typeof(f.init),typeof(f.shared))(f.init,f.shared,RK.kernel_prepared_handles(pf))
    RK._seed_nuts_children!(f); f
end

module _NativeFakeStructural
    const Bool = Core.Bool
end

struct _NativeCountedGrad
    count::Vector{Int}
end
@inline function (g::_NativeCountedGrad)(dst,p)
    @inbounds g.count[1]+=1
    @. dst=2*p
    sum(abs2,p)
end
struct _NativeThrowGrad
    count::Vector{Int}
    fail::Vector{Bool}
end
@inline function (g::_NativeThrowGrad)(dst,p)
    @inbounds g.count[1]+=1
    @inbounds g.fail[1] && error("NATIVE_GRAD_SENTINEL")
    @. dst=2*p
    sum(abs2,p)
end
struct _NativeThrowRNG <: Random.AbstractRNG end
Random.randn!(::_NativeThrowRNG, x::AbstractArray) = error("NATIVE_RNG_SENTINEL")

function _native_vals_with_grad(pf,T,grad;metric=T[2 0;0 2])
    d=_native_vals(pf,T); P=RK.kernel_prepared_plan(pf)
    for s in RK.kernel_plan_slots(P)
        n=String(s.path[1])
        n=="grad_f" && (d[s.canon]=grad)
        n=="metric" && (d[s.canon]=metric)
        n=="chol_metric" && (d[s.canon]=cholesky(metric))
    end
    d
end

function _native_build_instrumented(pf,T;grad=_NativeCountedGrad([0]),metric=T[2 0;0 2],
        md=5,min_dham=T(-1000),stats=_NativeNutsFix.nuts_stats!)
    vals=_native_vals_with_grad(pf,T,grad;metric=metric)
    k=RK._build_nuts_instrumented_sampler(pf,vals,_NativeNutsFix.nuts_state,
        _NativeNutsFix.refresh_momentum!!,_NativeNutsFix.nuts!!;
        step_f=RK.partial(_NativeNutsFix.leapfrog!;stepsize=T(.1)),max_depth=md,
        min_dham=min_dham,stats_f=stats)
    k,grad
end

mutable struct _NativeCountEnsure{F}
    f::F
    count::Vector{Int}
end
function (c::_NativeCountEnsure)(owned, shared, handles)
    c.count[1] += 1
    c.f(owned, shared, handles)
end
struct _NativeThrowEnsure end
(::_NativeThrowEnsure)(owned, shared, handles) = throw(ErrorException("ensure boundary sentinel"))
mutable struct _NativeIndexCounter
    count::Vector{Int}
end
function (c::_NativeIndexCounter)()
    c.count[1] += 1
    1
end


@testset "native NUTS — generated root executes the real fixture" begin
    pf=_native_pf(); fr=_native_frame(pf,Float64,3)
    C=RK.compile_nuts_native(pf,_NativeNutsFix.nuts_state,_NativeNutsFix.refresh_momentum!!,
                             _NativeNutsFix.nuts!!,fr)
    @test C.root!(fr,C.scratch,Random.Xoshiro(1)) === fr
    @test RK.diagnostics_committed_mask(fr.diag) == UInt(0x0f)
end

function _native_obs(pf,f)
    P=RK.kernel_prepared_plan(pf)
    slot(ep,n)=RK._canon_slot(ep,RK.kernel_plan_named_slot_val(P,Val(n)))
    (pos=copy(slot(f.init,:pos)),mom=copy(slot(f.init,:mom)),pot=slot(f.init,:pot),
     dpot=copy(slot(f.init,:dpot_dpos)),dkin=copy(slot(f.init,:dkin_dmom)),
     kin=slot(f.init,:kin),ham=slot(f.init,:ham),gofwd=f.gofwd,may_sample=f.may_sample,
     may_continue=f.may_continue,diverged=f.diverged,n_steps=RK._diag_slot(f.diag,Val(1)),
     reached=RK._diag_slot(f.diag,Val(2)),accept=RK._diag_slot(f.diag,Val(3)),
     dham=RK._diag_slot(f.diag,Val(4)),pending=RK.diagnostics_pending_mask(f.diag),
     committed=RK.diagnostics_committed_mask(f.diag))
end
@inline function _native_batch(root,fr,sc,rng,::Val{N}) where {N}
    i=0; @inbounds while i<N; root(fr,sc,rng); i+=1; end; nothing
end
function _native_alloc(root,fr,sc,rng,::Val{N}) where {N}
    _native_batch(root,fr,sc,rng,Val(N)); GC.gc(); @allocated _native_batch(root,fr,sc,rng,Val(N))
end

@testset "native NUTS — writeret is write-then-return; forced divergence cannot fall through" begin
    for T in (Float64,Float32)
        pf=_native_pf(); fc=_native_frame(pf,T,5;min_dham=T(1e6)); fn=_native_frame(pf,T,5;min_dham=T(1e6))
        Cc=RK.compile_nuts(pf,_NativeNutsFix.nuts_state,_NativeNutsFix.refresh_momentum!!,_NativeNutsFix.nuts!!,fc)
        Cn=RK.compile_nuts_native(pf,_NativeNutsFix.nuts_state,_NativeNutsFix.refresh_momentum!!,_NativeNutsFix.nuts!!,fn)
        Cc.root!(fc,Cc.scratch,Random.Xoshiro(7)); Cn.root!(fn,Cn.scratch,Random.Xoshiro(7))
        @test _native_obs(pf,fn) == _native_obs(pf,fc)
        @test fn.diverged && !fn.may_continue && !fn.may_sample
        @test RK._diag_slot(fn.diag,Val(2)) == 1 # post-return depth-loop work did not execute
    end
end


@testset "native NUTS — explicit post-return sentinel and lazy guards" begin
    pf=_native_pf(); f=_native_frame(pf,Float64,3); PT=typeof(RK.kernel_prepared_plan(pf)); OT=:sentinel_owner
    W=RK._NNPlaceWrite{RK._NNSelfField{(:may_continue,)},:self,(:may_continue,),nothing,RK._NNLit{false},false}
    S=RK._NNPlaceWrite{RK._NNSelfField{(:reached_depth,)},:self,(:reached_depth,),nothing,RK._NNLit{999},false}
    M=RK._NNMethod{101,:sentinel,Tuple{},Tuple{RK._NNWriteReturn{W},S}}
    P=RK._NativeProgram{OT,PT,101,Tuple{M},(),RK._NNNoStats}
    RK._diag_set_value!(f.diag,Val(2),17); f.may_continue=true
    @test RK._nn_method0(P,Val(101),(;),f) === f
    @test !f.may_continue && RK._diag_slot(f.diag,Val(2)) == 17 # the statement after writeret is unreachable

    G1=RK._NNGuard{:&&,RK._NNLit{false},Tuple{S}}
    G2=RK._NNGuard{:||,RK._NNLit{true},Tuple{S}}
    GM=RK._NNMethod{307,:guards,Tuple{},Tuple{G1,G2}}
    GP=RK._NativeProgram{:guard_owner,PT,307,Tuple{GM},(),RK._NNNoStats}
    RK._diag_set_value!(f.diag,Val(2),23); @test RK._nn_method0(GP,Val(307),(;),f) === f
    @test RK._diag_slot(f.diag,Val(2)) == 23
end

@testset "native NUTS — current derived reads bypass the ensure callable; dirty reads repair once" begin
    pf = _native_pf(); f = _native_frame(pf, Float64, 3)
    C = RK.compile_nuts_native(pf, _NativeNutsFix.nuts_state, _NativeNutsFix.refresh_momentum!!,
                               _NativeNutsFix.nuts!!, f)
    readham = RK._NNSelfField{(:init, :ham)}
    M = RK._NNMethod{111, :ensure_boundary, Tuple{}, Tuple{RK._NNExprStmt{readham}}}
    P = RK._NativeProgram{:ensure_boundary, typeof(RK.kernel_prepared_plan(pf)), 111,
                          Tuple{M}, (:ham,), RK._NNNoStats}
    calls = [0]
    counted = _NativeCountEnsure(getfield(C.cfg.ensures, :ham), calls)
    cfg = (ensures = (ham = counted,), handles = C.cfg.handles)

    # Warm/current is the dominant recursive path: the generated caller reads the physical slot directly.
    @test RK._nn_method0(P, Val(111), cfg, f) === f
    @test calls[1] == 0

    # Dirty still delegates to the exact prepared ensure once, then subsequent reads bypass it again.
    plan = RK.kernel_prepared_plan(pf)
    hslot = RK.kernel_plan_named_slot_val(plan, Val(:ham))
    RK._canon_kill!(f.init, hslot)
    @test RK._nn_method0(P, Val(111), cfg, f) === f
    @test calls[1] == 1
    @test RK._canon_current(f.init, hslot)
    @test RK._nn_method0(P, Val(111), cfg, f) === f
    @test calls[1] == 1

    # A failing dirty repair is neither bypassed nor followed by a raw stale read/bless.
    RK._canon_kill!(f.init, hslot)
    badcfg = (ensures = (ham = _NativeThrowEnsure(),), handles = C.cfg.handles)
    @test_throws ErrorException RK._nn_method0(P, Val(111), badcfg, f)
    @test !RK._canon_current(f.init, hslot)

    # An indexed endpoint expression remains once-evaluated on both the bypass and repair paths.
    index_calls = [0]; indexer = _NativeIndexCounter(index_calls)
    idx = RK._NNRegistered{1, false, :index_counter, Tuple{}, Tuple{}, false}
    proposal = RK._NNIndex{RK._NNSelfField{(:proposals,)}, Tuple{idx}}
    indexed_ham = RK._NNGetfield{proposal, :ham}
    MI = RK._NNMethod{112, :indexed_ensure_boundary, Tuple{}, Tuple{RK._NNExprStmt{indexed_ham}}}
    PI = RK._NativeProgram{:indexed_ensure_boundary, typeof(plan), 112, Tuple{MI},
                           (:ham,), RK._NNNoStats}
    icfg = (ensures = (ham = counted,), handles = C.cfg.handles, callees = (indexer,))
    ensure_before = calls[1]
    @test RK._nn_method0(PI, Val(112), icfg, f) === f
    @test index_calls[1] == 1
    @test calls[1] == ensure_before
    RK._canon_kill!(f.proposals[1], hslot)
    @test RK._nn_method0(PI, Val(112), icfg, f) === f
    @test index_calls[1] == 2
    @test calls[1] == ensure_before + 1

    # A derived shared slot is guarded and reread on `frame.shared`, while its repair still receives the
    # physical owned endpoint plus shared authority.  Treating every derived field as owned would return the
    # endpoint's placeholder slot rather than the sealed Cholesky value.
    readchol = RK._NNSelfField{(:init, :chol_metric)}
    child = RK._NNMethod{114, :read_shared_chol, Tuple{}, Tuple{RK._NNReturn{readchol}}}
    childcall = RK._NNCallExpr{:read_shared_chol, 114, RK._NNSelf, Tuple{}, Tuple{}}
    parent = RK._NNMethod{113, :shared_ensure_parent, Tuple{}, Tuple{RK._NNExprStmt{childcall}}}
    PS = RK._NativeProgram{:shared_ensure_boundary, typeof(plan), 113, Tuple{parent, child},
                           (:chol_metric,), RK._NNNoStats}
    chol_ensure = RK.compile_prepared_ensure(pf, typeof(f.init), typeof(f.shared), :chol_metric)
    chol_calls = [0]; counted_chol = _NativeCountEnsure(chol_ensure, chol_calls)
    scfg = (ensures = (chol_metric = counted_chol,), handles = C.cfg.handles)
    expected_chol = RK._canon_slot(f.shared,
        RK.kernel_plan_named_slot_val(plan, Val(:chol_metric)))
    @test RK._nn_method0(PS, Val(114), scfg, f) === expected_chol
    @test chol_calls[1] == 0
    cslot = RK.kernel_plan_named_slot_val(plan, Val(:chol_metric))
    RK._canon_kill!(f.shared, cslot)
    @test RK._nn_method0(PS, Val(114), scfg, f) ===
          RK._canon_slot(f.shared, cslot)
    @test chol_calls[1] == 1
    @test RK._canon_current(f.shared, cslot)
end

@testset "native NUTS — full parity, concrete return, exact loop 0-B, two RNG, public Mode-2" begin
    for T in (Float64,Float32)
        pf=_native_pf(); fc=_native_frame(pf,T,5); fn=_native_frame(pf,T,5)
        Cc=RK.compile_nuts(pf,_NativeNutsFix.nuts_state,_NativeNutsFix.refresh_momentum!!,_NativeNutsFix.nuts!!,fc)
        Cn=RK.compile_nuts_native(pf,_NativeNutsFix.nuts_state,_NativeNutsFix.refresh_momentum!!,_NativeNutsFix.nuts!!,fn)
        for seed in 1:8
            Cc.root!(fc,Cc.scratch,Random.Xoshiro(seed)); Cn.root!(fn,Cn.scratch,Random.Xoshiro(seed))
            @test _native_obs(pf,fn) == _native_obs(pf,fc)
        end
        @inferred Cn.root!(fn,Cn.scratch,Random.Xoshiro(11))
        for rg in (Random.Xoshiro(91),Random.MersenneTwister(2))
            @test _native_alloc(Cn.root!,fn,Cn.scratch,rg,Val(64)) == 0
            @test _native_alloc(Cn.root!,fn,Cn.scratch,rg,Val(256)) == 0
        end
        owner=RK.kernel_token(_NativeNutsFix.nuts_state)
        sampler=RK.nuts_sampler(Val(owner),Val(Cn.RootToken),fn,Cn.root!,Cn.scratch)
        @test _NativeNutsFix.nuts!!(sampler;rng=Random.Xoshiro(3)) === sampler
        @inferred _NativeNutsFix.nuts!!(sampler;rng=Random.Xoshiro(4))
    end
end

@testset "native NUTS — effectful stats program is encoded and executes" begin
    pf=_native_pf(); f=_native_frame(pf,Float64,4;stats=_NativeNutsFix.nuts_stats!)
    C=RK.compile_nuts_native(pf,_NativeNutsFix.nuts_state,_NativeNutsFix.refresh_momentum!!,_NativeNutsFix.nuts!!,f)
    C.root!(f,C.scratch,Random.Xoshiro(1))
    @test RK._diag_slot(f.diag,Val(1)) > 0
    @test isfinite(RK._diag_slot(f.diag,Val(3)))
    @test _native_alloc(C.root!,f,C.scratch,Random.Xoshiro(9),Val(64)) == 0
end


@testset "native NUTS — sparse MethodId and left-to-right authored default" begin
    pf=_native_pf(); f=_native_frame(pf,Float64,3)
    SM=RK._NNMethod{9001,:sparse,Tuple{},Tuple{}}
    SP=RK._NativeProgram{:sparse_owner,typeof(RK.kernel_prepared_plan(pf)),9001,Tuple{SM},(),RK._NNNoStats}
    @test RK._nn_method0(SP,Val(9001),(;),f) === f
    @test_throws ArgumentError RK._nn_method0(SP,Val(17),(;),f)

    C=RK.compile_nuts_native(pf,_NativeNutsFix.nuts_state,_NativeNutsFix.refresh_momentum!!,_NativeNutsFix.nuts!!,f)
    last=length(f.proposals); a=f.proposals[1]; z=f.proposals[last]
    @test RK._nn_method1(C.program,Val(4),C.cfg,f,1) === f # omitted j evaluates length(proposals)
    @test f.proposals[1] === z && f.proposals[last] === a
    @test_throws MethodError RK._nn_method0(C.program,Val(4),C.cfg,f) # required i is not guessed
end

@testset "native NUTS — two owner tokens interleave without metadata collision" begin
    p1=_native_pf()
    p2=RK._prepare_factory(_NativeNutsFix2.euclidean_phasepoint,RK.kernel_registration(_NativeNutsFix2.leapfrog!))
    f1=_native_frame(p1,Float64,3)
    # The second fixture has the same source shape but a distinct definition token; construct with its own helpers.
    P2=RK.kernel_prepared_plan(p2); m=[2.0 0;0 2.0]; d=Dict{Int,Any}()
    for s in RK.kernel_plan_slots(P2)
        n=String(s.path[1]); d[s.canon]=n=="grad_f" ? ((dst,p)->(dst .= 2 .* p;sum(abs2,p))) :
          n=="metric" ? m : n=="chol_metric" ? cholesky(m) : startswith(n,"##node") ? 0.0 :
          n=="pos" ? [1.0,2] : n=="mom" ? [3.0,4] : n in ("dpot_dpos","dham_dpos","dkin_dmom","dham_dmom") ? [0.0,0] : 0.0
    end
    f2=RK._construct_nuts_frame(p2,d,3;step_f=RK.partial(_NativeNutsFix2.leapfrog!;stepsize=.1),stats_f=nothing,min_dham=-1000)
    RK.compile_prepared_initialization(p2,typeof(f2.init),typeof(f2.shared))(f2.init,f2.shared,RK.kernel_prepared_handles(p2)); RK._seed_nuts_children!(f2)
    c1=RK.compile_nuts_native(p1,_NativeNutsFix.nuts_state,_NativeNutsFix.refresh_momentum!!,_NativeNutsFix.nuts!!,f1)
    c2=RK.compile_nuts_native(p2,_NativeNutsFix2.nuts_state,_NativeNutsFix2.refresh_momentum!!,_NativeNutsFix2.nuts!!,f2)
    @test RK._native_program_parts(c1.program).owner !== RK._native_program_parts(c2.program).owner
    c1.root!(f1,(),Random.Xoshiro(1)); c2.root!(f2,(),Random.Xoshiro(1)); c1.root!(f1,(),Random.Xoshiro(2))
    @test all(isfinite,_native_obs(p1,f1).pos) && all(isfinite,_native_obs(p2,f2).pos)
end

@testset "native NUTS — registry-free total type-tree encoder" begin
    irs = RK.method_irs(_NativeNutsFix.nuts_state)
    E = RK._native_encode_program(irs, Nothing, RK.kernel_token(_NativeNutsFix.nuts_state),
                                  RK.kernel_module(_NativeNutsFix.nuts_state))
    @test E.program <: RK._NativeProgram
    @test RK._native_program_node_count(E.program) == 444
    @test Tuple(ir.id.decl for ir in irs) == (1,2,3,4,5,6,7,8,9)
    @test Tuple(ir.id.name for ir in irs) ==
          (:reset!, :collectstats!, :logadvanceprob, :swapproposal!, :step!, :flip!, :flip_neg!, :finish!, :start!)
    @test length(E.refs) == length(E.callees) == length(E.registrations)
    @test length(E.refs) > 10
    @test all(r -> r isa RK._CapturedCalleeRef, E.refs)
    @test !any(v -> v isa Dict || v isa Set || v isa Base.RefValue, E.callees)
    # Re-encoding one definition is deterministic and yields the identical program/callee tuple types.
    E2 = RK._native_encode_program(RK.method_irs(_NativeNutsFix.nuts_state), Nothing,
                                   RK.kernel_token(_NativeNutsFix.nuts_state),
                                   RK.kernel_module(_NativeNutsFix.nuts_state))
    @test E2.program === E.program
    @test typeof(E2.callees) === typeof(E.callees)
    # Independent raw-vs-type semantic fingerprints pin every load-bearing path/target/callee/default.
    @test RK._native_program_fingerprint(E.program,E.refs,E.registrations) ==
          RK._native_raw_program_fingerprint(irs)

    stats = only(RK.method_irs(_NativeNutsFix.nuts_stats!))
    ES = RK._native_encode_program(irs,Nothing,RK.kernel_token(_NativeNutsFix.nuts_state),
        RK.kernel_module(_NativeNutsFix.nuts_state);stats_ir=stats,stats_produced=(1,3))
    @test !(RK._native_program_parts(ES.program).stats <: RK._NNNoStats)
    @test RK._native_program_fingerprint(ES.program,ES.refs,ES.registrations) ==
          RK._native_raw_program_fingerprint(irs;stats_ir=stats)
end

@testset "native NUTS — totality rejects unsupported/ambiguous authority" begin
    b = RK._NativeCalleeBuilder(@__MODULE__)
    @test_throws RK._NativeEncodeReject RK._native_encode(
        RK._OpCall(GlobalRef(Base, :identity), (RK._Lit(1),), (), false), b)
    @test_throws RK._NativeEncodeReject RK._native_encode(
        RK._NodeExpr(RK._Lit(1), :(1), @__MODULE__), b)
    @test_throws RK._NativeEncodeReject RK._native_encode(
        RK._ExtRef(GlobalRef(@__MODULE__, :not_structural)), b)
    @test_throws RK._NativeEncodeReject RK._native_encode(RK._Lit(UInt(1)), b)
    @test_throws RK._NativeEncodeReject RK._native_encode(RK._Lit(Ptr{Nothing}(0)), b)
    @test_throws RK._NativeEncodeReject RK._native_encode(
        RK._ExtRef(GlobalRef(_NativeFakeStructural, :Bool)), b)
    # A full captured registration paired with a different live authored slot is a rebind, never a name match.
    irs=RK.method_irs(_NativeNutsFix.nuts_state); found=Ref{Any}(nothing)
    function findreg(x)
        x isa RK._RegisteredCall && x.registration.kind !== :intrinsic && found[]===nothing && (found[]=x)
        if x isa Tuple || x isa AbstractVector; foreach(findreg,x)
        elseif x isa Pair; findreg(x.second)
        elseif x isa RK._MExpr || x isa RK._MStmt; foreach(f->findreg(getfield(x,f)),fieldnames(typeof(x))) end
    end
    foreach(ir->findreg(ir.body),irs); x=found[]
    badref=RK._CapturedCalleeRef(GlobalRef(@__MODULE__,:identity),nothing)
    bad=RK._RegisteredCall(badref,x.registration,false,x.args,x.kw,x.broadcast)
    @test_throws RK._NativeEncodeReject RK._native_encode(bad,b)
end


@testset "native NUTS — encoded sibling graph is closed and root-unique" begin
    irs = collect(RK.method_irs(_NativeNutsFix.nuts_state))
    stepidx = only(i for i in eachindex(irs) if irs[i].id.name === :step!)
    step = irs[stepidx]; c = step.body[1]
    oldid = c.candidates[1].id
    badid = RK.MethodId(oldid.name,999,oldid.npos_req,oldid.npos_opt,oldid.kw_req,oldid.kw_opt,
                        oldid.argtypes,oldid.wheres)
    badcall = RK._Call(c.name,(RK._CallCandidate(badid,c.candidates[1].kind),),c.target,c.pos,c.kw)
    badbody = (badcall,step.body[2:end]...)
    irs[stepidx] = RK.MethodIR(step.id,step.self,step.formals,badbody,step.control,step.effects,
        step.resolution_deps,step.kind,step.ok,step.reason,step.signature)
    @test_throws RK._NativeEncodeReject RK._native_encode_program(irs,Nothing,
        RK.kernel_token(_NativeNutsFix.nuts_state),RK.kernel_module(_NativeNutsFix.nuts_state))

    irs2 = collect(RK.method_irs(_NativeNutsFix.nuts_state)); firstir=irs2[1]; id=firstir.id
    rootid=RK.MethodId(:step!,id.decl,id.npos_req,id.npos_opt,id.kw_req,id.kw_opt,id.argtypes,id.wheres)
    irs2[1]=RK.MethodIR(rootid,firstir.self,firstir.formals,firstir.body,firstir.control,firstir.effects,
        firstir.resolution_deps,firstir.kind,firstir.ok,firstir.reason,firstir.signature)
    @test_throws RK._NativeEncodeReject RK._native_encode_program(irs2,Nothing,
        RK.kernel_token(_NativeNutsFix.nuts_state),RK.kernel_module(_NativeNutsFix.nuts_state))
end


@inline function _sealed_public_batch(skel, sampler, rng, ::Val{N}) where {N}
    i=0
    @inbounds while i<N
        skel(sampler; rng=rng)
        i+=1
    end
    nothing
end
function _sealed_public_alloc(skel, sampler, rng, ::Val{N}) where {N}
    _sealed_public_batch(skel,sampler,rng,Val(N)); GC.gc()
    @allocated _sealed_public_batch(skel,sampler,rng,Val(N))
end

# Test-only reconstruction of the PRE-FAST-PATH emitter: replace each exact emitted guarded-helper call by the
# old generic `ldiv!(dest,factor,rhs)`. This operates on the compiler-emitted RGF body, not on an independently authored
# sampler, and returns the replacement count so a disconnected/no-op oracle cannot false-green the parity gate.
function _force_generic_diag_ldiv(x)
    x isa Expr || return (x, 0)
    if x.head === :call && x.args[1] === :_pp_diag_cholesky_ldiv! && length(x.args) == 4
        return (Expr(:call,:ldiv!,x.args[2],x.args[3],x.args[4]),1)
    end
    if x.head === :block
        ys = Any[]; n = 0; i = 1
        while i <= length(x.args)
            y, m = _force_generic_diag_ldiv(x.args[i]); push!(ys, y); n += m; i += 1
        end
        return (Expr(:block, ys...), n)
    end
    ys = Any[]; n = 0
    for a in x.args
        y, m = _force_generic_diag_ldiv(a); push!(ys, y); n += m
    end
    Expr(x.head, ys...), n
end

function _generic_diag_native_root(C)
    leafbody, nl = _force_generic_diag_ldiv(getfield(C.cfg.leaf, :body))
    leaf = RK.compile(:((owned,shared,handles,__lf_stepkw) -> $leafbody))
    names = keys(C.cfg.ensures); ens = Any[]; ne = 0
    for name in names
        body, n = _force_generic_diag_ldiv(getfield(getfield(C.cfg.ensures, name), :body))
        push!(ens, RK.compile(:((owned,shared,handles) -> $body))); ne += n
    end
    nl == 1 || error("generic oracle expected exactly one leaf ldiv replacement, got $nl")
    ne == 2 || error("generic oracle expected exactly two ensure ldiv replacements, got $ne")
    ensures = NamedTuple{names}(Tuple(ens))
    cfg = merge(C.cfg, (leaf=leaf, ensures=ensures))
    R = Core.apply_type(RK._CompiledNutsRootNative, C.program, typeof(C.refresh), typeof(cfg), typeof(C.cfg.handles))
    (root=R(C.refresh,cfg,C.cfg.handles), replacements=nl+ne)
end

function _native_full_obs(pf, f)
    base = _native_obs(pf,f)
    masks = (init=RK._canon_current_mask(f.init),fwd=RK._canon_current_mask(f.fwd),
             bwd=RK._canon_current_mask(f.bwd),
             proposals=Tuple(RK._canon_current_mask(p) for p in f.proposals),
             shared=RK._canon_current_mask(f.shared),
             diverged_pending=RK.nuts_frame_diverged_pending(f),
             diverged_committed=RK.nuts_frame_diverged_committed(f))
    (base=base,masks=masks)
end

@testset "native NUTS — fast Diagonal public trajectory is byte-identical to forced-generic emitted root" begin
    for T in (Float64,Float32), scaled in (false,true)
        pf=_native_pf(); metric=Diagonal(scaled ? T[2,3] : T[1,1])
        fast=RK._build_nuts_sampler(pf,_native_vals(pf,T;metric),_NativeNutsFix.nuts_state,
            _NativeNutsFix.refresh_momentum!!,_NativeNutsFix.nuts!!;
            step_f=RK.partial(_NativeNutsFix.leapfrog!;stepsize=T(.1)),max_depth=5,
            min_dham=T(-1000),stats_f=_NativeNutsFix.nuts_stats!)
        fg=_native_frame(pf,T,5;metric,stats=_NativeNutsFix.nuts_stats!)
        Cg=RK.compile_nuts_native(pf,_NativeNutsFix.nuts_state,_NativeNutsFix.refresh_momentum!!,
                                  _NativeNutsFix.nuts!!,fg)
        oracle=_generic_diag_native_root(Cg)
        @test oracle.replacements == 3
        generic=RK.nuts_sampler(Val(RK.kernel_token(_NativeNutsFix.nuts_state)),Val(Cg.RootToken),
                                fg,oracle.root,())
        rf=Random.Xoshiro(71); rg=Random.Xoshiro(71); ff=RK.nuts_sealed_frame(fast)
        for _ in 1:80
            @test _NativeNutsFix.nuts!!(fast;rng=rf) === fast
            @test _NativeNutsFix.nuts!!(generic;rng=rg) === generic
            @test _native_full_obs(pf,ff) == _native_full_obs(pf,fg)
        end
    end
end

@testset "native NUTS — sealed public Diagonal solve is inferred + exact loop 0-B, two RNG" begin
    for T in (Float64,Float32)
        pf=_native_pf(); metric=Diagonal(T[2,3])
        k=RK._build_nuts_sampler(pf,_native_vals(pf,T;metric),_NativeNutsFix.nuts_state,
            _NativeNutsFix.refresh_momentum!!,_NativeNutsFix.nuts!!;
            step_f=RK.partial(_NativeNutsFix.leapfrog!;stepsize=T(.1)),max_depth=5,
            min_dham=T(-1000),stats_f=nothing)
        @test RK.nuts_sealed_metric(k) isa Diagonal{T,Vector{T}}
        @test RK.nuts_sealed_chol_metric(k) isa Cholesky{T,<:Diagonal{T,Vector{T}}}
        @test (@inferred _NativeNutsFix.nuts!!(k;rng=Random.Xoshiro(3))) === k
        for rg in (Random.Xoshiro(8),Random.MersenneTwister(9))
            @test _sealed_public_alloc(_NativeNutsFix.nuts!!,k,rg,Val(64)) == 0
            @test _sealed_public_alloc(_NativeNutsFix.nuts!!,k,rg,Val(256)) == 0
        end
    end
end

@testset "native NUTS — production sampler carries one compiler-derived sealed evidence chain" begin
    pf=_native_pf()
    k=RK._build_nuts_sampler(pf,_native_vals(pf,Float64),_NativeNutsFix.nuts_state,
        _NativeNutsFix.refresh_momentum!!,_NativeNutsFix.nuts!!;
        step_f=RK.partial(_NativeNutsFix.leapfrog!;stepsize=.1),max_depth=5,
        min_dham=-1000,stats_f=nothing)
    cert=RK.nuts_sealed_certificate(k); cp=RK._nuts_certificate_parts(cert)
    h=getfield(k,:handles); frame=getfield(k,:state)
    @test fieldcount(typeof(cert)) == 0
    @test cp.mode === :production
    @test cp.owner === RK.kernel_token(_NativeNutsFix.nuts_state) === RK.kernel_token(k)
    @test cp.root_token === RK.kernel_token(_NativeNutsFix.nuts!!)
    @test cp.plan === typeof(RK.kernel_prepared_plan(pf))
    @test cp.plan_key === RK.kernel_plan_key(RK.kernel_prepared_plan(pf))
    @test cp.program <: RK._NativeProgram
    @test cp.control === RK._NutsControlFingerprint{cp.program,
        RK._native_program_parts(cp.program).root,RK._native_program_node_count(cp.program)}
    @test cp.recipes === RK.kernel_plan_recipes(RK.kernel_prepared_plan(pf))
    @test cp.integrator === RK.prepared_callable_token(RK.nuts_frame_step(frame))
    @test (@inferred RK.nuts_sealed_certificate(k)) === cert
    @test (@inferred RK.nuts_sealed_root(k)) === getfield(h,:root)
    @test RK.nuts_sealed_scratch(k) === getfield(h,:scratch)
    @test RK.nuts_sealed_frame(k) === frame
    @test RK.nuts_sealed_shared(k) === getfield(frame,:shared)
    plan=RK.kernel_prepared_plan(pf)
    @test (@inferred RK.nuts_sealed_metric(k)) === RK._canon_slot(frame.shared,
        RK.kernel_plan_named_slot_val(plan,Val(:metric)))
    @test RK.nuts_sealed_chol_metric(k) === RK._canon_slot(frame.shared,
        RK.kernel_plan_named_slot_val(plan,Val(:chol_metric)))

    @test (@inferred _NativeNutsFix.nuts!!(k;rng=Random.Xoshiro(3))) === k
    for rg in (Random.Xoshiro(8),Random.MersenneTwister(9))
        @test _sealed_public_alloc(_NativeNutsFix.nuts!!,k,rg,Val(64)) == 0
        @test _sealed_public_alloc(_NativeNutsFix.nuts!!,k,rg,Val(256)) == 0
    end

    # Legacy control construction is intentionally callable but cannot enter the sealed evidence path.
    legacy_frame=_native_frame(pf,Float64,5)
    control=RK.compile_nuts(pf,_NativeNutsFix.nuts_state,_NativeNutsFix.refresh_momentum!!,
                            _NativeNutsFix.nuts!!,legacy_frame)
    legacy=RK.nuts_sampler(Val(RK.kernel_token(_NativeNutsFix.nuts_state)),Val(control.RootToken),
                           legacy_frame,control.root!,control.scratch)
    @test _NativeNutsFix.nuts!!(legacy;rng=Random.Xoshiro(2)) === legacy
    @test_throws ArgumentError RK.nuts_sealed_certificate(legacy)

    # A coordinated adapter-shaped report is not an evidence input: only the concrete KernelObject dispatches.
    coordinated=(sampler=k,root=RK.nuts_sealed_root(k),frame=frame,
                 metric=RK.nuts_sealed_metric(k),certificate=cert)
    @test_throws MethodError RK.nuts_sealed_certificate(coordinated)

    # Detached root, root token, owner token, and frame/shared-metric authority each fail the one traversal.
    RT=RK.nuts_handles_root_token(h); OT=RK.kernel_token(k)
    badroot=Returns(nothing)
    hr=RK._NutsHandles(Val(RT),badroot,getfield(h,:scratch),frame,cert)
    kr=RK.KernelObject{OT,typeof(frame),typeof(hr)}(frame,hr)
    @test_throws ArgumentError RK.nuts_sealed_root(kr)
    ht=RK._NutsHandles(Val(:detached_root_token),getfield(h,:root),getfield(h,:scratch),frame,cert)
    kt=RK.KernelObject{OT,typeof(frame),typeof(ht)}(frame,ht)
    @test_throws ArgumentError RK.nuts_sealed_certificate(kt)
    ko=RK.KernelObject{:detached_owner_token,typeof(frame),typeof(h)}(frame,h)
    @test_throws ArgumentError RK.nuts_sealed_certificate(ko)
    frame2=_native_frame(pf,Float64,5)
    kf=RK.KernelObject{OT,typeof(frame2),typeof(h)}(frame2,h)
    @test_throws ArgumentError RK.nuts_sealed_metric(kf)
    good_step=getfield(frame,:step_f)
    wrong_step=RK._prepare_callable(:step_f,
        RK.partial(_NativeNutsFix2.leapfrog!;stepsize=.1))
    setfield!(frame,:step_f,wrong_step)
    @test_throws ArgumentError RK.nuts_sealed_certificate(k)
    setfield!(frame,:step_f,good_step)
    @test RK.nuts_sealed_certificate(k) === cert

    # Same owner/root strings cannot hide a different Plan type inside the zero-field certificate.
    pf2=RK._prepare_factory(_NativeNutsFix2.euclidean_phasepoint,
                            RK.kernel_registration(_NativeNutsFix2.leapfrog!))
    plan2=RK.kernel_prepared_plan(pf2); q=typeof(cert).parameters
    BadCert=Core.apply_type(RK._NutsCertificate,q[1],q[2],q[3],typeof(plan2),
        RK.kernel_plan_key(plan2),q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],
        q[15],q[16],q[17])
    hc=RK._NutsHandles(Val(RT),getfield(h,:root),getfield(h,:scratch),frame,BadCert())
    kc=RK.KernelObject{OT,typeof(frame),typeof(hc)}(frame,hc)
    @test_throws ArgumentError RK.nuts_sealed_certificate(kc)

    @test_throws ArgumentError RK._native_nuts_certificate(Val(:instrumented),pf,
        _NativeNutsFix.nuts_state,Val(RT),getfield(h,:root),getfield(h,:scratch),frame)
end


@testset "instrumented native NUTS — compiler-owned seal and exact emitted tape" begin
    pf=_native_pf()
    kp=RK._build_nuts_sampler(pf,_native_vals(pf,Float64),_NativeNutsFix.nuts_state,
        _NativeNutsFix.refresh_momentum!!,_NativeNutsFix.nuts!!;
        step_f=RK.partial(_NativeNutsFix.leapfrog!;stepsize=.1),max_depth=5,
        min_dham=-1000,stats_f=_NativeNutsFix.nuts_stats!)
    ki,_=_native_build_instrumented(pf,Float64)
    cp=RK._nuts_certificate_parts(RK.nuts_sealed_certificate(kp))
    ci=RK._nuts_certificate_parts(RK.nuts_sealed_certificate(ki))
    @test fieldcount(typeof(RK.nuts_sealed_certificate(ki))) == 0
    @test cp.mode === :production && ci.mode === :instrumented
    @test cp.owner === ci.owner && cp.root_token === ci.root_token
    @test cp.plan === ci.plan && cp.plan_key === ci.plan_key
    @test cp.program === ci.program && cp.control === ci.control
    @test cp.recipe_manifest === ci.recipe_manifest
    @test cp.recipes === ci.recipes && cp.roles === ci.roles && cp.integrator === ci.integrator
    @test RK.nuts_sealed_root(kp) !== RK.nuts_sealed_root(ki)
    @test RK.nuts_sealed_frame(kp) !== RK.nuts_sealed_frame(ki)
    @test RK.nuts_sealed_scratch(kp) === ()
    @test RK.nuts_sealed_scratch(ki) === getfield(RK.nuts_sealed_root(ki),:scratch)
    prod_lowered=join(string.(Base.code_lowered(RK.nuts_sealed_root(kp),
        Tuple{typeof(RK.nuts_sealed_frame(kp)),Tuple{},Random.Xoshiro})))
    @test !occursin("_nuts_instrument",prod_lowered)
    @test RK.nuts_instrumentation_equivalent(kp,ki)
    @test RK._nuts_real_op_signature(cp.emitted_ops) === RK._nuts_real_op_signature(ci.emitted_ops)
    @test RK._nuts_validate_emitted_ops(:production,cp.emitted_ops)
    @test RK._nuts_validate_emitted_ops(:instrumented,ci.emitted_ops)
    @test !any(T->T<:RK._NutsInstrumentWrite,cp.emitted_ops.parameters)
    its=ci.emitted_ops.parameters
    @test any(T->T<:RK._NutsInstrumentWrite,its)
    @test all(i->begin
        T=its[i]
        !(T<:RK._NutsInstrumentWrite) ||
          (i>1 && its[i-1]<:RK._NutsRealOp && T.parameters[2]==its[i-1].parameters[1])
    end,eachindex(its))
    realids=[T.parameters[1] for T in its if T<:RK._NutsRealOp]
    instrids=[T.parameters[1] for T in its if T<:RK._NutsInstrumentWrite]
    @test length(realids)==length(unique(realids))
    @test length(instrids)==length(unique(instrids))
    @test isempty(intersect(Set(realids),Set(instrids)))
    @test count(T->T<:RK._NutsRealOp && T.parameters[2]===:leaf_write,its) == 3
    R1=RK._NutsRealOp{1,:probe,:a}; R2=RK._NutsRealOp{2,:probe,:b}
    I1=RK._NutsInstrumentWrite{100001,1,:probe}; Ibad=RK._NutsInstrumentWrite{100002,1,:probe}
    I2=RK._NutsInstrumentWrite{100002,2,:probe}; Iextra=RK._NutsInstrumentWrite{100003,2,:probe}
    @test !RK._nuts_validate_emitted_ops(:production,Tuple{})
    @test !RK._nuts_validate_emitted_ops(:instrumented,Tuple{})
    @test !RK._nuts_validate_emitted_ops(:instrumented,Tuple{I1,R1})       # write before real
    @test !RK._nuts_validate_emitted_ops(:instrumented,Tuple{R1,R2,Ibad}) # declared adjacency is elsewhere
    @test !RK._nuts_validate_emitted_ops(:production,Tuple{R1,I1})
    @test !RK._nuts_validate_emitted_ops(:instrumented,Tuple{R1,I1,R1})   # duplicate real lexical id
    @test !RK._nuts_validate_emitted_ops(:instrumented,Tuple{R1,I1,R2})   # trailing real omitted its write
    @test !RK._nuts_validate_emitted_ops(:instrumented,Tuple{R1,I1,R2,I2,Iextra}) # extra write
    @test RK._nuts_validate_emitted_ops(:instrumented,Tuple{R1,I1,R2,I2})

    # No adapter-shaped object can enter the seal.  The exact scratch VALUE is retained in the root, so even
    # a same-concrete-type deep copy is detached and rejected rather than accepted by shape alone.
    coordinated=(root=RK.nuts_sealed_root(ki),scratch=RK.nuts_sealed_scratch(ki),
                 certificate=RK.nuts_sealed_certificate(ki))
    @test_throws MethodError RK.nuts_instrumented_counts(coordinated)
    h=getfield(ki,:handles); sc2=deepcopy(getfield(h,:scratch)); OT=RK.kernel_token(ki)
    h2=RK._NutsHandles(Val(RK.nuts_handles_root_token(h)),getfield(h,:root),sc2,
                       getfield(h,:frame),getfield(h,:certificate))
    k2=RK.KernelObject{OT,typeof(getfield(ki,:state)),typeof(h2)}(getfield(ki,:state),h2)
    @test_throws ArgumentError RK.nuts_instrumented_counts(k2)
    @test_throws ArgumentError getfield(h,:root)(getfield(h,:frame),sc2,Random.Xoshiro(1))
    @test !RK.nuts_instrumentation_equivalent(kp,kp)
    @test !RK.nuts_instrumentation_equivalent(ki,ki)
    # A different frame carries equal metric values but is not the sampler's actual metric authority.
    frame2=_native_frame(pf,Float64,5;stats=_NativeNutsFix.nuts_stats!)
    hf=RK._NutsHandles(Val(RK.nuts_handles_root_token(h)),getfield(h,:root),getfield(h,:scratch),
                       frame2,getfield(h,:certificate))
    kf=RK.KernelObject{OT,typeof(frame2),typeof(hf)}(frame2,hf)
    @test_throws ArgumentError RK.nuts_sealed_metric(kf)
end


@testset "instrumented native NUTS — ordinary/divergence/throw parity" begin
    for T in (Float64,Float32)
        pf=_native_pf()
        fp=_native_frame(pf,T,5;stats=_NativeNutsFix.nuts_stats!)
        fi=_native_frame(pf,T,5;stats=_NativeNutsFix.nuts_stats!)
        Cp=RK.compile_nuts_native(pf,_NativeNutsFix.nuts_state,_NativeNutsFix.refresh_momentum!!,
                                  _NativeNutsFix.nuts!!,fp)
        Ci=RK.compile_nuts_native_instrumented(pf,_NativeNutsFix.nuts_state,
            _NativeNutsFix.refresh_momentum!!,_NativeNutsFix.nuts!!,fi)
        for seed in 1:8
            Cp.root!(fp,Cp.scratch,Random.Xoshiro(seed))
            Ci.root!(fi,Ci.scratch,Random.Xoshiro(seed))
            @test _native_obs(pf,fi)==_native_obs(pf,fp)
        end
        fpd=_native_frame(pf,T,5;stats=_NativeNutsFix.nuts_stats!,min_dham=T(1e6))
        fid=_native_frame(pf,T,5;stats=_NativeNutsFix.nuts_stats!,min_dham=T(1e6))
        Dp=RK.compile_nuts_native(pf,_NativeNutsFix.nuts_state,_NativeNutsFix.refresh_momentum!!,
                                  _NativeNutsFix.nuts!!,fpd)
        Di=RK.compile_nuts_native_instrumented(pf,_NativeNutsFix.nuts_state,
            _NativeNutsFix.refresh_momentum!!,_NativeNutsFix.nuts!!,fid)
        Dp.root!(fpd,(),Random.Xoshiro(7)); Di.root!(fid,Di.scratch,Random.Xoshiro(7))
        @test _native_obs(pf,fid)==_native_obs(pf,fpd)
        @test fid.diverged && !fid.may_sample && !fid.may_continue
        @test RK._diag_slot(fid.diag,Val(2))==1

        bp=RK._nuts_instrumentation_counts(Ci.scratch)
        @test_throws ArgumentError Ci.root!(fi,Ci.scratch,_NativeThrowRNG())
        ap=RK._nuts_instrumentation_counts(Ci.scratch)
        @test ap.transitions==bp.transitions && ap.leaf_bodies==bp.leaf_bodies &&
              ap.gradients==bp.gradients
        @test RK.diagnostics_pending_mask(fi.diag)==0 && RK.diagnostics_committed_mask(fi.diag)==0
    end

    # A destination gradient that throws after entering the callable is not a completed recipe, leaf, or
    # transition.  The completion hook is physically after assignment+bless and therefore cannot false-count.
    pf=_native_pf(); g=_NativeThrowGrad([0],[false]); k,_=_native_build_instrumented(pf,Float64;grad=g)
    _NativeNutsFix.nuts!!(k;rng=Random.Xoshiro(1)); before=RK.nuts_instrumented_counts(k)
    g.fail[1]=true
    @test_throws ErrorException _NativeNutsFix.nuts!!(k;rng=Random.Xoshiro(2))
    after=RK.nuts_instrumented_counts(k)
    @test after.gradients==before.gradients
    @test after.leaf_bodies==before.leaf_bodies
    @test after.transitions==before.transitions
end


@testset "instrumented native NUTS — G10 identity chain and fixed exact-0B scratch" begin
    for T in (Float64,Float32)
        pf=_native_pf(); g=_NativeCountedGrad([0]); k,_=_native_build_instrumented(pf,T;grad=g)
        @test RK.nuts_sealed_gradient(k) === g
        _NativeNutsFix.nuts!!(k;rng=Random.Xoshiro(1))
        for seed in (2,3)
            c0=RK.nuts_instrumented_counts(k); e0=g.count[1]
            @inferred _NativeNutsFix.nuts!!(k;rng=Random.Xoshiro(seed))
            c1=RK.nuts_instrumented_counts(k); e1=g.count[1]
            n=RK._diag_slot(RK.nuts_sealed_frame(k).diag,Val(1))
            @test e1-e0 == c1.gradients-c0.gradients == c1.leaf_bodies-c0.leaf_bodies ==
                  c1.diagnostics-c0.diagnostics == n
            @test n>0
        end
        for rng in (Random.Xoshiro(91),Random.MersenneTwister(92))
            @test _sealed_public_alloc(_NativeNutsFix.nuts!!,k,rng,Val(64))==0
            @test _sealed_public_alloc(_NativeNutsFix.nuts!!,k,rng,Val(256))==0
        end
        # The fixed ring has overflowed many times, but neither it nor the complete recipe counter bank grows.
        c=RK.nuts_instrumented_counts(k)
        @test c.trace_overflows>0
        @test c.trace_length==length(RK.nuts_sealed_scratch(k).trace)
        tr=RK.nuts_instrumented_trace(k)
        static=RK.nuts_sealed_op_stream(k).parameters
        adj=Dict(T.parameters[1]=>T.parameters[2] for T in static if T<:RK._NutsInstrumentWrite)
        @test all(i->get(adj,tr[i+1],nothing)==tr[i],1:2:length(tr))
    end
end


@testset "instrumented native NUTS — G13 complete recipe census and real probes" begin
    pf=_native_pf(); metric=Diagonal(ones(Float64,2)); k,_=_native_build_instrumented(
        pf,Float64;metric=metric)
    _NativeNutsFix.nuts!!(k;rng=Random.Xoshiro(1))
    s0=RK.nuts_instrumented_schedule(k); selected=Set(s0.plan.selected_recipe_keys)
    keys(s)=Set((x.owner,x.recipe) for x in s.recompute)
    @test keys(s0)==selected
    @test length(s0.recompute)==length(unique((x.owner,x.recipe) for x in s0.recompute))
    @test length(unique((s0.plan.chol_role,s0.plan.logdet_role,s0.plan.grad_role)))==3
    @test all(in(selected),(s0.plan.chol_role,s0.plan.logdet_role,s0.plan.grad_role))
    @test any(x->x.kind===:inlined && x.id===s0.integrator.body_marker,s0.leaf_ops)
    @test any(x->x.kind===:recipe && (x.owner,x.id)==s0.plan.grad_role,s0.leaf_ops)
    @test count(x->x.kind===:inlined && x.id===s0.integrator.refresh_body_marker,s0.root_ops)==1
    @test !any(x->x.id in (s0.integrator.refresh_token,s0.integrator.refresh_body_marker),s0.leaf_ops)
    @test any(x->(x.owner,x.id)==s0.plan.logdet_role,s0.nodes)
    @test !any(x->x.kind===:recipe && (x.owner,x.id)==s0.plan.logdet_role,s0.leaf_ops)
    @test !any(x->x.kind in (:opaque,:dynamic,:registered),(s0.root_ops...,s0.leaf_ops...))

    m(s)=Dict((x.owner,x.recipe)=>x.count for x in s.recompute)
    M2=Diagonal(fill(2.0,2)); b=m(s0)
    @test RK.nuts_instrumented_mutate_metric!(k,M2) === RK.nuts_sealed_metric(k)
    s1=RK.nuts_instrumented_schedule(k); a=m(s1)
    @test keys(s1)==selected
    @test a[s0.plan.chol_role]-b[s0.plan.chol_role]==1
    @test a[s0.plan.logdet_role]-b[s0.plan.logdet_role]==1
    @test a[s0.plan.grad_role]-b[s0.plan.grad_role]==0
    pp=RK.nuts_instrumented_phasepoint(k); L=cholesky(M2)
    @test pp.dham_dmom ≈ L\pp.mom
    @test pp.kin ≈ (logdet(L)+dot(pp.mom,L\pp.mom))/2
    spre=RK.nuts_instrumented_schedule(k); pre=m(spre)
    RK.nuts_instrumented_leaf_probe!(k,pp.pos,pp.mom)
    s2=RK.nuts_instrumented_schedule(k); post=m(s2)
    @test keys(s2)==selected
    @test post[s0.plan.grad_role]-pre[s0.plan.grad_role]==1
    @test post[s0.plan.chol_role]-pre[s0.plan.chol_role]==0
    @test post[s0.plan.logdet_role]-pre[s0.plan.logdet_role]==0
    @test all(key->post[key]>=pre[key],selected)
end
