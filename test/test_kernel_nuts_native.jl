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
function _native_vals(pf,T)
    P=RK.kernel_prepared_plan(pf); m=T[2 0;0 2]; d=Dict{Int,Any}()
    for s in RK.kernel_plan_slots(P)
        n=String(s.path[1]); d[s.canon] = n=="grad_f" ? ((dst,p)->(dst .= 2 .* p;sum(abs2,p))) :
          n=="metric" ? m : n=="chol_metric" ? cholesky(m) : startswith(n,"##node") ? zero(T) :
          n=="pos" ? T[1,2] : n=="mom" ? T[3,4] :
          n in ("dpot_dpos","dham_dpos","dkin_dmom","dham_dmom") ? T[0,0] : zero(T)
    end; d
end
function _native_frame(pf,T,md;stats=nothing,min_dham=-1000)
    f=RK._construct_nuts_frame(pf,_native_vals(pf,T),md;step_f=RK.partial(_NativeNutsFix.leapfrog!;stepsize=T(.1)),
                               stats_f=stats,min_dham=min_dham)
    RK.compile_prepared_initialization(pf,typeof(f.init),typeof(f.shared))(f.init,f.shared,RK.kernel_prepared_handles(pf))
    RK._seed_nuts_children!(f); f
end
module _NativeFakeStructural
    const Bool = Core.Bool
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
