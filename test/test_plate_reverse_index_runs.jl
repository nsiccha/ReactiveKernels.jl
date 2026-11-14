module PlateReverseIndexRunTests
using ReactiveKernels, Test
const RK = ReactiveKernels

# Direct generated-code checks exercise both schedules with independent
# derivatives. The fixture has the exact run/flat shape emitted by index_runs.jl.
function scheduling_program(density;second=false)
    cell=quote
        index=$(GlobalRef(Base,:CartesianIndex))(linear)
        v=$(GlobalRef(Base.Broadcast,:_broadcast_getindex))(prepared,index)
        $(second ? :(w=$(GlobalRef(Base.Broadcast,:_broadcast_getindex))(prepared_z,index)) : nothing)
        term=$density
        total=$(GlobalRef(RK,:_plate_total_add))(total,term)
    end
    filter!(x->x!==nothing,cell.args)
    flat=macroexpand(RK,Expr(:macrocall,GlobalRef(Base,Symbol("@simd")),LineNumberNode(0),
        Expr(:for,:(linear=$(GlobalRef(RK,:_plate_simd_cells))(plate_axes)),cell)))
    # Native lowering retains the SIMD let, excluding its macro wrapper.
    Meta.isexpr(flat,:block) && (flat=only(filter(a->Meta.isexpr(a,:let),flat.args)))
    run_cell=deepcopy(cell)
    run_cell=RK._index_run_replace(run_cell) do ex
        ex==:($(GlobalRef(Base.Broadcast,:_broadcast_getindex))(prepared,index)) && return :run_read
        ex==:($(GlobalRef(RK,:_plate_total_add))(total,term)) && return :($(GlobalRef(RK,:_plate_total_add))(partial,term))
        Meta.isexpr(ex,:(=),2) && ex.args[1]===:total && return :(partial=$(GlobalRef(RK,:_plate_total_add))(partial,term))
        ex
    end
    inner=macroexpand(RK,Expr(:macrocall,GlobalRef(Base,Symbol("@simd")),LineNumberNode(0),
        Expr(:for,:(linear=lo:hi),run_cell)))
    grouped=:(for run=eachindex(getfield(metadata,1))
        lo=getfield(metadata,1)[run]
        hi=getfield(metadata,2)[run]
        run_read=$(GlobalRef(Base.Broadcast,:_broadcast_getindex))(prepared,Base.CartesianIndex(lo))
        partial=$(GlobalRef(RK,:_plate_run_seed))(total)
        $inner
        total=$(GlobalRef(RK,:_plate_total_add))(total,partial)
    end)
    guard=:(getfield(metadata,4) && length(plate_axes)==1 && Base.only(plate_axes)==Base.OneTo(getfield(metadata,3)) && total isa Number)
    params=second ? (:x,:s,:metadata,:z) : (:x,:s,:metadata)
    body=quote
        raw=Base.broadcastable(x)
        prepared=Base.Broadcast.preprocess(nothing,raw)
        $(second ? :(raw_z=Base.broadcastable(z)) : nothing)
        $(second ? :(prepared_z=Base.Broadcast.preprocess(nothing,raw_z)) : nothing)
        plate_axes=Base.axes(raw)
        total=$(GlobalRef(RK,:_plate_total_seed))(Float64,0.0)
        $(Expr(:if,guard,grouped,flat))
        total
    end
    filter!(x->x!==nothing,body.args)
    Expr(:function,Expr(:tuple,params...),body)
end

# These checks call the generator directly and cross no backend rule boundary,
# so they run without a backend's refusal of run scheduling.
function without_refusal(f)
    saved=RK._PR_INDEX_RUN_REFUSAL[]
    RK._PR_INDEX_RUN_REFUSAL[]=nothing
    try
        f()
    finally
        RK._PR_INDEX_RUN_REFUSAL[]=saved
    end
end
compile_scheduling(density;second=false)=without_refusal() do
    compile_scheduling_unrefused(density;second)
end
function compile_scheduling_unrefused(density;second=false)
    ast=scheduling_program(density;second)
    active=second ? [:x,:s,:z] : [:x,:s]
    flat, runs = RK._pr_index_run_source(ast)
    @test length(runs) == 1
    calls = map((flat, ast)) do program
        rev, scalar, batched = RK._pr_program_reverse(program, active, RK, Dict{Symbol,Any}())
        @test scalar == [:s]
        @test batched == (second ? [:x,:z] : [:x])
        @test occursin("_pr_run_disjoint", string(rev)) == (program === ast)
        RK._PlateCall(RK.compile(rev))
    end
    calls
end
@noinline direct_call(f::F,args...) where F=f(args...)
@testset "Per-run scheduling: seeds, inactive shadows, maps and internal overlap" begin
    for (label,density) in ((:gaussian,:(-log(s)-0.5*(v/s)^2)),
                            (:lazy,:(s*(v>0 ? sqrt(v) : v*v))))
        calls=compile_scheduling(density)
        for idx in (Int[],[1],repeat([1,2,3],inner=2),repeat([1,2,3],inner=8),
                    [fill(2,17);fill(1,8);fill(3,9);fill(1,13)],repeat([3,1],inner=128))
            metadata=RK._index_run_metadata(idx)
            q=[-.6,.3,1.2];x=view(q,idx);s=1.4
            storage=zeros(6)
            shadow=Base.SubArray(view(storage,2:4),(idx,))
            @test RK._pr_run_shadow_matches(shadow,x)
            @test RK._pr_run_disjoint((x,s,metadata),(shadow,))
            @test !RK._pr_run_shadow_matches(zeros(length(idx)),x)
            @test !RK._pr_run_disjoint((x,s,metadata),(x,)) || isempty(idx)
            for seed in (0.,-.6,1.7),want in (Val(false),Val(true))
                expected=label===:gaussian ? -length(idx)*log(s)-sum(abs2,x)/(2s^2) :
                    s*sum(v->v>0 ? sqrt(v) : v*v,x;init=0.)
                ds=label===:gaussian ? -length(idx)/s+sum(abs2,x)/s^3 :
                    sum(v->v>0 ? sqrt(v) : v*v,x;init=0.)
                dx=label===:gaussian ? -collect(x)/s^2 : map(v->s*(v>0 ? inv(2sqrt(v)) : 2v),x)
                for f in calls
                    dq=zeros(3);result=direct_call(f,x,s,metadata,seed,want,view(dq,idx))
                    expected_dq=zeros(3)
                    for i in eachindex(idx);expected_dq[idx[i]]+=seed*dx[i];end
                    @test dq≈expected_dq rtol=3e-11 atol=3e-11
                    @test result[2]≈seed*ds rtol=3e-11 atol=3e-11
                    want===Val(true) && @test result[1]≈expected rtol=3e-11 atol=3e-11
                    inactive=direct_call(f,x,s,metadata,seed,want,nothing)
                    @test inactive[2]≈seed*ds rtol=3e-11 atol=3e-11
                    # A dense destination needs one cotangent per cell. It
                    # must take the preserved flat fallback, not a run scatter.
                    dense=zeros(length(idx))
                    direct_call(f,x,s,metadata,seed,want,dense)
                    @test dense≈seed*dx rtol=3e-11 atol=3e-11
                end
            end
            # Nonzero incoming cotangent is supported by reverse-only mode.
            for f in calls
                dq=fill(.2,3)
                result=direct_call(f,x,s,metadata,.7,Val(false),view(dq,idx))
                baseline=fill(.2,3)
                reference=direct_call(first(calls),x,s,metadata,.7,Val(false),view(baseline,idx))
                @test dq≈baseline rtol=3e-11 atol=3e-11
                @test result[2]≈reference[2] rtol=3e-11 atol=3e-11
            end
        end
        # Internal primal/shadow overlap is a differential fallback witness,
        # not permission for a public gradient call to overwrite its inputs.
        idx=repeat([1,2,3],inner=16);metadata=RK._index_run_metadata(idx)
        baseline=[.4,.8,1.2];r=direct_call(first(calls),view(baseline,idx),1.4,metadata,.2,Val(false),view(baseline,idx))
        for f in Base.tail(calls)
            q=[.4,.8,1.2];v=direct_call(f,view(q,idx),1.4,metadata,.2,Val(false),view(q,idx))
            @test q≈baseline rtol=3e-11 atol=3e-11
            @test v[2]≈r[2] rtol=3e-11 atol=3e-11
        end
    end
    # Two independent gather maps: only the map in the compiler run metadata
    # may be hoisted/combined. The other scatter must retain its per-cell index.
    calls=compile_scheduling(:(s*v*w);second=true)
    idx=repeat([1,2,3],inner=16);other=repeat([3,1,2],outer=16)
    metadata=RK._index_run_metadata(idx);q=[.4,.8,1.2];z=[-.6,.3,.9]
    for f in calls,want in (Val(false),Val(true)), packed in (false,true)
        tape=zeros(6)
        dx,dz=packed ? (view(tape,1:3),view(tape,4:6)) : (zeros(3),zeros(3))
        sx=Base.SubArray(dx,(idx,));sz=Base.SubArray(dz,(other,))
        @test RK._pr_run_disjoint((view(q,idx),view(z,other)),(sx,sz))
        v=direct_call(f,view(q,idx),1.3,metadata,view(z,other),.7,want,sx,sz)
        a=zeros(3);b=zeros(3)
        for i in eachindex(idx)
            a[idx[i]]+=.7*1.3*z[other[i]]
            b[other[i]]+=.7*1.3*q[idx[i]]
        end
        @test dx≈a rtol=3e-11 atol=3e-11
        @test dz≈b rtol=3e-11 atol=3e-11
        @test v[2]≈.7*sum(q[idx].*z[other]) rtol=3e-11 atol=3e-11
        want===Val(true) && @test v[1]≈1.3*sum(q[idx].*z[other]) rtol=3e-11 atol=3e-11
    end
    # Actual overlapping tape slices retain the flat schedule. Compare against
    # the original reverse because these internal destinations share entries.
    for want in (Val(false),Val(true))
        baseline=zeros(5)
        sx=Base.SubArray(view(baseline,1:3),(idx,))
        sz=Base.SubArray(view(baseline,3:5),(other,))
        @test !RK._pr_run_disjoint((view(q,idx),view(z,other)),(sx,sz))
        reference=direct_call(first(calls),view(q,idx),1.3,metadata,view(z,other),.7,want,sx,sz)
        for f in Base.tail(calls)
            tape=zeros(5)
            sx=Base.SubArray(view(tape,1:3),(idx,))
            sz=Base.SubArray(view(tape,3:5),(other,))
            value=direct_call(f,view(q,idx),1.3,metadata,view(z,other),.7,want,sx,sz)
            @test tape≈baseline rtol=3e-11 atol=3e-11
            @test value[2]≈reference[2] rtol=3e-11 atol=3e-11
            want===Val(true) && @test value[1]≈reference[1] rtol=3e-11 atol=3e-11
        end
    end
    # Proving disjoint value buffers must not discard the index arrays' reads.
    indices=ones(Int,6)
    sx=Base.SubArray(reinterpret(Float64,view(indices,1:3)),(ones(Int,6),))
    sz=Base.SubArray(zeros(3),(indices,))
    @test !Base.mightalias(parent(sx),parent(sz))
    @test RK._pr_run_shadow_overlap(sx,sz)
    @test RK._pr_run_shadow_overlap(sz,sx)
    @test !RK._pr_run_disjoint((),(sx,sz))
end
@testset "A backend's refusal leaves a run program to ordinary reverse" begin
    ast=scheduling_program(:(-log(s)-0.5*(v/s)^2))
    flat=without_refusal(()->first(RK._pr_index_run_source(ast)))
    reason="this backend's rule cannot read run metadata"
    saved=RK._PR_INDEX_RUN_REFUSAL[]
    RK._PR_INDEX_RUN_REFUSAL[]=reason
    try
        failure=try
            RK._pr_program_reverse(ast,[:x,:s],RK,Dict{Symbol,Any}())
            nothing
        catch err
            err
        end
        @test failure isa RK._PRIneligible
        @test failure.reason==reason
        # A program without a run arm reads no run metadata and stays eligible.
        rev,scalar,batched=RK._pr_program_reverse(flat,[:x,:s],RK,Dict{Symbol,Any}())
        @test scalar==[:s] && batched==[:x]
        @test !occursin("_pr_run_disjoint",string(rev))
    finally
        RK._PR_INDEX_RUN_REFUSAL[]=saved
    end
end
end # module
