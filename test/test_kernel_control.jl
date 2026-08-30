# Adversaries for the ISOLATED design-B control compiler (src/kernel_control.jl). Synthetic typed stores
# are the sanctioned pre-rebase scaffold (RK 09:19c); the real callable-frame binding lands on the rebase
# onto syntax's approved seam. Each @kernel adversary lives in its own module (no global collision); each
# proves correctness + warmed exact 0-B + @inferred, and the loop adversary also proves LLVM no-Box.
using ReactiveKernels, Test, InteractiveUtils
const RKC = ReactiveKernels

mutable struct _CtlCount; count::Int; end
mutable struct _CtlAfter; count::Int; after::Int; hits::Int; end
mutable struct _CtlSum;   count::Int; sum::Int; end
mutable struct _CtlAcc;   acc::Int; end

module _CtlMS  # mutual+self recursion + acyclic-inlined base
  using ReactiveKernels
  @kernel adv(s;) = begin
    ping!(k) = begin k <= 0 && return done!(__self__); pong!(__self__, k-1); pong!(__self__, k-1) end
    pong!(x) = begin x <= 0 && return done!(__self__); ping!(__self__, x-1); ping!(__self__, x-1) end
    done!() = begin s.count = s.count + 1 end
  end
end
module _CtlEarly  # inlined helper EARLY-returns; caller writes AFTER (callee-return continuation)
  using ReactiveKernels
  @kernel adv(s;) = begin
    ping!(k) = begin
      k <= 0 && return done!(__self__)
      helper!(__self__, k-1); s.after = s.after + 1; pong!(__self__, k-1)
    end
    pong!(x) = begin x <= 0 && return done!(__self__); ping!(__self__, x-1) end
    helper!(j) = begin j <= 0 && return; s.hits = s.hits + 1 end
    done!() = begin s.count = s.count + 1 end
  end
end
module _CtlSparse  # sparse decl ordinals: SCC = decls [2,4], acyclic before/between/after
  using ReactiveKernels
  @kernel adv(s;) = begin
    pre!(z) = begin s.count = s.count + 0 end
    ping!(k) = begin k <= 0 && return done!(__self__); mid!(__self__, k); pong!(__self__, k-1) end
    mid!(z) = begin s.count = s.count + 0 end
    pong!(x) = begin x <= 0 && return done!(__self__); ping!(__self__, x-1) end
    done!() = begin s.count = s.count + 1 end
  end
end
module _CtlLoop  # SUSPENDING for-loop: loop var i is a cross-suspension local (read after the SCC call)
  using ReactiveKernels
  @kernel adv(s;) = begin
    ping!(k) = begin k <= 0 && return done!(__self__); pong!(__self__, k-1) end
    pong!(x) = begin x <= 0 && return done!(__self__); ping!(__self__, x-1) end
    done!() = begin s.count = s.count + 1 end
    driver!(m) = begin for i in 1:m; ping!(__self__, i); s.sum = s.sum + i end end
  end
end
module _CtlLoopB  # SUSPENDING for-loop with BREAK
  using ReactiveKernels
  @kernel adv(s;) = begin
    ping!(k) = begin k <= 0 && return done!(__self__); pong!(__self__, k-1) end
    pong!(x) = begin x <= 0 && return done!(__self__); ping!(__self__, x-1) end
    done!() = begin s.count = s.count + 1 end
    driverB!(m) = begin for i in 1:m; i > 3 && break; ping!(__self__, i); s.sum = s.sum + i end end
  end
end
module _CtlWhile  # SUSPENDING while-loop + continue; local counter i is a cross-suspension local
  using ReactiveKernels
  @kernel adv(s;) = begin
    ping!(k) = begin k <= 0 && return done!(__self__); pong!(__self__, k-1) end
    pong!(x) = begin x <= 0 && return done!(__self__); ping!(__self__, x-1) end
    done!() = begin s.count = s.count + 1 end
    driver!(m) = begin
      i = 0
      while i < m
        i = i + 1
        i == 2 && continue
        ping!(__self__, i)
        s.sum = s.sum + i
      end
    end
  end
end
module _CtlNative  # ACYCLIC helper with a NON-suspending native for-loop (inlined natively)
  using ReactiveKernels
  @kernel adv(s;) = begin
    ping!(k) = begin k <= 0 && return reset!(__self__); pong!(__self__, k-1) end
    pong!(x) = begin x <= 0 && return reset!(__self__); ping!(__self__, x-1) end
    reset!() = begin for j in 1:3; s.acc = s.acc + j end end
  end
end

_ctl_st1(cap) = RKC._FrameStore{1,Tuple{Vector{Int}}}((Vector{Int}(undef,cap),))
_ctl_st2(cap) = RKC._FrameStore{2,Tuple{Vector{Int}}}((Vector{Int}(undef,cap),))

@testset "control — backend-neutral program retains source CFG and effects" begin
    irs = RKC.method_irs(_CtlMS.adv)
    program = RKC._control_program_from_irs(irs; root_mid=1)
    @test program.methods == (1, 2)
    @test program.names == Dict(1 => :ping!, 2 => :pong!)
    @test program.stored == Dict(1 => (:k,), 2 => (:x,))
    @test program.formal_positions ==
          Dict(1 => Dict(:k => 1), 2 => Dict(:x => 1))
    @test program.root_mid == 1
    @test program.root_entry == program.entries[1]
    @test all(block -> block.mid in program.methods, program.blocks)
    @test all(block -> block.name == program.names[block.mid], program.blocks)
    @test any(block -> block.term === :call && block.callee_mid != 0,
              program.blocks)
    @test any(block -> block.term === :branch && hasproperty(block, :condition),
              program.blocks)
    @test any(block -> !isempty(block.effects), program.blocks)

    by_name = RKC._control_program(_CtlMS.adv; root_name=:ping!)
    @test by_name.root_mid == program.root_mid
    @test map(block -> (block.mid, block.pc, block.term), by_name.blocks) ==
          map(block -> (block.mid, block.pc, block.term), program.blocks)

    @test_throws ArgumentError RKC._control_program(_CtlMS.adv;
                                                     root_name=:missing!)
end

@testset "control — mutual+self recursion (2^n) + acyclic-inlined base, 0-B/@inferred" begin
    irs = RKC.method_irs(_CtlMS.adv); cap=256
    @test sort(collect(RKC.defunctionalized_mids(irs))) == [1,2]        # only ping!/pong! defunctionalized
    fn = RKC.compile_dispatcher(irs; typemap=Dict(1=>Dict(:k=>Int),2=>Dict(:x=>Int)), cap=cap, root_mid=1)
    mk()=(_ctl_st1(cap),_ctl_st2(cap),Vector{RKC._CtrlFrame}(undef,cap))
    for n in 0:5; st=_CtlCount(0); fn(st,mk(),n); @test st.count == 2^n; end
    st=_CtlCount(0); sc=mk(); fn(st,sc,5); fn(st,sc,5); @test (@allocated fn(st,sc,5)) == 0
    Test.@inferred fn(_CtlCount(0), mk(), 5)
end

@testset "control — callee-return continuation: inlined helper early-returns, caller writes AFTER" begin
    irs = RKC.method_irs(_CtlEarly.adv); cap=256
    fn = RKC.compile_dispatcher(irs; typemap=Dict(1=>Dict(:k=>Int),2=>Dict(:x=>Int)), cap=cap, root_mid=1)
    mk()=(_ctl_st1(cap),_ctl_st2(cap),Vector{RKC._CtrlFrame}(undef,cap))
    st=_CtlAfter(0,0,0); fn(st,mk(),1)                                  # ping!(1): helper!(0) early-returns
    @test st.after == 1 && st.hits == 0 && st.count == 1               # caller write AFTER still executed
    st=_CtlAfter(0,0,0); sc=mk(); fn(st,sc,3); fn(st,sc,3); @test (@allocated fn(st,sc,3)) == 0
end

@testset "control — sparse decl ordinals (SCC=[2,4], acyclic before/between/after)" begin
    irs = RKC.method_irs(_CtlSparse.adv); cap=256
    @test sort(collect(RKC.defunctionalized_mids(irs))) == [2,4]
    fn = RKC.compile_dispatcher(irs; typemap=Dict(2=>Dict(:k=>Int),4=>Dict(:x=>Int)), cap=cap, root_mid=2)
    mk()=(RKC._FrameStore{2,Tuple{Vector{Int}}}((Vector{Int}(undef,cap),)),
          RKC._FrameStore{4,Tuple{Vector{Int}}}((Vector{Int}(undef,cap),)), Vector{RKC._CtrlFrame}(undef,cap))
    for n in 1:4; st=_CtlCount(0); fn(st,mk(),n); @test st.count == 1; end   # linear chain -> one base hit
    st=_CtlCount(0); sc=mk(); fn(st,sc,4); fn(st,sc,4); @test (@allocated fn(st,sc,4)) == 0
    Test.@inferred fn(_CtlCount(0), mk(), 4)
end

@testset "control — SUSPENDING loop + cross-suspension local + break, 0-B/@inferred/LLVM-no-Box" begin
    irs = RKC.method_irs(_CtlLoop.adv); cap=512
    drv = irs[findfirst(ir->ir.id.name===:driver!, irs)]
    tm = Dict(1=>Dict(:k=>Int),2=>Dict(:x=>Int),drv.id.decl=>Dict(:m=>Int,:i=>Int))
    fn = RKC.compile_dispatcher(irs; typemap=tm, cap=cap, root_mid=drv.id.decl)
    SD()=RKC._FrameStore{drv.id.decl,Tuple{Vector{Int},Vector{Int}}}((Vector{Int}(undef,cap),Vector{Int}(undef,cap)))
    mk()=(_ctl_st1(cap),_ctl_st2(cap),SD(),Vector{RKC._CtrlFrame}(undef,cap))
    for m in 1:4; st=_CtlSum(0,0); fn(st,mk(),m); @test st.count==m && st.sum==m*(m+1)÷2; end
    st=_CtlSum(0,0); sc=mk(); fn(st,sc,4); fn(st,sc,4); @test (@allocated fn(st,sc,4)) == 0
    Test.@inferred fn(_CtlSum(0,0), mk(), 4)
    # LLVM no-Box / no dynamic dispatch (RK 09:58)
    io=IOBuffer(); code_llvm(io, fn, (_CtlSum, typeof(mk()), Int); debuginfo=:none); llvm=String(take!(io))
    @test !occursin("jl_box", llvm) && !occursin("jl_apply_generic", llvm)
end

@testset "control — SUSPENDING loop with BREAK (i>3 -> only i=1,2,3)" begin
    irs = RKC.method_irs(_CtlLoopB.adv); cap=512
    dvb=irs[findfirst(ir->ir.id.name===:driverB!,irs)]
    fb = RKC.compile_dispatcher(irs; typemap=Dict(1=>Dict(:k=>Int),2=>Dict(:x=>Int),dvb.id.decl=>Dict(:m=>Int,:i=>Int)),
                                cap=cap, root_mid=dvb.id.decl)
    SDB()=RKC._FrameStore{dvb.id.decl,Tuple{Vector{Int},Vector{Int}}}((Vector{Int}(undef,cap),Vector{Int}(undef,cap)))
    mkb()=(_ctl_st1(cap),_ctl_st2(cap),SDB(),Vector{RKC._CtrlFrame}(undef,cap))
    st=_CtlSum(0,0); fb(st,mkb(),10); @test st.count==3 && st.sum==6
    st=_CtlSum(0,0); sc=mkb(); fb(st,sc,10); fb(st,sc,10); @test (@allocated fb(st,sc,10)) == 0
end

@testset "control — NON-suspending native loop (acyclic reset! inlined as a native for), 0-B" begin
    irs = RKC.method_irs(_CtlNative.adv); cap=256
    fn = RKC.compile_dispatcher(irs; typemap=Dict(1=>Dict(:k=>Int),2=>Dict(:x=>Int)), cap=cap, root_mid=1)
    mk()=(_ctl_st1(cap),_ctl_st2(cap),Vector{RKC._CtrlFrame}(undef,cap))
    for n in 1:3; st=_CtlAcc(0); fn(st,mk(),n); @test st.acc == 6; end   # reset! native loop 1+2+3
    st=_CtlAcc(0); sc=mk(); fn(st,sc,3); fn(st,sc,3); @test (@allocated fn(st,sc,3)) == 0
    Test.@inferred fn(_CtlAcc(0), mk(), 3)
end

@testset "control — native-loop accumulator is spilled to its successor block" begin
    acc = RKC._LocalAssign((:acc,), RKC._Lit(1))
    loop = RKC._For((:i,), RKC._Lit(1), (acc,))
    raw = RKC._RawStmt((:for_native, loop))
    @test RKC._block_writes([raw], Set([:acc])) == [:acc]
    @test isempty(RKC._block_writes([raw], Set([:other])))
end

@testset "control — SUSPENDING while-loop + continue (skip i==2), cross-suspension local, 0-B" begin
    irs = RKC.method_irs(_CtlWhile.adv); cap=512
    drv = irs[findfirst(ir->ir.id.name===:driver!, irs)]
    tm = Dict(1=>Dict(:k=>Int),2=>Dict(:x=>Int),drv.id.decl=>Dict(:m=>Int,:i=>Int))
    fn = RKC.compile_dispatcher(irs; typemap=tm, cap=cap, root_mid=drv.id.decl)
    SD()=RKC._FrameStore{drv.id.decl,Tuple{Vector{Int},Vector{Int}}}((Vector{Int}(undef,cap),Vector{Int}(undef,cap)))
    mk()=(_ctl_st1(cap),_ctl_st2(cap),SD(),Vector{RKC._CtrlFrame}(undef,cap))
    # m=4: i=1(ping,+1) i=2(continue) i=3(ping,+3) i=4(ping,+4) -> count=3, sum=8
    st=_CtlSum(0,0); fn(st,mk(),4); @test st.count==3 && st.sum==8
    st=_CtlSum(0,0); fn(st,mk(),5); @test st.count==4 && st.sum==1+3+4+5   # skip i==2
    st=_CtlSum(0,0); sc=mk(); fn(st,sc,5); fn(st,sc,5); @test (@allocated fn(st,sc,5)) == 0
    Test.@inferred fn(_CtlSum(0,0), mk(), 5)
end
