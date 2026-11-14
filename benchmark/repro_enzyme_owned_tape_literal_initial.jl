# Diagnostic difference: pass the literal initial total 0.0 at the call site.
# Standalone diagnostic. CUT is the actual emitted expression; only its module
# references are relocated below. The helper subset is copied from RK
# b6569aa6b. ReactiveKernels is not imported.
using Enzyme, DifferentiationInterface, Test
const ER = Enzyme.EnzymeRules
const DI = DifferentiationInterface
module CutHelpers
@inline _native_mul(args...) = *(args...)
@inline _plate_total_add(total, cell) = total + cell
@inline _plate_simd_cells(axes::Tuple{Any}) = only(axes)
@inline _plate_simd_cells(axes::Tuple) = CartesianIndices(axes)
@inline _plate_spans_axis(output_axes, arg::AbstractArray) = axes(arg) == output_axes
@inline _plate_spans_axis(output_axes, arg) = false
Base.@propagate_inbounds _plate_read(arg::AbstractArray, index) = arg[index]
Base.@propagate_inbounds _plate_read(arg, index) = Base.Broadcast._broadcast_getindex(arg,index)
@inline _pr_spans(A::Base.Broadcast.Extruded,cells) = ndims(A.x)==ndims(cells) && all(A.keeps)
@inline _pr_spans(::Union{Number,Base.RefValue},cells) = true
@inline _pr_spans(_,cells) = false
@inline _pr_spans_all(cells,operands...) = all(A->_pr_spans(A,cells),operands)
@inline _pr_direct_read(A::Base.Broadcast.Extruded,i) = @inbounds A.x[i]
@inline _pr_direct_read(x::Number,i) = x
@inline _pr_direct_read(r::Base.RefValue,i) = r[]
_pr_check_shadow(::Nothing,x) = nothing
function _pr_check_shadow(shadow::AbstractArray,x::AbstractArray)
    axes(shadow)==axes(x) && ndims(x)==1 || error("incompatible shadow")
    nothing
end
@inline _pr_shadow_add!(::Nothing,i,v) = nothing
@inline function _pr_shadow_add!(shadow::AbstractArray,i,v)
    index=length(shadow)==1 ? firstindex(shadow) : i
    @inbounds shadow[index] += v
    nothing
end
@inline _pr_scaled(k,acc) = iszero(acc) ? zero(k*acc) : k*acc
_pr_scale_buffer!(::Nothing,k) = nothing
_pr_scale_buffer!(buffer::SubArray,k) = _pr_scale_owned!(parent(buffer),k)
_pr_scale_buffer!(buffer::Array,k) = _pr_scale_owned!(buffer,k)
function _pr_scale_owned!(q,k)
    @inbounds for j in eachindex(q)
        q[j]=_pr_scaled(k,q[j])
    end
    nothing
end
end
relocate(x) = x
relocate(x::Symbol) = x === :ReactiveKernels ? :CutHelpers : x
relocate(x::Expr) = Expr(x.head,map(relocate,x.args)...)
const CUT = eval(relocate(
:((var"##plate_argument#1", var"##plate_argument#2", var"##plate_axes#3", s, var"##plate_total#4", var"##ybar#5", _pr_want_value, var"##plate_argument_shadow#6")->begin
          ReactiveKernels._pr_check_shadow(var"##plate_argument_shadow#6", var"##plate_argument#1")
          var"##plate_argument_bar#7" = 0.0
          var"##plate_prepared_bar#8" = 0.0
          var"##plate_total_bar#9" = var"##ybar#5"
          var"##plate_prepared#10" = Base.Broadcast.preprocess(nothing, var"##plate_argument#1")
          var"##plate_prepared#11" = Base.Broadcast.preprocess(nothing, var"##plate_argument#2")
          if ReactiveKernels._plate_spans_axis(var"##plate_axes#3", var"##plate_argument#1") && ReactiveKernels._plate_spans_axis(var"##plate_axes#3", var"##plate_argument#2")
              begin
                  var"#s#direct#op5#12" = s
                  _pr_cells7 = ReactiveKernels._plate_simd_cells(var"##plate_axes#3")
                  _pr_ncells7 = Float64(length(_pr_cells7))
                  _pr_coef9 = var"##plate_total_bar#9" * var"#s#direct#op5#12"
                  _pr_value11 = 0.0
                  if ReactiveKernels._pr_spans_all(_pr_cells7, var"##plate_argument#1", var"##plate_argument#2")
                      #= /home/niko/scratch/plate-tape-storage-20261114/src/plate_reverse.jl:1618 =# @inbounds for var"##plate_linear#direct#13" = _pr_cells7
                              _pr_t1 = Base.CartesianIndex(var"##plate_linear#direct#13")
                              var"##plate_index#direct#14" = _pr_t1
                              _pr_t2 = ReactiveKernels._pr_direct_read(var"##plate_argument#1", var"##plate_linear#direct#13")
                              var"##inbounds_value#direct#15" = _pr_t2
                              var"#v#direct#op5#16" = var"##inbounds_value#direct#15"
                              _pr_t3 = ReactiveKernels._pr_direct_read(var"##plate_argument#2", var"##plate_linear#direct#13")
                              var"##inbounds_value#direct#17" = _pr_t3
                              var"#y#direct#op5#18" = var"##inbounds_value#direct#17"
                              _pr_t4 = ReactiveKernels._native_mul(var"#s#direct#op5#12", var"#v#direct#op5#16")
                              _pr_t5 = _pr_t4 + var"#y#direct#op5#18"
                              _pr_t6 = Base.sin(_pr_t5)
                              var"##plate___plate_value__#direct#19" = _pr_t6
                              _pr_want_value isa Val{true} && (_pr_value11 = _pr_value11 + _pr_t6)
                              _pr_f8 = Base.cos(_pr_t5)
                              if _pr_want_value isa Val{true} && !(var"##plate_argument_shadow#6" isa Nothing)
                                  ReactiveKernels._pr_shadow_add!(var"##plate_argument_shadow#6", var"##plate_linear#direct#13", _pr_f8)
                              else
                                  _pr_f10 = _pr_coef9 * _pr_f8
                                  ReactiveKernels._pr_shadow_add!(var"##plate_argument_shadow#6", var"##plate_linear#direct#13", _pr_f10)
                              end
                          end
                  else
                      #= /home/niko/scratch/plate-tape-storage-20261114/src/plate_reverse.jl:1624 =# @inbounds for var"##plate_linear#direct#13" = _pr_cells7
                              _pr_t1 = Base.CartesianIndex(var"##plate_linear#direct#13")
                              var"##plate_index#direct#14" = _pr_t1
                              _pr_t2 = ReactiveKernels._plate_read(var"##plate_argument#1", var"##plate_index#direct#14")
                              var"##inbounds_value#direct#15" = _pr_t2
                              var"#v#direct#op5#16" = var"##inbounds_value#direct#15"
                              _pr_t3 = ReactiveKernels._plate_read(var"##plate_argument#2", var"##plate_index#direct#14")
                              var"##inbounds_value#direct#17" = _pr_t3
                              var"#y#direct#op5#18" = var"##inbounds_value#direct#17"
                              _pr_t4 = ReactiveKernels._native_mul(var"#s#direct#op5#12", var"#v#direct#op5#16")
                              _pr_t5 = _pr_t4 + var"#y#direct#op5#18"
                              _pr_t6 = Base.sin(_pr_t5)
                              var"##plate___plate_value__#direct#19" = _pr_t6
                              _pr_want_value isa Val{true} && (_pr_value11 = _pr_value11 + _pr_t6)
                              _pr_f8 = Base.cos(_pr_t5)
                              if _pr_want_value isa Val{true} && !(var"##plate_argument_shadow#6" isa Nothing)
                                  ReactiveKernels._pr_shadow_add!(var"##plate_argument_shadow#6", var"##plate_linear#direct#13", _pr_f8)
                              else
                                  _pr_f10 = _pr_coef9 * _pr_f8
                                  ReactiveKernels._pr_shadow_add!(var"##plate_argument_shadow#6", var"##plate_linear#direct#13", _pr_f10)
                              end
                          end
                  end
                  if _pr_want_value isa Val{true} && _pr_ncells7 > 0
                      var"##plate_total#4" = ReactiveKernels._plate_total_add(var"##plate_total#4", 0.0 + 1.0_pr_value11)
                  end
                  if _pr_want_value isa Val{true} && !(var"##plate_argument_shadow#6" isa Nothing)
                      ReactiveKernels._pr_scale_buffer!(var"##plate_argument_shadow#6", _pr_coef9)
                  end
              end
              Core.nothing
          else
              begin
                  var"#s#op5#20" = s
                  _pr_cells18 = ReactiveKernels._plate_simd_cells(var"##plate_axes#3")
                  _pr_ncells18 = Float64(length(_pr_cells18))
                  _pr_coef20 = var"##plate_total_bar#9" * var"#s#op5#20"
                  _pr_value22 = 0.0
                  if ReactiveKernels._pr_spans_all(_pr_cells18, var"##plate_prepared#10", var"##plate_prepared#11")
                      #= /home/niko/scratch/plate-tape-storage-20261114/src/plate_reverse.jl:1618 =# @inbounds for var"##plate_linear#21" = _pr_cells18
                              _pr_t12 = Base.CartesianIndex(var"##plate_linear#21")
                              var"##plate_index#22" = _pr_t12
                              _pr_t13 = ReactiveKernels._pr_direct_read(var"##plate_prepared#10", var"##plate_linear#21")
                              var"##inbounds_value#23" = _pr_t13
                              var"#v#op5#24" = var"##inbounds_value#23"
                              _pr_t14 = ReactiveKernels._pr_direct_read(var"##plate_prepared#11", var"##plate_linear#21")
                              var"##inbounds_value#25" = _pr_t14
                              var"#y#op5#26" = var"##inbounds_value#25"
                              _pr_t15 = ReactiveKernels._native_mul(var"#s#op5#20", var"#v#op5#24")
                              _pr_t16 = _pr_t15 + var"#y#op5#26"
                              _pr_t17 = Base.sin(_pr_t16)
                              var"##plate___plate_value__#27" = _pr_t17
                              _pr_want_value isa Val{true} && (_pr_value22 = _pr_value22 + _pr_t17)
                              _pr_f19 = Base.cos(_pr_t16)
                              if _pr_want_value isa Val{true} && !(var"##plate_argument_shadow#6" isa Nothing)
                                  ReactiveKernels._pr_shadow_add!(var"##plate_argument_shadow#6", var"##plate_linear#21", _pr_f19)
                              else
                                  _pr_f21 = _pr_coef20 * _pr_f19
                                  ReactiveKernels._pr_shadow_add!(var"##plate_argument_shadow#6", var"##plate_linear#21", _pr_f21)
                              end
                          end
                  else
                      #= /home/niko/scratch/plate-tape-storage-20261114/src/plate_reverse.jl:1624 =# @inbounds for var"##plate_linear#21" = _pr_cells18
                              _pr_t12 = Base.CartesianIndex(var"##plate_linear#21")
                              var"##plate_index#22" = _pr_t12
                              _pr_t13 = Base.Broadcast._broadcast_getindex(var"##plate_prepared#10", var"##plate_index#22")
                              var"##inbounds_value#23" = _pr_t13
                              var"#v#op5#24" = var"##inbounds_value#23"
                              _pr_t14 = Base.Broadcast._broadcast_getindex(var"##plate_prepared#11", var"##plate_index#22")
                              var"##inbounds_value#25" = _pr_t14
                              var"#y#op5#26" = var"##inbounds_value#25"
                              _pr_t15 = ReactiveKernels._native_mul(var"#s#op5#20", var"#v#op5#24")
                              _pr_t16 = _pr_t15 + var"#y#op5#26"
                              _pr_t17 = Base.sin(_pr_t16)
                              var"##plate___plate_value__#27" = _pr_t17
                              _pr_want_value isa Val{true} && (_pr_value22 = _pr_value22 + _pr_t17)
                              _pr_f19 = Base.cos(_pr_t16)
                              if _pr_want_value isa Val{true} && !(var"##plate_argument_shadow#6" isa Nothing)
                                  ReactiveKernels._pr_shadow_add!(var"##plate_argument_shadow#6", var"##plate_linear#21", _pr_f19)
                              else
                                  _pr_f21 = _pr_coef20 * _pr_f19
                                  ReactiveKernels._pr_shadow_add!(var"##plate_argument_shadow#6", var"##plate_linear#21", _pr_f21)
                              end
                          end
                  end
                  if _pr_want_value isa Val{true} && _pr_ncells18 > 0
                      var"##plate_total#4" = ReactiveKernels._plate_total_add(var"##plate_total#4", 0.0 + 1.0_pr_value22)
                  end
                  if _pr_want_value isa Val{true} && !(var"##plate_argument_shadow#6" isa Nothing)
                      ReactiveKernels._pr_scale_buffer!(var"##plate_argument_shadow#6", _pr_coef20)
                  end
              end
              Core.nothing
          end
          (var"##plate_total#4", var"##plate_total_bar#9")
      end)
))

mutable struct Slot
    tape::Vector{Float64}
    empty::Vector{Float64}
end
Slot() = (empty=Float64[];Slot(empty,empty))
@inline take!(slot,n,::Val{false}) = zeros(n)
@inline function take!(slot,n,::Val{true})
    tape=slot.tape
    length(tape)==n || return zeros(n)
    slot.tape=slot.empty
    fill!(tape,0.0)
end
@inline give!(slot,tape,::Val{true}) = (slot.tape=tape;nothing)
@inline give!(slot,tape,::Val{false}) = nothing
struct Cut{Reuse} end
function (::Cut)(x,data,s,initial,slot)
    total=initial
    for i in eachindex(x)
        total+=sin(s*x[i]+data[i])
    end
    total
end
struct Gathered{F}
    f::F
end
(g::Gathered)(q,idx,data,s,initial,slot) = g.f(Base.SubArray(q,(idx,)),data,s,0.0,slot)
function ER.augmented_primal(config::ER.RevConfig,f::Const{Cut{Reuse}},
        ::Type{Active{Float64}},x::Duplicated,data::Const,s::Const,initial::Const,
        slot::Enzyme.Annotation) where {Reuse}
    n=length(parent(x.val))
    tape=take!(slot.val,n+1,Val(Reuse))
    buffer=Base.SubArray(view(tape,1:n),parentindices(x.val))
    value,partial=CUT(x.val,data.val,axes(x.val),s.val,initial.val,1.0,Val(true),buffer)
    tape[end]=partial
    ER.AugmentedReturn(ER.needs_primal(config) ? value : nothing,nothing,tape)
end
function ER.reverse(config::ER.RevConfig,f::Const{Cut{Reuse}},dy::Active,tape,
        x::Duplicated,data::Const,s::Const,initial::Const,slot::Enzyme.Annotation) where {Reuse}
    for i in eachindex(parent(x.dval))
        parent(x.dval)[i]+=dy.val*tape[i]
    end
    give!(slot.val,tape,Val(Reuse))
    (nothing,nothing,nothing,nothing,nothing)
end
@testset "emitted cut without ReactiveKernels" begin
    for reuse in (false,true)
        @testset "reuse=$reuse" begin
            println("CASE reuse=",reuse);flush(stdout)
            idx=[mod1(i,3) for i in 1:128]
            data=[sin(i/3)/2 for i in 1:128]
            q,direction=[.1,-.4,.3],[.2,-.7,1.1]
            contexts=(DI.Constant(idx),DI.Constant(data),DI.Constant(1.2),DI.Constant(0.0),DI.Cache(Slot()))
            f=Gathered(Cut{reuse}())
            backend=SecondOrder(AutoEnzyme(;mode=Enzyme.Forward),AutoEnzyme(;mode=Enzyme.Reverse))
            prep=DI.prepare_hvp(f,backend,q,(direction,),contexts...)
            for point in (q,[-.8,.6,1.2],q)
                expected=zeros(3)
                for (i,j) in enumerate(idx)
                    expected[j]-=1.2^2*sin(1.2*point[j]+data[i])*direction[j]
                end
                @test only(DI.hvp(f,prep,backend,point,(direction,),contexts...)) ≈ expected rtol=3e-10 atol=3e-10
            end
        end
    end
end
