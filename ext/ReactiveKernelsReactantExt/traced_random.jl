# Fixed integer-range draws use rejection before reduction, preserving a
# uniform distribution without a widened traced integer or a trajectory tape.
struct SlotRangeSampler{T,Full}
    first_bits::UInt64
    span::UInt64
    threshold::UInt64
end
function SlotRangeSampler(range::UnitRange{T}) where {T}
    RK._kernel_dom_int_scalar(T) && sizeof(T)<=8 ||
        error("traced slots: integer-range draws require builtin integers up to 64 bits")
    isempty(range) && error("traced slots: cannot draw from an empty integer range")
    span=UInt128(Int128(last(range))-Int128(first(range)))+one(UInt128)
    full=span==(one(UInt128)<<64)
    SlotRangeSampler{T,full}(first(range)%UInt64,
        full ? zero(UInt64) : UInt64(span),
        full ? zero(UInt64) : UInt64((one(UInt128)<<64)%span))
end
Reactant.make_tracer(seen,previous::SlotRangeSampler,path,mode;kwargs...)=previous
Reactant.traced_type_inner(::Type{T},seen,mode::Reactant.TraceMode,
    track_numbers::Type,ndevices,runtime) where {T<:SlotRangeSampler}=T

slot_range_integer(::Type{T},value) where {T}=value%T
slot_range_integer(::Type{T},value::Reactant.TracedRNumber) where {T}=
    Reactant.promote_to(Reactant.TracedRNumber{T},value)

function slot_rand_range(rng,sampler::SlotRangeSampler{T,false}) where {T}
    bits=Random.rand(rng,UInt64)
    Reactant.@trace while bits<sampler.threshold
        bits=Random.rand(rng,UInt64)
    end
    slot_range_integer(T,bits%sampler.span+sampler.first_bits)
end
function slot_rand_range(rng,sampler::SlotRangeSampler{T,true}) where {T}
    slot_range_integer(T,Random.rand(rng,UInt64)+sampler.first_bits)
end
