# Array operand snapshots do not by themselves preserve a producer callable's
# reads of mutable global state. This is a bounded semantic diagnostic.
using ReactiveKernels, MutatingFunctions, TOML, SHA
include("prototypes.jl")
const GAIN=Ref(1.0)
scaled_exp(x)=GAIN[]*exp(x)
struct ChangingGain <: AbstractVector{Float64}
    data::Vector{Float64}
end
Base.size(x::ChangingGain)=size(x.data)
function Base.getindex(x::ChangingGain,i::Int)
    GAIN[]=2.0
    x.data[i]
end
@kernel producer_state(u::Vector{Float64},peer::ChangingGain) = begin
    w=scaled_exp.(u)
    squared=w.^2
    terms=plate(peer,w) do yi,wi
        yi+log(wi)
    end
    total=sum(squared)+sum(terms)
    return total
end
x=[0.1,0.2,0.3]; peer=ChangingGain([0.3,0.4,0.5])
p=plan(producer_state)
expected=sum(exp.(2 .* x))+sum(peer.data)+sum(x)
rows=Any[]
for mode in (:control,:early,:early_snapshot,:late,:late_snapshot)
    k=if mode===:control
        prepare_nonallocating(prepare(p;bound_statistics=false),x,peer)
    elseif mode in (:early,:early_snapshot)
        StorageAlternatives.early(p,x,peer;snapshot=mode===:early_snapshot)
    else
        StorageAlternatives.late(p,x,peer;snapshot=mode===:late_snapshot)
    end
    GAIN[]=1.0
    value=only(k(x,peer))
    push!(rows,Dict("mode"=>string(mode),"expected"=>expected,"value"=>value,
        "preserved"=>isapprox(value,expected;rtol=1e-12,atol=1e-12)))
end
out=abspath(only(ARGS));mkpath(dirname(out))
open(out,"w") do io
    TOML.print(io,Dict("checks"=>rows,
        "prototype_sha256"=>bytes2hex(sha256(read(joinpath(@__DIR__,"prototypes.jl")))));sorted=true)
end
println("CALLABLE_EFFECTS_COMPLETE ",rows)
