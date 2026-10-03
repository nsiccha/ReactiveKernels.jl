# Backend-only reproducer of the existing chained-slice optimizer failure.
# Run `default` and `only_enzyme` in separate processes. The latter preserves
# the trace and reverse pass while omitting the failing optimization pipeline.
using Reactant, Enzyme
const _GATHER_OFFSETS=[0.,1.,2.,3.,4.,5.]
const _GATHER_MAP=[2,4,6]
function loss(u)
    s=sum(view(u,1:1))
    r=exp.(s .+ _GATHER_OFFSETS)[_GATHER_MAP]
    a,b,c=sum(view(r,1:1)),sum(view(r,2:2)),sum(view(r,3:3))
    m=min(a,b)
    m+min(m,c)
end
gradient(u)=only(Enzyme.gradient(Enzyme.Reverse,loss,u))
u=[2.];ru=Reactant.to_rarray(u)
expected=[2exp(3.)]
mode=Symbol(get(ARGS,1,"default"))
compiled=mode===:default ? Reactant.compile(gradient,(ru,)) :
    Reactant.compile(gradient,(ru,);optimize=mode)
got=Array(compiled(ru))
println("expected=",expected," compiled=",got)
@assert got≈expected
