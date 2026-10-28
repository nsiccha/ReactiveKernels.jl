# Backend-only count loop inside a batched cell. On Reactant 0.2.290,
# reverse with `only_enzyme` segfaults inside Enzyme's LoopCheckpointing.
using Reactant,Enzyme
function count_cell(a0,b0)
    a=copy(Reactant.@allowscalar a0[])
    b=copy(Reactant.@allowscalar b0[])
    total=zero(a+b);i=0
    Reactant.@trace checkpointing=Reactant.Binomial(4) while i<3
        total+=exp(i*a-b)
        i+=1
    end
    total
end
function loss(u)
    a=sum(view(u,1:1));b=sum(view(u,2:2))
    av=exp.(a .+ zeros(2));bv=b .+ zeros(2)
    values=only(Reactant.Ops.batch(count_cell,[av,bv],Int64[2]))
    sum(log.(values))
end
gradient(u)=only(Enzyme.gradient(Enzyme.Reverse,loss,u))
u=[.2,.3];ru=Reactant.to_rarray(u)
mode=Symbol(get(ARGS,1,"only_enzyme"))
compiled=mode===:default ? Reactant.compile(gradient,(ru,)) :
    Reactant.compile(gradient,(ru,);optimize=mode)
got=Array(compiled(ru))
weights=exp.((0:2).*exp(u[1]).-u[2])
expected=[2exp(u[1])*sum((0:2).*weights)/sum(weights),-2.]
println("expected=",expected," compiled=",got)
@assert got≈expected
