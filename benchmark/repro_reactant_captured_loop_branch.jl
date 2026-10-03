# Backend-only read-only tracer ownership reproducer. A retained loop updates
# its traced input handles, so handles captured by a surrounding lazy thunk
# must be copied on entry. Run `borrowed` and `copied` in separate processes.
using Reactant, Enzyme
const COPY_INPUTS=get(ARGS,1,"borrowed")=="copied"
function mass(a,b,k)
    if COPY_INPUTS
        a,b=copy(a),copy(b)
    end
    total=zero(a+b);i=0
    Reactant.@trace checkpointing=Reactant.Binomial(4) while i<k
        total+=a+b
        i+=1
    end
    total
end
function pick(pred,yes,no)
    Reactant.@trace if pred
        result=yes()
    else
        result=no()
    end
    result
end
cell(a,b)=pick(a>b,()->mass(a,b,2),()->mass(a,b,3))
function loss(v)
    a,b=sum(view(v,1:1)),sum(view(v,2:2))
    cell(a,b)+a+b
end
gradient(v)=only(Enzyme.gradient(Enzyme.Reverse,loss,v))
v=[.3,.2];rv=Reactant.to_rarray(v)
hlo=repr(Reactant.@code_hlo optimize=false loss(rv))
@assert occursin("stablehlo.while",hlo)
c=Reactant.@compile loss(rv)
@assert Float64(c(rv))≈loss(v)
cg=Reactant.@compile gradient(rv)
@assert Array(cg(rv))≈only(Enzyme.gradient(Enzyme.Reverse,loss,v))
println("compiled primal and reverse match native")
