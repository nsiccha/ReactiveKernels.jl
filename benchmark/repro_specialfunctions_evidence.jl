# Numerical backend only: no ReactiveKernels code. Run each case in a fresh
# process, e.g. julia --project=<env> this_file.jl beta primal.
using Reactant, Enzyme, SpecialFunctions
family=Symbol(get(ARGS,1,"gamma")); mode=Symbol(get(ARGS,2,"primal"))
function primitive(v)
    z=sum(v)
    family===:gamma && return first(gamma_inc(2.,z))
    family===:beta && return first(beta_inc(2.,3.,z))
    family===:bessel && return besselix(1,z)
    family===:erfcx && return erfcx(z)
    error("unknown primitive")
end
v=Reactant.to_rarray([.4])
if mode===:primal
    compiled=Reactant.@compile primitive(v)
    println("compiled=",compiled(v))
else
    gradient(v)=only(Enzyme.gradient(Enzyme.Reverse,primitive,v))
    compiled=Reactant.@compile gradient(v)
    println("compiled_gradient=",Array(compiled(v)))
end
