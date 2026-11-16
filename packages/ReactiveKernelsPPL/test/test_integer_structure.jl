module IntegerStructureChecks
using ReactiveKernels, ReactiveKernelsPPL, Distributions, Enzyme, Test
using DifferentiationInterface: AutoEnzyme

function inventory(ex)
    ex isa Expr || return (nodes=1, loops=0, functions=0)
    children = map(inventory, ex.args)
    return (nodes=1+sum(c->c.nodes,children;init=0),
        loops=Int(ex.head in (:for,:while))+sum(c->c.loops,children;init=0),
        functions=Int(ex.head in (:function,:(->)))+sum(c->c.functions,children;init=0))
end
# A native plate is an executable operand in the outer expression. Its
# printer already shows this exact AST; inspect the reachable body too.
inventory(program::ReactiveKernels._PlateProgram) = inventory(program.ast)

@testset "integer enumeration retains runtime bodies" begin
    model = marginalize(@rkppl(begin
        p ~ Beta(2,3)
        k ~ Binomial(n,p)
        y .~ Normal.(k*p,1)
    end), :k)
    plan = model(;n=2) | (;y=[0.2,0.7])
    built = build_kernel(plan)
    u=[0.3]; p=1/(1+exp(-only(u)))
    inventories = NamedTuple[]
    for (n,ny) in ((0,0),(2,2),(20,7),(50,40))
        y=collect(range(0.2,1.7;length=ny))
        rebound = model(;n) | (;y)
        query = prepare_query(built,rebound,:sampler)
        sampler = prepare_sampler(built,rebound,u;
            backend=AutoEnzyme(;mode=Enzyme.Reverse),nonallocating=false)
        scores=[logpdf(Binomial(n,p),k)+sum(logpdf.(Normal(k*p,1),y)) for k in 0:n]
        mass=ReactiveKernelsPPL.logsumexp(scores)
        weights=exp.(scores.-mass)
        reference=logpdf(Beta(2,3),p)+log(p)+log1p(-p)+mass
        derivative=2-5p+sum(weights[k+1]*(k-n*p+sum(y.-k*p)*k*p*(1-p)) for k in 0:n)
        value,gradient=sampler_value_and_gradient!(sampler,zeros(1),u)
        @test query(u) ≈ reference
        @test value ≈ reference
        @test only(gradient) ≈ derivative atol=1e-10
        code=code_expr(query)
        metrics=inventory(code)
        push!(inventories,(;n,ny,metrics...))
    end
    println("INTEGER NATIVE BODY INVENTORY ",inventories)
    @test all(i->i.loops>0,inventories[2:end])
    @test inventories[end].nodes < 2inventories[2].nodes
    @test inventories[end].functions <= inventories[2].functions + 2
end
end
