length(ARGS) == 3 || error("usage: joint_preparation_costs.jl definitions baseline|joint family")
include(abspath(ARGS[1]))

function measure_first_stages(rule::R, x::X, variant, family, value_op::F) where {R,X,F}
    (y,tape), first_forward = timed_phase(() -> Base.invokelatest(stage_primal,rule,Val(1),x))
    (dx,), first_reverse = timed_phase(() -> Base.invokelatest(stage_reverse,rule,Val(1),tape,1.0))
    println((;stage=:first_forward,variant,family,first_forward...))
    println((;stage=:first_reverse,variant,family,first_reverse...))
    @test y ≈ (family == :scalar_exp ? exp(x) : value_op(x))
    @test dx ≈ (family == :array_square ? 2 .* x : exp.(x))
    @test stage_residuals(rule,Val(1)) == (variant == :joint ? (:partial,) : (:x,))
    first_forward, first_reverse
end
function preparation_graph(family)
    value_op, joint_op, direct_op = family == :array_square ?
        (square_value,square_joint,square_direct) : (exp_value,exp_joint,exp_direct)
    family == :scalar_exp ? scalar_graph(5) : array_graph(value_op,joint_op,direct_op,5)
end
function measure_preparation(variant, family)
    println((;stage=:environment,variant,family,julia=VERSION,rk=pkgversion(ReactiveKernels),enzyme=pkgversion(Enzyme)))
    flush(stdout)
    value_op = family == :array_square ? square_value : exp_value
    spec, graph_cost = timed_phase(() -> Base.invokelatest(preparation_graph,family))
    println((;stage=:first_graph,variant,family,graph_cost...))
    rule, prep = timed_phase(() -> Base.invokelatest(make_rule,spec))
    println((;stage=:first_rule,variant,family,prep...))
    x = family == :scalar_exp ? 0.3 : [0.7sin(i) for i in 1:1000]
    first_forward, first_reverse = measure_first_stages(rule,x,variant,family,value_op)
    phases = (graph_cost,prep,first_forward,first_reverse)
    println((;stage=:measured_total,variant,family,seconds=sum(p.seconds for p in phases),
        compile_seconds=sum(p.compile_seconds for p in phases),
        bytes=sum(p.bytes for p in phases)))
    for round in 1:5
        trial=@benchmark make_rule($spec) samples=20 evals=1 seconds=1.0
        mid=median(trial)
        println((;stage=:warm_rule,variant,family,round,ns=mid.time,bytes=mid.memory,allocations=mid.allocs))
    end
    println("JOINT_PREPARATION_COMPLETE")
end
measure_preparation(Symbol(ARGS[2]), Symbol(ARGS[3]))
