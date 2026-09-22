# Standalone CPU reproducer: only Reactant and Enzyme are required.
using Reactant, Enzyme

function conditional_add(total, q, i)
    Reactant.@trace if i > 1
        a = Reactant.@allowscalar q[i]
        result = total + a
    else
        result = total
    end
    result
end

function conditional_sum(q)
    total = zero(sum(q))
    Reactant.@trace for i in 1:length(q)
        total = conditional_add(total, q, i)
    end
    total
end

q = [1., 2., 3., 4., 5., 6.]
rq = Reactant.to_rarray(q)
primal = Reactant.@compile sync=true conditional_sum(rq)
@assert conditional_sum(q) == Float64(primal(rq)) == 20.0
gradient(q) = only(Enzyme.gradient(Enzyme.Reverse, conditional_sum, q))
compiled_gradient = Reactant.@compile sync=true gradient(rq)
@assert Array(compiled_gradient(rq)) == [0., 1., 1., 1., 1., 1.]
