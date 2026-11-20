"""
Compare RKPPL's constructor reuse with ordinary Julia lowering on the exact
same emissions. Pass a series of fresh definitions from
`kernel_expr(plan, assign_layout(plan))` to `KernelConstructionReplay.compare`.
Each arm builds a fresh graph. The rows report wall seconds in alternating
order. The first pair includes compilation.
"""
module KernelConstructionReplay

import ReactiveKernels as RK
import ReactiveKernelsPPL as PPL

function compare(definitions)
    mod = PPL.PPLGeneratedModels
    cache = RK._KernelConstructors(mod)
    map(enumerate(definitions)) do (i, definition)
        if isodd(i)
            ordinary = @elapsed reference = RK._kernel_eval_definition(mod, definition)
            reused = @elapsed candidate = RK._kernel_eval_definition(cache, definition)
        else
            reused = @elapsed candidate = RK._kernel_eval_definition(cache, definition)
            ordinary = @elapsed reference = RK._kernel_eval_definition(mod, definition)
        end
        @assert reference.graph !== candidate.graph
        @assert reference.port_order == candidate.port_order
        (; ordinary, reused, recipes = length(candidate.graph.recipes),
           cached_shapes = length(cache.thunks))
    end
end

end
