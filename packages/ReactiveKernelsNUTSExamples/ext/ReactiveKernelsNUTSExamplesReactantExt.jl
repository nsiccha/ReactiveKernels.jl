module ReactiveKernelsNUTSExamplesReactantExt

using ReactiveKernelsNUTSExamples
import Reactant

# Exact normalized CFG of the accepted authored step!/finish!/start! fixture.
# Method ids are deliberately absent: the source validator substitutes the ids
# captured in this Julia session, then the generated transition freezes them as
# literals. Any edge/terminator/callee drift rejects before Reactant tracing.
const _NUTS_REACTANT_CFG = (
    (:step,1,:goto,0,0,:none,0,0), (:step,2,:branch,19,1,:none,0,0),
    (:step,3,:goto,2,0,:none,0,0), (:step,4,:goto,1,0,:none,0,0),
    (:step,5,:branch,3,4,:none,0,0), (:step,6,:goto,5,0,:none,0,0),
    (:step,7,:branch,6,5,:none,0,0), (:step,8,:goto,1,0,:none,0,0),
    (:step,9,:branch,7,8,:none,0,0), (:step,10,:call,0,0,:finish,15,9),
    (:step,11,:call,0,0,:finish,15,9), (:step,12,:branch,10,11,:none,0,0),
    (:step,13,:goto,12,0,:none,0,0), (:step,14,:goto,12,0,:none,0,0),
    (:step,15,:branch,13,14,:none,0,0), (:step,16,:goto,15,0,:none,0,0),
    (:step,17,:branch,16,12,:none,0,0), (:step,18,:branch,17,12,:none,0,0),
    (:step,19,:goto,18,0,:none,0,0), (:step,20,:goto,2,0,:none,0,0),
    (:step,21,:goto,20,0,:none,0,0), (:step,22,:goto,21,0,:none,0,0),
    (:step,23,:goto,21,0,:none,0,0), (:step,24,:branch,22,23,:none,0,0),
    (:step,25,:goto,24,0,:none,0,0), (:step,26,:goto,25,0,:none,0,0),
    (:finish,1,:goto,0,0,:none,0,0), (:finish,2,:goto,1,0,:none,0,0),
    (:finish,3,:goto,2,0,:none,0,0), (:finish,4,:goto,0,0,:none,0,0),
    (:finish,5,:goto,4,0,:none,0,0), (:finish,6,:goto,5,0,:none,0,0),
    (:finish,7,:branch,3,6,:none,0,0), (:finish,8,:goto,7,0,:none,0,0),
    (:finish,9,:goto,0,0,:none,0,0), (:finish,10,:branch,8,9,:none,0,0),
    (:finish,11,:call,0,0,:start,15,10), (:finish,12,:goto,11,0,:none,0,0),
    (:finish,13,:goto,11,0,:none,0,0), (:finish,14,:branch,12,13,:none,0,0),
    (:finish,15,:goto,14,0,:none,0,0),
    (:start,1,:goto,0,0,:none,0,0), (:start,2,:goto,0,0,:none,0,0),
    (:start,3,:branch,2,1,:none,0,0), (:start,4,:goto,3,0,:none,0,0),
    (:start,5,:branch,3,4,:none,0,0), (:start,6,:goto,5,0,:none,0,0),
    (:start,7,:goto,0,0,:none,0,0), (:start,8,:branch,7,0,:none,0,0),
    (:start,9,:branch,8,0,:none,0,0), (:start,10,:call,0,0,:finish,15,9),
    (:start,11,:goto,10,0,:none,0,0), (:start,12,:goto,0,0,:none,0,0),
    (:start,13,:branch,11,12,:none,0,0), (:start,14,:call,0,0,:start,15,13),
    (:start,15,:branch,6,14,:none,0,0),
)

function _normalized_nuts_cfg(plan)
    kinds = Dict(b.mid => b.kind for b in plan.blocks)
    Tuple((b.kind, b.pc, b.tt, b.ta, b.tb,
           b.cm == 0 ? :none : kinds[b.cm], b.ce, b.resume) for b in plan.blocks)
end

# Build the one traced control loop only after the example-owned NUTS compiler
# has validated and frozen its captured CFG. Each emitted arm carries literal
# method/pc/terminator metadata; the loop itself carries tensors only.
function compile_nuts_reactant_transition(plan, cfg)
    _normalized_nuts_cfg(plan) == _NUTS_REACTANT_CFG || throw(ArgumentError(
        "compile_nuts_reactant rejects control-flow drift in the authored NUTS fixture"))
    cfgname = gensym(:nuts_reactant_cfg)
    fname = gensym(:nuts_reactant_transition)
    Core.eval(@__MODULE__, :(const $cfgname = $cfg))
    arms = Expr(:block, map(plan.blocks) do b
        :(st = ReactiveKernelsNUTSExamples._nr_block(st,
            Val($(b.mid)), Val($(QuoteNode(b.kind))), Val($(b.pc)), Val($(QuoteNode(b.tt))),
            Val($(b.ta)), Val($(b.tb)), Val($(b.cm)), Val($(b.ce)), Val($(b.resume)),
            Val($(b.mp)), Val($(b.mpc)), m, p, e, d, cs, ONE, $cfgname))
    end...)
    definition = quote
        function $fname(st)
            st = ReactiveKernelsNUTSExamples._nr_refresh(st, $cfgname)
            ONE = st.dep[1:1] .* 0 .+ 1
            Reactant.@trace while (sum(st.csp) >= 1) & (sum(st._step) < sum(st.stepcap))
                st = merge(st, (_step = st._step .+ 1,))
                cs = st.csp
                m = st.mids[cs]; p = st.pcs[cs]; e = st.ep[cs]; d = st.dep[cs]
                $arms
            end
            ReactiveKernelsNUTSExamples._nr_finish(st)
        end
    end
    Core.eval(@__MODULE__, definition)
    Base.invokelatest(() -> Core.getglobal(@__MODULE__, fname))
end

compile_nuts_reactant_executable(transition, state; sync::Bool=false) =
    Base.invokelatest(
        Reactant.compile, transition, (state,); serializable=true, sync)

end # module ReactiveKernelsNUTSExamplesReactantExt
