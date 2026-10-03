# Backend-only Enzyme-JAX optimizer regression; no RK/PPL or model imports.
# A non-identity pad is split into multiplies and a concatenation; pushing
# the multiplies back across that concatenation and folding its constant
# prefix into a pad recreates the trigger, adding slices on every round.
# The downstream slices cover the whole padded dimension, so the split has
# no unread region to eliminate. EnzymeAD/Enzyme-JAX PR #3330 adds that guard.
#
# Reactant 0.2.289 / Reactant_jll 0.0.413+0, artifact 9dd6114d: `cycle`
# exceeds a 30-second bound; removing any one of the four patterns finishes.
# A standalone compiler containing merged #3330 finishes the same input.
# This is an optimizer-stage test, not proof of a released Reactant runtime.
#
# Run each case in its own bounded process:
#   timeout --kill-after=5s 30 julia --project=<env> \
#       benchmark/repro_reactant_pad_concat_rewrite.jl cycle
#   ... omit=1              # likewise omit=2, omit=3, omit=4
#   ... source              # print a module for standalone compiler testing
using Reactant

const SOURCE = """
module {
  func.func @main(%a: tensor<1xf64>, %x: tensor<3xf64>, %y: tensor<3xf64>) -> tensor<3xf64> {
    %c = stablehlo.constant dense<-0.5> : tensor<f64>
    %p = stablehlo.pad %a, %c, low = [2], high = [0], interior = [0] : (tensor<1xf64>, tensor<f64>) -> tensor<3xf64>
    %m = stablehlo.multiply %p, %x : tensor<3xf64>
    %v = stablehlo.add %m, %y : tensor<3xf64>
    %s0 = stablehlo.slice %v [0:1] : (tensor<3xf64>) -> tensor<1xf64>
    %s1 = stablehlo.slice %v [1:3] : (tensor<3xf64>) -> tensor<2xf64>
    %r = stablehlo.concatenate %s0, %s1, dim = 0 : (tensor<1xf64>, tensor<2xf64>) -> tensor<3xf64>
    return %r : tensor<3xf64>
  }
}
"""
const PATTERNS = [
    "broadcast_in_dim_simplify<16>(1024)",
    "concat_push_binop_mul<1>",
    "binop_pad_to_concat_mul<1>",
    "concat_to_pad<1>",
]

case = isempty(ARGS) ? "cycle" : only(ARGS)
if case == "source"
    print(SOURCE)
else
    patterns = copy(PATTERNS)
    if startswith(case, "omit=")
        deleteat!(patterns, parse(Int, split(case, '='; limit=2)[2]))
    else
        case == "cycle" || error("expected cycle, source, or omit=1 through omit=4")
    end
    pipeline = "enzyme-hlo-generate-td{patterns=$(join(patterns, ';'))}," *
               "transform-interpreter,enzyme-hlo-remove-transform"
    println("PAD_CONCAT_REWRITE_BEGIN case=", case, " julia=", VERSION,
            " Reactant=", pkgversion(Reactant))
    flush(stdout)
    result = Reactant.Compiler.run_pass_pipeline_on_source(SOURCE, pipeline)
    println("PAD_CONCAT_REWRITE_DONE case=", case)
    println(result)
end
