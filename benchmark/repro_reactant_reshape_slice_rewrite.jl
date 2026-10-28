# Reactant only; no ReactiveKernels code. Two Enzyme-JAX rewrite patterns,
# `reshape_dynamic_slice` and `reshape_dus`, never finish on a reshape that
# inserts a unit dimension ahead of a dropped one. Each pattern is run alone
# on a module of at most four operations.
#
# Both patterns call `transformReshapeSlice<mlir::Value>`, whose fill callback
# creates a `stablehlo.constant` for the inserted dimension. A later check
# then fails (the dropped dimension has a dynamic start, or the operand
# extent is not one) and the pattern reports failure with the constant left
# behind. MLIR's greedy driver erases the dead constant, records a change
# with no counted rewrite, and repeats. `max_num_rewrites` does not bound it.
#
# Recorded on strato2, Julia 1.10.11, 2026-09-30, on Reactant 0.2.284
# (Reactant_jll 0.0.407) and Reactant 0.2.289 (Reactant_jll 0.0.413, Enzyme-JAX
# a86561d8): the control finishes in under a second, each of the two cases
# exceeds a 25-second bound. `repro_reactant_passthrough_loop.jl` reaches the
# first pattern through the default pipeline.
#
# Run one case per process under an execution bound:
#   timeout --kill-after=15s 60 julia --project=<env> \
#       benchmark/repro_reactant_reshape_slice_rewrite.jl control
#   ... reshape_dynamic_slice
#   ... reshape_dus
using Reactant

const CASES = Dict(
    # Dropping the sliced unit dimension alone: the pattern fails before it
    # creates anything, and the pipeline finishes.
    "control" => ("reshape_dynamic_slice(1)", """
        func.func @main(%x: tensor<2x3xf64>, %i: tensor<i32>) -> tensor<2xf64> {
          %c = stablehlo.constant dense<0> : tensor<i32>
          %s = stablehlo.dynamic_slice %x, %c, %i, sizes = [2, 1] : (tensor<2x3xf64>, tensor<i32>, tensor<i32>) -> tensor<2x1xf64>
          %r = stablehlo.reshape %s : (tensor<2x1xf64>) -> tensor<2xf64>
          return %r : tensor<2xf64>
        }
        """),
    "reshape_dynamic_slice" => ("reshape_dynamic_slice(1)", """
        func.func @main(%x: tensor<2x3xf64>, %i: tensor<i32>) -> tensor<1x2xf64> {
          %c = stablehlo.constant dense<0> : tensor<i32>
          %s = stablehlo.dynamic_slice %x, %c, %i, sizes = [2, 1] : (tensor<2x3xf64>, tensor<i32>, tensor<i32>) -> tensor<2x1xf64>
          %r = stablehlo.reshape %s : (tensor<2x1xf64>) -> tensor<1x2xf64>
          return %r : tensor<1x2xf64>
        }
        """),
    "reshape_dus" => ("reshape_dus<1>", """
        func.func @main(%x: tensor<2x1xf64>, %u: tensor<1x1xf64>, %i: tensor<i32>, %j: tensor<i32>) -> tensor<1x2xf64> {
          %d = stablehlo.dynamic_update_slice %x, %u, %i, %j : (tensor<2x1xf64>, tensor<1x1xf64>, tensor<i32>, tensor<i32>) -> tensor<2x1xf64>
          %r = stablehlo.reshape %d : (tensor<2x1xf64>) -> tensor<1x2xf64>
          return %r : tensor<1x2xf64>
        }
        """),
)

case = only(ARGS)
pattern, source = CASES[case]
pipeline = "enzyme-hlo-generate-td{patterns=$pattern},transform-interpreter," *
           "enzyme-hlo-remove-transform"
println("RESHAPE_SLICE_REWRITE_BEGIN case=", case, " pattern=", pattern,
        " julia=", VERSION, " Reactant=", pkgversion(Reactant))
flush(stdout)
result = Reactant.Compiler.run_pass_pipeline_on_source(source, pipeline)
println("RESHAPE_SLICE_REWRITE_DONE case=", case)
println(result)
