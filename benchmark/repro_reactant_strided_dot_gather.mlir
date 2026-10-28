// Compiler-only companion: run each split module with slice_dot_general.
// The input modules verify; stock Enzyme-JAX floors nondivisible extents.
// These are proposed upstream regression inputs, not a passing-fix receipt.
// RUN: enzymexlamlir-opt --split-input-file --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=slice_dot_general<1>},transform-interpreter,enzyme-hlo-remove-transform)" %s

module {
  func.func @lhs_even(%a: tensor<5x2xf64>, %u: tensor<2xf64>) -> tensor<2xf64> {
    %d = stablehlo.dot_general %a, %u, contracting_dims = [1] x [0] : (tensor<5x2xf64>, tensor<2xf64>) -> tensor<5xf64>
    %s = stablehlo.slice %d [1:4:2] : (tensor<5xf64>) -> tensor<2xf64>
    return %s : tensor<2xf64>
  }
}
// -----
module {
  func.func @lhs_odd(%a: tensor<5x2xf64>, %u: tensor<2xf64>) -> tensor<3xf64> {
    %d = stablehlo.dot_general %a, %u, contracting_dims = [1] x [0] : (tensor<5x2xf64>, tensor<2xf64>) -> tensor<5xf64>
    %s = stablehlo.slice %d [0:5:2] : (tensor<5xf64>) -> tensor<3xf64>
    return %s : tensor<3xf64>
  }
}
// -----
module {
  func.func @rhs_free(%a: tensor<2xf64>, %b: tensor<2x5xf64>) -> tensor<2xf64> {
    %d = stablehlo.dot_general %a, %b, contracting_dims = [0] x [0] : (tensor<2xf64>, tensor<2x5xf64>) -> tensor<5xf64>
    %s = stablehlo.slice %d [1:4:2] : (tensor<5xf64>) -> tensor<2xf64>
    return %s : tensor<2xf64>
  }
}
// -----
module {
  func.func @batch(%a: tensor<5x2xf64>, %b: tensor<5x2xf64>) -> tensor<3xf64> {
    %d = stablehlo.dot_general %a, %b, batching_dims = [0] x [0], contracting_dims = [1] x [1] : (tensor<5x2xf64>, tensor<5x2xf64>) -> tensor<5xf64>
    %s = stablehlo.slice %d [0:5:2] : (tensor<5xf64>) -> tensor<3xf64>
    return %s : tensor<3xf64>
  }
}
// -----
module {
  func.func @unit_stride(%a: tensor<5x2xf64>, %u: tensor<2xf64>) -> tensor<3xf64> {
    %d = stablehlo.dot_general %a, %u, contracting_dims = [1] x [0] : (tensor<5x2xf64>, tensor<2xf64>) -> tensor<5xf64>
    %s = stablehlo.slice %d [1:4] : (tensor<5xf64>) -> tensor<3xf64>
    return %s : tensor<3xf64>
  }
}
// -----
module {
  func.func @empty(%a: tensor<5x2xf64>, %u: tensor<2xf64>) -> tensor<0xf64> {
    %d = stablehlo.dot_general %a, %u, contracting_dims = [1] x [0] : (tensor<5x2xf64>, tensor<2xf64>) -> tensor<5xf64>
    %s = stablehlo.slice %d [1:1:2] : (tensor<5xf64>) -> tensor<0xf64>
    return %s : tensor<0xf64>
  }
}
