module JAXTranslationExample

using ReactiveKernels
using LinearAlgebra
const RK = ReactiveKernels

# A package owns these Julia functions; its RK extension owns the mappings.
function triple(x)
    result = zero(x)
    for _ in 1:3
        result += x
    end
    result
end

RK._jax_elementwise(::typeof(triple)) = "_example_triple" => """
def _example_triple(x):
    return 3 * x
"""

# An ordinary call on an array has its own translation. Here its mathematics
# happens to be elementwise too, so preserve the input's padded axes as well.
function RK._jax_function_call(ctx::RK._JaxContext, ::typeof(triple),
                               args::Vector{RK._PyVal})
    RK._jax_elementwise_result(triple, args, "3 * (" * only(args).code * ")")
end

matrix_exponential(A::AbstractMatrix) = exp(A)
function RK._jax_function_call(ctx::RK._JaxContext, ::typeof(matrix_exponential),
                               args::Vector{RK._PyVal})
    RK._jax_result(matrix_exponential, args,
                   "jax.scipy.linalg.expm(" * only(args).code * ")")
end

# Types supplied to jax_source select scalar, vector, or matrix inputs.
@kernel mapped(x) = begin
    y = triple(x)
    total = sum(y)
    return y, total
end

# The matrix result must keep its rank for the following matrix-vector product.
@kernel mapped_product(A::Matrix{Float64}, v::Vector{Float64}) = begin
    B = triple(A)
    total = sum(B * v)
    return total
end

@kernel prepared(scale::Float64, x) = begin
    y = triple(x)
    total = scale * sum(y)
    return total
end

@kernel exponential(scale::Float64, A::Matrix{Float64}) = begin
    E = matrix_exponential(A)
    total = scale * sum(E)
    return total
end

end # module
