module DerivativeRuleArrayPrimalTests
using ReactiveKernels, Test, LinearAlgebra
import Enzyme

@kernel spec(A::Matrix{Float64}, x::Vector{Float64}, seed::Vector{Float64}) = begin
    y::Vector{Float64} = A*x
    dx::Vector{Float64} = transpose(A)*seed
    return y, dx
end
const rule = derivative_rule(spec; primal=:y, covector=:seed,
    cotangents=(A=nothing, x=:dx))
native(A, x) = sum(A*x; init=0.0)
loss(A, x) = sum(rule(A, x); init=0.0)
function overwrite_matrix(A, x)
    result = loss(A, x)
    fill!(A, 0.0)
    result
end
function overwrite_result(A, x)
    y = rule(A, x)
    result = sum(y; init=0.0)
    fill!(y, 0.0)
    result
end
function gradient!(dx, f::F, A, x, ::Val{P}) where {F,P}
    fill!(dx, 0.0)
    mode = P ? Enzyme.ReverseWithPrimal : Enzyme.Reverse
    Enzyme.autodiff(mode, Enzyme.Const(f), Enzyme.Active,
        Enzyme.Const(A), Enzyme.Duplicated(x, dx))
end

@testset "array primal retains readonly residuals safely" begin
    @test ReactiveKernels.stage_residuals(rule, Val(2)) == (:A,)
    ext = Base.get_extension(ReactiveKernels, :ReactiveKernelsEnzymeExt)
    for n in (0, 7, 128), p in (false, true)
        A = [sin(0.2i + 0.3j) for i in 1:n, j in 1:3]
        original = copy(A)
        for x in ([0.1, -0.3, 0.4], [-0.4, 0.2, 0.8])
            original_x = copy(x)
            reference = zeros(3)
            reference_value = gradient!(reference, native, A, x, Val(p))
            @test reference ≈ vec(sum(original; dims=1)) atol=1e-10 rtol=1e-10
            primal, residuals = ReactiveKernels.stage_primal(rule, Val(2), A, x)
            retained = ext._retain_residuals((1,), residuals, primal,
                (false, false, false))
            @test retained[1] === A
            for f in (loss, overwrite_matrix, overwrite_result)
                A .= original
                dx = zeros(3)
                actual = gradient!(dx, f, A, x, Val(p))
                @test dx ≈ reference atol=1e-10 rtol=1e-10
                p && @test last(actual) ≈ last(reference_value) atol=1e-10 rtol=1e-10
                @test x == original_x
                @test f === overwrite_matrix ? all(iszero, A) : A == original
            end
            A .= original
        end
    end
end
end
