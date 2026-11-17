# A prepared non-allocating gradient is a function of its point only: a call
# whose reverse pass meets a non-finite adjoint must not change the gradients
# of later calls. The gradient reuses one shadow per cache slot across calls
# (DifferentiationInterface's `Cache` context), so a matrix product's buffer
# needs Enzyme's reverse BLAS rules to clear the shadow of the output they
# overwrite. The Enzyme dogfood line has that repair
# (`blas_overwritten_output`); registered Enzyme keeps the `NaN` in the
# shadow (Enzyme snag `reverse-blas-rul-f21de298`, ReactiveKernels snag
# `rkppl-samplerque-d71bddde`). Runs through
# `test/run_nonallocating_integration.jl`.
using ReactiveKernels
using MutatingFunctions
using DifferentiationInterface: AutoEnzyme
import Enzyme
using Test

const NF_BACKEND = AutoEnzyme(mode = Enzyme.Reverse)
const NF_BLAS_REPAIRED = isdefined(Enzyme, :has_dogfood_repair) &&
    Enzyme.has_dogfood_repair(:blas_overwritten_output)

# `exp` overflows to `Inf` past 709.78, so `r / (1 + r)` is `Inf / Inf`: the
# value is `NaN`, and the reverse pass seeds a `NaN` adjoint into the product.
nf_vector_spec = @kernel nf_vector_target(u, X, y) = begin
    eta = X * u
    r = exp.(eta)
    p = r ./ (1 .+ r)
    lp = -sum(abs2, y .- p)
    return lp
end
nf_matrix_spec = @kernel nf_matrix_target(B, X, Y) = begin
    eta = X * B
    r = exp.(eta)
    p = r ./ (1 .+ r)
    lp = -sum(abs2, Y .- p)
    return lp
end

const NF_X = [1.0 0.5; 0.2 1.0; -0.3 0.4]

# One prepared gradient at `good`, then `bad`, then `good` again.
function nf_gradients(spec, active, bound, good, bad; nonallocating::Bool)
    ordinary = prepare(spec; have = (active, keys(bound)...), want = :lp, bound)
    # As `prepare_sampler` does: the exemplar types the cache slots.
    kernel = nonallocating ? prepare_nonallocating(ordinary, good) : ordinary
    ad = prepare_ad(kernel, NF_BACKEND, good; active)
    g = similar(good)
    map((good, bad, good)) do point
        value, returned = ad_value_and_gradient!(ad, g, point)
        @test returned === g
        (value, copy(g))
    end
end

same_bits(a, b) = a[1] === b[1] && all(map(===, a[2], b[2]))

@testset "a non-finite adjoint does not reach later non-allocating gradients" begin
    # gemv: registered Enzyme keeps the `NaN` with OpenBLAS 0.3.29 and later
    # and not with 0.3.23, so without the repair the outcome depends on the
    # Julia build and is not asserted. gemm keeps it with every build.
    cases = (("matrix-vector product", nf_vector_spec, :u,
              (; X = NF_X, y = [0.2, 0.7, 0.5]),
              [0.3, -0.2], [900.0, 0.0], false),
             ("matrix-matrix product", nf_matrix_spec, :B,
              (; X = NF_X, Y = [0.2 0.6; 0.7 0.1; 0.5 0.4]),
              [0.3 0.1; -0.2 0.4], [900.0 0.0; 0.0 0.0], true))
    for (label, spec, active, bound, good, bad, every_build) in cases
        @testset "$label" begin
            initial, nonfinite, again = nf_gradients(spec, active, bound, good, bad;
                                                     nonallocating = true)
            # The middle point does seed a non-finite adjoint.
            @test isnan(nonfinite[1])
            @test !all(isfinite, nonfinite[2])
            @test isfinite(initial[1]) && all(isfinite, initial[2])
            if NF_BLAS_REPAIRED
                @test same_bits(again, initial)
            elseif every_build
                @test_broken same_bits(again, initial)
            end
            # The ordinary kernel writes a fresh product on every call.
            ordinary = nf_gradients(spec, active, bound, good, bad;
                                    nonallocating = false)
            @test same_bits(ordinary[3], ordinary[1])
            @test initial[1] == ordinary[1][1]
            @test initial[2] ≈ ordinary[1][2]
        end
    end
end
