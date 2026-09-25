# Joint PPL posterior through `prepare_nonallocating`: value parity plus a
# zero-byte steady-state call. Runs only through
# `test/run_nonallocating_integration.jl`, not the default suite, because it
# needs the unregistered MutatingFunctions weak dependency (and the PPL
# parity fixture, included by path).
using ReactiveKernels
using ReactiveKernelsPPL
using MutatingFunctions
using Test
using Random

const _PPL_TESTDIR = joinpath(@__DIR__, "..", "packages", "ReactiveKernelsPPL",
    "test")
include(joinpath(_PPL_TESTDIR, "parity", "joint_parity_fixture.jl"))
include(joinpath(_PPL_TESTDIR, "parity", "joint_tiling.jl"))

# Measure through a function barrier with concrete argument types: `@allocated`
# at top level over globals boxes the dynamic-call boundary (one 16-byte float
# box), which would pollute the reading.
function _na_joint_lp_bytes(na_q, u)
    na_q(u)
    @allocated na_q(u)
end

@testset "joint posterior non-allocating parity + zero bytes" begin
    cols = tiled_columns(1)
    plan = final_plan("continuous")
    bound = bind_data(plan, cols; dims = Dict(:kernel_nsub_pk_loc => 3))
    built = build_kernel(bound)
    post_q = prepare_query(built, bound, :sampler)
    u = Vector{Float64}(0.1 .* randn(Xoshiro(20260917),
            assign_layout(bound).total))
    na_q = Base.invokelatest(prepare_nonallocating, post_q)
    ref = post_q(u)
    got = na_q(u)
    @test abs(got - ref) <= 1e-14 * abs(ref)
    @test _na_joint_lp_bytes(na_q, u) == 0
end
