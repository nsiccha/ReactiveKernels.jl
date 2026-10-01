# Grouped linear-PK PPL posterior through `prepare_nonallocating`: value
# parity plus a zero-byte steady-state call. Runs only through
# `test/run_nonallocating_integration.jl`, not the default suite, because it
# needs the unregistered MutatingFunctions weak dependency.
using ReactiveKernels
using ReactiveKernelsPPL
using MutatingFunctions
using Test
using Random

# Measure through a function barrier with concrete argument types: `@allocated`
# at top level over globals boxes the dynamic-call boundary (one 16-byte float
# box), which would pollute the reading.
function _na_pk_lp_bytes(na_q, u)
    na_q(u)
    @allocated na_q(u)
end

@testset "grouped PK posterior non-allocating parity + zero bytes" begin
    # Two subjects; subject 1 has a repeated-dose segment, subject 2 a single
    # dose, so both the propagate and the regular-repeat branches run.
    plan = lower_rkppl(Meta.parse("begin\nsigma ~ Exponential(1.0)\n" *
            "b0_vc ~ Normal(0.0, 1.0)\nb1_vc ~ Normal(0.0, 1.0)\n" *
            "b0_k10 ~ Normal(0.0, 1.0)\nb0_k12 ~ Normal(0.0, 1.0)\n" *
            "b0_k21 ~ Normal(0.0, 1.0)\nb0_ka ~ Normal(0.0, 1.0)\n" *
            "log_Vc = b0_vc .+ b1_vc .* age_s\nlog_k10 = b0_k10\n" *
            "log_k12 = b0_k12\nlog_k21 = b0_k21\nlog_ka = b0_ka\n" *
            "pk_sched = linear_pk_schedule(obs = (:subj, :time), " *
            "dose = (:dsubj, :dtime, :damt))\n" *
            "@plate conc for s in 1:kernel_nsub_conc\n" *
            " read_locs = linear_pk_read_locs(pk_sched, log_Vc, log_k10, " *
            "log_k12, log_k21, log_ka)\n" *
            " mu = read_locs[pk_sched.obs_map]\n" *
            " dv .~ Normal.(mu, sigma)\n mu\nend\nend"),
        (:subj, :time, :dsubj, :dtime, :damt, :dv, :age_s))
    cols = Dict{Symbol,AbstractVector}(
        :subj => [1, 1, 2, 2], :time => [96.0, 120.0, 0.0, 5.0],
        :dsubj => [1, 1, 1, 1, 2], :dtime => [0.0, 24.0, 48.0, 72.0, 0.0],
        :damt => [100.0, 100.0, 100.0, 100.0, 50.0],
        :dv => [10.0, 8.0, 0.5, 7.0], :age_s => [35.0, 52.0])
    bound = bind_data(plan, cols; dims = Dict{Symbol,Int}(:kernel_nsub_conc => 2))
    built = build_kernel(bound)
    post_q = prepare_query(built, bound, :sampler)
    u = Vector{Float64}(0.1 .* randn(Xoshiro(20260917),
            assign_layout(bound).total))
    na_q = Base.invokelatest(prepare_nonallocating, post_q)
    ref = post_q(u)
    got = na_q(u)
    @test isfinite(ref)
    @test abs(got - ref) <= 1e-14 * abs(ref)
    @test _na_pk_lp_bytes(na_q, u) == 0
end
