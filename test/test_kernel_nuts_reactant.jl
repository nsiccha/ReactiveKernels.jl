using ReactiveKernels
using Reactant
using LinearAlgebra
using Random
using Test

include(joinpath(@__DIR__, "..", "examples", "nuts_runtime.jl"))

module ReactantNutsFixture
include(joinpath(@__DIR__, "..", "benchmark", "nuts_kernel_authoring_fixture.jl"))
end

Random.eval(quote
    mutable struct RKReactantTestRNG <: AbstractRNG
        inner::Xoshiro
        log::Vector{Any}
    end
end)
const _NRTestRNG = Random.RKReactantTestRNG
Random.randn!(rng::_NRTestRNG, x::AbstractArray) =
    (Random.randn!(rng.inner, x); push!(rng.log, (:momentum, copy(x))); x)
Base.rand(rng::_NRTestRNG, ::Type{Bool}) =
    (value = Base.rand(rng.inner, Bool); push!(rng.log, (:direction, value)); value)
Random.randexp(rng::_NRTestRNG) =
    (value = Random.randexp(rng.inner); push!(rng.log, (:exponential, value)); value)

const _NR_FIX = ReactantNutsFixture
const _NR_PF = ReactiveKernels._prepare_factory(
    _NR_FIX.euclidean_phasepoint,
    ReactiveKernels.kernel_registration(_NR_FIX.leapfrog!))

function _nr_test_values(::Type{T}; metric=T[2 0; 0 2]) where {T}
    values = Dict{Int,Any}()
    for slot in ReactiveKernels.kernel_plan_slots(
            ReactiveKernels.kernel_prepared_plan(_NR_PF))
        name = String(slot.path[1])
        values[slot.canon] = name == "pot_f" ? (p -> sum(abs2, p)) :
            name == "grad_f" ? ((dst, p) -> (dst .= 2 .* p; sum(abs2, p))) :
            name == "metric" ? metric :
            name == "chol_metric" ? cholesky(metric) :
            startswith(name, "##node") ? zero(T) :
            name == "pos" ? T[1, 2] :
            name == "mom" ? T[3, 4] :
            name in ("dpot_dpos", "dham_dpos", "dkin_dmom", "dham_dmom") ?
                T[0, 0] : zero(T)
    end
    values
end

function _nr_test_frame(::Type{T}, max_depth;
        metric=T[2 0; 0 2], min_dham=T(-1000)) where {T}
    frame = ReactiveKernels._construct_nuts_frame(
        _NR_PF, _nr_test_values(T; metric), max_depth;
        step_f=ReactiveKernels.partial(_NR_FIX.leapfrog!; stepsize=T(0.1)),
        stats_f=_NR_FIX.nuts_stats!, min_dham)
    ReactiveKernels.compile_prepared_initialization(
        _NR_PF, typeof(frame.init), typeof(frame.shared))(
            frame.init, frame.shared,
            ReactiveKernels.kernel_prepared_handles(_NR_PF))
    ReactiveKernels._seed_nuts_children!(frame)
    frame
end

function _nr_test_bundle(log, max_depth)
    momentum = Float64[]; directions = Bool[]; exponentials = Float64[]
    for item in log
        item[1] === :momentum ? (momentum = copy(item[2])) :
        item[1] === :direction ? push!(directions, item[2]) :
        push!(exponentials, item[2])
    end
    bundle = ReactiveKernels.nuts_reactant_bundle(
        momentum, directions, exponentials, max_depth)
    bundle, directions, exponentials
end

_nr_close(a, b) = all(isapprox.(a, b; rtol=0, atol=64eps(Float64)))

function _nr_test_match(native_frame, output, directions, exponentials)
    phase_ok = _nr_close(getfield(native_frame.init, :f4), Array(output.pp_pos)[:, 1]) &&
        _nr_close(getfield(native_frame.init, :f5), Array(output.pp_mom)[:, 1]) &&
        _nr_close(getfield(native_frame.init, :f8), Array(output.pp_dpot)[:, 1]) &&
        _nr_close(getfield(native_frame.init, :f10), Array(output.pp_dkin)[:, 1]) &&
        _nr_close([getfield(native_frame.init, :f7), getfield(native_frame.init, :f11),
                   getfield(native_frame.init, :f12)],
                  [Array(output.pp_pot)[1], Array(output.pp_kin)[1],
                   Array(output.pp_ham)[1]])
    diagnostics_ok = native_frame.diag.n_steps == Array(output.n_steps)[1] &&
        native_frame.diag.reached_depth == Array(output.reached_depth)[1] &&
        _nr_close([native_frame.diag.acceptance_rate, native_frame.diag.dham],
                  [Array(output.acc)[1], Array(output.dham)[1]]) &&
        Int(native_frame.diverged) == Array(output.diverged)[1]
    control_ok = Array(output.csp)[1] == 0 &&
        all(iszero, Array(output.overflow)) &&
        Array(output.kd)[1] == length(directions) &&
        Array(output.ke)[1] == length(exponentials)
    phase_ok && diagnostics_ok && control_ok
end

function _nr_test_native!(frame, seed)
    native = ReactiveKernels.compile_nuts(
        _NR_PF, _NR_FIX.nuts_state, _NR_FIX.refresh_momentum!!,
        _NR_FIX.nuts!!, frame)
    rng = _NRTestRNG(Xoshiro(seed), Any[])
    native.root!(frame, native.scratch, rng)
    _nr_test_bundle(rng.log, frame.max_depth)
end

@testset "kernel-derived adaptive NUTS compiles through Reactant" begin
    seed_frame = _nr_test_frame(Float64, 1)
    compiled = ReactiveKernels.compile_nuts_reactant(
        _NR_PF, _NR_FIX.nuts_state, _NR_FIX.refresh_momentum!!,
        _NR_FIX.nuts!!, seed_frame)

    @testset "backend loop and zero-exponential path" begin
        native_frame = _nr_test_frame(Float64, 1)
        bundle, directions, exponentials = _nr_test_native!(native_frame, 999)
        traced_frame = _nr_test_frame(Float64, 1)
        state = map(Reactant.to_rarray,
            ReactiveKernels.nuts_reactant_state(compiled, traced_frame, bundle))
        executable = ReactiveKernels.nuts_reactant_compile(compiled, state)
        output = executable(state)
        hlo = executable.module_string

        @test _nr_test_match(native_frame, output, directions, exponentials)
        @test output.pp_pos isa Reactant.RArray
        @test count(_ -> true, eachmatch(r"stablehlo\.while", hlo)) == 1
        @test count(_ -> true, eachmatch(r"stablehlo\.select", hlo)) > 0
        @test !occursin("enzyme.batch", lowercase(hlo))
        @test !occursin("ConcretePJRT", hlo)
        @test isempty(exponentials)
        written = _nr_test_frame(Float64, 1)
        @test ReactiveKernels.nuts_reactant_writeback!(written, output) === written
        @test _nr_close(getfield(written.init, :f4), getfield(native_frame.init, :f4))
        @test written.diag.n_steps == native_frame.diag.n_steps
    end

    @testset "deep tree and carried second transition" begin
        native_frame = _nr_test_frame(Float64, 6)
        bundle1, directions1, exponentials1 = _nr_test_native!(native_frame, 12345)
        traced_frame = _nr_test_frame(Float64, 6)
        state1 = map(Reactant.to_rarray,
            ReactiveKernels.nuts_reactant_state(compiled, traced_frame, bundle1))
        executable = ReactiveKernels.nuts_reactant_compile(compiled, state1)
        output1 = executable(state1)
        @test _nr_test_match(native_frame, output1, directions1, exponentials1)
        @test Array(output1.n_steps)[1] == 31

        bundle2, directions2, exponentials2 = _nr_test_native!(native_frame, 54321)
        device_bundle2 = map(Reactant.to_rarray, bundle2)
        state2 = ReactiveKernels.nuts_reactant_rebundle(output1, device_bundle2)
        output2 = executable(state2)
        @test _nr_test_match(native_frame, output2, directions2, exponentials2)
    end

    @testset "early U-turn and divergence" begin
        native_frame = _nr_test_frame(Float64, 8)
        bundle, directions, exponentials = _nr_test_native!(native_frame, 777)
        state = map(Reactant.to_rarray, ReactiveKernels.nuts_reactant_state(
            compiled, _nr_test_frame(Float64, 8), bundle))
        executable = ReactiveKernels.nuts_reactant_compile(compiled, state)
        output = executable(state)
        @test _nr_test_match(native_frame, output, directions, exponentials)
        @test Array(output.reached_depth)[1] < 8

        divergent_native = _nr_test_frame(Float64, 1; min_dham=Inf)
        divergent_bundle, divergent_directions, divergent_exponentials =
            _nr_test_native!(divergent_native, 20260829)
        divergent_state = map(Reactant.to_rarray,
            ReactiveKernels.nuts_reactant_state(compiled,
                _nr_test_frame(Float64, 1; min_dham=Inf), divergent_bundle))
        divergent_executable = ReactiveKernels.nuts_reactant_compile(
            compiled, divergent_state)
        divergent_output = divergent_executable(divergent_state)
        @test Array(divergent_output.diverged)[1] == 1
        @test _nr_test_match(divergent_native, divergent_output,
            divergent_directions, divergent_exponentials)
    end

    @testset "unsupported specializations reject" begin
        float32_frame = _nr_test_frame(Float32, 1)
        @test_throws ArgumentError ReactiveKernels.compile_nuts_reactant(
            _NR_PF, _NR_FIX.nuts_state, _NR_FIX.refresh_momentum!!,
            _NR_FIX.nuts!!, float32_frame)
        nondiagonal = Float64[2 0.1; 0.1 2]
        nondiagonal_frame = _nr_test_frame(Float64, 1; metric=nondiagonal)
        @test_throws ArgumentError ReactiveKernels.compile_nuts_reactant(
            _NR_PF, _NR_FIX.nuts_state, _NR_FIX.refresh_momentum!!,
            _NR_FIX.nuts!!, nondiagonal_frame)
        @test_throws DimensionMismatch ReactiveKernels.nuts_reactant_bundle(
            [1.0, 2.0], [true, false], Float64[], 1)
        @test_throws DimensionMismatch ReactiveKernels.nuts_reactant_bundle(
            [1.0, 2.0], Bool[], [1.0, 2.0, 3.0], 1)
        @test_throws ArgumentError ReactiveKernels.nuts_reactant_bundle(
            Float32[1, 2], Bool[], Float64[], 1)
    end
end
