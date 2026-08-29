using ReactiveKernels
using Reactant
using LinearAlgebra
using Random
using Test

include(joinpath(@__DIR__, "..", "examples", "nuts_runtime.jl"))
include(joinpath(@__DIR__, "nuts_source_oracle.jl"))

module ReactantNutsFixture
include(joinpath(@__DIR__, "..", "benchmark", "nuts_kernel_authoring_fixture.jl"))
end

Random.eval(quote
    mutable struct RKReactantTestRNG <: AbstractRNG
        inner::Xoshiro
        log::Vector{Any}
    end
    mutable struct RKReactantReplayRNG <: AbstractRNG
        momentum::Vector{Float64}
        directions::Vector{Bool}
        exponentials::Vector{Float64}
        direction_index::Int
        exponential_index::Int
    end
end)
const _NRTestRNG = Random.RKReactantTestRNG
const _NRReplayRNG = Random.RKReactantReplayRNG
Random.randn!(rng::_NRTestRNG, x::AbstractArray) =
    (Random.randn!(rng.inner, x); push!(rng.log, (:momentum, copy(x))); x)
Base.rand(rng::_NRTestRNG, ::Type{Bool}) =
    (value = Base.rand(rng.inner, Bool); push!(rng.log, (:direction, value)); value)
Random.randexp(rng::_NRTestRNG) =
    (value = Random.randexp(rng.inner); push!(rng.log, (:exponential, value)); value)
Random.randn!(rng::_NRReplayRNG, x::AbstractArray) =
    (copyto!(x, rng.momentum); x)
Base.rand(rng::_NRReplayRNG, ::Type{Bool}) =
    (rng.direction_index += 1; rng.directions[rng.direction_index])
Random.randexp(rng::_NRReplayRNG) =
    (rng.exponential_index += 1; rng.exponentials[rng.exponential_index])

const _NR_FIX = ReactantNutsFixture
const _NR_PF = ReactiveKernels._prepare_factory(
    _NR_FIX.euclidean_phasepoint,
    ReactiveKernels.kernel_registration(_NR_FIX.leapfrog!))

_nr_quadratic_pot(p) = sum(abs2, p)
function _nr_quadratic_grad!(dst, p)
    dst .= 2 .* p
    sum(abs2, p)
end
_nr_nan_pot(p) = oftype(sum(abs2, p), NaN)
function _nr_nan_grad!(dst, p)
    fill!(dst, zero(eltype(dst)))
    oftype(sum(abs2, p), NaN)
end

function _nr_test_values(::Type{T}; metric=T[2 0; 0 2],
        pot_f=_nr_quadratic_pot, grad_f=_nr_quadratic_grad!) where {T}
    values = Dict{Int,Any}()
    for slot in ReactiveKernels.kernel_plan_slots(
            ReactiveKernels.kernel_prepared_plan(_NR_PF))
        name = String(slot.path[1])
        values[slot.canon] = name == "pot_f" ? pot_f :
            name == "grad_f" ? grad_f :
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
        metric=T[2 0; 0 2], min_dham=T(-1000),
        pot_f=_nr_quadratic_pot, grad_f=_nr_quadratic_grad!) where {T}
    frame = ReactiveKernels._construct_nuts_frame(
        _NR_PF, _nr_test_values(T; metric, pot_f, grad_f), max_depth;
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

function _nr_oracle_transition!(oracle, seed)
    rng = _NRTestRNG(Xoshiro(seed), Any[])
    NutsSourceOracle.transition!(oracle, rng)
    snapshot = NutsSourceOracle.snapshot(oracle)
    bundle, directions, exponentials = _nr_test_bundle(rng.log, oracle.max_depth)
    snapshot, bundle, directions, exponentials
end

function _nr_replay_native!(frame, bundle, directions, exponentials)
    native = ReactiveKernels.compile_nuts(
        _NR_PF, _NR_FIX.nuts_state, _NR_FIX.refresh_momentum!!,
        _NR_FIX.nuts!!, frame)
    rng = _NRReplayRNG(copy(bundle.momentum), copy(directions), copy(exponentials), 0, 0)
    native.root!(frame, native.scratch, rng)
    @test rng.direction_index == length(directions)
    @test rng.exponential_index == length(exponentials)
    frame
end

_nr_slot(frame, field) = ReactiveKernels._canon_slot(
    frame.init,
    ReactiveKernels.kernel_plan_named_slot_val(
        ReactiveKernels.kernel_prepared_plan(_NR_PF), Val(field)))

function _nr_native_matches_oracle(expected, frame)
    _nr_close(_nr_slot(frame, :pos), expected.pos) &&
        _nr_close(_nr_slot(frame, :mom), expected.mom) &&
        _nr_close(_nr_slot(frame, :dpot_dpos), expected.dpot) &&
        _nr_close(_nr_slot(frame, :dkin_dmom), expected.dkin) &&
        _nr_close([_nr_slot(frame, :pot), _nr_slot(frame, :kin), _nr_slot(frame, :ham)],
            [expected.pot, expected.kin, expected.ham]) &&
        frame.gofwd == expected.gofwd &&
        frame.may_sample == expected.may_sample &&
        frame.may_continue == expected.may_continue &&
        Bool(frame.diverged) == expected.diverged &&
        frame.diag.n_steps == expected.n_steps &&
        frame.diag.reached_depth == expected.reached_depth &&
        _nr_close([frame.diag.acceptance_rate, frame.diag.dham],
            [expected.acceptance_rate, expected.dham])
end

function _nr_reactant_matches_oracle(expected, output, directions, exponentials)
    _nr_close(Array(output.pp_pos)[:, 1], expected.pos) &&
        _nr_close(Array(output.pp_mom)[:, 1], expected.mom) &&
        _nr_close(Array(output.pp_dpot)[:, 1], expected.dpot) &&
        _nr_close(Array(output.pp_dkin)[:, 1], expected.dkin) &&
        _nr_close([Array(output.pp_pot)[1], Array(output.pp_kin)[1],
                   Array(output.pp_ham)[1]],
            [expected.pot, expected.kin, expected.ham]) &&
        Bool(Array(output.gofwd)[1]) == expected.gofwd &&
        Bool(Array(output.may_sample)[1]) == expected.may_sample &&
        Bool(Array(output.may_continue)[1]) == expected.may_continue &&
        Bool(Array(output.diverged)[1]) == expected.diverged &&
        Array(output.n_steps)[1] == expected.n_steps &&
        Array(output.reached_depth)[1] == expected.reached_depth &&
        _nr_close([Array(output.acc)[1], Array(output.dham)[1]],
            [expected.acceptance_rate, expected.dham]) &&
        Array(output.csp)[1] == 0 &&
        all(iszero, Array(output.overflow)) &&
        Array(output.kd)[1] == length(directions) &&
        Array(output.ke)[1] == length(exponentials)
end

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

@testset "kernel-derived adaptive NUTS compiles through Reactant" begin
    seed_frame = _nr_test_frame(Float64, 1)
    compiled = ReactiveKernels.compile_nuts_reactant(
        _NR_PF, _NR_FIX.nuts_state, _NR_FIX.refresh_momentum!!,
        _NR_FIX.nuts!!, seed_frame)

    @testset "backend loop and zero-exponential path" begin
        oracle = NutsSourceOracle.State(max_depth=1)
        expected, bundle, directions, exponentials =
            _nr_oracle_transition!(oracle, 999)
        native_frame = _nr_test_frame(Float64, 1)
        _nr_replay_native!(native_frame, bundle, directions, exponentials)
        traced_frame = _nr_test_frame(Float64, 1)
        state = map(Reactant.to_rarray,
            ReactiveKernels.nuts_reactant_state(compiled, traced_frame, bundle))
        executable = ReactiveKernels.nuts_reactant_compile(compiled, state)
        output = executable(state)
        hlo = executable.module_string

        @test _nr_test_match(native_frame, output, directions, exponentials)
        @test _nr_native_matches_oracle(expected, native_frame)
        @test _nr_reactant_matches_oracle(expected, output, directions, exponentials)
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
        oracle = NutsSourceOracle.State(max_depth=6)
        expected1, bundle1, directions1, exponentials1 =
            _nr_oracle_transition!(oracle, 12345)
        native_frame = _nr_test_frame(Float64, 6)
        _nr_replay_native!(native_frame, bundle1, directions1, exponentials1)
        traced_frame = _nr_test_frame(Float64, 6)
        state1 = map(Reactant.to_rarray,
            ReactiveKernels.nuts_reactant_state(compiled, traced_frame, bundle1))
        executable = ReactiveKernels.nuts_reactant_compile(compiled, state1)
        output1 = executable(state1)
        @test _nr_test_match(native_frame, output1, directions1, exponentials1)
        @test _nr_native_matches_oracle(expected1, native_frame)
        @test _nr_reactant_matches_oracle(expected1, output1, directions1, exponentials1)
        @test Array(output1.n_steps)[1] == expected1.n_steps == 31

        expected2, bundle2, directions2, exponentials2 =
            _nr_oracle_transition!(oracle, 54321)
        _nr_replay_native!(native_frame, bundle2, directions2, exponentials2)
        device_bundle2 = map(Reactant.to_rarray, bundle2)
        state2 = ReactiveKernels.nuts_reactant_rebundle(output1, device_bundle2)
        output2 = executable(state2)
        @test _nr_test_match(native_frame, output2, directions2, exponentials2)
        @test _nr_native_matches_oracle(expected2, native_frame)
        @test _nr_reactant_matches_oracle(expected2, output2, directions2, exponentials2)
    end

    @testset "adversarial backward U-turn and divergence" begin
        oracle = NutsSourceOracle.State(max_depth=8)
        expected, bundle, directions, exponentials =
            _nr_oracle_transition!(oracle, 777)
        native_frame = _nr_test_frame(Float64, 8)
        _nr_replay_native!(native_frame, bundle, directions, exponentials)
        state = map(Reactant.to_rarray, ReactiveKernels.nuts_reactant_state(
            compiled, _nr_test_frame(Float64, 8), bundle))
        executable = ReactiveKernels.nuts_reactant_compile(compiled, state)
        output = executable(state)
        @test _nr_test_match(native_frame, output, directions, exponentials)
        @test _nr_native_matches_oracle(expected, native_frame)
        @test _nr_reactant_matches_oracle(expected, output, directions, exponentials)
        @test directions[1:2] == [false, true]
        @test Array(output.reached_depth)[1] == expected.reached_depth == 5
        @test Array(output.n_steps)[1] == expected.n_steps == 31
        @test _nr_close(Array(output.pp_pos)[:, 1],
            [1.5941958933255589, 0.23013198901670232])

        divergent_oracle = NutsSourceOracle.State(max_depth=1, min_dham=Inf)
        divergent_expected, divergent_bundle, divergent_directions,
            divergent_exponentials = _nr_oracle_transition!(divergent_oracle, 20260829)
        divergent_native = _nr_test_frame(Float64, 1; min_dham=Inf)
        _nr_replay_native!(divergent_native, divergent_bundle,
            divergent_directions, divergent_exponentials)
        divergent_state = map(Reactant.to_rarray,
            ReactiveKernels.nuts_reactant_state(compiled,
                _nr_test_frame(Float64, 1; min_dham=Inf), divergent_bundle))
        divergent_executable = ReactiveKernels.nuts_reactant_compile(
            compiled, divergent_state)
        divergent_output = divergent_executable(divergent_state)
        @test Array(divergent_output.diverged)[1] == 1
        @test _nr_test_match(divergent_native, divergent_output,
            divergent_directions, divergent_exponentials)
        @test _nr_native_matches_oracle(divergent_expected, divergent_native)
        @test _nr_reactant_matches_oracle(divergent_expected, divergent_output,
            divergent_directions, divergent_exponentials)
    end

    @testset "admitted depth zero derives divergence without controls" begin
        oracle = NutsSourceOracle.State(max_depth=0, min_dham=Inf)
        expected, bundle, directions, exponentials =
            _nr_oracle_transition!(oracle, 20260829)
        @test isempty(directions)
        @test isempty(exponentials)
        @test expected.diverged
        @test expected.reached_depth == 0
        @test expected.n_steps == 0

        native_frame = _nr_test_frame(Float64, 0; min_dham=Inf)
        _nr_replay_native!(native_frame, bundle, directions, exponentials)
        state = map(Reactant.to_rarray, ReactiveKernels.nuts_reactant_state(
            compiled, _nr_test_frame(Float64, 0; min_dham=Inf), bundle))
        executable = ReactiveKernels.nuts_reactant_compile(compiled, state)
        output = executable(state)
        @test _nr_native_matches_oracle(expected, native_frame)
        @test _nr_reactant_matches_oracle(expected, output, directions, exponentials)
        @test Array(output.diverged)[1] == 1
        @test Array(output.kd)[1] == Array(output.ke)[1] == 0
        @test all(iszero, Array(output.overflow))

        # A prior negative-infinity energy diagnostic must not leak NaN through reset (`-Inf * 0`).
        poisoned = merge(state, (dham=Reactant.to_rarray([-Inf]),))
        reset_output = executable(poisoned)
        @test Array(reset_output.dham)[1] == 0.0
        @test Array(reset_output.diverged)[1] == 1
        @test _nr_reactant_matches_oracle(expected, reset_output, directions, exponentials)
    end

    @testset "repeated nonfinite leaves retain negative-infinity sentinel" begin
        bundle = ReactiveKernels.nuts_reactant_bundle(
            [1.0, 0.5], [false, false], [1.0, 1.0, 1.0], 2)
        native_frame = _nr_test_frame(Float64, 2;
            min_dham=-Inf, pot_f=_nr_nan_pot, grad_f=_nr_nan_grad!)
        _nr_replay_native!(native_frame, bundle, [false, false], [1.0, 1.0, 1.0])

        traced_frame = _nr_test_frame(Float64, 2;
            min_dham=-Inf, pot_f=_nr_nan_pot, grad_f=_nr_nan_grad!)
        nonfinite_compiled = ReactiveKernels.compile_nuts_reactant(
            _NR_PF, _NR_FIX.nuts_state, _NR_FIX.refresh_momentum!!,
            _NR_FIX.nuts!!, traced_frame)
        state = map(Reactant.to_rarray, ReactiveKernels.nuts_reactant_state(
            nonfinite_compiled, traced_frame, bundle))
        executable = ReactiveKernels.nuts_reactant_compile(nonfinite_compiled, state)
        output = executable(state)

        @test native_frame.diag.dham == -Inf
        @test !native_frame.diverged
        @test native_frame.diag.n_steps == 3
        @test native_frame.diag.reached_depth == 2
        @test Array(output.dham)[1] == native_frame.diag.dham == -Inf
        @test Bool(Array(output.diverged)[1]) == native_frame.diverged == false
        @test Array(output.n_steps)[1] == native_frame.diag.n_steps == 3
        @test Array(output.reached_depth)[1] == native_frame.diag.reached_depth == 2
        @test Array(output.kd)[1] == 2
        @test Array(output.ke)[1] == 3
        @test all(iszero, Array(output.overflow))
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
