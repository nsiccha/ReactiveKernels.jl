module StructuredLoopReactantTests

using ReactiveKernels, Reactant, Test
include(joinpath(@__DIR__, "fixtures", "mutation_profile_b_generic_control.jl"))
const F = MutationProfileBGenericControl
const RK = ReactiveKernels

@kernel stopping_loop(total, limit, stop) = begin
    step!() = begin
        for _ in 1:limit
            total += 1
            total == stop && return
            total += 100
        end
        total += 1000
    end
end

@testset "a retained loop preserves early return" begin
    kernel = RK.compile_stateful(stopping_loop, 0, 4, 102)
    state = RK.stateful_snapshot(kernel(0, 4, 102))
    transition = RK.functionalize_stateful(kernel, Val(:step!);
        max_iterations=4, argument_types=Tuple{})
    native = transition(state)
    @test native.state.total == 102
    @test native.returned
    input = Reactant.to_rarray(state; track_numbers=true)
    fn = Reactant.compile(transition, (input,); sync=true, donated_args=:none)
    result = fn(input)
    @test Int(result.state.total) == 102
    @test Bool(result.returned)
    @test !Bool(result.control_overflow)
    @test Int(input.total) == 0
    next = fn(result.state)
    @test Int(next.state.total) == 1506
    @test !Bool(next.returned)
end

energy(values) = sum(abs2, values)
@kernel loop_endpoint(authority, values) = begin
    mirror = values
    energy = authority(values)
end
@kernel increment_endpoint!(point) = begin
    @. point.values += 1
end
@kernel endpoint_loop(left, limit, total; operation) = begin
    right = deepcopy(left)
    step!() = begin
        for _ in 1:limit
            operation(right)
            total += right.energy
        end
        left .= right
    end
end
struct EndpointAuthority <: Function end
(::EndpointAuthority)(args...) = error("functional compiler authority")
struct EndpointLowering{T}
    transition::T
end
function (op::EndpointLowering)(effect, point)
    updated = op.transition(point)
    (arguments=(updated,), result=nothing,
     effect_state=(calls=effect.calls + 1, last_energy=updated.energy))
end

@testset "retained loops preserve structured aliases and static authorities" begin
    endpoint = RK.compile_state_transition(
        loop_endpoint, increment_endpoint!, (energy, [1.0, 2.0]))
    point = RK.initial_transition_state(endpoint)
    structured = RK.structured_state_port(endpoint)
    source = EndpointAuthority()
    port = RK.effect_lowering_port(source, Tuple{typeof(point)}, Nothing;
        written_arguments=(1,), initial_effect_state=(calls=0, last_energy=0.0),
        functional_lowering=RK.total_functional_lowering(EndpointLowering(endpoint)))
    bindings = RK.stateful_compiler_bindings(
        left=structured, right=structured, operation=port)
    kernel = RK.compile_stateful(endpoint_loop, bindings, point, 2, 0.0;
        operation=source)
    state = RK.stateful_snapshot(kernel(point, 2, 0.0; operation=source))
    transition = RK.functionalize_stateful(kernel, Val(:step!);
        max_iterations=2, argument_types=Tuple{})
    native = transition(state)
    @test native.state.total == 38
    @test native.state.left.values == [3, 4]
    @test native.state.left.values === native.state.left.mirror
    @test native.state.left.values !== native.state.right.values
    @test native.state.left.authority === energy
    @test native.effects.operation.calls == 2
    input = Reactant.to_rarray(state; track_numbers=true)
    fn = Reactant.compile(transition, (input,); sync=true, donated_args=:none)
    result = fn(input)
    @test Float64(result.state.total) == 38
    @test Array(result.state.left.values) == [3, 4]
    @test result.state.left.values === result.state.left.mirror
    @test result.state.left.values !== result.state.right.values
    @test result.state.left.authority === energy
    @test Array(input.left.values) == [1, 2]
    @test !Bool(result.control_overflow)
    @test Int(result.effects.operation.calls) == 2
    @test Float64(result.effects.operation.last_energy) == 25
end

@testset "retained unit ranges and integer boundaries" begin
    for (T, budget, inputs) in (
        (Int, 4, ((typemin(Int), typemax(Int), true, 0),
                  (typemax(Int) - 1, typemax(Int), false, 2),
                  (4, 2, false, 0), (2, 5, false, 4), (2, 6, true, 0))),
        (Int8, 300, ((typemin(Int8), typemax(Int8), false, 256),
                    (Int8(127), Int8(127), false, 1))),
        (UInt8, 256, ((UInt8(0), typemax(UInt8), false, 256),
                     (UInt8(5), UInt8(4), false, 0))),
    )
        case = F.unit_range_case(T; max_iterations=budget)
        state = Reactant.to_rarray(case.state; track_numbers=true)
        scalar(x) = Reactant.to_rarray(x; track_numbers=true)
        fn = Reactant.compile(case.transition, (state, scalar(T(1)), scalar(T(2)));
            sync=true, donated_args=:none, serializable=true)
        @test occursin("stablehlo.while", fn.module_string)
        for (lower, upper, overflow, visits) in inputs
            native = case.transition(case.state, lower, upper)
            @test native.control_overflow == overflow
            @test native.state.visits == visits
            result = fn(state, scalar(lower), scalar(upper))
            @test Bool(result.control_overflow) == overflow
            @test Int(result.state.visits) == visits
            @test Int(state.visits) == 0
        end
    end
end

@testset "large finite allowance retains a compact body" begin
    case = F.readonly_index_case(; max_iterations=1_000_000)
    state = Reactant.to_rarray(case.state; track_numbers=true)
    fn = Reactant.compile(case.transition, (state,);
        sync=true, donated_args=:none, serializable=true)
    @test occursin("stablehlo.while", fn.module_string)
    result = fn(state)
    @test Float64(result.state.total) == 5
    @test !Bool(result.control_overflow)
    @test Float64(fn(result.state).state.total) == 10
end

end
