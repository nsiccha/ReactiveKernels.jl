module MutationProfileBGenericControl

using ReactiveKernels
using Random

macro define_recursive_probe(kernel_name, field_name, bound_name,
        emit_name, lower_name, upper_name, drive_name)
    field = field_name
    emit_definition = Expr(:(=), Expr(:call, emit_name),
        Expr(:block, Expr(:(+=), field, 1)))
    lower_branch = Expr(:if, Expr(:call, :(==), :n, 0),
            Expr(:block, Expr(:call, emit_name, :__self__)),
            Expr(:block, Expr(:call, upper_name, :__self__,
                Expr(:call, :-, :n, 1))))
    lower_definition = Expr(:(=), Expr(:call, lower_name, :n),
        Expr(:block, lower_branch, Expr(:(+=), field, 0)))
    upper_branch = Expr(:if, Expr(:call, :(>=), :n, 0),
        Expr(:block, Expr(:call, lower_name, :__self__, :n)))
    upper_definition = Expr(:(=), Expr(:call, upper_name, :n),
        Expr(:block, upper_branch, Expr(:(+=), field, 0)))
    loop = Expr(:for,
        Expr(:(=), :level, Expr(:call, :(:), 0, bound_name)),
        Expr(:block, Expr(:call, upper_name, :__self__, :level)))
    drive_definition = Expr(:(=), Expr(:call, drive_name),
        Expr(:block, loop, Expr(:(+=), field, 0)))
    signature = Expr(:call, kernel_name, Expr(:parameters), field_name, bound_name)
    body = Expr(:block, emit_definition, lower_definition,
        upper_definition, drive_definition)
    esc(Expr(:macrocall, Symbol("@kernel"), __source__,
        Expr(:(=), signature, body)))
end

# Deliberately consumer-neutral: generate the same bounded-SCC shape twice
# with different method and state-field names. This keeps the alpha-renamed
# variant coupled structurally without maintaining a second handwritten body.
@define_recursive_probe bounded_counter total ceiling emit! phase_a! phase_b! drive!
@define_recursive_probe renamed_machine tally horizon note! descend! ascend! run!

# Couple the same bounded-control contract to a fixed-shape, aliased state ABI.
# This makes shape and topology rejection part of the generic recursive seam,
# rather than relying only on unrelated straight-line structural tests.
@kernel structured_counter(payload, total, ceiling) = begin
    emit!() = total += 1
    descend!(n) = begin
        if n == 0
            emit!(__self__)
        else
            descend!(__self__, n - 1)
        end
        total += 0
    end
    drive!() = begin
        for level in 0:ceiling
            descend!(__self__, level)
        end
        total += 0
    end
end

# Both finite loop authorities reach the same recursive SCC. Automatic
# authority selection must therefore fail closed unless the certificate names
# one authority (or an explicit tuple) rather than choosing by source name.
@kernel ambiguous_counter(total, left_extent, right_extent) = begin
    emit!() = total += 1
    descend!(n) = begin
        if n == 0
            emit!(__self__)
        else
            descend!(__self__, n - 1)
        end
        total += 0
    end
    step!() = begin
        for depth in 0:left_extent
            descend!(__self__, depth)
        end
        for depth in 0:right_extent
            descend!(__self__, depth)
        end
        total += 0
    end
end

# A runtime while condition has no finite trip proof in MethodIR alone. It is
# admissible only when stateful_control_bounds receives max_iterations.
@kernel uncertified_while_counter(total, target) = begin
    step!() = begin
        while total < target
            total += 1
        end
        total += 0
    end
end

# Runtime extents need a supplied budget even when the iterator is a finite
# range. Reassigned locals exercise the same rule after a loop changes it.
@kernel runtime_for_counter(total) = begin
    step!(extent) = begin
        for _ in 1:extent
            total += 1
        end
        total += 0
    end
    local_step!(extent) = begin
        count = 1
        count = extent
        for _ in 1:count
            total += 1
        end
        total += 0
    end
end

@kernel broadcast_reset(values, marker) = begin
    reset!() = begin
        values .= 2
        values .+= 3
        marker += 0
    end
end

# Inlined loops retain Julia's separate induction-variable scopes. The
# anonymous-index case mirrors repeated grids without any sampler structure;
# the named case also reads the caller's index after the inlined helper loop.
@kernel loop_scope_counter(total, rounds) = begin
    descend!(depth) = begin
        if depth > 0
            descend!(__self__, depth - 1)
        end
        total += 0
    end
    inner!(extent) = begin
        for _ in 1:extent
            total += 1
        end
    end
    grids!() = begin
        descend!(__self__, rounds)
        count = 1
        for _ in 1:rounds
            inner!(__self__, count)
            count *= 2
        end
        total += 0
    end
    inner_index!() = begin
        for index in 1:2
            total += index
        end
    end
    nested!() = begin
        descend!(__self__, rounds)
        for index in 1:rounds
            inner_index!(__self__)
            total += index
        end
        total += 0
    end
end

function loop_scope_case(method; max_iterations=8)
    kernel = compile_stateful(loop_scope_counter, 0, 4)
    state = stateful_snapshot(kernel(0, 4))
    bounds = stateful_control_bounds(kernel, method, state;
        argument_types=Tuple{}, max_iterations, recursion_bound=:rounds)
    transition = functionalize_stateful(kernel, method, bounds)
    (; transition, state)
end

# The ordered replay is one root provider across every recursive suspension.
# This unrelated counter catches redundant provider-frame columns without using
# sampler names, layouts, or algorithm structure.
@kernel recursive_rng_counter(total, ceiling, marker) = begin
    observe!(rng) = begin
        take = marker
        take = rand(rng, Bool)
        if take
            total += 1
        else
            total += 0
        end
    end
    descend!(rng, n) = begin
        if n == 0
            observe!(__self__, rng)
        else
            descend!(__self__, rng, n - 1)
        end
        total += 0
    end
    drive!(rng) = begin
        if ceiling >= 0
            for level in 0:ceiling
                descend!(__self__, rng, level)
            end
        end
        total += 0
    end
end

@kernel owned_point(values) = begin
    mirror = values
    total = sum(values) + 0
end
@kernel bump_point!(point; delta=1) = begin
    point.values .+= delta
end
struct PointEffectAuthority <: Function end
(::PointEffectAuthority)(point) = error("functional only")

@kernel owned_alias_counter(init; ceiling=2, step_f) = begin
    left = deepcopy(init)
    right = deepcopy(init)
    counter = 0
    descend!(point, n) = begin
        if n > 0
            descend!(__self__, point, n - 1)
        end
        step_f(point)
        counter += point.total
    end
    drive!() = begin
        for level in 0:ceiling
            descend!(__self__, left, level)
        end
        descend!(__self__, right, ceiling)
        counter += 0
    end
end

function owned_alias_case()
    endpoint = compile_state_transition(owned_point, bump_point!, ([1, 2],))
    point = initial_transition_state(endpoint)
    port = structured_state_port(endpoint)
    source = PointEffectAuthority()
    lowering = total_functional_lowering((effect, value) -> (
        arguments=(endpoint(value),), result=nothing, effect_state=effect))
    step = effect_lowering_port(source, Tuple{typeof(point)}, Nothing;
        written_arguments=(1,), initial_effect_state=nothing,
        functional_lowering=lowering)
    bindings = stateful_compiler_bindings(left=port, right=port, step_f=step)
    kernel = compile_stateful(owned_alias_counter, bindings, point; step_f=source)
    state = stateful_snapshot(kernel(point; step_f=source))
    bounds = stateful_control_bounds(kernel, Val(:drive!), state; argument_types=Tuple{})
    transition = functionalize_stateful(kernel, Val(:drive!), bounds)
    (; transition, state)
end

end
