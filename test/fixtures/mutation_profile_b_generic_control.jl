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

end
