module WALNUTSBMutationAuthoringFixture

# Parallel mutation-profile-B translation of the WALNUTS-D compiler fixture.
# The locked mathematical fixture remains byte-identical in
# walnuts_kernel_authoring_fixture.jl. Ordinary `=` retains value/as-if
# semantics and permits only unobservable backend storage reuse;
# `destination .= source` requires identity-preserving mutation (including
# structured RK state). Bare causal calls remain only at exact captured RK
# kernel/method boundaries. This is mathematical source, not package API.
#
# The surrounding NUTS recursion is the locked ReactiveHMC-shaped fixture.  The
# only algorithmic substitution is its leaf: one fixed macro-time transition is
# resolved into a dyadic number of leapfrog micro steps, then rejected unless a
# reverse trajectory would choose the same critical grid.  The compiler must
# discover all control, effects, ownership, and replay-stream consumption from
# the ordinary @kernel source below; no WALNUTS name or layout is compiler input.
include(joinpath(@__DIR__, "nuts_kernel_authoring_fixture_b.jl"))
using ReactiveKernels
using .NUTSBMutationAuthoringFixture: fillf, tree, leapfrog!, nuts_stats!

"""Construct the external WALNUTS-D fixture with a fixed depth-10 default."""
example_walnuts_binding(init, macro_time; max_step_halvings = 10,
                         min_micro_steps = 1, max_error = oftype(init.ham, 0.1),
                         min_dham = oftype(init.ham, -1000)) =
    walnuts_state(init; step_f = leapfrog!, macro_time, max_step_halvings,
                  min_micro_steps, max_error, min_dham, stats_f = nuts_stats!)

@kernel walnuts_state(init; step_f, macro_time,
                       max_depth = 10, max_step_halvings = 10,
                       min_micro_steps = 1,
                       max_error = oftype(init.ham, 0.1),
                       min_dham = oftype(init.ham, -1000), stats_f = nothing) = begin
    gofwd = true
    may_sample = true
    may_continue = true
    fwd = deepcopy(init)
    bwd = deepcopy(init)
    candidate = deepcopy(init)
    reverse_candidate = deepcopy(init)
    trees = fillf(tree, init, max_depth + 1)
    proposals = fillf(deepcopy, init, max_depth + 2)
    dham = zero(init.ham)
    diverged = !(dham >= min_dham)
    n_steps = 0
    reached_depth = 0
    acceptance_rate = zero(init.ham)

    # Explicit, fixed-capacity control receipt.  One depth-d NUTS transition
    # can attempt at most 2^d-1 macro leaves.  These arrays are observables for
    # the independent upstream-source comparison, not compiler metadata.
    max_macro_steps = 2^max_depth - 1
    macro_count = 0
    total_micro_steps = 0
    attempted_micro_steps = fill(0, max_macro_steps)
    forward_attempts = fill(0, max_macro_steps)
    reverse_checks = fill(0, max_macro_steps)
    macro_accepted = fill(false, max_macro_steps)
    direction_index = 0
    exponential_index = 0
    replay_overflow = false

    finiteorneginf(x) = begin
        result = (x - x == zero(x)) ? x : -(one(x) / zero(x))
        result
    end
    reset!() = begin
        gofwd = true
        may_sample = true
        may_continue = true
        dham = zero(init.ham)
        n_steps = 0
        reached_depth = 0
        acceptance_rate = zero(init.ham)
        macro_count = 0
        total_micro_steps = 0
        direction_index = 0
        exponential_index = 0
        replay_overflow = false
        attempted_micro_steps .= 0
        forward_attempts .= 0
        reverse_checks .= 0
        macro_accepted .= false
        fwd .= init
        bwd .= init
        proposals[1] .= init
        proposals[length(proposals)] .= init
    end
    collectstats!() = isnothing(stats_f) || stats_f(__self__)
    logadvanceprob(depth) = trees[depth-1].log_weight[1] - trees[depth].log_weight[1]
    swapproposal!(i, j = length(proposals)) = begin
        proposals[i], proposals[j] = proposals[j], proposals[i]
    end
    record_macro!(num_steps, attempts, checks, accepted) = begin
        macro_count += 1
        if macro_count > length(attempted_micro_steps)
            replay_overflow = true
            return false
        end
        attempted_micro_steps[macro_count] = num_steps
        forward_attempts[macro_count] = attempts
        reverse_checks[macro_count] = checks
        macro_accepted[macro_count] = accepted
        return true
    end
    integrate!(ep, stepsize, num_steps) = begin
        for _ in 1:num_steps
            step_f(ep; stepsize)
            total_micro_steps += 1
        end
    end

    # Released Walnutpie WALNUTS-D leaf (walnuts.hpp:308-345): retry the
    # fixed macro time on grids m,2m,4m,... and accept the first endpoint whose
    # endpoint Hamiltonian error is within tolerance.  From that endpoint,
    # reject if any coarser reverse grid would itself have been acceptable.
    macro_step!(ep) = begin
        num_steps = min_micro_steps
        stepsize = macro_time / min_micro_steps
        attempts = 0
        for _ in 1:max_step_halvings
            attempts += 1
            candidate .= ep
            integrate!(__self__, candidate, stepsize, num_steps)
            if abs(candidate.ham - ep.ham) <= max_error
                checks = 0
                reversible = true
                reverse_num_steps = num_steps
                reverse_stepsize = stepsize
                while reverse_num_steps >= 2 * min_micro_steps
                    reverse_num_steps = div(reverse_num_steps, 2)
                    reverse_stepsize *= 2
                    reverse_candidate .= candidate
                    @. reverse_candidate.mom *= -1
                    integrate!(__self__, reverse_candidate, reverse_stepsize,
                               reverse_num_steps)
                    checks += 1
                    if abs(reverse_candidate.ham - candidate.ham) <= max_error
                        reversible = false
                        break
                    end
                end
                record_macro!(__self__, num_steps, attempts, checks, reversible) ||
                    return false
                if reversible
                    ep .= candidate
                end
                return reversible
            end
            num_steps *= 2
            stepsize *= oftype(stepsize, 0.5)
        end
        record_macro!(__self__, 0, attempts, 0, false)
        return false
    end

    next_direction!(stream) = begin
        direction_index += 1
        if direction_index > length(stream)
            replay_overflow = true
            return false
        end
        return stream[direction_index]
    end
    next_exponential!(stream) = begin
        exponential_index += 1
        if exponential_index > length(stream)
            replay_overflow = true
            return one(init.ham) / zero(init.ham)
        end
        return stream[exponential_index]
    end

    step!(directions, exponentials) = begin
        reset!(__self__)
        gofwd ? (@. bwd.mom *= -1) : (@. fwd.mom *= -1)
        trees[1].log_weight[1] = zero(init.ham)
        for depth in 1:max_depth
            reached_depth = depth
            direction = next_direction!(__self__, directions)
            replay_overflow && return
            direction && flip!(__self__, depth)
            gofwd ? finish!(__self__, fwd, depth, exponentials) :
                    finish!(__self__, bwd, depth, exponentials)
            may_sample || break
            ratio = trees[depth].log_weight[1] - trees[depth].log_weight[2]
            if !(ratio > zero(ratio))
                exponential = next_exponential!(__self__, exponentials)
                replay_overflow && return
                if -exponential < ratio
                    swapproposal!(__self__, depth)
                end
            else
                swapproposal!(__self__, depth)
            end
            may_continue || break
        end
        init .= proposals[length(proposals)]
    end
    flip!(depth) = if depth > 1
        gofwd = !gofwd
        gofwd ? flip_neg!(__self__, bwd, depth) :
                flip_neg!(__self__, fwd, depth)
    end
    flip_neg!(ep, depth) = begin
        tree = trees[depth]
        @. tree.bwd.mom = -ep.mom
        @. tree.bwd.dham_dmom = -ep.dham_dmom
        @. tree.summed_mom.fwd *= -1
    end
    finish!(ep, depth, exponentials) = begin
        tree = trees[depth]
        suptree = trees[depth+1]
        tree.log_weight[2] = tree.log_weight[1]
        if depth == 1
            @. suptree.bwd.mom = ep.mom
            @. suptree.bwd.dham_dmom = ep.dham_dmom
        else
            @. suptree.bwd.mom = tree.bwd.mom
            @. suptree.bwd.dham_dmom = tree.bwd.dham_dmom
            @. tree.bwd_fwd.mom = ep.mom
            @. tree.bwd_fwd.dham_dmom = ep.dham_dmom
            @. tree.summed_mom.bwd = tree.summed_mom.fwd
        end
        start!(__self__, ep, depth, exponentials)
        may_continue || return may_sample = false
        suptree.log_weight[1] = logaddexp(tree.log_weight[1], tree.log_weight[2])
        if depth == 1
            @. suptree.summed_mom.fwd = suptree.bwd.mom + ep.mom
            backward_dot = zero(dham)
            forward_dot = zero(dham)
            for i in 1:length(suptree.summed_mom.fwd)
                backward_dot += suptree.summed_mom.fwd[i] * suptree.bwd.dham_dmom[i]
                forward_dot += suptree.summed_mom.fwd[i] * ep.dham_dmom[i]
            end
            may_continue = backward_dot > zero(backward_dot) &&
                           forward_dot > zero(forward_dot)
        else
            @. suptree.summed_mom.fwd = tree.summed_mom.bwd + tree.summed_mom.fwd
            base_backward_dot = zero(dham)
            base_forward_dot = zero(dham)
            sum1_backward_dot = zero(dham)
            sum1_forward_dot = zero(dham)
            sum2_backward_dot = zero(dham)
            sum2_forward_dot = zero(dham)
            for i in 1:length(suptree.summed_mom.fwd)
                base_backward_dot += suptree.summed_mom.fwd[i] * suptree.bwd.dham_dmom[i]
                base_forward_dot += suptree.summed_mom.fwd[i] * ep.dham_dmom[i]
                sum1_backward_dot += (tree.summed_mom.bwd[i] + tree.bwd.mom[i]) *
                                     suptree.bwd.dham_dmom[i]
                sum1_forward_dot += (tree.summed_mom.bwd[i] + tree.bwd.mom[i]) *
                                    tree.bwd.dham_dmom[i]
                sum2_backward_dot += (tree.bwd_fwd.mom[i] + tree.summed_mom.fwd[i]) *
                                     tree.bwd_fwd.dham_dmom[i]
                sum2_forward_dot += (tree.bwd_fwd.mom[i] + tree.summed_mom.fwd[i]) *
                                    ep.dham_dmom[i]
            end
            may_continue = (
                base_backward_dot > zero(base_backward_dot) &&
                base_forward_dot > zero(base_forward_dot) &&
                sum1_backward_dot > zero(sum1_backward_dot) &&
                sum1_forward_dot > zero(sum1_forward_dot) &&
                sum2_backward_dot > zero(sum2_backward_dot) &&
                sum2_forward_dot > zero(sum2_forward_dot)
            )
        end
    end
    start!(ep, depth, exponentials) = if depth == 1
        accepted = macro_step!(__self__, ep)
        accepted || return may_continue = false
        raw_dham = init.ham - ep.ham
        dham = finiteorneginf(__self__, raw_dham)
        collectstats!(__self__)
        diverged && return may_continue = false
        trees[1].log_weight[1] = dham
        proposals[1] .= ep
    else
        start!(__self__, ep, depth - 1, exponentials)
        may_continue || return may_sample = false
        swapproposal!(__self__, depth - 1, depth)
        finish!(__self__, ep, depth - 1, exponentials)
        if may_sample
            ratio = trees[depth - 1].log_weight[1] - trees[depth].log_weight[1]
            if !(ratio > zero(ratio))
                exponential = next_exponential!(__self__, exponentials)
                replay_overflow && return may_sample = false
                if -exponential < ratio
                    swapproposal!(__self__, depth - 1, depth)
                end
            else
                swapproposal!(__self__, depth - 1, depth)
            end
        end
    end
end

# All stochastic inputs are explicit values.  This makes random consumption a
# source-path observable shared by native and Reactant execution.
@kernel walnuts!!(state; momentum, directions, exponentials) = begin
    @. state.init.mom = momentum
    step!(state, directions, exponentials)
    return state
end

end # module WALNUTSBMutationAuthoringFixture
