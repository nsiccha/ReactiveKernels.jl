module ScanStripRefillADTests

using ReactiveKernels, Enzyme, Test
using ReactiveKernels.DifferentiationInterface: AutoEnzyme

# Keep the gathered plate inside each group and retain every scan output:
# these are the loops in benchmark/repro_enzyme_strip_refill_stored_recurrence.jl.
@kernel grouped_walks(theta::Vector{Float64}, index_by, groups) = begin
    cells = plate(groups) do g
        amount = plate(index_by[g]) do q
            theta[q]
        end
        walk = scan(amount; init = 0.0) do carry, amt
            next = 0.5 * carry + amt
            (next, next)
        end
        sum(abs2, walk; init = 0.0)
    end
    total::Float64 = sum(cells)
    return total
end

# Independent, unstriped Julia evaluation for both values and central differences.
reference(theta, index_by) = sum(index_by; init = 0.0) do idx
    walk = accumulate((carry, q) -> 0.5 * carry + theta[q], idx; init = 0.0)
    sum(abs2, walk; init = 0.0)
end

function central(f, theta; h = 1e-6)
    map(eachindex(theta)) do i
        up, down = copy(theta), copy(theta)
        up[i] += h
        down[i] -= h
        (f(up) - f(down)) / (2h)
    end
end

@testset "stored scans fed by strip-refilled plates" begin
    # Unrepaired Enzyme on Julia 1.10 aborts the process at the third strip.
    # CONTRIBUTING.md's Derivative authority permits naming such skipped cases.
    # Remove this guard once registered Enzyme carries replayed_cache_limits.
    repaired = isdefined(Enzyme, :has_dogfood_repair) &&
        Enzyme.has_dogfood_repair(:replayed_cache_limits)
    for lengths in ([256], [257], [387], [3000], [0, 1, 127, 128, 129, 256, 257, 387, 513])
        @testset "group lengths $lengths" begin
            index_by = [[1 + mod(i, 3) for i in 1:n] for n in lengths]
            original = deepcopy(index_by)
            groups = collect(eachindex(lengths))
            theta = [0.3, -0.2, 0.7]
            k = prepare(grouped_walks; bound = (; index_by, groups))
            @test k(theta) ≈ reference(theta, index_by)
            if v"1.10" <= VERSION < v"1.11" && maximum(lengths) > 256 && !repaired
                @test_skip "ordinary Reverse: third-strip process abort without replayed_cache_limits"
                continue
            end
            ad = prepare_ad(k, AutoEnzyme(; mode = Enzyme.Reverse), theta; active = :theta)
            for parameters in (theta, [-0.4, 0.6, 0.2], theta)
                unchanged = copy(parameters)
                value, gradient = ad_value_and_gradient(ad, parameters)
                @test value ≈ reference(parameters, index_by)
                @test gradient ≈ central(t -> reference(t, index_by), parameters) rtol = 1e-7 atol = 1e-7
                @test parameters == unchanged
                @test index_by == original
            end
        end
    end
end

end
