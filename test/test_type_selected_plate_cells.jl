module TypeSelectedPlateCellTests
using ReactiveKernels
using ReactiveKernels: plate
using DifferentiationInterface: AutoEnzyme
import Enzyme
using Test

# A plate cell value that is a branch its operands' concrete types decide,
# Stan's scalar and array `+` written as Julia (`typeof(b) <: Number ? b .+ v :
# b + v`), takes the same arm in every cell of every call. The cell is lowered
# with that arm in its place, so the arm's broadcast stays the lazy broadcast
# its reader consumes, as the arm written alone does. Before, the cell kept the
# branch and its arm materialized `v` and `b .+ v` into fresh arrays in every
# cell: on Julia 1.10.12 and Enzyme dogfood 484b27a4 a native Reverse gradient
# over 100 cells of five entries allocated 42,560 B in 406 allocations, against
# 80 B in one for the arm alone (snag `type-decided-bra-2686ef6f`). The cost
# was in a cell holding a nested plate, the shape RKPPL gives `y[i] .~ D.(m)`:
# here 7,536 B over 20 cells and 30,816 B over 80 against 32 B for the arm. A
# condition the cell's data decide stays a lazy branch.

const TSB_BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)

# The cell reads `m` elementwise through a nested plate (`nested`) or through
# a broadcast (`direct`), the two cell lowerings.
const TSB_DIRECT_BRANCH = @kernel tsb_direct_branch(t, y, θ::Vector{Float64}) = begin
    b = θ[1]
    k = θ[2]
    cells = plate(t, y) do ti, yi
        v = ti .* exp(k)
        m = typeof(b) <: Number ? b .+ v : b + v
        sum(abs2.(yi .- m))
    end
    total = sum(cells)
    return total
end

const TSB_DIRECT_ARM = @kernel tsb_direct_arm(t, y, θ::Vector{Float64}) = begin
    b = θ[1]
    k = θ[2]
    cells = plate(t, y) do ti, yi
        v = ti .* exp(k)
        m = b .+ v
        sum(abs2.(yi .- m))
    end
    total = sum(cells)
    return total
end

# The else arm taken, with a compound condition reading a cell local.
const TSB_NESTED_ELSE = @kernel tsb_nested_else(t, y, θ::Vector{Float64}) = begin
    b = θ[1]
    k = θ[2]
    cells = plate(t, y) do ti, yi
        v = ti .* exp(k)
        m = typeof(b) <: Integer && typeof(v) <: AbstractArray ? v : b .+ v
        terms = plate(yi, m) do yj, mj
            abs2(yj - mj)
        end
        sum(terms)
    end
    total = sum(cells)
    return total
end

const TSB_NESTED_BRANCH = @kernel tsb_nested_branch(t, y, θ::Vector{Float64}) = begin
    b = θ[1]
    k = θ[2]
    cells = plate(t, y) do ti, yi
        v = ti .* exp(k)
        m = typeof(b) <: Number ? b .+ v : b + v
        terms = plate(yi, m) do yj, mj
            abs2(yj - mj)
        end
        sum(terms)
    end
    total = sum(cells)
    return total
end

const TSB_NESTED_ARM = @kernel tsb_nested_arm(t, y, θ::Vector{Float64}) = begin
    b = θ[1]
    k = θ[2]
    cells = plate(t, y) do ti, yi
        v = ti .* exp(k)
        m = b .+ v
        terms = plate(yi, m) do yj, mj
            abs2(yj - mj)
        end
        sum(terms)
    end
    total = sum(cells)
    return total
end

# A condition the cell's data decide: the branch stays lazy, each cell takes
# its own arm, and the arm that would fail on another cell's data never runs.
const TSB_DATA_BRANCH = @kernel tsb_data_branch(t, y, θ::Vector{Float64}) = begin
    b = θ[1]
    k = θ[2]
    cells = plate(t, y) do ti, yi
        v = ti .* exp(k)
        m = isempty(yi) ? yi : (length(yi) > 2 ? b .+ v : v .- b)
        sum(abs2.(yi .- m))
    end
    total = sum(cells)
    return total
end

# Distinct rows per cell, one cell empty: identical rows would run once.
function tsb_data(n)
    lengths = [g == 2 ? 0 : 2 + mod(g, 4) for g in 1:n]
    t = [collect(range(0.5, 2.0; length = l)) .+ 0.01 * g for (g, l) in enumerate(lengths)]
    y = [fill(0.4, l) .+ 0.001 * g for (g, l) in enumerate(lengths)]
    t, y
end

function tsb_oracle(t, y, θ, arm)
    b, k = θ
    sum(sum(abs2.(yi .- arm(b, ti .* exp(k), yi)); init = 0.0)
        for (ti, yi) in zip(t, y); init = 0.0)
end

function tsb_gradient(spec, t, y, θ)
    prepared = prepare(spec; exemplars = (t, y, θ))
    ad = prepare_ad(prepared, TSB_BACKEND, t, y, θ; active = :θ)
    grad = similar(θ)
    value, _ = ad_value_and_gradient!(ad, grad, t, y, θ)
    ad_value_and_gradient!(ad, grad, t, y, θ)
    bytes = @allocated ad_value_and_gradient!(ad, grad, t, y, θ)
    (; value, grad = copy(grad), bytes)
end

@testset "a branch the cell's operand types decide is its taken arm" begin
    θ = [0.3, -0.2]
    # `free`: the arm alone allocates nothing per cell. A broadcast reader's
    # cell materializes `m` and `abs2.(yi .- m)` with or without the branch.
    for (branch, arm, label, free) in (
            (TSB_NESTED_BRANCH, TSB_NESTED_ARM, "nested plate reader", true),
            (TSB_NESTED_ELSE, TSB_NESTED_ARM, "else arm", true),
            (TSB_DIRECT_BRANCH, TSB_DIRECT_ARM, "broadcast reader", false))
        @testset "$label" begin
            small, large = tsb_data(20), tsb_data(80)
            for (t, y) in (small, large)
                expected = tsb_oracle(t, y, θ, (b, v, yi) -> b .+ v)
                taken = tsb_gradient(branch, t, y, θ)
                alone = tsb_gradient(arm, t, y, θ)
                @test taken.value ≈ expected rtol = 1e-12
                @test taken.value == alone.value
                @test taken.grad == alone.grad
                # As many bytes as the arm alone.
                @test taken.bytes == alone.bytes
            end
            free && @test tsb_gradient(branch, small..., θ).bytes ==
                          tsb_gradient(branch, large..., θ).bytes
        end
    end
end

# A short-circuit condition its first operand decides leaves a deleted
# statement before its return; one that reads a value is not decided.
@testset "short-circuit conditions decided by their first operand" begin
    decided(f) = ReactiveKernels._type_constant_bool(f, Any[Float64, Vector{Float64}])
    @test decided((x, y) -> typeof(x) <: Number || typeof(y) <: Number) === true
    @test decided((x, y) -> typeof(x) <: Integer && typeof(y) <: AbstractArray) === false
    @test decided((x, y) -> typeof(x) <: Number && typeof(y) <: AbstractArray) === true
    @test decided((x, y) -> x > 0 || typeof(y) <: Number) === nothing
    @test decided((x, y) -> typeof(x) <: Integer || isempty(y)) === nothing
end

@testset "a branch the cell's data decide stays lazy" begin
    θ = [0.3, -0.2]
    t, y = tsb_data(20)
    data_arm(b, v, yi) = isempty(yi) ? yi : (length(yi) > 2 ? b .+ v : v .- b)
    expected = tsb_oracle(t, y, θ, data_arm)
    prepared = prepare(TSB_DATA_BRANCH; exemplars = (t, y, θ))
    @test prepared(t, y, θ) ≈ expected rtol = 1e-12
    taken = tsb_gradient(TSB_DATA_BRANCH, t, y, θ)
    @test taken.value ≈ expected rtol = 1e-12
    h = 1e-6
    fd = map(1:2) do i
        up, down = copy(θ), copy(θ)
        up[i] += h; down[i] -= h
        (tsb_oracle(t, y, up, data_arm) - tsb_oracle(t, y, down, data_arm)) / 2h
    end
    @test taken.grad ≈ fd rtol = 1e-6
end

end
