module ScanRangeArgumentTests
using ReactiveKernels, DifferentiationInterface, Enzyme, Test

@kernel free_index(xs::Vector{Float64}, selected) = begin
    trajectory = scan(xs; init = 0.5, include_init = true) do carry, x
        next = x > 0 ? sin(x) : x * x
        (next, next)
    end
    window = trajectory[selected]
    return window
end

@kernel prefix_index(xs::Vector{Float64}, selected) = begin
    trajectory = scan(xs; init = 0.5, include_init = true) do carry, x
        next = carry + x
        (next, next)
    end
    window = trajectory[selected]
    return window
end

allocated(f::F, args::Vararg{Any,N}) where {F,N} = @allocated f(args...)
loops(ex) = ex isa Expr ? (ex.head === :for ? 1 : 0) +
    sum(loops, ex.args; init = 0) : 0

@testset "scan reader accepts a runtime range argument" begin
    free, prefix = prepare(free_index), prepare(prefix_index)
    for xs in (Float64[], [-0.2], [-0.2, 0.1, 0.3, -0.4, 0.5])
        original = copy(xs)
        free_values = vcat(0.5, [x > 0 ? sin(x) : x * x for x in xs])
        prefix_values = accumulate(+, xs; init = 0.5)
        prefix_values = vcat(0.5, prefix_values)
        n = length(xs) + 1
        for selected in (1:n, 1:1, n:n, 2:1, 0:-1, n+1:n,
                         1, n, 1:2:n, [n, 1], Colon())
            @test free(xs, selected) ≈ free_values[selected]
            @test prefix(xs, selected) ≈ prefix_values[selected]
        end
        for selected in (0:1, n:n+1)
            @test_throws BoundsError free(xs, selected)
            @test_throws BoundsError prefix(xs, selected)
        end
        @test xs == original
        result = free(xs, 1:n)
        @test !Base.mightalias(result, xs)
        saved = copy(result)
        free(xs, 1:1)
        @test result == saved
    end
    xs = collect(range(-1.0, 1.0; length = 10_000))
    free(xs, 3:5); prefix(xs, 3:5)
    @test allocated(free, xs, 3:5) < 1024
    @test allocated(prefix, xs, 3:5) < 1024
    # Binding changes the values, not the number of emitted recurrence bodies.
    structures = [loops(code_expr(prepare(free_index;
        bound = (; xs = zeros(n), selected = 1:1)))) for n in (0, 7, 1000)]
    @test allequal(structures)
end

@kernel range_objective(q::Vector{Float64}, xs::Vector{Float64}, selected::UnitRange{Int}) = begin
    gain::Float64 = q[1]
    trajectory = scan(xs, Ref(gain); init = q[2], include_init = true) do carry, x, gain
        next = x > 0 ? sqrt(gain * x) : gain * x * x
        (next, next)
    end
    window = trajectory[selected]
    objective::Float64 = sum(abs2, window)
    return objective
end

# Native full-trajectory hand-loop control: retain every step and slice after
# materialization, independently of RK's demand analysis.
function range_reference(q, xs, selected)
    trajectory = Vector{eltype(q)}(undef, length(xs) + 1)
    trajectory[1] = q[2]
    for i in eachindex(xs)
        x = xs[i]
        trajectory[i + 1] = x > 0 ? sqrt(q[1] * x) : q[1] * x * x
    end
    sum(abs2, trajectory[selected])
end

@testset "runtime scan ranges retain ordinary reverse derivatives" begin
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    for (xs, selected) in ((Float64[], 1:1), ([-0.2], 2:2),
                           ([-0.2, 0.1, 0.3, -0.4], 2:4),
                           ([-0.2, 0.1, 0.3, -0.4], 3:2))
        kernel = prepare(range_objective; bound = (; xs, selected))
        q = [0.3, 0.6]
        ad = prepare_ad(kernel, backend, q; active = :q)
        for delta in (0.0, 0.1)
            point = q .+ delta
            value, grad = ad_value_and_gradient(ad, point)
            expected_gradient = zero(point)
            Enzyme.autodiff(Enzyme.Reverse, range_reference, Enzyme.Active,
                           Enzyme.Duplicated(point, expected_gradient),
                           Enzyme.Const(xs), Enzyme.Const(selected))
            @test value ≈ range_reference(point, xs, selected)
            @test grad ≈ expected_gradient atol = 1e-12 rtol = 1e-11
        end
    end
end

@kernel ragged_objective(q::Vector{Float64}, sequences::Vector{Vector{Float64}},
                         windows::Vector{UnitRange{Int}}) = begin
    totals = plate(q, sequences, windows) do gain, xs, selected
        trajectory = scan(xs; init = gain, include_init = true) do carry, x
            next = x > 0 ? sqrt(gain * x) : gain * x * x
            (next, next)
        end
        window = trajectory[selected]
        sum(abs2, window)
    end
    objective::Float64 = sum(totals)
    return objective
end

function ragged_reference(q, sequences, windows)
    total = 0.0
    for i in eachindex(q, sequences, windows)
        gain = q[i]
        xs = sequences[i]
        trajectory = Vector{eltype(q)}(undef, length(xs) + 1)
        trajectory[1] = gain
        for j in eachindex(xs)
            x = xs[j]
            trajectory[j + 1] = x > 0 ? sqrt(gain * x) : gain * x * x
        end
        total += sum(abs2, trajectory[windows[i]])
    end
    total
end

@testset "range arguments inside ragged bound plates" begin
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    sequences = [Float64[], [-0.2], [-0.2, 0.1, 0.3, -0.4]]
    windows = [1:1, 2:1, 2:4]
    structures = Int[]
    for n in (0, 1, 3)
        xs, selected = sequences[1:n], windows[1:n]
        q = fill(0.3, n)
        kernel = prepare(ragged_objective; bound = (; sequences = xs, windows = selected))
        push!(structures, loops(code_expr(kernel)))
        ad = prepare_ad(kernel, backend, q; active = :q)
        value, grad = ad_value_and_gradient(ad, q)
        expected_gradient = zero(q)
        Enzyme.autodiff(Enzyme.Reverse, ragged_reference, Enzyme.Active,
                       Enzyme.Duplicated(q, expected_gradient),
                       Enzyme.Const(xs), Enzyme.Const(selected))
        @test value ≈ ragged_reference(q, xs, selected)
        @test grad ≈ expected_gradient atol = 1e-12 rtol = 1e-11
    end
    @test allequal(structures)
end
end
