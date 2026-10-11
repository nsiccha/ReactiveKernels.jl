module PPLKernelCompositionTests
using ReactiveKernels, ReactiveKernelsPPL, DifferentiationInterface, Enzyme, Test
using Distributions: Normal, Exponential, logpdf

module Models
using ReactiveKernels, ReactiveKernelsPPL
@rkppl scaled_values(sd, z) = begin
    z .* sd[1]
end
read_column(values, index) = values[index, 1]
@kernel ragged_cell(xs, gain) = begin
    return xs * gain
end
@kernel ragged_reader(n, xs, gains) = begin
    cells = plate(1:n) do i
        cell_input = xs[i]
        cell_gain = gains[i]
        ragged_cell(cell_input, cell_gain)
    end
    result = convert(Vector{Float64}, reduce(vcat, cells; init=Float64[]))
    return result
end
@kernel recurrence(xs, gain) = begin
    updates = scan(xs; init=0.0) do carry, x
        next = carry + x * gain
        (next, next)
    end
    return updates
end
const alias = recurrence
@kernel panel(x, gain) = begin
    totals = plate(eachcol(x)) do xs
        history = recurrence(xs, gain)
        sum(history)
    end
    return totals
end
@kernel basis_methods(axis::Vector{Float64}) = begin
    center = sum(axis) / length(axis)
    width = 1.5 * maximum(abs.(axis .- center))
    omega2 = plate(1:3, Ref(width)) do mode, width
        (mode * pi / (2width))^2
    end
    rows = plate(axis, Ref(omega2), Ref(width), Ref(center)) do x, omega2, width, center
        values = plate(1:3, Ref(x), Ref(omega2), Ref(width), Ref(center)) do b, x, omega2, width, center
            sin(sqrt(omega2[b]) * (x - center + width)) / sqrt(width)
        end
        values
    end
    PHI = stack(rows; dims=1)
    basis_matrix() = PHI
    squared_frequencies() = omega2
end
const basis_alias = basis_methods

end

const BACKEND = AutoEnzyme(; mode=Enzyme.Reverse)

function build_case(n; head=:recurrence, grouped=false)
    x = sin.(1:n)
    y = cos.(1:n) .* 0.2
    ast = grouped ? quote
        a ~ Normal(0, 0.7)
        @plate for i in eachindex(y)
            path = recurrence(x, a)
            y[i] ~ Normal(path[i], 0.8)
        end
    end : quote
        a ~ Normal(0, 0.7)
        loc = $head(x, a)
        y .~ Normal.(loc, 0.8)
    end
    data = (; x, y)
    bound = bind_data(lower_rkppl(ast, data; conditioned=(:y,), mod=Models), data)
    built = build_kernel(bound)
    return bound, built, data
end

reference(data, a) = logpdf(Normal(0, 0.7), a) +
    sum(logpdf.(Normal.(cumsum(data.x .* a), 0.8), data.y))
gradient(data, a) = -a / 0.7^2 +
    sum((data.y .- cumsum(data.x .* a)) .* cumsum(data.x)) / 0.8^2

function build_subject_case(n, groups)
    data = (; x=reshape(sin.(1:n*groups), n, groups), y=cos.(1:groups) .* 0.2)
    ast = quote
        a ~ Normal(0, 0.7)
        loc = panel(x, a)
        y .~ Normal.(loc, 0.8)
    end
    bound = bind_data(lower_rkppl(ast, data; conditioned=(:y,), mod=Models), data)
    return bound, build_kernel(bound), data
end

subject_weights(data) = [sum(cumsum(xs)) for xs in eachcol(data.x)]
subject_reference(data, a) = logpdf(Normal(0, 0.7), a) +
    sum(logpdf.(Normal.(a .* subject_weights(data), 0.8), data.y))
subject_gradient(data, a) = -a / 0.7^2 +
    sum((data.y .- a .* subject_weights(data)) .* subject_weights(data)) / 0.8^2

# Structure checks read the public inspection contract: `recipe_inventory`
# walks every plate and scan body recursively and `recipe_kind` classifies.
scan_count(program) = count(entry -> entry.kind === :scan, recipe_inventory(program))
structure(program) = [(entry.kind, entry.depth) for entry in recipe_inventory(program)
                      if entry.kind !== :ordinary]

@testset "PPL module kernels retain their scans and source authority" begin
    for head in (:recurrence, :alias, GlobalRef(Models, :recurrence), :(Models.recurrence))
        counts = Int[]
        for n in (0, 1, 3, 17)
            bound, built, data = build_case(n; head)
            original = deepcopy(data)
            @test scan_count(built.spec) == 1
            push!(counts, length(built.spec.graph.recipes))
            # Evaluate the very expression exported by the public generator;
            # neither construction path repairs the plan after emission.
            def = kernel_expr(bound, built.layout)
            replayed = ReactiveKernelsPPL._eval_kernel_def(def)
            @test scan_count(replayed) == 1
            replay = prepare(replayed; bound=data)
            # Once the data are bound, the residual plan keeps the scan over a
            # nonempty sequence; preparation folds an empty sequence's output.
            n > 0 && @test scan_count(replay) == 1
            sampler = prepare_sampler(built, bound, [0.2]; backend=BACKEND)
            for a in (0.2, -0.4)
                u = [a]
                value, grad = sampler_value_and_gradient!(sampler, similar(u), u)
                @test value ≈ reference(data, a) rtol=1e-12
                @test grad ≈ [gradient(data, a)] rtol=1e-10 atol=1e-12
                @test Base.invokelatest(replay, u) ≈ value rtol=1e-12
                @test u == [a]
            end
            @test data == original
        end
        @test all(==(first(counts)), counts)
    end
end

@testset "PPL plate cells retain module kernel scans" begin
    for n in (3, 11)
        bound, built, data = build_case(n; grouped=true)
        @test scan_count(built.spec) == 1
        sampler = prepare_sampler(built, bound, [0.2]; backend=BACKEND)
        value, grad = sampler_value_and_gradient!(sampler, [0.0], [0.2])
        @test value ≈ reference(data, 0.2)
        @test grad ≈ [gradient(data, 0.2)]
    end
end

@testset "PPL subject plates contain the original child scan" begin
    counts = Int[]
    for (n, groups) in ((0, 2), (3, 2), (11, 5))
        bound, built, data = build_subject_case(n, groups)
        original = deepcopy(data)
        # The subject plate holds the child scan in its body in the built
        # graph, its printed-source replay, and once the data are bound. The
        # second top-level plate is the observation plate.
        subject_structure = [(:plate, 0), (:scan, 1), (:plate, 0)]
        @test structure(built.spec) == subject_structure
        entries = recipe_inventory(built.spec)
        child = only(entry for entry in entries if entry.kind === :scan)
        subject = entries[child.parent]
        @test subject.kind === :plate && subject.depth == 0
        @test count(recipe -> recipe_kind(recipe) === :scan,
                    plate_body(subject.recipe).recipes) == 1
        replayed = ReactiveKernelsPPL._eval_kernel_def(kernel_expr(bound, built.layout))
        @test structure(replayed) == subject_structure
        @test structure(prepare(replayed; bound=data)) == subject_structure
        push!(counts, length(built.spec.graph.recipes))
        sampler = prepare_sampler(built, bound, [0.2]; backend=BACKEND)
        @test structure(sampler.kernel) == subject_structure
        for a in (0.2, -0.4)
            value = Base.invokelatest(sampler.kernel, [a])
            @test value ≈ subject_reference(data, a)
            # Ordinary native Reverse, including an empty child scan (n == 0)
            # inside the bound eachcol plate.
            _, grad = sampler_value_and_gradient!(sampler, [0.0], [a])
            @test grad ≈ [subject_gradient(data, a)]
        end
        @test data == original
    end
    @test all(==(first(counts)), counts)
end
function grouped_case(n)
    xs = [i == 2 ? Float64[] : sin.(1:mod(i, 4)+1) for i in 1:n]
    index = reverse(collect(1:n))
    data = (; n, xs, index, y=cos.(1:sum(length, xs)) .* 0.2)
    ast = quote
        beta ~ Normal(0, 0.7)
        sd[1:1] .~ Exponential.(0.9)
        z[levels(index), 1:1] .~ Normal.(0, 1)
        draws ~ scaled_values(sd, z)
        gains = beta .* ones(n) .+ read_column(draws, index)
        loc = ragged_reader(n, xs, gains)
        y .~ Normal.(loc, 0.8)
    end
    bound = bind_data(lower_rkppl(ast, data; conditioned=(:y,), mod=Models), data)
    bound, build_kernel(bound), data
end

function grouped_reference(data, u)
    beta, sd = u[1], exp(u[2])
    z = u[3:end]
    value = logpdf(Normal(0, 0.7), beta) +
        logpdf(Exponential(0.9), sd) + sum(logpdf.(Normal(), z)) + u[2]
    grad = vcat(-beta / 0.7^2, 1 - sd / 0.9, -z)
    row = 0
    for i in eachindex(data.xs), x in data.xs[i]
        row += 1
        zi = data.index[i]
        mu = x * (beta + sd * z[zi])
        value += logpdf(Normal(mu, 0.8), data.y[row])
        residual = (data.y[row] - mu) * x / 0.8^2
        grad[1] += residual
        grad[2] += residual * sd * z[zi]
        grad[2 + zi] += residual * sd
    end
    value, grad
end

@testset "computed submodel results enter transparent ragged readers" begin
    counts = Int[]
    for n in (3, 7, 19)
        bound, built, data = grouped_case(n)
        original = deepcopy(data)
        @test built.layout.total == n + 2
        # The subject plate: its domain and the two vectors its cells read.
        subject_plate = only(filter(r -> recipe_kind(r) === :plate &&
            length(r.inputs) == 3, built.spec.graph.recipes))
        @test length(plate_body(subject_plate).recipes) == 3
        push!(counts, length(built.spec.graph.recipes))
        # The public pre-build diagnostic needs no successfully built spec.
        def = kernel_expr(bound, assign_layout(bound))
        replayed = ReactiveKernelsPPL._eval_kernel_def(def)
        replay = prepare_query((; spec=replayed, layout=built.layout), bound, :sampler)
        u = vcat(0.2, -0.3, sin.(1:n) .* 0.3)
        sampler = prepare_sampler(built, bound, u; backend=BACKEND)
        for shift in (0.0, -0.4)
            input = u .+ shift
            saved = copy(input)
            expected, expected_grad = grouped_reference(data, input)
            value, grad = sampler_value_and_gradient!(sampler, similar(input), input)
            @test value ≈ expected rtol=1e-12
            @test grad ≈ expected_grad rtol=1e-10 atol=1e-12
            @test Base.invokelatest(replay, input) ≈ value rtol=1e-12
            @test input == saved
        end
        @test data == original
    end
    @test all(==(first(counts)), counts)
end

function method_case(data; head=:basis_methods, live=true)
    axis = live ? :(exp.(a .+ t)) : :t
    matrix_call = Expr(:call, Expr(:., Expr(:call, head, :x),
        QuoteNode(:basis_matrix)))
    frequency_call = Expr(:call, Expr(:., Expr(:call, head, :x),
        QuoteNode(:squared_frequencies)))
    ast = quote
        a ~ Normal(0, 1)
        x = $axis
        P = $matrix_call
        omega2 = $frequency_call
        z[1:3] .~ Normal.(0, 1)
        mu = P * (omega2 .* z)
        y .~ Normal.(mu, 0.7)
    end
    plan = lower_rkppl(ast, data; conditioned=(:y,), mod=Models)
    bound = bind_data(plan, data)
    return plan, bound, build_kernel(bound)
end

function method_reference(data, u; live=true)
    a, z = u[1], u[2:4]
    axis = live ? exp.(a .+ data.t) : data.t
    center = sum(axis) / length(axis)
    width = 1.5 * maximum(abs.(axis .- center))
    omega2 = [(mode * pi / (2width))^2 for mode in 1:3]
    P = [sin(sqrt(omega2[b]) * (x - center + width)) / sqrt(width)
         for x in axis, b in 1:3]
    mu = P * (omega2 .* z)
    return logpdf(Normal(), a) + sum(logpdf.(Normal(), z)) +
        sum(logpdf.(Normal.(mu, 0.7), data.y))
end

@testset "PPL composes constructed object endpoints" begin
    data = (; t=[0.5, 0.9, 1.4, 2.1], y=[0.2, -0.1, 0.4, 0.3])
    original = deepcopy(data)
    u = [0.1, -0.2, 0.3, 0.15]
    for head in (:basis_methods, :basis_alias, :(Models.basis_methods)),
        live in (true, false)
        plan, bound, built = method_case(data; head, live)
        @test any(entry -> entry.kind === :plate && entry.depth == 1,
            recipe_inventory(built.spec))
        sampler = prepare_sampler(built, bound, u; backend=BACKEND)
        value, grad = sampler_value_and_gradient!(sampler, similar(u), u)
        oracle = v -> method_reference(data, v; live)
        @test value ≈ oracle(u) rtol=1e-11
        for j in eachindex(u)
            step = 1e-6
            lo, hi = copy(u), copy(u)
            lo[j] -= step
            hi[j] += step
            @test grad[j] ≈ (oracle(hi) - oracle(lo)) / (2step) rtol=1e-5 atol=1e-7
        end
        @test data == original
        if head === :basis_methods && !live
            changed = (; t=data.t .+ [0.1, 0.0, -0.1, 0.2], y=data.y)
            rebound = bind_data(plan, changed)
            resampled = prepare_sampler(build_kernel(rebound), rebound, u;
                backend=BACKEND)
            rebound_value, _ = sampler_value_and_gradient!(resampled,
                similar(u), u)
            @test rebound_value ≈ method_reference(changed, u; live=false) rtol=1e-11
            @test rebound_value != value
        end
    end
end

end
