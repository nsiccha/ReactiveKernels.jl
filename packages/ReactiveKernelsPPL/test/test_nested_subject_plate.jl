module TestNestedSubjectPlate

using Test, ReactiveKernels, ReactiveKernelsPPL, ReactiveKernelsDistributionKernels, Enzyme
using ReactiveKernels.DifferentiationInterface: AutoEnzyme
import Distributions

@kernel signal_path(raw, shift) = begin
    signal = plate(raw) do subject
        subject .+ shift
    end
    return signal
end

@rkppl channel(rows, signal, picks) = begin
    offset ~ Normal(0, 1)
    width = 1.0
    @plate for s in eachindex(rows)
        loc[s] = signal[s][picks[s]] .+ offset
        rows[s] .~ Normal.(loc[s], width)
    end
    return loc
end

@testset "An endpoint submodel observes subjects on their own reading schedules" begin
    model = quote
        shift ~ Normal(0, 1)
        trajectory = signal_path(raw, shift)
        @plate for e in eachindex(readings)
            endpoint[e] ~ channel(readings[e], trajectory, picks[e])
        end
        fitted = endpoint.loc
    end
    small = (; readings=[[[0.1, 0.2], Float64[]], [[0.3], [0.4, 0.5]]],
        raw=[[1., 2.], [3., 4.]], picks=[[[1, 2], Int[]], [[2], [1, 2]]])
    large = (; readings=[Vector{Vector{Float64}}(), [[0.3], Float64[], [0.6]],
            [Float64[], [0.7], [0.3, 0.4]]], raw=[[2.], [0.1, -0.4], [2., 4., 5.]],
        picks=[Vector{Vector{Int}}(), [[1], Int[], [3]], [Int[], [1], [1, 2]]])
    empty = (; readings=empty!(copy(small.readings)), raw=small.raw, picks=empty!(copy(small.picks)))
    plan = lower_rkppl(model, small; mod=@__MODULE__, conditioned=(:readings,))
    programs = Expr[]
    structures = Vector{Tuple{Symbol,Int}}[]
    emitted = Tuple{Int,Int}[]
    nodes(p::ReactiveKernels._PlateProgram) = nodes(p.ast)
    loops(p::ReactiveKernels._PlateProgram) = loops(p.ast)
    nodes(ex) = ex isa Expr ? 1 + sum(nodes, ex.args; init=0) : 1
    loops(ex) = ex isa Expr ? (ex.head in (:for, :while)) + sum(loops, ex.args; init=0) : 0
    sum_subjects(ys) = sum(y -> sum(y; init=0.0), ys; init=0.0)
    for (d, shift, offsets) in ((small, 0.3, [0.25, -0.5]),
            (large, -0.2, [0.8, 0.15, -0.1]), (empty, 0.4, Float64[]))
        bound = bind_data(plan, d)
        built = build_kernel(bound)
        u = [shift; offsets]
        @test length(coordinate_names(built.layout)) == length(u)
        fits = [[(d.raw[s][ps] .+ shift) .+ offsets[e] for (s, ps) in pairs(subjects)]
            for (e, subjects) in pairs(d.picks)]
        residuals = [map(-, y, m) for (y, m) in zip(d.readings, fits)]
        pointwise = [[-0.5 .* (r.^2 .+ log(2pi)) for r in subjects] for subjects in residuals]
        expected = Float64[sum_subjects(p) for p in pointwise]
        query(w) = Base.invokelatest(prepare_query(built, bound, w), u)
        actual_fits = query((:fitted,))
        @test actual_fits == fits
        @test actual_fits isa AbstractVector{<:AbstractVector{<:AbstractVector{Float64}}}
        got = query(:pointwise).readings
        @test got isa AbstractVector{<:AbstractVector{<:AbstractVector{Float64}}}
        @test length.(got) == length.(pointwise)
        for (p, q) in zip(got, pointwise)
            @test length.(p) == length.(q)
            @test all(x ≈ y for (x, y) in zip(p, q))
        end
        cell = prepare_cell_query(built, bound, :readings)
        @test all(Base.invokelatest(cell, u, e) ≈ expected[e] for e in eachindex(expected))
        @test query(:likelihood) ≈ sum(expected)
        backend = AutoEnzyme(; mode=Enzyme.Reverse)
        sampler = prepare_sampler(built, bound, u; backend)
        value, gradient = Base.invokelatest(sampler_value_and_gradient!, sampler, similar(u), u)
        @test value ≈ sum(expected) - 0.5 * sum(abs2.(u) .+ log(2pi))
        scores = Float64[sum_subjects(r) for r in residuals]
        @test gradient ≈ [sum(scores); scores] .- u
        source = Base.remove_linenums!(deepcopy(code_expr(sampler.kernel)))
        push!(emitted, (nodes(source), loops(source)))
        push!(programs, Base.remove_linenums!(kernel_expr(bound, built.layout)))
        push!(structures, [(r.kind, r.depth) for r in recipe_inventory(built.spec)])
    end
    @test all(==(first(programs)), programs)
    @test all(==(first(structures)), structures)
    @test all(==(first(emitted)), emitted)
    @test last(first(emitted)) > 0
    @test (:plate, 2) in first(structures)
end

@kernel score(y, mu, w) = begin
    residual = y - mu
    return -w * residual * residual
end

@rkppl scored(rows, traj, picks, e) = begin
    weight = e
    offset ~ Normal(0, 1)
    @plate for e in axes(rows, 1)
        selected = traj[e][picks[e]]
        fit[e] = selected .+ offset
        rows[e] .~ LogDensity.(score, fit[e], weight)
    end
    return fit
end

@testset "Nested plate locals and caller-owned densities" begin
    data = (; readings=[[[2.5], Float64[]], [[0.6], [2.0, 5.0]]],
        signal=[[0.5, 2.0], [1.0, 5.0]], picks=[[[2], Int[]], [[1], [1, 2]]])
    model = quote
        @plate for e in eachindex(readings)
            endpoint[e] ~ scored(readings[e], signal, picks[e], 2.0)
        end
    end
    bound = bind_data(lower_rkppl(model, data; mod=@__MODULE__, conditioned=(:readings,)), data)
    built = build_kernel(bound)
    q = prepare_query(built, bound, :pointwise)
    measured = Base.invokelatest(q, [0.5, 0.0]).readings
    @test measured[1] == [[0.0], Float64[]]
    @test measured[2][1] ≈ [-0.02]
    @test measured[2][2] ≈ [-2.0, 0.0]
end

@rkppl scaled_channel(rows, signal, picks, sigma) = begin
    offset ~ Normal(0, 1)
    width = sigma
    @plate for s in eachindex(rows)
        loc[s] = signal[s][picks[s]] .+ offset
        rows[s] .~ Normal.(loc[s], width)
    end
    return loc
end

@testset "Nested Normal scales follow their authored observation axes" begin
    data = (; readings=[[[0.1, 0.2], Float64[]], [[0.3], [0.4, 0.5]]],
        signal=[[1., 2.], [3., 4.]], picks=[[[1, 2], Int[]], [[2], [1, 2]]],
        widths=[0.7, 1.2])
    offsets = [0.12, -0.42]
    sum_rows(x) = sum(row -> sum(row; init=0.0), x; init=0.0)
    for (scale, widths) in ((1.0, ones(2)), (:(widths[e]), data.widths))
        model = quote
            @plate for e in eachindex(readings)
                endpoint[e] ~ scaled_channel(readings[e], signal, picks[e], $scale)
            end
            fitted = endpoint.loc
        end
        plan = lower_rkppl(model, data; mod=@__MODULE__, conditioned=(:readings,))
        bound = bind_data(plan, data)
        built = build_kernel(bound)
        fits = [[data.signal[s][j] .+ offsets[e] for (s, j) in pairs(p)]
            for (e, p) in pairs(data.picks)]
        errors = [map(-, ys, ms) for (ys, ms) in zip(data.readings, fits)]
        expected = [[-0.5 .* ((r ./ w).^2 .+ log(2pi)) .- log(w) for r in group]
            for (group, w) in zip(errors, widths)]
        query(w) = Base.invokelatest(prepare_query(built, bound, w), offsets)
        @test query((:fitted,)) == fits
        got = query(:pointwise).readings
        @test all(all(isapprox.(p, q)) for (p, q) in zip(got, expected))
        @test query(:likelihood) ≈ sum(sum_rows, expected)
        sampler = prepare_sampler(built, bound, offsets; backend=AutoEnzyme(; mode=Enzyme.Reverse))
        _, grad = Base.invokelatest(sampler_value_and_gradient!, sampler, similar(offsets), offsets)
        @test grad ≈ sum_rows.(errors) ./ widths.^2 .- offsets
        if scale isa Expr
            @test_throws ContractValidationError bind_data(plan, merge(data, (; widths=[0.7, -1.0])))
        end
    end
end

@rkppl student_channel(rows, df, width) = begin
    offset ~ Normal(0, 1)
    @plate for s in eachindex(rows)
        rows[s] .~ StudentT.(df, offset, width)
    end
    return offset
end

@testset "Nested Student degrees of freedom and scale" begin
    data = (; readings=[[[0.2], Float64[]], [[-0.3, 0.7]]],
        degrees=[3, 7], widths=[0.7, 1.2])
    model = quote
        @plate for e in eachindex(readings)
            endpoint[e] ~ student_channel(readings[e], degrees[e], widths[e])
        end
    end
    plan = lower_rkppl(model, data; mod=@__MODULE__, conditioned=(:readings,))
    bound = bind_data(plan, data)
    query = prepare_query(build_kernel(bound), bound, :pointwise)
    got = Base.invokelatest(query, zeros(2)).readings
    for e in eachindex(data.readings)
        law = Distributions.LocationScale(0.0, data.widths[e], Distributions.TDist(data.degrees[e]))
        @test length(got[e]) == length(data.readings[e])
        for s in eachindex(data.readings[e])
            @test got[e][s] ≈ Distributions.logpdf.(Ref(law), data.readings[e][s])
        end
    end
    @test_throws ContractValidationError bind_data(plan, merge(data, (; degrees=[0, 7])))
end

end
