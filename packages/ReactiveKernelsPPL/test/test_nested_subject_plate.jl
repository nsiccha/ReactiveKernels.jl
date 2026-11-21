module TestNestedSubjectPlate

using Test, ReactiveKernels, ReactiveKernelsPPL, ReactiveKernelsDistributionKernels, Enzyme
using ReactiveKernels.DifferentiationInterface: AutoEnzyme

@kernel signal_path(raw, shift) = begin
    signal = plate(raw) do subject
        subject .+ shift
    end
    return signal
end

@rkppl channel(rows, signal, picks) = begin
    offset ~ Normal(0, 1)
    @plate for s in eachindex(rows)
        loc[s] = signal[s][picks[s]] .+ offset
        rows[s] .~ Normal.(loc[s], 1.0)
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
        expected = sum_subjects.(pointwise)
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
        scores = sum_subjects.(residuals)
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

end
