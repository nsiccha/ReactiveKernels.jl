using DifferentiationInterface
using Distributions
using Enzyme
using ReactiveKernels
using ReactiveKernelsPPL
using Test

# A response whose scale or shape is a bound number costs no more per call than
# the same response with that parameter sampled, and its support guard stays
# lazy (snag bound-response-s-9739d33d). The guard reads bound data, so
# preparation selects its arm; the arm's work over that number is one value of
# the binding, not a statement of every row.
module BoundResponseScale

using ReactiveKernels
using ReactiveKernelsPPL
using Distributions
using Enzyme
using DifferentiationInterface

const backend = AutoEnzyme(; mode = Enzyme.Reverse)

function bind(source, data; conditioned = (:y,))
    plan = bind_data(lower_rkppl(source, data; conditioned, mod = @__MODULE__), data)
    built = Base.invokelatest(build_kernel, plan)
    (; built, plan)
end

function query(source, data; conditioned = (:y,))
    b = bind(source, data; conditioned)
    Base.invokelatest(prepare_query, b.built, b.plan, :sampler)
end

function gradient_at(source, data, u; conditioned = (:y,))
    b = bind(source, data; conditioned)
    q = Base.invokelatest(prepare_sampler, b.built, b.plan, u; backend)
    g = similar(u)
    value, _ = Base.invokelatest(sampler_value_and_gradient!, q, g, u)
    (; value, gradient = copy(g))
end

cached(q) = Any[r.op.value for r in q.plan.recipes
                if r.op isa ReactiveKernels._BoundConstant &&
                   startswith(String(only(r.outputs).name), "bound_plate_")]
plates(q) = filter(r -> r.op isa ReactiveKernels._AuthoredPlateOp, q.plan.recipes)
code(q) = string(Base.remove_linenums!(deepcopy(ReactiveKernels.code_expr(q))))

end

@testset "a bound response scale runs its arm's work once per binding" begin
    B = BoundResponseScale
    y = collect(range(0.1, 0.9; length = 40))
    u = [0.3]

    # Each row: the family with a bound scale or shape, its reference density,
    # and the parameter's value in the cache.
    families = (
        (name = "Normal", source = :(begin a ~ Normal(0, 1); y .~ Normal.(a, noise) end),
         data = (; y, noise = 0.2), parameter = :noise,
         density = (a, s) -> sum(logpdf.(Normal(a, s), y))),
        (name = "LogNormal", source = :(begin a ~ Normal(0, 1); y .~ LogNormal.(a, noise) end),
         data = (; y, noise = 0.2), parameter = :noise,
         density = (a, s) -> sum(logpdf.(LogNormal(a, s), y))),
        (name = "StudentT", source = :(begin a ~ Normal(0, 1); y .~ StudentT.(4.0, a, noise) end),
         data = (; y, noise = 0.2), parameter = :noise,
         density = (a, s) -> sum(logpdf.(TDist(4.0), (y .- a) ./ s) .- log(s))),
    )

    @testset "$(f.name)" for f in families
        original = deepcopy(f.data)
        q = B.query(f.source, f.data)
        reference(v, s) = logpdf(Normal(0, 1), v) + f.density(v, s)

        # The guard selected its arm, and no cell statement computes the scale's
        # log: it is a cached value of this binding, read at run time.
        @test length(B.plates(q)) == 1
        caches = B.cached(q)
        @test any(v -> v isa Real && v ≈ log(0.2), caches)
        @test !occursin(r"\(\$\(QuoteNode\(log\)\)\)\(var\"#noise", B.code(q))

        @test Base.invokelatest(q, u) ≈ reference(u[1], 0.2)
        r = B.gradient_at(f.source, f.data, u)
        @test r.value ≈ reference(u[1], 0.2)
        h = 1e-6
        @test r.gradient[1] ≈ (reference(u[1] + h, 0.2) - reference(u[1] - h, 0.2)) / 2h rtol = 1e-6

        # A second value of the number is another binding of the same program.
        other = merge(f.data, (; noise = 0.5))
        q2 = B.query(f.source, other)
        @test typeof(q2) === typeof(q)
        @test Base.invokelatest(q2, u) ≈ reference(u[1], 0.5)
        @test Base.invokelatest(q, u) ≈ reference(u[1], 0.2)

        @test f.data == original
    end

    @testset "written parameters" begin
        # Base's `log` over a literal folds in the arm's closure, which stays.
        written = B.query(:(begin a ~ Normal(0, 1); y .~ Normal.(a, 0.2) end), (; y))
        @test isempty(B.cached(written))
        @test length(plate_body(only(B.plates(written))).recipes) == 1
        @test Base.invokelatest(written, u) ≈
              logpdf(Normal(0, 1), u[1]) + sum(logpdf.(Normal(u[1], 0.2), y))
        # A written response's `log(y)` reads bound data alone.
        lognormal = B.query(:(begin a ~ Normal(0, 1); y .~ LogNormal.(a, 0.2) end), (; y))
        @test any(v -> v isa AbstractVector && v ≈ log.(y), B.cached(lognormal))
        @test Base.invokelatest(lognormal, u) ≈
              logpdf(Normal(0, 1), u[1]) + sum(logpdf.(LogNormal(u[1], 0.2), y))
        # A written Student-t normalizer calls `logbeta`, which does not fold.
        student = :(begin a ~ Normal(0, 1); y .~ StudentT.(4.0, a, 0.2) end)
        q = B.query(student, (; y))
        @test any(v -> v isa Real && v ≈ log(0.2), B.cached(q))
        reference(v) = logpdf(Normal(0, 1), v) + sum(logpdf.(TDist(4.0), (y .- v) ./ 0.2) .- log(0.2))
        @test Base.invokelatest(q, u) ≈ reference(u[1])
        r = B.gradient_at(student, (; y), u)
        h = 1e-6
        @test r.gradient[1] ≈ (reference(u[1] + h) - reference(u[1] - h)) / 2h rtol = 1e-6
    end

    @testset "mixtures over a bound scale" begin
        source = :(begin
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            y .~ MixtureModel.(vcat.(Normal.(a, sigma), Normal.(b, sigma)), Ref([0.3, 0.7]))
        end)
        q = B.query(source, (; y, sigma = 0.3))
        @test any(v -> v isa Real && v ≈ log(0.3), B.cached(q))
        position = [0.2, 0.6]
        reference(v) = sum(logpdf.(Normal(0, 1), v)) +
            sum(logpdf.(MixtureModel([Normal(v[1], 0.3), Normal(v[2], 0.3)], [0.3, 0.7]), y))
        @test Base.invokelatest(q, position) ≈ reference(position)
        r = B.gradient_at(source, (; y, sigma = 0.3), position)
        @test r.value ≈ reference(position)
        h = 1e-6
        for k in 1:2
            e = [i == k ? h : 0.0 for i in 1:2]
            @test r.gradient[k] ≈ (reference(position .+ e) - reference(position .- e)) / 2h rtol = 1e-6
        end
    end

    @testset "a sampled scale gives the same density" begin
        bound = B.query(:(begin a ~ Normal(0, 1); y .~ Normal.(a, noise) end), (; y, noise = 0.2))
        sampled = B.query(:(begin a ~ Normal(0, 1); s ~ LogNormal(0, 1); y .~ Normal.(a, s) end), (; y))
        v, s = 0.3, 0.2
        likelihood = sum(logpdf.(Normal(v, s), y))
        @test Base.invokelatest(bound, [v]) ≈ logpdf(Normal(0, 1), v) + likelihood
        @test Base.invokelatest(sampled, [v, log(s)]) ≈
              logpdf(Normal(0, 1), v) + logpdf(LogNormal(0, 1), s) + log(s) + likelihood
        # A live scale stays a guard decided once per call, with its `log` above
        # the lane loop.
        @test occursin("plate_log_scale", B.code(sampled))
    end

    @testset "a per-row bound scale caches each row's log" begin
        noise = collect(range(0.2, 0.4; length = length(y)))
        source = :(begin a ~ Normal(0, 1); y .~ Normal.(a, noise) end)
        q = B.query(source, (; y, noise))
        @test any(v -> v isa AbstractVector && length(v) == length(y) && v ≈ log.(noise),
                  B.cached(q))
        reference(v) = logpdf(Normal(0, 1), v) + sum(logpdf.(Normal.(v, noise), y))
        @test Base.invokelatest(q, u) ≈ reference(u[1])
        r = B.gradient_at(source, (; y, noise), u)
        h = 1e-6
        @test r.gradient[1] ≈ (reference(u[1] + h) - reference(u[1] - h)) / 2h rtol = 1e-6
    end

    # A valid scale computed from data reaches the guard as a bound value too.
    # `bind_data` refuses an invalid one, here and for a bound scale, so RKPPL has
    # no bound invalid scale to take the fallback; the core plate tests cover that.
    @testset "a scale computed from data" begin
        source = :(begin a ~ Normal(0, 1); s = noise - 1.0; y .~ Normal.(a, s) end)
        reference(v, s) = logpdf(Normal(0, 1), v) + sum(logpdf.(Normal(v, s), y))
        q = B.query(source, (; y, noise = 1.5))
        @test any(v -> v isa Real && v ≈ log(0.5), B.cached(q))
        @test Base.invokelatest(q, u) ≈ reference(u[1], 0.5)
        r = B.gradient_at(source, (; y, noise = 1.5), u)
        h = 1e-6
        @test r.gradient[1] ≈ (reference(u[1] + h, 0.5) - reference(u[1] - h, 0.5)) / 2h rtol = 1e-6
    end

    # The arm's cell holds several recipes and reads the cached value, so a query
    # of part of the plate must read that value as well.
    @testset "partition and cell queries" begin
        subject = repeat(1:4; inner = 3)
        response = cos.(eachindex(subject))
        data = (; subject, y = response, noise = 0.3)
        source = :(begin
            z[levels(subject)] .~ Normal.(0, 1)
            y .~ Normal.(z[subject], noise)
        end)
        b = B.bind(source, data)
        position = collect(range(-0.3, 0.4; length = 4))
        density(label) = sum(logpdf.(Normal(position[label], 0.3), response[subject .== label]))
        partition = Base.invokelatest(prepare_partition, b.built, b.plan, :y; by = :subject)
        sampler = Base.invokelatest(prepare_partition_sampler, b.built, b.plan, :y, position;
            by = :subject, backend = B.backend, compact = false)
        gradient = similar(position)
        for label in 1:4
            @test Base.invokelatest(partition, position, label) ≈ density(label)
            value, _ = Base.invokelatest(partition_value_and_gradient!, sampler, gradient,
                                         position, label)
            @test value ≈ density(label)
            expected = zeros(4)
            expected[label] = sum(response[subject .== label] .- position[label]) / 0.3^2
            @test gradient ≈ expected
        end
        cell = Base.invokelatest(prepare_cell_query, b.built, b.plan, :y)
        for i in eachindex(subject)
            @test Base.invokelatest(cell, position, i) ≈
                  logpdf(Normal(position[subject[i]], 0.3), response[i])
        end
    end

    @testset "the program does not grow with the rows" begin
        shapes = map((10, 1000)) do n
            ys = collect(range(0.1, 0.9; length = n))
            q = B.query(:(begin a ~ Normal(0, 1); y .~ Normal.(a, noise) end), (; y = ys, noise = 0.2))
            @test Base.invokelatest(q, u) ≈
                  logpdf(Normal(0, 1), u[1]) + sum(logpdf.(Normal(u[1], 0.2), ys))
            (length(q.plan.recipes),
             length(plate_body(only(B.plates(q))).recipes))
        end
        @test shapes[1] == shapes[2]
    end
end
