module PPLKernelCompositionTests
using ReactiveKernels, ReactiveKernelsPPL, DifferentiationInterface, Enzyme, Test
using Distributions: Normal, logpdf

module Models
using ReactiveKernels
@kernel recurrence(xs, gain) = begin
    updates = scan(xs, Ref(gain); init=0.0) do carry, x, g
        next = carry + x * g
        (next, next)
    end
    return updates
end
const alias = recurrence
@kernel panel(x, gain) = begin
    totals = plate(eachcol(x), Ref(gain)) do xs, g
        history = recurrence(xs, g)
        sum(history)
    end
    return totals
end
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

function scan_count(recipes)
    sum(recipes; init=0) do r
        r.op isa ReactiveKernels._AuthoredScanOp && return 1
        r.op isa ReactiveKernels._AuthoredPlateOp && return scan_count(plate_body(r).recipes)
        return 0
    end
end

@testset "PPL module kernels retain their scans and source authority" begin
    for head in (:recurrence, :alias, GlobalRef(Models, :recurrence), :(Models.recurrence))
        counts = Int[]
        for n in (0, 1, 3, 17)
            bound, built, data = build_case(n; head)
            original = deepcopy(data)
            @test scan_count(built.spec.graph.recipes) == 1
            push!(counts, length(built.spec.graph.recipes))
            # Evaluate the very expression exported by the public generator;
            # neither construction path repairs the plan after emission.
            def = kernel_expr(bound, built.layout)
            replayed = ReactiveKernelsPPL._eval_kernel_def(def)
            @test scan_count(replayed.graph.recipes) == 1
            replay = prepare(replayed; bound=data)
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
        @test scan_count(built.spec.graph.recipes) == 1
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
        @test scan_count(built.spec.graph.recipes) == 1
        @test any(built.spec.graph.recipes) do r
            r.op isa ReactiveKernels._AuthoredPlateOp && scan_count(plate_body(r).recipes) == 1
        end
        push!(counts, length(built.spec.graph.recipes))
        sampler = prepare_sampler(built, bound, [0.2]; backend=BACKEND)
        for a in (0.2, -0.4)
            value = Base.invokelatest(sampler.kernel, [a])
            @test value ≈ subject_reference(data, a)
            if n == 0
                # Capability gap: native Enzyme static activity analysis fails
                # for an empty child scan inside the bound eachcol plate.
                @test_broken sampler_value_and_gradient!(sampler, [0.0], [a])[2] ≈
                    [subject_gradient(data, a)]
            else
                _, grad = sampler_value_and_gradient!(sampler, [0.0], [a])
                @test grad ≈ [subject_gradient(data, a)]
            end
        end
        @test data == original
    end
    @test all(==(first(counts)), counts)
end
end
