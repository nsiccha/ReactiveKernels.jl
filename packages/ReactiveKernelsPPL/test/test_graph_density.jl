module GraphDensityTests
using Test, ReactiveKernels, ReactiveKernelsPPL
import DifferentiationInterface as DI
import Enzyme
include("fixtures/graph_density.jl")
using .GraphDensityFixtures

module ImportedDensity
using ReactiveKernelsPPL
import ..GraphDensityFixtures: scalar_score
end

function query(fx, u, cut=:sampler)
    Base.invokelatest(prepare_query(fx.built, fx.bound, cut), u)
end
function independent_gradient(fx, u)
    h = 1e-5
    map(eachindex(u)) do i
        plus, minus = copy(u), copy(u)
        plus[i] += h; minus[i] -= h
        (GraphDensityFixtures.oracle(fx, plus).posterior-
            GraphDensityFixtures.oracle(fx, minus).posterior)/(2h)
    end
end
function prepared_cell_sources(spec)
    [join((string(cell.source) for cell in plate_body(r).recipes), "\n") for r in spec.graph.recipes
        if r.op isa ReactiveKernels._AuthoredPlateOp]
end

@testset "caller graph densities compose in the observation plate" begin
    for head in (:threshold_score, :alias, GlobalRef(GraphDensityFixtures, :threshold_score),
            :(GraphDensityFixtures.threshold_score)), rowwise in (false, true), n in (0, 1, 5, 19)
        fx = GraphDensityFixtures.fixture(n; head, rowwise)
        saved = deepcopy(fx.data)
        source = kernel_expr(fx.bound, fx.built.layout)
        @test !occursin("sampling_logdensity", string(source))
        replayed = ReactiveKernelsPPL._eval_kernel_def(source)
        cell_sources = prepared_cell_sources(fx.built.spec)
        replay_sources = prepared_cell_sources(replayed)
        @test !isempty(cell_sources)
        @test any(s->occursin("value <= lower", s) && occursin("value >= upper", s) &&
            occursin("log", s) && occursin("sqrt", s), cell_sources)
        @test any(s->occursin("value <= lower", s) && occursin("value >= upper", s) &&
            occursin("log", s) && occursin("sqrt", s), replay_sources)
        @test all(s->!occursin("sampling_logdensity", s), cell_sources)
        @test coordinate_names(fx.built.layout) == [:a, :b, :add, :prop]
        sampler = prepare_sampler(fx.built, fx.bound, fx.u;
            backend=DI.AutoEnzyme(;mode=Enzyme.Reverse))
        replay = prepare_query((;spec=replayed, layout=fx.built.layout), fx.bound, :sampler)
        for shift in (0.0, 0.07)
            u = fx.u .+ shift
            before = copy(u)
            expected = GraphDensityFixtures.oracle(fx, u)
            value, gradient = sampler_value_and_gradient!(sampler, similar(u), u)
            @test value ≈ expected.posterior rtol=2e-12
            @test gradient ≈ independent_gradient(fx, u) atol=2e-8 rtol=2e-7
            @test Base.invokelatest(replay, u) ≈ expected.posterior rtol=2e-12
            points = query(fx, u, :pointwise).y
            @test axes(points) == axes(expected.points)
            @test all(isapprox.(points, expected.points; rtol=2e-12))
            @test query(fx, u, :prior) ≈ expected.prior rtol=2e-12
            @test query(fx, u, :likelihood) ≈ expected.likelihood rtol=2e-12
            @test u == before
        end
        @test fx.data == saved
    end
end

@testset "scalar, keyword, extracted endpoint and numerical density paths" begin
    for (rhs, scale_at) in ((:(LogDensity(scalar_score, a, 1+exp(a))), w->1+exp(w)),
            (:(density_alias(extracted_score, a, 1+exp(a))), w->1+exp(w)),
            (:(density_alias(extracted_score, a, 2.0)), w->2.0),
            (:(LogDensity(numerical_score, a; scale=1+exp(a))), w->1+exp(w)))
        data = (;y=0.4)
        bound = bind_data(lower_rkppl(quote a ~ Normal(0,1); y ~ $rhs end,
            data;mod=GraphDensityFixtures,conditioned=(:y,)), data)
        built = build_kernel(bound)
        sampler = prepare_sampler(built, bound, [0.2];backend=DI.AutoEnzyme(;mode=Enzyme.Reverse))
        for a in (0.2, -0.3)
            expected(w) = -log(2pi)/2-w^2/2 +
                GraphDensityFixtures.Distributions.logpdf(
                    GraphDensityFixtures.Distributions.Normal(w, scale_at(w)), data.y)
            u = [a]
            value, gradient = sampler_value_and_gradient!(sampler, similar(u), u)
            @test value ≈ expected(a)
            @test only(gradient) ≈ (expected(a+1e-5)-expected(a-1e-5))/2e-5 atol=2e-8
            @test u == [a]
        end
    end
end

@testset "imported graph density and literal broadcast arguments" begin
    for n in (0, 3)
        data = (;y=[0.2sin(i) for i in 1:n])
        saved = deepcopy(data)
        bound = bind_data(lower_rkppl(quote
            a ~ Normal(0,1)
            y .~ LogDensity.(scalar_score, a, 2.0)
        end, data; mod=ImportedDensity, conditioned=(:y,)), data)
        built = build_kernel(bound)
        @test !occursin("sampling_logdensity", string(kernel_expr(bound, built.layout)))
        @test coordinate_names(built.layout) == [:a]
        sampler = prepare_sampler(built, bound, [0.2];
            backend=DI.AutoEnzyme(;mode=Enzyme.Reverse))
        for a in (0.2, -0.3)
            u = [a]
            value, gradient = sampler_value_and_gradient!(sampler, similar(u), u)
            expected = GraphDensityFixtures.Distributions.logpdf(
                GraphDensityFixtures.Distributions.Normal(), a) +
                sum((GraphDensityFixtures.Distributions.logpdf(
                    GraphDensityFixtures.Distributions.Normal(a,2), y) for y in data.y); init=0.0)
            @test value ≈ expected
            @test only(gradient) ≈ -a+sum((data.y .- a)./4)
            @test u == [a]
        end
        @test data == saved
    end
end
end
