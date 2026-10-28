module ArrayLiteralTests
using Test, ReactiveKernels, ReactiveKernelsPPL
import DifferentiationInterface as DI
import Enzyme

normal_logpdf(x, m, s=1.0) = -log(2pi)/2 - log(s) - ((x-m)/s)^2/2

function fixture(ast, data; conditioned=(:y,))
    plan = lower_rkppl(ast, data; mod=@__MODULE__, conditioned)
    bound = bind_data(plan, data)
    built = build_kernel(bound)
    u = zeros(built.layout.total)
    likelihood = prepare_query(built, bound, :likelihood)
    pointwise = prepare_query(built, bound, :pointwise)
    sampler = prepare_sampler(built, bound, u;
        backend=DI.AutoEnzyme(; mode=Enzyme.Reverse))
    return (; plan, bound, built, likelihood, pointwise, sampler)
end

function value_query(fx, name)
    names = sort!(collect(keys(fx.bound.columns)))
    bound = NamedTuple{Tuple(names)}(Tuple(fx.bound.columns[n] for n in names))
    Base.invokelatest(prepare, fx.built.spec;
        have=(:unconstrained,names...), want=name, bound)
end

function compare(left, right, points, oracle)
    @test coordinate_names(left.built.layout) == coordinate_names(right.built.layout)
    for u in points
        expected = oracle(constrain(left.built.layout, u))
        @test Base.invokelatest(left.likelihood, u) ≈ expected rtol=1e-12
        @test Base.invokelatest(right.likelihood, u) ≈ expected rtol=1e-12
        gl, gr = similar(u), similar(u)
        vl, _ = sampler_value_and_gradient!(left.sampler, gl, u)
        vr, _ = sampler_value_and_gradient!(right.sampler, gr, u)
        @test vl ≈ vr rtol=1e-13
        @test gl ≈ gr rtol=1e-12 atol=1e-12
        for i in eachindex(u)
            plus, minus = copy(u), copy(u)
            plus[i] += 1e-6
            minus[i] -= 1e-6
            expected_gradient = (Base.invokelatest(left.sampler, plus) -
                Base.invokelatest(left.sampler, minus)) / 2e-6
            @test gl[i] ≈ expected_gradient rtol=1e-6 atol=1e-6
        end
    end
end

@testset "model-level literals preserve vector and block semantics" begin
    for (literal, call, value) in (
        (:([a, b]), :(Base.vect(a, b)), c -> [c.a,c.b]),
        (:([[a,b] z]), :(hcat([a,b],z)), c -> hcat([c.a,c.b],c.z)),
        (:([z; a]), :(vcat(z, a)), c -> vcat(c.z,c.a)),
        (:([a b; b a]), :(hvcat((2,2), a,b,b,a)), c -> [c.a c.b;c.b c.a]),
        (:([[a,b] z; z [b,a]]), :(hvcat((2,2), [a,b],z,z,[b,a])),
            c -> [[c.a,c.b] c.z; c.z [c.b,c.a]]),
        (:([[a,b], [b,a]]), :(Base.vect(Base.vect(a,b), Base.vect(b,a))),
            c -> [[c.a,c.b],[c.b,c.a]]),
    )
        model(ex) = quote
            a ~ Normal(0,1)
            b ~ Normal(0,1)
            z[1:2] .~ Normal.(0,1)
            A = $ex
            loc = sum(sum.(A))
            y .~ Normal.(loc,1)
        end
        data = (;y=[-0.3,0.2,0.8])
        left, right = fixture(model(literal), data), fixture(model(call), data)
        points = ([0.3,-0.2,0.5,0.1],[-0.7,0.6,-0.4,0.9])
        for fx in (left,right), u in points
            actual = Base.invokelatest(value_query(fx,:A),u)
            expected = value(constrain(fx.built.layout,u))
            @test actual == expected
            @test size(actual) == size(expected)
        end
        compare(left, right, points,
            c -> sum(normal_logpdf.(data.y, sum(sum.(value(c))))))
    end
end

@testset "bracket matrices in named and inline products" begin
    for (n, k) in ((3, 2), (5, 3)), named in (false, true)
        u = reshape(collect(range(-0.6, 0.8; length=n*k)), n, k)
        data = (;u, y=collect(range(-0.2, 0.7; length=n)))
        function model(ex)
            product = named ? :(begin
                W = $ex
                R = u * W
            end) : :(begin
                R = u * $ex
            end)
            quote
                w1[axes(u,2)] .~ Normal.(0,1)
                w2[axes(u,2)] .~ Normal.(0,1)
                $(product.args...)
                E = exp.(R)
                P = E ./ sum.(eachrow(E))
                y .~ Normal.(P[:,1],1)
            end
        end
        left = fixture(model(:([w1 w2])),data)
        right = fixture(model(:(hcat(w1,w2))),data)
        points = (collect(range(-0.4,0.7;length=2k)),
            collect(range(0.6,-0.2;length=2k)))
        compare(left,right,points,c -> begin
            E = exp.(u * hcat(c.w1,c.w2))
            sum(normal_logpdf.(data.y,(E ./ sum(E;dims=2))[:,1]))
        end)
    end
end

@testset "brackets retain data and live design-matrix behavior" begin
    x = collect(range(-0.5,0.8;length=5))
    data = (;x,y=[-0.3,0.1,0.8,0.7,0.4])
    for live in (false,true)
        literal = live ? :([ones(length(x)) x exp.(a .* x)]) :
            :([ones(length(x)) x])
        call = live ? :(hcat(ones(length(x)),x,exp.(a .* x))) :
            :(hcat(ones(length(x)),x))
        qualified = Expr(:call,:(Base.hcat),call.args[2:end]...)
        model(ex) = quote
            a ~ Normal(0,1)
            X = $ex
            b[axes(X,2)] .~ Normal.(0,1)
            y .~ Normal.(a .+ X * b,1)
        end
        left = fixture(model(literal),data)
        points = (fill(0.2,live ? 4 : 3),fill(-0.3,live ? 4 : 3))
        oracle(c) = begin
            X = live ? hcat(ones(length(x)),x,exp.(c.a .* x)) :
                hcat(ones(length(x)),x)
            sum(normal_logpdf.(data.y,c.a .+ X*c.b))
        end
        for spelling in (call,qualified)
            compare(left,fixture(model(spelling),data),points,oracle)
        end
    end
end

@testset "numeric literal values retain Julia matrix semantics" begin
    for (literal, call, value) in (
        (:([tau kappa]), :(hcat(tau,kappa)), [0.2 0.7]),
        (:([0.0 1.0; 2.0 3.0]), :(hvcat((2,2),0.0,1.0,2.0,3.0)),
            [0.0 1.0;2.0 3.0]),
    )
        model(ex) = quote
            m ~ Normal(0,1)
            A = $ex
            loc = m + sum(A)
            y .~ Normal.(loc,1)
        end
        data = (;tau=0.2,kappa=0.7,y=[-0.3,0.8])
        compare(fixture(model(literal),data),fixture(model(call),data),
            ([0.3],[-1.2]),c -> sum(normal_logpdf.(data.y,c.m+sum(value))))
    end
end

@testset "array-literal observations bind once with Julia axes" begin
    for n in (0, 3, 7)
        v = collect(range(0.1,0.7;length=n))
        w = collect(range(-0.2,0.3;length=n))
        data = (;v,w)
        original = deepcopy(data)
        for (literal, call, value) in (
            (:([sum(v),sum(w)]), :(Base.vect(sum(v),sum(w))), [sum(v),sum(w)]),
            (:([v;w]), :(vcat(v,w)), vcat(v,w)),
            (:([v w]), :(hcat(v,w)), hcat(v,w)),
            (:([sum(v) 1.0; 2.0 sum(w)]), :(hvcat((2,2),sum(v),1.0,2.0,sum(w))),
                [sum(v) 1.0;2.0 sum(w)]),
        )
            model(ex) = quote
                m ~ Normal(0,5)
                ys = $ex
                ys .~ Normal.(m,1)
            end
            left = fixture(model(literal), data;conditioned=(:ys,))
            right = fixture(model(call), data;conditioned=(:ys,))
            direct = fixture(quote
                m ~ Normal(0,5)
                ys .~ Normal.(m,1)
            end, (;ys=value);conditioned=(:ys,))
            @test left.bound.columns[:ys] == value
            @test size(left.bound.columns[:ys]) == size(value)
            compare(left,right,([0.3],[-1.2]), c -> sum(normal_logpdf.(value,c.m)))
            compare(left,direct,([0.3],[-1.2]), c -> sum(normal_logpdf.(value,c.m)))
            @test size(Base.invokelatest(left.pointwise,[0.3]).ys) == size(value)
            @test data == original
        end
    end
end

const calls = Ref(0)
counted_sum(x) = (calls[] += 1; sum(x))
@testset "literal response preprocessing is not repeated by prepared queries" begin
    calls[] = 0
    data = (;v=[0.2,0.5], w=[-0.1,0.7])
    ast = quote
        m ~ Normal(0,1)
        ys = [counted_sum(v),counted_sum(w)]
        ys .~ Normal.(m,1)
    end
    fx = fixture(ast,data;conditioned=(:ys,))
    @test calls[] == 2
    for m in (0.3,-1.2)
        Base.invokelatest(fx.likelihood,[m])
        sampler_value_and_gradient!(fx.sampler,zeros(1),[m])
    end
    @test calls[] == 2
    rebound = bind_data(fx.plan, (;v=[1.0],w=[2.0]))
    @test calls[] == 4
    @test rebound.columns[:ys] == [1.0,2.0]
end
end
