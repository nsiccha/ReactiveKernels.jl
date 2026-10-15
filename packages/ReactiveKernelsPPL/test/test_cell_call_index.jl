module CellCallIndexTests
using ReactiveKernels, ReactiveKernelsPPL, DifferentiationInterface, Enzyme, Test
import Distributions as D

column_sums(A) = vec(sum(A; dims=1))
column_pair(A) = (A[:, 1], A[:, 2])
const BACKEND = AutoEnzyme(; mode=Enzyme.Reverse)

plate_expr(cells...) = Expr(:macrocall, Symbol("@plate"), LineNumberNode(0),
    Expr(:for, :(i = eachindex(y)), Expr(:block, cells...)))

function differences(f, p)
    h = cbrt(eps(Float64))
    map(eachindex(p)) do i
        up, down = copy(p), copy(p)
        up[i] += h
        down[i] -= h
        (f(up) - f(down)) / (2h)
    end
end

function fixture(expr; site=:definition, n=4, rows=3, values=false)
    data = (; x=[0.2cos(i) for i in 1:n], y=[0.3sin(i) for i in 1:n],
        domain=zeros(rows))
    declarations = [:(W[axes(domain, 1), 1:2] .~ Normal.(0, 1))]
    if site === :hoisted
        append!(declarations, [:(k = $expr), plate_expr(:(r[i] = x[i] * k)),
            :(y .~ Normal.(r, 1))])
    elseif site === :local
        append!(declarations, [plate_expr(:(k = $expr), :(r[i] = x[i] * k)),
            :(y .~ Normal.(r, 1))])
    elseif site === :observation
        push!(declarations, plate_expr(:(y[i] ~ Normal(x[i] * $expr, 1))))
    elseif site === :shared_observation
        push!(declarations, plate_expr(:(y[i] ~ Normal($expr, 1))))
    elseif site === :prior
        append!(declarations, [plate_expr(:(z[i] ~ Normal($expr, 1))),
            :(y .~ Normal.(z, 1))])
    elseif site === :shared_output
        append!(declarations, [plate_expr(:(r[i] = $expr)), :(y .~ Normal.(r, 1))])
    else
        append!(declarations, [plate_expr(:(r[i] = x[i] * $expr)),
            :(y .~ Normal.(r, 1))])
    end
    ast = Expr(:block, declarations...)
    plan = lower_rkppl(ast, values ? data : Tuple(keys(data));
        mod=@__MODULE__, conditioned=(:y,))
    bound = bind_data(plan, data)
    built = build_kernel(bound)
    p = [0.1sin(i) for i in 1:built.layout.total]
    sampler = prepare_sampler(built, bound, p; backend=BACKEND)
    function oracle(p)
        q = constrain(built.layout, p)
        k = sum(q.W[:, 2])
        m = site in (:shared_observation, :shared_output) ? fill(k, n) : data.x .* k
        prior = sum(D.logpdf.(D.Normal(), q.W))
        if site === :prior
            prior += sum(D.logpdf.(D.Normal(k, 1), q.z))
            m = q.z
        end
        prior + sum(D.logpdf.(D.Normal.(m, 1), data.y)) + logjac(built.layout, p)
    end
    (; data, plan, bound, built, p, sampler, oracle, site)
end

function check(fx; query=false)
    (; data, built, p, sampler, oracle, site) = fx
    saved = deepcopy(data)
    reader = query ? Base.invokelatest(prepare, built.spec;
        have=(:unconstrained, keys(data)...), want=:r, bound=data) : nothing
    for shift in (0.0, 0.35)
        v = p .+ shift
        original = copy(v)
        @test sampler(v) ≈ oracle(v)
        value, grad = sampler_value_and_gradient!(sampler, similar(v), v)
        @test value ≈ oracle(v)
        @test grad ≈ differences(oracle, v) rtol=1e-5 atol=1e-7
        @test v == original
        if query
            q = constrain(built.layout, v)
            k = sum(q.W[:, 2])
            expected = site === :shared_output ? fill(k, length(data.x)) : data.x .* k
            @test Base.invokelatest(reader, v) ≈ expected
        end
    end
    @test data == saved
end

@testset "literal indexing of whole call results in plate cells" begin
    exprs = (:(column_sums(W)[2]), :(sum(W; dims=1)[2]),
        :(Base.sum(W; dims=1)[2]), :(sum(W; dims=1)[1, 2]))
    for expr in exprs
        fx = fixture(expr)
        check(fx; query=true)
        control = fixture(expr; site=:hoisted)
        actual = sampler_value_and_gradient!(fx.sampler, similar(fx.p), fx.p)
        expected = sampler_value_and_gradient!(control.sampler, similar(control.p), control.p)
        @test actual[1] ≈ expected[1]
        @test actual[2] ≈ expected[2]
    end
    for site in (:local, :observation, :shared_observation, :prior, :shared_output)
        check(fixture(:(column_sums(W)[2]); site); query=site in (:local, :shared_output))
    end
    for n in (0, 1, 9), rows in (2, 4)
        check(fixture(:(sum(W; dims=1)[1, 2]); n, rows, values=true); query=true)
    end
end

@testset "indexed call results retain their value rank" begin
    for rows in (2, 4)
        data = (; x=[0.2, -0.1, 0.7], y=[0.3, 0.1, -0.2], domain=zeros(rows))
        ast = quote
            W[axes(domain, 1), 1:2] .~ Normal.(0, 1)
            @plate for i in eachindex(y)
                v[i] = x[i] * column_pair(W)[2]
                r[i] = sum(v[i])
            end
            y .~ Normal.(r, 1)
        end
        bound = bind_data(lower_rkppl(ast, data; mod=@__MODULE__, conditioned=(:y,)), data)
        built = build_kernel(bound)
        p = [0.1sin(i) for i in 1:built.layout.total]
        sampler = prepare_sampler(built, bound, p; backend=BACKEND)
        function oracle(p)
            q = constrain(built.layout, p)
            sum(D.logpdf.(D.Normal(), q.W)) +
                sum(D.logpdf.(D.Normal.(data.x .* sum(q.W[:, 2]), 1), data.y))
        end
        @test sampler(p) ≈ oracle(p)
        value, grad = sampler_value_and_gradient!(sampler, similar(p), p)
        @test value ≈ oracle(p)
        @test grad ≈ differences(oracle, p) rtol=1e-5 atol=1e-7
        q = constrain(built.layout, p)
        read_v = Base.invokelatest(prepare, built.spec; have=(:unconstrained, keys(data)...), want=:v, bound=data)
        got = Base.invokelatest(read_v, p)
        @test length(got) == length(data.x)
        @test all(size(v) == (rows,) for v in got)
        @test all(got[i] ≈ data.x[i] .* q.W[:, 2] for i in eachindex(got))
    end
end

@testset "call-result indexing preserves bounds and lazy arms" begin
    for expr in (:(column_sums(W)[0]), :(sum(W; dims=1)[1, 3]))
        # Refused by Julia bounds, not by the model surface: the result has
        # two entries and shape (1, 2), respectively.
        @test_throws BoundsError begin
            fx = fixture(expr)
            fx.sampler(fx.p)
        end
    end
    check(fixture(:(x[i] > -1.0 ? column_sums(W)[2] : column_sums(W)[20])); query=true)
    check(fixture(:(W[1, 1] > -1.0 ? column_sums(W)[2] : column_sums(W)[20])); query=true)
    empty = fixture(:(column_sums(W)[20]); n=0)
    check(empty)
    # Known RK limitation (0j2nqi0): materializing an empty plate still
    # evaluates shared invariants. Its empty likelihood needs no cell work.
    read_empty = Base.invokelatest(prepare, empty.built.spec;
        have=(:unconstrained, keys(empty.data)...), want=:r, bound=empty.data)
    @test_broken Base.invokelatest(read_empty, empty.p) == Float64[]
    # The same out-of-bounds arm must throw when it is selected.
    @test_throws BoundsError begin
        fx = fixture(:(x[i] < -1.0 ? column_sums(W)[2] : column_sums(W)[20]))
        fx.sampler(fx.p)
    end
end

const LIVE_CALLS = Ref(0)
counted_sums(A) = (LIVE_CALLS[] += 1; column_sums(A))
const DATA_CALLS = Ref(0)
data_sums(A) = (DATA_CALLS[] += 1; column_sums(A))

@testset "shared calls hoist and data-only calls stay bound" begin
    for n in (2, 17)
        fx = fixture(:(counted_sums(W)[2]); n)
        fx.sampler(fx.p)
        LIVE_CALLS[] = 0
        @test fx.sampler(fx.p) ≈ fx.oracle(fx.p)
        @test LIVE_CALLS[] == 1
        sampler_value_and_gradient!(fx.sampler, similar(fx.p), fx.p)
        LIVE_CALLS[] = 0
        sampler_value_and_gradient!(fx.sampler, similar(fx.p), fx.p)
        @test LIVE_CALLS[] == 1
    end
    ast = quote
        a ~ Normal(0, 1)
        @plate for i in eachindex(y)
            r[i] = a * x[i] * data_sums(A)[2]
        end
        y .~ Normal.(r, 1)
    end
    plan = lower_rkppl(ast, (:A, :x, :y); mod=@__MODULE__, conditioned=(:y,))
    for n in (2, 7)
        data = (; A=[0.2j - 0.3k for j in 1:3, k in 1:2],
            x=[0.1i for i in 1:n], y=[0.2sin(i) for i in 1:n])
        saved = deepcopy(data)
        DATA_CALLS[] = 0
        bound = bind_data(plan, data)
        built = build_kernel(bound)
        sampler = prepare_sampler(built, bound, [0.3]; backend=BACKEND)
        bound_calls = DATA_CALLS[]
        @test bound_calls > 0
        oracle(p) = D.logpdf(D.Normal(), p[1]) +
            sum(D.logpdf.(D.Normal.(p[1] .* data.x .* sum(data.A[:, 2]), 1), data.y))
        for p in ([0.3], [-0.6])
            value, grad = sampler_value_and_gradient!(sampler, similar(p), p)
            @test value ≈ oracle(p)
            @test grad ≈ differences(oracle, p) rtol=1e-5 atol=1e-7
        end
        @test DATA_CALLS[] == bound_calls
        @test data == saved
    end
end

end # module
