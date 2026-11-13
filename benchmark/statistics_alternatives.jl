# Bounded architecture experiment, not an automatic rewrite or a public API.
# Run in a consumer env containing this checkout, its nested PPL/distribution
# packages, Enzyme, DifferentiationInterface and MutatingFunctions.
# Only deterministic synthetic data. Optional ARGS select sections:
# selection, guard, numerical, accuracy, ppl. Rank-deficient cases must retain
# accurate row fallback, including the former exact-null-space regression.
using ReactiveKernels, ReactiveKernelsPPL, LinearAlgebra, Random, Test
using DifferentiationInterface: AutoEnzyme
import Enzyme, MutatingFunctions
const RK = ReactiveKernels
BLAS.set_num_threads(1)

structure(x) = (1,0,0)
structure(x::RK._PlateProgram) = structure(x.ast)
structure(x::RK._PlateCall) = structure(RK._program_expression(x.program))
function structure(x::Expr)
    parts = map(structure,x.args)
    (1+sum(first,parts;init=0),
     (x.head in (:for,:while))+sum(p->p[2],parts;init=0),
     (x.head === :if)+sum(last,parts;init=0))
end

println((stage=:environment, julia=VERSION, enzyme=pkgversion(Enzyme),
    enzyme_source=pkgdir(Enzyme), library=Enzyme.Compiler.Enzyme_jll.libEnzyme_path))
@assert Base.get_extension(RK, :ReactiveKernelsEnzymeExt) !== nothing
@assert Base.get_extension(RK, :ReactiveKernelsMutatingFunctionsExt) !== nothing

row_residual(X, y, beta) = y - X * beta
row_total(residual) = sum(abs2, residual)
function guarded_statistics(s, X, y, beta)
    s.usable ? RK._ls_rss_dense(s, (beta,)) : row_total(row_residual(X, y, beta))
end
scalar_first(k, beta) = first(k(beta))

# Costs here are explicitly declared scalar objective weights. They demonstrate
# selection and sharing, not a claim that these numbers predict wall time.
function alternative_graph(; evaluations=1, residual_cost=9.0, reduction_cost=1.0,
                             preparation_cost=20.0, statistics_cost=2.0)
    g = Graph()
    X = value!(g, :X, Matrix{Float64})
    y = value!(g, :y, Vector{Float64})
    beta = value!(g, :beta, Vector{Float64})
    r = value!(g, :residual, Vector{Float64})
    s = value!(g, :statistics, RK._LSStatistics)
    total = value!(g, :total, Float64)
    add!(g, (X, y, beta) => r, row_residual; cost=residual_cost)
    add!(g, r => total, row_total; cost=reduction_cost)
    add!(g, (X, y) => s, RK._ls_statistics_dense; cost=preparation_cost/evaluations)
    add!(g, (s, X, y, beta) => total, guarded_statistics; cost=statistics_cost)
    (; g, X, y, beta, r, s, total)
end

function median_ns(f; samples=9, iterations=100)
    f()
    times = Float64[]
    for _ in 1:samples
        t = time_ns()
        for _ in 1:iterations
            f()
        end
        push!(times, (time_ns()-t)/iterations)
    end
    sort!(times)
    times[(length(times)+1)÷2]
end

function reverse!(k, dbeta, beta)
    fill!(dbeta, 0)
    Enzyme.autodiff(Enzyme.ReverseWithPrimal, Enzyme.Const(k), Enzyme.Active,
        Enzyme.Duplicated(beta, dbeta))[2]
end

function synthetic(n, k; shift=0.0, rank_deficient=false)
    rng = Xoshiro(415+n+k)
    X = hcat(ones(n), randn(rng, n, k-1))
    rank_deficient && (X[:,end] .= X[:,1])
    truth = collect(range(-0.3, 0.4; length=k))
    y = X*truth + 0.03randn(rng,n) .+ shift
    beta = truth .+ 0.1
    beta[1] += shift
    (;X,y,beta)
end

if isempty(ARGS) || "selection" in ARGS
@testset "costed numerical alternatives before planning" begin
    for (evaluations, want_residual, expected, expected_cost) in (
            (1,false,[1,2],10.0), (100,false,[3,4],2.2),
            (100,true,[1,2],10.0))
        a = alternative_graph(;evaluations)
        want = want_residual ? (a.total,a.r) : (a.total,)
        p = plan(a.g; have=(a.X,a.y,a.beta), want)
        @test [r.id for r in p.recipes] == expected
        @test p.cost ≈ expected_cost
        d = synthetic(40,3)
        q = prepare(a.g; have=(a.X,a.y,a.beta), want,
            bound=(a.X=>d.X,a.y=>d.y))
        @test first(q(d.beta)) ≈ row_total(row_residual(d.X,d.y,d.beta))
        want_residual && @test last(q(d.beta)) ≈ row_residual(d.X,d.y,d.beta)
        println((stage=:selection, evaluations, want_residual, recipes=expected,
            cost=p.cost, plan_ns=median_ns(()->plan(a.g;have=(a.X,a.y,a.beta),want))))
    end
end
end

# Same graph and compiled type across threshold/empty/rank changes. Costs are
# fixed here. A runtime guard, not the planner, determines numerical eligibility.
if isempty(ARGS) || "guard" in ARGS
@testset "guarded alternative retains runtime eligibility" begin
    a = alternative_graph(;evaluations=100)
    kernels = []
    structures = []
    for (n,rank_deficient) in ((0,false),(5,false),(6,false),(40,false),(40,true),(400,false))
        d = synthetic(n,3;rank_deficient)
        original = deepcopy((d.X,d.y,d.beta))
        q = prepare(a.g; have=(a.X,a.y,a.beta), want=(a.total,), bound=(a.X=>d.X,a.y=>d.y))
        push!(kernels,q)
        push!(structures,structure(code_expr(q)))
        f = b -> scalar_first(q,b)
        grad = zeros(3)
        @test reverse!(f,grad,d.beta) ≈ row_total(row_residual(d.X,d.y,d.beta))
        @test grad ≈ 2d.X'*(d.X*d.beta-d.y)
        @test (d.X,d.y,d.beta) == original
        stats = only(filter(x->x isa RK._LSStatistics,q.constants))
        @test stats.usable == (n >= 6 && !rank_deficient)
        println((stage=:runtime_guard,n,rank_deficient,usable=stats.usable,
            stats_bytes=Base.summarysize(stats),constants_bytes=Base.summarysize(q.constants),
            source_structure=last(structures)))
    end
    @test allequal(typeof.(kernels))
    @test allequal(structures)
end
@testset "manual alternative graph execution" begin
    for (n,k) in ((40,3),(4000,3),(40,16)), evaluations in (1,100)
        a = alternative_graph(;evaluations)
        d = synthetic(n,k)
        q = prepare(a.g;have=(a.X,a.y,a.beta),want=(a.total,),bound=(a.X=>d.X,a.y=>d.y))
        grad = zeros(k)
        f = b->scalar_first(q,b)
        run = ()->reverse!(f,grad,d.beta)
        @test run() ≈ row_total(row_residual(d.X,d.y,d.beta))
        @test grad ≈ 2d.X'*(d.X*d.beta-d.y)
        println((stage=:manual_alternative,n,k,evaluations,
            gradient_ns=median_ns(run),julia_gradient_bytes=@allocated(run()),
            constants_bytes=Base.summarysize(q.constants)))
    end
end
end

@kernel scaled_rss(X::Matrix{Float64}, y::Vector{Float64}, beta::Vector{Float64}, scale::Float64) = begin
    mu = X * beta
    terms = plate(y,mu) do yi,mi
        scale * (yi-mi)^2
    end
    total::Float64 = sum(terms)
    return total
end

if isempty(ARGS) || "numerical" in ARGS
@testset "current natural numerical path cost and ordinary reverse" begin
    structures = Dict(false=>Any[],true=>Any[])
    for (n,k) in ((4,3),(40,3),(400,3),(4000,3),(40,16),(400,16),(40,24))
        d = synthetic(n,k)
        pairs = map((false,true)) do statistics
            prepare(scaled_rss; bound=(;d.X,d.y),bound_statistics=statistics)
        end
        for (statistics,native) in zip((false,true),pairs)
            push!(structures[statistics],structure(code_expr(native)))
            println((stage=:natural_structure,n,k,statistics,
                source_structure=last(structures[statistics])))
        end
        for (statistics,native) in zip((false,true),pairs), reuse in (false,true)
            q = reuse ? prepare_nonallocating(native,d.beta,0.5) : native
            ad = prepare_ad(q,AutoEnzyme(;mode=Enzyme.Reverse),d.beta,0.5;active=:beta)
            grad = zeros(k)
            f = ()->ad_value_and_gradient!(ad,grad,d.beta,0.5)
            first_gradient = @elapsed value = first(f())
            @test value ≈ 0.5row_total(row_residual(d.X,d.y,d.beta))
            @test grad ≈ d.X'*(d.X*d.beta-d.y)
            iterations = max(10,min(200,100000÷max(n*k,1)))
            gradient_ns = median_ns(f;iterations)
            bytes = @allocated f()
            primal_ns = median_ns(()->q(d.beta,0.5);iterations)
            println((stage=:natural_rss,n,k,statistics,reuse,primal_ns,gradient_ns,
                julia_gradient_bytes=bytes,first_gradient_seconds=first_gradient,
                constants_bytes=Base.summarysize(native.constants)))
        end
        for statistics in (false,true)
            bind = ()->prepare(scaled_rss;bound=(;d.X,d.y),bound_statistics=statistics)
            println((stage=:rebind,n,k,statistics,ns=median_ns(bind;iterations=2,samples=5)))
        end
        s = RK._ls_statistics_dense(d.X,d.y)
        println((stage=:statistics_preparation,n,k,usable=s.usable,
            ns=median_ns(()->RK._ls_statistics_dense(d.X,d.y);iterations=2,samples=5),
            bytes=Base.summarysize(s)))
    end
    @test allequal(structures[false])
    @test allequal(structures[true])
end
end

if isempty(ARGS) || "accuracy" in ARGS
@testset "stable residual accuracy" begin
    for shift in (0.0,1e4,1e8), rank_deficient in (false,true)
        d = synthetic(400,3;shift,rank_deficient)
        # The duplicated final column is omitted for this bounded fitted point.
        # Large null-space components are tested separately, without a solver.
        beta = rank_deficient ? vcat(d.X[:,1:2]\d.y,0.0) : d.X\d.y
        s = RK._ls_statistics_dense(d.X,d.y)
        reference = setprecision(BigFloat,256) do
            r = big.(d.y)-big.(d.X)*big.(beta)
            (sum(abs2,r), -2big.(d.X)'*r)
        end
        grad = zeros(3)
        f = b->guarded_statistics(s,d.X,d.y,b)
        stable = reverse!(f,grad,beta)
        row = row_total(row_residual(d.X,d.y,beta))
        naive = dot(d.y,d.y)-2dot(beta,d.X'*d.y)+dot(beta,(d.X'*d.X)*beta)
        err(v)=Float64(abs(big(v)-reference[1])/reference[1])
        # At a fitted point the gradient is near zero; report absolute norm
        # rather than an ill-conditioned relative error.
        graderr = Float64(norm(big.(grad)-reference[2]))
        rowgraderr = Float64(norm(big.(-2d.X'*(d.y-d.X*beta))-reference[2]))
        @test s.usable == !rank_deficient
        @test err(stable) <= max(4err(row),1e-13)
        @test graderr <= max(10rowgraderr,1e-11)
        println((stage=:accuracy,shift,rank_deficient,reference=Float64(reference[1]),
            stable_error=err(stable),row_error=err(row),expanded_error=err(naive),
            stable_gradient_absolute_error=graderr,row_gradient_absolute_error=rowgraderr))
    end
end
@testset "rank-deficient cancellation regression" begin
    # No solver or probabilistic semantics are needed to trigger this gap.
    # Exact reference: beta[1] + beta[2] == 1, hence residual and gradient zero.
    # Regression from statistics-alternatives todo 17bzmgx. The compressed
    # factor formerly returned RSS 15.6055 and gradient approximately [120,120].
    X, y = ones(400,2), ones(400)
    beta = [1e14,1-1e14]
    reference = sum(abs2,big.(y)-big.(X)*big.(beta))
    @test iszero(reference)
    for statistics in (false,true)
        q = prepare(scaled_rss;bound=(;X,y),bound_statistics=statistics)
        ad = prepare_ad(q,AutoEnzyme(;mode=Enzyme.Reverse),beta,1.0;active=:beta)
        grad = zeros(2)
        value = first(ad_value_and_gradient!(ad,grad,beta,1.0))
        @test value == 0
        @test all(iszero,grad)
        println((stage=:cancellation_regression,statistics,beta,value,gradient=copy(grad),
            reference=Float64(reference)))
    end
end
end

if isempty(ARGS) || "ppl" in ARGS
@testset "natural PPL remains on the shared numerical path" begin
    model = @rkppl begin
        beta[1:k] .~ Normal.(0.0,1.0)
        y .~ Normal.(X*beta,noise)
    end
    types = Dict(false=>Any[],true=>Any[])
    for n in (0,4,40,400)
        d = synthetic(n,3)
        data = (;d.X,d.y,k=3,noise=0.7)
        fit = model(;data.X,data.k,data.noise) | (;data.y)
        built = build_kernel(fit)
        for statistics in (false,true)
            q = with_bound_statistics(statistics) do
                prepare_sampler(built,bind_data(fit,data),d.beta;
                    backend=AutoEnzyme(;mode=Enzyme.Reverse),nonallocating=true)
            end
            push!(types[statistics],typeof(q))
            grad = zeros(3)
            f = ()->sampler_value_and_gradient!(q,grad,d.beta)
            first_gradient = @elapsed value = first(f())
            reference = -sum(abs2,d.beta)/2 - 3log(2pi)/2 -
                row_total(row_residual(d.X,d.y,d.beta))/(2data.noise^2) - n*log(data.noise) - n*log(2pi)/2
            @test value ≈ reference
            @test grad ≈ -d.beta + d.X'*(d.y-d.X*d.beta)/data.noise^2
            println((stage=:ppl,n,statistics,gradient_ns=median_ns(f),
                julia_gradient_bytes=@allocated(f()),first_gradient_seconds=first_gradient))
        end
    end
    @test allequal(types[false])
    @test allequal(types[true])
end
end
println("STATISTICS_ALTERNATIVES_COMPLETE")
