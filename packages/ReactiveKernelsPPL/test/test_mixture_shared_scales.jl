module MixtureSharedScalesTests
using ReactiveKernels, ReactiveKernelsPPL, Test
using DifferentiationInterface: AutoEnzyme
import Enzyme
import Distributions as D

# Natural spelling of the reported mixture: the component scales are scalar
# aliases of an abstractly declared array produced by the transform plate.
const MODEL = @rkppl begin
    mu ~ Ordered(Normal(0, 2), 2)
    sigma[1:2] .~ HalfNormal.(2)
    theta ~ Beta(5, 5)
    mu1 = mu[1]
    mu2 = mu[2]
    sigma1 = sigma[1]
    sigma2 = sigma[2]
    y .~ MixtureModel.(vcat.(Normal.(mu1, sigma1), Normal.(mu2, sigma2)),
        Ref([theta, 1 - theta]))
end

function reference(u, y)
    theta = inv(1 + exp(-u[1]))
    mu = [u[2], u[2] + exp(u[3])]
    sigma = exp.(u[4:5])
    prior = D.logpdf(D.Beta(5, 5), theta) + log(theta) + log1p(-theta)
    prior += sum(D.logpdf.(D.Normal(0, 2), mu)) + u[3]
    prior += sum(D.logpdf.(D.Normal(0, 2), sigma)) + 2log(2) + sum(u[4:5])
    likelihood = sum(y; init=0.0) do yi
        a = log(theta) + D.logpdf(D.Normal(mu[1], sigma[1]), yi)
        b = log1p(-theta) + D.logpdf(D.Normal(mu[2], sigma[2]), yi)
        max(a, b) + log1p(exp(-abs(a - b)))
    end
    prior + likelihood
end

function finite_gradient(f, u)
    map(eachindex(u)) do i
        p, m = copy(u), copy(u)
        p[i] += 1e-5
        m[i] -= 1e-5
        (f(p) - f(m)) / 2e-5
    end
end
nodes(ex) = ex isa Expr ? 1 + sum(nodes, ex.args; init=0) : 1
loops(ex) = ex isa Expr ? (ex.head === :for) + sum(loops, ex.args; init=0) : 0
function scale_log_bindings(ex, in_loop=false)
    ex isa Expr || return (0, 0)
    in_loop |= ex.head in (:for, :while)
    own = if ex.head === :(=) && ex.args[1] isa Symbol &&
            occursin("plate_", String(ex.args[1])) && occursin("log_scale", String(ex.args[1]))
        in_loop ? (0, 1) : (1, 0)
    else
        (0, 0)
    end
    foldl((a, b) -> a .+ b, (scale_log_bindings(x, in_loop) for x in ex.args); init=own)
end

function check(q, u, y)
    g = similar(u)
    value, _ = sampler_value_and_gradient!(q, g, u)
    @test value ≈ reference(u, y) rtol=2e-12
    @test g ≈ finite_gradient(v -> reference(v, y), u) rtol=2e-6 atol=3e-7
end

@testset "mixture scales from a preceding plate remain shared" begin
    sizes = Int[]
    for n in (0, 3, 103)
        y = [0.4cos(i) + 0.2sin(0.7i) for i in 1:n]
        bound = MODEL() | (; y)
        built = build_kernel(bound)
        u = collect(Float64, unconstrain(built.layout,
            (theta=0.4, mu=[-0.5, 0.7], sigma=[0.8, 1.2])))
        q = prepare_sampler(built, bound, u; backend=AutoEnzyme(mode=Enzyme.Reverse),
            warm=true, native_forward_batch=false)
        for delta in (zeros(5), [0.3, -0.2, 0.1, -0.4, 0.2])
            Base.invokelatest(check, q, u + delta, y)
        end
        program = Base.remove_linenums!(deepcopy(code_expr(q.kernel)))
        @test loops(program) > 0
        # Two mixture components and two prior plates have shared scales.
        # Their logarithm bindings must precede their observation loops.
        outside, inside = scale_log_bindings(program)
        @test outside >= 4
        @test inside == 0
        push!(sizes, nodes(program))
    end
    @test all(==(first(sizes)), sizes)
end
end
