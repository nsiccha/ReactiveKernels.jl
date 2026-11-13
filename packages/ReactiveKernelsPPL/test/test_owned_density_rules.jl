module OwnedDensityRuleTests
using Test, ReactiveKernels, ReactiveKernelsPPL
using LogExpFunctions: log1pexp
import DifferentiationInterface as DI
import Enzyme

# A user-owned density supplies its math once. PPL's open LogDensity interface
# composes it with a nonlinear predictor and ordinary latent/prior lowering.
@kernel logistic_graph(y::Float64, mu::Float64, s::Float64) = begin
    z::Float64 = (y - mu) / exp(s)
    t::Float64 = tanh(z / 2)
    value::Float64 = -s - log1pexp(z) - log1pexp(-z)
    dy::Float64 = -t / exp(s)
    dmu::Float64 = t / exp(s)
    ds::Float64 = z * t - 1
    return value, dy, dmu, ds
end
const logistic = scalar_derivative_rule(logistic_graph; primal = :value,
    partials = (y = :dy, mu = :dmu, s = :ds))

@testset "owned density graph through ordinary PPL reverse" begin
    ast = quote
        a ~ Normal(0, 1)
        s ~ Normal(0, 1)
        mu = sin(a)
        y .~ LogDensity.(logistic, mu, s)
    end
    for n in (0, 3, 17)
        y = [i == 2 ? -1000.0 : i == 3 ? 1000.0 : sin(i) for i in 1:n]
        data = Dict{Symbol,Any}(:y => y)
        before = deepcopy(data)
        plan = bind_data(lower_rkppl(ast, data; mod = @__MODULE__, conditioned = keys(data)), data)
        built = build_kernel(plan)
        u0 = unconstrain(built.layout, (; a = 0.3, s = -0.2))
        pointwise = prepare_query(built, plan, :pointwise)
        for enabled in (false, true)
            sampler = with_plate_reverse(enabled) do
                prepare_sampler(built, plan, u0;
                    backend = DI.AutoEnzyme(; mode = Enzyme.Reverse))
            end
            for (a, s) in ((0.3, -0.2), (-0.4, 0.7))
                u = unconstrain(built.layout, (; a, s))
                saved = copy(u)
                value, gradient = sampler_value_and_gradient!(sampler, zeros(length(u)), u)
                z = (y .- sin(a)) ./ exp(s)
                cells = -s .- log1pexp.(z) .- log1pexp.(-z)
                @test value ≈ -log(2pi) - (a^2+s^2)/2 + sum(cells; init = 0.0)
                ga = -a + cos(a) * sum(tanh.(z ./ 2); init = 0.0) / exp(s)
                gs = -s + sum(z .* tanh.(z ./ 2) .- 1; init = 0.0)
                # Use the public layout to respect the model's packing order.
                expected = unconstrain(built.layout, (; a = ga, s = gs))
                @test gradient ≈ expected rtol = 1e-10 atol = 1e-11
                @test Base.invokelatest(pointwise, u).y ≈ cells
                @test u == saved
            end
        end
        @test data == before
    end
end
end
