module DistributionGraphExamples

const SOURCE_ROOT = normpath(joinpath(@__DIR__, "..", "packages",
    "ReactiveKernelsDistributionKernels", "src"))

# Read the implemented kernels when the page builds; the displayed math and
# the executable example therefore change together.
function moment_source()
    "import SpecialFunctions, Distributions\n" *
    read(joinpath(SOURCE_ROOT, "lognormal_moments.jl"), String) * raw"""

moments = extract(lower_truncated_lognormal;
    have=(:location, :scale, :lower), want=(:mean, :var))
kernel = prepare(moments)
inputs = (; location=0.2, scale=0.3, lower=1.1)
output = kernel(Tuple(inputs)...)

# Independent central-region raw-moment oracle. The implementation above
# avoids this subtraction when the variance is tiny.
mu, sigma, lower = Tuple(inputs)
a = (log(lower) - mu) / sigma
tail(z) = Distributions.ccdf(Distributions.Normal(), z)
mean_reference = exp(mu + sigma^2 / 2) * tail(a - sigma) / tail(a)
second_reference = exp(2mu + 2sigma^2) * tail(a - 2sigma) / tail(a)
@assert isapprox(output[1], mean_reference; rtol=1e-12)
@assert isapprox(output[2], second_reference - mean_reference^2; rtol=1e-12)
ordinary = kernel(mu, sigma, 0.0)
@assert ordinary[1] ≈ Distributions.mean(Distributions.LogNormal(mu, sigma))
@assert ordinary[2] ≈ Distributions.var(Distributions.LogNormal(mu, sigma))

docs_example = (; name=:bounded_lognormal_moments,
    origin="owned one-sided LogNormal moments, including ordinary moments at lower=0",
    inputs, spec=moments, kernel, output)
"""
end

function count_source()
    "using SpecialFunctions: loggamma\nimport Distributions\n" *
    read(joinpath(SOURCE_ROOT, "dirichlet_multinomial.jl"), String) * raw"""

@kernel count_batch(events::Vector{Vector{Int}}, n::Int, alpha::Vector{Float64}) = begin
    scores = plate(events, Ref(n), Ref(alpha)) do counts, n, alpha
        dirichlet_multinomial(n, alpha).logpdf(counts)
    end
    return sum(scores)
end

events = [[2, 3, 1], [0, 6, 0]]
saved = deepcopy(events)
kernel = prepare(count_batch; bound=(; events, n=6))
inputs = (; alpha=[0.2, 0.5, 0.3])
output = kernel(inputs.alpha)
for alpha in (inputs.alpha, [1.2, 2.5, 0.7])
    reference = sum(Distributions.logpdf(
        Distributions.DirichletMultinomial(6, alpha), counts) for counts in events)
    @assert kernel(alpha) ≈ reference
end
@assert events == saved
generated = string(code_expr(kernel))
@assert occursin("bound_plate_logpdf__coefficient", generated)
@assert !occursin("coefficient.normalizers", generated)

docs_example = (; name=:bound_dirichlet_multinomial_counts,
    origin="bound count coefficients with live shared concentrations",
    inputs, spec=count_batch, kernel, output)
"""
end

end
