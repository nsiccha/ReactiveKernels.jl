module LawDocsTests
# Every exported law object documents its parameters, and every statement a
# law docstring makes is checked against the object here: the call signature
# against the owner formals a positional construction binds, each named
# binding and the stated Distributions.jl equivalent numerically, and the
# endpoint list against the object's endpoints (snag distribution-con-f58a59c2).
using Test, ReactiveKernels
import ReactiveKernels as RK
import ReactiveKernelsDistributionKernels.DistributionKernelSources as Laws
import Distributions as D
import LogExpFunctions, SpecialFunctions
using LinearAlgebra: Cholesky, LowerTriangular, Symmetric, cholesky

_is_law(value) = value isa Union{RK._KernelObjectSkeleton,RK.KernelObjectSpec}
const LAWS = sort([name for name in names(Laws)
    if Base.isexported(Laws, name) && _is_law(getfield(Laws, name))])

function _docstring(name)
    meta = Base.Docs.meta(Laws)
    binding = Base.Docs.Binding(Laws, name)
    haskey(meta, binding) || return nothing
    join(string.(collect(only(values(meta[binding].docs)).text)))
end

# The paragraph a docstring opens with `label:`, up to the next blank line.
function _paragraph(doc, label)
    m = match(Regex("(?ms)^$(label):(.*?)(?:\\n\\s*\\n|\\z)"), doc)
    m === nothing ? "" : m.captures[1]
end

function _signatures(name, doc)
    map(eachmatch(Regex("(?m)^    $(name)\\((.*)\\)\$"), doc)) do m
        text = strip(m.captures[1])
        named = startswith(text, ";")
        fields = filter(!isempty, strip.(split(lstrip(text, ';'), ',')))
        (named, Tuple(Symbol.(fields)))
    end
end

_signature(law, endpoint) = typeof(extract(law; want = endpoint).call_signature).parameters
_owner_formals(law, endpoint) = Tuple(first(_signature(law, endpoint)))
_explicit_formals(law, endpoint) = Tuple(last(_signature(law, endpoint)))
_primary(law) = :logpdf in RK.kernel_endpoint_names(law) ? :logpdf :
    first(RK.kernel_endpoint_names(law))

_X = [1.0 0.4; 1.0 -1.2; 1.0 0.7]
_L = Matrix(cholesky(Symmetric([1.0 0.3 -0.2; 0.3 1.0 0.1; -0.2 0.1 1.0])).L)
_Σ = [1.4 0.3 0.1; 0.3 0.8 -0.2; 0.1 -0.2 1.1]
_zi(count_law, zi) =
    D.MixtureModel(D.DiscreteUnivariateDistribution[D.Dirac(0), count_law], [zi, 1 - zi])
_logistic(t) = 1 / (1 + exp(-t))
_ar1_cov(ϕ, scale, n) = [scale^2 * ϕ^abs(i - j) / (1 - ϕ^2) for i in 1:n, j in 1:n]

# One case per law: parameter values in the order of `formals` (default: the
# documented positional signature), the stated reference expression, sample
# observations, the reference log density and every named binding the
# docstring lists, as (name, the formal it replaces, its value from params).
const CASES = Dict{Symbol,Any}(
    :standard_normal => (params = (), doc = "Distributions.Normal()",
        xs = [-1.3, 0.0, 2.2], ref = x -> D.logpdf(D.Normal(), x)),
    :standard_cauchy => (params = (), doc = "Distributions.Cauchy()",
        xs = [-1.3, 0.0, 2.2], ref = x -> D.logpdf(D.Cauchy(), x)),
    :standard_laplace => (params = (), doc = "Distributions.Laplace()",
        xs = [-1.3, 0.0, 2.2], ref = x -> D.logpdf(D.Laplace(), x)),
    :standard_logistic => (params = (), doc = "Distributions.Logistic()",
        xs = [-1.3, 0.0, 2.2], ref = x -> D.logpdf(D.Logistic(), x)),
    :standard_student_t => (params = (3.5,), doc = "Distributions.TDist(nu)",
        xs = [-1.3, 0.0, 2.2], ref = (nu, x) -> D.logpdf(D.TDist(nu), x)),
    :location_scale => (law = Laws.location_scale(Laws.standard_laplace),
        params = (0.3, 1.7), doc = "location_scale(standard_normal)",
        xs = [-1.3, 0.0, 2.2], ref = (l, s, x) -> D.logpdf(D.Laplace(l, s), x),
        alts = ((:log_scale, :scale, (l, s) -> log(s)),)),
    :normal => (params = (0.3, 1.7), doc = "Distributions.Normal(location, scale)",
        xs = [-1.3, 0.0, 2.2], ref = (l, s, x) -> D.logpdf(D.Normal(l, s), x),
        alts = ((:log_scale, :scale, (l, s) -> log(s)),)),
    :cauchy => (params = (0.3, 1.7), doc = "Distributions.Cauchy(location, scale)",
        xs = [-1.3, 0.0, 2.2], ref = (l, s, x) -> D.logpdf(D.Cauchy(l, s), x),
        alts = ((:log_scale, :scale, (l, s) -> log(s)),)),
    :laplace => (params = (0.3, 1.7), doc = "Distributions.Laplace(location, scale)",
        xs = [-1.3, 0.0, 2.2], ref = (l, s, x) -> D.logpdf(D.Laplace(l, s), x),
        alts = ((:log_scale, :scale, (l, s) -> log(s)),)),
    :logistic => (params = (0.3, 1.7), doc = "Distributions.Logistic(location, scale)",
        xs = [-1.3, 0.0, 2.2], ref = (l, s, x) -> D.logpdf(D.Logistic(l, s), x),
        alts = ((:log_scale, :scale, (l, s) -> log(s)),)),
    :student_t => (params = (4.5, 0.3, 1.7),
        doc = "Distributions.TDist(nu) * scale + location",
        xs = [-1.3, 0.0, 2.2],
        ref = (nu, l, s, x) -> D.logpdf(D.TDist(nu) * s + l, x),
        alts = ((:log_scale, :scale, (nu, l, s) -> log(s)),)),
    :bernoulli => (params = (0.37,), doc = "Distributions.Bernoulli(p)",
        xs = [true, false], ref = (p, y) -> D.logpdf(D.Bernoulli(p), y),
        alts = ((:logit, :p, p -> log(p / (1 - p))),)),
    :lognormal => (params = (0.3, 0.8), doc = "Distributions.LogNormal(location, scale)",
        xs = [0.4, 1.4, 3.0], ref = (l, s, x) -> D.logpdf(D.LogNormal(l, s), x),
        alts = ((:log_scale, :scale, (l, s) -> log(s)),)),
    :exponential => (params = (1.7,), doc = "Distributions.Exponential(scale)",
        xs = [0.0, 0.4, 3.1], ref = (s, x) -> D.logpdf(D.Exponential(s), x),
        alts = ((:log_scale, :scale, s -> log(s)),)),
    :geometric => (params = (0.6,), doc = "Distributions.Geometric(p)",
        xs = [0, 1, 4], ref = (p, y) -> D.logpdf(D.Geometric(p), y),
        alts = ((:logitp, :p, p -> log(p / (1 - p))),)),
    :uniform => (params = (-1.0, 2.0), doc = "Distributions.Uniform(lower, upper)",
        xs = [-0.5, 1.9], ref = (a, b, x) -> D.logpdf(D.Uniform(a, b), x)),
    :mvnormal => (formals = (:μ, :covariance), params = ([-0.2, 0.3, 0.5], _Σ),
        doc = "Distributions.MvNormal(μ, covariance)", xs = [[0.4, -1.1, 0.7]],
        ref = (μ, Σ, x) -> D.logpdf(D.MvNormal(μ, Σ), x),
        alts = ((:chol, :covariance, (μ, Σ) -> Matrix(cholesky(Symmetric(Σ)).L)),
                (:precision, :covariance, (μ, Σ) -> inv(Σ)),
                (:precision_chol, :covariance,
                    (μ, Σ) -> Matrix(cholesky(Symmetric(inv(Σ))).L)))),
    :ar1 => (params = (0.2, 0.6, 0.9), doc = "Distributions.MvNormal",
        xs = [[0.1, -0.3, 0.5, 0.2]],
        ref = (μ, ϕ, s, x) -> D.logpdf(D.MvNormal(fill(μ, length(x)),
            _ar1_cov(ϕ, s, length(x))), x)),
    :categorical_logit => (params = ([0.2, -1.0, 0.7],),
        doc = "Distributions.Categorical(LogExpFunctions.softmax(logits))", xs = [1, 2, 3],
        ref = (l, y) -> D.logpdf(D.Categorical(LogExpFunctions.softmax(l)), y)),
    :categorical_logit_ref => (params = ([-1.0, 0.7],),
        doc = "categorical_logit([0.0; nonreference_logits])", xs = [1, 2, 3],
        ref = (l, y) -> D.logpdf(D.Categorical(LogExpFunctions.softmax([0.0; l])), y)),
    :poisson => (params = (2.3,), doc = "Distributions.Poisson(rate)",
        xs = [0, 1, 5], ref = (r, y) -> D.logpdf(D.Poisson(r), y),
        alts = ((:log_rate, :rate, r -> log(r)),)),
    :gamma => (params = (2.5, 1.5), doc = "Distributions.Gamma(shape, 1 / rate)",
        xs = [0.3, 1.2, 4.0], ref = (a, r, x) -> D.logpdf(D.Gamma(a, 1 / r), x),
        alts = ((:scale, :rate, (a, r) -> 1 / r), (:log_rate, :rate, (a, r) -> log(r)))),
    :beta => (params = (2.0, 3.5), doc = "Distributions.Beta(a, b)",
        xs = [0.1, 0.5, 0.9], ref = (a, b, x) -> D.logpdf(D.Beta(a, b), x)),
    :binomial => (params = (7, 0.35), doc = "Distributions.Binomial(n, p)",
        xs = [0, 3, 7], ref = (n, p, y) -> D.logpdf(D.Binomial(n, p), y),
        alts = ((:logit, :p, (n, p) -> log(p / (1 - p))),)),
    :negative_binomial2 => (params = (3.2, 1.7),
        doc = "Distributions.NegativeBinomial(phi, phi / (mu + phi))", xs = [0, 2, 9],
        ref = (mu, phi, y) -> D.logpdf(D.NegativeBinomial(phi, phi / (mu + phi)), y)),
    :negative_binomial => (params = (2.5, 0.4), doc = "Distributions.NegativeBinomial(r, p)",
        xs = [0, 2, 9], ref = (r, p, y) -> D.logpdf(D.NegativeBinomial(r, p), y)),
    :weibull => (params = (1.6, 2.2), doc = "Distributions.Weibull(k, theta)",
        xs = [0.3, 1.0, 4.0], ref = (k, t, x) -> D.logpdf(D.Weibull(k, t), x),
        alts = ((:log_theta, :theta, (k, t) -> log(t)),)),
    :beta_binomial => (params = (7, 1.5, 2.5), doc = "Distributions.BetaBinomial(n, alpha, beta)",
        xs = [0, 3, 7], ref = (n, a, b, y) -> D.logpdf(D.BetaBinomial(n, a, b), y)),
    :inverse_gamma => (params = (3.0, 2.0), doc = "Distributions.InverseGamma(shape, scale)",
        xs = [0.3, 1.0, 4.0], ref = (a, s, x) -> D.logpdf(D.InverseGamma(a, s), x),
        alts = ((:log_scale, :scale, (a, s) -> log(s)),)),
    :dirichlet => (params = ([1.5, 2.0, 0.7],), doc = "Distributions.Dirichlet(alpha)",
        xs = [[0.2, 0.5, 0.3]], ref = (a, x) -> D.logpdf(D.Dirichlet(a), x)),
    :lkj_corr_cholesky => (params = (1.7,), doc = "Distributions.LKJCholesky(K, eta, 'L')",
        xs = [_L], ref = (eta, L) -> D.logpdf(D.LKJCholesky(size(L, 1), eta, 'L'),
            Cholesky(LowerTriangular(L)))),
    :zero_inflated_poisson => (params = (2.3, 0.2),
        doc = "Distributions.MixtureModel([Distributions.Dirac(0), Distributions.Poisson(rate)], [zi, 1 - zi])",
        xs = [0, 1, 5], ref = (r, zi, y) -> D.logpdf(_zi(D.Poisson(r), zi), y),
        alts = ((:log_rate, :rate, (r, zi) -> log(r)),)),
    :zero_inflated_binomial => (params = (7, 0.35, 0.2),
        doc = "Distributions.MixtureModel([Distributions.Dirac(0), Distributions.Binomial(n, p)], [zi, 1 - zi])",
        xs = [0, 3, 7], ref = (n, p, zi, y) -> D.logpdf(_zi(D.Binomial(n, p), zi), y),
        alts = ((:logit, :p, (n, p, zi) -> log(p / (1 - p))),)),
    :zero_inflated_binomial_logit_zi => (params = (7, 0.35, -1.2),
        doc = "zero_inflated_binomial(n, p, 1 / (1 + exp(-zi_logit)))", xs = [0, 3, 7],
        ref = (n, p, t, y) -> D.logpdf(_zi(D.Binomial(n, p), _logistic(t)), y),
        alts = ((:logit, :p, (n, p, t) -> log(p / (1 - p))),)),
    :gumbel => (params = (0.3, 1.7), doc = "Distributions.Gumbel(location, scale)",
        xs = [-1.3, 0.0, 2.2], ref = (l, s, x) -> D.logpdf(D.Gumbel(l, s), x),
        alts = ((:log_scale, :scale, (l, s) -> log(s)),)),
    :rayleigh => (params = (1.3,), doc = "Distributions.Rayleigh(scale)",
        xs = [0.3, 1.0, 4.0], ref = (s, x) -> D.logpdf(D.Rayleigh(s), x),
        alts = ((:log_scale, :scale, s -> log(s)),)),
    :chi_squared => (params = (3.5,), doc = "Distributions.Chisq(nu)",
        xs = [0.3, 1.0, 4.0], ref = (nu, x) -> D.logpdf(D.Chisq(nu), x)),
    :chi => (params = (3.5,), doc = "Distributions.Chi(nu)",
        xs = [0.3, 1.0, 4.0], ref = (nu, x) -> D.logpdf(D.Chi(nu), x)),
    :inverse_gaussian => (params = (1.5, 2.0), doc = "Distributions.InverseGaussian(mu, lambda)",
        xs = [0.3, 1.0, 4.0], ref = (m, l, x) -> D.logpdf(D.InverseGaussian(m, l), x)),
    :beta_prime => (params = (2.0, 3.0), doc = "Distributions.BetaPrime(alpha, beta)",
        xs = [0.3, 1.0, 4.0], ref = (a, b, x) -> D.logpdf(D.BetaPrime(a, b), x)),
    :f_dist => (params = (4.0, 6.0), doc = "Distributions.FDist(nu1, nu2)",
        xs = [0.3, 1.0, 4.0], ref = (a, b, x) -> D.logpdf(D.FDist(a, b), x)),
    :dirichlet_multinomial => (params = (6, [0.2, 0.5, 0.3]),
        doc = "Distributions.DirichletMultinomial(n, alpha)", xs = [[2, 3, 1], [0, 6, 0]],
        ref = (n, a, y) -> D.logpdf(D.DirichletMultinomial(n, a), y)),
    :bernoulli_logit_glm => (params = (_X, [0.3, -0.8]), doc = "Distributions.Bernoulli",
        xs = [[1, 0, 1]],
        ref = (X, b, y) -> sum(D.logpdf.(D.Bernoulli.(_logistic.(X * b)), y))),
    :poisson_log_glm => (params = (_X, [0.3, -0.8]), doc = "Distributions.Poisson",
        xs = [[0, 3, 1]], ref = (X, b, y) -> sum(D.logpdf.(D.Poisson.(exp.(X * b)), y))),
    :normal_id_glm => (params = (_X, [0.3, -0.8], 1.3), doc = "Distributions.Normal",
        xs = [[0.2, -1.1, 0.9]],
        ref = (X, b, s, y) -> sum(D.logpdf.(D.Normal.(X * b, s), y))),
)

# The two truncated-moment objects have no density; their endpoints are the
# moments of the truncated law the docstring names. The registered
# Distributions.jl does not compute those, so the reference is the closed form
# in 256-bit arithmetic.
function _truncated_lognormal_moments(location, scale, bound, upper::Bool)
    setprecision(BigFloat, 256) do
        m, s, b = big(location), big(scale), big(bound)
        z = (log(b) - m) / s
        tail(x) = SpecialFunctions.erfc((upper ? -x : x) / sqrt(big(2))) / 2
        first = exp(m + s^2 / 2) * tail(z - s) / tail(z)
        second = exp(2m + 2s^2) * tail(z - 2s) / tail(z)
        Float64(first), Float64(second - first^2)
    end
end
const MOMENTS = Dict{Symbol,Any}(
    :lower_truncated_lognormal => (params = (0.3, 0.8, 1.1),
        doc = "Distributions.truncated(Distributions.LogNormal(location, scale); lower)",
        ref = (l, s, b) -> _truncated_lognormal_moments(l, s, b, false)),
    :upper_truncated_lognormal => (params = (0.3, 0.8, 1.1),
        doc = "Distributions.truncated(Distributions.LogNormal(location, scale); upper)",
        ref = (l, s, b) -> _truncated_lognormal_moments(l, s, b, true)),
)

@testset "law docstrings state each law's parameters" begin
    @test length(LAWS) >= 40
    @test sort(collect(union(keys(CASES), keys(MOMENTS)))) == LAWS
    for name in LAWS
        law = getfield(Laws, name)
        doc = _docstring(name)
        @testset "$name" begin
            @test doc !== nothing
            doc === nothing && continue
            signatures = _signatures(name, doc)
            @test !isempty(signatures)
            skeleton = RK._kernel_object_skeleton(law)
            bound = keys(RK._kernel_object_bindings(law))
            kernel_ports = Tuple(port for port in
                RK._kernel_endpoint_templates(skeleton).binding_names if !(port in bound))
            # An object with an unbound kernel-valued port is bound to a law
            # outside a kernel first, as `location_scale(standard_normal)`.
            owner = isempty(kernel_ports) ? _owner_formals(law, _primary(law)) : kernel_ports
            ports = RK.kernel_port_names(skeleton)
            for (named, fields) in signatures
                if named
                    @test all(in(ports), fields)
                else
                    # A positional construction binds exactly these, in order.
                    @test fields == owner
                end
            end
            listed = Set(Symbol(m.captures[1]) for m in
                eachmatch(r"`([A-Za-z_][A-Za-z_0-9]*)\(", _paragraph(doc, "Endpoints")))
            @test listed == Set(RK.kernel_endpoint_names(law))
            case = haskey(CASES, name) ? CASES[name] : MOMENTS[name]
            @test occursin(case.doc, doc)
            # Every listed binding is checked numerically below; the paragraph
            # may also name the parameters those bindings replace.
            documented = Set(Symbol(m.captures[1]) for m in
                eachmatch(r"`([A-Za-z_][A-Za-z_0-9]*)`", _paragraph(doc, "Named bindings")))
            alts = get(case, :alts, ())
            checked = Set(alt[1] for alt in alts)
            replaced = Set(alt[2] for alt in alts)
            @test checked ⊆ documented
            @test documented ⊆ union(checked, replaced)
        end
    end
end

@testset "law parameters match the documented Distributions.jl law" begin
    for (name, case) in CASES
        law = get(case, :law, getfield(Laws, name))
        formals = get(case, :formals, _owner_formals(law, :logpdf))
        explicit = _explicit_formals(law, :logpdf)
        routes = [(formals, case.params)]
        for (alt, replaced, value) in get(case, :alts, ())
            i = findfirst(==(replaced), formals)
            @test i !== nothing
            i === nothing && continue
            push!(routes, (Base.setindex(formals, alt, i),
                Base.setindex(case.params, value(case.params...), i)))
        end
        @testset "$name via $(route[1])" for route in routes
            density = prepare(extract(law; have = (route[1]..., explicit...), want = :logpdf))
            for x in case.xs
                @test density(route[2]..., x) ≈ case.ref(case.params..., x)
            end
        end
    end
    for (name, case) in MOMENTS
        law = getfield(Laws, name)
        mean, var = case.ref(case.params...)
        @test prepare(extract(law; want = :mean))(case.params...) ≈ mean
        @test prepare(extract(law; want = :var))(case.params...) ≈ var
    end
end

# A positional construction inside a kernel binds the documented order, and a
# named one selects another route the object carries.
@kernel _gamma_positional(x::Float64, shape::Float64, rate::Float64) =
    Laws.gamma(shape, rate).logpdf(x)
@kernel _gamma_named_scale(x::Float64, shape::Float64, θ::Float64) =
    Laws.gamma(; shape, scale = θ).logpdf(x)
@kernel _exponential_mean(x::Float64, mean::Float64) =
    Laws.exponential(mean).logpdf(x)

@testset "documented constructions inside a kernel" begin
    @test prepare(_gamma_positional)(1.2, 2.5, 1.5) ≈ D.logpdf(D.Gamma(2.5, 1 / 1.5), 1.2)
    @test prepare(_gamma_named_scale)(1.2, 2.5, 0.4) ≈ D.logpdf(D.Gamma(2.5, 0.4), 1.2)
    @test prepare(_exponential_mean)(1.2, 0.4) ≈ D.logpdf(D.Exponential(0.4), 1.2)
end
end
