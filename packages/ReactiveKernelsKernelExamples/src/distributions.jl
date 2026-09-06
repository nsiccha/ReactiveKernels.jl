# Executable native distribution-kernel examples for a future PPL layer.
#
# The location-scale examples share one transparent object graph. The compute
# path contains no `Distributions.jl` call; that package is an independent
# correctness/allocation oracle.
module DistributionExamples

export CONTINUOUS_SOURCE, DISCRETE_SOURCE, VECTORIZED_SOURCE
export CAUCHY_SOURCE, LAPLACE_SOURCE, LOGNORMAL_SOURCE
export HAVE_ROUTES_SOURCE, EXTRACT_JOINT_SOURCE
export BROADCAST_REF_SOURCE, INVARIANT_HOISTING_SOURCE
export lowering_sources, run_lowering_source
export LOCATION_SCALE_SOURCE
export normal, cauchy, laplace, bernoulli, lognormal
export exponential, geometric, uniform, mvnormal, ar1
export poisson, gamma, beta, binomial
export NORMAL_LOGDENSITY, CAUCHY_LOGDENSITY, LAPLACE_LOGDENSITY
export EXPONENTIAL_SOURCE, GEOMETRIC_SOURCE, UNIFORM_SOURCE
export MVNORMAL_SOURCE, AR1_SOURCE
export POISSON_SOURCE, GAMMA_SOURCE, BETA_SOURCE, BINOMIAL_SOURCE
export all_sources, evaluate_source, run

using Distributions
using LogExpFunctions: logistic

using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    LOCATION_SCALE_SOURCE, normal, cauchy, laplace,
    BERNOULLI_SOURCE, LOGNORMAL_SOURCE,
    bernoulli, lognormal, exponential, geometric, uniform, mvnormal, ar1,
    poisson, gamma, beta, binomial,
    NORMAL_LOGDENSITY, CAUCHY_LOGDENSITY, LAPLACE_LOGDENSITY,
    EXPONENTIAL_SOURCE, GEOMETRIC_SOURCE, UNIFORM_SOURCE,
    MVNORMAL_SOURCE, AR1_SOURCE,
    POISSON_SOURCE, GAMMA_SOURCE, BETA_SOURCE, BINOMIAL_SOURCE

_allocated(f, a, b) = @allocated f(a, b)
_allocated(f, a, b, c) = @allocated f(a, b, c)
_allocated(f, a, b, c, d) = @allocated f(a, b, c, d)

const CONTINUOUS_SOURCE = LOCATION_SCALE_SOURCE * raw"""

normal_kernel = prepare(normal.logpdf)

inputs = (; location = -0.2, scale = 1.3, x = 0.4)
output = normal_kernel(Tuple(inputs)...)

docs_example = (;
    name = :continuous_normal,
    origin = "shared location-scale Normal object (build executed)",
    inputs,
    spec = normal.logpdf,
    kernel = normal_kernel,
    output,
)
"""

const DISCRETE_SOURCE = BERNOULLI_SOURCE

const VECTORIZED_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal

@kernel normal_loglik(x, location, scale) = begin
    pointwise = plate(x, location, scale) do xi, li, si
        normal(li, si).logpdf(xi)
    end
    return sum(pointwise)
end

vectorized = prepare(normal_loglik)
pointwise = prepare(extract(normal_loglik; want = :pointwise))
both = prepare(extract(
    normal_loglik; want = (:pointwise, :__return__)))

x = collect(range(-1.5, 1.5; length = 8))
location = 0.3
scale = 1.2
inputs = (; x, location, scale)
output = vectorized(x, location, scale)
pointwise_output = pointwise(x, location, scale)
both_output = both(x, location, scale)

docs_example = (;
    name = :vectorized_normal,
    origin = "authored Normal likelihood via one transparent `plate` block (build executed)",
    inputs,
    spec = normal_loglik,
    kernel = vectorized,
    output,
    pointwise_output,
    both_output,
)
"""

const CAUCHY_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: cauchy

cauchy_kernel = prepare(cauchy.logpdf)

inputs = (; location = -0.3, scale = 1.1, x = 2.4)
output = cauchy_kernel(Tuple(inputs)...)

docs_example = (;
    name = :cauchy_heavy_tail,
    origin = "native Cauchy log density (build executed)",
    inputs,
    spec = cauchy.logpdf,
    kernel = cauchy_kernel,
    output,
)
"""

const LAPLACE_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: laplace

laplace_kernel = prepare(laplace.logpdf)

inputs = (; location = 0.2, scale = 0.8, x = -1.7)
output = laplace_kernel(Tuple(inputs)...)

docs_example = (;
    name = :laplace_sharp_peak,
    origin = "native Laplace log density (build executed)",
    inputs,
    kernel = laplace_kernel,
    output,
)
"""

# --- Interactive lowering-focused panels ------------------------------------
# These demonstrate authoring/planning semantics through build-executed asserts
# rather than a Distributions oracle, so they are NOT in `all_sources()` (which
# feeds the oracle checker). Each self-verifies the lowering property it teaches.

# C1: one object, three authoritative scale/log-scale HAVE routes.
const HAVE_ROUTES_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal

scale_kernel    = prepare(normal.logpdf; have = (:x, :location, :scale),     want = :logpdf)
logscale_kernel = prepare(normal.logpdf; have = (:x, :location, :log_scale), want = :logpdf)
both_kernel     = prepare(normal.logpdf; have = (:x, :location, :scale, :log_scale), want = :logpdf)

location = -0.2; scale = 1.3; log_scale = log(scale); x = 0.4
v_scale    = scale_kernel(x, location, scale)
v_logscale = logscale_kernel(x, location, log_scale)
v_both     = both_kernel(x, location, scale, log_scale)
@assert v_scale ≈ v_logscale ≈ v_both

# The plan differs by route: supplying scale computes log_scale, supplying
# log_scale computes scale, supplying both recomputes and validates neither.
recipe_outputs(k) = [only(r.outputs).name for r in k.plan.recipes]
@assert :log_scale in recipe_outputs(scale_kernel)
@assert !(:scale in recipe_outputs(scale_kernel))
@assert :scale in recipe_outputs(logscale_kernel)
@assert !(:log_scale in recipe_outputs(logscale_kernel))
@assert !(:scale in recipe_outputs(both_kernel))
@assert !(:log_scale in recipe_outputs(both_kernel))

docs_example = (; name = :normal_have_routes,
    origin = "one Normal object, three authoritative scale/log-scale HAVE routes (build executed)",
    inputs = (; x, location, scale), spec = normal.logpdf,
    kernel = scale_kernel, output = v_scale)
"""

# C2: extract a joint boundary; the standardized residual is shared (CSE).
const EXTRACT_JOINT_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal

joint = extract(normal; have = (:x, :location, :scale), want = (:logpdf, :cdf))
joint_kernel = prepare(joint)
location = -0.2; scale = 1.3; x = 0.4
lp, c = joint_kernel(x, location, scale)

# Both endpoints reuse ONE standardized node (structural CSE), not two.
@assert count(r -> only(r.outputs).name === :standardized, plan(joint).recipes) == 1

docs_example = (; name = :normal_extract_joint,
    origin = "extract logpdf and cdf together; the standardized residual is shared (build executed)",
    inputs = (; x, location, scale), spec = joint,
    kernel = joint_kernel, output = (lp, c))
"""

# C3: natural broadcast semantics with a Ref-wrapped atomic array argument.
const BROADCAST_REF_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal

# x zips with mu (both length-N); scale repeats (scalar); baseline is a whole
# vector passed atomically with Ref, so each body call sees all of it.
@kernel broadcast_ref_demo(x, mu, scale, baseline) = begin
    pointwise = plate(x, mu, scale, Ref(baseline)) do xi, mui, si, base
        normal(mui + sum(base), si).logpdf(xi)
    end
    return sum(pointwise)
end

x = [0.4, -1.1, 0.7, 0.2]
mu = [0.0, 0.5, -0.3, 0.1]
scale = 1.2
baseline = [0.1, -0.2, 0.05]
kernel = prepare(broadcast_ref_demo)
output = kernel(x, mu, scale, baseline)

docs_example = (; name = :normal_broadcast_ref,
    origin = "plate with zipped observations, a repeated scalar, and a Ref atom (build executed)",
    inputs = (; x, mu, scale, baseline), spec = broadcast_ref_demo,
    kernel = kernel, output = output)
"""

# C4: dependency-aware lowering hoists the scale-only invariant out of the loop.
const INVARIANT_HOISTING_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal

# Batched Normal log-likelihood: x varies over the batch; location and scale are
# shared. Scale-only work (log_scale) is a complete invariant and is hoisted to
# the loop preamble; standardization depends on x and stays in the loop.
@kernel normal_batch_loglik(x, location, scale) = begin
    pointwise = plate(x, location, scale) do xi, li, si
        normal(li, si).logpdf(xi)
    end
    return sum(pointwise)
end

x = collect(range(-1.5, 1.5; length = 8)); location = 0.3; scale = 1.2
kernel = prepare(normal_batch_loglik)
output = kernel(x, location, scale)

# The generated kernel is dependency-aware: the scale-only log_scale op is
# emitted ONCE in the hoisted preamble, before the batch loop.
generated = string(code_expr(kernel))
@assert occursin("_authored_plate_is_axis", generated)
@assert first(findfirst("plate_log_scale", generated)) < first(findfirst("for ", generated))

docs_example = (; name = :normal_invariant_hoisting,
    origin = "batched Normal plate: scale-only work hoisted out of the loop (build executed)",
    inputs = (; x, location, scale), spec = normal_batch_loglik,
    kernel = kernel, output = output)
"""

lowering_sources() = (
    HAVE_ROUTES_SOURCE, EXTRACT_JOINT_SOURCE,
    BROADCAST_REF_SOURCE, INVARIANT_HOISTING_SOURCE,
)

# Sandbox-evaluate a lowering panel exactly as the docs render it; the source's
# own build-executed `@assert`s verify the lowering property it teaches.
function run_lowering_source(source::AbstractString)
    sandbox = Module(gensym(:LoweringExample), true, true)
    Core.eval(sandbox, :(using ReactiveKernels))
    parsed = Meta.parseall(source; filename = "lowering-example.jl")
    expressions = parsed.head === :toplevel ? parsed.args : Any[parsed]
    for expression in expressions
        expression isa LineNumberNode && continue
        Core.eval(sandbox, expression)
    end
    Core.eval(sandbox, :docs_example)
end

all_sources() = (
    CONTINUOUS_SOURCE, DISCRETE_SOURCE, VECTORIZED_SOURCE,
    CAUCHY_SOURCE, LAPLACE_SOURCE, LOGNORMAL_SOURCE,
    EXPONENTIAL_SOURCE, GEOMETRIC_SOURCE, UNIFORM_SOURCE,
    MVNORMAL_SOURCE, AR1_SOURCE,
    POISSON_SOURCE, GAMMA_SOURCE, BETA_SOURCE, BINOMIAL_SOURCE,
)

function evaluate_source(source::AbstractString)
    sandbox = Module(gensym(:DistributionExample), true, true)
    Core.eval(sandbox, :(using ReactiveKernels))
    parsed = Meta.parseall(source; filename = "distribution-example.jl")
    expressions = parsed.head === :toplevel ? parsed.args : Any[parsed]
    for expression in expressions
        expression isa LineNumberNode && continue
        Core.eval(sandbox, expression)
    end
    artifact = Core.eval(sandbox, :docs_example)
    inputs = Tuple(artifact.inputs)
    argtypes = Tuple{map(typeof, inputs)...}
    inferred_return = only(Base.return_types(artifact.kernel, argtypes))
    allocated_bytes = Base.invokelatest(_allocated, artifact.kernel, inputs...)

    if artifact.name === :continuous_normal
        location, scale, x = inputs
        reference_call = (location, scale, x) ->
            logpdf(Normal(location, scale), x)
        reference = reference_call(inputs...)
        reference_allocated_bytes =
            _allocated(reference_call, location, scale, x)
        return merge(artifact, (;
            reference, allocated_bytes, reference_allocated_bytes, inferred_return,
        ))
    elseif artifact.name === :discrete_bernoulli_logit
        observed, logit = inputs
        reference_call = (observed, logit) ->
            logpdf(Bernoulli(logistic(logit)), observed)
        reference = reference_call(inputs...)
        reference_allocated_bytes = _allocated(reference_call, observed, logit)
        return merge(artifact, (;
            reference, allocated_bytes, reference_allocated_bytes, inferred_return,
        ))
    elseif artifact.name === :vectorized_normal
        x, location, scale = inputs
        reference_call = (x, location, scale) ->
            sum(logpdf.(Normal(location, scale), x))
        reference = reference_call(inputs...)
        reference_allocated_bytes =
            _allocated(reference_call, x, location, scale)
        per_obs = artifact.pointwise_output
        @assert artifact.both_output == (per_obs, artifact.output)
        return merge(artifact, (;
            reference, per_obs, allocated_bytes,
            reference_allocated_bytes, inferred_return,
        ))
    elseif artifact.name === :cauchy_heavy_tail
        location, scale, x = inputs
        reference_call = (location, scale, x) ->
            logpdf(Cauchy(location, scale), x)
        reference = reference_call(inputs...)
        reference_allocated_bytes =
            _allocated(reference_call, location, scale, x)
    elseif artifact.name === :laplace_sharp_peak
        location, scale, x = inputs
        reference_call = (location, scale, x) ->
            logpdf(Laplace(location, scale), x)
        reference = reference_call(inputs...)
        reference_allocated_bytes =
            _allocated(reference_call, location, scale, x)
    elseif artifact.name === :lognormal_positive_support
        x, location, log_scale = inputs
        reference_call = (x, location, log_scale) ->
            logpdf(LogNormal(location, exp(log_scale)), x)
        reference = reference_call(inputs...)
        reference_allocated_bytes =
            _allocated(reference_call, x, location, log_scale)
    elseif artifact.name === :exponential_logscale
        x, log_scale = inputs
        reference_call = (x, log_scale) ->
            logpdf(Exponential(exp(log_scale)), x)
        reference = reference_call(inputs...)
        reference_allocated_bytes = _allocated(reference_call, x, log_scale)
    elseif artifact.name === :geometric_logit
        observed, logitp = inputs
        reference_call = (observed, logitp) ->
            logpdf(Geometric(logistic(logitp)), observed)
        reference = reference_call(inputs...)
        reference_allocated_bytes = _allocated(reference_call, observed, logitp)
    elseif artifact.name === :uniform_bounded
        x, lower, upper = inputs
        reference_call = (x, lower, upper) -> logpdf(Uniform(lower, upper), x)
        reference = reference_call(inputs...)
        reference_allocated_bytes = _allocated(reference_call, x, lower, upper)
    elseif artifact.name === :multivariate_normal_have_want
        x, μ, chol = inputs
        reference_call = (x, μ, chol) -> logpdf(MvNormal(μ, chol * chol'), x)
        reference = reference_call(inputs...)
        reference_allocated_bytes = _allocated(reference_call, x, μ, chol)
    elseif artifact.name === :poisson_lograte
        observed, log_rate = inputs
        reference_call = (observed, log_rate) ->
            logpdf(Poisson(exp(log_rate)), observed)
        reference = reference_call(inputs...)
        reference_allocated_bytes = _allocated(reference_call, observed, log_rate)
    elseif artifact.name === :gamma_shape_rate
        x, shape, log_rate = inputs
        reference_call = (x, shape, log_rate) ->
            logpdf(Gamma(shape, 1 / exp(log_rate)), x)
        reference = reference_call(inputs...)
        reference_allocated_bytes = _allocated(reference_call, x, shape, log_rate)
    elseif artifact.name === :beta_unit_interval
        x, a, b = inputs
        reference_call = (x, a, b) -> logpdf(Beta(a, b), x)
        reference = reference_call(inputs...)
        reference_allocated_bytes = _allocated(reference_call, x, a, b)
    elseif artifact.name === :binomial_logit
        observed, n, logit = inputs
        reference_call = (observed, n, logit) ->
            logpdf(Binomial(n, logistic(logit)), observed)
        reference = reference_call(inputs...)
        reference_allocated_bytes = _allocated(reference_call, observed, n, logit)
    elseif artifact.name === :stationary_ar1
        x, μ, ϕ, log_scale = inputs
        reference_call = function (x, μ, ϕ, log_scale)
            σ = exp(log_scale)
            abs(ϕ) < 1 || return -Inf
            result = logpdf(Normal(μ, σ / sqrt(1 - ϕ^2)), first(x))
            for t in 2:length(x)
                conditional_mean = μ + ϕ * (x[t - 1] - μ)
                result += logpdf(Normal(conditional_mean, σ), x[t])
            end
            result
        end
        reference = reference_call(inputs...)
        reference_allocated_bytes =
            _allocated(reference_call, x, μ, ϕ, log_scale)
    else
        error("unknown distribution example $(artifact.name)")
    end

    merge(artifact, (;
        reference, allocated_bytes, reference_allocated_bytes, inferred_return,
    ))
end

function run(io::IO = stdout)
    artifacts = map(evaluate_source, all_sources())
    for artifact in artifacts
        println(io, artifact.name)
        println(io, "  output: ", artifact.output)
        println(io, "  reference: ", artifact.reference)
        println(io, "  allocated bytes: ", artifact.allocated_bytes)
        println(io, "  oracle allocated bytes: ",
                artifact.reference_allocated_bytes)
        println(io, "  inferred return: ", artifact.inferred_return)
    end
    artifacts
end

end # module DistributionExamples

if abspath(PROGRAM_FILE) == @__FILE__
    DistributionExamples.run()
end
