# Run with an environment containing ReactiveKernels, ReactiveKernelsPPL and
# ReactiveKernelsDistributionKernels. This law belongs to the caller.
module CallerOwnedDensityExample
using ReactiveKernels, ReactiveKernelsPPL
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal

# If Z ~ Normal(0, scale), this law describes exp(Z) + lower.
struct PositiveNormal{S,L}
    scale::S
    lower::L
end
PositiveNormal(scale) = PositiveNormal(scale, 0.0)
function ReactiveKernelsPPL.sampling_logdensity(d::PositiveNormal, value)
    delta = value-d.lower
    delta > 0 && d.scale > 0 || return -Inf
    z = log(delta)
    -log(2pi)/2-log(d.scale)-z^2/(2*d.scale^2)-z
end

function ReactiveKernelsPPL.sampling_distribution_graph(::Type{PositiveNormal}, args::Tuple)
    # Select by authored arity, never by evaluating the argument expressions.
    # Each return is a local spec; no registry or global graph binding is used.
    if length(args) == 1
        @kernel unshifted_density(value, scale) = begin
            score = if value > 0 && scale > 0
                z = log(value)
                normal(0.0, scale).logpdf(z) - z
            else
                -Inf
            end
            return score
        end
        return unshifted_density
    elseif length(args) == 2
        @kernel shifted_density(value, scale, lower) = begin
            delta = value-lower
            score = if delta > 0 && scale > 0
                z = log(delta)
                normal(0.0, scale).logpdf(z) - z
            else
                -Inf
            end
            return score
        end
        return shifted_density
    end
    nothing
end

function build_example(arity; y=[1.3,2.1])
    ast = arity == 1 ? quote
        a ~ Normal(0,1)
        y .~ PositiveNormal.(exp(a))
    end : quote
        a ~ Normal(0,1)
        lower ~ Normal(0,1)
        y .~ PositiveNormal.(exp(a),lower)
    end
    data = (;y)
    bound = bind_data(lower_rkppl(ast,data;mod=@__MODULE__,conditioned=(:y,)),data)
    built = build_kernel(bound)
    values = arity == 1 ? (;a=0.2) : (;a=0.2,lower=0.4)
    u = unconstrain(built.layout,values)
    (;data,bound,built,u)
end

# An observation needs no geometry. Parameter use supplies sampling_geometry
# separately; the -z above is the law's change-of-variable term, not the
# sampler-coordinate Jacobian.
if abspath(PROGRAM_FILE) == @__FILE__
    for arity in (1,2)
        fx = build_example(arity)
        q = prepare_query(fx.built,fx.bound,:sampler)
        println("arity=",arity," log density=",Base.invokelatest(q,fx.u))
    end
end
end
