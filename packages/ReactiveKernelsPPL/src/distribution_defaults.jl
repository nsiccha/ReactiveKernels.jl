# Positional defaults belong to the distribution constructor, on every
# lowering path. These expand fixed call syntax, never data-dependent work.
_distribution_args(head::Symbol, args) = _distribution_args(Val(head), args)
_distribution_args(::Val, args) = args

function _trailing_distribution_defaults(args, defaults)
    length(args) >= length(defaults) && return args
    return Any[args; defaults[(length(args) + 1):end]...]
end

_distribution_args(::Val{:Normal}, args) =
    _trailing_distribution_defaults(args, (0.0, 1.0))
_distribution_args(::Val{:Cauchy}, args) =
    _trailing_distribution_defaults(args, (0.0, 1.0))
_distribution_args(::Val{:Laplace}, args) =
    _trailing_distribution_defaults(args, (0.0, 1.0))
_distribution_args(::Val{:Logistic}, args) =
    _trailing_distribution_defaults(args, (0.0, 1.0))
_distribution_args(::Val{:LogNormal}, args) =
    _trailing_distribution_defaults(args, (0.0, 1.0))
_distribution_args(::Val{:Gamma}, args) =
    _trailing_distribution_defaults(args, (1.0, 1.0))
_distribution_args(::Val{:InverseGamma}, args) =
    _trailing_distribution_defaults(args, (1.0, 1.0))
_distribution_args(::Val{:Weibull}, args) =
    _trailing_distribution_defaults(args, (1.0, 1.0))
_distribution_args(::Val{:InverseGaussian}, args) =
    _trailing_distribution_defaults(args, (1.0, 1.0))
_distribution_args(::Val{:NegativeBinomial}, args) =
    _trailing_distribution_defaults(args, (1.0, 0.5))
_distribution_args(::Val{:Exponential}, args) =
    _trailing_distribution_defaults(args, (1.0,))
_distribution_args(::Val{:Uniform}, args) =
    isempty(args) ? Any[0.0, 1.0] : args
_distribution_args(::Val{:Beta}, args) =
    isempty(args) ? Any[1.0, 1.0] :
    length(args) == 1 ? Any[args[1], args[1]] : args
_distribution_args(::Val{:VonMises}, args) =
    isempty(args) ? Any[0.0, 1.0] :
    length(args) == 1 ? Any[0.0, args[1]] : args
function _distribution_args(::Val{:TDist}, args)
    length(args) == 1 || _sfail("TDist expects one degrees-of-freedom argument")
    return Any[args[1], 0.0, 1.0]
end

# Each constructor argument's role, by position, in the distribution
# kernels' vocabulary (`normal(location, scale)`). A value the lowering
# computes for one argument is named after its owner and that role
# (`y .~ Normal.(mu, f(x))` names `y_scale`). One row per family; a family
# without a row names a computed argument after its function.
const _DISTRIBUTION_ROLES = Dict{Symbol,Tuple{Vararg{Symbol}}}(
    :Normal => (:location, :scale), :Cauchy => (:location, :scale),
    :Laplace => (:location, :scale), :Logistic => (:location, :scale),
    :LogNormal => (:location, :scale), :StudentT => (:nu, :location, :scale),
    :TDist => (:nu,), :Exponential => (:scale,),
    :HalfNormal => (:scale,), :HalfCauchy => (:scale,),
    :Gamma => (:shape, :scale), :InverseGamma => (:shape, :scale),
    :Weibull => (:shape, :scale), :Beta => (:alpha, :beta),
    :Uniform => (:lower, :upper), :Poisson => (:rate,), :Bernoulli => (:p,),
    :Binomial => (:n, :p), :NegativeBinomial => (:r, :p),
    :InverseGaussian => (:location, :shape),
    :VonMises => (:location, :concentration),
    :ZeroInflatedPoisson => (:rate, :zi), :ZeroInflatedBinomial => (:n, :p, :zi),
)

# The role of argument `position` of `head(args...)` with `nargs` authored
# arguments, or `nothing`. One-argument forms keep `_distribution_args`'
# meaning: `VonMises(k)` is a concentration and `Beta(k)` both shapes.
function _distribution_role(head::Symbol, position::Int, nargs::Int)
    head === :VonMises && nargs == 1 && return :concentration
    head === :Beta && nargs == 1 && return :shape
    roles = get(_DISTRIBUTION_ROLES, head, ())
    return position in eachindex(roles) ? roles[position] : nothing
end
_distribution_role(head, position, nargs) = nothing

# The role of argument `position` of a prior family (`:normal`) with all
# `nargs` arguments, through its constructor head (`_PARAM_FAMILIES`,
# `_COEF_FAMILIES`).
function _family_role(family::Symbol, position::Int, nargs::Int)
    for table in (_PARAM_FAMILIES, _COEF_FAMILIES), (head, f) in table
        f === family && head !== :TDist &&
            return _distribution_role(head, position, nargs)
    end
    return nothing
end

# A literal in a log-linked response slot is a constrained value. Preserve
# its value through the existing link-space location representation.
_exp_response_location(lhs, arg::Real) = log(arg)
_exp_response_location(lhs, arg) = _lower_link_arg(lhs, arg, :exp)
