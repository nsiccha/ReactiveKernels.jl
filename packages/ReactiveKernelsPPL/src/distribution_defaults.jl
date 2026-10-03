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

# A literal in a log-linked response slot is a constrained value. Preserve
# its value through the existing link-space location representation.
_exp_response_location(lhs, arg::Real) = log(arg)
_exp_response_location(lhs, arg) = _lower_link_arg(lhs, arg, :exp)
