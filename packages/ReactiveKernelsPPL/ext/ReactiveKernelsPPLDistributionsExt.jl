module ReactiveKernelsPPLDistributionsExt

import Distributions
import ReactiveKernelsPPL

# Distribution objects use their ordinary public logpdf. Parameter geometry
# is a separate capability; it is never inferred by evaluating active args.
ReactiveKernelsPPL.sampling_logdensity(d::Distributions.Distribution, value) =
    Distributions.logpdf(d, value)

end
