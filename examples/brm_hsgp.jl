module BRMHSGPExample

using ReactiveKernels, ReactiveKernelsPPL
import BayesianRegressionModels
using Statistics
using SHA

const MCYCLE_PATH = joinpath(pkgdir(BayesianRegressionModels), "research",
    "adaptive_centering", "mcycle.csv")
const MODEL_SOURCE = joinpath(pkgdir(BayesianRegressionModels), "ext",
    "rk_statistical_gp.jl")
const model = BayesianRegressionModels.rk_model(:dual_hsgp)

const MCYCLE_SHA256 = "b89a1e4eb0391a982b32be3e378df00e8593ff9971e9425e9c5d7929b74f9801"

"""Read the exact MASS motorcycle observations used by BRM's case study."""
function motorcycle_data(path=MCYCLE_PATH)
    bytes2hex(sha256(read(path))) == MCYCLE_SHA256 || error("mcycle data hash mismatch")
    lines = readlines(path)
    first(lines) == "rownames,times,accel" || error("unexpected mcycle header")
    rows = split.(lines[2:end], ',')
    times = parse.(Float64, getindex.(rows, 2))
    accel = parse.(Float64, getindex.(rows, 3))
    length(times) == 133 || error("expected 133 observations")
    lo, hi = extrema(times)
    x = @. -1 + 2 * (times - lo) / (hi - lo)
    (; x, y=accel ./ std(accel))
end

# The statistical graph and preparation are owned by BRM. This optional
# consumer preserves the benchmark's loader and call boundary.
prepare_model(data; want=:posterior) =
    BayesianRegressionModels.StatisticalPreparation.prepare_dual_hsgp(data; want)

end
