module BRMHSGPExample

using ReactiveKernels
using Statistics
using SHA

const MCYCLE_SHA256 = "b89a1e4eb0391a982b32be3e378df00e8593ff9971e9425e9c5d7929b74f9801"

"""Read the exact MASS motorcycle observations used by BRM's case study."""
function motorcycle_data(path)
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

# q = [log(rho_mu), log(sd_mu), v_mu[1:20]...,
#      log(rho_logsigma), log(sd_logsigma), v_logsigma[1:20]...].
# c[j] = 0 is noncentered; c[j] = 1 is centered.
# v = z * exp(c * log_spectral_sd), so w = v * exp((1-c)*log_spectral_sd).
# Centeredness is a live, inactive HAVE: online-selected values reuse the
# same prepared graph and compiled executable.
# BEGIN MOTORCYCLE KERNEL
@kernel model(q::Vector{Float64}, x::Vector{Float64}, y::Vector{Float64},
              c::Vector{Float64}, modes::Vector{Float64}, half_width::Float64) = begin
    frequency = modes .* (pi / (2 * half_width))
    frequency_squared = frequency .^ 2
    basis = sin.((x .+ half_width) * transpose(frequency)) ./ sqrt(half_width)

    log_rho_mu = q[1]
    log_sd_mu = q[2]
    log_rho_sigma = q[23]
    log_sd_sigma = q[24]
    v_mu = q[3:22]
    v_sigma = q[25:44]
    c_mu = c[1:20]
    c_sigma = c[21:40]

    log_scale_mu = log_sd_mu .+ 0.5 * log_rho_mu .+ 0.25 * log(2pi) .-
        0.25 .* exp(2 * log_rho_mu) .* frequency_squared
    log_scale_sigma = log_sd_sigma .+ 0.5 * log_rho_sigma .+ 0.25 * log(2pi) .-
        0.25 .* exp(2 * log_rho_sigma) .* frequency_squared
    z_mu = v_mu .* exp.(-c_mu .* log_scale_mu)
    z_sigma = v_sigma .* exp.(-c_sigma .* log_scale_sigma)
    weights_mu = v_mu .* exp.((1 .- c_mu) .* log_scale_mu)
    weights_sigma = v_sigma .* exp.((1 .- c_sigma) .* log_scale_sigma)
    mu = basis * weights_mu
    log_sigma = basis * weights_sigma

    # LogNormal(0,4) priors plus the four positive-parameter Jacobians.
    hyperprior = -(log_rho_mu^2 + log_sd_mu^2 +
                   log_rho_sigma^2 + log_sd_sigma^2) / 32 -
        4 * log(4) - 2 * log(2pi)
    coordinate_jacobian = -sum(c_mu .* log_scale_mu) - sum(c_sigma .* log_scale_sigma)
    weight_prior = -0.5 * (sum(abs2, z_mu) + sum(abs2, z_sigma)) - 20 * log(2pi)
    pointwise = plate(y, mu, log_sigma) do yi, mui, lsi
        -0.5 * ((yi - mui) * exp(-lsi))^2 - lsi - 0.5 * log(2pi)
    end
    likelihood = sum(pointwise)
    posterior = hyperprior + weight_prior + coordinate_jacobian + likelihood
    return posterior
end
# END MOTORCYCLE KERNEL

"""Prepare the exact k=20 model, folding all data-only design work once."""
function prepare_model(data; want=:posterior)
    length(data.x) == length(data.y) == 133 || throw(DimensionMismatch("expected 133 rows"))
    prepare(model; have=(:q, :c, :x, :y, :modes, :half_width), want,
        bound=(; x=data.x, y=data.y, modes=collect(1.0:20.0), half_width=1.5))
end

end
