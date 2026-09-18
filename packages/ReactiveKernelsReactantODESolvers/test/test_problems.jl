# Non-toy test problems, out-of-place `(u, p, t) -> du` form.

lotka_volterra(u, p, t) = begin
    α, β, δ, γ = p
    x, y = u
    [α * x - β * x * y, δ * x * y - γ * y]
end

const LOTKA_PARAMS = (1.5, 1.0, 1.0, 3.0)
const LOTKA_U0 = [1.0, 1.0]
const LOTKA_TSPAN = (0.0, 10.0)

vanderpol(u, p, t) = begin
    μ = p
    x, y = u
    [y, μ * (1 - x^2) * y - x]
end

const VDP_MU = 10.0
const VDP_U0 = [2.0, 0.0]
const VDP_TSPAN = (0.0, 30.0)

function brusselator(u, p, t)
    B, α, N = p
    du = similar(u)
    for i in 1:N
        x = u[2i-1]
        y = u[2i]
        x_left = i == 1 ? 1.0 : u[2i-3]
        x_right = i == N ? 1.0 : u[2i+1]
        diffusion = α * N^2 * (x_left - 2x + x_right)
        du[2i-1] = 1 + x^2 * y - (B + 1) * x + diffusion
        du[2i] = B * x - x^2 * y
    end
    du
end

const BRUS_N = 20
const BRUS_PARAMS = (3.0, 0.02, BRUS_N)
const BRUS_U0 = repeat([1.5, 3.0], BRUS_N)
const BRUS_TSPAN = (0.0, 10.0)

exponential_decay(u, p, t) = -p .* u

const DECAY_RATES = [0.5, 1.0, 2.0, 5.0, 10.0]
const DECAY_U0 = ones(5)
const DECAY_TSPAN = (0.0, 5.0)
