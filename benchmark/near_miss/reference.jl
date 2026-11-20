# Hand-written plain-Julia row loops for the probe families: what a competent
# author gets from Julia + Enzyme with no model compiler. Priors are included
# so the values can be compared with the RKPPL programs; the per-row loop is
# what the timing measures.
using Enzyme, Chairmarks, Random, LinearAlgebra, Printf
BLAS.set_num_threads(1)

const LOG2PI = log(2π)
@inline nlp(v, m, s) = -0.5 * ((v - m) / s)^2 - log(s) - 0.5 * LOG2PI
@inline l1pe(v) = max(v, 0.0) + log1p(exp(-abs(v)))

function ref_normal(u, x, y)
    a, b, ls = u[1], u[2], u[3]
    sigma = exp(ls)
    lp = nlp(a, 0.0, 5.0) + nlp(b, 0.0, 2.0) - sigma + ls
    s = 0.0
    @inbounds @simd for i in eachindex(y)
        r = y[i] - (a + b * x[i])
        s += r * r
    end
    lp - 0.5 * s / (sigma * sigma) - length(y) * (ls + 0.5 * LOG2PI)
end

function ref_normal_exp(u, x, y)
    a, b, ls = u[1], u[2], u[3]
    sigma = exp(ls)
    lp = nlp(a, 0.0, 1.0) + nlp(b, 0.0, 1.0) - sigma + ls
    s = 0.0
    @inbounds @simd for i in eachindex(y)
        r = y[i] - exp(a + b * x[i])
        s += r * r
    end
    lp - 0.5 * s / (sigma * sigma) - length(y) * (ls + 0.5 * LOG2PI)
end

# Student-t with shared nu and sigma: one log1p per row, normalizer hoisted.
function ref_student_exp(u, x, y, lgamma)
    a, b, ls, lnu = u[1], u[2], u[3], u[4]
    sigma = exp(ls); nu = exp(lnu)
    lp = nlp(a, 0.0, 1.0) + nlp(b, 0.0, 1.0) - sigma + ls
    s = 0.0
    c = 1 / (nu * sigma * sigma)
    @inbounds @simd for i in eachindex(y)
        r = y[i] - exp(a + b * x[i])
        s += log1p(r * r * c)
    end
    norm = lgamma(0.5 * (nu + 1)) - lgamma(0.5 * nu) - 0.5 * log(nu * π) - ls
    lp - 0.5 * (nu + 1) * s + length(y) * norm
end

# Positive observations, with data-only logs available at binding.
function ref_gamma_exp(u, x, y, logy, lgamma)
    a, b, log_shape = u
    shape = exp(log_shape)
    prior = nlp(a, 0.0, 1.0) + nlp(b, 0.0, 1.0) - shape + log_shape
    result = prior + length(y) * (shape * log_shape - lgamma(shape))
    @inbounds for i in eachindex(x, y, logy)
        eta = a + b * x[i]
        result += (shape - 1) * logy[i] - shape * (eta + y[i] * exp(-eta))
    end
    result
end

function ref_weibull_exp(u, x, logy)
    a, b, log_k = u
    k = exp(log_k)
    prior = nlp(a, 0.0, 1.0) + nlp(b, 0.0, 1.0) + nlp(log_k, 0.0, 0.3)
    result = prior + length(logy) * log_k
    @inbounds for i in eachindex(x, logy)
        power = k * (logy[i] - (a + b * x[i]))
        result += power - exp(power) - logy[i]
    end
    result
end

function ref_cauchy_exp(u, x, y)
    a, b, ls = u[1], u[2], u[3]
    sigma = exp(ls)
    lp = nlp(a, 0.0, 1.0) + nlp(b, 0.0, 1.0) - sigma + ls
    s = 0.0
    c = 1 / (sigma * sigma)
    @inbounds @simd for i in eachindex(y)
        r = y[i] - exp(a + b * x[i])
        s += log1p(r * r * c)
    end
    lp - s - length(y) * (log(π) + ls)
end

function ref_bernoulli_logit(u, x, y)
    a, b = u[1], u[2]
    lp = nlp(a, 0.0, 1.0) + nlp(b, 0.0, 2.0)
    s = 0.0
    @inbounds @simd for i in eachindex(y)
        eta = a + b * x[i]
        s += y[i] * eta - l1pe(eta)
    end
    lp + s
end

# Poisson with log link; sum(y .* eta) and the factorials are data sums.
function ref_poisson(u, x, y, sy, sxy, slf)
    a, b = u[1], u[2]
    lp = nlp(a, 0.0, 1.0) + nlp(b, 0.0, 2.0)
    s = 0.0
    @inbounds @simd for i in eachindex(y)
        s += exp(a + b * x[i])
    end
    lp + a * sy + b * sxy - s - slf
end

function ref_normal_group(u, x, y, g, J)
    m, lt, b, ls = u[1], u[2], u[3], u[4]
    t = exp(lt); sigma = exp(ls)
    lp = nlp(m, 0.0, 5.0) - t + lt + nlp(b, 0.0, 2.0) - sigma + ls
    sa = 0.0
    @inbounds for j in 1:J
        r = u[4 + j] - m
        sa += r * r
    end
    lp += -0.5 * sa / (t * t) - J * (lt + 0.5 * LOG2PI)
    s = 0.0
    @inbounds @simd for i in eachindex(y)
        r = y[i] - (u[4 + g[i]] + b * x[i])
        s += r * r
    end
    lp - 0.5 * s / (sigma * sigma) - length(y) * (ls + 0.5 * LOG2PI)
end

function time_ref(name, f, u, args...)
    g = zero(u)
    v = f(u, args...)
    # Value and gradient together, as the RKPPL sampler call returns them: a
    # gradient-only reverse pass would skip the primal's transcendentals.
    grad!() = (fill!(g, 0.0); last(Enzyme.autodiff(Enzyme.ReverseWithPrimal, Enzyme.Const(f),
        Enzyme.Active, Enzyme.Duplicated(u, g), map(Enzyme.Const, args)...)))
    grad!() == v || abs(grad!() - v) <= 1e-9 * abs(v) || error("reference value mismatch for $name")
    bd = @b $f($u, $args...) seconds = 0.5
    bg = @b $grad!() seconds = 0.5
    (; name, value = v, gnorm = norm(g), t_density = bd.time * 1e9, t_gradient = bg.time * 1e9,
        allocs = Int(bg.allocs))
end
