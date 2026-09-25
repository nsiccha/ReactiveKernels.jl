module BranchPartition

using ReactiveKernels

# Instrumentation counts arm evaluations (a semantic control for laziness).
const arm_calls = Ref(0)
counted_log(x) = (arm_calls[] += 1; log(x))

# The `log` arm is invalid (DomainError) on the lanes it does not take.
@kernel guarded(x::Vector{Float64}, y::Vector{Float64}) = begin
    pointwise = plate(x, y) do xi, yi
        cell::Float64 = yi > 0 ? counted_log(xi * yi) : xi - 1.0
        cell
    end
    total::Float64 = sum(pointwise)
end

# Three arms by level; the interior arm gathers `c[l]`/`c[l-1]`, which are out
# of bounds on the first and last levels, and the nested condition is itself
# never evaluated on first-level lanes.
# (`nlev` is bound data; a condition reading a live input's shape, such as
# `length(c)`, is not bound data and would stay a lazy branch.)
@kernel leveled(eta::Vector{Float64}, level::Vector{Int}, cuts::Vector{Float64},
                w::Vector{Float64}, nlev::Int) = begin
    pointwise = plate(level, eta, Ref(cuts), w, nlev) do l, e, c, wi, k
        arm::Float64 = l == 1 ? c[1] - e :
            (l == k ? e - c[k - 1] : c[l] * e - c[l - 1])
        wi * arm
    end
    total::Float64 = sum(pointwise)
end

# A bound-only condition computed in the cell first (a cached frontier).
@kernel derived_condition(x::Vector{Float64}, y::Vector{Float64}) = begin
    pointwise = plate(x, y) do xi, yi
        valid::Bool = yi >= 1.0
        cell::Float64 = valid ? counted_log(xi) * yi : 2xi
        cell
    end
    total::Float64 = sum(pointwise)
end

# A constant fallback arm reads no lane argument; split off, it must still
# contribute once per lane.
@kernel constant_fallback(x::Vector{Float64}, y::Vector{Float64}) = begin
    pointwise = plate(x, y) do xi, yi
        cell::Float64 = yi > 0 ? xi * yi : 1.0
        cell
    end
    total::Float64 = sum(pointwise)
end

# A condition on a live value is never partitioned: it stays a lazy branch.
@kernel live_condition(x::Vector{Float64}, s::Float64) = begin
    pointwise = plate(x, s) do xi, si
        cell::Float64 = xi > si ? xi - si : si - xi
        cell
    end
    total::Float64 = sum(pointwise)
end

# A compact Normal-shaped object for constructed endpoints under branch arms.
@kernel endpoint_standard_normal() = begin
    logpdf(z::Float64)::Float64 = -0.5 * log(2π) - 0.5z^2
end

@kernel endpoint_location_scale(standard, location::Float64, scale::Float64) = begin
    log_scale::Float64 = log(scale)
    scale::Float64 = exp(log_scale)
    standardized(x::Float64)::Float64 = (x - location) / scale
    logpdf(x::Float64)::Float64 = begin
        z::Float64 = standardized(x)
        standard.logpdf(z) - log_scale
    end
end

@kernel endpoint_normal = endpoint_location_scale(endpoint_standard_normal)

# A constructed endpoint inside a ternary arm (a missingness guard): each
# taken arm carries its own endpoint evaluation after the split.
@kernel guarded_endpoint(y_full::Vector{Float64}, lp::Vector{Float64},
                         mis_pos::Vector{Int}) = begin
    pointwise = plate(y_full, lp, mis_pos) do yf, lpi, mp
        cell::Float64 = mp == 0 ? endpoint_normal(lpi, 1.5).logpdf(yf) : 0.0 * yf
        cell
    end
    total::Float64 = sum(pointwise)
end

# The endpoint is invalid off-arm (`log` of a nonpositive scale): it must
# never evaluate on lanes that do not take its arm, bound or not.
@kernel lazy_endpoint_arm(y::Vector{Float64}, m::Vector{Float64},
                          g::Vector{Int}) = begin
    pointwise = plate(y, m, g) do yf, mf, gf
        cell::Float64 = gf == 0 ? endpoint_normal(0.0, yf).logpdf(mf) : 0.0
        cell
    end
    total::Float64 = sum(pointwise)
end

# A value-combining call around an in-arm endpoint stays lazy too: no
# eager temporary is hoisted above the branch for it.
@kernel nested_endpoint_arm(y::Vector{Float64}, m::Vector{Float64},
                            g::Vector{Int}) = begin
    pointwise = plate(y, m, g) do yf, mf, gf
        cell::Float64 = gf == 0 ? 2 * endpoint_normal(0.0, yf).logpdf(mf) : 0.0
        cell
    end
    total::Float64 = sum(pointwise)
end

@kernel else_endpoint_arm(y::Vector{Float64}, m::Vector{Float64},
                          g::Vector{Int}) = begin
    pointwise = plate(y, m, g) do yf, mf, gf
        cell::Float64 = gf == 0 ? 0.0 : endpoint_normal(mf, 1.5).logpdf(yf)
        cell
    end
    total::Float64 = sum(pointwise)
end

@kernel named_endpoint_arm(y::Vector{Float64}, m::Vector{Float64},
                           s::Vector{Float64}, g::Vector{Int}) = begin
    pointwise = plate(y, m, s, g) do yf, mf, sf, gf
        cell::Float64 = gf == 0 ?
            endpoint_normal(; location = mf, scale = sf).logpdf(yf) : 0.0
        cell
    end
    total::Float64 = sum(pointwise)
end

@kernel computed_endpoint_arm(y::Vector{Float64}, m::Vector{Float64},
                              g::Vector{Int}) = begin
    pointwise = plate(y, m, g) do yf, mf, gf
        cell::Float64 = gf == 0 ?
            endpoint_normal(mf + 0.25, exp(mf)).logpdf(log(abs(yf) + 1)) : 0.0
        cell
    end
    total::Float64 = sum(pointwise)
end

@kernel and_endpoint_arm(y::Vector{Float64}, m::Vector{Float64},
                         g::Vector{Int}) = begin
    pointwise = plate(y, m, g) do yf, mf, gf
        cell::Float64 = (gf == 0 && yf > 0) ?
            endpoint_normal(mf, 1.5).logpdf(yf) : 0.0
        cell
    end
    total::Float64 = sum(pointwise)
end

# An endpoint in the branch condition (a strict position) keeps the
# established splice path.
@kernel condition_endpoint(y::Vector{Float64}, m::Vector{Float64}) = begin
    pointwise = plate(y, m) do yf, mf
        cell::Float64 = endpoint_normal(mf, 1.5).logpdf(yf) > -2.0 ? 1.0 : -1.0
        cell
    end
    total::Float64 = sum(pointwise)
end

@kernel toplevel_endpoint_arm(y::Float64, m::Float64, g::Int) = begin
    out::Float64 = g == 0 ? endpoint_normal(m, 1.5).logpdf(y) : 0.0 * y
    return out
end

# An endpoint reading a module-private global cannot be relocated into an
# arm (the name would rebind into the caller's module): rejected loudly.
nonbase_endpoint_helper(x) = 2x

@kernel helper_standard_normal() = begin
    logpdf(z::Float64)::Float64 = nonbase_endpoint_helper(z) - 0.5 * log(2π)
end

@kernel helper_normal =
    endpoint_location_scale(helper_standard_normal)

end
