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

end
