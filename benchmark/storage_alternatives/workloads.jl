module StorageWorkloads
using ReactiveKernels
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    _student_log_kernel, _student_log_kernel_at_log_abs

@kernel stored(u::Vector{Float64}, nu::Float64) = begin
    w = exp.(u)
    squared = w .^ 2
    prior = plate(w, Ref(nu)) do wi, ni
        _student_log_kernel(ni, wi)
    end
    total = sum(squared) + sum(prior)
    return total
end

@kernel explicit(u::Vector{Float64}, nu::Float64) = begin
    squared = exp.(2 .* u)
    prior = plate(u, Ref(nu)) do ui, ni
        _student_log_kernel_at_log_abs(ui, ni)
    end
    total = sum(squared) + sum(prior)
    return total
end

@kernel retained(u::Vector{Float64}, nu::Float64) = begin
    w = exp.(u)
    squared = w .^ 2
    prior = plate(w, Ref(nu)) do wi, ni
        _student_log_kernel(ni, wi)
    end
    total = sum(squared) + sum(prior) + sum(w)
    return total
end

@kernel logarithmic(u::Vector{Float64}, nu::Float64) = begin
    w = exp.(u)
    logarithms = log.(w)
    prior = plate(w, Ref(nu)) do wi, ni
        _student_log_kernel(ni, wi)
    end
    total = sum(logarithms) + sum(prior)
    return total
end

@kernel returned(u::Vector{Float64}, nu::Float64) = begin
    w = exp.(u)
    prior = plate(w, Ref(nu)) do wi, ni
        _student_log_kernel(ni, wi)
    end
    total = sum(prior)
    return w, total
end

# These fixtures deliberately exercise observable snapshot semantics. A pure
# recipe flag does not prove that a foreign array's element access is pure.
struct ReadOnlyPeer <: AbstractVector{Float64}
    data::Vector{Float64}
end
struct MutatingPeer <: AbstractVector{Float64}
    data::Vector{Float64}
    victim::Vector{Float64}
end
struct GlobalMutatingPeer <: AbstractVector{Float64}
    data::Vector{Float64}
end
const peer_victim = Ref(Float64[])
Base.size(x::Union{ReadOnlyPeer,MutatingPeer,GlobalMutatingPeer}) = size(x.data)
Base.getindex(x::ReadOnlyPeer, i::Int) = x.data[i]
function Base.getindex(x::MutatingPeer, i::Int)
    fill!(x.victim, 3.0)
    x.data[i]
end
function Base.getindex(x::GlobalMutatingPeer, i::Int)
    fill!(peer_victim[], 3.0)
    x.data[i]
end
@kernel peer(u::Vector{Float64}, y, nu::Float64) = begin
    w = exp.(u)
    squared = w .^ 2
    prior = plate(y, w, Ref(nu)) do yi, wi, ni
        yi + _student_log_kernel(ni, wi)
    end
    total = sum(squared) + sum(prior)
    return total
end
@kernel owned_peer(u::Vector{Float64}, data::Vector{Float64}, nu::Float64) = begin
    point = copy(u)
    w = exp.(point)
    squared = w .^ 2
    other = MutatingPeer(data, point)
    prior = plate(other, w, Ref(nu)) do yi, wi, ni
        yi + _student_log_kernel(ni, wi)
    end
    total = sum(squared) + sum(prior)
    return total
end

function reference(kind, x, nu)
    squared = exp.(2 .* x)
    prior = @. -(nu+1)/2*log1p(squared/nu)
    if kind === :logarithmic
        return sum(x .+ prior), @. 1 - (nu+1)*squared/(nu+squared)
    end
    y = sum(squared .+ prior)
    d = @. 2squared - (nu+1)*squared/(nu+squared)
    if kind === :retained
        y += sum(exp.(x))
        d .+= exp.(x)
    end
    y, d
end
end
