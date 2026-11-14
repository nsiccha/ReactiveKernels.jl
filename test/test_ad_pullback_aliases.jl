module ADPullbackAliasTests
# Pullback inputs that share memory with what the call reads or writes. A
# NonAllocatingKernel pullback returns borrowed values from storage its
# preparation owns; feeding such a value back (as the seed, the active point,
# an argument or the cotangent destination) must give the mathematical result
# for the inputs' contents at call time. References are closed forms.
using ReactiveKernels, MutatingFunctions, DifferentiationInterface, Test
import Enzyme
const RK = ReactiveKernels
const REV = AutoEnzyme(; mode = Enzyme.Reverse)

@kernel pullback_alias_exp(x::Vector{Float64}) = begin
    y::Vector{Float64} = exp.(x)
    return y
end
# Writing the product into storage that holds `x` would overwrite an input the
# product still reads.
@kernel pullback_alias_matvec(A::Matrix{Float64}, x::Vector{Float64}) = begin
    y::Vector{Float64} = A * x
    return y
end
@kernel pullback_alias_pair(x::Vector{Float64}, z::Vector{Float64}) = begin
    y::Vector{Float64} = exp.(x) .* z
    return y
end

na(spec, args...; want) = prepare_nonallocating(prepare(spec; want), args...)
function disjoint_bytes(p, dx, s, args::Vararg{Any,N}) where {N}
    ad_value_and_pullback!(p, dx, s, args...)
    @allocated ad_value_and_pullback!(p, dx, s, args...)
end

const n = 16
const x0 = sin.(1.0:n) ./ 2
const x1 = cos.(1.0:n) ./ 3
const s0 = collect(range(0.5, 1.5; length = n))

@testset "borrowed NonAllocatingKernel pullback values fed back" begin
    p = prepare_ad_pullback(na(pullback_alias_exp, x0; want = :y), REV, s0, x0;
                            active = :x)
    y, _ = ad_value_and_pullback(p, s0, x0)
    y2, _ = ad_value_and_pullback(p, s0, x1)
    @test y2 === y             # one borrowed buffer, overwritten by each call
    @test y2 ≈ exp.(x1)
    for (xa, xb) in ((x0, x1), (x1, x0), (x0, x0), (x1, x1))
        y, _ = ad_value_and_pullback(p, s0, xa); yc = copy(y)
        @test ad_pullback(p, y, xb) ≈ yc .* exp.(xb)
        y, _ = ad_value_and_pullback(p, s0, xa); yc = copy(y)
        value, cotangent = ad_value_and_pullback(p, y, xb)
        @test value ≈ exp.(xb) && cotangent ≈ yc .* exp.(xb)
        y, _ = ad_value_and_pullback(p, s0, xa)
        value, cotangent = ad_value_and_pullback!(p, y, s0, xb)
        @test cotangent === y
        @test cotangent ≈ s0 .* exp.(xb) && value ≈ exp.(xb)
        y, _ = ad_value_and_pullback(p, s0, xa); yc = copy(y)
        value, cotangent = ad_value_and_pullback(p, s0, y)
        @test value ≈ exp.(yc) && cotangent ≈ s0 .* exp.(yc)
        y, _ = ad_value_and_pullback(p, s0, xa); yc = copy(y)
        dx = zero(x0)
        value, cotangent = ad_value_and_pullback!(p, dx, y, y)
        @test cotangent === dx && dx ≈ yc .* exp.(yc) && value ≈ exp.(yc)
    end

    # Caller-owned overlaps: the seed or the point as the destination.
    s = copy(s0)
    value, cotangent = ad_value_and_pullback!(p, s, s, x1)
    @test cotangent === s && s ≈ s0 .* exp.(x1) && value ≈ exp.(x1)
    x = copy(x1)
    value, cotangent = ad_value_and_pullback!(p, x, s0, x)
    @test cotangent === x && x ≈ s0 .* exp.(x1) && value ≈ exp.(x1)
    x = copy(x0)
    @test ad_pullback(p, x, x) ≈ x0 .* exp.(x0)

    # A value from before a change of length, fed back after it.
    y, _ = ad_value_and_pullback(p, s0, x0); yc = copy(y)
    longer = vcat(x1, [0.25, -0.5]); longer_seed = vcat(s0, [1.0, 2.0])
    value, cotangent = ad_value_and_pullback(p, longer_seed, longer)
    @test value ≈ exp.(longer) && cotangent ≈ longer_seed .* exp.(longer)
    @test ad_pullback(p, y, x1) ≈ yc .* exp.(x1)

    # Disjoint inputs take no copy.
    dx = zero(x0)
    @test disjoint_bytes(p, dx, s0, x0) == 0
    @test dx ≈ s0 .* exp.(x0)
end

@testset "a borrowed value as a seed view" begin
    seed_view = view(copy(s0), 1:n)
    p = prepare_ad_pullback(na(pullback_alias_exp, x0; want = :y), REV,
                            seed_view, x0; active = :x)
    y, _ = ad_value_and_pullback(p, view(s0, 1:n), x0); yc = copy(y)
    @test ad_pullback(p, view(y, 1:n), x1) ≈ yc .* exp.(x1)
end

@testset "an input the product still reads, fed back" begin
    A = [1.0 / (i + j) + (i == j) for i in 1:n, j in 1:n]
    p = prepare_ad_pullback(na(pullback_alias_matvec, A, x0; want = :y), REV,
                            s0, A, x0; active = :x)
    for x in (x0, x1)
        y, _ = ad_value_and_pullback(p, s0, A, x); yc = copy(y)
        value, cotangent = ad_value_and_pullback(p, s0, A, y)
        @test value ≈ A * yc && cotangent ≈ transpose(A) * s0
        y, _ = ad_value_and_pullback(p, s0, A, x); yc = copy(y)
        value, cotangent = ad_value_and_pullback!(p, y, y, A, x)
        @test cotangent === y && y ≈ transpose(A) * yc && value ≈ A * x
    end
    dx = zero(x0)
    @test disjoint_bytes(p, dx, s0, A, x0) == 0
    @test dx ≈ transpose(A) * s0
end

@testset "tuple selector with a borrowed destination component" begin
    z0 = 1 .+ x1 .^ 2
    p = prepare_ad_pullback(na(pullback_alias_pair, x0, z0; want = :y), REV,
                            s0, x0, z0; active = (:x, :z))
    y, _ = ad_value_and_pullback(p, s0, x0, z0)
    dz = zero(z0)
    value, (cx, cz) = ad_value_and_pullback!(p, (y, dz), s0, x1, z0)
    @test cx === y && cz === dz
    @test cx ≈ s0 .* exp.(x1) .* z0 && cz ≈ s0 .* exp.(x1)
    @test value ≈ exp.(x1) .* z0
    y, _ = ad_value_and_pullback(p, s0, x0, z0); yc = copy(y)
    value, (cx, cz) = ad_value_and_pullback(p, y, x1, z0)
    @test cx ≈ yc .* exp.(x1) .* z0 && cz ≈ yc .* exp.(x1)
end

@testset "dataflow pullbacks own no storage and take no copy" begin
    p = prepare_ad_pullback(pullback_alias_exp, REV, s0, x0; active = :x,
                            want = :y)
    @test RK._ad_owned_storage(p) === ()
    y, _ = ad_value_and_pullback(p, s0, x0); yc = copy(y)
    value, cotangent = ad_value_and_pullback(p, y, x1)
    @test value ≈ exp.(x1) && cotangent ≈ yc .* exp.(x1)
end

end
