module TensorizedVectReactantTests

using ReactiveKernels, Reactant, Test
import Enzyme

# Scalar-vector literals lower to real traced vectors: an untyped
# `[a, b, c]` with a traced element and a typed `Float64[a, b, c]` both
# build a traced `Vector` with unchanged shape, type, and values, so
# gathering either at a host or traced scalar or vector index compiles
# and matches native execution.

@kernel _gather_fused(s1::Float64, s2::Float64, s3::Float64, i) = begin
    out = [s1, s2, s3][i]
    return out
end

@kernel _gather_steps(s1::Float64, s2::Float64, s3::Float64, i) = begin
    v = [s1, s2, s3]
    out = v[i]
    return out
end

@kernel _gather_typed(s1::Float64, s2::Float64, s3::Float64, i) = begin
    out = Float64[s1, s2, s3][i]
    return out
end

@kernel _construct_typed(s1::Float64, s2::Float64, s3::Float64) = begin
    out = Float64[s1, s2, s3]
    return out
end

@kernel _gather_mixed(s1::Float64, s3::Float64, i) = begin
    out = [s1, 5.0, s3][i]
    return out
end

@kernel _gather_f32(s1::Float32, s2::Float32, i) = begin
    out = [s1, s2][i]
    return out
end

@kernel _intpair(a::Int, b::Int) = begin
    out = [a, b]
    return out
end

@kernel _anypair(a::Float64, b::Float64) = begin
    out = Any[a, b]
    return out
end

@kernel _gather_plate(p::Vector{Float64}, x, ks) = begin
    pointwise = plate(x, ks, Ref(p)) do xi, k, pp
        xi * [pp[1], pp[2], pp[3]][k]
    end
    total::Float64 = sum(pointwise)
    return total
end

@kernel _gathersum(q::Vector{Float64}, i) = begin
    out::Float64 = sum([q[1], q[2], q[3]][i])
    return out
end

_traced(v) = v isa AbstractArray ? Reactant.to_rarray(v) :
             Reactant.to_rarray(v; track_numbers = true)
_host(v) = v isa Reactant.AbstractConcreteArray ? Array(v) :
           Reactant.to_number(v)

const _S1, _S2, _S3 = 0.5, 5.0, 3.0
const _IDX = [1, 3, 2, 1, 2, 3]

@testset "literal gathers lower at every index kind" begin
    rs1, rs2, rs3 = _traced(_S1), _traced(_S2), _traced(_S3)
    ri = _traced(_IDX)
    rscalar = _traced(2)
    for kernel in (_gather_fused, _gather_steps, _gather_typed)
        k = prepare(kernel; want = :out)
        for (label, idx, ridx) in (("traced vector", _IDX, ri),
                ("host vector", _IDX, _IDX),
                ("traced scalar", 2, rscalar),
                ("host scalar", 2, 2))
            native = k(_S1, _S2, _S3, idx)
            compiled = Reactant.@compile k(rs1, rs2, rs3, ridx)
            got = _host(compiled(rs1, rs2, rs3, ridx))
            @test got == native
        end
    end
end

@testset "typed literals construct typed traced vectors" begin
    rs1, rs2, rs3 = _traced(_S1), _traced(_S2), _traced(_S3)
    k = prepare(_construct_typed; want = :out)
    compiled = Reactant.@compile k(rs1, rs2, rs3)
    got = compiled(rs1, rs2, rs3)
    @test got isa Reactant.AbstractConcreteArray
    @test Array(got) == [_S1, _S2, _S3]
    @test eltype(Array(got)) === Float64
    @test k(_S1, _S2, _S3) == [_S1, _S2, _S3]
end

@testset "element types and mixes are preserved" begin
    rs1, rs3 = _traced(_S1), _traced(_S3)
    km = prepare(_gather_mixed; want = :out)
    compiled_mixed = Reactant.@compile km(rs1, rs3, _traced(_IDX))
    @test _host(compiled_mixed(rs1, rs3, _traced(_IDX))) ==
        km(_S1, _S3, _IDX)
    rf1 = Reactant.to_rarray(Float32(0.5); track_numbers = true)
    rf2 = Reactant.to_rarray(Float32(1.25); track_numbers = true)
    kf = prepare(_gather_f32; want = :out)
    compiled_f32 = Reactant.@compile kf(rf1, rf2, _traced([2, 1, 2]))
    got_f32 = Array(compiled_f32(rf1, rf2, _traced([2, 1, 2])))
    @test got_f32 == kf(Float32(0.5), Float32(1.25), [2, 1, 2])
    @test eltype(got_f32) === Float32
    ki = prepare(_intpair; want = :out)
    ra = Reactant.to_rarray(2; track_numbers = true)
    rb = Reactant.to_rarray(3; track_numbers = true)
    compiled_ints = Reactant.@compile ki(ra, rb)
    got_ints = Array(compiled_ints(ra, rb))
    @test got_ints == [2, 3]
    @test eltype(got_ints) === Int
end

@testset "abstract element types keep the host container" begin
    rs1, rs2 = _traced(_S1), _traced(_S2)
    k = prepare(_anypair; want = :out)
    compiled = Reactant.@compile k(rs1, rs2)
    got = compiled(rs1, rs2)
    @test got isa Vector
    @test Float64.(got) == [_S1, _S2]
end

@testset "literal construction emits one vector" begin
    rs1, rs2, rs3 = _traced(_S1), _traced(_S2), _traced(_S3)
    k = prepare(_construct_typed; want = :out)
    hlo = repr(Reactant.@code_hlo optimize = false k(rs1, rs2, rs3))
    @test occursin("tensor<3xf64>", hlo)
    @test count("stablehlo.while", hlo) == 0
end

@testset "compiled literals reuse live values and keep inputs" begin
    rs1, rs2, rs3 = _traced(_S1), _traced(_S2), _traced(_S3)
    ri = _traced(_IDX)
    k = prepare(_gather_fused; want = :out)
    compiled = Reactant.@compile k(rs1, rs2, rs3, ri)
    @test _host(compiled(rs1, rs2, rs3, ri)) == [_S1, _S2, _S3][_IDX]
    rs1b = _traced(0.55)
    @test _host(compiled(rs1b, rs2, rs3, ri)) == [0.55, _S2, _S3][_IDX]
    @test Array(ri) == _IDX
    @test Float64(rs1) == _S1
    @test Float64(rs1b) == 0.55
end

@testset "literal gathers keep the plate's retained loops" begin
    while_counts = Int[]
    p = [_S1, _S2, _S3]
    for n in (8, 32)
        x = collect(range(-1.0, 1.0; length = n))
        ks = [mod(i - 1, 3) + 1 for i in 1:n]
        expect = sum(x[i] * p[ks[i]] for i in 1:n)
        k = prepare(_gather_plate; bound = (; x, ks))
        @test k(p) ≈ expect
        rp = _traced(p)
        hlo = repr(Reactant.@code_hlo optimize = false k(rp))
        push!(while_counts, count("stablehlo.while", hlo))
        compiled = Reactant.@compile k(rp)
        @test Float64(compiled(rp)) ≈ expect
    end
    @test while_counts[1] == while_counts[2]
end

@testset "literal gathers differentiate like the native gather" begin
    q = [_S1, _S2, _S3]
    k = prepare(_gathersum; want = :out)
    native_gradient(v) = Enzyme.gradient(Enzyme.Reverse, w -> k(w, _IDX), v)
    compiled_gradient = Reactant.@compile native_gradient(_traced(q))
    @test _host(only(compiled_gradient(_traced(q)))) ≈
        only(native_gradient(q))
end


# An element read with one integer index per dimension, at least one traced,
# is one gather whatever the array: a host container of traced scalars is
# stacked into a traced array first (Reactant's own read of it at a traced
# index recursed without termination), an index of any integer type reads the
# element (a traced `Int32` index took Reactant's general path, which returns a
# 1×1 array), and a read may mix concrete and traced indices.
@testset "element reads at traced indices: stacked hosts, N-D, Int32" begin
    W = [0.5 -0.2 0.1; 0.3 0.7 -0.4; -0.1 0.2 0.9]
    ri, rj = _traced(2), _traced(3)
    stacked(s, i, j) = ReactiveKernels._tensorized_getindex(
        [s * W[a, b] for a in 1:3, b in 1:3], i, j)
    @test _host((Reactant.@compile stacked(_traced(1.5), ri, rj))(
        _traced(1.5), ri, rj)) ≈ 1.5 * W[2, 3]
    stacked_vector(s, i) = ReactiveKernels._tensorized_getindex([s, 2s, 3s], i)
    @test _host((Reactant.@compile stacked_vector(_traced(1.5), rj))(
        _traced(1.5), rj)) ≈ 4.5
    read_at(A, i...) = ReactiveKernels._tensorized_getindex(A, i...)
    for (A, indices, expected) in (
            (W, (Reactant.to_rarray(Int32(2); track_numbers = true), rj), W[2, 3]),
            (W[:, 1], (Reactant.to_rarray(Int32(3); track_numbers = true),), W[3, 1]),
            (W, (ri, 1), W[2, 1]))
        for array in (A, _traced(A))
            got = (Reactant.@compile read_at(array, indices...))(array, indices...)
            @test got isa Reactant.AbstractConcreteNumber
            @test _host(got) == expected
        end
    end
    A3 = reshape(collect(1.0:24.0), 2, 3, 4)
    read3(i, k) = ReactiveKernels._tensorized_getindex(A3, i, 3, k)
    @test _host((Reactant.@compile read3(ri, _traced(4)))(ri, _traced(4))) == A3[2, 3, 4]
end

end
