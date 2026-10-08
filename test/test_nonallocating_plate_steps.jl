using ReactiveKernels
using MutatingFunctions
using Test
using ReactiveKernels: plate

# A plate step of `prepare_nonallocating` runs the native cell loop of ordinary
# `prepare` and fills the step's own cache. However many `Ref` operands the
# plate has, a warmed-up call allocates nothing and returns `prepare`'s values.
# Before, the step broadcast the prepared cell kernel, which on Julia 1.10.12
# allocated per call: 240 B with one `Ref` operand and 352 B with three (the
# dose-superposition shape `plate(observations, Ref(plan), Ref(units),
# Ref(weights))`), against 128 B for `prepare`'s own output.

function plate_step_allocations(k, args)
    k(args...)
    k(args...)
    @allocated k(args...)
end

bitwise_equal(a, b) = typeof(a) == typeof(b) && axes(a) == axes(b) &&
                      all(map(===, a, b))

struct PlateShiftPlan
    shifts::Vector{Int}
end

const ONE_REF = @kernel one_ref(rows, u::Vector{Float64}) = begin
    y::Vector{Float64} = plate(rows, Ref(u)) do o, uu
        sum((0.5 * get(uu, o - j, 0.0) for j in 0:2); init = 0.0)
    end
    return y
end

const THREE_REFS = @kernel three_refs(rows, s::Vector{Int}, u::Vector{Float64},
                                      w::Vector{Float64}) = begin
    y::Vector{Float64} = plate(rows, Ref(s), Ref(u), Ref(w)) do o, ss, uu, ww
        sum((ww[j] * get(uu, o - ss[j], 0.0) for j in eachindex(ww)); init = 0.0)
    end
    return y
end

const STRUCT_REF = @kernel struct_ref(rows, plan, u::Vector{Float64},
                                      w::Vector{Float64}) = begin
    y::Vector{Float64} = plate(rows, Ref(plan), Ref(u), Ref(w)) do o, p, uu, ww
        sum((ww[j] * get(uu, o - p.shifts[j], 0.0) for j in eachindex(ww));
            init = 0.0)
    end
    return y
end

# Integer cells stored into the declared `Vector{Float64}`, as the ordinary
# kernel's typed local converts the plate's result.
const CONVERTED = @kernel converted(rows, u::Vector{Float64}) = begin
    y::Vector{Float64} = plate(rows, Ref(u)) do o, uu
        2o + length(uu)
    end
    return y
end

const MATRIX_CELLS = @kernel matrix_cells(a::Matrix{Float64},
                                          b::Vector{Float64}) = begin
    y::Matrix{Float64} = plate(a, b) do x, z
        x * z + 1.0
    end
    return y
end

@testset "nonallocating plate steps" begin
    rows = 1:9
    u = collect(1.0:9.0)
    w = [1.0, 0.5, 0.25]
    cases = (("one Ref operand", ONE_REF, (rows, u)),
             ("three Ref operands", THREE_REFS, (rows, [0, 2, 4], u, w)),
             ("a struct Ref operand", STRUCT_REF,
              (rows, PlateShiftPlan([0, 2, 4]), u, w)),
             ("cells converted to the declared element type", CONVERTED,
              (rows, u)),
             ("a two-dimensional broadcast", MATRIX_CELLS,
              ([1.0 2.0; 3.0 4.0; 5.0 6.0], [10.0, 20.0, 30.0])))
    for (label, spec, args) in cases
        @testset "$label" begin
            k = prepare_nonallocating(spec)
            reference = prepare(spec)
            @test bitwise_equal(k(args...), reference(args...))
            @test plate_step_allocations(k, args) == 0
            @test bitwise_equal(k(args...), reference(args...))
            println("NONALLOCATING_ALLOC_BYTES\tplate_step\t", label, "\t",
                    plate_step_allocations(k, args), "\tprepare\t",
                    plate_step_allocations(reference, args))
        end
    end

    @testset "a new domain size reseeds the plate cache" begin
        k = prepare_nonallocating(THREE_REFS)
        reference = prepare(THREE_REFS)
        for n in (9, 4, 12, 12)
            args = (1:n, [0, 2, 4], collect(1.0:n), w)
            @test bitwise_equal(k(args...), reference(args...))
        end
        @test plate_step_allocations(k, (1:12, [0, 2, 4], collect(1.0:12), w)) == 0
    end

    @testset "inputs are read-only" begin
        k = prepare_nonallocating(THREE_REFS)
        s, uu, ww = [0, 2, 4], copy(u), copy(w)
        k(rows, s, uu, ww)
        k(rows, s, uu, ww)
        @test s == [0, 2, 4]
        @test uu == u
        @test ww == w
    end
end
