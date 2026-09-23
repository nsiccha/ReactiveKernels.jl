using ReactiveKernels, Reactant, Test
import Enzyme

# Authored `if`/`?:`/`&&`/`||` keep their lazy Julia semantics under Reactant
# (docs/src/constraints.md): the inactive side is never evaluated, so an
# invalid inactive computation neither poisons the value nor the gradient.

@kernel lazy_scalar_guard(x::Float64) = begin
    guarded::Float64 = x > 0 ? log(x) : -1.0
    return guarded
end

@kernel lazy_plate_guard(x::Vector{Float64}) = begin
    pointwise = plate(x) do xi
        cell::Float64 = xi > 0 ? log(xi) : 0.0
        cell
    end
    total::Float64 = sum(pointwise)
    return total
end

# A branch whose condition reads bound data only (`y`) is split during
# preparation; the constant arm reads no lane argument.
@kernel data_bound_plate_guard(x::Vector{Float64}, y::Vector{Float64}) = begin
    pointwise = plate(x, y) do xi, yi
        cell::Float64 = yi > 0 ? log(xi * yi) : 1.0
        cell
    end
    total::Float64 = sum(pointwise)
    return total
end

@kernel lazy_short_circuit(x::Float64, flag::Bool) = begin
    both::Bool = flag && x > 0
    either::Bool = flag || x > 0
    picked::Float64 = both ? sqrt(x) : 0.0
    return picked
end

_traced(v) = v isa AbstractArray ? Reactant.to_rarray(v) :
             Reactant.to_rarray(v; track_numbers = true)
_host(v) = v isa Reactant.AbstractConcreteArray ? Array(v) : Reactant.to_number(v)

@testset "a scalar guard lowers to a lazy conditional region" begin
    k = prepare(lazy_scalar_guard; want = :guarded)
    hlo = repr(Reactant.@code_hlo optimize = false k(_traced(2.0)))
    @test occursin("stablehlo.if", hlo)
    @test !occursin("stablehlo.select", hlo)
    compiled = Reactant.@compile k(_traced(2.0))
    for x in (2.0, -1.0)
        @test _host(compiled(_traced(x))) == k(x)
    end
    gradient(x) = Enzyme.gradient(Enzyme.Reverse, k, x)
    compiled_gradient = Reactant.@compile gradient(_traced(2.0))
    @test _host(only(compiled_gradient(_traced(2.0)))) ≈ 0.5
    # The inactive `log` is never differentiated: exactly zero, never NaN.
    @test _host(only(compiled_gradient(_traced(-1.0)))) == 0.0
end

@testset "a plate cell guard stays lazy inside the batched region" begin
    k = prepare(lazy_plate_guard; want = :total)
    x = [2.0, -1.0, 0.5, -3.0]
    hlo = repr(Reactant.@code_hlo optimize = false k(_traced(x)))
    @test occursin("stablehlo.if", hlo)
    compiled = Reactant.@compile k(_traced(x))
    @test _host(compiled(_traced(x))) ≈ k(x)
    gradient(v) = Enzyme.gradient(Enzyme.Reverse, k, v)
    compiled_gradient = Reactant.@compile gradient(_traced(x))
    @test _host(only(compiled_gradient(_traced(x)))) ≈ [0.5, 0.0, 2.0, 0.0]
    # The emitted program does not grow with the plate length.
    sizes = [count("\n", repr(Reactant.@code_hlo optimize = false k(
        _traced(collect(range(-1.0, 1.0; length = n)))))) for n in (8, 32)]
    @test sizes[1] == sizes[2]
end

@testset "short-circuit forms are lazy branches" begin
    k = prepare(lazy_short_circuit; want = :picked)
    # A host `flag` would be a constant of the traced program; trace it so
    # one compiled program serves every case.
    compiled = Reactant.@compile k(_traced(4.0), _traced(true))
    for (x, flag) in ((4.0, true), (4.0, false), (-4.0, true), (-4.0, false))
        @test _host(compiled(_traced(x), _traced(flag))) == k(x, flag)
    end
    hlo = repr(Reactant.@code_hlo optimize = false k(_traced(4.0), _traced(true)))
    # `flag && x > 0` and the ternary are two nested lazy regions.
    @test count("stablehlo.if", hlo) >= 2
    @test !occursin("stablehlo.select", hlo)
end

@testset "reverse through a batched lazy branch: the upstream lane boundary" begin
    # Reactant's batching pass unrolls a small plate per lane and realizes a
    # larger one as a loop; Enzyme reverse through a lazy `if` inside that loop
    # does not lower yet (`benchmark/repro_reactant_batch_if_reverse.jl`).
    # The primal is lane-count independent either way. This locks the boundary
    # so an upstream fix (or a shift of the threshold) is noticed here.
    k = prepare(lazy_plate_guard; want = :total)
    gradient(v) = Enzyme.gradient(Enzyme.Reverse, k, v)
    small = [2.0, -1.0, 0.5, -3.0]
    large = [2.0, -1.0, 0.5, -3.0, 1.5, -0.5]
    compiled_small = Reactant.@compile gradient(_traced(small))
    @test _host(only(compiled_small(_traced(small)))) ≈ [0.5, 0.0, 2.0, 0.0]
    @test _host((Reactant.@compile k(_traced(large)))(_traced(large))) ≈ k(large)
    @test_throws Reactant.Compiler.CompilationError Reactant.@compile gradient(_traced(large))
end

@testset "a data-bound plate branch is split: reverse at every lane count" begin
    # The condition reads bound data, so preparation splits the lanes by arm
    # and no conditional region reaches the traced program; the batched
    # lazy-branch reverse boundary above does not apply.
    for n in (8, 32)
        x = collect(range(0.5, 3.0; length = n))
        y = [isodd(i) ? 0.5i : -1.0 for i in 1:n]
        k = prepare(data_bound_plate_guard; have = (:x, :y), want = :total,
                    bound = (; y))
        hlo = repr(Reactant.@code_hlo optimize = false k(_traced(x)))
        @test !occursin("stablehlo.if", hlo)
        compiled = Reactant.@compile k(_traced(x))
        @test _host(compiled(_traced(x))) ≈ k(x)
        @test k(x) ≈ sum(yi > 0 ? log(xi * yi) : 1.0 for (xi, yi) in zip(x, y))
        gradient(v) = Enzyme.gradient(Enzyme.Reverse, k, v)
        compiled_gradient = Reactant.@compile gradient(_traced(x))
        @test _host(only(compiled_gradient(_traced(x)))) ≈
              [yi > 0 ? 1 / xi : 0.0 for (xi, yi) in zip(x, y)]
    end
end
