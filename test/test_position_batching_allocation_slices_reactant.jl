using Test, ReactiveKernels, Reactant

@kernel compiled_slice_passthrough_source(position, shared) = begin
    result = position.scale * sum(shared)
end

@kernel compiled_slice_plate_inner(x, q) = begin
    pointwise = plate(x, Ref(q)) do xi, qq
        2 * xi + qq[1]
    end
    return pointwise
end
const COMPILED_SLICE_PLATE = prepare(compiled_slice_plate_inner)

@kernel compiled_slice_plate_outer(position, units, dose) = begin
    gathered = units .* position[1]
    out = COMPILED_SLICE_PLATE(gathered, dose)
    return out
end

# Slot columns are sliced and updated with the slot axis leading: a ranked
# dynamic slice takes one slot as its first extent and a ranked update carries
# one. Enzyme-JAX's reshape-slice rewrites do not finish on a trailing slot
# axis (benchmark/repro_reactant_reshape_slice_rewrite.jl), so a regression
# would otherwise surface as a compile that never returns.
function slot_extents(hlo)
    slices = [parse.(Int, split(found.captures[1], ", ")) for found in eachmatch(
        r"stablehlo\.dynamic_slice [^\n]*sizes = \[([0-9, ]+)\]", hlo)]
    updates = [parse.(Int, split(found.captures[1], "x")) for found in eachmatch(
        r"stablehlo\.dynamic_update_slice [^\n]*: \(tensor<[^>]*>, tensor<([0-9x]+)x[a-z]", hlo)]
    filter(extents -> length(extents) > 1, vcat(slices, updates))
end

operation_names(hlo) =
    [found.match for found in eachmatch(r"(?:stablehlo|enzyme)\.[a-z_]+", hlo)]

@testset "recipe-free structured passthrough compiler parity" begin
    batch = vectorize(compiled_slice_passthrough_source;
        have=(:position, :shared), want=(:position, :shared), batched=:position)
    @test isempty(plan(batch).recipes)
    position = (; scale=[1.0, 2.0, 3.0], curve=reshape(collect(1.0:6.0), 2, 3),
                  cube=reshape(collect(1.0:18.0), 2, 3, 3))
    shared = [2.0, 5.0]
    traced = map(Reactant.to_rarray, position)
    traced_shared = Reactant.to_rarray(shared)
    extents = slot_extents(repr(
        Reactant.@code_hlo optimize=false batch(traced, traced_shared)))
    @test !isempty(extents) && all(isone ∘ first, extents)
    compiled = Reactant.@compile batch(traced, traced_shared)
    actual, repeated = compiled(traced, traced_shared)
    native, native_repeated = batch(position, shared)
    @test Array(actual.scale) == native.scale
    @test Array(actual.curve) == native.curve
    @test Array(actual.cube) == native.cube
    @test Array(repeated) == native_repeated
    @test Array(traced.curve) == position.curve
    @test Array(traced_shared) == shared
    later = map(Reactant.to_rarray, map(value -> value .+ 1, position))
    compiled(later, traced_shared)
    @test Array(actual.curve) == native.curve

    function operations(lanes)
        input = (; scale=Reactant.to_rarray(fill(2.0, lanes)),
                   curve=Reactant.to_rarray(fill(3.0, 2, lanes)),
                   cube=Reactant.to_rarray(fill(4.0, 2, 3, lanes)))
        hlo = repr(Reactant.@code_hlo optimize=true batch(input, traced_shared))
        @test count("stablehlo.while", hlo) == 1
        operation_names(hlo)
    end
    small = operations(3)
    @test !isempty(small)
    @test small == operations(23)
end

@testset "embedded plate under position batching compiler parity" begin
    batch = prepare_batched(compiled_slice_plate_outer;
        have=(:position, :units, :dose), batched=(:position, :units), want=:out)
    dose = [5.0, 6.0]
    traced_dose = Reactant.to_rarray(dose)
    inputs(lanes) = (reshape(collect(1.0:(2lanes)), 2, lanes),
                     reshape(collect(10.0:10.0:(20.0lanes)), 2, lanes))
    for lanes in (1, 2, 5)
        position, units = inputs(lanes)
        traced = (Reactant.to_rarray(position), Reactant.to_rarray(units))
        extents = slot_extents(repr(
            Reactant.@code_hlo optimize=false batch(traced..., traced_dose)))
        @test !isempty(extents) && all(isone ∘ first, extents)
        compiled = Reactant.@compile batch(traced..., traced_dose)
        @test Array(compiled(traced..., traced_dose)) == batch(position, units, dose)
    end

    function operations(lanes)
        traced = map(Reactant.to_rarray, inputs(lanes))
        hlo = repr(Reactant.@code_hlo optimize=true batch(traced..., traced_dose))
        @test count("stablehlo.while", hlo) == 1
        operation_names(hlo)
    end
    @test operations(3) == operations(23)
end
