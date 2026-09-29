using Test
using ReactiveKernels
using Reactant
using DifferentiationInterface: AutoEnzyme
using Enzyme

@kernel compiled_record_positions(position, data) = begin
    result = (; trajectory=position.scale .* data .+ position.offset,
                total=position.scale * sum(data))
end

@testset "untyped and record batching under Reactant" begin
    batch = vectorize(compiled_record_positions; batched=:position)
    position = (; scale=[1.0, -2.0, 3.0], offset=[0.5, 1.0, -0.5])
    data = [0.0, 1.0, 2.0, 4.0]
    traced_position = map(Reactant.to_rarray, position)
    traced_data = Reactant.to_rarray(data)
    compiled = Reactant.@compile batch(traced_position, traced_data)
    result = compiled(traced_position, traced_data)
    expected = batch(position, data)
    @test Array(result.trajectory) ≈ expected.trajectory
    @test Array(result.total) ≈ expected.total
    later_position = (; scale=Reactant.to_rarray(position.scale .+ 1),
                        offset=Reactant.to_rarray(position.offset))
    compiled(later_position, traced_data)
    @test Array(result.trajectory) ≈ expected.trajectory
    @test Array(result.total) ≈ expected.total
    @test Array(traced_data) == data

    @kernel untyped_compiled_numeric(position, data) = begin
        result = sum(position .* position) + sum(data)
    end
    numeric = vectorize(untyped_compiled_numeric; batched=:position)
    positions = reshape(collect(1.0:15.0), 3, 5)
    traced = Reactant.to_rarray(positions)
    numeric_compiled = Reactant.@compile numeric(traced, traced_data)
    @test Array(numeric_compiled(traced, traced_data)) ≈ numeric(positions, data)

    # Position count is data, including a structured record's leaves. Trace a
    # small and a larger ensemble and compare the emitted operation sequence,
    # independently of SSA names and tensor shape attributes.
    function operation_names(count)
        positions = (; scale=Reactant.to_rarray(fill(2.0, count)),
                       offset=Reactant.to_rarray(fill(0.5, count)))
        hlo = repr(Reactant.@code_hlo optimize=true batch(positions, traced_data))
        [m.match for m in eachmatch(r"(?:stablehlo|enzyme)\.[a-z_]+", hlo)]
    end
    small = operation_names(3)
    large = operation_names(23)
    @test !isempty(small)
    @test small == large
end

@testset "replicated AD retains scalar semantics" begin
    @kernel compiled_batch_objective(position::Vector{Float64}, offset::Float64) = begin
        result::Float64 = sum(abs2, position) + offset
    end
    ad = prepare_ad(compiled_batch_objective, AutoEnzyme(mode=Enzyme.Reverse),
                    [1.0, 2.0], 0.5; active=:position, want=:result)
    batch = replica(ad; batched=:position)
    position = [1.0 -2.0 3.0; 2.0 0.5 -1.0]
    native = batch(position, 0.5)
    traced = Reactant.to_rarray(position)
    offset = Reactant.to_rarray(0.5; track_numbers=true)
    compiled = Reactant.@compile batch(traced, offset)
    values, gradients = compiled(traced, offset)
    @test Array(values) ≈ native[1]
    @test Array(gradients) ≈ native[2]
    @test native[2] ≈ 2 .* position
end

@testset "lazy position branches and derivative program size" begin
    @kernel lazy_position(position::Float64) = begin
        result::Float64 = position > 0 ? log(position) : -position
    end
    positions = [2.0, -1.0, 4.0]
    traced = Reactant.to_rarray(positions)
    primal = vectorize(lazy_position; batched=:position)
    compiled_primal = Reactant.@compile primal(traced)
    @test Array(compiled_primal(traced)) ≈ primal(positions)
    ad = replica(prepare_ad(lazy_position, AutoEnzyme(mode=Enzyme.Reverse),
                           2.0; active=:position, want=:result); batched=:position)
    compiled_ad = Reactant.@compile ad(traced)
    values, gradients = compiled_ad(traced)
    @test Array(values) ≈ primal(positions)
    @test Array(gradients) ≈ [0.5, -1.0, 0.25]
    function ad_operations(count)
        input = Reactant.to_rarray(fill(2.0, count))
        hlo = repr(Reactant.@code_hlo optimize=true ad(input))
        [m.match for m in eachmatch(r"(?:stablehlo|enzyme)\.[a-z_]+", hlo)]
    end
    small = ad_operations(3)
    @test !isempty(small)
    @test small == ad_operations(23)

    # Inspect the optimized primal too: backend batch lowering used to emit
    # one branch per position and compute log on inactive negative positions.
    hlo = repr(Reactant.@code_hlo optimize=true primal(traced))
    @test count("stablehlo.while", hlo) == 1
    regions = Symbol[]
    guarded_logs = Bool[]
    for line in split(hlo, '\n')
        occursin("stablehlo.log", line) && push!(guarded_logs, :if in regions)
        delta = count('{', line) - count('}', line)
        if delta > 0
            tag = occursin("stablehlo.if", line) ? :if : :other
            append!(regions, fill(tag, delta))
        elseif delta < 0
            resize!(regions, length(regions) + delta)
        end
    end
    @test !isempty(guarded_logs) && all(guarded_logs)
end

@testset "multiple compiled HAVE and WANT ports" begin
    @kernel joint_positions(position::Vector{Float64}, direction::Vector{Float64},
                            accept::Bool, step::Float64) = begin
        next::Vector{Float64} = accept ? position .+ step .* direction : position
        accepted::Bool = accept
        norm::Float64 = sum(abs2, next)
        return next, accepted, norm
    end
    batch = vectorize(joint_positions; batched=(:position, :direction, :accept))
    position = reshape(collect(1.0:12.0), 3, 4)
    direction = fill(0.5, 3, 4)
    accept = [true, false, false, true]
    reference = batch(position, direction, accept, 0.2)
    traced = (Reactant.to_rarray(position), Reactant.to_rarray(direction),
              Reactant.to_rarray(accept), Reactant.to_rarray(0.2; track_numbers=true))
    compiled = Reactant.@compile batch(traced...)
    actual = compiled(traced...)
    @test Array(actual[1]) ≈ reference[1]
    @test Array(actual[2]) == reference[2]
    @test Array(actual[3]) ≈ reference[3]
end

@testset "fixed batch with a changing scalar" begin
    @kernel fixed_units(units::Vector{Float64}, amount::Float64) = begin
        result::Float64 = sum(units) * amount
    end
    batch = vectorize(fixed_units; batched=:units)
    units = [1.0 2.0 3.0; 4.0 5.0 6.0]
    read_fixed(amount) = batch(units, amount)
    amount = Reactant.to_rarray(0.5; track_numbers=true)
    compiled = Reactant.@compile read_fixed(amount)
    @test Array(compiled(amount)) ≈ batch(units, 0.5)
    next_amount = Reactant.to_rarray(3.0; track_numbers=true)
    @test Array(compiled(next_amount)) ≈ batch(units, 3.0)
    hlo = repr(Reactant.@code_hlo optimize=true read_fixed(amount))
    @test count("stablehlo.while", hlo) == 1

    # Empty native batches work with declared output types. Compiled in-graph
    # zero-sized outputs remain the upstream tensor.empty export gap (#12).
    @test batch(zeros(2, 0), 0.5) == Float64[]
end

include("test_position_batching_allocation_slices_reactant.jl")
