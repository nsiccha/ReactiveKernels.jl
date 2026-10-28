using ReactiveKernels, Reactant, Enzyme, Test

Reactant.set_default_backend("cpu")

@kernel invariant_plate_guard(x::Vector{Float64}, scale::Float64) = begin
    pointwise = plate(x, scale) do xi, si
        cell::Float64 = si > 0 ? log(si) + xi / si : -2.0
        cell
    end
    total::Float64 = sum(pointwise)
end

@kernel nested_invariant_plate_guard(x::Vector{Float64}, scale::Float64,
                                      shift::Float64) = begin
    pointwise = plate(x, scale, shift) do xi, si, ti
        cell::Float64 = si > 0 ? (ti > 0 ? log(si) + log(ti) + xi / si : -3.0) : -2.0
        cell
    end
    total::Float64 = sum(pointwise)
end

@kernel atomic_invariant_plate_guard(x::Vector{Float64}, scale::Vector{Float64}) = begin
    pointwise = plate(x, Ref(scale)) do xi, shared
        cell::Float64 = shared[1] > 0 ? log(shared[1]) + xi / shared[1] : -2.0
        cell
    end
    total::Float64 = sum(pointwise)
end

@kernel lane_dependent_plate_guard(x::Vector{Float64}, scale::Float64) = begin
    pointwise = plate(x, scale) do xi, si
        cell::Float64 = xi > 0 ? log(xi) / si : -2.0
        cell
    end
    total::Float64 = sum(pointwise)
end

@kernel column_invariant_plate_guard(x::Matrix{Float64}, scale::Float64) = begin
    pointwise = plate(eachcol(x), scale) do xi, si
        cell::Float64 = si > 0 ? sum(xi) / si + log(si) : -2.0
        cell
    end
    total::Float64 = sum(pointwise)
end

@kernel anchored_invariant_plate_guard(x::Vector{Float64}, scale::Float64) = begin
    pointwise = plate(x, scale) do xi, si
        cell::Float64 = si > 0 ? log(si) : -2.0
        cell
    end
    total::Float64 = sum(pointwise)
end

@kernel column_array_invariant_plate_guard(x::Matrix{Float64}, scale::Float64) = begin
    pointwise = plate(eachcol(x), scale) do xi, si
        cell = si > 0 ? xi ./ si .+ log(si) : zero.(xi)
        cell
    end
    flat = vec(stack(pointwise))
    total::Float64 = sum(abs2, flat)
end

_ipb_traced(x::AbstractArray) = Reactant.to_rarray(x)
_ipb_traced(x::Number) = Reactant.to_rarray(x; track_numbers=true)
_ipb_host(x::Reactant.AbstractConcreteArray) = Array(x)
_ipb_host(x) = Reactant.to_number(x)

function _ipb_inventory(hlo)
    # Compare the complete serialized operation sequence, including reducer
    # bodies, packing, constants and control flow in raw and default modules.
    [strip(match.match) for match in eachmatch(
        r"\b(?:stablehlo|chlo|func|arith|enzyme|scf|cf|tensor|math|linalg|memref)\.\w+|(?m:^\s*(?:module|return|cond)\b)", hlo)]
end

function _ipb_check(spec, host_args, changed_args; structure=false, label="guard")
    k = prepare(spec; want=:total)
    gradient(args...) = Enzyme.gradient(Enzyme.Reverse, k, args...)
    traced = map(_ipb_traced, host_args)
    primal = Reactant.@compile k(traced...)
    reverse = Reactant.@compile gradient(traced...)
    for args in (host_args, changed_args...)
        saved = deepcopy(args)
        actual_args = map(_ipb_traced, args)
        @test _ipb_host(primal(actual_args...)) ≈ k(args...)
        expected = gradient(args...)
        actual = map(_ipb_host, reverse(actual_args...))
        @test all(isapprox(a, e) for (a, e) in zip(actual, expected))
        @test args == saved
        @test map(_ipb_host, actual_args) == args
    end
    if structure
        result = Dict{Tuple{String,Bool},Vector{String}}()
        for (mode, fn) in (("primal", k), ("reverse", gradient)), optimized in (false,true)
            hlo = repr(Reactant.@code_hlo optimize=optimized fn(traced...))
            result[(mode, optimized)] = _ipb_inventory(hlo)
            if haskey(ENV, "RK_INVARIANT_BRANCH_HLO_DIR")
                path=joinpath(ENV["RK_INVARIANT_BRANCH_HLO_DIR"],
                    "$label-$mode-$(optimized ? "default" : "raw").mlir")
                write(path, hlo)
            end
            @test occursin("stablehlo.if", hlo)
        end
        executable = Dict{String,Dict{String,Int}}()
        for (mode, compiled) in (("primal", primal), ("reverse", reverse))
            hlo = repr(only(Reactant.XLA.get_hlo_modules(compiled.exec)))
            instructions = collect(eachmatch(
                r"(?m)^\s*(?:ROOT\s+)?%?[\w.-]+ = .*?\s+([A-Za-z][A-Za-z0-9_-]*)\(", hlo))
            assignments = collect(eachmatch(r"(?m)^\s*(?:ROOT\s+)?%?[\w.-]+ = ", hlo))
            @test length(instructions) == length(assignments)
            counts = Dict{String,Int}()
            for m in instructions
                op = m.captures[1]
                counts[op] = get(counts, op, 0) + 1
            end
            executable[mode] = counts
            if haskey(ENV, "RK_INVARIANT_BRANCH_HLO_DIR")
                write(joinpath(ENV["RK_INVARIANT_BRANCH_HLO_DIR"],
                    "$label-$mode.executable.hlo"), hlo)
            end
            # Pure arithmetic may become a selection after MLIR AD while
            # preserving the values, gradients and ownership checked above.
            # Conditional counts are diagnostics, not a strict execution gate.
            println("EXECUTABLE_INVENTORY ", label, "-", mode, " ",
                sort!(collect(counts); by=first))
        end
        return (; mlir=result, executable)
    end
end

@testset "shared plate branches: default primal, ordinary reverse, fixed bodies" begin
    for extension in (:ReactiveKernelsReactantExt, :ReactiveKernelsEnzymeExt)
        @test Base.get_extension(ReactiveKernels, extension) !== nothing
    end
    inventories = []
    for n in (3,7,11)
        x = collect(range(0.2,1.0; length=n))
        push!(inventories, _ipb_check(invariant_plate_guard, (x,2.0),
            ((2x,0.7), (x,-1.0), (x,0.0)); structure=true, label="scalar-$n"))
        pointwise = prepare(invariant_plate_guard; want=:pointwise)
        compiled = Reactant.@compile pointwise(_ipb_traced(x),_ipb_traced(2.0))
        @test _ipb_host(compiled(_ipb_traced(x),_ipb_traced(-1.0))) == fill(-2.0,n)
        @test _ipb_host(compiled(_ipb_traced(x),_ipb_traced(2.0))) ≈ log(2.0) .+ x ./ 2.0
    end
    @test inventories[1] == inventories[2] == inventories[3]

    anchored_inventories = []
    for n in (3,7,11)
        x = collect(range(0.2,1.0;length=n))
        push!(anchored_inventories, _ipb_check(anchored_invariant_plate_guard,
            (x,2.0), ((2x,0.7), (x,-1.0));structure=true,label="anchored-$n"))
    end
    @test anchored_inventories[1] == anchored_inventories[2] == anchored_inventories[3]

    x = [0.2,0.5,1.0]
    _ipb_check(nested_invariant_plate_guard, (x,2.0,0.8),
        ((2x,0.7,1.2), (x,-1.0,-1.0), (x,2.0,-1.0)))
    _ipb_check(atomic_invariant_plate_guard, (x,[2.0,3.0]),
        ((2x,[0.7,5.0]), (x,[-1.0,4.0])))
    _ipb_check(lane_dependent_plate_guard, ([-1.0,0.5,2.0],2.0),
        (([2.0,-1.0,0.7],0.7),))
    column_inventories = []
    for n in (3,7,11)
        x=reshape(collect(range(0.2,1.0;length=2n)),2,n)
        push!(column_inventories, _ipb_check(column_invariant_plate_guard,
            (x,2.0), ((2x,0.7), (x,-1.0)); structure=true,label="columns-$n"))
    end
    @test column_inventories[1] == column_inventories[2] == column_inventories[3]
    array_inventories = []
    for n in (3,7,11)
        x=reshape(collect(range(0.2,1.0;length=2n)),2,n)
        push!(array_inventories, _ipb_check(column_array_invariant_plate_guard,
            (x,2.0), ((2x,0.7), (x,-1.0)); structure=true,label="array-$n"))
    end
    @test array_inventories[1] == array_inventories[2] == array_inventories[3]

    for n in (1,2)
        x=fill(0.5,n)
        _ipb_check(invariant_plate_guard,(x,2.0),((x,-1.0),))
    end
    bound=prepare(invariant_plate_guard;want=:total,bound=(;x=[0.2,0.5,1.0]))
    gradient(s)=only(Enzyme.gradient(Enzyme.Reverse,bound,s))
    primal=Reactant.@compile bound(_ipb_traced(2.0))
    reverse=Reactant.@compile gradient(_ipb_traced(2.0))
    for s in (2.0,0.7,-1.0)
        @test _ipb_host(primal(_ipb_traced(s))) ≈ bound(s)
        @test _ipb_host(reverse(_ipb_traced(s))) ≈ gradient(s)
    end

    # An empty domain takes no arm. Native reverse and compiled primal support
    # it; stock Reactant's separate empty-gradient export gap stays outside
    # this acceptance (benchmark/repro_reactant_empty_gradient.jl).
    empty_kernel=prepare(invariant_plate_guard;want=:total)
    @test empty_kernel(Float64[],-1.0) == 0.0
    @test only(Enzyme.gradient(Enzyme.Reverse,
        s->empty_kernel(Float64[],s),-1.0)) == 0.0
    compiled=Reactant.@compile empty_kernel(_ipb_traced(Float64[]),_ipb_traced(-1.0))
    @test _ipb_host(compiled(_ipb_traced(Float64[]),_ipb_traced(-1.0))) == 0.0

    # Bound empty data carry no lane marker. The empty plate takes the authored
    # cell's native element type, evaluates no cell or shared condition, and
    # its scalar-only reverse compiles.
    bound_empty=prepare(invariant_plate_guard;have=(:x,:scale),want=:total,
                        bound=(;x=Float64[]))
    bound_gradient(v)=only(Enzyme.gradient(Enzyme.Reverse,bound_empty,v))
    bound_primal=Reactant.@compile bound_empty(_ipb_traced(-1.0))
    bound_reverse=Reactant.@compile bound_gradient(_ipb_traced(-1.0))
    for s in (2.0,-1.0)
        @test bound_empty(s) == 0.0
        @test bound_gradient(s) == 0.0
        @test _ipb_host(bound_primal(_ipb_traced(s))) == 0.0
        @test _ipb_host(bound_reverse(_ipb_traced(s))) == 0.0
    end
end
