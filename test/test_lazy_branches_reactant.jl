using ReactiveKernels, Reactant, Test
import Enzyme

# Authored `if`/`?:`/`&&`/`||` retain conditional regions in Reactant MLIR
# (docs/src/constraints.md). Compiled values and gradients must match Julia,
# including when the backend speculates pure arithmetic in an inactive arm.

@kernel lazy_scalar_guard(x::Float64) = begin
    guarded::Float64 = x > 0 ? log(x) : -1.0
    return guarded
end

@kernel shared_split_arm(y::Vector{Float64}, scale::Float64) = begin
    pointwise=plate(y,scale) do yi,si
        cell::Float64=yi<=0 ? (si>0 ? log(si) : -Inf) : si*yi
        cell
    end
    total::Float64=sum(pointwise)
end

@kernel typed_count_conversion(x, scale::Float64) = begin
    count = ReactiveKernels._tensorized_trunc(Int, floor(x))
    total::Float64 = scale * count
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

# A constructed endpoint under a branch arm (a missingness guard): the arm
# carries its own endpoint evaluation, and a data-bound split removes the
# branch before any backend sees it.
@kernel arm_standard_normal() = begin
    logpdf(z::Float64)::Float64 = -0.5 * log(2π) - 0.5z^2
end

@kernel arm_location_scale(standard, location::Float64, scale::Float64) = begin
    log_scale::Float64 = log(scale)
    scale::Float64 = exp(log_scale)
    standardized(x::Float64)::Float64 = (x - location) / scale
    logpdf(x::Float64)::Float64 = begin
        z::Float64 = standardized(x)
        standard.logpdf(z) - log_scale
    end
end

@kernel arm_normal = arm_location_scale(arm_standard_normal)

@kernel reactant_endpoint_arm(y_full::Vector{Float64}, lp::Vector{Float64},
                              mis_pos::Vector{Int}) = begin
    pointwise = plate(y_full, lp, mis_pos) do yf, lpi, mp
        cell::Float64 = mp == 0 ? arm_normal(lpi, 1.5).logpdf(yf) : 0.0 * yf
        cell
    end
    total::Float64 = sum(pointwise)
    return total
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
    # The inactive arm contributes exactly zero to the gradient, never NaN.
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

@testset "reverse through a batched lazy branch matches at every lane count" begin
    # Enzyme reverse through a lazy `if` inside a batched plate cell lowers
    # since Reactant 0.2.289 (the upstream Enzyme-JAX remover gap, RK issue
    # #13, is fixed there). The primal is lane-count independent either way.
    # `benchmark/repro_reactant_batch_if_reverse.jl` (Reactant + Enzyme only)
    # guards the lift as a regression test.
    k = prepare(lazy_plate_guard; want = :total)
    gradient(v) = Enzyme.gradient(Enzyme.Reverse, k, v)
    small = [2.0, -1.0, 0.5, -3.0]
    large = [2.0, -1.0, 0.5, -3.0, 1.5, -0.5]
    xlarge = [isodd(i) ? 0.5i : -0.25i for i in 1:32]
    compiled_small = Reactant.@compile gradient(_traced(small))
    @test _host(only(compiled_small(_traced(small)))) ≈ [0.5, 0.0, 2.0, 0.0]
    @test _host((Reactant.@compile k(_traced(large)))(_traced(large))) ≈ k(large)
    compiled_large = Reactant.@compile gradient(_traced(large))
    @test _host(only(compiled_large(_traced(large)))) ≈
        [0.5, 0.0, 2.0, 0.0, 1 / 1.5, 0.0]
    compiled_xlarge = Reactant.@compile gradient(_traced(xlarge))
    @test _host(only(compiled_xlarge(_traced(xlarge)))) ≈
        [xi > 0 ? 1 / xi : 0.0 for xi in xlarge]
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

@testset "a split arm endpoint compiles with no backend branch" begin
    y_full = [0.5, -1.0, 2.0, 0.25]
    lp = [0.0, 1.0, -0.5, 0.75]
    mis_pos = [0, 1, 0, 1]
    nlogpdf(x, m, s) = -0.5 * log(2π) - log(s) - 0.5 * ((x - m) / s)^2
    ref = sum(mp == 0 ? nlogpdf(yf, lpi, 1.5) : 0.0 * yf
              for (yf, lpi, mp) in zip(y_full, lp, mis_pos))
    gref = [(mp == 0 ? (yf - lpi) / 1.5^2 : 0.0)
            for (yf, lpi, mp) in zip(y_full, lp, mis_pos)]
    k = prepare(reactant_endpoint_arm; have = (:y_full, :lp, :mis_pos),
                want = :total, bound = (; y_full, mis_pos))
    hlo = repr(Reactant.@code_hlo optimize = false k(_traced(lp)))
    @test !occursin("stablehlo.if", hlo)
    @test !occursin("stablehlo.select", hlo)
    compiled = Reactant.@compile k(_traced(lp))
    @test _host(compiled(_traced(lp))) ≈ ref
    gradient(v) = Enzyme.gradient(Enzyme.Reverse, k, v)
    compiled_gradient = Reactant.@compile gradient(_traced(lp))
    @test _host(only(compiled_gradient(_traced(lp)))) ≈ gref
end

@testset "zero-owner endpoints and bare operations stay inside their arm" begin
    if !isdefined(@__MODULE__, :BranchPartition)
        include("fixtures/branch_partition.jl")
    end
    structures = Dict{String,Int}[]
    for n in (4,8)
        x=fill(0.3,n); flag=repeat([0,1],n÷2)
        k=prepare(BranchPartition.bare_helper_branch;have=(:x,:flag),want=:total,bound=(;flag))
        hlo=repr(Reactant.@code_hlo optimize=false k(_traced(x)))
        @test !occursin("stablehlo.if",hlo)
        @test _host((Reactant.@compile k(_traced(x)))(_traced(x))) ≈ sum(2x[i] for i in eachindex(x) if flag[i]==0)
        gradient(v)=Enzyme.gradient(Enzyme.Reverse,k,v)
        cg=Reactant.@compile gradient(_traced(x))
        @test _host(only(cg(_traced(x)))) ≈ [f==0 ? 2. : 0. for f in flag]
        ops=Dict{String,Int}()
        for m in eachmatch(r"stablehlo\.[a-z_]+",hlo)
            ops[m.match]=get(ops,m.match,0)+1
        end
        push!(structures,ops)
    end
    @test structures[1]==structures[2]
end

@testset "a split shared arm retains a traced guard at every lane count" begin
    structures=Dict{String,Int}[]
    for n in (12,24)
        y=repeat([0.,1.,2.],n÷3)
        k=prepare(shared_split_arm;have=(:y,:scale),want=:total,bound=(;y))
        s=_traced(1.3)
        hlo=repr(Reactant.@code_hlo optimize=false k(s))
        @test occursin("stablehlo.if",hlo)
        c=Reactant.@compile k(s)
        @test _host(c(s))≈n÷3*(log(1.3)+3*1.3)
        gradient(v)=only(Enzyme.gradient(Enzyme.Reverse,k,v))
        cg=Reactant.@compile gradient(s)
        @test _host(cg(s))≈n÷3*(1/1.3+3)
        # Reuse the executable across both arms of the shared guard. Negative
        # and zero scales must keep the inactive logarithm out of the value
        # and ordinary derivative; the bound lane data remain caller-owned.
        saved=copy(y)
        for scale in (0.7,-0.4,0.0)
            input=_traced(scale)
            expected=n÷3*(scale>0 ? log(scale)+3scale : -Inf)
            expected_gradient=n÷3*(scale>0 ? inv(scale)+3 : 3.0)
            @test isequal(_host(c(input)),expected) || _host(c(input))≈expected
            @test _host(cg(input))≈expected_gradient
            @test k(scale)==expected || k(scale)≈expected
            @test y==saved
            @test _host(input)==scale
        end
        ops=Dict{String,Int}()
        for m in eachmatch(r"stablehlo\.[a-z_]+",hlo)
            ops[m.match]=get(ops,m.match,0)+1
        end
        push!(structures,ops)
    end
    @test structures[1]==structures[2]
end

@testset "an empty bound split-arm domain evaluates no cell" begin
    # Zero lanes run no cell. Before RK supplied the native element type,
    # the compiled primal read the first element of the zero-length operand
    # (`Create Operation 'stablehlo.dynamic_slice' failed`).
    y=Float64[]
    k=prepare(shared_split_arm;have=(:y,:scale),want=:total,bound=(;y))
    c=Reactant.@compile k(_traced(1.3))
    gradient(v)=only(Enzyme.gradient(Enzyme.Reverse,k,v))
    cg=Reactant.@compile gradient(_traced(1.3))
    for scale in (1.3,-0.4,0.0)
        input=_traced(scale)
        @test _host(c(input))==0.0
        @test _host(cg(input))==0.0
        @test k(scale)==0.0
        @test gradient(scale)==0.0
        @test _host(input)==scale
    end
    @test isempty(y)
    # One lane at the same shape keeps its cell.
    one=prepare(shared_split_arm;have=(:y,:scale),want=:total,bound=(;y=[2.0]))
    @test _host((Reactant.@compile one(_traced(1.3)))(_traced(0.7)))≈1.4
end

@testset "typed integer conversions accept traced integer and real inputs" begin
    k = prepare(typed_count_conversion; want=:total)
    for x in (3, 3.7)
        rx, rs = _traced(x), _traced(1.2)
        compiled = Reactant.@compile k(rx, rs)
        @test _host(compiled(rx, rs)) ≈ 3.6
        gradient(x, s) = only(Enzyme.gradient(Enzyme.Reverse, t -> k(x, t), s))
        cg = Reactant.@compile gradient(rx, rs)
        @test _host(cg(rx, rs)) ≈ 3.0
    end
end
