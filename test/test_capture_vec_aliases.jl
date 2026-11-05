module CaptureVecAliasTests
using ReactiveKernels, DifferentiationInterface, Enzyme, Test
const linearize = Base.vec

module DifferentVec
vec(x) = reverse(x)
end

# These are the same function, expressed as authored source or a resolved
# binding. The alias lives inside the cell, between its shared index input
# and the captured values, as in generated observation cells.
for (name, callee) in ((:qualified, :(Base.vec)), (:bare, :vec),
        (:resolved, GlobalRef(Base, :vec)), (:renamed, :linearize),
        (:different, :(DifferentVec.vec)))
    @eval @kernel $name(x::AbstractVector{Float64}, index::Vector{Int},
                        lanes::Vector{Int}, key::Vector{Int}) = begin
        values = exp.(x)
        density = plate(lanes) do lane
            lookup = $callee(index)
            values[lookup[lane]]^2
        end
    end
end

@kernel shadowed(x::AbstractVector{Float64}, index::Vector{Int},
                 lanes::Vector{Int}, key::Vector{Int}, vec) = begin
    values = exp.(x)
    density = plate(lanes) do lane
        lookup = vec(index)
        values[lookup[lane]]^2
    end
end

@kernel ragged(x::AbstractVector{Float64}, index::Vector{Vector{Int}},
              lanes::Vector{Int}) = begin
    values = exp.(x)
    density = plate(lanes) do lane
        lookup = Base.vec(index)
        sum(values[lookup[lane]])
    end
end

astsize(x::Expr) = 1 + sum(astsize, x.args; init=0)
astsize(x) = 0

@testset "vec capture aliases preserve selected reads and native Reverse" begin
    sizes = Int[]
    for n in (4, 400)
        x = sin.(1:n) ./ 10
        data = (;index=collect(n:-1:1), lanes=[1,3,2], key=[1,2,2])
        saved = deepcopy(data)
        for kernel in (qualified, bare, resolved, renamed)
            query = plate_partition(kernel, :density; by=:key, index=:which)
            for bound in (NamedTuple(), data)
                k = prepare(query; have=(:x,:index,:lanes,:key,:which),
                    want=:partition_total, bound)
                args(label) = isempty(bound) ? (x,values(data)...,label) : (x,label)
                for label in (1,2)
                    selected = data.index[data.lanes[findall(==(label),data.key)]]
                    @test k(args(label)...) ≈ sum(exp.(2 .* x[selected]))
                    @test partition_footprint(k,args(label)...).x == sort(selected)
                end
                if !isempty(bound)
                    ad = prepare_ad(k, AutoEnzyme(;mode=Enzyme.Reverse),x,1;active=:x)
                    value, gradient = ad_value_and_gradient(ad,x,1)
                    expected = zeros(n)
                    expected[n] = 2exp(2x[n])
                    @test value ≈ exp(2x[n])
                    @test gradient ≈ expected
                    kernel === qualified && push!(sizes,astsize(code_expr(k)))
                end
            end
        end
        @test data == saved
    end
    @test sizes[1] == sizes[2]
end

@testset "a foreign or shadowed vec is not an identity alias" begin
    x = [0.1,0.2,0.3,0.4]
    data = (;index=[4,2,3,1],lanes=[1,3,2],key=[1,2,2])
    for kernel in (different,shadowed)
        query = plate_partition(kernel,:density;by=:key,index=:which)
        extra = kernel === shadowed ? (;vec=reverse) : NamedTuple()
        for bound in (NamedTuple(),merge(data,extra))
            k = prepare(query;have=(:x,keys(data)...,keys(extra)...,:which),
                want=:partition_total,bound)
            args = isempty(bound) ? (x,values(data)...,values(extra)...,1) : (x,1)
            @test k(args...) ≈ exp(2x[1])
            reads = partition_footprint(k,args...).x
            @test reads === :whole || 1 in reads
        end
    end
end

@testset "empty and repeated vector indices retain cell granularity" begin
    x = [0.1,0.2,0.3,0.4]
    data = (;index=[Int[],[4,1],[2,2]],lanes=[1,2,3])
    k = prepare(plate_cell(ragged,:density;index=:i);
        have=(:x,keys(data)...,:i),want=:density_cell,bound=data)
    ad = prepare_ad(k,AutoEnzyme(;mode=Enzyme.Reverse),x,1;active=:x)
    for i in eachindex(data.lanes)
        selected = data.index[i]
        @test k(x,i) ≈ sum(exp.(x[selected]);init=0.0)
        @test partition_footprint(k,x,i).x == sort(unique(selected))
        value,gradient = ad_value_and_gradient(ad,x,i)
        expected = [count(==(j),selected)*exp(x[j]) for j in eachindex(x)]
        @test value ≈ k(x,i)
        @test gradient ≈ expected
    end
end
end
