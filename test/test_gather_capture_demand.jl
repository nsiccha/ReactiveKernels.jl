module GatherCaptureDemandTests
using ReactiveKernels, DifferentiationInterface, Enzyme, Test

# Binding the repeated indices interchanges exp and the gather inside means.
# y shares a producer with means, so both observation cells keep one emitter.
@kernel grouped(x::AbstractVector{Float64}, z::AbstractVector{Float64},
        M::Matrix{Float64}, nrows::Int, lanes::Vector{Int}, groups::Vector{Vector{Int}},
        key_y::Vector{Int}, key_soil::Vector{Int}) = begin
    shared = exp.(x)
    R = reshape(z, nrows, 2) * M
    mu = log.(R[:, 1])
    y = plate(lanes, Ref(shared)) do i, A
        A[i]^2
    end
    means = plate(eachindex(groups)) do g
        indices = Base.vec(groups)
        exp.(mu[indices[g]]) + shared[indices[g]]
    end
    soil = plate(eachindex(groups)) do g
        sum(means[g])
    end
end

# A superficially similar authored branch may demand more rows than its final
# index: it has no compiler proof that cover describes those indices.
@kernel authored(z::AbstractVector{Float64}, groups::Vector{Vector{Int}},
                 cover::Vector{Int}, key::Vector{Int}) = begin
    mu = log.(z)
    cells = plate(eachindex(groups)) do g
        c = cover[g]
        sum(c > 0 ? exp.(ReactiveKernels._rk_gather_source(mu, c))[groups[g]] :
                    exp.(mu[groups[g]]))
    end
end

astsize(ex::Expr) = 1 + sum(astsize, ex.args; init=0)
astsize(ex) = 0

@testset "bound gather maps preserve captured row demand" begin
    sizes = Int[]
    for n in (4, 400)
        x = sin.(1:n) ./ 10
        z = 1 .+ cos.(1:2n) ./ 10
        data = (; M=[1.0 0.2; 0.3 0.8], nrows=n, lanes=collect(1:n),
            groups=[[1,2,1], [n,n], Int[]], key_y=fill(2,n), key_soil=[1,3,4])
        saved = deepcopy(data)
        spec = plate_partition(grouped, (:y,:soil);
            by=(y=:key_y,soil=:key_soil),index=:unit,locality=:ignore)
        for bound in (NamedTuple(), data)
            k = prepare(spec;have=(:x,:z,keys(data)...,:unit),want=:partition_total,bound)
            args(a,b,label) = isempty(bound) ? (a,b,values(data)...,label) : (a,b,label)
            R = reshape(z,n,2)*data.M
            for label in 1:4
                rows = label == 1 ? [1,2,1] : label == 3 ? [n,n] : Int[]
                expected = label == 2 ? sum(exp.(2 .* x)) :
                    sum(exp.(log.(R[rows,1])) + exp.(x[rows]))
                @test k(args(x,z,label)...) ≈ expected
                footprint = partition_footprint(k,args(x,z,label)...)
                @test footprint.z == sort(unique([rows; rows .+ n]))
                @test footprint.x == (label == 2 ? collect(1:n) : sort(unique(rows)))
                isempty(bound) && continue
                for active in (:x,:z)
                    ad = prepare_ad(k,AutoEnzyme(;mode=Enzyme.Reverse),x,z,label;active)
                    value,gradient = ad_value_and_gradient(ad,x,z,label)
                    @test value ≈ expected
                    dx=zeros(n); dz=zeros(2n)
                    if label == 2
                        dx .= 2 .* exp.(2 .* x)
                    else
                        for row in rows
                            dx[row] += exp(x[row])
                            dz[row] += data.M[1,1]
                            dz[row+n] += data.M[2,1]
                        end
                    end
                    @test gradient ≈ (active === :x ? dx : dz) atol=2e-12 rtol=2e-12
                end
            end
            # An inactive whole predictor may not run to establish a cell type.
            @test k(args(x,-z,2)...) ≈ sum(exp.(2 .* x))
            @test k(args(x,-z,4)...) == 0.0
            @test_throws DomainError k(args(x,-z,1)...)
            isempty(bound) || push!(sizes,astsize(code_expr(k)))
        end
        @test data == saved
    end
    @test sizes[1] == sizes[2]
end

@testset "authored cover guards keep their actual reads" begin
    data=(;groups=[[1,1]],cover=[2],key=[1])
    k=prepare(plate_partition(authored,:cells;by=:key,index=:unit);
        have=(:z,keys(data)...,:unit),want=:partition_total,bound=data)
    @test k([1.0,2.0],1) ≈ 2.0
    @test_throws DomainError k([1.0,-1.0],1)
end
end
