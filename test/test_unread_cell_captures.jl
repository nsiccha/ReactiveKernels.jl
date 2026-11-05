module UnreadCellCaptureTests
using ReactiveKernels, DifferentiationInterface, Enzyme, Test

# The body is already planned when a cell/partition query sees it. Its unused
# shared formal must not keep the entire producer of b in that query.
@kernel captured(x::AbstractVector{Float64}, y::AbstractVector{Float64},
                 lanes::Vector{Int}, key::Vector{Int}) = begin
    a = exp.(x)
    b = log.(y)
    cells = plate(lanes, Ref(a), Ref(b)) do i, A, B
        A[i]^2
    end
end

@kernel zipped(x::Vector{Float64}, unused::Vector{Float64}) = begin
    cells = plate(x, unused) do xi, ui
        xi^2
    end
end

astsize(x::Expr) = 1 + sum(astsize,x.args;init=0)
astsize(x) = 0

@testset "already-built cells discard unread shared captures" begin
    sizes = Int[]
    for n in (3,300)
        x = sin.(1:n) ./ 10
        y = fill(-1.0,n) # log would fail if the unused producer ran.
        data = (;lanes=[n,1,1],key=[1,2,2])
        saved = deepcopy(data)
        for partition in (false,true)
            spec = partition ? plate_partition(captured,:cells;by=:key,index=:i) :
                plate_cell(captured,:cells;index=:i)
            want = partition ? :partition_total : :cells_cell
            for bound in (NamedTuple(),data)
                k = prepare(spec;have=(:x,:y,:lanes,:key,:i),want,bound)
                args(i) = isempty(bound) ? (x,y,values(data)...,i) : (x,y,i)
                for i in (1,2)
                    rows = partition ? data.lanes[findall(==(i),data.key)] : [data.lanes[i]]
                    @test k(args(i)...) ≈ sum(exp.(2 .* x[rows]))
                    fp = partition_footprint(k,args(i)...)
                    @test fp.x == sort(unique(rows))
                    @test isempty(fp.y)
                end
                isempty(bound) && continue
                ad = prepare_ad(k,AutoEnzyme(;mode=Enzyme.Reverse),x,y,1;active=:x)
                value,gradient = ad_value_and_gradient(ad,x,y,1)
                expected = zeros(n); expected[n] = 2exp(2x[n])
                @test value ≈ exp(2x[n])
                @test gradient ≈ expected
                inactive = prepare_ad(k,AutoEnzyme(;mode=Enzyme.Reverse),x,y,1;active=:y)
                @test all(iszero,last(ad_value_and_gradient(inactive,x,y,1)))
                partition && push!(sizes,astsize(code_expr(k)))
            end
        end
        @test data == saved
    end
    @test sizes[1] == sizes[2]
end

@testset "explicitly requested captures and batched domains remain live" begin
    data = (;lanes=[1,3],key=[1,2])
    q = prepare(plate_cell(captured,:cells;index=:i);
        have=(:x,:y,:lanes,:key,:i),want=(:cells_cell,:b),bound=data)
    x=[0.1,0.2,0.3]; y=[1.0,2.0,3.0]
    cell,b = q(x,y,1)
    @test cell ≈ exp(2x[1])
    @test b == log.(y)
    fp = partition_footprint(q,x,y,1).y
    @test fp === :whole || fp == [1,2,3]
    @test_throws DomainError q(x,-y,1)
    whole = prepare(zipped)
    @test whole([2.0],zeros(3)) == fill(4.0,3)
    @test isempty(whole([2.0],Float64[]))
    @test_throws DimensionMismatch whole([1.0,2.0],zeros(3))
end
end
