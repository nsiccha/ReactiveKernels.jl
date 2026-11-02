using Test, ReactiveKernels, ReactiveKernelsPPL, Distributions

_cb_field(w) = map(exp, w)
const _CB_WIDE = @rkppl begin
    w[levels(id)] .~ Normal.(0, 1)
    z[levels(id)] .~ Normal.(0, 1)
    field = _cb_field(w)
    @plate for i in eachindex(y)
        y[i] ~ Poisson(exp(z[id[i]]) * field[i])
    end
end
const _CB_BRANCH = @rkppl begin
    a ~ Normal(0, 1)
    z[1:4] .~ Normal.(0, 1)
    @plate for i in eachindex(y)
        mu = a > 0 ? sum(z[left[i]]) : sum(z[right[i]])
        y[i] ~ Normal(mu, 1)
    end
end
const _CB_LOCAL = @rkppl begin
    z[levels(id)] .~ Normal.(0, 1)
    @plate for i in eachindex(y)
        y[i] ~ Poisson(exp(z[id[i]]))
    end
end
const _CB_SUM = @rkppl begin
    w[levels(id)] .~ Normal.(0, 1)
    z[levels(id)] .~ Normal.(0, 1)
    field = _cb_field(w)
    rate = exp.(z[id]) .* field
    @plate for i in eachindex(y)
        y[i] ~ Normal(log1p(sum(rate[rows[i]])) + w[1] + z[anchor[i]], 1)
    end
end

struct _CBReads <: AbstractVector{Float64}
    values::Vector{Float64}
    reads::Vector{Int}
end
Base.size(v::_CBReads) = size(v.values)
Base.IndexStyle(::Type{_CBReads}) = IndexLinear()
Base.getindex(v::_CBReads, i::Int) = (push!(v.reads, i); v.values[i])

function _cb_fixture(n)
    bound = _CB_WIDE(; id=collect(1:n)) | (; y=ones(Int,n))
    built = build_kernel(bound)
    u = collect(range(-0.2,0.3; length=2n))
    q = prepare_cell_query(built,bound,:y; partitioned=:z,u0=u)
    q, u
end
function _cb_sweep(q, fp, u, n)
    with_cell_batch(q,u) do batch
        value, count = 0.0, 0
        for i in 1:n
            value += q(batch,i)
            count += length(fp(batch,i).unconstrained)
        end
        (; value, count)
    end
end
_cb_nodes(x) = x isa Expr ? 1 + sum(_cb_nodes,x.args; init=0) : 1

@testset "Local structural footprints and callback-scoped values" begin
    q,u = _cb_fixture(4)
    fp = partition_footprint_kernel(q; shared=false)
    full = partition_footprint_kernel(q)
    expected = [q(u,i) for i in 1:4]
    escaped = Ref{Any}()
    source = _CBReads(copy(u),Int[])
    result = with_cell_batch(q,source) do batch
        escaped[] = batch
        initial_reads = length(source.reads)
        source.values .= 99 # the batch owns its copied point
        for i in 1:4
            @test q(batch,i) ≈ expected[i]
            @test fp(batch,i).unconstrained == [4+i]
            @test fp(batch,i).unconstrained ==
                intersect(Base.invokelatest(full,u,i).unconstrained,5:8)
        end
        @test length(source.reads) == initial_reads
        :result
    end
    @test result === :result
    @test_throws ContractValidationError q(escaped[],1)
    @test_throws ContractValidationError fp(escaped[],1)
    q2,_ = _cb_fixture(4)
    with_cell_batch(q,u) do batch
        @test_throws ContractValidationError q2(batch,1)
        @test_throws ContractValidationError q(escaped[],1) # never revives
    end
    @test_throws ErrorException with_cell_batch(q,u) do batch
        escaped[] = batch
        error("callback failed")
    end
    @test_throws ContractValidationError q(escaped[],1)
    changed = copy(u); changed[1] += 0.4
    @test_throws ContractValidationError with_cell_batch(q,u) do batch
        q(changed,1)
        @test_throws ContractValidationError q(batch,1)
        @test_throws ContractValidationError fp(batch,1)
    end
    @test _cb_sweep(q,fp,changed,4).value ≈ sum(q(changed,i) for i in 1:4)
    @test fp(_CBReads(copy(changed),Int[]),2).unconstrained == [6]
    @test_throws ContractValidationError with_cell_batch(q,zeros(3)) do batch
        q(batch,1)
    end
end

@testset "Local footprints retain lazy branches without a prefix" begin
    localbound = _CB_LOCAL(; id=[1,2,3]) | (; y=[1,2,3])
    localbuilt = build_kernel(localbound)
    localq = prepare_cell_query(localbuilt,localbound,:y;partitioned=:z)
    localfp = partition_footprint_kernel(localq;shared=false)
    with_cell_batch(localq,[0.1,0.2,0.3]) do batch
        for i in 1:3
            @test localq(batch,i) ≈ logpdf(Poisson(exp(0.1i)),i)
            @test localfp(batch,i).unconstrained == [i]
        end
    end
    data = (; left=[[1,1,3],Int[],[2]], right=[[2,4],Int[],[2,2]],y=[0.2,-0.1,0.4])
    bound = _CB_BRANCH(; data.left,data.right) | (; data.y)
    built = build_kernel(bound)
    u = [0.3,0.2,0.5,-0.1,0.7]
    q = prepare_cell_query(built,bound,:y; partitioned=:z=>collect(1:4),u0=u)
    fp = partition_footprint_kernel(q; shared=false)
    full = partition_footprint_kernel(q.kernel)
    for sign in (-1,1)
        u[1] = sign*0.3
        with_cell_batch(q,u) do batch
            for i in eachindex(data.y)
                selected = sort!(unique([1 .+ data.left[i]; 1 .+ data.right[i]]))
                @test fp(batch,i).unconstrained == selected
                @test selected == intersect(Base.invokelatest(full,u,i).unconstrained,2:5)
                @test q(batch,i) ≈ q(u,i)
            end
        end
    end
    raw = prepare_cell_query(built,bound,:y)
    @test_throws ContractValidationError partition_footprint_kernel(raw;shared=false)
end

@testset "Summand and outer values share one local-footprint batch" begin
    data = (; id=collect(1:4),rows=[[1,1,3],Int[],[2,4]],anchor=[2,3,1],y=[0.2,-0.1,0.4])
    bound = _CB_SUM(; data.id,data.rows,data.anchor) | (; data.y)
    built = build_kernel(bound)
    u = collect(range(-0.2,0.3;length=8))
    f = prepare_cell_factors(built,bound,:y;partitioned=:z,u0=u)
    term,outer = prepare_cell_factor_query(f,1),prepare_cell_factor_query(f,:outer)
    tfp,ofp = partition_footprint_kernel(term;shared=false),partition_footprint_kernel(outer;shared=false)
    fullterm,fullouter = partition_footprint_kernel(term),partition_footprint_kernel(outer)
    q = prepare_cell_query(built,bound,:y;partitioned=:z,u0=u)
    with_cell_batch(f,u) do batch
        for i in eachindex(data.y)
            total = 0.0
            for (t,row) in enumerate(reduction_rows(f,1,i))
                @test term(batch,i,t) ≈ term(u,i,t)
                total += term(batch,i,t)
                @test tfp(batch,i,t).unconstrained == [4+row]
                @test tfp(batch,i,t).unconstrained ==
                    intersect(Base.invokelatest(fullterm,u,i,t).unconstrained,5:8)
            end
            @test outer(batch,[total],i) ≈ q(u,i)
            @test ofp(batch,[total],i).unconstrained == [4+data.anchor[i]]
            @test ofp(batch,[total],i).unconstrained ==
                intersect(Base.invokelatest(fullouter,u,[total],i).unconstrained,5:8)
        end
        @test_throws ContractValidationError outer(batch,Float64[],1)
    end
end

@testset "Preparation sweeps scale with local support" begin
    measurements = NamedTuple[]
    for n in (64,256,1024)
        q,u = _cb_fixture(n)
        fp = partition_footprint_kernel(q;shared=false)
        result = _cb_sweep(q,fp,u,n)
        @test result.count == n
        @test result.value ≈ sum(q(u,i) for i in 1:n)
        bytes = minimum(@allocated(_cb_sweep(q,fp,u,n)) for _ in 1:3)
        elapsed = minimum(@elapsed(_cb_sweep(q,fp,u,n)) for _ in 1:3)
        push!(measurements,(;n,bytes,elapsed,nodes=_cb_nodes(fp.kernel.ast)))
    end
    @info "local footprint and value batch scaling" measurements
    @test all(m -> m.nodes == first(measurements).nodes,measurements)
    @test last(measurements).bytes <= 5measurements[2].bytes + 4096
end
