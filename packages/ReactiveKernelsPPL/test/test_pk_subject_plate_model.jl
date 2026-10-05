module PKSubjectPlateModelTests
using ReactiveKernels, ReactiveKernelsPPL, DifferentiationInterface, Enzyme
using Distributions, Test
using ..PKSubjectPlateTests: oracle, finite_gradient

function rawdata(G)
    Dict{Symbol,Any}(
        :subj => [s for s in 1:G for _ in 1:s+1],
        :time => [t for s in 1:G for t in range(0.0, 16.0; length=s+1)],
        :dsubj => repeat(collect(1:G); inner=5),
        :dtime => repeat([0.0, 2.0, 4.0, 6.0, 8.0], G),
        :damt => [2.0+0.3s for s in 1:G for _ in 1:5],
        :dv => [0.2+0.01i for i in 1:sum(s+1 for s in 1:G)],
        :age_s => collect(range(-0.5, 0.5; length=G)))
end

# Include nested plans: counting only the outer recipe would miss hidden
# subject or event replication. The count must be independent of data sizes.
function structure(program)
    counts = Dict{Symbol,Int}()
    for (; kind) in recipe_inventory(program)
        counts[kind] = get(counts, kind, 0)+1
    end
    counts
end

@testset "PK subject plate: generated density, default reverse and graph growth" begin
    for form in (:legacy, :eachindex, :axes)
        file = form === :legacy ? "49_kernel_grouped_pk.jl" : "99_plate_49_grouped_pk.jl"
        source = read(joinpath(@__DIR__, "corpus", file), String)
        form === :axes && (source = replace(source, "eachindex(dv)" => "axes(dv, 1)"))
        ast = Meta.parse(source)
        inventories = Dict{Symbol,Int}[]
        for G in (2, 5)
            data = rawdata(G)
            names = Tuple(keys(data))
            plan = lower_rkppl(ast, names; conditioned=names)
            dims = form === :legacy ? Dict(:kernel_nsub_conc=>G) : Dict{Symbol,Int}()
            bound = bind_data(plan, data; dims)
            built = build_kernel(bound)
            u = [0.02cos(i) for i in 1:built.layout.total]
            query = prepare_query(built, bound, :sampler)
            function density(x)
                nt = constrain(built.layout, x)
                p = vcat(transpose(nt.b0_vc .+ nt.b1_vc .* data[:age_s]),
                    transpose(nt.b0_k10 .+ nt.b1_k10 .* data[:age_s]),
                    fill(nt.b0_k12, 1, G), fill(nt.b0_k21, 1, G), fill(nt.b0_ka, 1, G))
                cols = Tuple(bound.columns[Symbol(:pk_sched_, n)] for n in
                    (:op_type, :op_dt, :op_amount, :op_interval, :op_count, :op_read_idx))
                reads = oracle(bound.columns[:pk_sched_op_ends], cols,
                    zeros(length(first(cols))), p)
                mu = reads[bound.columns[:pk_sched_obs_map]]
                prior = logpdf(Exponential(1), nt.sigma) + log(nt.sigma)
                for name in propertynames(nt)
                    name === :sigma && continue
                    prior += logpdf(Normal(0, 1), getproperty(nt, name))
                end
                prior + sum(logpdf.(Normal.(mu, nt.sigma), data[:dv]))
            end
            @test Base.invokelatest(query, u) ≈ density(u) rtol=2e-11
            sampler = prepare_sampler(built, bound, u;
                backend=AutoEnzyme(; mode=Enzyme.Reverse))
            value, grad = sampler_value_and_gradient!(sampler, similar(u), u)
            @test value ≈ density(u) rtol=2e-11
            @test grad ≈ finite_gradient(density, u) rtol=2e-6 atol=2e-7
            inventory = structure(query)
            @test inventory[:plate] >= 1
            @test inventory[:scan] == 1
            push!(inventories, inventory)
        end
        @test first(inventories) == last(inventories)
    end
end
end
