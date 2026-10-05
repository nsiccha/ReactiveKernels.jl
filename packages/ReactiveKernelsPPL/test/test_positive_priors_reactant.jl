# _pp_backend_check checks native Enzyme against finite differences,
# Reactant primal/reverse against native, and returns the backend operation
# multiset. It is shared with test_parameter_priors_reactant.jl.
@testset "positive prior slots retain native and compiled values, gradients and structure" begin
    for slot in (:sampled, :vector, :matrix)
        structures = Dict{String,Int}[]
        for n in (5, 9)
            data = Dict(:y => fill(0.2, n))
            expr, q = if slot === :sampled
                (quote s ~ HalfCauchy(2); y .~ Normal.(s, 1) end, (s = 1.2,))
            elseif slot === :vector
                (quote
                    s[1:2] .~ truncated(Normal(0.7, 2), 0.3, 4)
                    mu = sum(s)
                    y .~ Normal.(mu, 1)
                end, (s = [1.2, 1.4],))
            else
                (quote
                    s[1:2, 1:2] .~ HalfNormal.(2)
                    mu = sum(s)
                    y .~ Normal.(mu, 1)
                end, (s = fill(0.8, 2, 2),))
            end
            before = deepcopy(data)
            push!(structures,_pp_backend_check(expr,data,q; structure_body = true))
            @test data == before
        end
        @test structures[1] == structures[2]
    end
end
