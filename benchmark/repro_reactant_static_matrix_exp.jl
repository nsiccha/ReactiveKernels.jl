# Backend-only reproduction: fixed-size matrices stage, but their ordinary
# exponential branches on a traced Bool. No ReactiveKernels code is loaded.
using Reactant, StaticArrays, LinearAlgebra, Test
Reactant.set_default_backend("cpu")

function pk_system(q)
    Reactant.@allowscalar begin
    k10, k12, k21, ka = exp(q[1]), exp(q[2]), exp(q[3]), exp(q[4])
    SMatrix{3,3}(-ka, ka, 0.0, 0.0, -(k10 + k12), k12,
        0.0, k21, -k21)
    end
end
pk_step(q) = exp(pk_system(q)) * SVector(1.0, 0.0, 0.0)

@testset "Reactant static-matrix exponential boundary" begin
    q = log.([0.1, 0.2, 0.3, 0.5])
    rq = Reactant.to_rarray(q)
    matrix = Reactant.@compile pk_system(rq)
    @test Float64.(matrix(rq)) ≈ pk_system(q)
    @test pk_step(q) ≈ exp(Matrix(pk_system(q))) * [1.0, 0.0, 0.0]
    result = try
        compiled = Reactant.@compile pk_step(rq)
        Float64.(compiled(rq))
    catch err
        @test err isa TypeError
        @test occursin("non-boolean", sprint(showerror, err)) &&
            occursin("TracedRNumber{Bool}", sprint(showerror, err))
        @test any(frame -> frame.func === :_exp &&
            endswith(string(frame.file), "expm.jl"), stacktrace(catch_backtrace()))
        nothing
    end
    @test_broken result !== nothing && isapprox(result, pk_step(q))
end
