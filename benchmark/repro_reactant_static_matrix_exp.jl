# Backend-only reproduction: fixed-size matrices stage, but their ordinary
# exponential branches on a traced Bool. No ReactiveKernels code is loaded.
using Reactant, StaticArrays, LinearAlgebra, Test
Reactant.set_default_backend("cpu")

function static_matrix(q)
    Reactant.@allowscalar begin
    SMatrix{3,3}(q[1], q[2], q[3], q[4], q[5], q[6],
        q[7], q[8], q[9])
    end
end
matrix_step(q) = exp(static_matrix(q)) * SVector(0.25, -0.5, 1.0)

@testset "Reactant static-matrix exponential boundary" begin
    q = vec([1.2 -0.4 0.7; 0.5 0.9 -0.2; 0.1 -0.3 1.5])
    rq = Reactant.to_rarray(q)
    matrix = Reactant.@compile static_matrix(rq)
    @test Float64.(matrix(rq)) ≈ static_matrix(q)
    @test matrix_step(q) ≈ exp(Matrix(static_matrix(q))) * [0.25, -0.5, 1.0]
    result = try
        compiled = Reactant.@compile matrix_step(rq)
        Float64.(compiled(rq))
    catch err
        @test err isa TypeError
        @test occursin("non-boolean", sprint(showerror, err)) &&
            occursin("TracedRNumber{Bool}", sprint(showerror, err))
        @test any(frame -> frame.func === :_exp &&
            endswith(string(frame.file), "expm.jl"), stacktrace(catch_backtrace()))
        nothing
    end
    @test_broken result !== nothing && isapprox(result, matrix_step(q))
end
