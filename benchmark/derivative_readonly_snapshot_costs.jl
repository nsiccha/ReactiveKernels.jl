# A readonly design matrix exposes the cost of copying bound-size residual data.
# Usage: this_file installed|baseline|borrowed [exact_baseline_retention_source]
using ReactiveKernels, Test, BenchmarkTools, Statistics, LinearAlgebra
using ReactiveKernels: stage_primal, stage_residuals
import Enzyme

loss(A, b, x) = sum(abs2, A*x - b; init=0.0)
@kernel readonly_spec(A::Matrix{Float64}, b::Vector{Float64}, x::Vector{Float64},
        seed::Float64) = begin
    residual::Vector{Float64} = A*x - b
    y::Float64 = sum(abs2, residual; init=0.0)
    dx::Vector{Float64} = (2seed) .* (transpose(A)*residual)
    return y, dx
end
function gradient!(dx, f::F, A, b, x) where {F}
    fill!(dx, 0.0)
    Enzyme.autodiff(Enzyme.ReverseWithPrimal, Enzyme.Const(f), Enzyme.Active,
        Enzyme.Const(A), Enzyme.Const(b), Enzyme.Duplicated(x, dx))[2]
end

function run_readonly_costs(mode)
    rule = derivative_rule(readonly_spec; primal=:y, covector=:seed,
        cotangents=(A=nothing, b=nothing, x=:dx))
    @test :A in stage_residuals(rule, Val(4))
    for (n, k) in ((0, 3), (128, 8), (2048, 32), (8192, 64))
        A = [sin(0.13i + 0.37j) / sqrt(k) for i in 1:n, j in 1:k]
        b = [cos(0.21i) for i in 1:n]
        x = [0.07sin(j) for j in 1:k]; dx = zeros(k)
        expected = 2 .* (transpose(A) * (A*x - b))
        for f in (loss, rule)
            @test gradient!(dx, f, A, b, x) ≈ loss(A,b,x)
            @test dx ≈ expected
        end
        n == 0 && continue
        arms = ((:native, () -> gradient!(dx,loss,A,b,x)),
                (:rule, () -> gradient!(dx,rule,A,b,x)))
        for round in 1:12
            for (arm,f) in (iseven(round) ? reverse(arms) : arms)
                trial = @benchmark $f() samples=60 evals=1 seconds=0.25
                r = median(trial)
                println((; mode, n, k, round, arm, ns=r.time, bytes=r.memory,
                    allocations=r.allocs, raw_ns=trial.times))
            end
        end
    end
end

mode = Symbol(first(ARGS))
mode in (:installed, :baseline, :borrowed) || error("unknown mode")
if mode == :baseline
    length(ARGS) == 2 || error("baseline needs its exact retention helper source")
    ext = Base.get_extension(ReactiveKernels, :ReactiveKernelsEnzymeExt)
    Base.include(ext, abspath(ARGS[2]))
elseif mode == :borrowed
    include(joinpath(@__DIR__, "derivative_snapshot_borrowing.jl"))
    ResidualBorrowPrototype.install!()
end
println((; mode, julia=VERSION, enzyme=pkgversion(Enzyme), rk=pkgversion(ReactiveKernels),
    library=Enzyme.Compiler.Enzyme_jll.libEnzyme_path))
run_readonly_costs(mode)
println("READONLY_SNAPSHOT_COSTS_COMPLETE")
