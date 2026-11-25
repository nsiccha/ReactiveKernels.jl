module JAXGuardedExpmTests

using ReactiveKernels, Test
include("jax_python.jl")
include(joinpath(@__DIR__, "..", "examples", "jax_translations.jl"))

# The (1, 2) entry of exp([d 1; 0 0]) is expm1(d)/d, including its
# removable singularity at zero. Exercise a package mapping in a plate.
@traceable jax_exp_entry(M) = JAXTranslationExample.matrix_exponential(M)[1, 2]
@kernel jax_divided_difference(ds::Vector{Float64}, A::Matrix{Float64}, B::Matrix{Float64}) = begin
    entries = plate(ds) do d
        abs(d) < 1e-5 ? jax_exp_entry(A + d * B) : expm1(d) / d
    end
    total = sum(entries)
    return total
end

@kernel jax_builtin_exp(A::Matrix{Float64}, enabled::Bool) = begin
    total = enabled ? sum(exp(A)) : sum(A)
    return total
end

@testset "jax_source: matrix exponential in a batched branch" begin
    builtin = jax_source(jax_builtin_exp; have = (:A, :enabled), want = :total)
    mapped = jax_source(jax_divided_difference; have = (:ds, :A, :B), want = :total)
    @test occursin("def _rk_guarded_expm(", string(builtin))
    @test occursin("def _rk_guarded_expm(", string(mapped))
    native = prepare(jax_divided_difference; have = (:ds, :A, :B), want = :total)
    ds = [-0.25, 0.0, 1e-7, 0.5]
    expected = native(ds, [0.0 1.0; 0.0 0.0], [1.0 0.0; 0.0 0.0])
    if rk_jax_available()
        mktempdir() do dir
            write(joinpath(dir, "model.py"), mapped)
            script = joinpath(@__DIR__, "jax_guarded_expm.py")
            @test success(`$RK_JAX_PYTHON $script $dir $expected`)
        end
    else
        @test_skip false
    end
end

end
