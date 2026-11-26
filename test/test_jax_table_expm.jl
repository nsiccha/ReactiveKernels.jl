module JAXTableExpmTests

using ReactiveKernels, Test
include("jax_python.jl")
include(joinpath(@__DIR__, "..", "examples", "jax_translations.jl"))

# A recurrence whose step reads a shared record table and takes a matrix
# exponential only where its closed form has a removable singularity: the
# (1, 2) entry of exp([d 1; 0 0]) is expm1(d)/d. The step is replayed in
# reverse and the exponential is a lazy arm; both must hold together.
@traceable jax_table_exp_entry(M) = JAXTranslationExample.matrix_exponential(M)[1, 2]
@kernel jax_table_expm_walk(a::Float64, keys::Vector{Float64}, orders::Vector{Vector{Int}},
                            A::Matrix{Float64}, B::Matrix{Float64}) = begin
    entries = plate(keys) do key
        (key, a * key)
    end
    totals = plate(orders, Ref(entries), Ref(A), Ref(B)) do order, table, A, B
        path = scan(order, Ref(table), Ref(A), Ref(B); init = 0.0) do state, index, rows, A, B
            row = rows[index]
            d = row[2]
            factor = abs(d) < 1e-5 ? jax_table_exp_entry(A + d * B) : expm1(d) / d
            next = 0.5 * state + factor * row[1]
            (next, next)
        end
        sum(path)
    end
    total = sum(totals)
    total
end

@testset "jax_source: a lazy matrix exponential in a replayed scan step" begin
    have = (:a, :keys, :orders, :A, :B)
    src = jax_source(jax_table_expm_walk; have, want = :total, data = (:keys, :orders))
    @test occursin("@_rk_table_scan_step", string(src))
    @test occursin("def _rk_guarded_expm(", string(src))
    native = prepare(jax_table_expm_walk; have, want = :total)
    # The data of `jax_table_expm.py`: ragged orders with an empty unit.
    keys = [0.25, -0.5]
    orders = [[1, 2, 1], [2], Int[], [2, 2, 1, 1, 2, 1, 2]]
    A, B = [0.0 1.0; 0.0 0.0], [1.0 0.0; 0.0 0.0]
    if rk_jax_available()
        mktempdir() do dir
            write(joinpath(dir, "model.py"), src)
            script = joinpath(@__DIR__, "jax_table_expm.py")
            away = native(0.6, keys, orders, A, B)
            tie = native(0.0, keys, orders, A, B)
            @test success(`$RK_JAX_PYTHON $script $dir $away $tie`)
        end
    else
        @test_skip false
    end
end

end
