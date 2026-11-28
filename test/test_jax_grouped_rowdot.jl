using ReactiveKernels, Test

isdefined(@__MODULE__, :rk_jax_available) || include(joinpath(@__DIR__, "jax_python.jl"))

# BRM's grouped HSGP term (`brm_grouped_hsgp`): each row reads its group's
# weighted coefficients and takes a row-wise product with the basis.
@kernel function jax_grouped_rowdot(z::Matrix{Float64}, s::Matrix{Float64}, PHI::Matrix{Float64},
                                    g::Vector{Int}, w::Vector{Float64})
    scaled = z .* s
    rows = scaled[g, :]
    value = vec(sum(PHI .* rows; dims=2))
    total = sum(w .* value)
    return total
end

# The same product with its operands swapped and without `vec`.
@kernel function jax_grouped_rowsum(z::Matrix{Float64}, PHI::Matrix{Float64}, g::Vector{Int},
                                    w::Vector{Float64})
    rows = z[g, :]
    value = sum(rows .* PHI; dims=2)
    total = sum(w .* vec(value))
    return total
end

# d total / d z for `total = sum(w .* vec(sum(PHI .* (z .* s)[g, :]; dims=2)))`.
function grouped_rowdot_gradient(z, s, PHI, g, w)
    grad = zero(z)
    for (r, k) in enumerate(g)
        grad[k, :] .+= w[r] .* PHI[r, :] .* s[k, :]
    end
    grad
end

function grouped_rowdot_cases()
    K, G = 5, 4
    cases = []
    unsorted = repeat(1:G, inner = 18)[sortperm(sin.(1:G*18))]
    # Groups averaging at least 16 rows are blocked (the first three); smaller
    # ones keep the literal gather.
    for g in (reduce(vcat, fill.(1:G, [18, 3, 40, 9])),   # sorted runs, padded blocks
              unsorted,                                   # unsorted
              fill(2, 37),                                # one group
              [3, 1, 4, 1, 2, 3, 3, 1, 4, 4, 2, 3],       # small groups
              [2, 4, 1, 3],                               # one row each
              Int[])                                      # no rows
        R = length(g)
        PHI = reshape(sin.(1:R*K) .+ 0.1, R, K)
        w = cos.(1:R)
        s = reshape(0.5 .+ 0.1 .* (1:G*K), G, K)
        z = reshape(0.3 .* cos.(1:G*K), G, K)
        push!(cases, (; z, s, PHI, g, w))
    end
    cases
end

@testset "Row products with rows gathered by a prepared index read each group's row once" begin
    have = (:z, :s, :PHI, :g, :w)
    native = prepare(jax_grouped_rowdot; have, want = :total)
    cases = grouped_rowdot_cases()
    region = ReactiveKernels._JaxSourceRegion(:grouped_term, [:z, :s, :PHI, :g], [:value, :rows],
        Dict(:value => :value, :rows => :rows), "A grouped term.")
    for regions in (ReactiveKernels._JaxSourceRegion[], [region]), layout in (:default, :padded, :chains)
        options = layout === :default ? (;) : (; plate_layout = layout)
        src = jax_source(jax_grouped_rowdot; have, want = :total, data = have[2:end],
                         source_regions = regions, options...)
        @test occursin("_rk_grouped_rowdot(", src.text)
        @test occursin("_rk_row_groups(", src.text)
        isempty(regions) || @test occursin("def grouped_term", src.text)
        rk_jax_available() || continue
        for c in cases
            args = (c.z, c.s, c.PHI, c.g, c.w)
            result = rk_jax_evaluate(src, :kernel, args; grad = true, prepare = :arguments)
            @test result.value ≈ native(args...) atol = 1e-12
            @test reduce(hcat, result.grad)' ≈ grouped_rowdot_gradient(args...) atol = 1e-12
        end
    end
end

@testset "A row sum with the gathered rows first keeps its matrix shape" begin
    have = (:z, :PHI, :g, :w)
    native = prepare(jax_grouped_rowsum; have, want = :total)
    src = jax_source(jax_grouped_rowsum; have, want = :total, data = have[2:end])
    @test occursin("_rk_grouped_rowdot(", src.text)
    if rk_jax_available()
        for c in grouped_rowdot_cases()
            args = (c.z, c.PHI, c.g, c.w)
            result = rk_jax_evaluate(src, :kernel, args; grad = true, prepare = :arguments)
            @test result.value ≈ native(args...) atol = 1e-12
            @test reduce(hcat, result.grad)' ≈
                  grouped_rowdot_gradient(c.z, one.(c.s), c.PHI, c.g, c.w) atol = 1e-12
        end
    end
end

@testset "An index that is not prepared data keeps the literal gather" begin
    have = (:z, :s, :PHI, :g, :w)
    src = jax_source(jax_grouped_rowdot; have, want = :total, data = (:s, :PHI, :w))
    @test !occursin("_rk_grouped_rowdot(", src.text)
end

# `_rk_row_groups` blocks an index whose groups average at least 16 rows (here
# exactly 16), and `_rk_grouped_rowdot` contracts those blocks. Smaller groups
# and an empty index get no layout, so the functions receive no extra arrays,
# and take the literal gather. All agree with the literal product.
const GROUPED_ROWDOT_PATHS = raw"""
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
rng = np.random.default_rng(0)
out = []
for index in (np.repeat([3, 1, 2], [20, 16, 12]), np.array([3, 1, 2, 2, 1]), np.zeros(0, dtype=np.int64)):
    layout = _rk_row_groups(index)
    a = rng.standard_normal((len(index), 4))
    table = rng.standard_normal((3, 4))
    f = lambda t: _rk_grouped_rowdot(jnp.asarray(a), t, jnp.asarray(index), layout)
    blocked = "dot_general" in str(jax.make_jaxpr(f)(table))
    literal = np.sum(a * table[index - 1, :], axis=1)
    out.append((len(jax.tree_util.tree_leaves(layout)), blocked,
                bool(np.allclose(f(table), literal, rtol=0, atol=1e-12))))
print(out)
"""

@testset "Only groups averaging 16 rows or more are read in blocks" begin
    if rk_jax_available()
        script = ReactiveKernels._JAX_ROW_GROUPS * ReactiveKernels._JAX_GROUPED_ROWDOT *
                 GROUPED_ROWDOT_PATHS
        @test strip(read(`$RK_JAX_PYTHON -c $script`, String)) ==
              "[(3, True, True), (0, False, True), (0, False, True)]"
    else
        @test_skip "needs RK_JAX_PYTHON"
    end
end
