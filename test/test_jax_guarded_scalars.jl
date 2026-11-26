module JAXGuardedScalarTests
using ReactiveKernels, Test
include("jax_python.jl")

@kernel root_graph(a::Float64, b::Float64) = begin
    value::Float64 = a + sqrt(b)
    da::Int = 1
    db::Float64 = 0.5 / sqrt(b)
    return value, da, db
end
const root = scalar_derivative_rule(root_graph; primal=:value, partials=(a=:da, b=:db))
@kernel offset(a::Float64, b::Float64) = begin
    norm::Float64 = b == Inf ? 0.0 : root(a, b / 2)
    score(x::Float64)::Float64 = norm + x
end
@kernel mapped(a::Float64, b::Float64, xs::Vector{Float64}) = begin
    terms = plate(xs, Ref(a), Ref(b)) do x, t, u
        (x > 0) & (u >= 0) ? offset(t,u).score(x) : 0.0
    end
    total = sum(terms)
end
@kernel ragged(a::Float64, b::Float64, rows::Vector{Vector{Float64}}) = begin
    totals = plate(rows, Ref(a), Ref(b)) do row, t, u
        terms = plate(row, Ref(t), Ref(u)) do x, v, w
            (x > 0) & (w >= 0) ? offset(v,w).score(x) : 0.0
        end
        sum(terms)
    end
    total = sum(totals)
end
@kernel unsafe(a::Float64, xs::Vector{Float64}) = begin
    terms = plate(xs, Ref(a)) do x, t
        (x > 0) & (t > 0) ? root(x, 1 / t) : 0.0
    end
    total = sum(terms)
end

@testset "guarded shared scalar rules" begin
    src = jax_source(mapped; have=(:a,:b,:xs), want=:total)
    @test occursin("_rk_scalar_demand(", string(src))
    @test !occursin("_rk_scalar_demand(", string(jax_source(unsafe; want=:total)))
    if rk_jax_available()
        native = prepare(ragged; have=(:a,:b,:rows), want=:total)
        rowsource = jax_source(ragged; have=(:a,:b,:rows), want=:total)
        for rows in ([Float64[], [1.,-2.,3.]], [Float64[], Float64[]], Vector{Float64}[])
            got = rk_jax_evaluate(rowsource, "kernel", (1.0, 2.0, rows); grad=true)
            @test got.value ≈ native(1.0, 2.0, rows)
            @test got.grad == sum(row -> count(>(0), row), rows; init=0)
        end
        mktempdir() do dir
            path = joinpath(dir, "mapped.py")
            write(path, src)
            driver = joinpath(@__DIR__, "jax_guarded_scalars.py")
            @test success(pipeline(`$RK_JAX_PYTHON $driver $path`; stdout, stderr))
        end
    else
        @test_skip false
    end
end
end
