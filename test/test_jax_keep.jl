module JAXKeepTests
using ReactiveKernels, Test
include("jax_python.jl")

@kernel metadata(u::Float64, x::Vector{Float64}) = begin
    @recipe (cse_key = :extent) n::Int = length(x)
    @recipe (cse_key = :extent) alias::Int = length(x)
    total = u * u
end

@testset "JAX retains a data-only cut beyond the function's outputs" begin
    @test canon_id(kernel_graph(metadata), metadata[:n].id) ==
          canon_id(kernel_graph(metadata), metadata[:alias].id)
    sources = [jax_source(metadata; want = :total, data = (; x), keep = (:n, :alias))
               for x in (Float64[], [1.0, 2.0], collect(1.0:20.0))]
    @test all(s -> haskey(s.prepared, :n), sources)
    @test all(s -> s.prepared[:n] == s.prepared[:alias], sources)
    @test all(s -> s.text == first(sources).text, sources)
    @test [f.name for f in first(sources).functions] == ["prepare_data", "kernel"]
    # The same operation through the Plan entry point retains the same metadata.
    planned = jax_source(plan(metadata; want = :total); data = (; x = [1.0]), keep = (:n, :alias))
    @test planned.prepared == first(sources).prepared
    src = first(sources)
    @test_throws ArgumentError jax_source(metadata; want = :total, data = (; x = [1.0]), keep = (:u,))
    if rk_jax_available()
        for x in (Float64[], [1.0, 2.0], collect(1.0:20.0))
            result = rk_jax_evaluate(src, :kernel, (0.4, x); grad = true, prepare = :arguments)
            @test result.value ≈ 0.16
            @test result.grad ≈ 0.8
        end
    else
        @test_skip "RK_JAX_PYTHON not set"
    end
end

@testset "retained metadata selects the data route of an inverse graph" begin
    g = Graph()
    x = value!(g, :x, Vector{Float64})
    u = value!(g, :u, Vector{Float64})
    n = value!(g, :n, Int)
    add!(g, u => n, length; cost = 1.0)
    add!(g, x => n, length; cost = 2.0)
    p = plan(g; have = (u, x), want = (n,))
    @test only(p.recipes).inputs == (u,)
    src = jax_source(p; data = (; x = ones(3)), keep = (:n,))
    @test haskey(src.prepared, :n)
    if rk_jax_available()
        @test rk_jax_evaluate(src, :kernel, (ones(3), ones(3)); prepare = :arguments).value == [3]
    else
        @test_skip "RK_JAX_PYTHON not set"
    end
end

function unavailable end
struct OpaqueMetadata
    n::Int
end
@kernel opaque_metadata(u::Float64, metadata::OpaqueMetadata) = begin
    n = metadata.n
    total = u * u
end

@testset "retained metadata reports its missing host operation" begin
    error = try
        jax_source(opaque_metadata; want = :total, data = (; metadata = OpaqueMetadata(3)), keep = (:n,))
        nothing
    catch e
        e
    end
    @test error isa JAXEmissionError
    @test occursin("retained data value `n`", sprint(showerror, error))
    @test occursin("field read", sprint(showerror, error))
    @test occursin("OpaqueMetadata", sprint(showerror, error))
end

@kernel missing_method(x::Vector{Float64}) = begin
    y = unavailable(x)
    out = plate(x, y) do a, b
        a + b
    end
end

@testset "a missing Julia method is diagnosed at the call" begin
    error = try
        jax_source(missing_method; want = :out)
        nothing
    catch e
        e
    end
    @test error isa JAXEmissionError
    @test occursin("unavailable", sprint(showerror, error))
    @test occursin("no applicable Julia method", sprint(showerror, error))
    @test !ReactiveKernels._jax_numeric_tuple(Union{})
    @test !ReactiveKernels._jax_numeric_tuple(Tuple{Vararg{Float64}})
    @test ReactiveKernels._jax_numeric_tuple(Tuple{Float64,Float64})
end
end
