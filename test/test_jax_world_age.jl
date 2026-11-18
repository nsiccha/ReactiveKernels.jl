module JaxWorldAgeTests
using ReactiveKernels, Test

# `jax_source` and `js_source` read a program's helpers, `@traceable` records
# and kernels in the latest world, as construction and `prepare` read a fresh
# program from any world (`test_source_world_age.jl`): a driver that builds a
# program and translates it in one function call needs no `invokelatest`.
# Before, `jax_source` there read a helper newer than the caller's world as a
# function with no method, "the function `fresh_scale` has no applicable
# Julia method for argument types (Float64)" (snag
# jax-source-world-4606dd56).

# Each call defines its helpers in a new module, after the calling function's
# world began.
function fresh_namespace()
    namespace = Module(gensym(:JaxWorldAge))
    Core.eval(namespace, :(using ReactiveKernels))
    Core.eval(namespace, :(fresh_scale(x) = (x - 2.5) / 1.2))
    Core.eval(namespace, :(@traceable function fresh_cumsum(x::AbstractVector)
        out = similar(x, Float64)
        acc = 0.0
        for i in eachindex(x)
            acc += x[i]
            out[i] = acc
        end
        return out
    end))
    namespace
end

function translate_fresh()
    namespace = fresh_namespace()
    spec = Core.eval(namespace, :(@kernel fresh_kernel(x::Float64, v::Vector{Float64}) = begin
        y = fresh_scale(x) + sum(fresh_cumsum(v))
        return y
    end))
    scalar = Core.eval(namespace, :(@kernel fresh_scalar(x::Float64) = begin
        y = fresh_scale(x)
        return y
    end))
    (; spec, scalar, jax = jax_source(spec; want = :y),
       planned = jax_source(plan(spec; want = :y)),
       js = ReactiveKernels.js_source(scalar; want = :y))
end

@testset "jax_source and js_source translate a program newer than the caller's world" begin
    fresh = translate_fresh()
    @test fresh.jax isa JAXSource
    @test occursin("def fresh_scale(", string(fresh.jax))
    @test occursin("def fresh_cumsum(", string(fresh.jax))
    @test occursin("fresh_scale", string(fresh.planned))
    @test occursin("fresh_scale", fresh.js)
    # The same modules as translated from the latest world.
    @test string(fresh.jax) == string(jax_source(fresh.spec; want = :y))
    @test string(fresh.planned) == string(jax_source(plan(fresh.spec; want = :y)))
    @test fresh.js == ReactiveKernels.js_source(fresh.scalar; want = :y)
end
end
