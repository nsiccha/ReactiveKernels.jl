module PPLJaxWorldAgeTests
using ReactiveKernels, ReactiveKernelsPPL, Test

# `jax_source(built, plan)` translates a program whose functions are newer
# than its caller's world, as a driver that builds a program and translates it
# in one function call has (snag jax-source-world-4606dd56). Before, the
# data-only definition `xs = fresh_scale.(x)` stayed a `prepare_data` input
# with "the function `fresh_scale` has no applicable Julia method for argument
# types (Float64)" in `UNTRANSLATED`, while a translation from the latest
# world computed it.
function fresh_program()
    namespace = Module(gensym(:PPLJaxWorldAge))
    Core.eval(namespace, :(using ReactiveKernelsPPL))
    Core.eval(namespace, :(fresh_scale(x) = (x - 2.5) / 1.2))
    plan = Core.eval(namespace, :((@rkppl begin
        xs = fresh_scale.(x)
        mu ~ Normal(0, 1)
        y .~ Normal.(mu .+ xs, 1.0)
    end)() | (; x = [1.0, 2.0, 3.0, 4.0], y = [0.1, 0.3, 0.2, 0.5])))
    built = build_kernel(plan)
    (; plan, built, src = jax_source(built, plan))
end

@testset "jax_source(built, plan) translates a program newer than the caller's world" begin
    fresh = fresh_program()
    text = string(fresh.src)
    @test occursin("UNTRANSLATED = {}", text)
    @test !occursin("no applicable Julia method", text)
    @test text == string(jax_source(fresh.built, fresh.plan))
end
end
