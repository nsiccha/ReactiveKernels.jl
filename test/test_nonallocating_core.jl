using ReactiveKernels
using Test

@testset "optional non-allocating preparation interface" begin
    @test Base.get_extension(ReactiveKernels,
                             :ReactiveKernelsMutatingFunctionsExt) === nothing

    g = Graph()
    x = value!(g, :x, Vector{Float64})
    y = value!(g, :y, Vector{Float64})
    add!(g, x => y, copy)
    p = plan(g; have = (x,), want = (y,))

    err = try
        prepare_nonallocating(p)
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("optional MutatingFunctions extension", sprint(showerror, err))
    @test occursin("using MutatingFunctions", sprint(showerror, err))
end
