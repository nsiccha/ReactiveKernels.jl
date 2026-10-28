using ReactiveKernels, Reactant, Enzyme, Test
isdefined(@__MODULE__, :ErrorPolicyFixture) || include("test_error_policy.jl")

@testset "compiled visible throw stripping" begin
    F = ErrorPolicyFixture
    kernel = prepare(F.total; on_error = :ignore)
    gradient(X) = Enzyme.gradient(Enzyme.Reverse, kernel, X)
    maps, reverse_maps = [], []
    for N in (3, 8)
        X = [-Float64(i) for i in 1:N]
        rx = Reactant.to_rarray(X)
        hlo = repr(Reactant.@code_hlo optimize = false kernel(rx))
        @test count("stablehlo.while", hlo) == 1
        @test !occursin("throw", hlo)
        counts(text) = Dict(op => count(op, text) for op in
            unique(m.match for m in eachmatch(r"\b(?:stablehlo|chlo|func|arith|enzyme)\.\w+", text)))
        push!(maps, counts(hlo))
        compiled = Reactant.@compile kernel(rx)
        @test Float64(compiled(rx)) == sum(abs2, X)
        native_gradient = only(gradient(X))
        @test native_gradient == 2X
        reverse = repr(Reactant.@code_hlo optimize = :only_enzyme gradient(rx))
        push!(reverse_maps, counts(reverse))
        compiled_gradient = Reactant.@compile gradient(rx)
        @test Array(only(compiled_gradient(rx))) == native_gradient
        @test Array(rx) == X
    end
    @test maps[1] == maps[2]
    @test reverse_maps[1] == reverse_maps[2]
    for N in (0,)
        X = zeros(N)
        rx = Reactant.to_rarray(X)
        compiled = Reactant.@compile kernel(rx)
        @test Float64(compiled(rx)) == 0.0
    end
    direct = prepare(F.direct; on_error = :ignore)
    rx = Reactant.to_rarray(-2.0)
    compiled = Reactant.@compile direct(rx)
    @test Float64(compiled(rx)) == 4.0
    dotted = prepare(F.dotted; on_error = :ignore)
    rx = Reactant.to_rarray([-2.0, 3.0])
    compiled = Reactant.@compile dotted(rx)
    @test Float64(compiled(rx)) == 13.0
    nested = prepare(F.nested; on_error = :ignore)
    rx = Reactant.to_rarray([-3.0, 2.0])
    rm = Reactant.to_rarray(0.5; track_numbers = Number)
    compiled = Reactant.@compile nested(rx, rm)
    @test Float64(compiled(rx, rm)) == 14.5
    nested_gradient(X, mu) = Enzyme.gradient(Enzyme.Reverse, nested, X, mu)
    compiled_gradient = Reactant.@compile nested_gradient(rx, rm)
    result = compiled_gradient(rx, rm)
    @test Array(result[1]) == [-7.0, 3.0]
    @test Float64(result[2]) == 4.0
    converted = prepare(F.converted; on_error = :ignore)
    rx = Reactant.to_rarray(Float32[2, -1, 3])
    compiled = Reactant.@compile converted(rx)
    @test Float64(compiled(rx)) == 4.0
end
