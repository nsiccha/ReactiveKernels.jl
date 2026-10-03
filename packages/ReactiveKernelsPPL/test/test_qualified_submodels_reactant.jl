using Reactant, ReactiveKernels, ReactiveKernelsPPL, Test
using ._QualifiedSubmodelTests: _qs_ast, _qs_build, _qs_oracle, _qs_fd,
    _QS_BACKEND, _QS_HEADS, _QS_STREAM_HEADS

function _qs_compiled_measure(kind, head, n)
    x = collect(range(-0.7, 0.9; length = n))
    data = (; x, y = sin.(x))
    saved = deepcopy(data)
    ast = _qs_ast(head; cell = kind === :cell, stream = kind === :stream)
    bound, built, u = _qs_build(ast, data)
    reference(v) = _qs_oracle(kind, constrain(built.layout, v), data)
    q = prepare_sampler(built, bound, u; backend = _QS_BACKEND)
    ru = Reactant.to_rarray(u)
    operations = Base.invokelatest(_qs_compile_raw, q, ru, reference, u)
    @test data == saved
    return operations
end

function _qs_compile_raw(q, ru, reference, u)
    post = q.kernel
    hlo = repr(Reactant.@code_hlo optimize = false post(ru))
    optimized_hlo = repr(Reactant.@code_hlo post(ru))
    operations = _qs_operations(hlo)
    optimized_operations = _qs_operations(optimized_hlo)
    @test !isempty(operations)
    @test !isempty(optimized_operations)
    compiled = Reactant.@compile post(ru)
    @test Float64(compiled(ru)) ≈ reference(u) rtol = 1e-9
    cad = compile_ad_value_and_gradient(q.ad, ru)
    value, gradient = cad(ru)
    @test Float64(value) ≈ reference(u) rtol = 1e-9
    @test Array(gradient) ≈ _qs_fd(reference, u) rtol = 1e-5 atol = 1e-7
    return (; traced = operations, optimized = optimized_operations)
end

function _qs_operations(hlo)
    operations = Dict{String,Int}()
    for m in eachmatch(r"\b(?:stablehlo|chlo|func|arith|enzyme)\.\w+", hlo)
        operations[m.match] = get(operations, m.match, 0) + 1
    end
    return operations
end

@testset "qualified submodels: compiled parity and retained structure" begin
    for kind in (:latent, :cell, :stream)
        # The module path uses two module bindings; the alias uses the same
        # submodel object. All three execute the same expanded mathematics.
        heads = kind === :stream ? _QS_STREAM_HEADS :
            (_QS_HEADS[1], _QS_HEADS[2], _QS_HEADS[4])
        counts = NamedTuple[]
        for head in heads, n in (3, 8)
            push!(counts, Base.invokelatest(_qs_compiled_measure, kind, head, n))
        end
        # Per-cell Normal blocks vectorize. Both the batched trace and the
        # optimized array graph must keep the same operations as data grow.
        @test all(==(first(counts)), counts)
    end
end
