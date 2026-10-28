using Reactant, ReactiveKernels, ReactiveKernelsPPL, Test
using ._QualifiedSubmodelTests: _qs_ast, _qs_build, _qs_oracle, _qs_fd,
    _qs_cell_oracle, _QS_BACKEND, _QS_HEADS, _QS_STREAM_HEADS

function _qs_compiled_measure(kind, head, n;
        cell_location = kind === :cell_sum ?
            :(z[i] + z[i].nested.b) : :(z[i]),
        cell_scale = nothing, named = false)
    x = collect(range(-0.7, 0.9; length = n))
    data = (; x, y = sin.(x))
    saved = deepcopy(data)
    ast = _qs_ast(head; cell = kind in (:cell, :cell_sum),
        stream = kind === :stream,
        cell_location, named)
    bound, built, u = _qs_build(ast, data)
    reference(v) = cell_scale === nothing ?
        _qs_oracle(kind, constrain(built.layout, v), data) :
        _qs_cell_oracle(constrain(built.layout, v), data, cell_scale)
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
    ad = q.ad
    both(w) = ad_value_and_gradient(ad, w)
    ad_operations = _qs_operations(repr(Reactant.@code_hlo optimize = false both(ru)))
    optimized_ad_operations = _qs_operations(repr(Reactant.@code_hlo both(ru)))
    @test !isempty(operations)
    @test !isempty(optimized_operations)
    @test !isempty(ad_operations)
    @test !isempty(optimized_ad_operations)
    compiled = Reactant.@compile post(ru)
    @test Float64(compiled(ru)) ≈ reference(u) rtol = 1e-9
    cad = compile_ad_value_and_gradient(q.ad, ru)
    value, gradient = cad(ru)
    @test Float64(value) ≈ reference(u) rtol = 1e-9
    @test Array(gradient) ≈ _qs_fd(reference, u) rtol = 1e-5 atol = 1e-7
    return (; traced = operations, optimized = optimized_operations,
        ad_traced = ad_operations, ad_optimized = optimized_ad_operations)
end

function _qs_operations(hlo)
    operations = Dict{String,Int}()
    for m in eachmatch(r"\b(?:stablehlo|chlo|func|arith|enzyme)\.\w+", hlo)
        operations[m.match] = get(operations, m.match, 0) + 1
    end
    return operations
end

@testset "qualified submodels: compiled parity and retained structure" begin
    for kind in (:latent, :cell, :cell_sum, :stream)
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

@testset "cell submodels: compiled signed/repeated inline and named values" begin
    for (loc, scale) in ((:(z[i] - z[i].nested.b), -1),
            (:(z[i] + z[i].nested.b + z[i].nested.b), 2)),
            named in (false, true)
        counts = map((3, 8)) do n
            Base.invokelatest(_qs_compiled_measure, :cell, _QS_HEADS[2], n;
                cell_location = loc, cell_scale = scale, named)
        end
        @test first(counts) == last(counts)
    end
end
