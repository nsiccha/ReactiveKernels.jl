using Test
using ReactiveKernels

module _WalnutsCorpusTest
include(joinpath(@__DIR__, "..", "benchmark", "walnuts_external_corpus.jl"))
end

module _WalnutsFixtureTest
include(joinpath(@__DIR__, "..", "benchmark", "walnuts_kernel_authoring_fixture.jl"))
end

const WEC = _WalnutsCorpusTest.WalnutsExternalCorpus
const WFX = _WalnutsFixtureTest.WalnutsKernelAuthoringFixture

@testset "external WALNUTS-D source and corpus lock" begin
    @test WEC.UPSTREAM.revision ==
          "4f051db7df57762a58ac851b0274fe57de342198"
    @test only(WEC.UPSTREAM.source_sha256).second ==
          "ab00138be5f6dee2d67108cafa11d42e99da3018b29d52d29bf4a07c545bdab5"
    @test WEC.BOB_CARPENTER_LINEAGE.revision ==
          "895a9b7a595b1bf15e9bcd7267bf1fa4fc36789a"
    @test WEC.BOB_CARPENTER_LINEAGE.blamed_lines ==
          WEC.BOB_CARPENTER_LINEAGE.bob_carpenter_lines == 408
    @test WEC.ORACLE_TOOLCHAIN.eigen_repository ==
          "https://github.com/eigen-mirror/eigen.git"
    @test WEC.ORACLE_TOOLCHAIN.eigen_revision ==
          "bc3b39870ecb690a623a3f49149a358b95c5781d"

    entry = only(WEC.EXTERNAL_CORPUS)
    @test propertynames(entry) == WEC.ENTRY_FIELDS
    @test entry.id == :walnuts_d
    @test entry.minimum_acceptance == :native_and_reactant
    @test entry.oracle == :separate_process_pinned_cpp_replay
    @test entry.id ∉ map(e -> e.id,
        WEC.ReactiveHMCAlgorithmCorpus.CORPUS)
    @test all(path -> isfile(joinpath(@__DIR__, "..", path)),
              entry.current_reactive_sources)
    @test :accepted_endpoint_is_rejected_if_a_coarser_reverse_grid_passes in
          WEC.WALNUTS_D_INVARIANTS
    @test :full_max_depth_ten_has_no_product_cap in WEC.WALNUTS_D_INVARIANTS
end

function _walnuts_walk(f, x)
    f(x)
    if x isa Tuple || x isa AbstractVector || x isa NamedTuple
        foreach(y -> _walnuts_walk(f, y), x)
    elseif x isa Pair
        _walnuts_walk(f, x.second)
    elseif x isa ReactiveKernels._MExpr || x isa ReactiveKernels._MStmt
        foreach(n -> _walnuts_walk(f, getfield(x, n)), fieldnames(typeof(x)))
    end
end

@testset "mathematical WALNUTS-D @kernel MethodIR and generic gap" begin
    irs = ReactiveKernels.method_irs(WFX.walnuts_state)
    byname = Dict(ir.id.name => ir for ir in irs)
    @test length(irs) == 15
    @test all(ir -> ir.ok, irs)
    @test byname[:macro_step!].control == :loop
    @test byname[:integrate!].control == :loop
    @test byname[:start!].control == :recursive
    @test byname[:finish!].control == :recursive

    # The source is admitted into MethodIR without pretending the two missing
    # scalar operations are compiler-known.  These two opaque nodes are the
    # smallest reusable value-domain gap handed to the generic compiler owner:
    # abs(::builtin float) and div(::builtin Int, ::builtin Int).
    opaque = Symbol[]
    _walnuts_walk(byname[:macro_step!].body) do node
        if node isa ReactiveKernels._OpCall && node.hint === :opaque
            push!(opaque, node.op.name)
        end
    end
    @test sort!(unique(opaque)) == [:abs, :div]

    rec = ReactiveKernels.defunctionalized_mids(irs)
    @test Set(ir.id.name for ir in irs if ir.id.decl in rec) ==
          Set((:step!, :finish!, :start!))
    bymid = Dict(ir.id.decl => ir for ir in irs)
    cfg = ReactiveKernels.build_method(byname[:start!], bymid, rec)
    @test length(cfg.blks) == 36
    @test count(b -> b.term isa ReactiveKernels.TCall, cfg.blks) == 2

    root = only(ReactiveKernels.method_irs(WFX.walnuts!!))
    @test root.ok
    @test map(f -> (f.name, f.kind, f.required), root.formals) == (
        (:momentum, :kw, true),
        (:directions, :kw, true),
        (:exponentials, :kw, true),
    )
    source = read(joinpath(@__DIR__, "..", "benchmark",
                           "walnuts_kernel_authoring_fixture.jl"), String)
    @test !occursin("Random.rand", source)
    @test occursin("@kernel walnuts!!(state; momentum, directions, exponentials)",
                   source)
end

@testset "pinned upstream C++ macro-step receipt" begin
    path = joinpath(@__DIR__, "..", "benchmark", "receipts",
                    "walnuts-upstream-macro-v1.tsv")
    lines = readlines(path)
    @test "# execution=separate_cpp_process" in lines
    @test "# authority_revision=$(WEC.UPSTREAM.revision)" in lines
    @test "# authority_sha256=$(only(WEC.UPSTREAM.source_sha256).second)" in lines
    @test "# eigen_repository=$(WEC.ORACLE_TOOLCHAIN.eigen_repository)" in lines
    @test "# eigen_revision=$(WEC.ORACLE_TOOLCHAIN.eigen_revision)" in lines

    data = filter(line -> !startswith(line, '#'), lines)
    @test split(first(data), '\t') ==
          ["case", "accepted", "logp_grad_calls", "theta", "rho", "joint",
           "base_accept"]
    rows = Dict(begin
        cols = split(line, '\t')
        cols[1] => (accepted = cols[2] == "1",
                    calls = parse(Int, cols[3]),
                    values = parse.(Float64, cols[4:7]))
    end for line in data[2:end])

    @test rows["base_grid_accept"].accepted
    @test rows["base_grid_accept"].calls == 1
    @test rows["dyadic_reverse_accept"].accepted
    @test rows["dyadic_reverse_accept"].calls == 10 # 1+2+4 forward, 2+1 reverse
    @test !rows["reverse_grid_reject"].accepted
    @test rows["reverse_grid_reject"].calls == 4    # 1+2 forward, 1 reverse
    @test !rows["all_grids_reject"].accepted
    @test rows["all_grids_reject"].calls == 15      # 1+2+4+8 forward

    expected = Dict(
        "base_grid_accept" =>
            [1.0249999999999999, 0.19875000000000001,
             -0.54506328124999992, 0.99993672075221618],
        "dyadic_reverse_accept" =>
            [0.075936222076415927, -3.1054732918739321,
             -4.8508137323873521, 0.045331641611676264],
        "reverse_grid_reject" =>
            [-9.1381250000000005, -1.223390625,
             -4.9236087364501966, 0.33200259295192019],
        "all_grids_reject" =>
            [0.08240002202929364, -3.1504141277854654,
             -4.9965034064272675, 0.045331641611676264],
    )
    @test all(id -> rows[id].values == expected[id], keys(expected))
end
