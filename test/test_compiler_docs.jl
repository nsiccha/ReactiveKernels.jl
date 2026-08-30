using Test

_compiler_docs_lf(text) = replace(text, "\r\n" => "\n", "\r" => "\n")

@testset "compiler capability page is navigated, explicit, and prose-first" begin
    root = joinpath(@__DIR__, "..")
    page_path = joinpath(root, "docs", "src", "compiler.md")
    make_path = joinpath(root, "docs", "make.jl")
    index_path = joinpath(root, "docs", "src", "index.md")
    readme_path = joinpath(root, "README.md")
    distributions_path = joinpath(root, "docs", "src", "distributions.md")

    @test isfile(page_path)
    page = _compiler_docs_lf(read(page_path, String))
    make = _compiler_docs_lf(read(make_path, String))
    index = _compiler_docs_lf(read(index_path, String))
    readme = _compiler_docs_lf(read(readme_path, String))
    distributions = _compiler_docs_lf(read(distributions_path, String))

    # The exported prepared value+in-place-gradient surface must remain visible
    # at both public entry points, with its destination and Constant-rebinding
    # contract rather than only a symbol mention.
    for ad_docs in (readme, distributions)
        @test occursin("ad_value_and_gradient!", ad_docs)
        @test occursin("caller-owned", ad_docs)
        @test occursin("returned_gradient === gradient_buffer", ad_docs)
        @test occursin("Constant", ad_docs)
        @test occursin("AutoEnzyme(; mode = Enzyme.Reverse)", ad_docs)
    end

    @test occursin("\"Compiler capability and limits\" => \"compiler.md\"", make)
    @test occursin("compiler.md", index)
    @test occursin("warnonly = false", make)

    nuts = _compiler_docs_lf(read(joinpath(root, "docs", "src", "nuts.md"), String))
    @test occursin("not an RK package API", nuts)
    @test occursin("external NUTS compiler-acceptance exemplar", nuts)
    @test occursin("The former `@rk_pure`,", nuts)
    @test occursin("declarations have been removed", nuts)
    @test occursin("The build-loaded source below is the macro-free executable fixture", nuts)

    # Keep every public entry point aligned with the executable Reactant
    # acceptance boundary instead of reviving the obsolete CPU-only claim.
    for public_source in (readme, nuts, index)
        public_page = replace(public_source, r"(?m)^>\s?" => "")
        public_page = replace(public_page, r"\s+" => " ")
        for claim in (
            "optional external",
            "one full-depth transition",
            "one data-dependent traced `while`",
            "pre-generated momentum, direction, and exponential tensors",
            "no host RNG inside the trace",
            "`Float64`",
            "positive diagonal Euclidean metric",
            "locked authored control-flow graph",
            "current diagnostics callback",
            "Overflow and unsupported cases reject",
            "native adaptive API remains CPU execution",
        )
            @test occursin(claim, public_page)
        end
    end
    @test !occursin("adaptive NUTS remains CPU-only", readme)
    @test !occursin("Adaptive NUTS is currently a CPU sampler", nuts)
    @test !occursin("stays on the CPU because", index)
    @test occursin("examples/nuts_runtime/kernel_nuts_reactant.jl", nuts)
    @test occursin("test/test_kernel_nuts_reactant.jl", nuts)
    @test occursin("### Measured Reactant performance", nuts)
    @test occursin("render_nuts_reactant_benchmark()", nuts)
    @test occursin("benchmark/nuts_reactant_comparison.jl", nuts)
    @test occursin("benchmark/receipts/nuts-reactant-v1.toml", nuts)
    @test occursin("same authored adaptive transition", lowercase(nuts))
    @test occursin("matched-control compiler/runtime microbenchmark", nuts)
    @test occursin("atol = 128eps(Float64)", nuts)
    @test occursin("control counters and random consumption", nuts)
    @test occursin("one synchronous compiled call per transition", nuts)
    @test occursin("could amortize dispatch", nuts)
    @test occursin("not adaptation", nuts)

    for heading in (
        "## The stateless compiler",
        "## Batch and replica lowering",
        "## Incremental and compiled reactive execution",
        "## Source-captured method compiler",
        "## What the NUTS proof does and does not establish",
        "## Definitive support matrix",
    )
        @test occursin(heading, page)
    end

    for contract in (
        "branch-and-bound",
        "availability-based Kahn ordering",
        "Structural CSE is explicit and conservative",
        "does not call\n`code_lowered`",
        "Validity changes are exception-safe but values are not transactionally rolled\nback",
        "external compilation examples and acceptance evidence",
        "a bare `using ReactiveKernels` does not load or\nexport them",
        "former `@rk_pure`, `@rk_borrows`, and `@rk_rng` declarations have\nbeen removed",
        "`@node` is unrelated to this removal boundary",
        "NUTS, log-density, and\nPPL artifacts are external compilation examples and acceptance evidence",
        "not** a general Julia-recursion compiler proof",
        "`compile_state_transition(spec, transition, endpoint_args;",
        "Static loops are unrolled during\nlowering",
        "do not add geometry or integrator cases to compiler code",
    )
        @test occursin(contract, page)
    end

    # The compiler specification is deliberately algorithmic prose plus three
    # build-executed result/API panels. Source-code examples remain on the
    # focused example pages.
    @test count(==("```@eval"), split(page, '\n')) == 3
    @test count(==("```"), split(page, '\n')) == 3
    @test !occursin("```julia", page)
    @test !occursin("~~~", page)
end
