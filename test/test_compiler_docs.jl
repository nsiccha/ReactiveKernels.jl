using Test

_compiler_docs_lf(text) = replace(text, "\r\n" => "\n", "\r" => "\n")

@testset "compiler capability page is navigated, explicit, and prose-first" begin
    root = joinpath(@__DIR__, "..")
    page_path = joinpath(root, "docs", "src", "compiler.md")
    make_path = joinpath(root, "docs", "make.jl")
    index_path = joinpath(root, "docs", "src", "index.md")
    readme_path = joinpath(root, "README.md")
    ad_path = joinpath(root, "docs", "src", "automatic-differentiation.md")

    @test isfile(page_path)
    @test isfile(ad_path)
    page = _compiler_docs_lf(read(page_path, String))
    make = _compiler_docs_lf(read(make_path, String))
    index = _compiler_docs_lf(read(index_path, String))
    readme = _compiler_docs_lf(read(readme_path, String))
    ad_docs = _compiler_docs_lf(read(ad_path, String))

    # The exported prepared value+in-place-gradient surface has one public
    # prose authority. It shows the kernel, the interaction, caller-owned
    # storage, and Constant rebinding in a build-executed example.
    for marker in (
            "@kernel objective",
            "ad_value_and_gradient!",
            "caller-owned",
            "returned_gradient === gradient_buffer",
            "Constant",
            "AutoEnzyme(; mode = Enzyme.Reverse)",
            "Main.BatchedExamples.BATCHED_AD_SOURCE",
            "test_batched_nonallocating.jl",
        )
        @test occursin(marker, ad_docs)
    end

    # No other public prose page carries backend/API guidance. Algorithmic
    # uses of the word "gradient" in sampler pages remain domain terminology,
    # while the evaluation-throughput page belongs to the top-level AD group.
    ad_pages = Set(("automatic-differentiation.md", "eval-throughput.md"))
    forbidden_ad_prose = (
        "DifferentiationInterface",
        "AutoEnzyme",
        "Enzyme",
        "prepare_ad",
        "ad_gradient",
        "ad_value_and_gradient!",
        "automatic differentiation",
        "reverse-mode",
    )
    docs_src = joinpath(root, "docs", "src")
    for path in readdir(docs_src; join = true)
        endswith(path, ".md") || continue
        basename(path) in ad_pages && continue
        prose = _compiler_docs_lf(read(path, String))
        for marker in forbidden_ad_prose
            @test !occursin(lowercase(marker), lowercase(prose))
        end
    end
    for marker in forbidden_ad_prose
        @test !occursin(lowercase(marker), lowercase(readme))
    end

    @test occursin("\"Compiler capability and limits\" => \"compiler.md\"", make)
    @test occursin("\"Automatic differentiation\" => [", make)
    @test occursin("\"Prepared gradients\" => \"automatic-differentiation.md\"", make)
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
        "### Functional stateful methods and nested state contracts",
        "### Prototype-derived finite structural containers",
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
        "`structured_state_port(compiled_transition)`",
        "`StatefulStateValue`",
        "`OrderedRNGReplay`",
        "`effect_lowering_port`",
        "`total_functional_lowering`",
        "`initial_transition_effects(transition)`",
        "`transition_with_effects(transition)`",
        "one global ordered event\ntape/cursor",
        "exact recursive state layout, owned-array axes,\ncanonical aliases",
        "It does\nnot mislabel that token as an ordinary source callback",
        "authored Float64 `randexp`\nresult",
        "explicit ULP\ndistances, not an `isapprox` tolerance",
        "introduces no `@fastmath`, `muladd`, `fma`, or\narithmetic reassociation",
        "cannot approach a\ncontrol threshold",
        "widened Float64 normal tape is rejected",
        "None of this lowering dispatches on sampler, geometry,\nmethod, or field names",
        "a bare `using ReactiveKernels` does not load or\nexport them",
        "former `@rk_pure`, `@rk_borrows`, and `@rk_rng` declarations have\nbeen removed",
        "`@node` is unrelated to this removal boundary",
        "NUTS, log-density, and\nPPL artifacts are external compilation examples and acceptance evidence",
        "not** a general Julia-recursion compiler proof",
        "`compile_state_transition(spec, transition, endpoint_args;",
        "Static loops are unrolled during\nlowering",
        "do not add geometry or integrator cases to compiler code",
        "fixed-capacity\nstructural vectors",
        "`Diagonal` and `Cholesky` wrappers",
        "one typed numeric\ncolumn per logical owned leaf group",
        "topology-aware read, write, copy, swap, and whole-container\nselect",
        "Output validation\nhappens before reconstruction",
        "`Base.abs` is admitted here only for one builtin\n`AbstractFloat` scalar",
        "`Base.div` requires two operands of the identical\nbuiltin non-`Bool` integer type",
        "not a general exported container API",
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
