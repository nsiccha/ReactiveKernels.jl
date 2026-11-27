# Run in an environment with Pluto, ReactiveKernels and ReactiveKernelsPPL.
# Headless notebook acceptance; no HTTP server or browser is required.
# julia --startup-file=no --project=<env> benchmark/pluto_reactivity.jl
using Pluto, Test
using Pluto: Cell, Notebook, ServerSession, update_run!, WorkspaceManager

function check_pluto_reactivity(; distributed = true)
    original_dir = pwd()
    session = ServerSession()
    session.options.evaluation.workspace_use_distributed = distributed
    session.options.server.disable_writing_notebook_files = true
    activate = """
    begin
        import Pkg
        Pkg.activate($(repr(dirname(Base.active_project()))))
        using ReactiveKernels, ReactiveKernelsPPL
    end
    """
    model = """
    model = @rkppl begin
        a ~ Normal(0, 1)
        mu = location(a, xs)
        ys .~ Normal.(mu, 1)
    end
    """
    nb = Notebook(Cell.([
        activate,
        "location(a, x) = a .+ x",
        "begin\nxs = [-1.0, 0.0, 2.0]\nys = [0.2, -0.4, 1.1]\nend",
        model,
        "bound = model(; xs) | (; ys)",
        "built = build_kernel(bound)",
        "density = prepare_query(built, bound, :sampler)",
        "density_value = density(zeros(built.layout.total))",
        "reference = -0.5log(2pi) + sum(-0.5log(2pi)-(y-m)^2/2 for (y,m) in zip(ys,location(0.0,xs)))",
        "@kernel squareplus(x) = begin\ny = x*x + 1\nend",
        "simple = prepare(squareplus; have=(:x,), want=(:y,))",
        "@assert simple(3.0) == 10.0",
        "@rkppl predictor(a, x) = begin\nmu = location(a, x)\nreturn mu\nend",
        "module Helpers\nlocation(a, x) = a .+ 4 .* x\nend",
    ]))
    function phase(changed, label)
        println("PHASE ", label)
        flush(stdout)
        update_run!(session, nb, nb.cells[changed])
        for (i, c) in enumerate(nb.cells)
            c.errored && println("CELL_ERROR ", i, " ", c.output.body)
        end
        @test all(c -> !c.errored, nb.cells)
        actual = parse(Float64, nb.cells[8].output.body)
        expected = parse(Float64, nb.cells[9].output.body)
        @test actual ≈ expected
        println("DENSITY ", actual, " REFERENCE ", expected)
        flush(stdout)
        actual
    end
    try
        @testset "Pluto notebook reactivity" begin
            initial = phase(eachindex(nb.cells), "initial")
            # Only the helper cell is submitted. Pluto must discover and
            # rerun the model and all downstream construction/query cells.
            nb.cells[2].code = "location(a, x) = a .+ 2 .* x"
            edited = phase([2], "helper edit")
            @test edited != initial

            nb.cells[3].code = "begin\nxs = [-1.0, 0.0, 2.0, 3.0]\nys = [0.7, -0.2, 0.8, 1.5]\nend"
            phase([3], "data size and values edit")
            nb.cells[10].code = "@kernel squareplus(x) = begin\ny = x*x + 2\nend"
            nb.cells[12].code = "@assert simple(3.0) == 11.0"
            phase([10, 12], "kernel edit")

            nb.cells[4].code = replace(model, "mu = location(a, xs)" => "mu ~ predictor(a, xs)")
            phase([4], "submodel")
            nb.cells[2].code = "location(a, x) = a .+ 3 .* x"
            phase([2], "helper edit through submodel")

            nb.cells[4].code = replace(model, "location(a, xs)" => "Helpers.location.(a, xs)")
            nb.cells[9].code = replace(nb.cells[9].code, "location(0.0,xs)" => "Helpers.location.(0.0,xs)")
            phase([4, 9], "qualified broadcast helper")
            nb.cells[14].code = "module Helpers\nlocation(a, x) = a .+ 5 .* x\nend"
            phase([14], "qualified helper edit")
        end
    finally
        WorkspaceManager.unmake_workspace((session, nb); verbose = false)
        cd(original_dir)
    end
    println("PLUTO_REACTIVITY_PASS Julia=", VERSION, " Pluto=", pkgversion(Pluto))
end

if abspath(PROGRAM_FILE) == @__FILE__
    check_pluto_reactivity()
end
