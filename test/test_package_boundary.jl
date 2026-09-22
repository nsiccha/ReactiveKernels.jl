using TOML
import ReactiveKernelsNUTSExamples

@testset "package boundary — generic RK core, external NUTS exemplar" begin
    srcdir = joinpath(pkgdir(ReactiveKernels), "src")
    module_source = read(joinpath(srcdir, "ReactiveKernels.jl"), String)
    project = TOML.parsefile(joinpath(pkgdir(ReactiveKernels), "Project.toml"))

    @test haskey(project["deps"], "DifferentiationInterface")
    @test !haskey(project["extras"], "DifferentiationInterface")
    @test "DifferentiationInterface" ∉ project["targets"]["test"]

    @test !haskey(project["deps"], "Enzyme")
    @test haskey(project["extras"], "Enzyme")
    @test "Enzyme" in project["targets"]["test"]
    # Core must not DEPEND on Enzyme (a test/extension-only AD backend), but
    # explanatory prose may name it: a comment documenting why core code
    # sidesteps an AD failure mode (e.g. `codegen.jl` narrowing a plate
    # accumulator's element type), or a docstring telling the caller which
    # backend package to load (e.g. `using Enzyme` for `AutoEnzyme` in
    # `ad.jl`). Scan code with docstrings and comments stripped so a prose
    # mention is not mistaken for a dependency — a real
    # `using`/`import`/qualified reference in code still trips this, as does
    # the `deps` guard above. Docstrings strip first: no `"""` appears in a
    # src/ comment, and no `#` shares a line with a docstring delimiter, so
    # neither strip unbalances the other.
    strip_prose(code) = replace(
        replace(
            replace(code, r"\"\"\".*?\"\"\""s => " "),  # docstrings
            r"#=.*?=#"s => " "),  # block comments
        r"#[^\n]*" => "",  # line comments
    )
    @test all(readdir(srcdir; join = true)) do path
        !isfile(path) || !occursin(r"\bEnzyme\b", strip_prose(read(path, String)))
    end

    for file in ("kernel_nuts.jl", "kernel_nuts_native.jl", "hmc.jl", "reactive_nuts.jl")
        @test !isfile(joinpath(srcdir, file))
        @test !occursin("include(\"$file\")", module_source)
    end

    for name in (
        :ReactivePhasePoint, :reactive_nuts_group, :CompiledNUTSState,
        :NUTSDiagnostics, :nuts_state, :step!, :warmup!, :welford_var,
        :compile_leapfrog, :compile_nuts, :compile_nuts_native, :_NutsFrame,
    )
        @test !isdefined(ReactiveKernels, name)
        @test name ∉ names(ReactiveKernels)
    end

    for name in (Symbol("@rk_pure"), Symbol("@rk_borrows"), Symbol("@rk_rng"))
        @test !isdefined(ReactiveKernels, name)
        @test name ∉ names(ReactiveKernels)
    end

    @test isdefined(ReactiveKernels, Symbol("@node"))
    @test isdefined(ReactiveKernels, Symbol("@kernel"))
    retired_object_macro = Symbol("@", "reactive")
    @test !isdefined(ReactiveKernels, retired_object_macro)
    @test retired_object_macro ∉ names(ReactiveKernels, all = true)
    retired_object_type = Symbol("Reactive", "Object")
    @test !isdefined(ReactiveKernels, retired_object_type)
    @test retired_object_type ∉ names(ReactiveKernels, all = true)
    @test :partial in names(ReactiveKernels)
    @test :reactive_program in names(ReactiveKernels)

    @test isfile(joinpath(pkgdir(ReactiveKernels), "examples", "nuts_runtime.jl"))
    for file in ("kernel_factory.jl", "kernel_codegen.jl", "kernel_nuts.jl",
                 "kernel_nuts_native.jl", "hmc.jl", "reactive_nuts.jl")
        @test isfile(joinpath(pkgdir(ReactiveKernelsNUTSExamples), "src",
                              "nuts_runtime", file))
    end
end
