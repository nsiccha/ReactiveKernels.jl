import Pkg
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

    # Enzyme_jll 0.0.302 (required by Enzyme 0.13.211) segfaults native
    # reverse gradients through functions that return freshly allocated
    # arrays, such as index gathers, on Julia 1.10
    # (EnzymeAD/Enzyme.jl#3776). Core declares Enzyme_jll only so this compat
    # entry constrains every environment that contains ReactiveKernels; the
    # binary stays out of `[deps]` and no extension loads it.
    @test haskey(project["weakdeps"], "Enzyme_jll")
    @test !haskey(project["deps"], "Enzyme_jll")
    @test all(trigger -> "Enzyme_jll" ∉ (trigger isa String ? [trigger] : trigger),
        values(project["extensions"]))
    enzyme_jll = Pkg.Types.semver_spec(project["compat"]["Enzyme_jll"])
    @test v"0.0.301" in enzyme_jll
    @test v"0.0.302" ∉ enzyme_jll
    @test v"0.0.303" in enzyme_jll

    # Core must not DEPEND on Enzyme (a test/extension-only AD backend), but
    # explanatory prose may name it: a comment documenting why core code
    # sidesteps an AD failure mode (e.g. `codegen.jl` narrowing a plate
    # accumulator's element type), or a docstring telling the caller which
    # backend package to load (e.g. `using Enzyme` for `AutoEnzyme` in
    # `ad.jl`). Parse Julia syntax so prose in docstrings, comments, and
    # diagnostic strings is ignored, while imports, qualified calls,
    # and executable string interpolations retain their symbol references.
    function references_enzyme(ex)
        ex === :Enzyme && return true
        ex isa Expr && return any(references_enzyme, ex.args)
        ex isa QuoteNode && return references_enzyme(ex.value)
        return false
    end
    references_enzyme_source(code) = references_enzyme(Meta.parseall(code))
    @test !references_enzyme_source("# Enzyme\nerror(\"Enzyme must be loaded\")")
    @test !references_enzyme_source("\"\"\"Load Enzyme first.\"\"\"\nf() = 1")
    @test references_enzyme_source("using Enzyme")
    @test references_enzyme_source("import Enzyme: Reverse")
    @test references_enzyme_source("f() = Enzyme.Reverse")
    @test references_enzyme_source(raw"f() = \"backend: $(Enzyme.Reverse)\"")
    dependency_paths = filter(readdir(srcdir; join = true)) do path
        isfile(path) && references_enzyme_source(read(path, String))
    end
    @test isempty(dependency_paths)

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
