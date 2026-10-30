import Pkg
using TOML

@testset "package boundary — generic RK core, no iterative-method compiler" begin
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

    # The iterative-method compiler and the sampler examples live in a
    # separate package that depends on ReactiveKernels; ReactiveKernels
    # defines none of its source files or names.
    for file in ("binders.jl", "kernel_adaptation.jl", "kernel_codegen.jl",
                 "kernel_control.jl", "kernel_execution.jl", "kernel_factory.jl",
                 "kernel_lowering.jl", "kernel_methodir.jl",
                 "kernel_structural_container.jl", "native_slot_compiler.jl",
                 "native_slots.jl", "native_slots_factory.jl", "transpiled_program.jl",
                 "kernel_nuts.jl", "kernel_nuts_native.jl", "hmc.jl", "reactive_nuts.jl")
        @test !isfile(joinpath(srcdir, file))
        @test !occursin("include(\"$file\")", module_source)
    end

    for name in (
        :compile_stateful, :stateful_compiler_bindings, :functionalize_stateful,
        :stateful_snapshot, :compile_state_transition, :initial_transition_state,
        :drain_observations!, :rng_provider, :OrderedRNGReplay,
        :prepare_transpiled, :transpiled_endpoint, :partial, :PartialFunction,
        :NativeSlotCompiler, :MethodIR,
        :ReactivePhasePoint, :reactive_nuts_group, :CompiledNUTSState,
        :NUTSDiagnostics, :nuts_state, :step!, :warmup!, :welford_var,
        :compile_leapfrog, :compile_nuts, :compile_nuts_native, :_NutsFrame,
    )
        @test !isdefined(ReactiveKernels, name)
        @test name ∉ names(ReactiveKernels)
    end

    # Its front end too: stateful free methods, the object/view skeletons, the frozen
    # method and recipe source, the definition-time callee and type snapshots, the
    # primitive effect registry and `copy!!`. Method-bearing objects keep only what
    # their pure endpoints read.
    for name in (
        Symbol("copy!!"), :KernelObject, :KernelView, :_Mode2KernelSkeleton,
        :_kernel_mode2_expand, :_PrimitiveEffect, :_KernelIntrinsic,
        :_KernelRegistration, :kernel_registration, :kernel_rebound,
        :kernel_callee_registration, :kernel_callee_registrations,
        :_CapturedCalleeRef, :_CapturedTypeAuthority, :kernel_type_authorities,
        :kernel_methods, :kernel_recipe_ast, :kernel_module, :kernel_token,
        :_kernel_freeze_ast, :_kernel_thaw_ast,
    )
        @test !isdefined(ReactiveKernels, name)
        @test name ∉ names(ReactiveKernels)
    end
    @test fieldnames(ReactiveKernels._StatefulKernelSkeleton) ==
          (:name, :spec_snapshot, :endpoint_templates)

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
    @test :reactive_program in names(ReactiveKernels)
end
