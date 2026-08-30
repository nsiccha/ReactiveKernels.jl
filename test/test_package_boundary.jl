using TOML

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
    @test all(readdir(srcdir; join = true)) do path
        !isfile(path) || !occursin(r"\bEnzyme\b", read(path, String))
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
        @test isfile(joinpath(pkgdir(ReactiveKernels), "examples", "nuts_runtime", file))
    end
end
