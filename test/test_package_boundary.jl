using TOML

@testset "package boundary — generic RK core, external NUTS exemplar" begin
    srcdir = joinpath(pkgdir(ReactiveKernels), "src")
    module_source = read(joinpath(srcdir, "ReactiveKernels.jl"), String)
    project = TOML.parsefile(joinpath(pkgdir(ReactiveKernels), "Project.toml"))

    for optional_ad in ("DifferentiationInterface", "Enzyme")
        @test !haskey(project["deps"], optional_ad)
        @test haskey(project["extras"], optional_ad)
        @test optional_ad in project["targets"]["test"]
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
    @test :partial in names(ReactiveKernels)
    @test :reactive_program in names(ReactiveKernels)

    @test isfile(joinpath(pkgdir(ReactiveKernels), "examples", "nuts_runtime.jl"))
    for file in ("kernel_factory.jl", "kernel_codegen.jl", "kernel_nuts.jl",
                 "kernel_nuts_native.jl", "hmc.jl", "reactive_nuts.jl")
        @test isfile(joinpath(pkgdir(ReactiveKernels), "examples", "nuts_runtime", file))
    end
end
