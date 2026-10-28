using Pkg
using TOML
include(joinpath(@__DIR__, "..", "deps", "source_dependencies.jl"))

# Pkg.test installs every test extra, including Reactant. Build the native
# acceptance environment from the same dependency UUIDs and compat bounds,
# then run the ordinary test/runtests.jl with RKPPL_TEST_BACKENDS=native.
length(ARGS) == 1 || error("usage: julia packages/setup_ppl_native.jl <environment>")
root = normpath(joinpath(@__DIR__, ".."))
ppl = joinpath(@__DIR__, "ReactiveKernelsPPL")
project = TOML.parsefile(joinpath(ppl, "Project.toml"))
deps = copy(project["deps"])
for name in project["targets"]["test"]
    name == "Reactant" && continue
    deps[name] = project["extras"][name]
end
deps[project["name"]] = project["uuid"]
compat = Dict(name => bound for (name, bound) in project["compat"]
    if name == "julia" || haskey(deps, name))
env = abspath(only(ARGS))
mkpath(env)
open(joinpath(env, "Project.toml"), "w") do io
    TOML.print(io, Dict("deps" => deps, "compat" => compat))
end
Pkg.activate(env)
Pkg.develop([PackageSpec[
    PackageSpec(path = root),
    PackageSpec(path = joinpath(@__DIR__, "ReactiveKernelsDistributionKernels")),
    PackageSpec(path = ppl),
]; rk_source_dependencies()])
Pkg.instantiate()
Pkg.precompile()
println("Native PPL acceptance environment: Julia ", VERSION)
Pkg.status()
