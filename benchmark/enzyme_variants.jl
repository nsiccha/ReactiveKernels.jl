#!/usr/bin/env julia

# Registered-versus-dogfood Enzyme runs for a benchmark script that executes in
# a project environment instead of building its own. Each variant derives a
# fresh environment from that project and manifest, develops its local sources
# with ReactiveKernels' exact source dependencies, selects the variant's
# Enzyme, and runs the unchanged script with the same arguments. See
# `_enzyme_variants.jl` for the variants, recorded provenance and outputs.
#
#   julia --startup-file=no benchmark/enzyme_variants.jl \
#       --project=benchmark/distributions benchmark/distribution_gradients.jl \
#       --output=<path.toml>
#
# The derived environments are not precompiled as a whole: the script compiles
# only the packages it loads, so unrelated environment members (for example
# Reactant in the shared distributions project) are neither loaded nor built.

import Pkg
import TOML

include(joinpath(@__DIR__, "_enzyme_variants.jl"))
using .EnzymeVariants

# Relative local paths in the project and manifest are relative to `template`.
function _derive_environment!(environment, template)
    absolute(path) = isabspath(path) ? path : normpath(joinpath(template, path))
    project = TOML.parsefile(joinpath(template, "Project.toml"))
    local_paths = String[]
    for source in values(get(project, "sources", Dict{String,Any}()))
        haskey(source, "path") || continue
        source["path"] = absolute(source["path"])
        push!(local_paths, source["path"])
    end
    open(io -> TOML.print(io, project; sorted = true), joinpath(environment, "Project.toml"), "w")
    manifest_path = joinpath(template, "Manifest.toml")
    if isfile(manifest_path)
        manifest = TOML.parsefile(manifest_path)
        for entries in values(manifest["deps"]), entry in entries
            haskey(entry, "path") || continue
            entry["path"] = absolute(entry["path"])
            push!(local_paths, entry["path"])
        end
        open(io -> TOML.print(io, manifest; sorted = true), joinpath(environment, "Manifest.toml"), "w")
    end
    unique!(realpath.(local_paths))
end

function main(arguments)
    index = findfirst(argument -> startswith(argument, "--project="), arguments)
    index == 1 && length(arguments) >= 2 || error(
        "usage: julia benchmark/enzyme_variants.jl --project=<dir> <script.jl> [script arguments...]")
    template = abspath(split(arguments[1], '='; limit = 2)[2])
    isfile(joinpath(template, "Project.toml")) || error("no Project.toml in $template")
    script = abspath(arguments[2])
    script_arguments = arguments[3:end]
    root = normpath(joinpath(@__DIR__, ".."))
    run_enzyme_variants(; suite = splitext(basename(script))[1], root,
            output = requested_output(script_arguments), precompile = false) do environment, variant
        withenv("JULIA_PKG_PRECOMPILE_AUTO" => "0") do
            local_paths = _derive_environment!(environment, template)
            Pkg.develop([[Pkg.PackageSpec(path = path) for path in local_paths];
                         rk_source_dependencies()])
            select_enzyme!(variant)
            Pkg.instantiate()
        end
        `$(Base.julia_cmd()) --startup-file=no --project=$environment $script $(variant_arguments(script_arguments, variant))`
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
