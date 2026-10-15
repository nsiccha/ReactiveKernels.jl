using Pkg, TOML

# Julia 1.10 ignores Project.toml [sources]. Supply the same exact source pins
# to the batched Pkg.develop calls used by the repository's setup entrypoints.
# Checkouts live beside the active environment so later test processes can
# reuse its manifest. Each revision has its own immutable directory.
function rk_source_dependencies(; project = joinpath(@__DIR__, "..", "Project.toml"))
    sources = TOML.parsefile(project)["sources"]
    root = joinpath(dirname(Base.active_project()), ".rk-deps")
    mkpath(root)
    specs = PackageSpec[]
    for (name, source) in sort!(collect(sources); by=first)
        url, revision = source["url"], source["rev"]
        occursin(r"^[0-9a-f]{40}$", revision) || error("expected an exact source revision for $name")
        path = joinpath(root, name * "-" * revision)
        if !isdir(path)
            run(`git clone --quiet $url $path`)
            run(`git -C $path checkout --quiet --detach $revision`)
        end
        actual = strip(read(`git -C $path rev-parse HEAD`, String))
        actual == revision || error("$name source checkout has $actual; expected $revision")
        push!(specs, PackageSpec(path=path))
    end
    return specs
end

if abspath(PROGRAM_FILE) == @__FILE__
    # The root project already has URL-based [sources]. Adding the exact URL
    # pins works on 1.10 too and avoids a path/URL conflict on newer Julia.
    sources = TOML.parsefile(joinpath(@__DIR__, "..", "Project.toml"))["sources"]
    Pkg.add([PackageSpec(; name, url=source["url"], rev=source["rev"])
             for (name, source) in sources])
end
