using Pkg
include(joinpath(@__DIR__, "..", "deps", "source_dependencies.jl"))

# Exact public revisions make local and CI rendering use the same contracts.
const DOCS_SOURCE_PINS = [
    ("AlgebraOfVega", "f96a064557589494d2a1b6ae75c16449758009fb"),
    ("DynamicObjects", "fc41273bce8b3aaf736bb0b23c36c97e773a1c7c"),
    ("HTMX", "0b8dc2f94e879bf31237bd28e751cf2ebc8d95f6"),
    ("HTMXObjects", "0dce4099357c224236ca9b9a2f7d2dfd0e3334b7"),
    ("ReparametrizableDistributions", "8f7d382df700d3780ed31d093bc4983d3734a6c6"),
    ("Treebars", "e03f2c7be5579fafce38d8759b531a6e22996ba5"),
    ("WarmupHMC", "6da3299e09e14547ceb922461c71e4d833e21658"),
]

Pkg.activate(@__DIR__)
specs = rk_source_dependencies()
for (name, revision) in DOCS_SOURCE_PINS
    path = joinpath(@__DIR__, ".rk-deps", name * "-" * revision)
    if !isdir(path)
        run(`git clone --quiet https://github.com/nsiccha/$(name).jl.git $path`)
        run(`git -C $path checkout --quiet --detach $revision`)
    end
    strip(read(`git -C $path rev-parse HEAD`, String)) == revision ||
        error("unexpected docs dependency revision: $name")
    isempty(read(`git -C $path status --porcelain`, String)) ||
        error("docs dependency has local changes: $name")
    push!(specs, PackageSpec(; path))
end
push!(specs, PackageSpec(path=dirname(@__DIR__)))
for name in ("ReactiveKernelsPPL", "ReactiveKernelsDistributionKernels")
    push!(specs, PackageSpec(path=joinpath(dirname(@__DIR__), "packages", name)))
end
cd(@__DIR__) do
    Pkg.develop([PackageSpec(path=relpath(spec.path, @__DIR__)) for spec in specs])
end
Pkg.instantiate()
