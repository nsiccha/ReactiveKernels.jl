# Docs renderer for the discrete-latent page (docs/src/discrete-latents.md).
# Every example runs at docs-build time from the same files ReactiveKernelsPPL's
# `test_discrete_latents.jl` checks against independent references
# (`packages/ReactiveKernelsPPL/test/discrete_latent_examples/`), and the
# state-posterior kernel is read from DistributionKernels' source, so neither
# the programs nor the kernel shown on the page can drift from the code that
# runs.
module DiscreteLatentDocs

using Markdown, Printf
import ReactiveKernelsPPL, ReactiveKernelsDistributionKernels

const EXAMPLES = joinpath(pkgdir(ReactiveKernelsPPL), "test", "discrete_latent_examples")
const KERNELS = joinpath(pkgdir(ReactiveKernelsDistributionKernels), "src",
    "distribution_kernel_sources.jl")

_read(path) = replace(read(path, String), "\r\n" => "\n", "\r" => "\n")

"""Run one example file in a fresh module, exactly as displayed."""
function run_example(file::AbstractString)
    mod = Module(gensym(:DiscreteLatentExample))
    Base.include_string(mod, _read(joinpath(EXAMPLES, file)), file)
    return mod
end

render_source(file::AbstractString) =
    Markdown.MD(Markdown.Code("julia", rstrip(_read(joinpath(EXAMPLES, file)))))

"""The `@kernel` definition named `name` in DistributionKernels' source."""
function render_kernel(name::AbstractString)
    lines = split(_read(KERNELS), '\n')
    start = findfirst(l -> startswith(l, "@kernel $name("), lines)
    start === nothing && error("discrete-latent docs: no `@kernel $name(` in $KERNELS")
    stop = findnext(==("end"), lines, start)
    stop === nothing && error("discrete-latent docs: `@kernel $name` has no closing `end`")
    Markdown.MD(Markdown.Code("julia", join(lines[start:stop], '\n')))
end

_text(lines) = Markdown.MD(Markdown.Code("text", join(lines, '\n')))
_row(v) = join((@sprintf("%.3f", x) for x in v), "  ")

function render_hmm()
    m = run_example("hmm.jl")
    P = m.posterior.probabilities
    _text([
        "size(posterior.probabilities) = $(size(P))",
        "first steps' state probabilities (rows: steps 1-4, columns: states 1-3):",
        ("  " * _row(P[t, :]) for t in 1:4)...,
        "most probable state equals the simulated state at " *
            "$(count(m.most_probable .== m.true_s)) of $(length(m.true_s)) steps",
        "the drawn path equals it at $(count(m.path .== m.true_s)) of $(length(m.true_s)) steps",
    ])
end

function render_mixture()
    m = run_example("mixture.jl")
    P = m.memberships
    _text([
        "size(memberships) = $(size(P))",
        "first observations' component probabilities:",
        ("  y = $(@sprintf("%6.3f", m.y[i])):  " * _row(P[i, :]) for i in 1:4)...,
        "drawn components equal the simulated ones for " *
            "$(count(m.components .== m.true_z)) of $(length(m.true_z)) observations",
    ])
end

function render_changepoint()
    m = run_example("changepoint.jl")
    p = m.probabilities
    mode = argmax(p)
    _text([
        "length(probabilities) = $(length(p))",
        "most probable switch year: $mode (probability $(@sprintf("%.3f", p[mode])))",
        "probability of years 38-44: $(@sprintf("%.3f", sum(p[38:44])))",
    ])
end

end # module DiscreteLatentDocs
