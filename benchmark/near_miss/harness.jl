# Near-miss probe harness: time the density and the Enzyme reverse gradient of
# sibling spellings of one model family, on one data set, in one process.
using ReactiveKernelsPPL, ReactiveKernels, Enzyme, ADTypes, Chairmarks, Random, LinearAlgebra, Printf
BLAS.set_num_threads(1)

const BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)

struct ProbeResult
    family::String
    name::String
    dim::Int
    value::Float64
    gnorm::Float64
    t_density::Float64   # ns
    t_gradient::Float64  # ns
    a_gradient::Int      # allocations per gradient
    note::String
end

# An optional third command-line argument is a regular expression over the
# variant names; a probe it does not match is skipped.
const PROBE_FILTER = length(ARGS) >= 3 ? Regex(ARGS[3]) : r""
struct Skipped end
report(::Skipped, nrows) = nothing

function probe(family, name, model, inputs, observed; seed = 1)
    occursin(PROBE_FILTER, name) || return Skipped()
    try
        plan = model(; inputs...) | observed
        built = build_kernel(plan)
        n = built.layout.total
        u = 0.1 .* randn(MersenneTwister(seed), n)
        q = prepare_sampler(built, plan, u; backend = BACKEND)
        g = similar(u)
        v = q(u)
        sampler_value_and_gradient!(q, g, u)
        bd = @b $q($u) seconds = 0.5
        bg = @b sampler_value_and_gradient!($q, $g, $u) seconds = 0.5
        ProbeResult(family, name, n, v, norm(g), bd.time * 1e9, bg.time * 1e9, Int(bg.allocs), "")
    catch err
        msg = sprint(showerror, err)
        ProbeResult(family, name, 0, NaN, NaN, NaN, NaN, -1, first(msg, 300))
    end
end

# One line per probe, printed as soon as it is measured, so a run that stops
# early still leaves its finished rows.
function report_header()
    @printf("%-10s %-34s %4s %14s %12s %12s %8s %8s %6s\n", "family", "variant", "dim",
        "value", "dens ns", "grad ns", "d ns/row", "g ns/row", "allocs")
    flush(stdout)
end
function report(r::ProbeResult, nrows)
    @printf("%-10s %-34s %4d %14.6f %12.0f %12.0f %8.2f %8.2f %6d %s\n", r.family, r.name, r.dim,
        r.value, r.t_density, r.t_gradient, r.t_density / nrows, r.t_gradient / nrows,
        r.a_gradient, r.note)
    flush(stdout)
end
