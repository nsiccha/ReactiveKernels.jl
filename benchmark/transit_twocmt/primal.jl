# Compare the former tuple-projection graph with the same mathematics exposed
# as value-only and joint recipes. Both use the current numerical recurrence;
# the legacy graph reproduces the pre-multi-output rule's preparation boundary.
using ReactiveKernels, ReactiveKernelsPPL
using ReactiveKernels: stage_primal, stage_reverse
using DifferentiationInterface: AutoEnzyme, Constant, gradient
using LinearAlgebra: BLAS, dot
using Printf, Random, Statistics, Test
import Enzyme
BLAS.set_num_threads(1)

@kernel legacy_transit_graph(ts::Vector{Float64}, p::Vector{Float64},
        amounts_bar::Vector{Float64}) = begin
    solution = ReactiveKernelsPPL._transit_response_partials(ts, p, 1e-15, 8)
    amounts::Vector{Float64} = solution[1]
    jac::Matrix{Float64} = solution[2]
    dt::Vector{Float64} = solution[3]
    ts_bar::Vector{Float64} = dt .* amounts_bar
    p_bar::Vector{Float64} = transpose(jac) * amounts_bar
    return amounts, ts_bar, p_bar
end
const LEGACY_RULE = derivative_rule(legacy_transit_graph; primal = :amounts,
    covector = :amounts_bar, cotangents = (ts = :ts_bar, p = :p_bar))
legacy_objective(p, ts, weights) = dot(weights, LEGACY_RULE(ts, p))
current_objective(p, ts, weights) = dot(weights, transit_twocmt_rule(ts, p))

function measure(ts, p)
    weights = [cos(i) for i in eachindex(ts)]
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    contexts = (Constant(ts), Constant(weights))
    oldvalue = LEGACY_RULE(ts, p)
    newvalue = transit_twocmt_rule(ts, p)
    @test newvalue == oldvalue
    oldgrad = gradient(legacy_objective, backend, p, contexts...)
    newgrad = gradient(current_objective, backend, p, contexts...)
    @test newgrad == oldgrad
    for mask in (1, 2, 3)
        oldy, oldtape = stage_primal(LEGACY_RULE, Val(mask), ts, p)
        newy, newtape = stage_primal(transit_twocmt_rule, Val(mask), ts, p)
        @test newy == oldy
        @test stage_reverse(transit_twocmt_rule, Val(mask), newtape, weights) ==
            stage_reverse(LEGACY_RULE, Val(mask), oldtape, weights)
    end
    sinks = [Ref(oldvalue), Ref(newvalue), Ref(oldgrad), Ref(newgrad)]
    runs = (
        () -> begin sinks[1][] = LEGACY_RULE(ts, p); nothing end,
        () -> begin sinks[2][] = transit_twocmt_rule(ts, p); nothing end,
        () -> begin sinks[3][] = gradient(legacy_objective, backend, p, contexts...); nothing end,
        () -> begin sinks[4][] = gradient(current_objective, backend, p, contexts...); nothing end,
    )
    names = ("legacy-primal", "current-primal", "legacy-gradient", "current-gradient")
    for _ in 1:10, run in runs
        run()
    end
    GC.gc()
    samples = [Float64[] for _ in runs]
    rng = MersenneTwister(9)
    for _ in 1:60, j in randperm(rng, length(runs))
        start = time_ns()
        for _ in 1:5
            runs[j]()
        end
        push!(samples[j], (time_ns() - start) / 5e3)
    end
    return [(length(ts), p[5], names[j], median(samples[j]),
        quantile(samples[j], .25), quantile(samples[j], .75), @allocated(runs[j]()))
        for j in eachindex(runs)]
end

function main(output)
    println("Julia ", VERSION, "; Enzyme ", pkgversion(Enzyme), "; CPU ", Sys.CPU_NAME,
        "; Julia/BLAS threads=1; rtol=1e-15, Watson terms=8")
    println("Same-process comparison; 10 warmups, 60 shuffled batches of 5 calls.")
    open(output, "w") do io
        println(io, "lags\tshape\tpath\tmedian_us\tq25_us\tq75_us\tJulia_allocated_bytes")
        for n in (11, 1401), shape in (1.2, 8.)
            ts = collect(range(0., 168.; length = n))
            p = [.08, .15, .05, .2, shape]
            for row in measure(ts, p)
                println(io, join(row, '\t'))
                @printf("lags=%d shape=%.1f %s median=%.3f us bytes=%d\n",
                    row[1], row[2], row[3], row[4], row[7])
            end
            flush(io)
            flush(stdout)
        end
    end
    println("PASS: identical primal values, parameter gradients and staged pullbacks for all masks.")
end
length(ARGS) == 1 || error("Usage: primal.jl <results.tsv>")
main(only(ARGS))
