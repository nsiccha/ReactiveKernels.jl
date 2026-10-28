# Generic acceptance and storage receipt for the opt-in runtime API.
# The original private-consumer probe is unchanged; this uses only its public
# synthetic recurrence, never consumer code or data.
module NativePositionScheduledRuntime

include("native_position_scheduling.jl")
using .NativePositionSchedulingProbe
using ReactiveKernels
using Statistics
using Test
const Probe = NativePositionSchedulingProbe

function verify()
    @testset "opt-in runtime recurrence and storage growth" begin
        for want in ((:lowest, :below), :trajectory)
            scalar = Probe.raw(prepare(Probe.recurrence; want))
            serial = vectorize(scalar; batched=:position)
            scheduled = vectorize(scalar; batched=:position,
                schedule=NativeScheduling(workers=4, chunk_size=8))
            expression = code_expr(scheduled)
            for count in (32, 129, 512), steps in (13, 64)
                positions, data = Probe.fixtures(count), collect(1.0:steps)
                result = scheduled(positions, data, 0.0)
                expected = serial(positions, data, 0.0)
                @test isequal(result, expected)
                numbers = want === :trajectory ? vec(result) : result[1]
                expected_numbers = want === :trajectory ? vec(expected) : expected[1]
                @test reinterpret(UInt64, numbers) == reinterpret(UInt64, expected_numbers)
                @test code_expr(scheduled) === expression
                for worker in scheduled.workers
                    if worker.caches[1][] !== nothing
                        @test size(worker.caches[1][], ndims(worker.caches[1][])) <= 8
                    end
                end
            end
            special = (; initial=[-0.0, 0.0, Inf, -Inf, NaN, 1.0, -1.0],
                         offset=zeros(7), rate=ones(7))
            parallel = vectorize(scalar; batched=:position,
                schedule=NativeScheduling(workers=4, chunk_size=2))
            @test isequal(parallel(special, [0.0, 0.5], 0.0),
                          serial(special, [0.0, 0.5], 0.0))
        end
    end
end

function measure(; count=512, steps=1024, samples=7)
    scalar = Probe.raw(prepare(Probe.recurrence; want=:trajectory))
    original = Probe.candidate(scalar; batched=:position, workers=4, minimum_positions=1)
    scheduled = vectorize(scalar; batched=:position,
        schedule=NativeScheduling(workers=4, chunk_size=8))
    serial = vectorize(scalar; batched=:position)
    args = (Probe.fixtures(count), collect(1.0:steps), 0.0)
    variants = (("serial", serial), ("original_probe", original), ("scheduled_runtime", scheduled))
    expected = serial(args...)
    for (name, template) in variants
        result = template(args...)
        @assert isequal(result, expected)
        slots = name == "serial" ? 0 :
            Base.summarysize([worker.caches for worker in template.workers])
        println("STORAGE name=", name, " positions=", count, " steps=", steps,
                " retained_worker_slots_bytes=", slots,
                " owned_result_bytes=", Base.summarysize(result))
        name == "serial" || copy(template)(args...)
    end
    # Each scheduled/probe operation includes fresh instance construction.
    # Samples alternate order on a shared host; times are exploratory, not a
    # production speedup claim or worker-policy calibration.
    timings = Dict(name => NamedTuple[] for (name, _) in variants)
    for sample in 1:samples
        for (name, template) in (isodd(sample) ? variants : reverse(variants))
            trial = @timed((name == "serial" ? template : copy(template))(args...))
            push!(timings[name], (; seconds=trial.time, bytes=trial.bytes))
        end
    end
    for (name, _) in variants
        trials = timings[name]
        println("OPERATION name=", name, " median_s=", median(x.seconds for x in trials),
                " min_s=", minimum(x.seconds for x in trials),
                " max_s=", maximum(x.seconds for x in trials),
                " median_bytes=", median(x.bytes for x in trials))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    verify()
    measure()
end
end
