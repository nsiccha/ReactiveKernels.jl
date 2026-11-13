# Requires the provisional joint-staging implementation.
# Select scalar_exp, array_exp, or array_square as the sole command argument.
# Use a prepared environment with ReactiveKernels, Enzyme, and BenchmarkTools.
# Callable type parameters keep the experiment caller from adding dispatch costs.
# Mechanism comparison, not a model benchmark or an isolated version speedup.
# The high-cost control selects the same independent cuts as old staging.
using ReactiveKernels, Test, BenchmarkTools, Statistics, LinearAlgebra
import Enzyme
using ReactiveKernels: stage_primal, stage_reverse, stage_residuals
BLAS.set_num_threads(1)

scalar_value(x) = exp(x)
scalar_joint(x) = (y=exp(x); (y,y))
scalar_direct(x, seed) = seed * exp(x)
scalar_contract(partial, seed) = seed * partial
exp_value(x) = sum(exp, x; init=0.0)
function exp_joint(x)
    partial = exp.(x)
    sum(partial; init=0.0), partial
end
exp_direct(x, seed) = seed .* exp.(x)
square_value(x) = sum(abs2, x; init=0.0)
square_joint(x) = (square_value(x), 2 .* x)
square_direct(x, seed) = (2seed) .* x
array_contract(partial, seed) = seed .* partial

function scalar_graph(joint_cost)
    @kernel spec(x::Float64, seed::Float64) = begin
        @recipe (cost=4) y::Float64 = scalar_value(x)
        @recipe (cost=joint_cost) (y, partial::Float64) = scalar_joint(x)
        @recipe (cost=4) dx::Float64 = scalar_direct(x, seed)
        @recipe (cost=1) dx = scalar_contract(partial, seed)
        return y, dx
    end
    spec
end
function array_graph(value, joint, direct, joint_cost)
    @kernel spec(x::Vector{Float64}, seed::Float64) = begin
        @recipe (cost=4) y::Float64 = value(x)
        @recipe (cost=joint_cost) (y, partial::Vector{Float64}) = joint(x)
        @recipe (cost=4) dx::Vector{Float64} = direct(x, seed)
        @recipe (cost=1) dx = array_contract(partial, seed)
        return y, dx
    end
    spec
end
make_rule(spec) = derivative_rule(spec; primal=:y, covector=:seed,
    cotangents=(x=:dx,), name=:joint_tradeoff)

function timed_phase(f)
    Base.cumulative_compile_timing(true)
    before = Base.cumulative_compile_time_ns()
    local measurement, compilation
    try
        measurement = @timed f()
    finally
        compilation = Base.cumulative_compile_time_ns() .- before
        Base.cumulative_compile_timing(false)
    end
    measurement.value, (; seconds=measurement.time, bytes=measurement.bytes,
        compile_seconds=first(compilation)/1e9,
        recompile_seconds=last(compilation)/1e9)
end

# Separate loops keep the reverse seed unavailable during the forward sweep.
# Each loop retains runtime N; the tape arrays themselves are preallocated.
function scalar_stages!(values, tapes, gradient, rule::R, x, seeds) where {R}
    for i in eachindex(x)
        values[i], tapes[i] = stage_primal(rule, Val(1), x[i])
    end
    for i in eachindex(x)
        gradient[i] = only(stage_reverse(rule, Val(1), tapes[i], seeds[i]))
    end
    nothing
end
function array_stages(rule::R, x, seed) where {R}
    y, tape = stage_primal(rule, Val(1), x)
    y, only(stage_reverse(rule, Val(1), tape, seed))
end
function ordinary!(gradient, f::F, x) where {F}
    fill!(gradient, 0.0)
    Enzyme.autodiff(Enzyme.ReverseWithPrimal, Enzyme.Const(f),
        Enzyme.Active, Enzyme.Duplicated(x, gradient))[2]
end

function record_trials(label, n, arms)
    # Rotating forward/reversed arm order, including all raw within-round times.
    for round in 1:12
        order = collect(eachindex(arms))
        order = circshift(order, mod(round-1, length(order)))
        iseven(round) && reverse!(order)
        for index in order
            arm, f = arms[index]
            trial = @benchmark $f() samples=60 evals=1 seconds=0.25
            result = median(trial)
            println((;stage=:warm, label, n, round, arm, ns=result.time,
                bytes=result.memory, allocations=result.allocs, raw_ns=trial.times))
        end
    end
end

function run_family(label)
    scalar = label == :scalar_exp
    value, joint, direct = label == :array_square ?
        (square_value, square_joint, square_direct) : (exp_value, exp_joint, exp_direct)
    graph(cost) = scalar ? scalar_graph(cost) : array_graph(value,joint,direct,cost)
    # Preparation order is printed: these are sequential observations sharing
    # compiler caches, not an independent cold-cost comparison.
    independent, independent_cost = timed_phase(() -> make_rule(graph(10)))
    shared, shared_cost = timed_phase(() -> make_rule(graph(5)))
    println((;stage=:prepare, label, arm=:independent, independent_cost...))
    println((;stage=:prepare, label, arm=:shared, shared_cost...))
    @test stage_residuals(independent,Val(1)) == (:x,)
    @test stage_residuals(shared,Val(1)) == (:partial,)
    for n in (0, 8, 1000, 100000)
        x = [0.7sin(i) for i in 1:n]
        gradient = zeros(n)
        expected = label == :array_square ? 2 .* x : exp.(x)
        ordinary_value = ordinary!(gradient,value,x)
        @test gradient ≈ expected
        if scalar
            values = zeros(n); tapes = Vector{Tuple{Float64}}(undef,n)
            seeds = [iseven(i) ? -0.7 : 1.3 for i in 1:n]
            for rule in (independent,shared)
                scalar_stages!(values,tapes,gradient,rule,x,seeds)
                @test values ≈ expected
                @test gradient ≈ expected .* seeds
            end
            arms = ((:independent,()->scalar_stages!(values,tapes,gradient,independent,x,seeds)),
                    (:shared,()->scalar_stages!(values,tapes,gradient,shared,x,seeds)))
            # The ordinary sum has a unit seed; it is a numerical control only.
            println((;stage=:storage,label,n,tape_payload_bytes=8n))
        else
            for rule in (independent,shared), seed in (-0.7,0.0,1.3)
                y, dx = array_stages(rule,x,seed)
                @test y ≈ ordinary_value
                @test dx ≈ expected .* seed
                @test ordinary!(gradient,rule,x) ≈ ordinary_value
                @test gradient ≈ expected
            end
            arms = ((:independent,()->array_stages(independent,x,1.0)),
                    (:shared,()->array_stages(shared,x,1.0)),
                    (:ordinary,()->ordinary!(gradient,value,x)),
                    (:adapter_independent,()->ordinary!(gradient,independent,x)),
                    (:adapter_shared,()->ordinary!(gradient,shared,x)))
            println((;stage=:storage,label,n,extra_shared_tape_payload_bytes=8n))
        end
        for (arm,f) in arms
            _, measured_cost = timed_phase(f)
            println((;stage=:post_correctness_call,label,n,arm,measured_cost...))
        end
        n == 0 || record_trials(label,n,arms)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
println((;julia=VERSION,rk=pkgversion(ReactiveKernels),enzyme=pkgversion(Enzyme),
    enzyme_source=pathof(Enzyme),library=Enzyme.Compiler.Enzyme_jll.libEnzyme_path,
    threads=Threads.nthreads(),blas_threads=BLAS.get_num_threads()))
length(ARGS)==1 || error("usage: joint_tradeoffs.jl scalar_exp|array_exp|array_square")
label=Symbol(only(ARGS))
label in (:scalar_exp,:array_exp,:array_square) || error("unknown family")
run_family(label)
println("JOINT_TRADEOFFS_COMPLETE")
end
