# Bounded ownership experiment for general DerivativeRule staging.
# Usage: julia --project=<environment> this_file current|snapshot|snapshot_leaf|installed
# `snapshot` replaces only residual retention in this process. It uses Julia's
# deep copy as a conservative comparison, not as a selected production design.
# `current` records wrong gradients without treating them as passing checks.
# `installed` requires correct snapshots from the loaded adapter without replacing it.
using ReactiveKernels, Test, BenchmarkTools, Statistics
import Enzyme
using ReactiveKernels: stage_primal, stage_reverse, stage_residuals

module ResidualSnapshotExperiment
using ReactiveKernels
import Enzyme

struct Record{T}
    values::T
end
mutable struct MutableRecord{T}
    values::T
end
residual_values(x::AbstractVector{Float64}) = x
residual_values(x::Tuple) = only(x)
residual_values(x::NamedTuple) = x.values
residual_values(x::Union{Record,MutableRecord}) = x.values
residual_values(x::Vector{Vector{Float64}}) = only(x)
wrap_view(x) = view(x, :)
wrap_tuple(x) = (x,)
wrap_named_tuple(x) = (; values=x)
wrap_record(x) = Record(x)
wrap_mutable_record(x) = MutableRecord(x)
wrap_nested(x) = [x]

value(x) = sum(abs2, x; init=0.0)
direct(x, seed) = (2seed) .* x
contract(partial, seed) = (2seed) .* residual_values(partial)
struct Producer{F}
    wrap::F
end
(p::Producer)(x) = (value(x), p.wrap(x))

function graph(wrap::F, cost) where {F}
    producer = Producer(wrap)
    @kernel spec(x::Vector{Float64}, seed::Float64) = begin
        @recipe (cost=4) y::Float64 = value(x)
        @recipe (cost=cost) (y, partial) = producer(x)
        @recipe (cost=4) dx::Vector{Float64} = direct(x, seed)
        @recipe (cost=1) dx = contract(partial, seed)
        return y, dx
    end
    derivative_rule(spec; primal=:y, covector=:seed, cotangents=(x=:dx,))
end

function overwritten(f::F, x) where {F}
    y = f(x)
    fill!(x, 0.0)
    y
end
function gradient!(dx, f::F, x) where {F}
    fill!(dx, 0.0)
    Enzyme.autodiff(Enzyme.ReverseWithPrimal, Enzyme.Const(f),
        Enzyme.Active, Enzyme.Duplicated(x, dx))[2]
end

# The returned primal and retained residual are distinct objects with shared
# storage. Mutating the primal must not change the earlier derivative values.
array_value(x) = exp.(x)
function array_joint(x)
    y = exp.(x)
    y, view(y, :)
end
array_contract(partial, seed) = seed .* partial
function array_rule()
    @kernel spec(x::Vector{Float64}, seed::Vector{Float64}) = begin
        y::Vector{Float64} = array_value(x)
        (y, partial) = array_joint(x)
        dx::Vector{Float64} = array_contract(partial, seed)
        return y, dx
    end
    derivative_rule(spec; primal=:y, covector=:seed, cotangents=(x=:dx,))
end
function overwrite_result(f::F, x) where {F}
    y = f(x)
    result = sum(y; init=0.0)
    fill!(y, 0.0)
    result
end
end

function run_snapshot_experiment(mode)
    snapshots = mode in (:snapshot, :snapshot_leaf, :installed)
    ext = Base.get_extension(ReactiveKernels, :ReactiveKernelsEnzymeExt)
    @assert ext !== nothing
    G = ResidualSnapshotExperiment
    wrappers = ((:alias, identity), (:fresh, copy), (:view, G.wrap_view),
        (:tuple, G.wrap_tuple), (:named_tuple, G.wrap_named_tuple),
        (:record, G.wrap_record), (:mutable_record, G.wrap_mutable_record),
        (:nested, G.wrap_nested))
    println((;mode, julia=VERSION, rk=pkgversion(ReactiveKernels),
        enzyme=pkgversion(Enzyme), library=Enzyme.Compiler.Enzyme_jll.libEnzyme_path))
    flush(stdout)
    @testset "residual snapshots: $mode" begin
        for n in (0, 3, 17)
            x = [0.7sin(i) for i in 1:n]; dx = zeros(n)
            @test G.gradient!(dx, G.value, x) ≈ G.value(x)
            @test dx ≈ 2 .* x
        end
        for (label, wrap) in wrappers
            # Cost 2 also makes the old reverse-only planner retain partial.
            rule = G.graph(wrap, 2)
            @test stage_residuals(rule, Val(1)) == (:partial,)
            for n in (0, 3, 17)
                original = [0.7sin(i) for i in 1:n]
                x = copy(original)
                y, tape = stage_primal(rule, Val(1), x)
                retained = ext._retain_residuals((-1,), tape, y, (false, true))
                @test typeof(retained) === typeof(tape)
                fill!(x, 0.0)
                dx = only(stage_reverse(rule, Val(1), retained, 1.0))
                correct = isapprox(dx, 2 .* original)
                snapshots && @test correct
                println((;stage=:retained, mode, label, n, correct))

                x = copy(original); dx = zeros(n)
                caller = x -> G.overwritten(rule, x)
                primal = G.gradient!(dx, caller, x)
                @test primal ≈ G.value(original)
                @test all(iszero, x)
                correct = isapprox(dx, 2 .* original)
                (snapshots || label == :fresh) && @test correct
                println((;stage=:ordinary_reverse, mode, label, n, correct))
                flush(stdout)
            end
            # Audit the actual retention call, including concrete wrapper types.
            x = [0.2, -0.4, 0.7]
            y, tape = stage_primal(rule, Val(1), x)
            findings = ReactiveKernels._level_code_findings(:all, ext._retain_residuals,
                (Tuple{Int}, typeof(tape), typeof(y), Tuple{Bool,Bool}))
            println((;stage=:type_audit, mode, label, findings))
            flush(stdout)
            snapshots && @test isempty(findings)
        end
        rule = G.array_rule()
        for n in (0, 3, 17)
            x = [0.7sin(i) for i in 1:n]; dx = zeros(n)
            caller = x -> G.overwrite_result(rule, x)
            primal = G.gradient!(dx, caller, x)
            @test primal ≈ sum(exp, x; init=0.0)
            correct = isapprox(dx, exp.(x))
            snapshots && @test correct
            println((;stage=:primal_view, mode, n, correct))
        end
        # Deep copy must preserve aliases within a residual environment while
        # separating it from mutable caller storage, including nested records.
        x = [0.2, -0.4]
        tape = (G.Record(x), (x,), view(x, :))
        retained = ext._retain_residuals((-1,-1,-1), tape, 0.0, (false,true))
        if snapshots
            @test retained[1].values === retained[2][1]
            @test parent(retained[3]) === retained[1].values
            @test retained[1].values !== x
        end
        for n in (8, 1000, 100000)
            x = [0.7sin(i) for i in 1:n]
            for (label, wrap) in wrappers
                tape = (wrap(x),)
                retain = ext._retain_residuals
                trial = @benchmark $retain((-1,), $tape, 0.0,
                    (false,true)) samples=40 evals=1 seconds=0.25
                result = median(trial)
                println((;stage=:retention_cost, mode, label, n, ns=result.time,
                    bytes=result.memory, allocations=result.allocs, raw_ns=trial.times))
            end
        end
    end
    println("RESIDUAL_SNAPSHOT_EXPERIMENT_COMPLETE")
end

function install_snapshot_retention!(; leaf=false)
    # Invoke at top level before the driver so all calls see the new method.
    # Copy the tuple together so shared references inside it stay shared.
    ext = Base.get_extension(ReactiveKernels, :ReactiveKernelsEnzymeExt)
    if leaf
        # Diagnostic control: preserve the Base call boundary. Base's copying
        # internals still contain Any-typed state; this does not change them.
        @eval ext @inline function _retain_residuals(
                sources::NTuple{R,Int}, residuals::NTuple{R,Any},
                primal, overwritten_flags) where {R}
            Base.@noinline deepcopy(residuals)
        end
    else
        @eval ext @inline function _retain_residuals(
                sources::NTuple{R,Int}, residuals::NTuple{R,Any},
                primal, overwritten_flags) where {R}
            deepcopy(residuals)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 || error("usage: derivative_residual_snapshots.jl current|snapshot|snapshot_leaf|installed")
    mode = Symbol(only(ARGS))
    mode in (:current, :snapshot, :snapshot_leaf, :installed) || error("unknown mode")
    mode in (:snapshot, :snapshot_leaf) &&
        install_snapshot_retention!(; leaf=mode == :snapshot_leaf)
    run_snapshot_experiment(mode)
end
