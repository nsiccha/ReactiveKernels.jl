using ReactiveKernelsPPL
using Test

# Bounded concurrent-construction regression (snag rkppl-thread-saf-0a062a1c):
# independent lower/bind/build pipelines over distinct ASTs must be safe to
# run from concurrent tasks with no caller-side lock — a build binds nothing
# in `PPLGeneratedModels` and takes no package lock, each task gets back its
# own spec, and lowering never mutates its inputs. Passes on one thread too
# (tasks serialize).

_ccb_cols1() = Dict{Symbol,AbstractVector}(
    :y => [1.0, 2.0, 1.5, 2.5, 3.0, 2.0],
    :x => [0.5, -1.0, 1.5, 0.0, -0.5, 1.0])
_ccb_cols2() = Dict{Symbol,AbstractVector}(
    :y => [1.0, 2.0, 1.5, 2.5, 3.0, 2.0],
    :x => [0.5, -1.0, 1.5, 0.0, -0.5, 1.0],
    :z => [1.0, 0.5, -0.5, 1.5, 0.0, -1.0])

# Two tiny variants with DIFFERENT layout totals, so a build that silently
# read back a sibling's binding fails the per-task total check.
function _ccb_variant(v::Int)
    v == 1 && return Expr(:block,
        :(a ~ Normal(0, 5)),
        :(mu = a .+ b .* x),
        :(y .~ Normal.(mu, s)),
        :(b ~ Normal(0, 2)),
        :(s ~ Exponential(1))), _ccb_cols1(), (:y, :x)
    return Expr(:block,
        :(a ~ Normal(0, 5)),
        :(mu = a .+ b .* x .+ d .* z),
        :(y .~ Normal.(mu, s)),
        :(b ~ Normal(0, 2)),
        :(s ~ Exponential(1)),
        :(d ~ Normal(1, 3))), _ccb_cols2(), (:y, :x, :z)
end

_ccb_submodel_def() =
    :($(Expr(:call, :ccb_scale, :rate)) = $(Expr(:block,
        :(r ~ Exponential(rate)),
        :r)))

_ccb_submodel_main(sc) = Expr(:block,
    :(a ~ Normal(0, 5)),
    :(sig ~ ccb_scale($sc)),
    :(eta = a .+ b .* x),
    :(y .~ Normal.(eta, sig)),
    :(b ~ Normal(0, 2)))

function _ccb_emit_module(defs)
    mod = Module(gensym(:CCBEmitModels))
    Core.eval(mod, :(using ReactiveKernelsPPL))
    for d in defs
        Core.eval(mod, Expr(:macrocall, Symbol("@rkppl"),
            LineNumberNode(0), d))
    end
    mod
end

@testset "concurrent build: lowering never mutates its inputs" begin
    main = _ccb_variant(1)[1]
    def = _ccb_submodel_def()
    main_snap, def_snap = deepcopy(main), deepcopy(def)
    mod = _ccb_emit_module([def])
    unbound = lower_rkppl(main, (:y, :x);
        mod = _ccb_emit_module([deepcopy(def)]), conditioned = (:y, :x))
    # The submodel path lowers through the same pipeline with a def module.
    unbound_sm = lower_rkppl(_ccb_submodel_main(1.0), (:y, :x); mod = mod, conditioned = (:y, :x))
    @test main == main_snap
    @test def == def_snap
    bound = bind_data(unbound, _ccb_cols1())
    bound_sm = bind_data(unbound_sm, _ccb_cols1())
    @test main == main_snap
    @test def == def_snap
    build_kernel(bound)
    build_kernel(bound_sm)
    @test main == main_snap
    @test def == def_snap
end

@testset "concurrent build: independent plain builds" begin
    refs = Dict{Int,Int}()
    for v in (1, 2)
        ast, cols, dn = _ccb_variant(v)
        refs[v] = build_kernel(bind_data(lower_rkppl(ast, dn; conditioned = dn), cols)).layout.total
    end
    @test refs[1] != refs[2] # variants must stay distinguishable
    n_tasks, n_rounds = 4, 3
    results = Vector{Any}(undef, n_tasks * n_rounds)
    @sync for t in 1:n_tasks
        Threads.@spawn begin
            v = isodd(t) ? 1 : 2
            for r in 1:n_rounds
                ast, cols, dn = _ccb_variant(v)
                built = build_kernel(bind_data(lower_rkppl(ast, dn; conditioned = dn), cols))
                results[(t - 1) * n_rounds + r] =
                    (v, built.layout.total, objectid(built.spec))
            end
        end
    end
    for (v, total, _) in results
        @test total == refs[v]
    end
    @test length(Set(oid for (_, _, oid) in results)) == length(results)
end

@testset "concurrent build: fresh-module submodel builds" begin
    ref = build_kernel(bind_data(
        lower_rkppl(_ccb_submodel_main(1.0), (:y, :x);
            mod = _ccb_emit_module([_ccb_submodel_def()]), conditioned = (:y, :x)),
        _ccb_cols1())).layout.total
    n_tasks, n_rounds = 4, 3
    results = Vector{Any}(undef, n_tasks * n_rounds)
    @sync for t in 1:n_tasks
        Threads.@spawn begin
            for r in 1:n_rounds
                mod = _ccb_emit_module([_ccb_submodel_def()])
                unbound = lower_rkppl(_ccb_submodel_main(1.0), (:y, :x);
                    mod = mod, conditioned = (:y, :x))
                built = build_kernel(bind_data(unbound, _ccb_cols1()))
                results[(t - 1) * n_rounds + r] =
                    (built.layout.total, objectid(built.spec))
            end
        end
    end
    for (total, _) in results
        @test total == ref
    end
    @test length(Set(oid for (_, oid) in results)) == length(results)
end

# Concurrency progress: concurrent builds must not serialize each other. A
# build takes no package lock, and construction compiles little per-model
# code, because Julia 1.10-1.13 serialize compilation process-wide (1.12/1.13:
# type inference). Before both held, eight concurrent builds took as long as
# eight serial ones.
include("concurrent_progress.jl")

@testset "concurrent build: construction compiles little per model" begin
    build_kernel(_ccb_wide(6, 0))   # compile this shape's construction path
    Base.cumulative_compile_timing(true)
    compile0 = Base.cumulative_compile_time_ns()[1]
    start = time_ns()
    build_kernel(_ccb_wide(6, 1))
    wall = time_ns() - start
    compile = Base.cumulative_compile_time_ns()[1] - compile0
    Base.cumulative_compile_timing(false)
    # Measured 0.20-0.35 on Julia 1.10.12 and 1.12.7; about 0.95 before
    # construction stopped compiling per-model code.
    @test compile < 0.6 * wall
end

@testset "concurrent build: independent builds overlap" begin
    # Tasks interleave only on several threads; on one thread, measure in a
    # two-thread child process.
    ratio = if Threads.nthreads() >= 2
        _ccb_progress()
    else
        fixture = "include($(repr(joinpath(@__DIR__, "concurrent_progress.jl")))); " *
                  "print(_ccb_progress())"
        parse(Float64, read(`$(Base.julia_cmd()) --startup-file=no --threads=2
                             --project=$(Base.active_project()) -e $fixture`, String))
    end
    # Measured on two threads: 0.69-0.73 on Julia 1.10.12 and 0.48-0.51 on 1.12.7;
    # with a lock around construction, 0.99-1.0 and 0.90.
    @test ratio < 0.8
end
