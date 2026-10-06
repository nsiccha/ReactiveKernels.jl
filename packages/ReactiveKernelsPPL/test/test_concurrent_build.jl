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
