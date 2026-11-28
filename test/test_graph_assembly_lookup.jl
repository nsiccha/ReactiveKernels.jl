# Graph assembly finds an earlier recipe by its CSE key, files each recipe's
# producers once per value class, rebuilds the producer index after a nested
# kernel call only when an alias moved a class that had producers, and
# `compose` clones every kernel once. These pin that each result equals the one
# a scan of all recipes, a full re-index and pairwise merges give
# (snag fitted-open-on-a-3082f909).
using ReactiveKernels
using Test

module GraphAssemblyLookup
using ReactiveKernels
@kernel shift(x::Float64) = begin
    shifted::Float64 = x + 1
    return shifted
end
@kernel model(a::Float64, b::Float64) = begin
    left::Float64 = shift(a)
    right::Float64 = shift(b)
    again::Float64 = shift(a)
    total::Float64 = left + right + again
    return total
end
@kernel double(total::Float64) = begin
    doubled::Float64 = 2 * total
    return doubled
end
@kernel square(total::Float64, doubled::Float64) = begin
    squared::Float64 = total * doubled
    return squared
end
@kernel mismatched(total::Int) = begin
    twice::Int = 2 * total
    return twice
end
# The child's typed result joins a value that already has a producer.
@kernel alternative(a::Float64) = begin
    y = a + 1
    y = shift(a)
    return y
end
end

# A graph's structure by name: every recipe with its inputs and outputs as
# value classes, the classes, and the producer index.
function _assembly_structure(spec)
    g = kernel_graph(spec)
    root(v) = g.values[canon_id(g, v.id)].name
    classes = Dict{Symbol,Vector{Symbol}}()
    for id in sort!(collect(keys(g.values)))
        push!(get!(classes, root(g.values[id]), Symbol[]), g.values[id].name)
    end
    (; recipes = [(map(root, r.inputs), map(root, r.outputs), r.cost, r.effectful,
                   r.cse_key === nothing) for r in g.recipes],
       classes, producers = Dict(g.values[id].name => copy(ids) for (id, ids) in g.producers),
       order = copy(spec.port_order), have = copy(spec.have_names),
       want = copy(spec.want_names), checks = length(g.checks))
end

# The producer index of `g` is the one a full re-index gives.
function _producers_current(g)
    rebuilt = deepcopy(g)
    ReactiveKernels._reindex_producers!(rebuilt)
    rebuilt.producers == g.producers
end

@testset "graph assembly: keyed recipes" begin
    A = GraphAssemblyLookup

    @testset "a table finds the recipe a scan finds" begin
        function assemble(keyed)
            g = Graph()
            x = value!(g, :x, Float64)
            outs = [value!(g, Symbol(:o, i), Float64) for i in 1:7]
            f(x) = x + 1
            h(x) = 2x
            add_keyed(out, op, key; effectful = false) = ReactiveKernels._add_recipe!(
                g, (x,), (out,), op, 1.0, key, effectful, nothing, keyed)
            # Two recipes the table has not seen when it is first used.
            add!(g, x => outs[1], f; cse_key = (:f, 1))
            add!(g, x => outs[2], h; cse_key = (:h, 1.0))
            found = Int[add_keyed(outs[3], f, (:f, 1)).id,
                        add_keyed(outs[4], h, (:h, 2)).id,
                        add_keyed(outs[5], h, (:h, 2)).id,
                        # `isequal(1, 1.0)`: equal keys of different types.
                        add_keyed(outs[6], h, (:h, 1)).id,
                        add_keyed(outs[7], f, (:f, 1); effectful = true).id]
            classes = [canon_id(g, o.id) == canon_id(g, outs[1].id) for o in outs]
            found, classes, length(g.recipes), _producers_current(g)
        end
        scanned = assemble(nothing)
        keyed = assemble(ReactiveKernels._KeyedRecipes())
        @test scanned == keyed
        @test scanned[1] == [1, 3, 3, 2, 4]
        @test scanned[4]
        # One assembly gives each graph it builds a table of its own.
        @test ReactiveKernels._assembly_table(Graph()) === nothing
        ReactiveKernels._with_assembly_tables() do
            g = Graph()
            table = ReactiveKernels._assembly_table(g)
            @test table isa ReactiveKernels._KeyedRecipes
            @test ReactiveKernels._assembly_table(g) === table
            @test ReactiveKernels._assembly_table(Graph()) !== table
            @test assemble(nothing) == scanned
        end
        @test ReactiveKernels._assembly_table(Graph()) === nothing
    end

    @testset "compose equals pairwise merges" begin
        for specs in ((A.model, A.double, A.square), (A.model, A.double, A.double, A.square),
                      (A.double, A.model), (A.model,))
            composed = compose(specs...)
            pairwise = length(specs) == 1 ? compose(only(specs)) : foldl(merge, specs)
            @test _assembly_structure(composed) == _assembly_structure(pairwise)
            @test _producers_current(kernel_graph(composed))
        end
        # The same fragment twice adds its recipes once.
        @test length(kernel_graph(compose(A.model, A.double, A.double)).recipes) ==
            length(kernel_graph(compose(A.model, A.double)).recipes)
        fragment = compose(A.model, A.double, A.square; boundary = :fragment)
        @test fragment.have_names == A.square.have_names
        @test fragment.want_names == A.square.want_names
        @test _assembly_structure(compose(A.model, A.double, A.square)).recipes ==
            _assembly_structure(fragment).recipes
        k = prepare(compose(A.model, A.double, A.square); have = (:a, :b), want = (:squared,))
        @test k(1.0, 2.0) == 98.0
        # A later kernel's port type is checked against the first declaration.
        mismatch = try
            compose(A.model, A.double, A.mismatched)
        catch err
            err
        end
        @test mismatch isa ArgumentError
        @test mismatch.msg == try
            merge(merge(A.model, A.double), A.mismatched)
        catch err
            err.msg
        end
    end

    @testset "the producer index after nested kernel calls" begin
        for spec in (A.model, A.alternative, compose(A.model, A.double),
                     ReactiveKernels.rename(A.model, :total => :sum))
            @test _producers_current(kernel_graph(spec))
        end
        @test prepare(A.alternative)(2.0) == 3.0
        # Two outputs of one recipe in one value class are one producer.
        g = Graph()
        x = value!(g, :x, Float64)
        a = value!(g, :a, Float64)
        b = value!(g, :b, Float64)
        g.aliases[b.id] = a.id
        add!(g, x => (a, b), x -> (x, x))
        @test g.producers[canon_id(g, a.id)] == [1]
        @test _producers_current(g)
    end
end
