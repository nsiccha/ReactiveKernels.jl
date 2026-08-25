# Executable ports of the examples on ReactiveObjects.jl `dev` at
# 118e73b86dcd8bb8854d1f535249b49008575a6e.
module ReactiveObjectsExamples

using ReactiveKernels

export chain_example, diamond_example, shared_example, run

function _allocated(k, args...)
    k(args...)
    @allocated k(args...)
end

"Port of ReactiveObjects.jl's `chain` gallery example."
function chain_example()
    graph = Graph()
    x = value!(graph, :x, Float64)
    z = value!(graph, :z, Float64)
    w = value!(graph, :w, Float64)

    add!(graph, x => z, x -> 2x)
    add!(graph, z => w, z -> z + 1)

    state = ReactiveState(graph; materialize = (z,))
    set!(state, x, 3.0)
    initial = get!(state, w)
    set!(state, x, 10.0)
    updated = get!(state, w)

    kernel = prepare(graph; have = (x,), want = (w,))
    (; initial, updated, recipe_count = length(kernel.plan.recipes),
       allocations = _allocated(kernel, 10.0))
end

"Port of ReactiveObjects.jl's `diamond` gallery example."
function diamond_example()
    graph = Graph()
    a = value!(graph, :a, Float64)
    b = value!(graph, :b, Float64)
    c = value!(graph, :c, Float64)
    d = value!(graph, :d, Float64)

    calls = Dict(:b => Ref(0), :c => Ref(0), :d => Ref(0))
    add!(graph, a => b, a -> (calls[:b][] += 1; a + 1))
    add!(graph, a => c, a -> (calls[:c][] += 1; a + 2))
    add!(graph, (b, c) => d, (b, c) -> (calls[:d][] += 1; b + c))

    state = ReactiveState(graph; materialize = (b, c))
    set!(state, a, 1.0)
    initial = get!(state, d)

    foreach(counter -> counter[] = 0, values(calls))
    set!(state, a, 10.0)
    b_only = get!(state, b)
    updated = get!(state, d)

    kernel = prepare(graph; have = (a,), want = (d,))
    (; initial, b_only, updated,
       calls = (; b = calls[:b][], c = calls[:c][], d = calls[:d][]),
       allocations = _allocated(kernel, 10.0))
end

"Port of ReactiveObjects.jl's `shared` gallery example."
function shared_example()
    graph = Graph()
    x = value!(graph, :x, Float64)
    a = value!(graph, :a, Float64)
    b = value!(graph, :b, Float64)
    c = value!(graph, :c, Float64)

    a_calls = Ref(0)
    add!(graph, x => a, x -> (a_calls[] += 1; x + 1))
    add!(graph, a => b, a -> 2a)
    add!(graph, a => c, a -> 3a)

    state = ReactiveState(graph; materialize = (a,))
    set!(state, x, 5.0)
    b_value = get!(state, b)
    c_value = get!(state, c)

    kernel = prepare(graph; have = (x,), want = (b, c))
    (; b = b_value, c = c_value, a_calls = a_calls[],
       allocations = _allocated(kernel, 5.0))
end

function run(io::IO = stdout)
    println(io, "ReactiveObjects.jl compatibility examples")
    println(io, "  chain:   ", chain_example())
    println(io, "  diamond: ", diamond_example())
    println(io, "  shared:  ", shared_example())
    nothing
end

end # module ReactiveObjectsExamples

if abspath(PROGRAM_FILE) == @__FILE__
    ReactiveObjectsExamples.run()
end
