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
    @kernel graph(x::Float64) = begin
        z::Float64 = 2x
        w::Float64 = z + 1
    end

    state = ReactiveState(graph; materialize = :z)
    set!(state, graph.x, 3.0)
    initial = get!(state, graph.w)
    set!(state, graph.x, 10.0)
    updated = get!(state, graph.w)

    kernel = prepare(graph)
    (; initial, updated, recipe_count = length(kernel.plan.recipes),
       allocations = _allocated(kernel, 10.0), kernel)
end

"Port of ReactiveObjects.jl's `diamond` gallery example."
function diamond_example()
    calls = Dict(:b => Ref(0), :c => Ref(0), :d => Ref(0))
    @kernel graph(a::Float64) = begin
        b::Float64 = (calls[:b][] += 1; a + 1)
        c::Float64 = (calls[:c][] += 1; a + 2)
        d::Float64 = (calls[:d][] += 1; b + c)
    end

    state = ReactiveState(graph; materialize = (:b, :c))
    set!(state, graph.a, 1.0)
    initial = get!(state, graph.d)

    foreach(counter -> counter[] = 0, values(calls))
    set!(state, graph.a, 10.0)
    b_only = get!(state, graph.b)
    updated = get!(state, graph.d)

    kernel = prepare(graph)
    (; initial, b_only, updated,
       calls = (; b = calls[:b][], c = calls[:c][], d = calls[:d][]),
       allocations = _allocated(kernel, 10.0), kernel)
end

"Port of ReactiveObjects.jl's `shared` gallery example."
function shared_example()
    a_calls = Ref(0)
    @kernel graph(x::Float64) = begin
        a::Float64 = (a_calls[] += 1; x + 1)
        b::Float64 = 2a
        c::Float64 = 3a
    end

    state = ReactiveState(graph; materialize = :a)
    set!(state, graph.x, 5.0)
    b_value = get!(state, graph.b)
    c_value = get!(state, graph.c)

    kernel = prepare(graph)
    (; b = b_value, c = c_value, a_calls = a_calls[],
       allocations = _allocated(kernel, 5.0), kernel)
end

function run(io::IO = stdout)
    chain = chain_example()
    diamond = diamond_example()
    shared = shared_example()
    println(io, "ReactiveObjects.jl compatibility examples")
    println(io, "  chain:   ", (;
        chain.initial, chain.updated, chain.recipe_count, chain.allocations,
    ))
    println(io, "  diamond: ", (;
        diamond.initial, diamond.b_only, diamond.updated, diamond.calls,
        diamond.allocations,
    ))
    println(io, "  shared:  ", (;
        shared.b, shared.c, shared.a_calls, shared.allocations,
    ))
    nothing
end

end # module ReactiveObjectsExamples

if abspath(PROGRAM_FILE) == @__FILE__
    ReactiveObjectsExamples.run()
end
