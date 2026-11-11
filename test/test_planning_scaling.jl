using ReactiveKernels
using Test
using Random

# Preparation must stay linear or log-linear in the size of the program, never
# quadratic: the user's standing principle, review point
# `scalable-never-quadratic` (user decision `0v0xrxl`). Planning a program used to rescan every
# recipe, or the whole selection, at every step, and plate program extraction
# walked the whole body again for every plate block; both allocated
# quadratically in the program (todo 0xis7be, a downstream application’s per-call preparation). Four
# times the program now allocates about four times as much; a quadratic step
# would allocate about sixteen times as much.

const RK = ReactiveKernels

# A forced wide graph: `n` independent two-step routes summed in a chain.
function wide_planning_graph(n)
    g = Graph()
    x = value!(g, :x, Float64)
    total = x
    for i in 1:n
        a = value!(g, Symbol(:a, i), Float64)
        b = value!(g, Symbol(:b, i), Float64)
        s = value!(g, Symbol(:s, i), Float64)
        add!(g, x => a, sin)
        add!(g, a => b, cos)
        add!(g, (total, b) => s, +)
        total = s
    end
    g, x, total
end

# The order an executable selection runs in: always the smallest ready recipe.
function smallest_ready_order(g, recipes, have)
    available = Set{RK.ValueId}(canon_id(g, v.id) for v in have)
    pending = sort!(collect(recipes); by = r -> r.id)
    order = Recipe[]
    while !isempty(pending)
        at = findfirst(r -> all(v -> canon_id(g, v.id) in available, r.inputs), pending)
        at === nothing && break
        r = popat!(pending, at)
        push!(order, r)
        union!(available, (canon_id(g, v.id) for v in r.outputs))
    end
    order
end

# A function body of `n` plate blocks, each assigning a name the body declares
# before it and the running total reads after it.
function plate_blocks_body(n)
    statements = Any[:(total = 0.0)]
    for i in 1:n
        a = Symbol(:a, i)
        push!(statements, :(local $a::Float64))
        push!(statements, RK._plate_block(Expr(:block, :($a = x * $i))))
        push!(statements, :(total = total + $a))
    end
    Expr(:function, Expr(:tuple, :x), Expr(:block, statements..., :total))
end

@testset "planning allocates linearly in the program" begin
    allocated = map((500, 2000)) do n
        g, x, total = wide_planning_graph(n)
        p = plan(g; have = (x,), want = (total,))
        @test length(p.recipes) == 3n
        @test p.recipes == smallest_ready_order(g, p.recipes, (x,))
        @allocated plan(g; have = (x,), want = (total,))
    end
    @test allocated[2] < 8 * allocated[1]
end

@testset "a plan runs its smallest ready recipe first" begin
    rng = MersenneTwister(20261108)
    for _ in 1:200
        g = Graph()
        vals = [value!(g, Symbol(:v, i), Float64) for i in 1:8]
        for _ in 1:12
            split = rand(rng, 1:7)
            ins = Tuple(vals[randperm(rng, split)[1:rand(rng, 0:min(3, split))]])
            later = vals[(split + 1):8]
            outs = Tuple(later[randperm(rng, length(later))[1:rand(rng, 1:min(2, length(later)))]])
            add!(g; inputs = ins, outputs = outs, op = identity,
                 cost = rand(rng, (0.0, 0.25, 1.0)))
        end
        have = (vals[1],)
        want = (vals[7], vals[8])
        p = try
            plan(g; have, want)
        catch e
            e isa PlanningError || rethrow()
            continue
        end
        @test p.recipes == smallest_ready_order(g, p.recipes, have)
    end
end

@testset "plate program extraction allocates linearly in the body" begin
    program_call(arg) = arg isa Expr && arg.head === :(=) && arg.args[2] isa Expr &&
        arg.args[2].head === :call && first(arg.args[2].args) isa RK._PlateProgram
    allocated = map((250, 1000)) do n
        ast = plate_blocks_body(n)
        extracted = RK._extract_plate_programs(ast)
        calls = filter(program_call, extracted.args[2].args)
        @test length(calls) == n
        # Each program declares again the name the body declared before it.
        program = first(first(calls).args[2].args)
        @test first(program.ast.args[2].args) == :(local a1::Float64)
        @allocated RK._extract_plate_programs(ast)
    end
    @test allocated[2] < 8 * allocated[1]
end
