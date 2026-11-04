using ReactiveKernels, ReactiveKernelsPPL, Test

@rkppl _draw_query_block() = begin
    scale[1:3] .~ Exponential.(1)
    factor ~ LKJCholesky(3, 2.0)
    return scale
end

@testset "host endpoint initialization is shared by concurrent callers" begin
    # Yielding during initialization makes duplicate construction observable
    # even with one Julia thread. A hit must not race another key's insertion.
    cache = Dict{Int,Vector{Int}}()
    calls = Threads.Atomic{Int}(0)
    tasks = map(1:64) do i
        Threads.@spawn ReactiveKernelsPPL._host_bijector_endpoint(cache, mod1(i, 4)) do
            Threads.atomic_add!(calls, 1)
            yield()
            [mod1(i, 4)]
        end
    end
    results = fetch.(tasks)
    @test calls[] == 4
    @test length(cache) == 4
    @test all(i -> results[i] === cache[mod1(i, 4)], eachindex(results))
end

@testset "scoped host draws and selected graph values" begin
    data = (; n = 23)
    source = quote
        block ~ _draw_query_block()
        probability ~ Beta(2, 3)
        spare[1:n] .~ Normal.(0, 1)
        selected = (scale = block.scale, factor = block.factor)
    end
    plan = bind_data(lower_rkppl(source, data; mod = @__MODULE__), data)
    built = build_kernel(plan)
    layout = built.layout
    U = [0.3 * sin(i + j) for i in 1:layout.total, j in 1:32]
    saved = copy(U)

    # Both exp and logistic are reached, with one layout shared by every task.
    tasks = map(axes(U, 2)) do j
        Threads.@spawn constrain(layout, view(U, :, j))
    end
    draws = fetch.(tasks)
    @test draws == [constrain(layout, view(U, :, j)) for j in axes(U, 2)]
    @test all(j -> unconstrain(layout, draws[j]) ≈ U[:, j], axes(U, 2))
    @test U == saved
    @test draws[1].block.scale !== draws[2].block.scale
    @test draws[1].block.factor !== draws[2].block.factor

    restored = restore_draws(layout, U)
    @test size(restored.probability) == (32,)
    @test size(restored.block.scale) == (3, 32)
    @test restored.block.factor isa Vector{Matrix{Float64}}
    @test length(restored.block.factor) == 32
    @test all(j -> restored.block.factor[j] == draws[j].block.factor, axes(U, 2))
    empty = restore_draws(layout, U[:, 1:0])
    @test size(empty.block.scale) == (3, 0)
    @test empty.block.factor isa Vector{Matrix{Float64}}
    @test isempty(empty.block.factor)

    # Only selected transforms are demanded; n is not needed by this cut.
    cut = prepare(built.spec; have = (:unconstrained,), want = :selected)
    batch = vectorize(cut; batched = :unconstrained)
    selected = Base.invokelatest(batch, U)
    @test size(selected.scale) == (3, 32)
    @test size(selected.factor) == (3, 3, 32)
    @test selected.scale ≈ restored.block.scale
    @test all(j -> selected.factor[:, :, j] ≈ restored.block.factor[j], axes(U, 2))
    # Changing unused coordinates cannot affect this selected graph result.
    changed = copy(U)
    spare = findfirst(e -> e.name == :spare, layout.entries)
    entry = layout.entries[spare]
    changed[entry.offset:(entry.offset + entry.size - 1), :] .+= 100
    @test Base.invokelatest(batch, changed) == selected
end
