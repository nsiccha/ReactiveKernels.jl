using ReactiveKernels, Test

# `compose` joins ports by name. `ReactiveKernels.rename` lets a kernel written
# against one port name read a value of another name, such as a summary kernel
# composed onto a model whose prediction value is named at run time (snag
# `kernel-input-bou-ce13eae3`, user pick on decision `1frxgh5`).

@kernel rename_model(a, b, t) = begin
    mu = a .+ b .* t
end

@kernel rename_summaries(prediction, rows, lo, hi) = begin
    cells = plate(eachindex(rows)) do s
        cell::Float64 = sum(prediction[rows[s]]) + lo[s] * hi[s]
    end
    peaks = plate(eachindex(lo)) do k
        cell::Float64 = maximum(prediction) - lo[k]
    end
    total = sum(cells)
    return total, peaks
end

# The same summary written against the model's name: the reference.
@kernel rename_summaries_mu(mu, rows, lo, hi) = begin
    cells = plate(eachindex(rows)) do s
        cell::Float64 = sum(mu[rows[s]]) + lo[s] * hi[s]
    end
    peaks = plate(eachindex(lo)) do k
        cell::Float64 = maximum(mu) - lo[k]
    end
    total = sum(cells)
    return total, peaks
end

@testset "rename: compose a kernel onto a value of another name" begin
    t = collect(0.0:0.5:3.0)
    rows = [[1, 2], [3, 4, 5], [6, 7]]
    lo = [0.1, 0.2, 0.3]
    hi = [1.0, 2.0, 3.0]
    node = Symbol("m", "u")   # named at run time
    renamed = ReactiveKernels.rename(rename_summaries, :prediction => node)
    composed = compose(rename_model, renamed)
    reference = compose(rename_model, rename_summaries_mu)
    have = (:a, :b, :t, :rows, :lo, :hi)
    want = (:total, :peaks, :mu)

    @testset "values" begin
        k = prepare(composed; have, want)
        r = prepare(reference; have, want)
        for (a, b) in ((1.0, 2.0), (-0.5, 0.25), (3.0, -1.0))
            @test k(a, b, t, rows, lo, hi) == r(a, b, t, rows, lo, hi)
        end
        @test k(1.0, 2.0, t, rows, lo, hi) ==
              (29.4, [6.9, 6.8, 6.7], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        # The renamed port is the model's value itself: no alias recipe.
        @test length(kernel_graph(composed).recipes) ==
              length(kernel_graph(reference).recipes)
        authored(spec) = filter(name -> !startswith(string(name), "#"), keys(spec))
        @test authored(composed) == authored(reference)
    end

    @testset "bound data and position batching" begin
        bound = (; t, rows, lo, hi)
        k = vectorize(prepare(composed; have, want = (:total, :peaks), bound);
                      batched = (:a, :b))
        r = vectorize(prepare(reference; have, want = (:total, :peaks), bound);
                      batched = (:a, :b))
        @test k([1.0, 2.0], [2.0, 3.0]) == r([1.0, 2.0], [2.0, 3.0])
        @test k([1.0, 2.0], [2.0, 3.0]) ==
              ([29.4, 46.9], [6.9 10.9; 6.8 10.8; 6.7 10.7])
    end

    @testset "the renamed kernel alone" begin
        @test keys(renamed) ==
              map(n -> n === :prediction ? :mu : n, keys(rename_summaries))
        @test renamed.have_names == [:mu, :rows, :lo, :hi]
        @test renamed.want_names == [:total, :peaks]
        mu = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        @test prepare(renamed)(mu, rows, lo, hi) ==
              prepare(rename_summaries)(mu, rows, lo, hi)
        @test prepare(renamed; have = (:mu, :rows, :lo, :hi), want = :total)(
            mu, rows, lo, hi) == 29.4
    end

    @testset "the input kernel is unchanged" begin
        ports = copy(rename_summaries.port_order)
        values = copy(kernel_graph(rename_summaries).values)
        ReactiveKernels.rename(rename_summaries, :prediction => :other)
        @test rename_summaries.port_order == ports
        @test kernel_graph(rename_summaries).values == values
        @test haskey(rename_summaries, :prediction)
        @test !haskey(rename_summaries, :other)
    end
end

@kernel rename_keyword(x; scale = 2.0) = begin
    y = x * scale
end

@kernel rename_difference(a, b) = begin
    c = a - b
end

@kernel rename_scaled(scale::Float64) = begin
    density(x::Float64)::Float64 = -abs(x) / scale
end

@testset "rename: call signature and positions" begin
    @testset "a renamed keyword port renames its keyword" begin
        r = prepare(ReactiveKernels.rename(rename_keyword, :scale => :s))
        @test r(3.0; s = 4.0) == 12.0
        @test r(3.0) == 6.0
        @test_throws ArgumentError r(3.0; scale = 4.0)
    end

    @testset "renamed ports keep their positions; names may swap" begin
        swapped = ReactiveKernels.rename(rename_difference, :a => :b, :b => :a)
        @test swapped.have_names == [:b, :a]
        @test prepare(swapped)(5.0, 2.0) == 3.0
        # Naming the boundary selects by the new names.
        @test prepare(swapped; have = (:a, :b), want = :c)(5.0, 2.0) == -3.0
    end

    @testset "an endpoint view keeps its signature kind" begin
        view = rename_scaled.density
        renamed = ReactiveKernels.rename(view, :scale => :width)
        @test haskey(renamed, :width)
        @test prepare(renamed; have = (:x, :width), want = :density)(2.0, 4.0) ==
              prepare(view; have = (:x, :scale), want = :density)(2.0, 4.0)
    end
end

@testset "rename: refusals" begin
    # refused: a rename names existing ports; an unknown port is a caller error
    # (decision `1frxgh5`, option 1).
    @test_throws ArgumentError ReactiveKernels.rename(rename_difference, :z => :y)
    # refused: one port cannot take two names at once (decision `1frxgh5`).
    @test_throws ArgumentError ReactiveKernels.rename(rename_difference,
                                                      :a => :x, :a => :y)
    # refused: a rename is one-to-one; joining two ports under one name is
    # `compose`'s job, not a rename's (decision `1frxgh5`, option 1).
    @test_throws ArgumentError ReactiveKernels.rename(rename_difference, :a => :b)
    @test_throws ArgumentError ReactiveKernels.rename(rename_difference,
                                                      :a => :x, :b => :x)
    message = try
        ReactiveKernels.rename(rename_difference, :a => :c)
        ""
    catch err
        sprint(showerror, err)
    end
    @test occursin("compose", message) && occursin(":a and :c", message)
end
