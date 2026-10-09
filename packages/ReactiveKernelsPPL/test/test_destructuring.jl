module PPLDestructuringTests
using ReactiveKernels, ReactiveKernelsPPL, DifferentiationInterface, Enzyme, Test
using Distributions: Normal, logpdf

# `(a, b) = f(x)` is Julia's destructuring: `f(x)` runs once and each name
# binds its element. The generated kernel keeps it one statement, so a
# function-shaped KernelSpec with several WANT ports is spliced at that
# boundary, exactly as in a hand-written `@kernel`.
module Models
using ReactiveKernels, ReactiveKernelsPPL
@kernel moments(x, g) = begin
    scaled = plate(x, Ref(g)) do xi, g
        return xi * g
    end
    path = scan(scaled; init=0.0) do carry, s
        next = carry + s
        (next, next)
    end
    total = sum(scaled)
    return path, total
end
const CALLS = Ref(0)
function parts(x)
    CALLS[] += 1
    return (sum(x), 2 .* x)
end
@rkppl moment_sum(x, g) = begin
    (path, total) = moments(x, g)
    path .+ total
end
end

const BACKEND = AutoEnzyme(; mode=Enzyme.Reverse)

scan_count(program) = count(entry -> entry.kind === :scan, recipe_inventory(program))
# Generated statements that call `head`, however they are spelled.
function call_count(def, head::Symbol)
    n = 0
    walk(ex) = ex isa Expr ? (ex.head === :call && ex.args[1] == GlobalRef(Models, head) &&
        (n += 1); foreach(walk, ex.args)) : nothing
    walk(def)
    return n
end

function build(ast, data)
    bound = bind_data(lower_rkppl(ast, data; conditioned=(:y,), mod=Models), data)
    return bound, build_kernel(bound)
end

function case(n)
    x = sin.(1:n)
    (; x, y=cos.(1:n) .* 0.2)
end

# mu = cumsum(a x) + sum(a x) = a w for the full read; a cumsum(x) for the path.
weights(data; total=true) = cumsum(data.x) .+ (total ? sum(data.x) : 0.0)
reference(data, a; total=true) = logpdf(Normal(0, 0.7), a) +
    sum(logpdf.(Normal.(a .* weights(data; total), 0.8), data.y))
gradient(data, a; total=true) = -a / 0.7^2 +
    sum((data.y .- a .* weights(data; total)) .* weights(data; total)) / 0.8^2

function check_sampler(bound, built, data; total=true)
    sampler = prepare_sampler(built, bound, [0.2]; backend=BACKEND)
    for a in (0.2, -0.4)
        u = [a]
        value, grad = sampler_value_and_gradient!(sampler, similar(u), u)
        @test value ≈ reference(data, a; total) rtol=1e-12
        @test grad ≈ [gradient(data, a; total)] rtol=1e-10 atol=1e-12
        @test u == [a]
    end
end

@testset "destructuring a multi-output KernelSpec splices its WANT boundary" begin
    ast = quote
        a ~ Normal(0, 0.7)
        (path, total) = moments(x, a)
        y .~ Normal.(path .+ total, 0.8)
    end
    counts = Int[]
    for n in (1, 3, 17)
        data = case(n)
        original = deepcopy(data)
        bound, built = build(ast, data)
        def = kernel_expr(bound, built.layout)
        # One destructuring statement, as authored: no runtime KernelSpec call.
        @test call_count(def, :moments) == 1
        @test any(st -> Meta.isexpr(st, :(=)) && st.args[1] == :((path, total)),
            def.args[2].args)
        @test scan_count(built.spec) == 1
        replayed = ReactiveKernelsPPL._eval_kernel_def(def)
        @test scan_count(replayed) == 1
        push!(counts, length(built.spec.graph.recipes))
        check_sampler(bound, built, data)
        @test data == original
    end
    @test all(==(first(counts)), counts)
end

@testset "a partial destructuring keeps the whole WANT boundary" begin
    ast = quote
        a ~ Normal(0, 0.7)
        (path,) = moments(x, a)
        y .~ Normal.(path, 0.8)
    end
    data = case(5)
    bound, built = build(ast, data)
    @test call_count(kernel_expr(bound, built.layout), :moments) == 1
    @test scan_count(built.spec) == 1
    check_sampler(bound, built, data; total=false)
end

@testset "a destructured ordinary function runs once per evaluation" begin
    ast = quote
        a ~ Normal(0, 0.7)
        w = a .* x
        (s, v) = parts(w)
        y .~ Normal.(v ./ 2 .+ s .- sum(w), 0.8)
    end
    data = case(4)
    bound, built = build(ast, data)
    @test call_count(kernel_expr(bound, built.layout), :parts) == 1
    sampler = prepare_sampler(built, bound, [0.2]; backend=BACKEND)
    for a in (0.2, -0.4)
        Models.CALLS[] = 0
        value = Base.invokelatest(sampler.kernel, [a])
        @test Models.CALLS[] == 1
        expected = logpdf(Normal(0, 0.7), a) + sum(logpdf.(Normal.(a .* data.x, 0.8), data.y))
        @test value ≈ expected rtol=1e-12
        _, grad = sampler_value_and_gradient!(sampler, [0.0], [a])
        @test grad ≈ [-a / 0.7^2 + sum((data.y .- a .* data.x) .* data.x) / 0.8^2] rtol=1e-10
    end
end

@testset "a data-only destructuring still runs once per bind" begin
    ast = quote
        a ~ Normal(0, 0.7)
        (s, v) = parts(x)
        y .~ Normal.(a .* (v ./ 2) .+ (s .- sum(x)), 0.8)
    end
    data = case(4)
    Models.CALLS[] = 0
    bound, built = build(ast, data)
    @test Models.CALLS[] == 1
    @test call_count(kernel_expr(bound, built.layout), :parts) == 0
    sampler = prepare_sampler(built, bound, [0.2]; backend=BACKEND)
    @test Models.CALLS[] == 1
    for a in (0.2, -0.4)
        expected = logpdf(Normal(0, 0.7), a) + sum(logpdf.(Normal.(a .* data.x, 0.8), data.y))
        value, grad = sampler_value_and_gradient!(sampler, [0.0], [a])
        @test value ≈ expected rtol=1e-12
        @test grad ≈ [-a / 0.7^2 + sum((data.y .- a .* data.x) .* data.x) / 0.8^2] rtol=1e-10
    end
    @test Models.CALLS[] == 1
end

@testset "destructuring in a submodel body splices the same boundary" begin
    ast = quote
        a ~ Normal(0, 0.7)
        mu ~ moment_sum(x, a)
        y .~ Normal.(mu, 0.8)
    end
    data = case(6)
    bound, built = build(ast, data)
    @test call_count(kernel_expr(bound, built.layout), :moments) == 1
    @test scan_count(built.spec) == 1
    check_sampler(bound, built, data)
end

end
