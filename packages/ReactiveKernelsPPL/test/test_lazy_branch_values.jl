module LazyBranchValueTests
using ReactiveKernels, ReactiveKernelsPPL, Distributions, DifferentiationInterface, Enzyme, Test

# A definition's value may be a lazy branch, as in Julia: `c ? a : b`,
# `if`/`elseif`/`else`, `&&` and `||`, at model scope as in a plate cell.
# Only the arm the condition selects runs, and types are values
# (`eltype(cells) <: Number`, `convert(Float64, a)`). So the flatten of
# reactivekernels-use §4c is written in the model itself and keeps its lane
# slot in a position-batched reader (snag brm-emitting-rk-d04ac327).

const BACKEND = AutoEnzyme(; mode=Enzyme.Reverse)

const x = [0.3, -0.5, 1.2, 0.8, -0.1]
const y = [0.1, -0.2, 0.9, 0.4, 0.0]

function differences(f, u)
    h = cbrt(eps(Float64))
    [begin
        up, down = copy(u), copy(u)
        up[i] += h
        down[i] -= h
        (f(up)-f(down))/(2h)
    end for i in eachindex(u)]
end

lowered(ast, data; values = false) = lower_rkppl(ast, values ? data : Tuple(keys(data));
    mod=@__MODULE__, conditioned=(:y,))

# Posterior value, ordinary Enzyme reverse and the printed-source replay
# against `oracle(p, data)`, an independent Distributions density of the
# constrained values `p`, at points on both sides of zero.
function check(plan, data, oracle)
    original = deepcopy(data)
    bound = bind_data(plan, data)
    built = build_kernel(bound)
    sampler = prepare_sampler(built, bound, zeros(built.layout.total); backend=BACKEND)
    replayed = ReactiveKernelsPPL._eval_kernel_def(kernel_expr(bound, built.layout))
    replay = prepare_query((; spec=replayed, layout=built.layout), bound, :sampler)
    target(u) = oracle(constrain(built.layout, u), data) + logjac(built.layout, u)
    for shift in (0.0, -0.9, 0.7)
        u = [0.3sin(i) + shift for i in 1:built.layout.total]
        saved = copy(u)
        value, grad = sampler_value_and_gradient!(sampler, similar(u), u)
        @test value ≈ target(u)
        @test Base.invokelatest(replay, u) ≈ target(u)
        @test grad ≈ differences(target, u) rtol=1e-5 atol=1e-7
        @test u == saved
    end
    @test data == original
    return built
end

folded(a) = a > 0 ? a : -a
folded_oracle(p, d) = logpdf(Normal(0, 1), p.a) + logpdf(Exponential(1), p.sigma) +
    sum(logpdf.(Normal.(folded(p.a), p.sigma), d.y))

@rkppl folded_value(a) = begin
    m = a > 0 ? a : -a
    return m
end

@testset "a lazy branch on a parameter at model scope" begin
    data = (; y)
    spellings = (
        :(m = a > 0 ? a : -a),
        :(m = if a > 0
            a
        else
            -a
        end),
        :(m = if a > 0
            a
        elseif a == 0
            0.0
        else
            -a
        end),
        # An arm's locals are read by its later statements.
        :(m = if a > 0
            t = 2a
            t - a
        else
            -a
        end),
        :(m = (a > 0 || a == 0) && true ? a : -a),
        :(m = 2 * (a > 0 ? a : -a) / 2),
        :(m ~ folded_value(a)),
    )
    for def in spellings
        ast = quote
            a ~ Normal(0, 1)
            sigma ~ Exponential(1)
            $def
            y .~ Normal.(m, sigma)
        end
        check(lowered(ast, data), data, folded_oracle)
    end
    # Written inline in an observation's or a prior's argument, a branch is
    # its named definition.
    inline = quote
        a ~ Normal(0, 1)
        sigma ~ Exponential(1)
        y .~ Normal.(a > 0 ? a : -a, sigma)
    end
    check(lowered(inline, data), data, folded_oracle)
    prior = quote
        sigma ~ Exponential(1)
        a ~ Normal(0, sigma > 1 ? 2.0 : 1.0)
        y .~ Normal.(a, sigma)
    end
    check(lowered(prior, data), data, (p, d) -> logpdf(Exponential(1), p.sigma) +
        logpdf(Normal(0, p.sigma > 1 ? 2.0 : 1.0), p.a) +
        sum(logpdf.(Normal.(p.a, p.sigma), d.y)))
end

@testset "a branch over observation columns" begin
    data = (; x, y)
    shifted_oracle(p, d) = logpdf(Normal(0, 1), p.a) + logpdf(Exponential(1), p.sigma) +
        sum(logpdf.(Normal.(p.a > 0 ? p.a .+ d.x : p.a .- d.x, p.sigma), d.y))
    for def in (:(mu = a > 0 ? a .+ x : a .- x),
            :(mu = typeof(x) <: Vector{Float64} && a > 0 ? a .+ x : a .- x))
        ast = quote
            a ~ Normal(0, 1)
            sigma ~ Exponential(1)
            $def
            y .~ Normal.(mu, sigma)
        end
        check(lowered(ast, data), data, shifted_oracle)
    end
end

@testset "a data-only branch runs once, at binding, and only its selected arm" begin
    data = (; x, y)
    weight(w) = (p, d) -> logpdf(Normal(0, 1), p.a) + logpdf(Exponential(1), p.sigma) +
        sum(logpdf.(Normal.(p.a .+ w .* d.x, p.sigma), d.y))
    cases = (
        (:(w = isempty(x) ? 0.0 : 2.0), 2.0),
        (:(w = length(x) > 1 ? 0.5 : 2.0), 0.5),
        # The unselected arm would throw a `BoundsError`.
        (:(w = length(x) > 10 ? x[11] : 0.5), 0.5),
        (:(w = eltype(x) <: Real ? 1.5 : 0.0), 1.5),
        (:(w = typeof(x) <: AbstractVector{<:Real} ? 1.5 : 0.0), 1.5),
        (:(w = Float64 >: eltype(x) ? 1.5 : 0.0), 1.5),
        (:(w = (isempty(x) || length(x) > 3) && true ? 1.5 : 0.0), 1.5),
        (:(w = if isempty(x)
            0.0
        else
            n = length(x)
            2n / length(x)
        end), 2.0),
    )
    for (def, w) in cases, values in (false, true)
        ast = quote
            a ~ Normal(0, 1)
            sigma ~ Exponential(1)
            $def
            mu = a .+ w .* x
            y .~ Normal.(mu, sigma)
        end
        plan = lowered(ast, data; values)
        bound = bind_data(plan, data)
        @test bound.columns[:w] == w
        check(plan, data, weight(w))
    end
end

@testset "an unselected arm never runs in the kernel" begin
    # `x[11]` is out of bounds for the five-entry `x`.
    data = (; x, y)
    ast = quote
        a ~ Normal(0, 1)
        sigma ~ Exponential(1)
        m = a > 100 ? x[11] : (a > 0 ? a : -a)
        y .~ Normal.(m, sigma)
    end
    check(lowered(ast, data), data, folded_oracle)
end

@testset "types are values" begin
    data = (; y)
    plain_oracle(p, d) = logpdf(Normal(0, 1), p.a) + logpdf(Exponential(1), p.sigma) +
        sum(logpdf.(Normal.(p.a, p.sigma), d.y))
    for def in (:(m = convert(Float64, a)), :(m = first(zeros(Float64, 2)) + a),
            :(m = a isa Real ? a : 0.0), :(m = typeof(a) <: Real ? a : 0.0))
        ast = quote
            a ~ Normal(0, 1)
            sigma ~ Exponential(1)
            $def
            y .~ Normal.(m, sigma)
        end
        check(lowered(ast, data), data, plain_oracle)
    end
end

@testset "a branch local is a new name" begin
    # refused: an arm assigning `sigma` would rebind the model's `sigma`, and
    # a name is assigned once (rkppl-use §9, single assignment).
    ast = quote
        a ~ Normal(0, 1)
        sigma ~ Exponential(1)
        m = if a > 0
            sigma = 2a
            sigma - a
        else
            -a
        end
        y .~ Normal.(m, sigma)
    end
    err = try
        bind_data(lowered(ast, (; y)), (; y))
        nothing
    catch e
        e
    end
    @test err isa ReactiveKernelsPPL.ContractValidationError
    @test occursin("a branch arm assigns sigma", sprint(showerror, err))
end

# reactivekernels-use §4c's flatten: number cells as they are, vector cells
# concatenated (a typed empty vector for no cells).
const FLATTENS = (
    :(loc = eltype(loc_cells) <: Number ? loc_cells :
        isempty(loc_cells) ? eltype(eltype(loc_cells))[] : reduce(vcat, loc_cells)),
    :(loc = if eltype(loc_cells) <: Number
        loc_cells
    elseif isempty(loc_cells)
        eltype(eltype(loc_cells))[]
    else
        reduce(vcat, loc_cells)
    end),
    :(loc = eltype(loc_cells) <: AbstractVector{<:Real} ?
        (isempty(loc_cells) ? eltype(eltype(loc_cells))[] : reduce(vcat, loc_cells)) :
        loc_cells),
)

flatten_model(cell, def) = quote
    log_k ~ Normal(0, 1)
    sigma ~ Exponential(1)
    @plate for i in eachindex(t)
        $cell
    end
    $def
    y .~ Normal.(loc, sigma)
end

const S = 16
const LENGTHS = [20 + 3 * (i % 5) for i in 1:S]
const T = [collect(range(0.1, 8.0; length=n)) for n in LENGTHS]
const DOSE = [1.0 + 0.01i for i in 1:S]
const VECTOR_CELLS = (:(loc_cells[i] = dose[i] .* exp.(-exp(log_k) .* t[i])),
    (; t=T, dose=DOSE, y=reduce(vcat, [0.3 .* exp.(-0.4 .* ti) for ti in T])),
    :(loc = reduce(vcat, loc_cells)),
    (p, d) -> reduce(vcat, [di .* exp.(-exp(p.log_k) .* ti) for (di, ti) in zip(d.dose, d.t)]))
const NUMBER_CELLS = (:(loc_cells[i] = dose[i] * exp(-exp(log_k) * t[i])),
    (; t=[0.1 + 0.05i for i in 1:4S], dose=repeat(DOSE, 4), y=[0.2 + 0.001i for i in 1:4S]),
    :(loc = loc_cells),
    (p, d) -> d.dose .* exp.(-exp(p.log_k) .* d.t))

bytes_per_call(reader, args...) = (reader(args...); reader(args...);
    minimum(@allocated(reader(args...)) for _ in 1:5))

# A warm borrowed reader of `loc` at `P` positions: its values against the
# unbatched cut's, and its bytes per call.
function batched_loc(built, bound, P)
    columns = NamedTuple(Symbol(k) => v for (k, v) in bound.columns)
    cut = ReactiveKernels.prepare(built.spec; have=(:unconstrained, keys(columns)...),
        want=(:loc,), bound=columns)
    reader = copy(ReactiveKernels.vectorize(cut; batched=:unconstrained, reuse=true))
    U = [0.1 * sin(j + 3k) for j in 1:built.layout.total, k in 1:P]
    single(u) = (v = Base.invokelatest(cut, u); v isa Tuple ? only(v) : v)
    stacked = Base.invokelatest(reader, U)
    stacked = stacked isa Tuple ? only(stacked) : stacked
    at(k) = stacked isa AbstractMatrix ? stacked[:, k] : stacked[k]
    ok = all(isequal(collect(at(k)), single(U[:, k])) for k in 1:P)
    return (; ok, bytes=Base.invokelatest(bytes_per_call, reader, U))
end

@testset "the flatten of reactivekernels-use §4c at model scope" begin
    for (cell, data, plain, cells) in (VECTOR_CELLS, NUMBER_CELLS)
        oracle(p, d) = logpdf(Normal(0, 1), p.log_k) + logpdf(Exponential(1), p.sigma) +
            sum(logpdf.(Normal.(cells(p, d), p.sigma), d.y))
        reference = let bound = bind_data(lowered(flatten_model(cell, plain), data;
                values=true), data)
            built = build_kernel(bound)
            (; built, bound, sampler=prepare_sampler(built, bound, zeros(2); backend=BACKEND))
        end
        flat = sizeof(Float64) * length(data.y)
        for def in FLATTENS
            plan = lowered(flatten_model(cell, def), data; values=true)
            built = check(plan, data, oracle)
            bound = bind_data(plan, data)
            # The same density and gradient as the model without the branch.
            sampler = prepare_sampler(built, bound, zeros(2); backend=BACKEND)
            for u in ([0.1, 0.2], [-0.4, 0.3])
                v, g = sampler_value_and_gradient!(sampler, similar(u), u)
                vr, gr = sampler_value_and_gradient!(reference.sampler, similar(u), u)
                @test v ≈ vr rtol=1e-14
                @test g ≈ gr rtol=1e-12
            end
            # The flat value keeps a lane slot: each of 24 more positions
            # costs less than one flat value. (The bounded condition
            # `AbstractVector{<:Real}` builds its `where` type at each
            # position, as Julia does in any loop: reactivekernels-use §4c.)
            few, many = batched_loc(built, bound, 8), batched_loc(built, bound, 32)
            @test few.ok && many.ok
            @test many.bytes - few.bytes < 24 * flat
        end
    end
end

end # module
