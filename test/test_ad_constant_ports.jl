module ADConstantPortsTests
using ReactiveKernels, DifferentiationInterface, Enzyme, Test

# WANT ports that no active HAVE port reaches have a zero derivative. A port
# that bound partial evaluation folds into a constant (a gather of bound data,
# here also as a view), one computed from inactive HAVE values alone, and an
# inactive HAVE port itself leave the differentiated call: pullbacks add
# nothing for them, pushforwards return zero tangents, and their values come
# from the same sweep (snag prepare-ad-pullb-8ca897b0). Plain Enzyme modes, no
# runtime activity; oracles are analytic.

const REVERSE = AutoEnzyme(; mode = Enzyme.Reverse)
const FORWARD = AutoEnzyme(; mode = Enzyme.Forward)

@kernel constant_ports(q::Vector{Float64}, s::Float64, loc::Vector{Float64},
                       rows::Vector{Int}, d::Vector{Float64}) = begin
    located = s .* q[rows]
    gathered = view(loc, rows)
    folded = loc[rows] .* 2.0
    shifted = d .+ 1.0
    total::Float64 = sum(abs2, q) + s
end

const HAVE = (:q, :s, :loc, :rows, :d)

# Port values and the pullback of each seeded port, in `q` and in `s`.
port_value(name, q, s, loc, rows, d) = (
    located = s .* q[rows], gathered = loc[rows], folded = 2 .* loc[rows],
    shifted = d .+ 1.0, d = d, total = sum(abs2, q) + s)[name]
function vjp(name, seed, q, s, rows)
    name === :located || name === :total || return (zero(q), 0.0)
    name === :total && return (2 .* q .* seed, seed)
    cotangent = zero(q)
    cotangent[rows] .+= s .* seed
    cotangent, sum(seed .* q[rows])
end
seed_for(value::AbstractArray) = collect(1.0:length(value)) ./ 4
seed_for(::Number) = 0.75

@testset "WANT ports that no active port reaches" begin
    q = [0.1, -0.2, 0.3, 0.4, -0.5]
    s = 1.5
    loc = [1.0, 2.0, 3.0, 4.0, 5.0]
    rows = [2, 3, 5]
    d = [1.0, -1.0, 2.0]
    loc_before = copy(loc)
    wants = ((:located, :gathered), (:gathered,), (:gathered, :folded),
             (:located, :d), (:located, :folded, :shifted, :total))

    @testset "pullbacks over $want" for want in wants
        selected = length(want) == 1 ? only(want) : want
        kernel = prepare(constant_ports; have = HAVE, want = selected,
                         bound = (; loc, rows))
        primal = kernel(q, s, d)
        ports = length(want) == 1 ? (primal,) : primal
        seeds = map(seed_for, ports)
        seed = length(want) == 1 ? only(seeds) : seeds
        expected_q = sum(first(vjp(name, port_seed, q, s, rows))
                         for (name, port_seed) in zip(want, seeds))
        expected_s = sum(last(vjp(name, port_seed, q, s, rows))
                         for (name, port_seed) in zip(want, seeds))

        prepared = prepare_ad_pullback(kernel, REVERSE, seed, q, s, d;
                                       active = :q)
        @test ad_pullback(prepared, seed, q, s, d) ≈ expected_q
        # The values of the differentiated sweep, the constant ports' included.
        value, cotangent = ad_value_and_pullback(prepared, seed, q, s, d)
        @test cotangent ≈ expected_q
        @test all(map(==, length(want) == 1 ? (value,) : value, ports))
        destination = fill(NaN, length(q))
        value, returned = ad_value_and_pullback!(
            prepared, destination, seed, q, s, d)
        @test returned === destination
        @test destination ≈ expected_q
        @test ad_pullback(kernel, REVERSE, seed, q, s, d; active = :q) ≈ expected_q

        both = prepare_ad_pullback(kernel, REVERSE, seed, q, s, d;
                                   active = (:q, :s))
        cotangent_q, cotangent_s = ad_pullback(both, seed, q, s, d)
        @test cotangent_q ≈ expected_q
        @test cotangent_s ≈ expected_s

        # A backend that does not keep its primal sweep evaluates the values
        # separately.
        forward_pullback = prepare_ad_pullback(kernel, FORWARD, seed, q, s, d;
                                               active = :q)
        value, cotangent = ad_value_and_pullback(forward_pullback, seed, q, s, d)
        @test cotangent ≈ expected_q
        @test all(map(==, length(want) == 1 ? (value,) : value, ports))
    end

    @testset "pushforwards over $want" for want in wants
        selected = length(want) == 1 ? only(want) : want
        kernel = prepare(constant_ports; have = HAVE, want = selected,
                         bound = (; loc, rows))
        primal = kernel(q, s, d)
        v = [1.0, 0.5, -2.0, 0.25, 3.0]
        tangents = map(want) do name
            name === :located ? s .* v[rows] :
            name === :total ? 2 * sum(q .* v) :
            zero(port_value(name, q, s, loc, rows, d))
        end
        prepared = prepare_ad_pushforward(kernel, FORWARD, (v,), q, s, d;
                                          active = :q)
        value, (tangent,) = ad_value_and_pushforward(prepared, (v,), q, s, d)
        @test all(map(==, length(want) == 1 ? (value,) : value,
                      length(want) == 1 ? (primal,) : primal))
        @test all(map(≈, length(want) == 1 ? (tangent,) : tangent, tangents))
        (tangent,) = ad_pushforward(prepared, (v,), q, s, d)
        @test all(map(≈, length(want) == 1 ? (tangent,) : tangent, tangents))
    end

    @testset "values of inactive ports follow each call" begin
        kernel = prepare(constant_ports; have = HAVE,
                         want = (:located, :shifted), bound = (; loc, rows))
        seed = ([1.0, 2.0, 3.0], [1.0, 1.0, 1.0])
        prepared = prepare_ad_pullback(kernel, REVERSE, seed, q, s, d;
                                       active = :q)
        for (newq, newd) in (([0.2, 0.1, 0.0, -0.3, 0.6], [4.0, 5.0, 6.0]),
                             ([1.0, 2.0, 3.0, 4.0, 5.0], [-1.0, 0.0, 1.0]))
            value, cotangent = ad_value_and_pullback(prepared, seed, newq, s, newd)
            @test value[1] ≈ s .* newq[rows]
            @test value[2] == newd .+ 1.0
            @test cotangent ≈ first(vjp(:located, seed[1], newq, s, rows))
        end
    end

    @testset "unbound views of inactive HAVE values" begin
        seed = ([1.0, 2.0, 3.0], [1.0, 1.0, 1.0])
        prepared = prepare_ad_pullback(constant_ports, REVERSE, seed,
                                       q, s, loc, rows, d; active = :q,
                                       want = (:located, :gathered))
        value, cotangent = ad_value_and_pullback(prepared, seed,
                                                 q, s, loc, rows, d)
        @test value[1] ≈ s .* q[rows]
        @test value[2] == loc[rows]
        @test cotangent ≈ first(vjp(:located, seed[1], q, s, rows))
        constant = prepare_ad_pullback(constant_ports, REVERSE, seed[2],
                                       q, s, loc, rows, d; active = :q,
                                       want = :gathered)
        @test ad_pullback(constant, seed[2], q, s, loc, rows, d) == zero(q)
    end

    @testset "retained WANT outputs" begin
        kernel = prepare(constant_ports; have = HAVE,
                         want = (:total, :gathered, :shifted, :located),
                         bound = (; loc, rows))
        prepared = prepare_ad(kernel, REVERSE, q, s, d; active = :q,
                              retain = (:gathered, :shifted, :located))
        for newd in (d, [5.0, 6.0, 7.0])
            value, gradient, retained = ad_value_gradient_and_retained!(
                prepared, similar(q), q, s, newd)
            @test value ≈ sum(abs2, q) + s
            @test gradient ≈ 2 .* q
            @test keys(retained) == (:gathered, :shifted, :located)
            @test retained.gathered == loc[rows]
            @test retained.shifted == newd .+ 1.0
            @test retained.located ≈ s .* q[rows]
            _, _, retained_o = ad_value_gradient_and_retained(prepared, q, s, newd)
            @test retained_o.shifted == newd .+ 1.0
        end
    end

    @testset "bound data and the prepared kernel stay unchanged" begin
        kernel = prepare(constant_ports; have = HAVE, want = :gathered,
                         bound = (; loc, rows))
        prepared = prepare_ad_pullback(kernel, REVERSE, [1.0, 2.0, 3.0],
                                       q, s, d; active = :q)
        for _ in 1:3
            @test ad_pullback(prepared, [10.0, 20.0, 30.0], q, s, d) == zero(q)
        end
        @test loc == loc_before
        @test kernel(q, s, d) == loc[rows]
    end

    # refused: a cotangent pairs with a port value of its own structure, so a
    # misshapen one has no meaning, whether or not the port reads the point.
    @testset "a misshapen seed of a constant port names the boundary" begin
        kernel = prepare(constant_ports; have = HAVE, want = :gathered,
                         bound = (; loc, rows))
        @test_throws ArgumentError prepare_ad_pullback(
            kernel, REVERSE, [1.0], q, s, d; active = :q)
    end
end

end
