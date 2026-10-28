using DifferentiationInterface
using Distributions
using Enzyme
using LinearAlgebra
using ReactiveKernels
using ReactiveKernelsPPL
using Test

_apd_data(n = 8, K = 3) = Dict(:y => [0.2 * sin(i) for i in 1:n],
    :k => [mod1(i, K) for i in 1:n])

function _apd_build(ast, data; values = false)
    plan = lower_rkppl(ast, values ? data : keys(data); conditioned = (:y,))
    bound = bind_data(plan, data)
    return (; plan, bound, built = build_kernel(bound), data)
end

function _apd_difference(f, u; h = cbrt(eps(Float64)))
    map(eachindex(u)) do i
        up, dn = copy(u), copy(u)
        up[i] += h
        dn[i] -= h
        (f(up) - f(dn)) / (2h)
    end
end

function _apd_check(ast, data, prior, locations; values = false)
    original = deepcopy(data)
    fx = _apd_build(ast, data; values)
    @test fx.bound.n_obs == length(data[:y])
    u = [0.25 * cos(i) for i in 1:fx.built.layout.total]
    function oracle(u)
        th = constrain(fx.built.layout, u)
        likelihood = sum(logpdf.(Normal.(locations(th)[data[:k]], 0.7), data[:y]))
        return prior(th) + likelihood + logjac(fx.built.layout, u)
    end
    th = constrain(fx.built.layout, u)
    query = prepare_query(fx.built, fx.bound, :prior)
    @test Base.invokelatest(query, u) ≈ prior(th)
    q = prepare_sampler(fx.built, fx.bound, u;
        backend = AutoEnzyme(; mode = Enzyme.Reverse))
    value, grad = sampler_value_and_gradient!(q, similar(u), u)
    @test value ≈ oracle(u)
    @test grad ≈ _apd_difference(oracle, u) rtol = 1e-5 atol = 1e-7
    @test data == original
    return fx
end

@testset "whole data arguments of declared-array priors" begin
    mu0 = [0.2, -0.3, 0.5]
    L = [1.0 0.0 0.0; 0.2 0.8 0.0; -0.1 0.3 1.2]
    S0 = [1.0 0.2; 0.2 0.8]
    alpha = [1.2, 2.1, 0.8]
    for values in (false, true)
        _apd_check(:(begin
            z[1:3] .~ Normal.(mu0, 1)
            y .~ Normal.(z[k], 0.7)
        end), merge(_apd_data(), Dict(:mu0 => mu0)),
            th -> sum(logpdf.(Normal.(mu0, 1), th.z)), th -> th.z; values)
        _apd_check(:(begin
            eachrow(B[levels(k), 1:3]) .~ MvNormalCholesky(mu0, L)
            y .~ Normal.(B[k, 1], 0.7)
        end), merge(_apd_data(), Dict(:mu0 => mu0, :L => L)),
            th -> sum(logpdf(MvNormal(mu0, L * L'), row) for row in eachrow(th.B)),
            th -> th.B[:, 1]; values)
        _apd_check(:(begin
            eachrow(B[levels(k), 1:2]) .~ MvNormal(zeros(2), S0)
            y .~ Normal.(B[k, 1], 0.7)
        end), merge(_apd_data(), Dict(:S0 => S0)),
            th -> sum(logpdf(MvNormal(zeros(2), S0), row) for row in eachrow(th.B)),
            th -> th.B[:, 1]; values)
        _apd_check(:(begin
            eachrow(P[levels(k), 1:3]) .~ Dirichlet(alpha)
            y .~ Normal.(P[k, 1], 0.7)
        end), merge(_apd_data(), Dict(:alpha => alpha)),
            th -> sum(logpdf(Dirichlet(alpha), row) for row in eachrow(th.P)),
            th -> th.P[:, 1]; values)
        # A names-only plan can bind either a shared scalar or an element vector.
        _apd_check(:(begin
            z[1:3] .~ Normal.(mu0, 1)
            y .~ Normal.(z[k], 0.7)
        end), merge(_apd_data(), Dict(:mu0 => 0.2)),
            th -> sum(logpdf.(Normal(0.2, 1), th.z)), th -> th.z; values)
    end
end

@testset "array prior data retains its value through definitions" begin
    mu0 = [0.2, -0.3, 0.5]
    for argument in (:m0, :(mu0 .+ a))
        ast = quote
            a ~ Normal(0, 1)
            m0 = mu0 .+ a
            z[1:3] .~ Normal.($argument, 1)
            y .~ Normal.(z[k], 0.7)
        end
        _apd_check(ast, merge(_apd_data(), Dict(:mu0 => mu0)),
            th -> logpdf(Normal(0, 1), th.a) +
                sum(logpdf.(Normal.(mu0 .+ th.a, 1), th.z)), th -> th.z)
    end
end

@testset "array prior data keeps shape and observation checks" begin
    ast = :(begin
        z[1:3] .~ Normal.(mu0, 1)
        y .~ Normal.(z[k], 0.7)
    end)
    plan = lower_rkppl(ast, (:y, :k, :mu0); conditioned = (:y,))
    # Refused: per-element prior arguments must match the declared array length (§3).
    @test_throws ContractValidationError bind_data(plan,
        merge(_apd_data(), Dict(:mu0 => [0.2, 0.5])))
    # Rebinding the same plan preserves both shared and per-element semantics.
    for mean in (0.2, [0.2, -0.3, 0.5])
        @test bind_data(plan, merge(_apd_data(), Dict(:mu0 => mean))).n_obs == 8
    end
    dual = :(begin
        z[1:3] .~ Normal.(mu0, 1)
        eta = z[k] .+ mu0
        y .~ Normal.(eta, 0.7)
    end)
    aligned = lower_rkppl(dual, (:y, :k, :mu0); conditioned = (:y,))
    # Refused: a value also read per observation must match that observation axis.
    @test_throws ContractValidationError bind_data(aligned,
        merge(_apd_data(), Dict(:mu0 => [0.2, -0.3, 0.5])))
end

# Two-axis elementwise priors read a per-element argument of the declared
# array's size, one value per element, as one-axis declarations read a
# vector (rkppl-use §3).
const _APD_LOC = [0.2 -0.4; -0.3 0.1; 0.5 0.0]
const _APD_SC = [1.0 0.5; 0.8 1.2; 0.6 0.9]

@testset "per-element arguments of two-axis array priors" begin
    prior(loc, sc) = th -> sum(logpdf.(Normal.(loc, sc), th.B))
    column(j) = th -> th.B[:, j]
    data = merge(_apd_data(), Dict(:LOC => _APD_LOC, :SC => _APD_SC))
    for values in (false, true)
        _apd_check(:(begin
            B[1:3, 1:2] .~ Normal.(LOC, SC)
            y .~ Normal.(B[k, 1], 0.7)
        end), data, prior(_APD_LOC, _APD_SC), column(1); values)
        # A per-element matrix beside a shared number.
        _apd_check(:(begin
            B[1:3, 1:2] .~ Normal.(LOC, 0.5)
            y .~ Normal.(B[k, 2], 0.7)
        end), data, prior(_APD_LOC, 0.5), column(2); values)
    end
    # Literal matrices, `hcat` of literal columns, integer entries and a
    # levels axis.
    _apd_check(:(begin
        B[1:3, 1:2] .~ Normal.([0.2 -0.4; -0.3 0.1; 0.5 0.0],
            [1.0 0.5; 0.8 1.2; 0.6 0.9])
        y .~ Normal.(B[k, 1], 0.7)
    end), _apd_data(), prior(_APD_LOC, _APD_SC), column(1))
    _apd_check(:(begin
        B[1:3, 1:2] .~ Normal.(hcat([0.2, -0.3, 0.5], [-0.4, 0.1, 0.0]), SC)
        y .~ Normal.(B[k, 1], 0.7)
    end), data, prior(_APD_LOC, _APD_SC), column(1))
    _apd_check(:(begin
        B[levels(k), 1:2] .~ Normal.([0 1; 2 0; 1 1], 1)
        y .~ Normal.(B[k, 2], 0.7)
    end), _apd_data(), prior([0 1; 2 0; 1 1], 1), column(2))
    # A positive family keeps its transform and Jacobian.
    _apd_check(:(begin
        B[1:3, 1:2] .~ Gamma.(SC .+ 1, LOC .+ 1)
        y .~ Normal.(B[k, 1], 0.7)
    end), data, th -> sum(logpdf.(Gamma.(_APD_SC .+ 1, _APD_LOC .+ 1), th.B)),
        column(1))
    # Live arguments: a computed matrix and a declared two-axis array.
    _apd_check(:(begin
        a ~ Normal(0, 1)
        M = LOC .+ a
        B[1:3, 1:2] .~ Normal.(M, SC)
        y .~ Normal.(B[k, 1], 0.7)
    end), data, th -> logpdf(Normal(0, 1), th.a) +
        sum(logpdf.(Normal.(_APD_LOC .+ th.a, _APD_SC), th.B)), column(1))
    _apd_check(:(begin
        Z[1:3, 1:2] .~ Normal.(0, 1)
        B[1:3, 1:2] .~ Normal.(Z, SC)
        y .~ Normal.(B[k, 1], 0.7)
    end), data, th -> sum(logpdf.(Normal(0, 1), th.Z)) +
        sum(logpdf.(Normal.(th.Z, _APD_SC), th.B)), column(1))
end

@testset "two-axis per-element priors equal per-column declarations" begin
    data = merge(_apd_data(), Dict(:LOC => _APD_LOC, :SC => _APD_SC,
        :l1 => _APD_LOC[:, 1], :l2 => _APD_LOC[:, 2],
        :s1 => _APD_SC[:, 1], :s2 => _APD_SC[:, 2]))
    joint = _apd_build(:(begin
        B[1:3, 1:2] .~ Normal.(LOC, SC)
        y .~ Normal.(B[k, 1] .- B[k, 2], 0.7)
    end), data)
    cols = _apd_build(:(begin
        b1[1:3] .~ Normal.(l1, s1)
        b2[1:3] .~ Normal.(l2, s2)
        y .~ Normal.(b1[k] .- b2[k], 0.7)
    end), data)
    # Column-major packing: B.i.j runs down column 1, then column 2, exactly
    # as b1 then b2.
    @test coordinate_names(joint.built.layout) ==
        Symbol.(["B.$i.$j" for j in 1:2 for i in 1:3])
    u = [0.3 * sin(i) for i in 1:6]
    for preset in (:prior, :likelihood, :sampler)
        qj = prepare_query(joint.built, joint.bound, preset)
        qc = prepare_query(cols.built, cols.bound, preset)
        @test Base.invokelatest(qj, u) ≈ Base.invokelatest(qc, u)
    end
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    gj = sampler_value_and_gradient!(prepare_sampler(joint.built, joint.bound,
        u; backend), similar(u), u)[2]
    gc = sampler_value_and_gradient!(prepare_sampler(cols.built, cols.bound,
        u; backend), similar(u), u)[2]
    @test gj ≈ gc
end

@testset "per-target coefficient priors of a joint design product" begin
    n = 12
    x1 = [0.4 * sin(i) for i in 1:n]
    x2 = [0.3 * cos(2i) for i in 1:n]
    X = hcat(ones(n), x1, x2)
    Y = [0.1 * sin(i + j) + 0.2j for i in 1:n, j in 1:3]
    LOC = [2.3 -1.41 0.0; 0.0 0.0 0.5; 0.0 0.0 0.0]
    SCALE = [1.0 2.0 1.0; 0.5 0.5 1.0; 0.5 0.5 1.0]
    data = Dict(:x1 => x1, :x2 => x2, :Y => Y, :LOC => LOC, :SCALE => SCALE)
    function oracle(th)
        sum(logpdf.(Normal.(LOC, SCALE), th.B)) +
            logpdf(Exponential(1.0), th.sigma) +
            sum(logpdf.(Normal.(X * th.B, th.sigma), Y))
    end
    for loc in (:LOC, :([2.3 -1.41 0.0; 0.0 0.0 0.5; 0.0 0.0 0.0]))
        ast = :(begin
            X = hcat(ones(length(x1)), x1, x2)
            B[axes(X, 2), 1:3] .~ Normal.($loc, SCALE)
            sigma ~ Exponential(1.0)
            Y .~ Normal.(X * B, sigma)
        end)
        bound = bind_data(lower_rkppl(ast, data; conditioned = (:Y,)), data)
        built = build_kernel(bound)
        @test built.layout.total == 10
        u = [0.2 * cos(i) for i in 1:built.layout.total]
        th = constrain(built.layout, u)
        full(u) = (t = constrain(built.layout, u); oracle(t) + logjac(built.layout, u))
        q = prepare_sampler(built, bound, u;
            backend = AutoEnzyme(; mode = Enzyme.Reverse))
        value, grad = sampler_value_and_gradient!(q, similar(u), u)
        @test value ≈ full(u)
        @test grad ≈ _apd_difference(full, u) rtol = 1e-5 atol = 1e-7
        @test data[:LOC] == LOC
    end
end

@testset "two-axis per-element prior shapes follow Julia" begin
    ast = :(begin
        B[1:3, 1:2] .~ Normal.(M, 1)
        y .~ Normal.(B[k, 1], 0.7)
    end)
    plan = lower_rkppl(ast, (:y, :k, :M); conditioned = (:y,))
    # Refused: Julia cannot broadcast a 2×2 argument over a 3×2 array
    # (principle 3, standard-Julia semantics).
    err = try
        bind_data(plan, merge(_apd_data(), Dict(:M => ones(2, 2))))
        nothing
    catch e
        e
    end
    @test err isa ContractValidationError
    @test occursin("has size (3, 2) but its prior arg1 has size (2, 2)",
        sprint(showerror, err))
    # Refused: a 2×3 value cannot broadcast over the 3×2 array either
    # (principle 3). A data-only definition is checked at binding; a value
    # the graph computes is checked where it is computed.
    data = merge(_apd_data(), Dict(:LOC => _APD_LOC))
    @test_throws ContractValidationError _apd_build(:(begin
        M = permutedims(LOC)
        B[1:3, 1:2] .~ Normal.(M, 1)
        y .~ Normal.(B[k, 1], 0.7)
    end), data)
    fx = _apd_build(:(begin
        a ~ Normal(0, 1)
        M = permutedims(LOC) .+ a
        B[1:3, 1:2] .~ Normal.(M, 1)
        y .~ Normal.(B[k, 1], 0.7)
    end), data)
    q = prepare_query(fx.built, fx.bound, :prior)
    @test_throws DimensionMismatch Base.invokelatest(q, zeros(fx.built.layout.total))
    # Not built yet: Julia broadcasts a 3-vector over the rows of a 3×2
    # array (one value per row); per-element arguments currently have the
    # array's own size. Capability gap, not a refusal.
    @test_broken try
        bind_data(plan, merge(_apd_data(), Dict(:M => [0.1, 0.2, 0.3])))
        true
    catch
        false
    end
end

@testset "per-slice concentrations and means from data matrices" begin
    A = [1.0 2.0 1.0; 1.0 1.0 3.0]
    M = [0.1 -0.2 0.3; 0.4 0.0 -0.1]
    L = [1.0 0.0; 0.3 0.8]
    data = merge(_apd_data(), Dict(:A => A, :At => permutedims(A), :M => M, :L => L))
    for values in (false, true)
        _apd_check(:(begin
            eachcol(S[1:2, levels(k)]) .~ Dirichlet.(eachcol(A))
            y .~ Normal.(S[1, k], 0.7)
        end), data,
            th -> sum(logpdf(Dirichlet(A[:, j]), th.S[:, j]) for j in 1:3),
            th -> th.S[1, :]; values)
        _apd_check(:(begin
            eachrow(S[levels(k), 1:2]) .~ Dirichlet.(eachrow(At))
            y .~ Normal.(S[k, 1], 0.7)
        end), data,
            th -> sum(logpdf(Dirichlet(A[:, j]), th.S[j, :]) for j in 1:3),
            th -> th.S[:, 1]; values)
        _apd_check(:(begin
            eachcol(B[1:2, levels(k)]) .~ MvNormalCholesky.(eachcol(M), Ref(L))
            y .~ Normal.(B[2, k], 0.7)
        end), data,
            th -> sum(logpdf(MvNormal(M[:, j], L * L'), th.B[:, j]) for j in 1:3),
            th -> th.B[2, :]; values)
    end
end
