using ReactiveKernels, Test, DifferentiationInterface, LinearAlgebra
import Enzyme

# Prepared pushforwards (Jacobian-vector products) and Hessian-vector products
# share the prepared-gradient boundary: one active HAVE, every other HAVE
# rebound as a Constant per call. Oracles are analytic.

@kernel tangent_density(q::Vector{Float64}, s::Float64;
                        data::Vector{Float64}) = begin
    density::Float64 = sum(data .* exp.(s .* q)) - 0.5 * sum(abs2, q) - s^2
end

@kernel tangent_mean(q::Vector{Float64}, s::Float64;
                     data::Vector{Float64}) = begin
    mean = data .* exp.(s .* q)
end

# Per-group effects `z` never interact across groups, so the Hessian in `z`
# is block diagonal and one direction per within-group coordinate, summed
# across groups, recovers every block.
@kernel tangent_groups(z::Matrix{Float64}; w::Vector{Float64}) = begin
    density::Float64 = sum(exp.(z[:, 1] .+ w .* z[:, 2])) - 0.5 * sum(abs2, z)
end

_tangent_hessian(q, s, data) = Diagonal(data .* s^2 .* exp.(s .* q)) - I
_tangent_jacobian(q, s, data) = Diagonal(data .* s .* exp.(s .* q))
_tangent_unit(n, i) = (e = zeros(n); e[i] = 1.0; e)

@testset "prepared pushforwards and Hessian-vector products" begin
    second_order = SecondOrder(AutoEnzyme(; mode = Enzyme.Forward),
                               AutoEnzyme(; mode = Enzyme.Reverse))
    forward = AutoEnzyme(; mode = Enzyme.Forward)
    data = [2.0, -1.0, 0.5]
    q = [0.3, -0.4, 0.2]
    s = 0.7
    v = [1.0, 0.5, -2.0]

    @testset "Hessian-vector products rebind constants per call" begin
        prepared = prepare_ad_hvp(tangent_density, second_order, (v,), q, s;
                                  data, active = :q, want = :density)
        @test prepared isa PreparedADHVP
        @test prepared isa PreparedADOperator
        @test inputs(prepared) == inputs(tangent_density)
        @test occursin("PreparedADHVP(active=:q, want=:density",
                       sprint(show, prepared))
        @test only(ad_hvp(prepared, (v,), q, s; data)) ≈
              _tangent_hessian(q, s, data) * v
        for (newq, news, newdata) in (([0.1, 0.0, -0.1], 1.1, [1.0, 2.0, 3.0]),
                                      ([-0.5, 0.4, 0.3], 0.2, [0.5, 0.5, -4.0]))
            w = [-1.0, 2.0, 0.25]
            @test only(ad_hvp(prepared, (w,), newq, news; data = newdata)) ≈
                  _tangent_hessian(newq, news, newdata) * w
        end

        gradient, hvps = ad_gradient_and_hvp(prepared, (v,), q, s; data)
        @test gradient ≈ data .* s .* exp.(s .* q) .- q
        @test only(hvps) ≈ _tangent_hessian(q, s, data) * v

        results = (similar(q),)
        gradient_buffer = similar(q)
        @test ad_hvp!(prepared, results, (v,), q, s; data) === results
        @test only(results) ≈ _tangent_hessian(q, s, data) * v
        returned = ad_gradient_and_hvp!(prepared, gradient_buffer, results,
                                        (2v,), q, s; data)
        @test returned[1] === gradient_buffer
        @test gradient_buffer ≈ gradient
        @test only(results) ≈ 2 * _tangent_hessian(q, s, data) * v
    end

    @testset "batched directions and bound preparations" begin
        units = ntuple(i -> _tangent_unit(3, i), 3)
        batched = prepare_ad_hvp(tangent_density, second_order, units, q, s;
                                 data, active = :q, want = :density)
        @test reduce(hcat, ad_hvp(batched, units, q, s; data)) ≈
              _tangent_hessian(q, s, data)

        bound = prepare_ad_hvp(tangent_density, second_order, (v,), q, s;
                               active = :q, want = :density, bound = (; data))
        @test only(ad_hvp(bound, (v,), q, s)) ≈ _tangent_hessian(q, s, data) * v
        @test_throws ArgumentError prepare_ad_hvp(
            tangent_density, second_order, (v,), q, s;
            active = :q, want = :density, bound = (; data), extra = 1.0)
    end

    @testset "compressed block Hessian from colored directions" begin
        groups = 5
        z = [0.1 * sin(i + j) for i in 1:groups, j in 1:2]
        w = [0.3 * i for i in 1:groups]
        colors = ntuple(j -> (seed = zeros(groups, 2); seed[:, j] .= 1.0; seed), 2)
        prepared = prepare_ad_hvp(tangent_groups, second_order, colors, z;
                                  w, active = :z, want = :density)
        compressed = ad_hvp(prepared, colors, z; w)
        for group in 1:groups
            e = exp(z[group, 1] + w[group] * z[group, 2])
            block = [e - 1 w[group] * e; w[group] * e w[group]^2 * e - 1]
            @test [compressed[j][group, k] for k in 1:2, j in 1:2] ≈ block
        end
    end

    @testset "pushforwards of non-scalar WANTs" begin
        prepared = prepare_ad_pushforward(tangent_mean, forward, (v,), q, s;
                                          data, active = :q, want = :mean)
        @test prepared isa PreparedADPushforward
        @test occursin("PreparedADPushforward(active=:q, want=:mean",
                       sprint(show, prepared))
        @test only(ad_pushforward(prepared, (v,), q, s; data)) ≈
              _tangent_jacobian(q, s, data) * v
        value, jvps = ad_value_and_pushforward(prepared, (v,), q, s; data)
        @test value ≈ data .* exp.(s .* q)
        @test only(jvps) ≈ _tangent_jacobian(q, s, data) * v

        units = ntuple(i -> _tangent_unit(3, i), 3)
        batched = prepare_ad_pushforward(tangent_mean, forward, units, q, s;
                                         data, active = :q, want = :mean)
        @test reduce(hcat, ad_pushforward(batched, units, q, s; data)) ≈
              _tangent_jacobian(q, s, data)

        scalar = prepare_ad_pushforward(tangent_mean, forward, (1.0,), q, s;
                                        data, active = :s, want = :mean)
        @test only(ad_pushforward(scalar, (1.0,), q, s; data)) ≈
              data .* q .* exp.(s .* q)

        both = prepare_ad_pushforward(tangent_mean, forward, ((v, 1.0),), q, s;
                                      data, active = (:q, :s), want = :mean)
        @test only(ad_pushforward(both, ((v, 1.0),), q, s; data)) ≈
              _tangent_jacobian(q, s, data) * v .+ data .* q .* exp.(s .* q)

        low_level = prepare(plan(tangent_mean; want = :mean))
        direct = prepare_ad_pushforward(low_level, forward, (v,), q, s, data;
                                        active = :q)
        @test only(ad_pushforward(direct, (v,), q, s, data)) ≈
              _tangent_jacobian(q, s, data) * v
        @test_throws ArgumentError prepare_ad_pushforward(
            low_level, forward, (v,), q, s, data; active = :q, data)
    end

    @testset "tangent and WANT checks name the boundary" begin
        # A bare vector is one direction's tangent, not a direction tuple.
        @test_throws ArgumentError prepare_ad_hvp(
            tangent_density, second_order, v, q, s;
            data, active = :q, want = :density)
        @test_throws ArgumentError prepare_ad_hvp(
            tangent_density, second_order, (), q, s;
            data, active = :q, want = :density)
        @test_throws ArgumentError prepare_ad_hvp(
            tangent_density, second_order, ([1.0, 2.0],), q, s;
            data, active = :q, want = :density)
        prepared = prepare_ad_hvp(tangent_density, second_order, (v,), q, s;
                                  data, active = :q, want = :density)
        @test_throws ArgumentError ad_hvp(prepared, ([1.0],), q, s; data)
        # Hessian-vector products need a scalar WANT.
        @test_throws ArgumentError prepare_ad_hvp(
            tangent_mean, second_order, (v,), q, s;
            data, active = :q, want = :mean)
    end

    # DifferentiationInterface's forward-over-reverse HVP allocates its inner
    # gradient with `similar(x)`, which has no method for the structured
    # (Vector, Ref) point a tuple selector differentiates (DI 0.7.21). The
    # shape is valid; it waits on DI accepting tuple points.
    @testset "tuple-selector Hessian-vector products (DI tuple points)" begin
        @test_broken try
            prepared = prepare_ad_hvp(
                tangent_density, second_order, ((v, 1.0),), q, s;
                data, active = (:q, :s), want = :density)
            ad_hvp(prepared, ((v, 1.0),), q, s; data) isa Tuple
        catch error
            error isa MethodError || rethrow()
            false
        end
    end
end
