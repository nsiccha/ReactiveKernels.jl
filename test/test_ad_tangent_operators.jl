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

# Two WANT ports read one shared intermediate, so one sweep computes both.
@kernel tangent_ports(q::Vector{Float64}, s::Float64;
                      data::Vector{Float64}) = begin
    shared = exp.(s .* q)
    mean = data .* shared
    total::Float64 = sum(shared) + s^2
end

# A design-matrix product: reads its constant matrix to build an
# intermediate array.
@kernel tangent_design(beta::Vector{Float64}; X::Matrix{Float64}) = begin
    density::Float64 = -0.5 * sum(abs2, X * beta)
end

# Per-group effects `z` never interact across groups, so the Hessian in `z`
# is block diagonal and one direction per within-group coordinate, summed
# across groups, recovers every block.
@kernel tangent_groups(z::Matrix{Float64}; w::Vector{Float64}) = begin
    density::Float64 = sum(exp.(z[:, 1] .+ w .* z[:, 2])) - 0.5 * sum(abs2, z)
end

# A design-matrix product with nonzero third derivatives.
@kernel tangent_exp_design(beta::Vector{Float64}; X::Matrix{Float64}) = begin
    density::Float64 = sum(exp.(X * beta)) - 0.5 * sum(abs2, X * beta)
end

# Two regression ports read constant design matrices and a group index, as a
# location and a shape predictor do.
@kernel tangent_regression_ports(u::Vector{Float64}; X::Matrix{Float64},
                                 Z::Matrix{Float64}, g::Vector{Int}) = begin
    beta = u[1:2]
    gamma = u[3:4]
    effects = u[5:7]
    location = exp.(X * beta .+ effects[g])
    shape = exp.(Z * gamma)
end

_tangent_hessian(q, s, data) = Diagonal(data .* s^2 .* exp.(s .* q)) - I
# ∇(vᵀ H v) of `tangent_density` in `q`, and of `tangent_exp_design`.
_tangent_third(q, s, data, v) = data .* s^3 .* exp.(s .* q) .* v .^ 2
_tangent_design_third(X, beta, v) = X' * (exp.(X * beta) .* (X * v) .^ 2)
_tangent_jacobian(q, s, data) = Diagonal(data .* s .* exp.(s .* q))
_tangent_unit(n, i) = (e = zeros(n); e[i] = 1.0; e)

@testset "prepared pushforwards and Hessian-vector products" begin
    # Reverse over forward: the documented Hessian-vector composition.
    second_order = SecondOrder(AutoEnzyme(; mode = Enzyme.Reverse),
                               AutoEnzyme(; mode = Enzyme.Forward))
    forward_over_reverse = SecondOrder(AutoEnzyme(; mode = Enzyme.Forward),
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

    # Several WANT ports differentiate as the kernel's tuple result: one
    # cotangent per port in, the sum of the ports' vector-Jacobian products
    # out; one tangent per port out of each direction.
    @testset "several WANT ports in one sweep" begin
        reverse = AutoEnzyme(; mode = Enzyme.Reverse)
        wants = (:mean, :total)
        shared(q, s) = exp.(s .* q)
        port_values(q, s, data) = (data .* shared(q, s), sum(shared(q, s)) + s^2)
        # Jᵀ seed in q and in s, summed over both ports.
        vjp_q(q, s, data, (a, b)) =
            data .* s .* shared(q, s) .* a .+ b .* s .* shared(q, s)
        vjp_s(q, s, data, (a, b)) =
            dot(a, data .* q .* shared(q, s)) +
            b * (sum(q .* shared(q, s)) + 2s)
        jvp_q(q, s, data, w) =
            (data .* s .* shared(q, s) .* w, dot(s .* shared(q, s), w))
        seed = ([1.0, -0.5, 2.0], 0.75)
        seed2 = ([0.3, 0.2, -1.0], -2.0)

        prepared = prepare_ad_pullback(tangent_ports, reverse, seed, q, s;
                                       data, active = :q, want = wants)
        @test prepared isa PreparedADPullback
        @test occursin("PreparedADPullback(active=:q, want=(:mean, :total)",
                       sprint(show, prepared))
        @test ad_pullback(prepared, seed, q, s; data) ≈ vjp_q(q, s, data, seed)
        for (newq, news, newdata) in (([0.1, 0.0, -0.1], 1.1, [1.0, 2.0, 3.0]),
                                      ([-0.5, 0.4, 0.3], 0.2, [0.5, 0.5, -4.0]))
            value, cotangent = ad_value_and_pullback(
                prepared, seed2, newq, news; data = newdata)
            @test all(value .≈ port_values(newq, news, newdata))
            @test cotangent ≈ vjp_q(newq, news, newdata, seed2)
        end
        # The same as one pullback per port, summed.
        per_port = sum(zip(wants, seed)) do (want, port_seed)
            ad_pullback(prepare_ad_pullback(tangent_ports, reverse, port_seed,
                                            q, s; data, active = :q, want),
                        port_seed, q, s; data)
        end
        @test ad_pullback(prepared, seed, q, s; data) ≈ per_port
        destination = similar(q)
        value, returned = ad_value_and_pullback!(
            prepared, destination, seed, q, s; data)
        @test returned === destination
        @test destination ≈ vjp_q(q, s, data, seed)
        @test all(value .≈ port_values(q, s, data))
        @test ad_pullback(tangent_ports, reverse, seed, q, s;
                          data, active = :q, want = wants) ≈
              vjp_q(q, s, data, seed)

        # A tuple selector: one cotangent per active port.
        both = prepare_ad_pullback(tangent_ports, reverse, seed, q, s;
                                   data, active = (:q, :s), want = wants)
        cotangent_q, cotangent_s = ad_pullback(both, seed2, q, s; data)
        @test cotangent_q ≈ vjp_q(q, s, data, seed2)
        @test cotangent_s ≈ vjp_s(q, s, data, seed2)

        # A low-level kernel prepared with several WANT ports, with its data
        # bound, as `prepare(spec; have, want, bound)` builds it.
        low_level = prepare(tangent_ports; have = (:q, :s, :data), want = wants,
                            bound = (; data))
        direct = prepare_ad_pullback(low_level, reverse, seed, q, s;
                                     active = :q)
        @test ad_pullback(direct, seed2, q, s) ≈ vjp_q(q, s, data, seed2)
        @test ad_pullback(low_level, reverse, seed2, q, s; active = :q) ≈
              vjp_q(q, s, data, seed2)

        # A backend that does not keep its primal sweep evaluates the port
        # values separately.
        @test !ad_retains_primal_sweep(forward)
        forward_pullback = prepare_ad_pullback(
            tangent_ports, forward, seed, q, s; data, active = :q, want = wants)
        value, cotangent = ad_value_and_pullback(forward_pullback, seed2, q, s;
                                                 data)
        @test all(value .≈ port_values(q, s, data))
        @test cotangent ≈ vjp_q(q, s, data, seed2)

        # Pushforwards: one tangent per port in each direction.
        units = ntuple(i -> _tangent_unit(3, i), 3)
        pushforward = prepare_ad_pushforward(tangent_ports, forward, units, q, s;
                                             data, active = :q, want = wants)
        @test occursin("PreparedADPushforward(active=:q, want=(:mean, :total)",
                       sprint(show, pushforward))
        value, jvps = ad_value_and_pushforward(pushforward, units, q, s; data)
        @test all(value .≈ port_values(q, s, data))
        for (unit, jvp) in zip(units, jvps)
            @test jvp isa Tuple{Vector{Float64},Float64}
            @test all(jvp .≈ jvp_q(q, s, data, unit))
        end
        direct_pushforward = prepare_ad_pushforward(low_level, forward, (v,),
                                                    q, s; active = :q)
        @test all(only(ad_pushforward(direct_pushforward, (v,), q, s)) .≈
                  jvp_q(q, s, data, v))

        # refused: a vector-Jacobian product pairs each port with a cotangent
        # of that port's structure, so a missing port or a misshapen cotangent
        # has no meaning.
        for wrong in (first(seed), (first(seed),), (first(seed), [1.0]),
                      ([1.0, 2.0], 0.75))
            @test_throws ArgumentError prepare_ad_pullback(
                tangent_ports, reverse, wrong, q, s;
                data, active = :q, want = wants)
        end
        # refused: a gradient or Hessian-vector product is of one scalar
        # objective; several ports have no single gradient.
        @test_throws ArgumentError prepare_ad(
            tangent_ports, reverse, q, s; data, active = :q, want = wants)
        @test_throws ArgumentError prepare_ad_hvp(
            tangent_ports, second_order, (v,), q, s;
            data, active = :q, want = wants)
    end

    # Plain reverse mode, no runtime activity, over constant design matrices,
    # unbound and bound (snag several-port-pul-e6344d44).
    @testset "several WANT ports over constant design matrices" begin
        reverse = AutoEnzyme(; mode = Enzyme.Reverse)
        wants = (:location, :shape)
        X = [1.0 0.5; -0.3 1.2; 0.8 -1.0; 0.2 0.4; -1.1 0.7; 0.6 0.9]
        Z = [0.4 -0.2; 1.0 0.3; -0.5 0.8; 0.9 -0.6; 0.1 1.1; -0.7 0.2]
        g = [1, 2, 3, 1, 2, 3]
        u = [0.2, -0.1, 0.3, 0.05, -0.2, 0.1, 0.25]
        seed = ([1.0, -0.5, 2.0, 0.3, -1.2, 0.7], [0.4, 1.1, -0.6, 0.9, 0.2, -0.3])
        port_values(u) = (exp.(X * u[1:2] .+ u[5:7][g]), exp.(Z * u[3:4]))
        function vjp(u, (a, b))
            location, shape = port_values(u)
            weighted = a .* location
            [X' * weighted; Z' * (b .* shape);
             [sum(weighted[g .== group]) for group in 1:3]]
        end

        unbound = prepare_ad_pullback(tangent_regression_ports, reverse, seed,
                                      u; X, Z, g, active = :u, want = wants)
        @test ad_pullback(unbound, seed, u; X, Z, g) ≈ vjp(u, seed)
        value, cotangent = ad_value_and_pullback(unbound, seed, u; X, Z, g)
        @test all(value .≈ port_values(u))
        @test cotangent ≈ vjp(u, seed)

        low_level = prepare(tangent_regression_ports; have = (:u, :X, :Z, :g),
                            want = wants, bound = (; X, Z, g))
        bound = prepare_ad_pullback(low_level, reverse, seed, u; active = :u)
        @test ad_pullback(bound, seed, u) ≈ vjp(u, seed)
        destination = similar(u)
        value, returned = ad_value_and_pullback!(bound, destination, seed, u)
        @test returned === destination
        @test destination ≈ vjp(u, seed)
        @test all(value .≈ port_values(u))
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

    @testset "forward over reverse" begin
        prepared = prepare_ad_hvp(tangent_density, forward_over_reverse, (v,),
                                  q, s; data, active = :q, want = :density)
        @test only(ad_hvp(prepared, (v,), q, s; data)) ≈
              _tangent_hessian(q, s, data) * v

        X = [1.0 0.5 0.0; 1.0 0.0 0.5; 1.0 1.0 1.0; 1.0 -1.0 2.0]
        beta = [0.1, 0.2, -0.3]
        reverse_over_forward = prepare_ad_hvp(
            tangent_design, second_order, (beta,), beta;
            X, active = :beta, want = :density)
        @test only(ad_hvp(reverse_over_forward, (beta,), beta; X)) ≈
              -(X' * X) * beta
        # Enzyme forward mode over a reverse pass that reads a constant array
        # into an intermediate array fails static activity analysis
        # (`benchmark/repro_enzyme_forward_over_reverse_const_array.jl`,
        # Enzyme 0.13.210). Reverse over forward differentiates it.
        @test_broken try
            prepared = prepare_ad_hvp(tangent_design, forward_over_reverse,
                                      (beta,), beta; X, active = :beta,
                                      want = :density)
            only(ad_hvp(prepared, (beta,), beta; X)) ≈ -(X' * X) * beta
        catch error
            nameof(typeof(error)) === :EnzymeRuntimeActivityError || rethrow()
            false
        end
    end

    # Reverse over forward over forward: the Hessian-vector product of the
    # directional derivative `x -> Df(x)[v]` is `∇(vᵀ H v) = D³f[v, v, ·]`.
    third_order = SecondOrder(second_order, forward)

    @testset "third-order directional derivatives rebind constants per call" begin
        prepared = prepare_ad_third(tangent_density, third_order, (v,), q, s;
                                    data, active = :q, want = :density)
        @test prepared isa PreparedADThird
        @test prepared isa PreparedADOperator
        @test inputs(prepared) == inputs(tangent_density)
        @test occursin("PreparedADThird(active=:q, want=:density",
                       sprint(show, prepared))
        @test only(ad_third(prepared, (v,), q, s; data)) ≈
              _tangent_third(q, s, data, v)
        for (newq, news, newdata) in (([0.1, 0.0, -0.1], 1.1, [1.0, 2.0, 3.0]),
                                      ([-0.5, 0.4, 0.3], 0.2, [0.5, 0.5, -4.0]))
            w = [-1.0, 2.0, 0.25]
            @test only(ad_third(prepared, (w,), newq, news; data = newdata)) ≈
                  _tangent_third(newq, news, newdata, w)
        end

        gradient, hvps, thirds = ad_gradient_hvp_and_third(
            prepared, (v,), q, s; data)
        @test gradient ≈ data .* s .* exp.(s .* q) .- q
        @test only(hvps) ≈ _tangent_hessian(q, s, data) * v
        @test only(thirds) ≈ _tangent_third(q, s, data, v)

        results = (similar(q),)
        @test ad_third!(prepared, results, (2v,), q, s; data) === results
        @test only(results) ≈ 4 * _tangent_third(q, s, data, v)
        gradient_buffer, hvp_buffers, third_buffers =
            similar(q), (similar(q),), (similar(q),)
        returned = ad_gradient_hvp_and_third!(
            prepared, gradient_buffer, hvp_buffers, third_buffers, (v,), q, s;
            data)
        @test returned[1] === gradient_buffer
        @test returned[2] === hvp_buffers && returned[3] === third_buffers
        @test gradient_buffer ≈ gradient
        @test only(hvp_buffers) ≈ only(hvps)
        @test only(third_buffers) ≈ only(thirds)
        @test_throws DimensionMismatch ad_third!(
            prepared, (similar(q), similar(q)), (v,), q, s; data)
    end

    @testset "third-order: batched directions, bound and low-level preparations" begin
        units = ntuple(i -> _tangent_unit(3, i), 3)
        batched = prepare_ad_third(tangent_density, third_order, units, q, s;
                                   data, active = :q, want = :density)
        for (unit, third) in zip(units, ad_third(batched, units, q, s; data))
            @test third ≈ _tangent_third(q, s, data, unit)
        end

        bound = prepare_ad_third(tangent_density, third_order, (v,), q, s;
                                 active = :q, want = :density,
                                 bound = (; data))
        @test only(ad_third(bound, (v,), q, s)) ≈ _tangent_third(q, s, data, v)

        low_level = prepare(plan(tangent_density; want = :density))
        direct = prepare_ad_third(low_level, third_order, (v,), q, s, data;
                                  active = :q)
        @test only(ad_third(direct, (v,), q, s, data)) ≈
              _tangent_third(q, s, data, v)
    end

    @testset "third-order derivatives of a constant-matrix product" begin
        # The design matrix multiplies the active vector; the outermost pass
        # is plain reverse mode with no activity annotation.
        X = [1.0 0.5 0.0; 1.0 0.0 0.5; 1.0 1.0 1.0; 1.0 -1.0 2.0]
        beta = [0.1, 0.2, -0.3]
        unbound = prepare_ad_third(tangent_exp_design, third_order, (v,), beta;
                                   X, active = :beta, want = :density)
        bound = prepare_ad_third(tangent_exp_design, third_order, (v,), beta;
                                 active = :beta, want = :density,
                                 bound = (; X))
        # On Julia 1.12 and later, registered Enzyme returns NaN on some calls:
        # its forward-mode `gemv` rule scales the output tangent by `beta`
        # instead of overwriting it when `beta == 0`. The direction `w` is
        # constant at both outer levels, so the doubly forward tangent product
        # starts from an uninitialized shadow, which OpenBLAS 0.3.28 and later
        # no longer zero-fill (Enzyme snag `nested-forward-o-9434176c`, whose
        # proposed fix is a libEnzyme change). A NaN there is pinned as broken;
        # every other result must be exact.
        third_or_gap(result, expected) =
            VERSION >= v"1.12" && any(isnan, result) ? (@test_broken false) :
                (@test result ≈ expected)
        for w in (v, [0.3, -1.0, 0.7])
            expected = _tangent_design_third(X, beta, w)
            third_or_gap(only(ad_third(unbound, (w,), beta; X)), expected)
            third_or_gap(only(ad_third(bound, (w,), beta)), expected)
        end
    end

    @testset "third-order contractions from colored directions" begin
        # Groups never interact, so a direction summed across groups returns
        # the sum of every group's own contraction.
        groups = 5
        z = [0.1 * sin(i + j) for i in 1:groups, j in 1:2]
        w = [0.3 * i for i in 1:groups]
        colors = ntuple(j -> (seed = zeros(groups, 2); seed[:, j] .= 1.0; seed), 2)
        prepared = prepare_ad_third(tangent_groups, third_order, colors, z;
                                    w, active = :z, want = :density)
        compressed = ad_third(prepared, colors, z; w)
        for group in 1:groups, j in 1:2
            e = exp(z[group, 1] + w[group] * z[group, 2])
            a = j == 1 ? 1.0 : w[group]
            @test compressed[j][group, 1] ≈ e * a^2
            @test compressed[j][group, 2] ≈ e * a^2 * w[group]
        end
    end

    @testset "third-order backend, tangent and WANT checks" begin
        # The backend nests a Hessian-vector backend over a pushforward one.
        for backend in (second_order, forward,
                        SecondOrder(forward, second_order))
            @test_throws ArgumentError prepare_ad_third(
                tangent_density, backend, (v,), q, s;
                data, active = :q, want = :density)
        end
        @test_throws ArgumentError prepare_ad_third(
            tangent_density, third_order, v, q, s;
            data, active = :q, want = :density)
        @test_throws ArgumentError prepare_ad_third(
            tangent_density, third_order, ([1.0, 2.0],), q, s;
            data, active = :q, want = :density)
        prepared = prepare_ad_third(tangent_density, third_order, (v,), q, s;
                                    data, active = :q, want = :density)
        @test_throws ArgumentError ad_third(prepared, ([1.0],), q, s; data)
        @test_throws ArgumentError prepare_ad_third(
            tangent_mean, third_order, (v,), q, s;
            data, active = :q, want = :mean)
    end

    # DifferentiationInterface 0.7.21 cannot take a Hessian-vector product at
    # the structured (Vector, Ref) point a tuple selector differentiates:
    # forward over reverse calls `similar(x)` and reverse over forward
    # `pick_batchsize`, neither defined for a tuple. The shape is valid; it
    # waits on DI accepting tuple points.
    @testset "tuple-selector Hessian-vector products (DI tuple points)" begin
        for backend in (second_order, forward_over_reverse)
            @test_broken try
                prepared = prepare_ad_hvp(
                    tangent_density, backend, ((v, 1.0),), q, s;
                    data, active = (:q, :s), want = :density)
                ad_hvp(prepared, ((v, 1.0),), q, s; data) isa Tuple
            catch error
                error isa MethodError || rethrow()
                false
            end
        end
    end
end
