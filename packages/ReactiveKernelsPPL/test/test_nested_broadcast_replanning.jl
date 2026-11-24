module TestNestedBroadcastReplanning

using Test, ReactiveKernelsPPL, Enzyme
using ReactiveKernels: @kernel
using DifferentiationInterface: AutoEnzyme

@rkppl meter(y, x) = begin
    offset ~ Normal(0, 1)
    @plate for s in eachindex(y)
        y[s] .~ Normal.(offset .+ x[s], 1)
    end
    return offset
end

@kernel flagged_score(y, mu, flag) = begin
    weight = flag > 0 ? 2.0 : 1.0
    residual = y - mu
    return -0.5 * weight * residual^2
end

@rkppl flagged_meter(y, x, flags) = begin
    offset ~ Normal(0, 1)
    @plate for s in eachindex(y)
        y[s] .~ LogDensity.(flagged_score, offset .+ x[s], flags[s])
    end
    return offset
end

function scalar_posterior(data, point, fitted, external)
    logpdf(r) = -(abs2(r) + log(2π)) / 2
    value = sum(logpdf, point)
    gradient = -point
    for k in eachindex(data.ids), s in eachindex(data.y[k]), j in eachindex(data.y[k][s])
        residual = data.y[k][s][j] - (point[k + 1] + data.x[k][s][j])
        weight = external && data.flags[k][s][j] > 0 ? 2.0 : 1.0
        value += external ? -weight * residual^2 / 2 : logpdf(residual)
        gradient[k + 1] += weight * residual
    end
    for i in eachindex(data.z), j in eachindex(data.z[i])
        u = data.u[i][j]
        residual = data.z[i][j] - (fitted ? point[1] * u : u)
        value += logpdf(residual)
        fitted && (gradient[1] += residual * u)
    end
    return value, gradient
end

@testset "Nested observation axes survive predictor replanning" begin
    small = (; ids=[1], y=[[[0.2, 0.3], [0.9]]], x=[[[1., 2.], [0.1]]],
        flags=[[[1., 0.], [0.]]], z=[[0.8, 1.3], [0.4]], u=[[0.5, 0.8], [0.1]])
    large = (; ids=[1, 2], y=[[Float64[], [0.7]], [[0.1, 0.2, 0.3]]],
        x=[[Float64[], [1.1]], [[0.2, -1.1, 0.5]]],
        flags=[[Float64[], [0.]], [[1., 0., 1.]]],
        z=[[1.2], Float64[], [0.4, -0.3, 1.0]], u=[[0.3], Float64[], [1.2, -0.1, 0.8]])
    for external in (false, true), fitted in (false, true)
        loc = fitted ? :(expected[i]) : :(u[i])
        reading = external ? :(flagged_meter(y[k], x[k], flags[k])) : :(meter(y[k], x[k]))
        model = quote
            gain ~ Normal(0, 1)
            @plate for i in eachindex(u)
                expected[i] = gain .* u[i]
            end
            @plate for k in eachindex(ids)
                reading[k] ~ $reading
            end
            @plate for i in eachindex(z)
                z[i] .~ Normal.($loc, 1)
            end
        end
        plan = lower_rkppl(model, small; mod=@__MODULE__, conditioned=(:y, :z))
        @test plan.nested_broadcasts == Set([:y])
        for (data, point) in ((small, [0.3, -0.4]), (large, [-0.2, 0.5, 0.7]))
            bound = bind_data(plan, data)
            built = build_kernel(bound)
            @test coordinate_names(built.layout) ==
                [:gain; [Symbol("reading.offset.", k) for k in eachindex(data.ids)]]
            refvalue, refgradient = scalar_posterior(data, point, fitted, external)
            primal = prepare_query(built, bound, :sampler)
            @test Base.invokelatest(primal, point) ≈ refvalue
            sampler = prepare_sampler(built, bound, point;
                backend=AutoEnzyme(; mode=Enzyme.Reverse), native_forward_batch=false)
            value, gradient = Base.invokelatest(sampler_value_and_gradient!, sampler, similar(point), point)
            @test value ≈ refvalue
            @test gradient ≈ refgradient
        end
    end
end

end
