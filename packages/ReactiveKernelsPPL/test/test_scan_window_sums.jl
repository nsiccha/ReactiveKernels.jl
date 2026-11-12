using ReactiveKernels, ReactiveKernelsPPL, Distributions
using DifferentiationInterface, Enzyme, Test

function _window_program(coupled; history = false)
    read = history ? :(sum(h[1:(t - 1)]) / (t - 1)) :
        :(sum(h[(t - 2):(t - 1)]))
    seeds = coupled ? Any[:(b[1] ~ Normal()), :(b[2] ~ Normal())] : Any[]
    steps = coupled ? Any[:(b[t] ~ Normal(0.5 * b[t - 1], 0.7))] : Any[]
    quote
        sigma ~ LogNormal(0, 0.3)
        @scan begin
            h[1] ~ Normal()
            h[2] ~ Normal()
            $(seeds...)
            for t in 3:T
                h[t] ~ Normal(enabled ? -0.2 * $read : 0.0, sigma)
                $(steps...)
            end
        end
        y .~ Normal.(h, 0.9)
    end
end

function _window_oracle(u, T, coupled, enabled, history)
    sigma = exp(u[1])
    if coupled
        # Multi-state scans pack all seeds, then one contiguous block per
        # sampled step. Centered single-state scans pack the trajectory.
        h = vcat(u[2:3], u[6:(T + 3)])
        b = vcat(u[4:5], u[(T + 4):end])
    else
        h, b = u[2:end], Float64[]
    end
    lp = logpdf(LogNormal(0, 0.3), sigma) + u[1] + sum(logpdf.(Normal(), h[1:2]))
    for t in 3:T
        m = history ? sum(h[1:(t - 1)]) / (t - 1) : sum(h[(t - 2):(t - 1)])
        lp += logpdf(Normal(enabled ? -0.2 * m : 0.0, sigma), h[t])
    end
    if coupled
        lp += sum(logpdf.(Normal(), b[1:2]))
        for t in 3:T
            lp += logpdf(Normal(0.5 * b[t - 1], 0.7), b[t])
        end
    end
    return lp + sum(logpdf.(Normal.(h, 0.9), 0.1 .* sin.(1:T)))
end

function _window_allocations(q, u, g)
    q(u)
    sampler_value_and_gradient!(q, g, u)
    return (@allocations(q(u)), @allocations(sampler_value_and_gradient!(q, g, u)))
end

_window_nodes(ex) = ex isa Expr ? 1 + sum(_window_nodes, ex.args; init = 0) : 0

@testset "scan window sums: scalar lags and runtime history" begin
    for coupled in (false, true), history in (false, true)
        allocations = Tuple{Int,Int}[]
        emitted = Int[]
        for T in (2, 8, 24), enabled in (false, true)
            data = (; T, enabled, y = 0.1 .* sin.(1:T))
            plan = bind_data(lower_rkppl(_window_program(coupled; history), data;
                conditioned = (:y,)), data)
            built = build_kernel(plan)
            u = 0.13 .* sin.(1:built.layout.total)
            q = prepare_sampler(built, plan, u;
                backend = AutoEnzyme(; mode = Enzyme.Reverse), warm = true,
                native_forward_batch = false)
            g = similar(u)
            oracle(x) = _window_oracle(x, T, coupled, enabled, history)
            for shift in (0.0, 0.07)
                point = u .+ shift
                v, _ = sampler_value_and_gradient!(q, g, point)
                @test v ≈ oracle(point) rtol = 1e-11
                epsilon = cbrt(eps(Float64))
                fd = map(eachindex(point)) do i
                    up, dn = copy(point), copy(point)
                    up[i] += epsilon
                    dn[i] -= epsilon
                    (oracle(up) - oracle(dn)) / (2epsilon)
                end
                @test g ≈ fd rtol = 1e-5 atol = 1e-7
            end
            if enabled && T > 2
                push!(allocations, _window_allocations(q, u, g))
                # Count actual emitted expression nodes; both data extents
                # must retain one body rather than replicate it per step.
                push!(emitted, _window_nodes(ReactiveKernels.code_expr(q.kernel)))
            end
        end
        @test emitted[1] == emitted[2]
        @test allocations[2][1] <= allocations[1][1] + 2
        # Constant Reverse allocation is the fixed-window regression. A
        # growing full-history reduction retains a runtime sum; its backend
        # tape may allocate as the reduction grows.
        if !history
            @test allocations[2][2] <= allocations[1][2] + 2
        end
    end
end

@testset "scan window sums: only literal nonempty backward windows expand" begin
    window(ex) = ReactiveKernels._history_window_sum(ex, [:h], :t, 11)
    @test window(:(sum(h[(t - 2):(t - 1)]))) == :(sum((h[t - 2], h[t - 1])))
    @test window(:(sum(h[(t - 11):(t - 1)]))) isa Expr
    for ex in (:(sum(h[1:(t - 1)])), :(sum(h[(t - k):(t - 1)])),
               :(sum(h[(t - 1):(t - 2)])), :(sum(h[(t - 12):(t - 1)])),
               :(sum(h[(t - 2):t])), :(sum(h[(t - 2):2:(t - 1)])))
        @test window(ex) === nothing
    end
    # sum's promotion differs from raw + for Bool and narrow integers.
    for values in ([true, true], Int8[100, 100], Float64[0.5, -0.25])
        @test sum(Tuple(values)) === sum(values)
    end
end
