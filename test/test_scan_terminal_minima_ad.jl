module ScanTerminalMinimaADTests
using ReactiveKernels, DifferentiationInterface, Enzyme, Test
materialized_minimum(xs) = minimum(xs)
@kernel fused_minima(q::Vector{Float64}, xs::Vector{Float64}) = begin
    seed::Float64 = q[1]
    gain::Float64 = q[2]
    trajectory::Vector{Float64} = scan(xs, Ref(gain); init=seed, include_init=true) do carry, x, gain
        next = carry + gain*x
        (next,next)
    end
    relative::Vector{Float64} = @.(100 * (trajectory - trajectory[1]) / trajectory[1])
    low::Float64 = minimum(trajectory)
    relative_low::Float64 = minimum(relative)
    objective::Float64 = low + 0.01*relative_low
    return objective
end
@kernel reference_minima(q::Vector{Float64}, xs::Vector{Float64}) = begin
    seed::Float64 = q[1]
    gain::Float64 = q[2]
    trajectory::Vector{Float64} = scan(xs, Ref(gain); init=seed, include_init=true) do carry, x, gain
        next = carry + gain*x
        (next,next)
    end
    relative::Vector{Float64} = @.(100 * (trajectory - trajectory[1]) / trajectory[1])
    low::Float64 = materialized_minimum(trajectory)
    relative_low::Float64 = materialized_minimum(relative)
    objective::Float64 = low + 0.01*relative_low
    return objective
end
@testset "ordinary AD terminal minima" begin
    backend=AutoEnzyme(;mode=Enzyme.Reverse)
    # Two separated equal minima exercise Base's long-array reduction path.
    tied = zeros(256)
    tied[4] = tied[255] = -1.0
    tied[5] = tied[256] = 1.0
    for q in ([2.0,0.7], [-2.0,0.7]),
            xs in (Float64[], [1.0], [-1.0, 0.5, -0.2],
                   [-1.0, 1.0, -1.0], [0.0, 0.0], tied), bound in (false,true)
        saved = copy(q), copy(xs)
        fused=bound ? prepare(fused_minima;bound=(;xs)) : prepare(fused_minima)
        reference=bound ? prepare(reference_minima;bound=(;xs)) : prepare(reference_minima)
        args=bound ? (q,) : (q,xs)
        f_ad=prepare_ad(fused,backend,args...;active=:q)
        r_ad=prepare_ad(reference,backend,args...;active=:q)
        fg=ad_value_and_gradient(f_ad,args...)
        rg=ad_value_and_gradient(r_ad,args...)
        @test fg[1] ≈ rg[1]
        @test fg[2] ≈ rg[2]
        if !bound
            f_xs=prepare_ad(fused,backend,args...;active=:xs)
            r_xs=prepare_ad(reference,backend,args...;active=:xs)
            @test ad_gradient(f_xs,args...) ≈ ad_gradient(r_xs,args...)
        end
        @test (q, xs) == saved
    end
end

end # module
