# Vectorized saveat emission (one masked dense-buffer update per step)
# against the per-point dense evaluation it replaces.
const RKRO_DENSE = ReactiveKernelsReactantODESolvers

@testset "vector dense evaluation is the per-point evaluation column-wise" begin
    dense = RKRO_DENSE.Tsit5DenseCoefficients{Float64}()
    uprev = [1.0, -2.0, 0.5]
    stages = ntuple(i -> [0.1i, -0.2i, 0.3i] .+ 0.05, 7)
    dt = 0.25
    θs = [-0.4, 0.0, 0.4, 1.0, 1.6]
    mat = RKRO_DENSE.tsit5_dense_eval(uprev, stages, dt, θs, dense)
    @test size(mat) == (3, 5)
    for j in eachindex(θs)
        @test mat[:, j] ≈ RKRO_DENSE.tsit5_dense_eval(uprev, stages, dt, θs[j],
            dense)
    end
end

@testset "vectorized saveat emission matches per-point dense output" begin
    dense = RKRO_DENSE.Tsit5DenseCoefficients{Float64}()
    uprev = [1.0, -2.0, 0.5]
    stages = ntuple(i -> [0.1i, -0.2i, 0.3i] .+ 0.05, 7)
    t, dt = 0.5, 0.25
    lo, hi = t, t + dt
    saveat = [0.4, 0.5, 0.6, 0.75, 0.9]
    out0 = fill(-1.0, 3, 5)
    out = RKRO_DENSE._emit_saveat(out0, saveat, uprev, stages, t, dt, lo, hi,
        true, dense)
    @test out0 == fill(-1.0, 3, 5)
    for (j, ts) in enumerate(saveat)
        if lo < ts <= hi
            @test out[:, j] ≈ RKRO_DENSE.tsit5_dense_eval(uprev, stages, dt,
                (ts - t) / dt, dense)
        else
            @test out[:, j] == out0[:, j]
        end
    end
    @test count(j -> out[:, j] != out0[:, j], 1:5) == 2
    # A rejected step leaves the buffer untouched.
    @test RKRO_DENSE._emit_saveat(out0, saveat, uprev, stages, t, dt, lo, hi,
        false, dense) == out0
    # Backward integration: the window is still `(lo, hi]` in time order.
    back = RKRO_DENSE._emit_saveat(out0, saveat, uprev, stages, hi, -dt, lo, hi,
        true, dense)
    @test back[:, 3] ≈ RKRO_DENSE.tsit5_dense_eval(uprev, stages, -dt,
        (0.6 - hi) / -dt, dense)
    @test back[:, 1] == out0[:, 1]
end
