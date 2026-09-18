# Standard-kernel expression of the Tsit5 stage block: the kernel graph
# prepares, infers, and its two executors — functional (native) and prepared
# (traced) — agree bit-for-bit.
using ReactiveKernels: KernelSpec, PreparedKernel

const TSIT5_KERNEL = RKRO.prepare_tsit5_stage()

function check_kernel_parity(f, uprev, p, t, dt, atol, rtol)
    T = eltype(uprev)
    # The driver promotes tolerances to the compute type (`atol = T(abstol)`
    # in `solve_ode`); mirror that so the mixed-precision shape stays out.
    atol_T, rtol_T = T(atol), T(rtol)
    tab = RKRO.Tsit5Tableau{T}()
    k1 = f(uprev, p, t)
    ref = RKRO.tsit5_step(f, uprev, k1, p, t, dt, tab, atol_T, rtol_T)
    u_k, kk, EEst_k = TSIT5_KERNEL(f, uprev, k1, p, t, dt, tab, atol_T,
        rtol_T, T(inv(length(uprev))))
    return (ref=ref, u_k=u_k, kk=kk, EEst_k=EEst_k)
end

@testset "stage kernel prepares and infers" begin
    @test RKRO.tsit5_stage isa KernelSpec
    @test TSIT5_KERNEL isa PreparedKernel
    f = exponential_decay
    uprev = [1.0, 2.0]
    k1 = f(uprev, [0.5, 1.5], 0.0)
    tab = RKRO.Tsit5Tableau{Float64}()
    Rt = only(Base.return_types(TSIT5_KERNEL,
        Tuple{typeof(f),Vector{Float64},Vector{Float64},Vector{Float64},
            Float64,Float64,typeof(tab),Float64,Float64,Float64}))
    @test Rt == Tuple{Vector{Float64},NTuple{7,Vector{Float64}},Float64}
end

@testset "functional and prepared executors agree bit-for-bit" begin
    cases = (
        (exponential_decay, [1.0, 2.0], [0.5, 1.5], 0.0, 0.1, 1e-10, 1e-8),
        (exponential_decay, Float32[1.0, 2.0], Float32[0.5, 1.5], 0.0f0,
            0.1f0, 1e-6, 1e-3),
        (lotka_volterra, [1.0, 3.0], LOTKA_PARAMS, 0.0, 0.1, 1e-6, 1e-3),
        (lotka_volterra, Float32[1.0, 3.0], Float32.(LOTKA_PARAMS), 0.0f0,
            0.1f0, 1e-6, 1e-3),
        ((u, p, t) -> -0.5 .* u, [2.0, 0.5], nothing, 0.25, 0.05, 1e-8,
            1e-6),
    )
    for (f, uprev, p, t, dt, atol, rtol) in cases
        out = check_kernel_parity(f, uprev, p, t, dt, atol, rtol)
        @test out.u_k == out.ref.u
        @test out.EEst_k == out.ref.EEst
        @test length(out.kk) == 7
        for j in 1:7
            @test out.kk[j] == out.ref.k[j]
        end
        @test eltype(out.u_k) == eltype(uprev)
        @test typeof(out.EEst_k) == eltype(uprev)
    end
end
