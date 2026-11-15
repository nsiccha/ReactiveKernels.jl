module DistributionJSSourceTests
# `ReactiveKernels.js_source` of this package's laws (RK 0.1.264): a browser
# module computes the same law the RK, RKPPL and BRM models compute, its
# log-domain tails included, and Node agrees with native Julia.
using Test, ReactiveKernels
using ReactiveKernelsDistributionKernels.DistributionKernelSources: standard_student_t, student_t, normal,
    lognormal, exponential, gamma

const NODE = ReactiveKernels._js_default_node() !== nothing

agree(a::Real, b::Real) = (isnan(a) && isnan(b)) || isapprox(a, b; rtol = 1e-12)
agree(a::Union{AbstractVector,Tuple}, b::Union{AbstractVector,Tuple}) =
    length(a) == length(b) && all(agree.(a, b))
agree(a, b) = false

# Each case: the native value and the module's, or the same Julia exception.
function check_js(spec, have, want, types, cases)
    source = ReactiveKernels.js_source(spec; have, want, types, name = :law)
    native = prepare(spec; have, want)
    for (c, r) in zip(cases, ReactiveKernels.js_run(source, :law, cases))
        expected = try
            native(c...)
        catch e
            e
        end
        if expected isa Exception
            @test r.julia_error && r.error == string(nameof(typeof(expected)))
        else
            @test r.error === nothing && agree(expected, r.value)
        end
    end
    source
end

const ENDPOINTS = (:cdf, :ccdf, :logcdf, :logccdf, :logpdf)
const Z = [-Inf, -1e9, -1e6, -40.0, -3.0, -1.0, -1e-6, -0.0, 0.0, 1e-6, 0.5, 2.0, 40.0, 1e6, 1e9, Inf, NaN]
const NU = [0.5, 1.0, 2.5, 5.0, 16.5, 30.0, 1e6, Inf]
const STANDARD_T = Dict(ep => (@eval @kernel $(Symbol(:js_standard_t_, ep))(z::Float64, nu::Float64) = begin
    p = standard_student_t(nu).$ep(z)
    return p
end) for ep in ENDPOINTS)
const LOCATION_SCALE_T = Dict(ep => (@eval @kernel $(Symbol(:js_student_t_, ep))(x::Float64, nu::Float64,
        m::Float64, s::Float64) = begin
    p = student_t(nu, m, s).$ep(x)
    return p
end) for ep in ENDPOINTS)
const NORMAL = Dict(ep => (@eval @kernel $(Symbol(:js_normal_, ep))(x::Float64, m::Float64, s::Float64) = begin
    p = normal(m, s).$ep(x)
    return p
end) for ep in ENDPOINTS)
const LOGNORMAL = Dict(ep => (@eval @kernel $(Symbol(:js_lognormal_, ep))(x::Float64, m::Float64, s::Float64) = begin
    p = lognormal(m, s).$ep(x)
    return p
end) for ep in ENDPOINTS)
# The probability mass below each value of a vector (a detection limit's
# picture), written as a comprehension, as a plate and in a loop of a plate.
@kernel js_t_below(x::Vector{Float64}, nu::Float64) = begin
    p = [standard_student_t(nu).cdf(v) for v in x]
    return p
end
@kernel js_t_below_plate(x::Vector{Float64}, nu::Float64) = begin
    p = plate(eachindex(x)) do i
        standard_student_t(nu).cdf(x[i])
    end
    return p
end
@kernel js_lower_truncated_lognormal(x::Float64, m::Float64, s::Float64, lower::Float64) = begin
    p = lognormal(m, s).logpdf(x) - lognormal(m, s).logccdf(lower)
    return p
end
@kernel js_exponential(x::Float64, scale::Float64) = begin
    lp = exponential(scale).logpdf(x)
    c = exponential(scale).cdf(x)
    return lp
end
@kernel js_gamma_density(x::Float64, shape::Float64, rate::Float64) = begin
    lp = gamma(shape, rate).logpdf(x)
    return lp
end

@testset "js_source of the distribution laws" begin
    if !NODE
        @test_skip "no node on the PATH"
    else
        @testset "standard_student_t(nu).$ep" for ep in ENDPOINTS
            source = check_js(STANDARD_T[ep], (:z, :nu), :p, (; z = Float64, nu = Float64),
                              [(z, nu) for z in Z for nu in NU])
            # the continued fraction of the tails stays a loop
            ep in (:cdf, :ccdf, :logcdf, :logccdf) && @test occursin("while (true)", source)
        end
        @testset "student_t(nu, location, scale).$ep" for ep in ENDPOINTS
            check_js(LOCATION_SCALE_T[ep], (:x, :nu, :m, :s), :p,
                     (; x = Float64, nu = Float64, m = Float64, s = Float64),
                     [(x, nu, m, s) for x in (-3.0, 0.2, 1.5, 40.0) for nu in (1.0, 4.0, Inf)
                      for m in (0.0, 2.0) for s in (0.5, 3.0, -1.0)])
        end
        @testset "normal and lognormal .$ep" for ep in ENDPOINTS, (name, k) in ((:normal, NORMAL), (:lognormal, LOGNORMAL))
            check_js(k[ep], (:x, :m, :s), :p, (; x = Float64, m = Float64, s = Float64),
                     [(x, m, s) for x in (-Inf, -40.0, -1.0, 0.0, 0.3, 8.0, 40.0, Inf, NaN)
                      for m in (0.0, 1.5) for s in (0.0, 0.25, 2.0, -1.0)])
        end
        @testset "the mass below a limit, over a vector" begin
            cases = [([-2.0, 0.0, 2.0], 5.0), (Float64[], 2.0), ([-1e9, -1e6, -40.0, 3.0, 1e9], 1.0),
                     ([1.0], -1.0), ([0.5, NaN], 3.5)]
            types = (; x = Vector{Float64}, nu = Float64)
            check_js(js_t_below, (:x, :nu), :p, types, cases)
            check_js(js_t_below_plate, (:x, :nu), :p, types, cases)
            # the Cauchy tail, exact below 1e-15: atan(1 / 1e6) / π
            source = ReactiveKernels.js_source(js_t_below; have = (:x, :nu), want = :p, types, name = :below)
            r = only(ReactiveKernels.js_run(source, :below, [([-1e6, -1e9], 1.0)]))
            @test isapprox(r.value[1], atan(1e-6) / π; rtol = 1e-14)
            @test isapprox(r.value[2], atan(1e-9) / π; rtol = 1e-14)
        end
        @testset "lower-truncated lognormal, exponential and gamma densities" begin
            check_js(js_lower_truncated_lognormal, (:x, :m, :s, :lower), :p,
                     (; x = Float64, m = Float64, s = Float64, lower = Float64),
                     [(x, m, s, l) for x in (0.5, 3.0, 50.0) for m in (0.0, 1.0) for s in (0.3, 2.0)
                      for l in (0.1, 2.0, 30.0)])
            check_js(js_exponential, (:x, :scale), (:lp, :c), (; x = Float64, scale = Float64),
                     [(x, a) for x in (-1.0, 0.0, 0.5, 30.0, Inf) for a in (0.5, 2.0, -1.0)])
            check_js(js_gamma_density, (:x, :shape, :rate), :lp, (; x = Float64, shape = Float64, rate = Float64),
                     [(x, a, b) for x in (-1.0, 0.0, 0.5, 3.0, 30.0, Inf) for a in (0.5, 1.0, 2.5, 40.0)
                      for b in (0.5, 3.0)])
        end
    end
end

end
