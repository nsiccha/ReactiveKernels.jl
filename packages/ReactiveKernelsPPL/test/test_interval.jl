# Interval-censored response (`interval_censored.(dist, hi)` evidence kind;
# the response IS the lower endpoint): surface admission (Gaussian/Poisson
# × literal/column upper, fused head, weighted composition), value parity
# vs Distributions.jl hand oracles, Enzyme-vs-findiff gradients,
# Reactant/XLA value+grad on the Gaussian cell, the pinned upstream
# Poisson-XLA gap (`poisson.cdf` needs `SpecialFunctions.gamma_inc`,
# which has no Reactant tracing rule), and the I1/I2 SB-parity probes
# (N=80, Xoshiro recipes, vectors inlined so the test is immune to
# RNG/Distributions drift; SB literals filled from the peer lane's
# BridgeStan brief before landing). (`_findiff_grad` / `_GEN_BACKEND`
# come from test_generator.jl, included first.)
using DifferentiationInterface
using Distributions: Poisson, Normal, Exponential, logpdf, cdf
using Enzyme
using ReactiveKernels
using ReactiveKernelsPPL
using Reactant
using SHA
using Test

# Lower + bind + build + query an interval program; return
# `(bound, built, kern, layout)`.
function _int_query(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    kern = prepare_query(built, bound, :sampler)
    return bound, built, kern, built.layout
end

# Posterior at a constrained probe (world-age-safe call).
_int_posterior(kern, lay, q::NamedTuple) =
    Base.invokelatest(kern, unconstrain(lay, q))

# Scalar interval log-densities (the response is the lower endpoint:
# Gaussian `log(F(hi) - F(y))`, Poisson `log(F(ub) - F(y-1))` with
# F(-1) = 0 definitionally — the inclusive-cdf shift).
_int_gref(y::Real, mu::Real, s::Real, hi::Real) =
    log(cdf(Normal(mu, s), hi) - cdf(Normal(mu, s), y))
_int_pF(lam::Real, k::Integer) = k < 0 ? 0.0 : cdf(Poisson(lam), k)
_int_pref(y::Integer, lam::Real, ub::Integer) =
    log(_int_pF(lam, ub) - _int_pF(lam, y - 1))

const _INT_X = [0.5, -1.0, 1.5, 0.0, -0.5, 1.0]
const _INT_YG = [1.0, 2.0, 1.5, 2.5, 3.0, 2.0]
const _INT_HI = [4.0, 3.0, 5.0, 4.5, 6.0, 3.5]
const _INT_YP = [0, 1, 2, 1, 3, 2]
const _INT_UB = [2, 3, 4, 2, 5, 3]
const _INT_W = [1.0, 2.0, 1.0, 0.5, 1.5, 1.0]
_int_gcols() = Dict{Symbol,AbstractVector}(:y => copy(_INT_YG),
    :x => copy(_INT_X), :hi => copy(_INT_HI))
_int_pcols() = Dict{Symbol,AbstractVector}(:y => copy(_INT_YP),
    :x => copy(_INT_X), :ub => copy(_INT_UB))

@testset "interval surface admission" begin
    @testset "gaussian literal upper" begin
        plan = lower_rkppl(quote
                mu = a .+ b .* x
                y .~ interval_censored.(Normal.(mu, s), 5.0)
                s ~ Exponential(1)
            end, (:y, :x))
        r = only(plan.responses)
        @test r.family === GaussianFam
        @test r.link === IdentityLink
        @test r.predictor === :mu
        @test r.scale === :s
        @test r.weights === nothing
        @test r.evidence.kind === :interval_censored
        @test r.evidence.lower === nothing
        @test r.evidence.upper == 5.0
        @test r.trials === nothing
        @test r.range === nothing
    end
    @testset "gaussian column upper" begin
        plan = lower_rkppl(quote
                mu = a .+ b .* x
                y .~ interval_censored.(Normal.(mu, s), hi)
                s ~ Exponential(1)
            end, (:y, :x, :hi))
        r = only(plan.responses)
        @test r.family === GaussianFam
        @test r.evidence.kind === :interval_censored
        @test r.evidence.lower === nothing
        @test r.evidence.upper === :hi
    end
    @testset "poisson literal upper" begin
        plan = lower_rkppl(quote
                eta = a .+ b .* x
                y .~ interval_censored.(Poisson.(exp.(eta)), 4)
            end, (:y, :x))
        r = only(plan.responses)
        @test r.family === PoissonLogFam
        @test r.link === LogLink
        @test r.predictor === :eta
        @test r.scale === nothing
        @test r.evidence.kind === :interval_censored
        @test r.evidence.lower === nothing
        @test r.evidence.upper == 4
    end
    @testset "poisson column upper" begin
        plan = lower_rkppl(quote
                eta = a .+ b .* x
                y .~ interval_censored.(Poisson.(exp.(eta)), ub)
            end, (:y, :x, :ub))
        r = only(plan.responses)
        @test r.family === PoissonLogFam
        @test r.evidence.kind === :interval_censored
        @test r.evidence.upper === :ub
    end
    @testset "fused poisson head lowers identically" begin
        plan = lower_rkppl(quote
                eta = a .+ b .* x
                y .~ interval_censored.(PoissonLog.(eta), 4)
            end, (:y, :x))
        r = only(plan.responses)
        @test r.family === PoissonLogFam
        @test r.link === LogLink
        @test r.evidence.kind === :interval_censored
        @test r.evidence.upper == 4
    end
    @testset "weighted composition (weights outermost)" begin
        plan = lower_rkppl(quote
                mu = a .+ b .* x
                y .~ weighted.(interval_censored.(Normal.(mu, s), hi), w)
                s ~ Exponential(1)
            end, (:y, :x, :hi, :w))
        r = only(plan.responses)
        @test r.weights === :w
        @test r.evidence.kind === :interval_censored
        @test r.evidence.upper === :hi
    end
    @testset "rejections" begin
        # Missing upper.
        @test_throws SurfaceLoweringError lower_rkppl(quote
                mu = a .+ b .* x
                y .~ interval_censored.(Normal.(mu, s))
                s ~ Exponential(1)
            end, (:y, :x))
        # Evidence is Gaussian/Poisson-only (slice 1 family gate).
        @test_throws ContractValidationError lower_rkppl(quote
                eta = a .+ b .* x
                y .~ interval_censored.(Bernoulli.(logistic.(eta)), 1)
            end, (:y, :x))
        # Response must sit strictly below the upper every row (bind-time
        # data gate).
        plan = lower_rkppl(quote
                mu = a .+ b .* x
                y .~ interval_censored.(Normal.(mu, s), hi)
                s ~ Exponential(1)
            end, (:y, :x, :hi))
        bad = _int_gcols()
        bad[:y] = copy(_INT_HI)
        @test_throws ContractValidationError bind_data(plan, bad)
    end
end

# Gaussian interval posterior oracle at a constrained probe (interval
# cells + Normal(0,1) coef priors + Exponential(1) scale prior +
# log(s) Jacobian).
function _int_gauss_oracle(y, x, hi, a, b, s)
    mu = a .+ b .* x
    ll = sum(_int_gref(y[i], mu[i], s, hi[i]) for i in eachindex(y))
    return ll + logpdf(Normal(0, 1), a) + logpdf(Normal(0, 1), b) +
        logpdf(Exponential(1), s) + log(s)
end

# Poisson interval posterior oracle (interval cells + Normal(0,1) coef
# priors; no scale, no Jacobian).
function _int_pois_oracle(y, x, ub, a, b)
    lam = exp.(a .+ b .* x)
    ll = sum(_int_pref(y[i], lam[i], ub[i]) for i in eachindex(y))
    return ll + logpdf(Normal(0, 1), a) + logpdf(Normal(0, 1), b)
end

@testset "interval values vs oracle" begin
    @testset "gaussian literal upper" begin
        _, _, kern, lay = _int_query(quote
                mu = a .+ b .* x
                y .~ interval_censored.(Normal.(mu, s), 5.0)
                s ~ Exponential(1)
            end, _int_gcols())
        q = (mu = [0.5, -0.25], s = 1.3)
        want = _int_gauss_oracle(_INT_YG, _INT_X, fill(5.0, 6), q.mu[1],
            q.mu[2], q.s)
        @test _int_posterior(kern, lay, q) ≈ want rtol = 1e-12
    end
    @testset "gaussian column upper" begin
        _, _, kern, lay = _int_query(quote
                mu = a .+ b .* x
                y .~ interval_censored.(Normal.(mu, s), hi)
                s ~ Exponential(1)
            end, _int_gcols())
        q = (mu = [0.5, -0.25], s = 1.3)
        want = _int_gauss_oracle(_INT_YG, _INT_X, _INT_HI, q.mu[1],
            q.mu[2], q.s)
        @test _int_posterior(kern, lay, q) ≈ want rtol = 1e-12
    end
    @testset "gaussian weighted" begin
        cols = _int_gcols()
        cols[:w] = copy(_INT_W)
        _, _, kern, lay = _int_query(quote
                mu = a .+ b .* x
                y .~ weighted.(interval_censored.(Normal.(mu, s), hi), w)
                s ~ Exponential(1)
            end, cols)
        q = (mu = [0.5, -0.25], s = 1.3)
        mu = q.mu[1] .+ q.mu[2] .* _INT_X
        ll = sum(_INT_W[i] *
            _int_gref(_INT_YG[i], mu[i], q.s, _INT_HI[i]) for i in 1:6)
        want = ll + logpdf(Normal(0, 1), q.mu[1]) +
            logpdf(Normal(0, 1), q.mu[2]) + logpdf(Exponential(1), q.s) +
            log(q.s)
        @test _int_posterior(kern, lay, q) ≈ want rtol = 1e-12
    end
    @testset "poisson literal upper" begin
        _, _, kern, lay = _int_query(quote
                eta = a .+ b .* x
                y .~ interval_censored.(Poisson.(exp.(eta)), 4)
            end, _int_pcols())
        q = (eta = [0.1, -0.2],)
        want = _int_pois_oracle(_INT_YP, _INT_X, fill(4, 6), q.eta[1],
            q.eta[2])
        @test _int_posterior(kern, lay, q) ≈ want rtol = 1e-12
    end
    @testset "poisson column upper" begin
        _, _, kern, lay = _int_query(quote
                eta = a .+ b .* x
                y .~ interval_censored.(Poisson.(exp.(eta)), ub)
            end, _int_pcols())
        q = (eta = [0.1, -0.2],)
        want = _int_pois_oracle(_INT_YP, _INT_X, _INT_UB, q.eta[1],
            q.eta[2])
        @test _int_posterior(kern, lay, q) ≈ want rtol = 1e-12
    end
end

# One Enzyme-vs-findiff gradient check at a constrained probe (no oracle
# needed).
function _int_enzyme_check(prog::Expr, cols::Dict{Symbol,AbstractVector},
        q::NamedTuple)
    bound, built, kern, lay = _int_query(prog, cols)
    u = unconstrain(lay, q)
    prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
    g = similar(u)
    _, _ = sampler_value_and_gradient!(prep, g, u)
    @test all(isfinite, g)
    @test isapprox(g, _findiff_grad(w -> Base.invokelatest(kern, w), u);
        rtol = 1e-5, atol = 1e-7)
    return g
end

@testset "interval Enzyme gradients" begin
    @testset "gaussian literal upper" begin
        _int_enzyme_check(quote
                mu = a .+ b .* x
                y .~ interval_censored.(Normal.(mu, s), 5.0)
                s ~ Exponential(1)
            end, _int_gcols(), (mu = [0.5, -0.25], s = 1.3))
    end
    @testset "gaussian column upper" begin
        _int_enzyme_check(quote
                mu = a .+ b .* x
                y .~ interval_censored.(Normal.(mu, s), hi)
                s ~ Exponential(1)
            end, _int_gcols(), (mu = [0.5, -0.25], s = 1.3))
    end
    @testset "poisson literal upper" begin
        _int_enzyme_check(quote
                eta = a .+ b .* x
                y .~ interval_censored.(Poisson.(exp.(eta)), 4)
            end, _int_pcols(), (eta = [0.1, -0.2],))
    end
    @testset "poisson column upper" begin
        _int_enzyme_check(quote
                eta = a .+ b .* x
                y .~ interval_censored.(Poisson.(exp.(eta)), ub)
            end, _int_pcols(), (eta = [0.1, -0.2],))
    end
end

# Reactant/XLA value+grad parity at an unconstrained probe (no oracle —
# native vs compiled), plus the traced program size.
function _int_reactant(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    post_q = prepare_query(built, bound, :sampler)
    u = [0.3 * sin(1.7i) for i in 1:built.layout.total]
    return Base.invokelatest(_int_reactant_measure, built, bound, post_q, u)
end

function _int_reactant_measure(built, bound, post_q, u)
    hlo = repr(Reactant.@code_hlo optimize = false post_q(Reactant.to_rarray(u)))
    native = post_q(u)
    compiled = Reactant.@compile post_q(Reactant.to_rarray(u))
    primal = Float64(compiled(Reactant.to_rarray(u)))
    q = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
    g = similar(u)
    val, _ = sampler_value_and_gradient!(q, g, u)
    cad = compile_ad_value_and_gradient(q.ad, Reactant.to_rarray(u))
    rval, rgrad = cad(Reactant.to_rarray(u))
    return (; lines = count(==('\n'), hlo), native, primal, val, g,
        rval = Float64(rval), rgrad = Array(rgrad))
end

@testset "interval under Reactant" begin
    prog = quote
        mu = a .+ b .* x
        y .~ interval_censored.(Normal.(mu, s), hi)
        s ~ Exponential(1)
    end
    @testset "gaussian column upper" begin
        fx = _int_reactant(prog, _int_gcols())
        @test fx.primal ≈ fx.native rtol = 1e-9
        @test fx.val ≈ fx.native rtol = 1e-12
        @test fx.rval ≈ fx.native rtol = 1e-9
        @test fx.rgrad ≈ fx.g rtol = 1e-8
    end
    @testset "traced program is O(1) in n_obs" begin
        small = _int_reactant(prog, _int_gcols())
        bigcols = Dict{Symbol,AbstractVector}(:y => vcat(_INT_YG, _INT_YG),
            :x => vcat(_INT_X, _INT_X), :hi => vcat(_INT_HI, _INT_HI))
        large = _int_reactant(prog, bigcols)
        @test small.lines == large.lines
    end
end

# Attempt the Poisson-interval HLO trace, catching the expected upstream
# failure (returns the exception, or `:traced` if Reactant ever closes
# the gap).
function _int_poisson_hlo_attempt()
    plan = lower_rkppl(quote
            eta = a .+ b .* x
            y .~ interval_censored.(Poisson.(exp.(eta)), ub)
        end, (:y, :x, :ub))
    bound = bind_data(plan, _int_pcols())
    built = build_kernel(bound)
    post_q = prepare_query(built, bound, :sampler)
    u = [0.3 * sin(1.7i) for i in 1:built.layout.total]
    return Base.invokelatest(_int_try_hlo, post_q, u)
end

function _int_try_hlo(post_q, u)
    try
        Reactant.@code_hlo optimize = false post_q(Reactant.to_rarray(u))
        return :traced
    catch err
        return err
    end
end

@testset "poisson interval XLA gap is pinned upstream" begin
    # `poisson.cdf` lowers through `SpecialFunctions.gamma_inc`, which
    # has no Reactant tracing rule (the hurdle truncated-Poisson note).
    # If this test starts failing, the gap closed: promote Poisson
    # interval to the trio testset above and delete this pin.
    err = _int_poisson_hlo_attempt()
    @test err isa MethodError
    if err isa MethodError
        @test nameof(err.f) === :gamma_inc
    end
end

# I1/I2 parity probes (N=80; Xoshiro(90210)/Xoshiro(90211) recipes — see
# the SB-request todo on BayesianRegressionModels:rk:parity-fam-interval;
# vectors inlined so the test is immune to RNG/Distributions drift.
# Stable byte hashes:
# I1 = bytes2hex(sha256(vcat(reinterpret(UInt8, x1), reinterpret(UInt8, y1), reinterpret(UInt8, hi1))))
#    = f859b733502222f25f8663c876c2bb6d25c44c72acc6cf040805cae5a52067c2
# I2 = bytes2hex(sha256(vcat(reinterpret(UInt8, x2), reinterpret(UInt8, y2), reinterpret(UInt8, ub2))))
#    = fcafb293baa16dc421cbe5b2da42a4a6bd1006c0e7840495495bb3cb419b8c11
# ).
const _INT_I1_X = Float64[
    0.5758533047695249, 0.4857141711633878, 0.6538203494355899, 1.4193234599207312, -0.4514319327003183, -1.4746325529508209, 0.983893112973223, 0.49050313965876124,
    0.7412701070366418, 1.9268510476215033, -0.037800435968808103, -0.5193904972478445, -0.888569333320613, -0.20068282082992028, 1.0857722257663807, 1.1922284560249587,
    0.9996196641822946, 0.3384848824944231, 0.38867901890052836, 1.0617874158285001, 0.9055704794132495, -0.8851057419850155, 0.38478538362574716, 0.8196021880362718,
    -0.16183449693424415, -1.6661590873437002, -1.7558274944931618, -0.30852626515741205, 0.9354108833414416, 1.2513988191846157, -1.2525312759085387, -0.5130937201978506,
    0.35724763435374457, -0.1339485263218724, -1.352002946213746, 0.3198333292357278, 0.6480596047812515, 0.5272764452538399, 1.648358638192172, -0.03860687973901382,
    -0.7187465430930281, 0.09118456056109493, -1.2499085572409379, -0.11869599732684115, 0.3801889547180739, 1.1498757552043213, -0.5991488171755999, 0.6871210152410007,
    -1.4985385725822922, -0.4498497423427658, 0.3867440790695889, 1.0229195684163674, -0.366796483682315, 1.3359916208509484, -0.9747869832878195, -0.10803697767736899,
    2.676121679348735, -0.2898959267603609, -0.4655903869464306, 0.6961584921288654, -0.2813379109719391, 0.3809512365256197, 0.8243538801009267, 0.9990596722275582,
    0.8312898562110762, -2.1115276340260927, -0.786973250066989, -1.3201295282127972, -0.020489979954670657, -0.22045856597966493, -0.5370248620102156, -1.3990394869529965,
    -0.821056505811844, 0.33557397349286494, 1.2172934000078957, 0.3330006116629562, 0.19322528639851796, -2.205471852205272, -1.424080842546198, 0.27987082182521206,
]
const _INT_I1_Y = Float64[
    0.06305789615761542, 1.003552331319971, 0.43402454269981927, 2.535971482030315, 2.046426700141506, 1.5101385388656638, -0.49073383415454586, 0.8474302301346859,
    1.35518591829801, -1.9557689697071465, -0.22902916779436866, 1.8468327936780773, 1.9887011715384366, -1.9804239405515833, 1.6055129360559603, 0.340062578362734,
    0.762870973233669, 0.1306396863341327, 1.3779432494906136, -1.1820748592189751, 1.2627003125741743, 0.9034773245478149, 1.325562063927666, 0.3710752000402805,
    0.30582160124874924, 1.6547708861777013, -0.5805486138942717, -1.1041279590310094, 0.05805630485863822, -0.6851850258148979, -0.4479450827932774, 0.9512940832617406,
    -0.0642039517047745, 0.41287427344546607, -0.08382513029988115, -1.265213175414314, -0.7053947693842126, -0.685315473117154, -2.15402436152449, 1.2634884838291236,
    2.3380604143936194, -0.04505241738808574, 2.926580158969046, 0.14043362019478906, -0.7951911258016797, 0.7719181260461094, 0.8460976063135895, 0.5663769192742405,
    2.39122988429696, -0.17069157701764215, 1.2518124116179574, -0.32930682041609954, -0.37023335682083913, -0.4901375002439431, 1.3167838968661085, -0.0017836908992867606,
    -1.4431187658990785, 2.656373046811603, 2.290129671577981, -0.6923400339257004, 1.0067256054700298, 1.3965874417144755, 2.193323924894746, 1.8954234894550928,
    0.4004397198561234, 0.1383254274994905, 0.0059516285104042055, 1.2039196951598479, -1.1864999381179038, 0.5165522501354486, 0.3794515942243328, 1.3568945301947182,
    1.805755046924412, 0.012055533576156796, -0.7608412564771381, 1.1381851866855146, 0.12695427678610371, -0.1027359088255928, 1.8690251296168179, 1.779689209956922,
]
const _INT_I1_HI = Float64[
    0.7948794119906133, 1.7669753312612553, 1.6193523084453305, 3.318828207340093, 3.1704578961467433, 2.3990353416192214, 0.25522037088259947, 2.4373370622140964,
    2.304159912912381, 0.9443232223475015, 0.7746876121702105, 3.61489060788837, 3.0155938850241233, 0.24151093224946463, 2.3994726034621676, 1.714829116834185,
    2.238709856845375, 1.0989099828956579, 4.814491709640376, -0.29971765480301493, 3.8343975640409274, 3.5122266123821877, 1.9431144309630437, 1.8215796159456061,
    0.9685598796500756, 2.2601470103201278, 0.3059614001362156, -0.45826435767375095, 0.8897218335249972, 0.22004915584820262, 0.9864198849747264, 3.3570255693498696,
    1.7056618320131534, 2.449985664606955, 0.6869126378050536, 0.12626888825877747, 0.22856599773398328, 0.39491396913255106, -1.1026504556555312, 1.8656245182417974,
    4.382333679885432, 0.757182772092226, 5.502008179915064, 0.940636371745864, -0.02143585297458539, 1.375981358510211, 2.3137227240368725, 2.6341779698159122,
    3.2944814920735124, 1.0669880200431106, 1.7600802443522752, 0.8565030464601622, 0.5630366812136162, 1.490540593043998, 2.280979616925729, 2.4773470890713933,
    -0.8355644353450437, 4.200509670605125, 2.9050433721082496, 0.1657941010298976, 1.7192683871556922, 2.494752502228777, 3.4117938805315418, 2.9215377633319193,
    1.447052067333816, 1.355770578616768, 1.1784243989314775, 2.3442968752977134, -0.3576193529781734, 1.4099231897547222, 1.1220579320392425, 2.1111330282462872,
    3.423412189621356, 0.7637938724082541, 1.2183345523622284, 2.2572041593098917, 0.9837828118697166, 1.7353626283354957, 3.8267130775651736, 2.965059305120588,
]
const _INT_I2_X = Float64[
    -0.7010030650441363, 1.1532790123047962, -0.19915966223167772, 1.2364373087933493, 0.2045579549771363, -0.8265829297187625, 0.5664488429528816, -0.2315428108857454,
    -0.7306138451620792, 1.1558660845089141, 0.33827712247162633, 0.34944345292818846, 0.5921165450211545, -0.05210795448850637, -0.010892940655554989, -1.5350363175033266,
    -0.7712155395604448, -0.2952159247624898, -1.049146648232547, 0.561925616000451, -1.0075651479949725, 1.1823498368253302, -0.8124269871044666, 0.11375266702457076,
    0.23867260068447393, -0.5004779954235872, -0.9169287918577902, -1.3703351218982789, 1.117500063486037, -0.2048952936281121, -0.2613862614120609, -1.1834491628236383,
    0.244692127263438, 1.9982601284329944, -1.1949263830397807, -0.7209971991601043, 1.3398427458410977, 0.47794789623354833, 0.4102668699191212, 0.37049244044748786,
    1.2298372385535432, -0.3505342494586053, 1.014421811884107, -0.22270363091088533, 1.0940821469117858, -1.0078479022727382, 0.030552720739414396, 0.2280778308686943,
    0.4224250797203201, 0.12814157868524576, 2.3330866004579227, 0.3408426992554014, 1.7139678747861677, -0.4467736793335272, -0.26916867314580045, 0.2680182370019038,
    0.2994104867320347, 0.5981573601414407, 0.4694658817287437, -0.36230886803399026, -0.9105993575695183, 1.091356657185403, 0.8338941216735699, 2.161173310380576,
    0.7177664670593352, -1.0745134637924403, -0.04908501287806115, -0.48368223122509946, 1.3438981216290344, -2.1879744595214317, 0.3815015514420813, 2.045038772108382,
    -0.7417611499761906, 0.5075713548681087, -0.8551693594856419, 1.142282216859539, 0.7620040711927162, -2.1338672639693477, -0.10387482602230935, -1.2927525268308362,
]
const _INT_I2_Y = Int[
    1, 1, 5, 3, 2, 0, 1, 1,
    2, 1, 4, 2, 0, 2, 4, 0,
    0, 1, 0, 1, 0, 1, 0, 1,
    1, 2, 3, 1, 0, 3, 1, 1,
    3, 1, 0, 0, 2, 2, 1, 1,
    2, 2, 3, 1, 3, 3, 0, 3,
    4, 1, 3, 3, 4, 3, 0, 1,
    2, 2, 2, 1, 3, 1, 2, 3,
    1, 1, 1, 1, 3, 1, 1, 1,
    0, 2, 1, 1, 1, 1, 1, 1,
]
const _INT_I2_UB = Int[
    4, 2, 6, 6, 5, 1, 2, 4,
    5, 4, 5, 4, 3, 5, 6, 1,
    3, 4, 1, 4, 3, 4, 1, 2,
    4, 5, 5, 2, 3, 5, 4, 4,
    4, 2, 1, 1, 5, 5, 3, 2,
    4, 3, 5, 4, 5, 4, 2, 6,
    7, 3, 5, 6, 6, 4, 2, 3,
    5, 5, 4, 2, 4, 2, 3, 6,
    4, 4, 3, 3, 4, 2, 4, 2,
    1, 5, 4, 3, 4, 2, 3, 3,
]
_int_i1_cols() = Dict{Symbol,AbstractVector}(:y => copy(_INT_I1_Y),
    :x => copy(_INT_I1_X), :hi => copy(_INT_I1_HI))
_int_i2_cols() = Dict{Symbol,AbstractVector}(:y => copy(_INT_I2_Y),
    :x => copy(_INT_I2_X), :ub => copy(_INT_I2_UB))

@testset "interval probe vectors match published hashes" begin
    @test bytes2hex(sha256(vcat(reinterpret(UInt8, _INT_I1_X),
        reinterpret(UInt8, _INT_I1_Y),
        reinterpret(UInt8, _INT_I1_HI)))) ==
        "f859b733502222f25f8663c876c2bb6d25c44c72acc6cf040805cae5a52067c2"
    @test bytes2hex(sha256(vcat(reinterpret(UInt8, _INT_I2_X),
        reinterpret(UInt8, _INT_I2_Y),
        reinterpret(UInt8, _INT_I2_UB)))) ==
        "fcafb293baa16dc421cbe5b2da42a4a6bd1006c0e7840495495bb3cb419b8c11"
end

# I1/I2 RK-side pins at the SB-parity u probes (RK value vs the hand
# oracle; the SB-literal assertions join this testset when the peer
# brief lands).
@testset "interval I1/I2 probe pins" begin
    @testset "I1 interval_gauss" begin
        # SB: mu ~ 1 + x1; effect(mu, Intercept) ~ Normal(0, 1);
        # effect(mu, x1) ~ Normal(0, 1); s ~ Exponential(1);
        # y1 ~ interval_censored(Normal(mu, s); upper=hi1);
        # u = [0.6, -0.2, 0.1].
        _, _, kern, lay = _int_query(quote
                mu = a .+ b .* x
                y .~ interval_censored.(Normal.(mu, s), hi)
                s ~ Exponential(1)
            end, _int_i1_cols())
        @test coordinate_names(lay) ==
            [Symbol("mu.Intercept"), Symbol("mu.x"), :s]
        u = [0.6, -0.2, 0.1]
        want = _int_gauss_oracle(_INT_I1_Y, _INT_I1_X, _INT_I1_HI, u[1],
            u[2], exp(u[3]))
        @test Base.invokelatest(kern, u) ≈ want rtol = 1e-12
    end
    @testset "I2 interval_poisson" begin
        # SB: eta ~ 1 + x2; effect(eta, Intercept) ~ Normal(0, 1);
        # effect(eta, x2) ~ Normal(0, 1);
        # y2 ~ interval_censored(Poisson(exp(eta)); upper=ub2);
        # u = [0.35, 0.45].
        _, _, kern, lay = _int_query(quote
                eta = a .+ b .* x
                y .~ interval_censored.(Poisson.(exp.(eta)), ub)
            end, _int_i2_cols())
        @test coordinate_names(lay) ==
            [Symbol("eta.Intercept"), Symbol("eta.x")]
        u = [0.35, 0.45]
        want = _int_pois_oracle(_INT_I2_Y, _INT_I2_X, _INT_I2_UB, u[1],
            u[2])
        @test Base.invokelatest(kern, u) ≈ want rtol = 1e-12
    end
end
