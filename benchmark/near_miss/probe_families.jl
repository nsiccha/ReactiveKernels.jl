# Near-miss families. Each family holds one benchmark-shaped base model and
# sibling spellings / sibling laws on the same data. A cliff between siblings
# whose mathematics has the same cost structure marks an overspecific fast path.
include(joinpath(@__DIR__, "harness.jl"))

# usage: julia probe_families.jl [families, e.g. ABCDR] [rows]
const N = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 2000
rng = MersenneTwister(42)
x = randn(rng, N)
w = 0.5 .+ rand(rng, N)
z = randn(rng, N)
X = randn(rng, N, 5)
xd = Float64.(rand(rng, 1:5, N))             # discrete covariate, 5 values
g_sorted = repeat(1:50; inner = N ÷ 50)       # 50 groups, contiguous runs
g_shuf = g_sorted[randperm(rng, N)]           # same groups, no runs
h_cross = rand(rng, 1:20, N)                  # a second, crossed grouping
y = 0.3 .+ 0.7 .* x .+ 0.5 .* randn(rng, N)
ypos = exp.(y)
ybin = Int.(rand(rng, N) .< 1 ./ (1 .+ exp.(-(0.2 .+ 0.5 .* xd))))
ybinc = Int.(rand(rng, N) .< 1 ./ (1 .+ exp.(-(0.2 .+ 0.5 .* x))))
ycnt = [rand(rng, 0:6) for _ in 1:N]
ntr = fill(8, N)

fam = isempty(ARGS) ? "ABCDGR" : ARGS[1]
report_header()
record(r) = report(r, N)

# ---------------------------------------------------------------- A: Gaussian
if occursin('A', fam)
    A0 = @rkppl begin
        a ~ Normal(0, 5); b ~ Normal(0, 2); sigma ~ Exponential(1.0)
        y .~ Normal.(a .+ b .* x, sigma)
    end
    record(probe("A-gauss", "A0 Normal.(a+b*x, sigma)", A0, (; x), (; y)))
    A1 = @rkppl begin
        a ~ Normal(0, 5); b ~ Normal(0, 2); sigma ~ Exponential(1.0)
        mu = a .+ b .* x
        y .~ Normal.(mu, sigma)
    end
    record(probe("A-gauss", "A1 mu on its own line", A1, (; x), (; y)))
    A2 = @rkppl begin
        a ~ Normal(0, 5); beta[axes(X, 2)] .~ Normal.(0, 2); sigma ~ Exponential(1.0)
        y .~ Normal.(a .+ X * beta, sigma)
    end
    record(probe("A-gauss", "A2 a + X*beta (K=5)", A2, (; X), (; y)))
    A3 = @rkppl begin
        a ~ Normal(0, 5); b ~ Normal(0, 2); sigma ~ Exponential(1.0)
        y .~ Normal.(a .+ b .* x, sigma .* w)
    end
    record(probe("A-gauss", "A3 scale sigma .* w (w data)", A3, (; x, w), (; y)))
    A3p = @rkppl begin
        a ~ Normal(0, 5); b ~ Normal(0, 2); sigma ~ Exponential(1.0)
        y .~ Normal.(a .+ b .* x, sigma ./ sqrt.(precision))
    end
    record(probe("A-gauss", "A3p scale sigma ./ sqrt.(precision)", A3p,
                 (; x, precision = inv.(w .^ 2)), (; y)))
    A3m = @rkppl begin
        a ~ Normal(0, 5); b ~ Normal(0, 2); sigma ~ Exponential(1.0)
        mu = muladd.(b, x, a)
        y .~ Normal.(mu, sigma .* w)
    end
    record(probe("A-gauss", "A3m muladd location, weighted scale", A3m,
                 (; x, w), (; y)))
    A3t = @rkppl begin
        a ~ Normal(0, 5); b ~ Normal(0, 2); sigma ~ Exponential(1.0)
        y .~ Normal.(a .+ b .* x .* z, sigma .* w)
    end
    record(probe("A-gauss", "A3t product location, weighted scale", A3t,
                 (; x, z, w), (; y)))
    A4 = @rkppl begin
        a ~ Normal(0, 5); b ~ Normal(0, 2); sigma ~ Exponential(1.0)
        y .~ LogNormal.(a .+ b .* x, sigma)
    end
    record(probe("A-gauss", "A4 LogNormal sibling", A4, (; x), (; y = ypos)))
    A5 = @rkppl begin
        a ~ Normal(0, 5); b ~ Normal(0, 2); sigma ~ Exponential(1.0)
        @plate for i in eachindex(y)
            y[i] ~ Normal(a + b * x[i], sigma)
        end
    end
    record(probe("A-gauss", "A5 @plate loop spelling", A5, (; x), (; y)))
    A6 = @rkppl begin
        a ~ Normal(0, 5); b ~ Normal(0, 2); c ~ Normal(0, 2); sigma ~ Exponential(1.0)
        y .~ Normal.(a .+ b .* x .+ c .* x .^ 2, sigma)
    end
    record(probe("A-gauss", "A6 + c .* x.^2 (linear in theta)", A6, (; x), (; y)))
    A7 = @rkppl begin
        a ~ Normal(0, 5); b ~ Normal(0, 2); sigma ~ Exponential(1.0)
        y .~ Normal.(exp.(a .+ b .* x), sigma)
    end
    record(probe("A-gauss", "A7 exp link (control: per-row)", A7, (; x), (; y)))
    A8 = @rkppl begin
        a ~ Normal(0, 5); b ~ Normal(0, 2); sigma ~ Exponential(1.0); nu ~ Gamma(2, 0.1)
        y .~ StudentT.(nu, a .+ b .* x, sigma)
    end
    record(probe("A-gauss", "A8 StudentT (control: per-row)", A8, (; x), (; y)))
    A9 = @rkppl begin
        a ~ Normal(0, 5); b ~ Normal(0, 2); c ~ Normal(0, 1); d ~ Normal(0, 1)
        y .~ Normal.(a .+ b .* x, exp.(c .+ d .* z))
    end
    record(probe("A-gauss", "A9 modeled scale exp(c+d*z)", A9, (; x, z), (; y)))
    A10 = @rkppl begin
        m ~ Normal(0, 5); sigma ~ Exponential(1.0)
        y .~ Normal.(m, sigma)
    end
    record(probe("A-gauss", "A10 iid Normal.(m, sigma) scalar m", A10, (;), (; y)))
    A11 = @rkppl begin
        b ~ Normal(0, 2); sigma ~ Exponential(1.0)
        alpha[levels(g)] .~ Normal.(0, 2); gamma[levels(h)] .~ Normal.(0, 2)
        y .~ Normal.(alpha[g] .+ gamma[h] .+ b .* x, sigma)
    end
    record(probe("A-gauss", "A11 crossed alpha[g] + gamma[h]", A11, (; x, g = g_sorted, h = h_cross), (; y)))
    A12 = @rkppl begin
        sigma ~ Exponential(1.0)
        alpha[levels(g)] .~ Normal.(0, 2); beta[axes(X, 2)] .~ Normal.(0, 2)
        y .~ Normal.(alpha[g] .+ X * beta, sigma)
    end
    record(probe("A-gauss", "A12 alpha[g] + X*beta", A12, (; X, g = g_sorted), (; y)))
    A13 = @rkppl begin
        b ~ Normal(0, 2); sigma ~ Exponential(1.0)
        alpha[levels(g)] .~ Normal.(0, 2)
        y .~ Normal.(alpha[g] .+ b .* x, sigma)
    end
    record(probe("A-gauss", "A13 alpha[g] + b*x sorted g", A13, (; x, g = g_sorted), (; y)))
    record(probe("A-gauss", "A13s alpha[g] + b*x shuffled g", A13, (; x, g = g_shuf), (; y)))
    A14 = @rkppl begin
        b ~ Normal(0, 2); sigma ~ Exponential(1.0); nu ~ Gamma(2, 0.1)
        alpha[levels(g)] .~ Normal.(0, 2)
        y .~ StudentT.(nu, alpha[g] .+ b .* x, sigma)
    end
    record(probe("A-gauss", "A14 StudentT alpha[g] sorted", A14, (; x, g = g_sorted), (; y)))
    record(probe("A-gauss", "A14s StudentT alpha[g] shuffled", A14, (; x, g = g_shuf), (; y)))
end

# ------------------------------------------------- B: discrete-covariate GLMs
if occursin('B', fam)
    B0 = @rkppl begin
        a ~ Normal(0, 1); b ~ Normal(0, 2)
        y .~ BernoulliLogit.(a .+ b .* x)
    end
    record(probe("B-glm", "B0 BernoulliLogit, x discrete", B0, (; x = xd), (; y = ybin)))
    record(probe("B-glm", "B0c BernoulliLogit, x continuous", B0, (; x), (; y = ybinc)))
    B1 = @rkppl begin
        a ~ Normal(0, 1); b ~ Normal(0, 2)
        y .~ Bernoulli.(logistic.(a .+ b .* x))
    end
    record(probe("B-glm", "B1 Bernoulli(logistic), discrete", B1, (; x = xd), (; y = ybin)))
    record(probe("B-glm", "B1c Bernoulli(logistic), contin.", B1, (; x), (; y = ybinc)))
    B2 = @rkppl begin
        a ~ Normal(0, 1); b ~ Normal(0, 2)
        y .~ Poisson.(exp.(a .+ b .* x))
    end
    record(probe("B-glm", "B2 Poisson(exp), discrete", B2, (; x = xd), (; y = ycnt)))
    record(probe("B-glm", "B2c Poisson(exp), continuous", B2, (; x), (; y = ycnt)))
    B3 = @rkppl begin
        a ~ Normal(0, 1); b ~ Normal(0, 2)
        y .~ Binomial.(n, logistic.(a .+ b .* x))
    end
    record(probe("B-glm", "B3 Binomial(n, logistic), discr.", B3, (; x = xd, n = ntr), (; y = ycnt)))
    record(probe("B-glm", "B3c Binomial(n, logistic), cont.", B3, (; x, n = ntr), (; y = ycnt)))
    B4 = @rkppl begin
        a ~ Normal(0, 1); b ~ Normal(0, 2); phi ~ Exponential(1.0)
        y .~ NegativeBinomial2.(exp.(a .+ b .* x), phi)
    end
    record(probe("B-glm", "B4 NegBinomial2, discrete", B4, (; x = xd), (; y = ycnt)))
    record(probe("B-glm", "B4c NegBinomial2, continuous", B4, (; x), (; y = ycnt)))
    B5 = @rkppl begin
        a ~ Normal(0, 1); b ~ Normal(0, 2)
        @plate for i in eachindex(y)
            y[i] ~ Bernoulli(logistic(a + b * x[i]))
        end
    end
    record(probe("B-glm", "B5 @plate Bernoulli, discrete", B5, (; x = xd), (; y = ybin)))
    record(probe("B-glm", "B5c @plate Bernoulli, continuous", B5, (; x), (; y = ybinc)))
end

# ---------------------------------------------------- C: grouped intercepts
if occursin('C', fam)
    C0 = @rkppl begin
        m ~ Normal(0, 5); t ~ Exponential(1.0); b ~ Normal(0, 2); sigma ~ Exponential(1.0)
        alpha[levels(g)] .~ Normal.(m, t)
        y .~ Normal.(alpha[g] .+ b .* x, sigma)
    end
    record(probe("C-group", "C0 Normal, alpha[g] sorted g", C0, (; x, g = g_sorted), (; y)))
    record(probe("C-group", "C0s Normal, alpha[g] shuffled g", C0, (; x, g = g_shuf), (; y)))
    C1 = @rkppl begin
        m ~ Normal(0, 5); t ~ Exponential(1.0); b ~ Normal(0, 2)
        alpha[levels(g)] .~ Normal.(m, t)
        y .~ BernoulliLogit.(alpha[g] .+ b .* x)
    end
    record(probe("C-group", "C1 BernoulliLogit, sorted g", C1, (; x, g = g_sorted), (; y = ybinc)))
    record(probe("C-group", "C1s BernoulliLogit, shuffled g", C1, (; x, g = g_shuf), (; y = ybinc)))
    C2 = @rkppl begin
        m ~ Normal(0, 5); t ~ Exponential(1.0); b ~ Normal(0, 2)
        alpha[levels(g)] .~ Normal.(m, t)
        y .~ Poisson.(exp.(alpha[g] .+ b .* x))
    end
    record(probe("C-group", "C2 Poisson, sorted g", C2, (; x, g = g_sorted), (; y = ycnt)))
    record(probe("C-group", "C2s Poisson, shuffled g", C2, (; x, g = g_shuf), (; y = ycnt)))
    C3 = @rkppl begin
        m ~ Normal(0, 5); t ~ Exponential(1.0); b ~ Normal(0, 2); sigma ~ Exponential(1.0); nu ~ Gamma(2, 0.1)
        alpha[levels(g)] .~ Normal.(m, t)
        y .~ StudentT.(nu, alpha[g] .+ b .* x, sigma)
    end
    record(probe("C-group", "C3 StudentT, sorted g", C3, (; x, g = g_sorted), (; y)))
    record(probe("C-group", "C3s StudentT, shuffled g", C3, (; x, g = g_shuf), (; y)))
end

# ------------------------------------------- D: shared-parameter normalizers
# Every law below has per-row work plus terms that read only shared values.
# The per-row minimum is one transcendental (or none) per row.
if occursin('D', fam)
    D0 = @rkppl begin
        a ~ Normal(0, 1); b ~ Normal(0, 1); sigma ~ Exponential(1.0)
        y .~ Normal.(exp.(a .+ b .* x), sigma)
    end
    record(probe("D-shared", "D0 Normal, shared sigma", D0, (; x), (; y)))
    D1 = @rkppl begin
        a ~ Normal(0, 1); b ~ Normal(0, 1); sigma ~ Exponential(1.0); nu ~ Gamma(2, 0.1)
        y .~ StudentT.(nu, exp.(a .+ b .* x), sigma)
    end
    record(probe("D-shared", "D1 StudentT, shared nu sigma", D1, (; x), (; y)))
    D2 = @rkppl begin
        a ~ Normal(0, 1); b ~ Normal(0, 1); sigma ~ Exponential(1.0)
        y .~ LogNormal.(exp.(a .+ b .* x), sigma)
    end
    record(probe("D-shared", "D2 LogNormal, shared sigma", D2, (; x), (; y = ypos)))
    D3 = @rkppl begin
        a ~ Normal(0, 1); b ~ Normal(0, 1); alpha ~ Exponential(1.0)
        y .~ Gamma.(alpha, exp.(a .+ b .* x) ./ alpha)
    end
    record(probe("D-shared", "D3 Gamma, shared shape", D3, (; x), (; y = ypos)))
    D4 = @rkppl begin
        a ~ Normal(0, 1); b ~ Normal(0, 1); k ~ LogNormal(0.0, 0.3)
        y .~ Weibull.(k, exp.(a .+ b .* x))
    end
    record(probe("D-shared", "D4 Weibull, shared shape", D4, (; x), (; y = ypos)))
    D5 = @rkppl begin
        a ~ Normal(0, 1); b ~ Normal(0, 1); sigma ~ Exponential(1.0)
        y .~ Cauchy.(exp.(a .+ b .* x), sigma)
    end
    record(probe("D-shared", "D5 Cauchy, shared sigma", D5, (; x), (; y)))
    D6 = @rkppl begin
        a ~ Normal(0, 1); b ~ Normal(0, 1); sigma ~ Exponential(1.0)
        y .~ Laplace.(exp.(a .+ b .* x), sigma)
    end
    record(probe("D-shared", "D6 Laplace, shared sigma", D6, (; x), (; y)))
    D7 = @rkppl begin
        a ~ Normal(0, 1); b ~ Normal(0, 1); sigma ~ Exponential(1.0)
        y .~ Logistic.(exp.(a .+ b .* x), sigma)
    end
    record(probe("D-shared", "D7 Logistic, shared sigma", D7, (; x), (; y)))
end


# ------------------------------------------- G: count regressions, small and wide
# The shapes of the small count regressions in the benchmark suites (a cubic
# trend over 40 rows, with and without one effect per row) and their wide
# siblings over a design matrix.
if occursin('G', fam)
    yr = collect(range(-1.0, 1.0; length = 40)); yr2 = yr .^ 2; yr3 = yr .^ 3
    cnt40 = [rand(rng, 0:30) for _ in 1:40]; trials40 = fill(40, 40)
    G0 = @rkppl begin
        alpha ~ Normal(0, 10); b1 ~ Normal(0, 10); b2 ~ Normal(0, 10); b3 ~ Normal(0, 10)
        counts .~ Poisson.(exp.(alpha .+ b1 .* yr .+ b2 .* yr2 .+ b3 .* yr3))
    end
    record(probe("G-count", "G0 Poisson cubic, 40 rows", G0, (; yr, yr2, yr3), (; counts = cnt40)))
    G1 = @rkppl begin
        alpha ~ Normal(0, 10); b1 ~ Normal(0, 10); b2 ~ Normal(0, 10)
        counts .~ Binomial.(trials, logistic.(alpha .+ b1 .* yr .+ b2 .* yr2))
    end
    record(probe("G-count", "G1 Binomial quadratic, 40 rows", G1, (; yr, yr2, trials = trials40), (; counts = cnt40)))
    G2 = @rkppl begin
        alpha ~ Normal(0, 10); b1 ~ Normal(0, 10); b2 ~ Normal(0, 10); b3 ~ Normal(0, 10)
        s ~ Exponential(1.0)
        eps[axes(counts, 1)] .~ Normal.(0, s)
        counts .~ Poisson.(exp.(alpha .+ b1 .* yr .+ b2 .* yr2 .+ b3 .* yr3 .+ eps))
    end
    record(probe("G-count", "G2 Poisson cubic + row effects", G2, (; yr, yr2, yr3), (; counts = cnt40)))
    G3 = @rkppl begin
        a ~ Normal(0, 1); beta[axes(X, 2)] .~ Normal.(0, 1)
        y .~ Poisson.(exp.(a .+ X * beta))
    end
    record(probe("G-count", "G3 Poisson a + X*beta (K=5)", G3, (; X), (; y = ycnt)))
    G4 = @rkppl begin
        a ~ Normal(0, 1); beta[axes(X, 2)] .~ Normal.(0, 1)
        y .~ Binomial.(n, logistic.(a .+ X * beta))
    end
    record(probe("G-count", "G4 Binomial a + X*beta (K=5)", G4, (; X, n = ntr), (; y = ycnt)))
    G5 = @rkppl begin
        a ~ Normal(0, 1); beta[axes(X, 2)] .~ Normal.(0, 1)
        y .~ BernoulliLogit.(a .+ X * beta)
    end
    record(probe("G-count", "G5 BernoulliLogit a + X*beta", G5, (; X), (; y = ybinc)))
end

# ------------------------------------------------------ R: hand-written loops
if occursin('R', fam)
    include(joinpath(@__DIR__, "reference.jl"))
    import ReactiveKernelsPPL.SpecialFunctions: loggamma
    refs = Any[]
    u3 = 0.1 .* randn(MersenneTwister(1), 3); u4 = 0.1 .* randn(MersenneTwister(1), 4)
    u2 = 0.1 .* randn(MersenneTwister(1), 2); u54 = 0.1 .* randn(MersenneTwister(1), 54)
    push!(refs, time_ref("R normal a+b*x (row loop)", ref_normal, u3, x, y))
    push!(refs, time_ref("R normal exp link", ref_normal_exp, u3, x, y))
    push!(refs, time_ref("R studentt exp link", ref_student_exp, u4, x, y, loggamma))
    push!(refs, time_ref("R cauchy exp link", ref_cauchy_exp, u3, x, y))
    push!(refs, time_ref("R bernoulli-logit continuous", ref_bernoulli_logit, u2, x, ybinc))
    push!(refs, time_ref("R poisson continuous", ref_poisson, u2, x, ycnt, Float64(sum(ycnt)),
        sum(x .* ycnt), sum(loggamma.(ycnt .+ 1.0))))
    push!(refs, time_ref("R normal alpha[g] shuffled", ref_normal_group, u54, x, y, g_shuf, 50))
    push!(refs, time_ref("R normal alpha[g] sorted", ref_normal_group, u54, x, y, g_sorted, 50))
    @printf("\n%-34s %14s %12s %12s %8s %8s %6s\n", "hand reference", "value", "dens ns", "grad ns",
        "d ns/row", "g ns/row", "allocs")
    for r in refs
        @printf("%-34s %14.6f %12.0f %12.0f %8.2f %8.2f %6d\n", r.name, r.value, r.t_density,
            r.t_gradient, r.t_density / N, r.t_gradient / N, r.allocs)
    end
end
