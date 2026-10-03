include("test_smooth_capabilities_helpers.jl")

@testset "smooth capability bases: cubic, TPS and tensor margins" begin
    data = _sc_data(); x, z, w = data[:x], data[:z], data[:w]
    saved = deepcopy(data)
    @test size.(cr_basis(x; k=4)) == ((20, 1), (20, 2))
    @test size.(tps_basis(x, z; k=6)) == ((20, 2), (20, 3))
    X, Z = tps_basis(x, z; k=6)
    # An isotropic TPS is invariant to rotations of its location plane.
    angle = 0.37
    Xr, Zr = tps_basis(cos(angle).*x .- sin(angle).*z,
        sin(angle).*x .+ cos(angle).*z; k=6)
    @test Z * Z' ≈ Zr * Zr' rtol=1e-9 atol=1e-10
    @test vec(sum(X; dims=1)) ≈ zeros(2) atol=1e-14
    one = t2_basis(x; k=4)
    @test all(isapprox.(one, cr_basis(x; k=4)))
    @test t2_basis(x, z; k=4) == t2_basis(x, z; k=(4, 4))
    three = t2_basis(x, z, w; k=(3, 4, 5))
    @test size.(three) == ((20, 7), (20, 6), (20, 4), (20, 6),
        (20, 4), (20, 12), (20, 8), (20, 12))
    # Independent row-wise products of each marginal null/range space.
    margins = map((a,k) -> ReactiveKernelsPPL._rk_apply_cr_spline(
        ReactiveKernelsPPL._rk_fit_cr_spline(a; k), a), (x,z,w), (3,4,5))
    for (block, mask) in zip(Base.tail(three), 7:-1:1)
        parts = [margins[j][iszero(mask & (1 << (3-j))) ? 1 : 2] for j in 1:3]
        expected = hcat(([parts[1][i,a] * parts[2][i,b] * parts[3][i,c]
            for i in 1:20] for c in axes(parts[3],2), b in axes(parts[2],2),
            a in axes(parts[1],2))...)
        @test block ≈ expected
    end
    @test data == saved
    # A model derives its basis size from bound data, then uses the result.
    fx = _sc_build(quote
        kk = min(length(x), 4)
        (X, Z) = cr_basis(x; k=kk)
        f ~ penalized_smooth(X, Z)
        y .~ Normal.(f, 1.0)
    end)
    XX, ZZ = cr_basis(x; k=4)
    @test _sc_node(fx, :likelihood) ≈ sum(logpdf.(Normal.(
        XX * fx.nt.f.b .+ ZZ * (fx.nt.f.sd .* fx.nt.f.z), 1), data[:y]))
    _sc_gradient(fx)
    # A positive scalar-data chain must not accidentally classify sampled
    # definitions as data. Changing a packed array's extent with a sampled
    # value is refused (fixed layout; Core constraints / P3).
    @test_throws SurfaceLoweringError lower_rkppl(quote
        kk ~ Normal(4,1)
        (X,Z) = cr_basis(x;k=kk)
        b[axes(Z,2)] .~ Normal.(0,1)
        y .~ Normal.(Z * b,1.0)
    end,(:x,:y);mod=@__MODULE__,conditioned=(:x,:y))
end

@testset "Matérn HSGP angular spectral normalization" begin
    data = _sc_data(); x, z = data[:x], data[:z]
    P, lam = hsgp_basis(x; k=4)
    sigma, rho = 1.3, 0.9
    for nu in (1.5, 2.5)
        a = sqrt(2nu) / rho
        expected = nu == 1.5 ?
            sqrt.(4sigma^2 * a^3 ./ (a^2 .+ lam[:,1]).^2) :
            sqrt.((16/3)*sigma^2 * a^5 ./ (a^2 .+ lam[:,1]).^3)
        @test hsgp_matern_sqrt_spd(lam, sigma, rho, nu) ≈ expected
        fx = _sc_build(quote
            (P, lam) = hsgp_basis(x; k=4)
            f ~ _sc_matern(P, lam, $nu)
            y .~ Normal.(f, 1.0)
        end)
        a = sqrt(2nu) / fx.nt.f.rho
        weight = nu == 1.5 ?
            sqrt.(4fx.nt.f.sigma^2 * a^3 ./ (a^2 .+ lam[:,1]).^2) :
            sqrt.((16/3)*fx.nt.f.sigma^2 * a^5 ./ (a^2 .+ lam[:,1]).^3)
        @test _sc_node(fx, :likelihood) ≈ sum(logpdf.(
            Normal.(P * (weight .* fx.nt.f.z), 1), data[:y]))
        _sc_gradient(fx)
    end
    _, lam2 = hsgp_basis(x, z; k=(2,3))
    r = [0.7,1.2]
    @test hsgp_matern_sqrt_spd(lam2, sigma, r, 1.5).^2 ≈
        sigma^2 * (4pi * 1.5 * 3^1.5 * prod(r)) ./
            (3 .+ lam2 * r.^2).^2.5
    @test hsgp_matern_sqrt_spd(lam2, sigma, rho, 1.5) ≈
        hsgp_matern_sqrt_spd(lam2, sigma, fill(rho,2), 1.5)
end

@testset "periodic tensor and grouped HSGP weights" begin
    data = _sc_data(); x, z, g = data[:x], data[:z], data[:g]
    saved = deepcopy(data)
    P, h = hsgp_periodic_basis(x,z; k=(2,3),period=(2.0,3.0))
    @test size(P) == (20,34) && size(h) == (34,2)
    # Includes terms varying along only one margin, excludes global DC.
    @test any(iszero, h[:,1]) && any(iszero, h[:,2])
    @test all(any(!iszero, row) for row in eachrow(h))
    sigma, rho = 1.2, [0.8,1.1]
    expected = [sigma * prod(sqrt((j==0 ? 1 : 2) *
        exp(-1/rho[d]^2) * besseli(j,1/rho[d]^2))
        for (d,j) in enumerate(row)) for row in eachrow(h)]
    @test hsgp_periodic_sqrt_spd(h,sigma,rho) ≈ expected
    Pg, hg = hsgp_periodic_basis(x,z;k=(2,3),period=(2.0,3.0),by=g)
    @test hg == h && size(Pg) == (20,102)
    @test Pg * ones(102) ≈ P * ones(34)
    for axes_ in ((x,), (x,z))
        PHI, harm = hsgp_periodic_basis(axes_...;k=2,period=2.0,by=g)
        sig, r = [0.9,1.2,0.7], [0.8,1.1,1.3]
        oracle = vec(hcat([hsgp_periodic_sqrt_spd(harm,sig[j],r[j])
            for j in 1:3]...)')
        @test hsgp_periodic_grouped_sqrt_spd(harm,sig,r) ≈ oracle
        @test hsgp_periodic_grouped_sqrt_spd(harm,sig,1.1) ≈
            vec(hcat([hsgp_periodic_sqrt_spd(harm,sig[j],1.1) for j in 1:3]...)')
    end
    for axes_ in ((:x,), (:x,:z))
        basis = Expr(:call,:hsgp_periodic_basis,
            Expr(:parameters,Expr(:kw,:k,2),Expr(:kw,:period,2.0),Expr(:kw,:by,:g)),axes_...)
        fx = _sc_build(quote
            (P,h) = $basis
            f ~ _sc_grouped_periodic(P,h,g)
            y .~ Normal.(f,1.0)
        end)
        PP, hh = hsgp_periodic_basis((data[a] for a in axes_)...;k=2,period=2.0,by=g)
        weights = vec(hcat([hsgp_periodic_sqrt_spd(hh,fx.nt.f.sigma[j],fx.nt.f.rho[j])
            for j in 1:3]...)')
        @test _sc_node(fx,:likelihood) ≈ sum(logpdf.(Normal.(PP * (weights .* fx.nt.f.z),1),data[:y]))
        _sc_gradient(fx)
    end
    @test data == saved
end

@testset "smooth values compose with ordinary Julia arithmetic" begin
    data = _sc_data(); x, y = data[:x], data[:y]
    for basis in (:spline,:hsgp)
        defs = basis === :spline ? quote
            (X,Z) = tps_basis(x;k=4); f ~ penalized_smooth(X,Z)
        end : quote
            (P,lam) = hsgp_basis(x;k=1); f ~ hsgp_effect(P,lam)
        end
        for (expr, oracle) in (
                (:(f .+ f), (nt,f) -> 2 .* f),
                (:(a .- f), (nt,f) -> nt.a .- f),
                (:(a .+ 2.0 .* f), (nt,f) -> nt.a .+ 2 .* f),
                (:(a .+ reuse), (nt,f) -> nt.a .+ f),
                (:(a .+ f .+ x), (nt,f) -> nt.a .+ f .+ x),
                (:(a .+ f .+ b .* x), (nt,f) -> nt.a .+ f .+ nt.b .* x))
            ast = quote a ~ Normal(0,1); b ~ Normal(0,1) end
            append!(ast.args,defs.args)
            append!(ast.args,[:(reuse=f),:(mu=$expr),:(y .~ Normal.(mu,1.0))])
            fx = _sc_build(ast)
            f = if basis === :spline
                X,Z = tps_basis(x;k=4)
                X * fx.nt.f.b .+ Z * (fx.nt.f.sd .* fx.nt.f.z)
            else
                P,lam = hsgp_basis(x;k=1)
                @test length(fx.nt.f.z) == 1
                P * (hsgp_sqrt_spd(lam,fx.nt.f.sigma,fx.nt.f.rho) .* fx.nt.f.z)
            end
            @test _sc_node(fx,:likelihood) ≈ sum(logpdf.(Normal.(oracle(fx.nt,f),1),y))
            _sc_gradient(fx)
        end
    end
    fx = _sc_build(quote
        (X,Z) = tps_basis(x;k=4)
        f ~ penalized_smooth(X,Z)
        mu = f .+ x
        nu = f .- x
        y .~ Normal.(mu,1.0)
        z .~ Normal.(nu,1.0)
    end)
    X,Z = tps_basis(x;k=4)
    f = X * fx.nt.f.b .+ Z * (fx.nt.f.sd .* fx.nt.f.z)
    @test _sc_node(fx,:likelihood) ≈ sum(logpdf.(Normal.(f .+ x,1),y)) +
        sum(logpdf.(Normal.(f .- x,1),data[:z]))
    _sc_gradient(fx)
    fx = _sc_build(quote
        (X,Z) = tps_basis(x;k=4)
        f ~ penalized_smooth(X,Z)
        s ~ Dirichlet([1.0,1.0])
        m ~ monotonic(c,s)
        mu = f .+ m
        y .~ Normal.(mu,1.0)
    end)
    f = X * fx.nt.f.b .+ Z * (fx.nt.f.sd .* fx.nt.f.z)
    m = cumsum(vcat(0.0,fx.nt.s))[data[:c]]
    @test _sc_node(fx,:likelihood) ≈ sum(logpdf.(Normal.(f .+ m,1),y))
    _sc_gradient(fx)
end

@testset "smooth hyper-priors retain authored distribution semantics" begin
    for basis in (:spline,:hsgp,:periodic),
            prior in (:(HalfNormal(2)),:(HalfCauchy(2)),:(Beta(1,1)),
            :(Normal(0,s)),:(truncated(Normal(0,1),0,Inf)))
        ast = quote
            s ~ Exponential(1)
            sd ~ $prior
        end
        body = basis === :spline ? quote
                (X,Z) = tps_basis(x;k=4)
                b[axes(X,2)] .~ Normal.(0,1)
                raw[axes(Z,2)] .~ Normal.(0,1)
                mu = X * b .+ Z * (sd .* raw)
            end : basis === :hsgp ? quote
                (P,lam) = hsgp_basis(x;k=4)
                rho ~ LogNormal(0,1)
                raw[axes(P,2)] .~ Normal.(0,1)
                mu = P * (hsgp_sqrt_spd(lam,sd,rho) .* raw)
            end : quote
                (P,h) = hsgp_periodic_basis(x;k=4,period=2.0)
                rho ~ LogNormal(0,1)
                raw[axes(P,2)] .~ Normal.(0,1)
                mu = P * (hsgp_periodic_sqrt_spd(h,sd,rho) .* raw)
            end
        append!(ast.args,body.args)
        push!(ast.args,:(y .~ Normal.(mu,1.0)))
        fx = _sc_build(ast)
        dist = prior == :(HalfNormal(2)) ? truncated(Normal(0,2),0,Inf) :
            prior == :(HalfCauchy(2)) ? truncated(Cauchy(0,2),0,Inf) :
            prior == :(Beta(1,1)) ? Beta(1,1) :
            prior == :(Normal(0,s)) ? Normal(0,fx.nt.s) : truncated(Normal(),0,Inf)
        expected = logpdf(Exponential(1),fx.nt.s) + logpdf(dist,fx.nt.sd) +
            sum(logpdf.(Normal(),fx.nt.raw))
        expected += basis === :spline ? sum(logpdf.(Normal(),fx.nt.b)) :
            logpdf(LogNormal(0,1),fx.nt.rho)
        @test _sc_node(fx,:prior) ≈ expected
        _sc_gradient(fx)
    end
end

@testset "multi-axis spline and grouped HSGP models" begin
    data = _sc_data()
    for basis in (:(tps_basis(x,z;k=6)),:(t2_basis(x;k=4)))
        fx = _sc_build(quote
            (X,Z) = $basis
            f ~ penalized_smooth(X,Z)
            y .~ Normal.(f,1.0)
        end)
        X,Z = basis == :(tps_basis(x,z;k=6)) ?
            tps_basis(data[:x],data[:z];k=6) : t2_basis(data[:x];k=4)
        expected = X * fx.nt.f.b .+ Z * (fx.nt.f.sd .* fx.nt.f.z)
        @test _sc_node(fx,:likelihood) ≈ sum(logpdf.(Normal.(expected,1),data[:y]))
        _sc_gradient(fx)
    end
    ast = _sc_tensor_body((3,4,5))
    push!(ast.args, :(y .~ Normal.(f,1.0)))
    fx = _sc_build(ast)
    X, blocks... = t2_basis(data[:x],data[:z],data[:w];k=(3,4,5))
    expected = X * fx.nt.b
    for (j,name) in enumerate((:rrr,:rrn,:rnr,:rnn,:nrr,:nrn,:nnr))
        expected = expected .+ blocks[j] * (fx.nt.sd[j] .* getproperty(fx.nt,name))
    end
    @test _sc_node(fx,:likelihood) ≈ sum(logpdf.(Normal.(expected,1),data[:y]))
    _sc_gradient(fx)
    fx = _sc_build(quote
        (P,lam) = hsgp_basis(x,z;k=(2,3),by=g .+ 1)
        f ~ hsgp_grouped_effect(P,lam,g)
        y .~ Normal.(f,1.0)
    end)
    P,lam = hsgp_basis(data[:x],data[:z];k=(2,3),by=data[:g])
    rho = max.(exp.(fx.nt.f.rho_mu .+ fx.nt.f.rho_sd .* fx.nt.f.rho_z),
        maximum(hsgp_rho_floors(lam)))
    sigma = exp.(fx.nt.f.sigma_mu .+ fx.nt.f.sigma_sd .* fx.nt.f.sigma_z)
    weights = vec(hcat([hsgp_sqrt_spd(lam,sigma[j],rho[j]) for j in 1:3]...)')
    @test _sc_node(fx,:likelihood) ≈ sum(logpdf.(
        Normal.(P * (weights .* fx.nt.f.z),1),data[:y]))
    _sc_gradient(fx)
end

@testset "whole covariance leaves in plates and matrix-location GP values" begin
    grid = [-0.2,0.4,0.9]
    ys = [0.3,0.6]
    @test_throws SurfaceLoweringError lower_rkppl(quote
        @plate for i in eachindex(y)
            y[i] ~ Normal(grid,1.0)
        end
    end,(:y,:grid); mod=@__MODULE__, conditioned=(:y,:grid))
    fx = _sc_build(quote
        rho ~ LogNormal(0,1)
        @plate for i in eachindex(y)
            K = gp_exp_quad_cov(grid,1.0,rho,1e-9)
            cs = cumsum(grid)
            y[i] ~ Normal(sum(K) + sum(cs),1.0)
        end
    end,Dict{Symbol,Any}(:y=>ys,:grid=>grid))
    K = [exp(-(x-z)^2/(2fx.nt.rho^2)) + (i==j ? 1e-9 : 0.0)
        for (i,x) in enumerate(grid),(j,z) in enumerate(grid)]
    @test _sc_node(fx,:likelihood) ≈ sum(logpdf.(Normal(sum(K)+sum(cumsum(grid)),1),ys))
    _sc_gradient(fx)
    locations = [-0.2 0.1;0.4 -0.3;0.9 0.7]
    saved = copy(locations)
    for covariance in (:gp_exp_quad_cov,:gp_periodic_cov)
        call = covariance === :gp_exp_quad_cov ?
            :(gp_exp_quad_cov(locations,sigma,rho,1e-7)) :
            :(gp_periodic_cov(locations,sigma,rho,2.0,1e-7))
        fx = _sc_build(quote
            rho ~ LogNormal(0,1)
            sigma ~ LogNormal(0,1)
            raw[1:3] .~ Normal.(0,1)
            f = gp_chol_latent($call,raw)
            y .~ Normal.(f[oi],1.0)
        end,Dict{Symbol,Any}(:y=>[0.3,-0.2,0.1],:locations=>locations,:oi=>[3,1,2]))
        K = [begin
            dist = norm(locations[i,:] - locations[j,:])
            arg = covariance === :gp_exp_quad_cov ? -dist^2/(2fx.nt.rho^2) :
                -2sin(pi * dist/2)^2/fx.nt.rho^2
            fx.nt.sigma^2 * exp(arg) + (i==j ? 1e-7 : 0)
        end for i in 1:3,j in 1:3]
        expected = cholesky(Symmetric(K)).L * fx.nt.raw
        @test _sc_node(fx,:likelihood) ≈ sum(logpdf.(
            Normal.(expected[fx.bound.columns[:oi]],1),fx.bound.columns[:y]))
        _sc_gradient(fx)
    end
    scales = [0.7,1.3]
    @test gp_exp_quad_cov(locations,1.2,scales,1e-9) ≈
        [1.2^2 * exp(-sum(((locations[i,:]-locations[j,:])./scales).^2)/2) +
            (i==j ? 1e-9 : 0) for i in 1:3,j in 1:3]
    @test locations == saved
end
