using Test, ReactiveKernels, ReactiveKernelsPPL
import Distributions as PD

_positive_half(d) = PD.truncated(d, 0, Inf)
function _positive_prior_query(expr, data, oracle)
    plan = bind_data(lower_rkppl(expr, data; conditioned=keys(data)), data)
    built = build_kernel(plan)
    u = [0.12sin(i) for i in 1:built.layout.total]
    q = constrain(built.layout, u)
    query = prepare_query(built, plan, :prior)
    @test Base.invokelatest(query, u) ≈ oracle(q, built.layout) atol=1e-12 rtol=1e-12
    @test unconstrain(built.layout, q) ≈ u atol=1e-12
    return plan, built, u
end

@testset "positive priors have one normalized meaning in every slot" begin
    data = Dict(:y=>[0.1, -0.2, 0.3, 0.4], :x=>[-1., -0.3, 0.2, 1.], :g=>[1,2,1,2])
    for (rhs, d) in ((:(HalfNormal(2)), PD.Normal(0,2)),
                    (:(HalfCauchy(2)), PD.Cauchy(0,2)),
                    (:(truncated(Normal(0,2),0,Inf)), PD.Normal(0,2)),
                    (:(truncated(Cauchy(0,2),0,Inf)), PD.Cauchy(0,2)))
        expected = _positive_half(d)
        _positive_prior_query(quote s ~ $rhs; y .~ Normal.(s,1) end,
            Dict(:y=>data[:y]), (q,l)->PD.logpdf(expected,q.s))
        _positive_prior_query(quote
            a ~ Normal(0,1)
            r ~ varying_effect(g,[1]; sd=$rhs)
            mu = a .+ r
            y .~ Normal.(mu,1)
        end, Dict(:y=>data[:y],:g=>data[:g]), (q,l)->
            PD.logpdf(PD.Normal(),q.a) + sum(PD.logpdf.(expected,q.tau_g)) +
            sum(PD.logpdf.(PD.Normal(),q.z_flat_g)))
        _positive_prior_query(quote
            a ~ Normal(0,1)
            spline_basis(:s_x,x;k=4,sd=$rhs)
            mu = a .+ spline(:s_x)
            y .~ Normal.(mu,1)
        end, Dict(:y=>data[:y],:x=>data[:x]), (q,l)->PD.logpdf(PD.Normal(),q.a) +
            sum(PD.logpdf.(expected,q.sd_s_x)) + sum(PD.logpdf.(PD.Normal(),q.b_s_x_raw)))
        _positive_prior_query(quote
            a ~ Normal(0,1)
            hsgp_basis(:h_x,x;k=4,length_scale=$rhs,sd=$rhs)
            mu = a .+ hsgp(:h_x)
            y .~ Normal.(mu,1)
        end, Dict(:y=>data[:y],:x=>data[:x]), (q,l)->PD.logpdf(PD.Normal(),q.a) +
            PD.logpdf(expected,q.rho_h_x) + PD.logpdf(expected,q.sigma_h_x) +
            sum(PD.logpdf.(PD.Normal(),q.beta_raw_h_x)))
    end
    # Nonzero location and finite upper bound exercise the full normalizer,
    # rather than proving only a hard-coded log(2) adjustment.
    rhs = :(truncated(Normal(0.7,2),0.3,4))
    expected = PD.truncated(PD.Normal(0.7,2),0.3,4)
    _positive_prior_query(quote
        a ~ Normal(0,1)
        hsgp_basis(:h_x,x;k=4,length_scale=$rhs,sd=$rhs)
        mu = a .+ hsgp(:h_x)
        y .~ Normal.(mu,1)
    end, Dict(:y=>data[:y],:x=>data[:x]), (q,l)->PD.logpdf(PD.Normal(),q.a) +
        PD.logpdf(expected,q.rho_h_x) + PD.logpdf(expected,q.sigma_h_x) +
        sum(PD.logpdf.(PD.Normal(),q.beta_raw_h_x)))
    _positive_prior_query(quote
        a ~ Normal(0,1)
        hsgp_basis(:h_x,x;k=4)
        mu = a .+ hsgp(:h_x)
        y .~ Normal.(mu,1)
    end, Dict(:y=>data[:y],:x=>data[:x]), (q,l)->begin
        floor = only(e.lo for e in l.entries if e.name===:rho_h_x)
        PD.logpdf(PD.Normal(),q.a) + PD.logpdf(PD.truncated(PD.LogNormal(),floor,Inf),q.rho_h_x) +
            PD.logpdf(PD.LogNormal(),q.sigma_h_x) + sum(PD.logpdf.(PD.Normal(),q.beta_raw_h_x))
    end)
end

@testset "legacy positive spellings name normalized replacements" begin
    # Refused: non-Julia constructor keywords and silently constrained bare
    # distributions violate P3 and the user-approved stan-halves decision 0m1j3iz.
    for rhs in (:(Normal(0,2;lower=0)), :(Cauchy(0,2;lower=0)), :(Flat(;lower=0)), :(Flat(;lower=0,upper=2)))
        err = try
            lower_rkppl(quote s ~ $rhs; y .~ Normal.(s,1) end,(:y,);conditioned=(:y,))
            nothing
        catch e; e end
        @test err isa SurfaceLoweringError
        @test occursin(rhs.args[1]===:Flat ? "Exponential" : "truncated",sprint(showerror,err))
    end
    for rhs in (:(Normal(0,2)), :(Cauchy(0,2)), :(StudentT(3,0,2)))
        for declaration in (:(hsgp_basis(:h_x,x;k=4,sd=$rhs)), :(spline_basis(:h_x,x;k=4,sd=$rhs)))
            err = try
                lower_rkppl(Expr(:block, declaration, :(a ~ Normal(0,1)), :(y .~ Normal.(a,1))),(:x,:y);conditioned=(:y,))
                nothing
            catch e; e end
            @test err isa SurfaceLoweringError
            @test occursin("truncated",sprint(showerror,err))
        end
    end
    for rhs in (:(Normal(0,2)), :(Cauchy(0,2)),
                :(truncated(Exponential(2),0.3,4)))
        err = try
            lower_rkppl(quote
                r ~ varying_effect(g,[1];sd=$rhs)
                y .~ Normal.(r,1)
            end,(:g,:y);conditioned=(:y,))
            nothing
        catch e; e end
        @test err isa SurfaceLoweringError
        @test occursin(rhs.args[1]===:truncated ? "not supported" : "Half",sprint(showerror,err))
    end
end
