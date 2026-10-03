# Modern DAR declarations state the innovation prior in a library submodel.
# Every fixture below has an independent sequential density/gradient oracle.
function _dar_capability(n, form; beta_prior = :normal, scale_prior = :normal)
    distributions = Dict(:normal => Normal(0.2, 0.7),
        :exponential => Exponential(1.3), :half_normal => truncated(Normal(0, 0.7), 0, Inf),
        :half_cauchy => truncated(Cauchy(0, 0.7), 0, Inf),
        :interval => truncated(Normal(0.2, 0.7), 0, 2))
    spellings = Dict(:normal => :(Normal(0.2, 0.7)),
        :exponential => :(Exponential(1.3)), :half_normal => :(HalfNormal(0.7)),
        :half_cauchy => :(HalfCauchy(0.7)),
        :interval => :(truncated(Normal(0.2, 0.7), 0, 2)))
    shared = form === :shared
    decl = shared ? Expr(:block) : :(sigma_d ~ $(spellings[scale_prior]))
    first_arg = form === :literal_argument ? 0.5 : :beta
    second_arg = shared ? :beta : :sigma_d
    second_path = form === :two_paths ? :(d2 ~ differenced_ar1(beta, sigma_d)) : Expr(:block)
    location = Dict(:bare => :(d), :literal => :(2 .* d),
        :data => :(x .* d), :negative => :(-d), :nested => :(2 .* (d .+ x)),
        :computed => :(c .* d), :shared => :(d), :literal_argument => :(d),
        :two_paths => :(d .+ d2), :dual_role => :(beta .* x .+ d),
        :collision => :(dar_mu .+ d))
    collision = form === :collision ? :(dar_mu ~ Normal(0, 1)) : Expr(:block)
    program = quote
        beta ~ $(spellings[beta_prior])
        $decl
        $collision
        c = beta * beta
        d ~ differenced_ar1($first_arg, $second_arg)
        $second_path
        mu = $(location[form])
        y .~ Normal.(mu, 1.0)
    end
    filter!(ex -> !(ex isa Expr && ex.head === :block && isempty(ex.args)), program.args)
    x, y = [0.2cos(t) for t in 1:n], [0.3sin(t) for t in 1:n]
    f = _scan_cap_sampler(program, Dict{Symbol,Any}(:x => x, :y => y))
    function parameter(u, name, family)
        e = only(e for e in f.built.layout.entries if e.name === name)
        a = u[e.offset]
        if family === :interval
            v = 2 / (1 + exp(-a))
            return v, log(v) + log1p(-v / 2)
        elseif family !== :normal
            return exp(a), a
        end
        return a, 0.0
    end
    function oracle(u)
        beta, jbeta = parameter(u, :beta, beta_prior)
        sigma, jsigma = shared ? (beta, 0.0) : parameter(u, :sigma_d, scale_prior)
        lp = logpdf(distributions[beta_prior], beta) + jbeta
        shared || (lp += logpdf(distributions[scale_prior], sigma) + jsigma)
        extra = 0.0
        if form === :collision
            extra, _ = parameter(u, :dar_mu, :normal)
            lp += logpdf(Normal(), extra)
        end
        zentries = [e for e in f.built.layout.entries if e.kind === :scan]
        zblocks = [u[e.offset:(e.offset + e.size - 1)] for e in zentries]
        isempty(zblocks) && push!(zblocks, Float64[])
        paths = [zeros(n) for _ in zblocks]
        for (path, z) in zip(paths, zblocks)
            lp += sum(logpdf.(Normal(), z))
            increment = 0.0
            for t in 2:n
                increment = (form === :literal_argument ? 0.5 : beta) * increment + sigma * z[t - 1]
                path[t] = path[t - 1] + increment
            end
        end
        for t in 1:n
            d = paths[1][t]
            mu = form === :literal ? 2d : form === :data ? x[t] * d :
                form === :negative ? -d : form === :nested ? 2(d + x[t]) :
                form === :computed ? beta^2 * d : form === :two_paths ? d + paths[2][t] :
                form === :dual_role ? beta * x[t] + d : form === :collision ? extra + d : d
            lp += logpdf(Normal(mu, 1), y[t])
        end
        return lp
    end
    return f, oracle
end

@testset "DAR capabilities: explicit library trajectories" begin
    for form in (:bare, :literal, :data, :negative, :nested, :computed,
            :shared, :literal_argument, :two_paths, :dual_role, :collision)
        f, oracle = _dar_capability(4, form)
        @test isempty(f.plan.dar_paths)
        _scan_cap_check(f, oracle)
    end
    for n in (1, 8)
        f, oracle = _dar_capability(n, :bare)
        @test f.built.layout.total == 2 + n - 1
        _scan_cap_check(f, oracle)
    end
    for prior in (:exponential, :half_normal, :half_cauchy, :interval)
        for slot in (:beta, :scale)
            f, oracle = slot === :beta ? _dar_capability(4, :bare; beta_prior = prior) :
                _dar_capability(4, :bare; scale_prior = prior)
            _scan_cap_check(f, oracle)
        end
    end
end
