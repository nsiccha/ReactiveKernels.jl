using Distributions, LogExpFunctions, ReactiveKernelsPPL, Test

@testset "indexed corpus responses preserve density and reverse" begin
    cases = (
        ("99_plate_32_gaussian", (; b0 = 0.3, sigma = 0.8), [-0.2, 0.4, 0.1]),
        ("99_plate_33_offset", (; sigma = 0.8), [-0.2, 0.4, 0.1]),
        ("99_plate_69_poisson", (; b0 = 0.3), [0, 2, 1]),
        ("99_plate_70_bernoulli", (; b0 = 0.3), [false, true, true]),
        ("99_plate_71_nb2", (; b0 = 0.3, phi = 1.4), [0, 2, 1]),
        ("99_plate_72_gamma", (; b0 = 0.3, alpha = 1.4), [0.3, 1.2, 0.8]),
        ("99_plate_73_beta", (; b0 = 0.3, kappa = 2.4), [0.2, 0.7, 0.5]),
        ("99_plate_74_studentt", (; b0 = 0.3, sigma = 0.8, nu = 4.2), [-0.2, 0.4, 0.1]),
        ("99_plate_76_gaussian_poisson", (; b0 = 0.3, sigma = 0.8), nothing),
    )
    for (name, initial, y) in cases
        @testset "$name" begin
            ast, data = _load_corpus_case(joinpath(_CORPUS_DIR, name * ".jl"))
            columns = name == "99_plate_33_offset" ?
                Dict(:dose => [0.5, 1.0, 1.5], :ls => [-0.2, 0.1, 0.3], :dv => y) :
                name == "99_plate_76_gaussian_poisson" ?
                Dict(:x1 => [-0.4, 0.2], :y1 => [-0.2, 0.4],
                    :x2 => [0.1, 0.4, 0.8], :y2 => [0, 2, 1]) :
                Dict(:dose => [0.5, 1.0, 1.5], :t => [0.2, 0.4, 0.8], :obs => y)
            saved = deepcopy(columns)
            bound = bind_data(lower_rkppl(ast, data; conditioned = data), columns)
            built = build_kernel(bound)
            u = unconstrain(built.layout, initial)
            function oracle(v)
                q = constrain(built.layout, v)
                prior = haskey(q, :b0) ? logpdf(Normal(), q.b0) : 0.0
                for parameter in (:sigma, :phi, :alpha)
                    if haskey(q, parameter)
                        value = getproperty(q, parameter)
                        prior += logpdf(Exponential(), value) + log(value)
                    end
                end
                for (parameter, shape, scale) in ((:kappa, 2.0, 1000.0), (:nu, 2.0, 0.1))
                    if haskey(q, parameter)
                        value = getproperty(q, parameter)
                        prior += logpdf(Gamma(shape, scale), value) + log(value)
                    end
                end
                if name == "99_plate_33_offset"
                    return prior + sum(logpdf.(Normal.(columns[:dose] ./ 10 .* exp.(columns[:ls]), q.sigma), y))
                elseif name == "99_plate_76_gaussian_poisson"
                    return prior + sum(logpdf.(Normal.(q.b0 .* columns[:x1], q.sigma), columns[:y1])) +
                        sum(logpdf.(Poisson.(exp.(q.b0 .* columns[:x2])), columns[:y2]))
                end
                eta = q.b0 .* columns[:dose] .* columns[:t]
                distributions = if name == "99_plate_32_gaussian"
                    Normal.(eta, q.sigma)
                elseif name == "99_plate_69_poisson"
                    Poisson.(exp.(eta))
                elseif name == "99_plate_70_bernoulli"
                    Bernoulli.(logistic.(eta))
                elseif name == "99_plate_71_nb2"
                    # Distributions uses success probability rather than mean.
                    NegativeBinomial.(q.phi, q.phi ./ (q.phi .+ exp.(eta)))
                elseif name == "99_plate_72_gamma"
                    Gamma.(q.alpha, exp.(eta) ./ q.alpha)
                elseif name == "99_plate_73_beta"
                    probability = logistic.(eta)
                    Beta.(probability .* q.kappa, (1 .- probability) .* q.kappa)
                else
                    LocationScale.(eta, q.sigma, TDist(q.nu))
                end
                return prior + sum(logpdf.(distributions, y))
            end
            _check_model_math(built, bound, u, oracle)
            @test columns == saved
        end
    end
end
