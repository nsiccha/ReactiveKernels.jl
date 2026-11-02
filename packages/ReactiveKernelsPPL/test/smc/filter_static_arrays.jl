using Test, Random, Distributions, SequentialMonteCarlo
using ReactiveKernels, ReactiveKernelsPPL

@testset "Calibrated array declarations in finite-state filters" begin
    models = (
        matrix = @rkppl(begin
            prior_scale = error("a calibrated prior must not execute")
            b[1:G, 1:2] .~ Normal(0, prior_scale)
            @plate for g in eachindex(y)
                y[g] ~ Normal(b[g, 2], 0.5)
            end
        end),
        tensor = @rkppl(begin
            b[1:G, 1:2, 1:1] .~ Normal()
            @plate for g in eachindex(y)
                y[g] ~ Normal(b[g, 2, 1], 0.5)
            end
        end),
        rows = @rkppl(begin
            eachrow(b[1:G, 1:2]) .~ MvNormal([0.0, 0.0], [1.0, 1.0])
            @plate for g in eachindex(y)
                y[g] ~ Normal(b[g, 2], 0.5)
            end
        end),
        cols = @rkppl(begin
            eachcol(b[1:2, 1:G]) .~ MvNormal([0.0, 0.0], [1.0, 1.0])
            @plate for g in eachindex(y)
                y[g] ~ Normal(b[2, g], 0.5)
            end
        end),
    )
    loops(ex) = ex isa Expr ? (ex.head === :for) + sum(loops, ex.args; init=0) : 0
    for (kind, model) in pairs(models), G in (2, 32)
        means = collect(range(-0.4, 0.4; length=G))
        b = kind === :cols ? permutedims(hcat(zeros(G), means)) :
            kind === :tensor ? reshape(hcat(zeros(G), means), G, 2, 1) :
            hcat(zeros(G), means)
        @test filter_parameter_names(model) == (:b,)
        pars = filter_parameters(model, (; b))
        @test pars.b == b
        @test pars.b !== b
        for fixed in (false, true)
            target = prepare_filter(model; model_id="static-$kind", groups=(; g=1:G),
                parameters=fixed ? (;) : pars, data=fixed ? (; b) : (;))
            @test loops(target.program) == 1 # data-sized groups remain a runtime loop
            @test isempty(target.prototype)
            state = initialize_filter(target; rng=Xoshiro(12), particles=4)
            y = means .+ 0.1
            state = filter_step(state, (; y))
            @test state.predictive_logscore ≈ sum(logpdf.(Normal.(means, 0.5), y))
            state = filter_step(state, (; y); groups=(; g=[G]))
            @test state.predictive_logscore ≈ logpdf(Normal(means[G], 0.5), y[G])
            @test state.group_times.g == [g == G ? 2 : 1 for g in 1:G]
            @test b == pars.b # preparation and scoring keep caller arrays read-only
        end
        @test filter_parameter_names(model; data=(; b)) == ()
        err = try
            prepare_filter(model; model_id="missing-$kind", groups=(; g=1:G))
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("needs a calibrated parameter value", sprint(showerror, err))
    end
end

@rkppl filter_static_matrix(G) = begin
    eachrow(z[1:G, 1:2]) .~ MvNormal([0.0, 0.0], [1.0, 1.0])
    z
end

@testset "Scoped calibrated matrix declarations" begin
    model = @rkppl begin
        coefficients ~ filter_static_matrix(2)
        y ~ Normal(coefficients[1, 2], 0.5)
    end
    draw = (; coefficients=(; z=[0.1 0.2; 0.3 0.4]))
    @test filter_parameter_names(model) == (Symbol("coefficients.z"),)
    pars = filter_parameters(model, draw)
    target = prepare_filter(model; model_id="scoped-matrix", parameters=pars)
    state = initialize_filter(target; rng=Xoshiro(2), particles=4)
    state = filter_step(state, (; y=0.6))
    @test state.predictive_logscore ≈ logpdf(Normal(0.2, 0.5), 0.6)
end
