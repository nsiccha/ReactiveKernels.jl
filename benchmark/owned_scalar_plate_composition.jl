# Compositional derivative-graph comparison motivated by the PB/PBR study.
# Run in an environment containing the imports below, using the pinned Enzyme
# source/library reported at startup. Optional ARGS select modes from
# ordinary, cell, cell_plate, reduction, captured. The default runs all five.
# cell and cell_plate differ only in the generated plate reverse opt-in.
# The compared models explicitly collect the residual, so the plate receives
# a writable array shadow; generated plate reverse does not support RK's lazy
# cell-array shadows. Materialization cost is included in every comparison.
# Captured immutable closures are a control for source visibility. Both owned
# scalar variants must engage the generated rule. No whole-model rule is installed.
using ReactiveKernels, MutatingFunctions
using LinearAlgebra, Random, Statistics, BenchmarkTools, LogExpFunctions, Test
using DifferentiationInterface
import Enzyme
const RK = ReactiveKernels
BLAS.set_num_threads(1)
println("PROVENANCE ", VERSION, " RK=", pkgversion(RK), " Enzyme=", pkgversion(Enzyme),
    " source=", pkgdir(Enzyme), " library=", Enzyme.Compiler.Enzyme_jll.libEnzyme_path)

module ComposedProbe
using ReactiveKernels, LogExpFunctions, LinearAlgebra

# A local mathematical component, independent of the surrounding model.
@kernel logistic_cell(r::Float64, s::Float64) = begin
    z::Float64 = r / exp(s)
    t::Float64 = tanh(z / 2)
    v::Float64 = -s - log1pexp(z) - log1pexp(-z)
    dr::Float64 = -t / exp(s)
    ds::Float64 = z * t - 1
    return v, dr, ds
end
const cell = scalar_derivative_rule(logistic_cell; primal=:v,
    partials=(r=:dr, s=:ds), name=:logistic_cell)

# Same mathematics over an array, with the scalar partial reduced before the
# covector arrives. No model recognizer or backend-specific rule is involved.
@kernel logistic_reduction(r::Vector{Float64}, s::Float64,
        w::Vector{Float64}, seed::Float64) = begin
    sigma::Float64 = exp(s)
    z::Vector{Float64} = r ./ sigma
    t::Vector{Float64} = tanh.(z ./ 2)
    v::Vector{Float64} = -s .- log1pexp.(z) .- log1pexp.(-z)
    total::Float64 = dot(w, v)
    r_partial::Vector{Float64} = -w .* t ./ sigma
    s_partial::Float64 = sum(w .* (z .* t .- 1))
    r_bar::Vector{Float64} = seed .* r_partial
    s_bar::Float64 = seed * s_partial
    w_bar::Vector{Float64} = seed .* v
    return total, r_bar, s_bar, w_bar
end
const reduction = derivative_rule(logistic_reduction; primal=:total,
    covector=:seed, cotangents=(r=:r_bar, s=:s_bar, w=:w_bar), name=:logistic_reduction)

@kernel cell_plate(r::Vector{Float64}, s::Float64) = begin
    terms = plate(r) do ri
        cell(ri, s)
    end
    total::Float64 = sum(terms)
    return total
end

function make_model(mode)
    loss = mode === :captured ? cell : reduction
    if mode === :ordinary
        @kernel graph(beta::Vector{Float64}, u::Vector{Float64}, s::Float64,
                X::Matrix{Float64}, g::Vector{Int}, y::Vector{Float64}, w::Vector{Float64}) = begin
            eta::Vector{Float64} = X * beta .+ u[g]
            r::Vector{Float64} = collect(y .- sin.(eta))
            sigma::Float64 = exp(s)
            terms = plate(r, w) do ri, wi
                z = ri / sigma
                wi * (-s - log1pexp(z) - log1pexp(-z))
            end
            total::Float64 = sum(terms) - sum(abs2, beta) / 2 - sum(abs2, u) / 2 - s^2 / 2
            return total
        end
    elseif mode in (:cell, :cell_plate)
        @kernel graph(beta::Vector{Float64}, u::Vector{Float64}, s::Float64,
                X::Matrix{Float64}, g::Vector{Int}, y::Vector{Float64}, w::Vector{Float64}) = begin
            eta::Vector{Float64} = X * beta .+ u[g]
            r::Vector{Float64} = collect(y .- sin.(eta))
            terms = plate(r, w) do ri, wi
                wi * cell(ri, s)
            end
            total::Float64 = sum(terms) - sum(abs2, beta) / 2 - sum(abs2, u) / 2 - s^2 / 2
            return total
        end
    elseif mode === :captured
        @kernel graph(beta::Vector{Float64}, u::Vector{Float64}, s::Float64,
                X::Matrix{Float64}, g::Vector{Int}, y::Vector{Float64}, w::Vector{Float64}) = begin
            eta::Vector{Float64} = X * beta .+ u[g]
            r::Vector{Float64} = collect(y .- sin.(eta))
            terms = plate(r, w) do ri, wi
                wi * loss(ri, s)
            end
            total::Float64 = sum(terms) - sum(abs2, beta) / 2 - sum(abs2, u) / 2 - s^2 / 2
            return total
        end
    else
        @kernel graph(beta::Vector{Float64}, u::Vector{Float64}, s::Float64,
                X::Matrix{Float64}, g::Vector{Int}, y::Vector{Float64}, w::Vector{Float64}) = begin
            eta::Vector{Float64} = X * beta .+ u[g]
            r::Vector{Float64} = collect(y .- sin.(eta))
            loglik::Float64 = loss(r, s, w)
            total::Float64 = loglik - sum(abs2, beta) / 2 - sum(abs2, u) / 2 - s^2 / 2
            return total
        end
    end
    graph
end

function reference(beta,u,s,X,g,y,w)
    eta = X*beta + u[g]
    z = (y - sin.(eta)) ./ exp(s)
    t = tanh.(z ./ 2)
    value = sum(w .* (-s .- log1pexp.(z) .- log1pexp.(-z))) -
        sum(abs2,beta)/2 - sum(abs2,u)/2 - s^2/2
    ebar = w .* t .* cos.(eta) ./ exp(s)
    db = X' * ebar - beta
    du = -u
    for i in eachindex(g)
        du[g[i]] += ebar[i]
    end
    ds = sum(w .* (z .* t .- 1)) - s
    value,(db,du,ds)
end
end

function measure(op, args)
    f() = ad_value_and_gradient(op, args...)
    f()
    t = @benchmark $f() evals=1 samples=1000 seconds=0.3
    m=median(t)
    (; ns=m.time, bytes=m.memory, allocs=m.allocs)
end

println("LOCAL RULE REPORT ", plate_reverse_report(prepare(ComposedProbe.cell_plate), (1,2)))
backend = AutoEnzyme(; mode=Enzyme.Reverse)
for mode in (isempty(ARGS) ? (:ordinary,:cell,:cell_plate,:reduction,:captured) : Tuple(Symbol.(ARGS)))
    println("BEGIN ",mode); flush(stdout)
    let
        build = @elapsed spec = ComposedProbe.make_model(mode)
        prep = @elapsed kernel = prepare(spec)
        report = plate_reverse_report(kernel,(1,2,3))
        println("BUILD ",build," PREP ",prep," REPORT ",report)
        mode in (:cell_plate, :captured) && @test !isempty(report) && all(r -> last(r) == "rule", report)
        rng=Xoshiro(43); n=128; p=4; k=7
        args=(randn(rng,p)./5, randn(rng,k)./5, .2,
              randn(rng,n,p), rand(rng,1:k,n), randn(rng,n), 0.5 .+ rand(rng,n))
        adprep=@elapsed op=with_plate_reverse(mode in (:cell_plate, :captured)) do
            prepare_ad(kernel,backend,args...;active=(:beta,:u,:s))
        end
        firstcall=@elapsed actual=ad_value_and_gradient(op,args...)
        expected=ComposedProbe.reference(args...)
        @test actual[1] ≈ expected[1]
        @test all(isapprox.(actual[2],expected[2]))
        println("ADPREP ",adprep," FIRST ",firstcall," MEDIAN ",measure(op,args))
        for n in (0,3,2048)
            args2=(args[1],args[2],-.3,randn(rng,n,p),rand(rng,1:k,n),randn(rng,n),0.5 .+ rand(rng,n))
            actual=ad_value_and_gradient(op,args2...)
            expected=ComposedProbe.reference(args2...)
            @test actual[1] ≈ expected[1]
            @test all(isapprox.(actual[2],expected[2]))
            println("N ",n," MEDIAN ",measure(op,args2))
        end
    end
    println("END ",mode); flush(stdout)
end
println("COMPLETE")
