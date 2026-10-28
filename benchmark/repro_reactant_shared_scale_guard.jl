# Generic reproducer for a shared scalar guard entering a retained plate.
# Reactant 0.2.290 reaches isless(0, EnsureReturnType{Any}) during tracing.
# Native densities remain valid. Exit nonzero when a control still fails;
# future successful compilation must preserve both selected outcomes.
using ReactiveKernels, Reactant, Test
using Distributions: Normal, logpdf
using ReactiveKernelsDistributionKernels.DistributionKernelSources
@kernel product_scale(u::Vector{Float64}, x::Vector{Float64}, y::Vector{Float64}) = begin
    a = sum(view(u, 1:1))
    z = sum(view(u, 2:2))
    lambda = exp(sum(view(u, 3:3)))
    tau = exp(sum(view(u, 4:4)))
    scale = z * lambda * tau
    mean = a .+ scale .* x
    pointwise = plate(y, mean, scale) do yy, mu, s
        if s > 0
            normal(mu, s).logpdf(yy)
        else
            -Inf
        end
    end
    total = sum(pointwise)
    return total
end
@kernel direct_scale(u::Vector{Float64}, x::Vector{Float64}, y::Vector{Float64}) = begin
    scale = u[2]
    mean = u[1] .+ scale .* x
    pointwise = plate(y, mean, scale) do yy, mu, s
        if s > 0
            normal(mu, s).logpdf(yy)
        else
            -Inf
        end
    end
    total = sum(pointwise)
    return total
end
@kernel typed_direct_scale(u::Vector{Float64}, x::Vector{Float64}, y::Vector{Float64}) = begin
    scale::Float64 = u[2]
    mean = u[1] .+ scale .* x
    pointwise = plate(y, mean, scale) do yy, mu, s
        if s > 0
            normal(mu, s).logpdf(yy)
        else
            -Inf
        end
    end
    total = sum(pointwise)
    return total
end
function compile_primal(k,ru)
    Reactant.@compile k(ru)
end
u=[0.4,0.6,0.1,-0.2];x=[-0.4,0.1,0.6];y=[0.7,1.2,2.3]
failures = 0
for (name,spec) in (("direct",direct_scale),("typed_direct",typed_direct_scale),("product",product_scale))
    k=prepare(spec;have=(:u,:x,:y),want=:total,bound=(;x,y))
    scale = name == "product" ? u[2] * exp(u[3]) * exp(u[4]) : u[2]
    expected = sum(logpdf.(Normal.(u[1] .+ scale .* x, scale), y))
    @test Base.invokelatest(k, u) ≈ expected
    invalid = copy(u)
    invalid[2] = -invalid[2]
    @test Base.invokelatest(k, invalid) == -Inf
    println(name," native=",Base.invokelatest(k,u))
    try
        compiled=Base.invokelatest(compile_primal,k,Reactant.to_rarray(u))
        @test Float64(compiled(Reactant.to_rarray(u))) ≈ expected
        @test Float64(compiled(Reactant.to_rarray(invalid))) == -Inf
        println(name," compiled density checks passed")
    catch error
        error isa MethodError && error.f === isless || rethrow()
        global failures += 1
        println(name," failure=",first(split(sprint(showerror,error),'\n')))
    end
end
failures == 0 || exit(1)
