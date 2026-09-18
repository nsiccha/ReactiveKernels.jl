# Tight-tolerance OrdinaryDiffEq Tsit5 reference solves. ODE's `Tsit5` is
# qualified (our package exports its own `Tsit5` tag) via explicit imports.
import OrdinaryDiffEqTsit5
using OrdinaryDiffEqTsit5: ODEProblem, solve

const ReferenceTsit5 = OrdinaryDiffEqTsit5.Tsit5

function ode_reference(f, u0, tspan, p; saveat=nothing, abstol=1e-13,
        reltol=1e-11)
    prob = ODEProblem(f, u0, tspan, p)
    if saveat === nothing
        solve(prob, ReferenceTsit5(); abstol=abstol, reltol=reltol)
    else
        solve(prob, ReferenceTsit5(); abstol=abstol, reltol=reltol,
            saveat=saveat)
    end
end

max_abs_diff(a::AbstractVector, b::AbstractVector) =
    maximum(abs.(a .- b))

function max_abs_diff(as::AbstractVector{<:AbstractVector},
        bs::AbstractVector{<:AbstractVector})
    @assert length(as) == length(bs)
    maximum(max_abs_diff(a, b) for (a, b) in zip(as, bs))
end
