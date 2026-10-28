# Tsitouras 5(4) coefficients.
#
# Transcribed from OrdinaryDiffEqTsit5 2.1.4
# `src/tsit_tableaus.jl` (`Tsit5ConstantCacheActual` / `Tsit5Interp`,
# `CompiledFloats` branch). The end-to-end agreement tests at tight tolerance
# are the check on this transcription: a wrong coefficient drops the order
# and fails them by orders of magnitude.
#
# FSAL note: the seventh A-row is the fifth-order propagator
# (`u = uprev + dt*Σa7ᵢkᵢ`, matching OrdinaryDiffEq's `perform_step!`, which
# never references a separate `b` row), and `k7 = f(u)` becomes the next
# step's `k1`. No separate `b` row is stored.

"""
    Tsit5Tableau{T<:Number}

Explicit Tsit5 tableau: stage nodes `c1..c6` (node of stage 1 is zero), the
strictly-lower stage matrix rows `a21..a76`, and the embedded-error weights
`btilde1..btilde7` (fifth- minus fourth-order weights).

`Tsit5Tableau{T}()` fills every entry from the reference coefficients. The
bound is `Number` (not `AbstractFloat`) so Reactant can promote the
container to traced coefficients when the struct is captured by a traced
loop; native construction always uses concrete floats.
"""
Base.@kwdef struct Tsit5Tableau{T<:Number}
    c1::T = T(0.161)
    c2::T = T(0.327)
    c3::T = T(0.9)
    c4::T = T(0.9800255409045097)
    c5::T = T(1)
    c6::T = T(1)
    a21::T = T(0.161)
    a31::T = T(-0.008480655492356989)
    a32::T = T(0.335480655492357)
    a41::T = T(2.8971530571054935)
    a42::T = T(-6.359448489975075)
    a43::T = T(4.3622954328695815)
    a51::T = T(5.325864828439257)
    a52::T = T(-11.748883564062828)
    a53::T = T(7.4955393428898365)
    a54::T = T(-0.09249506636175525)
    a61::T = T(5.86145544294642)
    a62::T = T(-12.92096931784711)
    a63::T = T(8.159367898576159)
    a64::T = T(-0.071584973281401)
    a65::T = T(-0.028269050394068383)
    a71::T = T(0.09646076681806523)
    a72::T = T(0.01)
    a73::T = T(0.4798896504144996)
    a74::T = T(1.379008574103742)
    a75::T = T(-3.290069515436081)
    a76::T = T(2.324710524099774)
    btilde1::T = T(-0.00178001105222577714)
    btilde2::T = T(-0.0008164344596567469)
    btilde3::T = T(0.007880878010261995)
    btilde4::T = T(-0.1447110071732629)
    btilde5::T = T(0.5823571654525552)
    btilde6::T = T(-0.45808210592918697)
    btilde7::T = T(0.015151515151515152)
end

"""
    Tsit5DenseCoefficients{T<:Number}

Free fourth-order dense-output coefficients (Tsitouras): `b1(θ)` is
`θ*evalpoly(θ, r11..r14)` and `bᵢ(θ)` for stages 2..7 is
`θ²*evalpoly(θ, rᵢ2..rᵢ4)`, so `y(t+θ*dt) = uprev + dt*Σbᵢ(θ)kᵢ`.
The bound is `Number` for the same traced-loop promotion reason as
[`Tsit5Tableau`](@ref).
"""
Base.@kwdef struct Tsit5DenseCoefficients{T<:Number}
    r11::T = T(1.0)
    r12::T = T(-2.763706197274826)
    r13::T = T(2.9132554618219126)
    r14::T = T(-1.0530884977290216)
    r22::T = T(0.13169999999999998)
    r23::T = T(-0.2234)
    r24::T = T(0.1017)
    r32::T = T(3.9302962368947516)
    r33::T = T(-5.941033872131505)
    r34::T = T(2.490627285651253)
    r42::T = T(-12.411077166933676)
    r43::T = T(30.33818863028232)
    r44::T = T(-16.548102889244902)
    r52::T = T(37.50931341651104)
    r53::T = T(-88.1789048947664)
    r54::T = T(47.37952196281928)
    r62::T = T(-27.896526289197286)
    r63::T = T(65.09189467479366)
    r64::T = T(-34.87065786149661)
    r72::T = T(1.5)
    r73::T = T(-4.0)
    r74::T = T(2.5)
end
