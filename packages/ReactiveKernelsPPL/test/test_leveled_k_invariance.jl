# Core constraint 1 (docs/src/constraints.md) for the leveled families: a
# data-inferred level count K must not replicate emitted statements. The
# emitted program at two level counts must be the SAME expression once
# literal constants are abstracted (level counts, data lengths, and the
# frozen literal payloads — Dirichlet α−1, R2D2 column variances — are
# constants; anything else that differs is K-dependent structure).
using ReactiveKernels
using ReactiveKernelsPPL
using Test

_kinv_levels(K, n) = [mod1(i, K) for i in 1:n]

function _kinv_plans(K::Int)
    n = 3K
    x = [sin(1.3i) for i in 1:n]
    y = _kinv_levels(K, n)
    yc = [cos(0.7i) for i in 1:n]
    w = [0.5 + mod(i, 3) / 2 for i in 1:n]
    alpha = Expr(:vect, fill(1.5, K - 1)...)
    phia = Expr(:vect, fill(1.0, K + 1)...)
    data(pairs...) = Dict{Symbol,AbstractVector}(pairs...)
    return [
        "ordered_logistic" => bind_data(lower_rkppl(quote
                eta = a .+ b .* x
                y .~ OrderedLogistic.(eta)
            end, (:y, :x)), data(:y => y, :x => x)),
        "ordinal_cumulative_probit" => bind_data(lower_rkppl(quote
                eta = b .* x
                y .~ Ordinal.(Cumulative(), ProbitLink(), eta)
            end, (:y, :x)), data(:y => y, :x => x)),
        "ordinal_stopping_logit" => bind_data(lower_rkppl(quote
                eta = b .* x
                y .~ Ordinal.(StoppingRatio(), LogitLink(), eta)
            end, (:y, :x)), data(:y => y, :x => x)),
        "categorical_simplex" => bind_data(lower_rkppl(quote
                s ~ Dirichlet($K, 1.0)
                y .~ Categorical.(s)
            end, (:y,)), data(:y => y)),
        "monotonic" => bind_data(lower_rkppl(quote
                s ~ Dirichlet($alpha)
                mu = a .+ b .* mo(c, s)
                sigma ~ Exponential(1.0)
                yc .~ Normal.(mu, sigma)
            end, (:yc, :c)), data(:yc => yc, :c => y)),
        "r2d2_factor" => bind_data(lower_rkppl(quote
                R2 ~ Beta(1.0, 1.0)
                phi ~ Dirichlet($phia)
                mu = b1 .* x1 .+ c[g]
                r2d2(mu, R2, phi)
                sigma ~ Exponential(1.0)
                yc .~ Normal.(mu, sigma)
            end, Set([:x1, :g, :yc])), data(:x1 => x, :g => y, :yc => yc)),
    ]
end

# Abstract every numeric literal and every literal constant vector
# (`[…]`, `Float64[…]` over numbers) to one placeholder.
_kinv_isnum(a) = a isa Number
function _kinv_normalize(ex)
    ex isa Number && return :__lit
    ex isa LineNumberNode && return nothing
    ex isa Expr || return ex
    if (ex.head === :vect && all(_kinv_isnum, ex.args)) ||
       (ex.head === :ref && ex.args[1] === :Float64 &&
        all(_kinv_isnum, ex.args[2:end]))
        return :__litvec
    end
    args = Any[_kinv_normalize(a) for a in ex.args if !(a isa LineNumberNode)]
    return Expr(ex.head, args...)
end

_kinv_program(bound) = _kinv_normalize(
    ReactiveKernelsPPL.kernel_expr(bound, assign_layout(bound)).args[2])

@testset "leveled emission is invariant in the level count K" begin
    small, large = Dict(_kinv_plans(3)), Dict(_kinv_plans(6))
    for name in sort!(collect(keys(small)))
        a, b = _kinv_program(small[name]), _kinv_program(large[name])
        @testset "$name" begin
            @test length(a.args) == length(b.args)
            @test a == b
        end
    end
end

@testset "leveled emission carries no per-level ifelse chain" begin
    for (name, bound) in _kinv_plans(5)
        src = string(ReactiveKernelsPPL.kernel_expr(bound, assign_layout(bound)))
        @test !occursin("_ppl_v_", src)
        @test !occursin(r"ifelse\(\w+ == \d", src)
    end
end
