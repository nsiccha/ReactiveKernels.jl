module ProductMixtureRowsTests
using Test, ReactiveKernels, ReactiveKernelsDistributionKernels
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    bernoulli_product_mixture_logdensity, bernoulli_logit_product_mixture_logdensity,
    binomial_product_mixture_logdensity, binomial_logit_product_mixture_logdensity,
    zero_inflated_binomial_logit_product_mixture_logdensity
using DifferentiationInterface: AutoEnzyme
import Enzyme

# Independent high-precision scalar laws, with broadcasting performed only
# by the reference. The production laws keep runtime position/component loops.
bernoulli_cell(y, p) = y ? log(big(p)) : log1p(-big(p))
bernoulli_logit_cell(y, l) = -log1p(exp(y ? -big(l) : big(l)))
function binomial_cell(y, n, p)
    0 <= y <= n || return big(-Inf)
    log(big(Base.binomial(big(n), y))) +
        (y == 0 ? big(0) : y * log(big(p))) +
        (y == n ? big(0) : (n - y) * log1p(-big(p)))
end
function binomial_logit_cell(y, n, l)
    0 <= y <= n || return big(-Inf)
    log(big(Base.binomial(big(n), y))) +
        (y == 0 ? big(0) : -y * log1p(exp(-big(l)))) +
        (y == n ? big(0) : -(n - y) * log1p(exp(big(l))))
end
function logadd(a, b)
    top = max(a, b)
    top == -Inf ? top : top + log(exp(a - top) + exp(b - top))
end
function zib_cell(y, n, l, z)
    count = binomial_logit_cell(y, n, l)
    regular = -log1p(exp(big(z))) + count
    y == 0 ? logadd(-log1p(exp(-big(z))), regular) : regular
end
expand(a, n) = size(a, 1) == 1 ? repeat(a, n, 1) : a
function structure(ex)
    ex isa Expr || return (0, 0)
    foldl(.+, (structure(a) for a in ex.args);
        init=(1, Int(ex.head in (:for, :while))))
end
function reference(cell, observed, arrays, weights)
    setprecision(BigFloat, 512) do
        full = map(a -> expand(a, length(observed)), arrays)
        scores = [log(big(weights[k])) + sum(
            (cell(observed[i], (a[i, k] for a in full)...) for i in eachindex(observed));
            init=big(0)) for k in eachindex(weights)]
        foldl(logadd, scores; init=big(-Inf))
    end
end

@testset "product laws broadcast only the position rows" begin
    weights = [0.35, 0.65]
    for (spec, cell, observed, rows, names, active) in (
            (bernoulli_product_mixture_logdensity, bernoulli_cell,
             Bool[false, true, false], ([0.25 0.7],), (:probabilities,), (:probabilities,)),
            (bernoulli_logit_product_mixture_logdensity, bernoulli_logit_cell,
             Bool[false, true, false], ([40.0 -Inf],), (:logits,), (:logits,)),
            (binomial_product_mixture_logdensity, binomial_cell,
             [0, 1, 3], ([4 5], [0.25 0.7]), (:trials, :probabilities), (:probabilities,)),
            (binomial_logit_product_mixture_logdensity, binomial_logit_cell,
             [0, 1, 3], ([4 5], [40.0 -Inf]), (:trials, :logits), (:logits,)),
            (zero_inflated_binomial_logit_product_mixture_logdensity, zib_cell,
             [0, 1, 3], ([4 5], [40.0 -Inf], [-35.0 0.4]),
             (:trials, :logits, :zi_logits), (:logits, :zi_logits)))
        law = prepare(spec; type_stability=:error)
        expected = Float64(reference(cell, observed, rows, weights))
        for mask in 0:(2^length(rows)-1)
            arrays = ntuple(j -> iszero(mask & (1 << (j-1))) ? rows[j] :
                expand(rows[j], length(observed)), length(rows))
            saved = deepcopy(arrays)
            @test law(observed, arrays..., weights) ≈ expected rtol=1e-13 atol=1e-13
            @test arrays == saved
            varied = map(arrays) do a
                size(a, 1) == 1 && return a
                b = copy(a)
                b[2, 1] += eltype(a) <: Integer ? 1 : 0.05
                b
            end
            @test law(observed, varied..., weights) ≈
                Float64(reference(cell, observed, varied, weights)) rtol=1e-13 atol=1e-13
            # Empty events allow each operand to have zero or one position row.
            empty_arrays = map(a -> size(a, 1) == 1 ? a : a[1:0, :], arrays)
            @test law(eltype(observed)[], empty_arrays..., weights) ≈ 0.0 atol=1e-15
        end
        @test law(observed, (a[:, 1:0] for a in rows)..., Float64[]) == -Inf
        @test_throws DimensionMismatch law(observed, rows..., [1.0])
        for j in eachindex(rows)
            bad_rows = Base.setindex(rows, repeat(rows[j], 2, 1), j)
            @test_throws DimensionMismatch law(observed, bad_rows..., weights)
            bad_columns = Base.setindex(rows, rows[j][:, 1:1], j)
            @test_throws DimensionMismatch law(observed, bad_columns..., weights)
        end
        for input in active
            j = findfirst(==(input), names)
            ad = prepare_ad(law, AutoEnzyme(; mode=Enzyme.Reverse),
                observed, rows..., weights; active=input)
            value, gradient = ad_value_and_gradient(ad, observed, rows..., weights)
            @test value ≈ expected rtol=1e-13 atol=1e-13
            @test all(isfinite, gradient)
            # Each singleton cotangent accumulates contributions from every
            # position. Check against a 512-bit derivative of the scalar oracle.
            setprecision(BigFloat, 512) do
                h = big"1e-40"
                for k in axes(rows[j], 2)
                    if !isfinite(rows[j][1,k])
                        @test iszero(gradient[1,k])
                        continue
                    end
                    plus = map(a -> big.(a), rows)
                    minus = deepcopy(plus)
                    plus[j][1,k] += h
                    minus[j][1,k] -= h
                    slope = (reference(cell, observed, plus, weights) -
                        reference(cell, observed, minus, weights)) / (2h)
                    @test gradient[1,k] ≈ Float64(slope) rtol=1e-12 atol=1e-13
                end
            end
            @test all(iszero, ad_gradient(ad, eltype(observed)[], rows..., weights))
        end
        # Binding observations and singleton data preserves the same emitted
        # program size when the event length changes.
        bodies = Tuple{Int,Int}[]
        for repeats in (1, 7)
            counts = repeat(observed, repeats)
            bound = :trials in names ? (; observed=counts, trials=rows[1], weights) :
                (; observed=counts, weights)
            prepared = prepare(spec; bound)
            args = :trials in names ? Base.tail(rows) : rows
            @test prepared(args...) ≈ Float64(reference(cell, counts, rows, weights)) rtol=1e-13
            push!(bodies, structure(code_expr(prepared)))
        end
        @test bodies[1] == bodies[2]
        @test last(bodies[1]) >= 2
    end
end
end
