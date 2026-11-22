module SharedSoftplusTests
using Test, ReactiveKernels, MutatingFunctions
using DifferentiationInterface: AutoEnzyme
using LogExpFunctions: log1pexp, logistic
using LinearAlgebra: dot
using InteractiveUtils: code_llvm
import Enzyme

@kernel softplus_total(x::Vector{Float64}, seed::Float64) = begin
    cells = plate(x) do xi
        rk_log1pexp(xi)
    end
    total::Float64 = seed * sum(cells)
    return total
end

@testset "shared stable softplus value and partial" begin
    f = rk_log1pexp
    setprecision(BigFloat, 512) do
        for x in (-1000.0,-745.0,-700.0,-40.0,-20.0,-1.0,-eps(),-0.0,
                0.0,eps(),1.0,20.0,40.0,700.0,1000.0)
            ref = Float64(log1p(exp(BigFloat(x))))
            dref = Float64(inv(1+exp(-BigFloat(x))))
            v,d = derivative_cut(f,(true,),x)
            @test v ≈ ref rtol=2eps(Float64)
            @test d ≈ dref rtol=2eps(Float64)
            @test f(x) == v
            for seed in (-1.3,0.0,1.0)
                wrapped = z -> seed*f(z)
                dr = Enzyme.autodiff(Enzyme.Reverse,wrapped,Enzyme.Active(x))[1][1]
                df = Enzyme.autodiff(Enzyme.Forward,f,Enzyme.Duplicated(x,seed))[1]
                @test dr ≈ seed*dref rtol=4eps(Float64)
                @test df ≈ seed*dref rtol=4eps(Float64)
            end
        end
    end
    @test derivative_cut(f,(true,),-Inf)==(0.0,0.0)
    @test derivative_cut(f,(true,),Inf)==(Inf,1.0)
    @test all(isnan,derivative_cut(f,(true,),NaN))
    @test isnan(f(NaN)) && f(Inf)==Inf && f(-Inf)==0.0

    # The primal call plans the one value-only recipe, whose tail shortcuts
    # (`x` above 34, `exp(x)` below -37) return the joint recipe's bits.
    @test length(typeof(f).parameters[4][2])==1
    @test f(40.0)===40.0 && f(-40.0)===exp(-40.0)
    for x in Float32[-2, -0.5, 0.5, 2]
        @test f(x) === max(x, 0.0) + log1p(exp(-abs(x)))
    end
    for x in (34.0,prevfloat(34.0),nextfloat(34.0),-37.0,prevfloat(-37.0),
            nextfloat(-37.0),prevfloat(-1.0),-1.0,nextfloat(-1.0),
            prevfloat(1.0),1.0,nextfloat(1.0),-745.2,-746.0,1e300,-1e300,
            range(-800.0,800.0;length=40_001)...)
        @test f(x)===derivative_cut(f,(true,),x)[1]
    end

    backend=AutoEnzyme(;mode=Enzyme.Reverse)
    for nonallocating in (false,true), enabled in (false,true)
        x0=[-.2,.3,40.0]
        op=with_plate_reverse(enabled) do
            k=nonallocating ? prepare_nonallocating(softplus_total,x0,1.0) : prepare(softplus_total)
            prepare_ad(k,backend,x0,1.0;active=:x)
        end
        for n in (0,3,257), seed in (-1.3,0.0,1.0)
            x=[sin(i)*100 for i in 1:n]
            saved=copy(x)
            g=fill(99.0,n)
            value,gradient=ad_value_and_gradient!(op,g,x,seed)
            @test value ≈ seed*sum(log1pexp,x;init=0.0)
            @test gradient ≈ seed.*logistic.(x)
            @test gradient === g
            @test x == saved
        end
    end
end

# A summed plate over the owned softplus, over LogExpFunctions' softplus, and a
# plate that keeps its cells; a broadcast under a dot product.
@kernel owned_sum(y::Vector{Bool}, eta::Vector{Float64}) = begin
    cells = plate(y, eta) do yi, ei
        -rk_log1pexp(ifelse(yi, -ei, ei))
    end
    total::Float64 = sum(cells)
    return total
end
@kernel foreign_sum(x::Vector{Float64}) = begin
    cells = plate(x) do xi
        log1pexp(xi)
    end
    total::Float64 = sum(cells)
    return total
end
@kernel kept_cells(x::Vector{Float64}) = begin
    cells = plate(x) do xi
        rk_log1pexp(xi)
    end
    return cells
end
@kernel weighted_softplus(n::Vector{Float64}, x::Vector{Float64}) = begin
    total::Float64 = dot(n, log1pexp.(x))
    return total
end
# Cells and a broadcast that also call Base's `log`, which keeps a loop scalar.
@kernel mixed_cells(y::Vector{Bool}, eta::Vector{Float64}, w::Vector{Float64}) = begin
    cells = plate(y, eta, w) do yi, ei, wi
        -rk_log1pexp(ifelse(yi, -ei, ei)) + log(wi)
    end
    total::Float64 = sum(cells)
    return total
end
@kernel weighted_cells(y::Vector{Bool}, eta::Vector{Float64}, w::Vector{Float64}) = begin
    cells = plate(y, eta, w) do yi, ei, wi
        -rk_log1pexp(ifelse(yi, -ei, ei)) * wi
    end
    total::Float64 = sum(cells)
    return total
end
@kernel fused_broadcast(eta::Vector{Float64}, w::Vector{Float64}) = begin
    total::Float64 = sum(log1pexp.(eta) .+ log.(w))
    return total
end
@kernel scanned_cells(X::Matrix{Float64}) = begin
    cells = plate(eachcol(X)) do xs
        history = scan(xs; init = 0.0) do carry, x
            next = carry + rk_log1pexp(x)
            (next, next)
        end
        last(history)
    end
    total::Float64 = sum(cells)
    return total
end

vector_sum(x) = (s = 0.0; @inbounds @simd for i in eachindex(x); s += ReactiveKernels._vector_softplus(x[i]); end; s)
scalar_sum(x) = (s = 0.0; @inbounds @simd for i in eachindex(x); s += rk_log1pexp(x[i]); end; s)
function _loop_llvm(f)
    io = IOBuffer()
    code_llvm(io, f, Tuple{Vector{Float64}}; debuginfo = :none)
    String(take!(io))
end
_has_vector_doubles(f) = occursin(r"<\d+ x double>", _loop_llvm(f))
_ulps(a, b) = a == b ? 0.0 : abs(a - b) / eps(abs(b))

@testset "a vector loop computes softplus with the vector rule" begin
    vs = ReactiveKernels._vector_softplus
    @test vs isa ScalarDerivativeRule
    # Its value: one unit in the last place from the owned softplus, the same
    # tails, and the authored partial unchanged.
    grid = (range(-800.0, 800.0; length = 40_001)..., -745.2, -745.0, -40.0, -37.0, -1.0, -0.0, 0.0, 1.0,
            33.9, 34.0, 40.0, 700.0, 1e300, -1e300)
    @test maximum(x -> _ulps(vs(x), rk_log1pexp(x)), grid) <= 1.0
    setprecision(BigFloat, 512) do
        for x in (-700.0, -40.0, -20.0, -1.0, -eps(), -0.0, 0.0, eps(), 1.0, 20.0, 40.0, 700.0, 1000.0)
            @test vs(x) ≈ Float64(log1p(exp(BigFloat(x)))) rtol = 2eps(Float64)
        end
    end
    @test vs(40.0) === 40.0 && vs(-40.0) === exp(-40.0) && vs(1e300) === 1e300
    @test vs(-Inf) === 0.0 && vs(Inf) === Inf && isnan(vs(NaN)) && vs(-1000.0) === 0.0
    for x in (-800.0, -37.5, -1.0, 0.0, 0.3, 34.5, 800.0, -Inf, Inf)
        @test derivative_cut(vs, (true,), x) === derivative_cut(rk_log1pexp, (true,), x)
    end
    # Any other argument keeps the function the source names.
    @test ReactiveKernels._vector_rk_log1pexp(0.5f0) === rk_log1pexp(0.5f0)
    @test ReactiveKernels._vector_log1pexp(0.5f0) === log1pexp(0.5f0)
    @test ReactiveKernels._vector_log1pexp(big"0.5") == log1pexp(big"0.5")
    @test ReactiveKernels._vector_log1pexp(0.5) === vs(0.5) === ReactiveKernels._vector_rk_log1pexp(0.5)

    # The vector rule's summed loop holds vector arithmetic; the owned
    # softplus's does not, which is why a vector loop swaps it.
    @test _has_vector_doubles(vector_sum)
    @test !_has_vector_doubles(scalar_sum)

    # The swap needs a CPU whose vectors hold at least four doubles: with two
    # the vector softplus would be level with the source's at best.
    twin = ReactiveKernels._vector_softplus_twin
    @test twin(log1pexp, 2) === nothing && twin(rk_log1pexp, 2) === nothing
    @test twin(log1pexp, 4) === :_vector_log1pexp && twin(rk_log1pexp, 8) === :_vector_rk_log1pexp
    @test twin(exp, 8) === nothing && twin(logistic, 8) === nothing
    wide = ReactiveKernels._vector_doubles() >= ReactiveKernels._VECTOR_SOFTPLUS_DOUBLES
    @test twin(log1pexp) === (wide ? :_vector_log1pexp : nothing)

    # Which programs call it there: a summed plate's cells, whether the cell is
    # an expression or one call, and a broadcast. A plate that keeps its cells
    # and a cell that runs a scan keep the source's softplus.
    @test occursin("_vector_rk_log1pexp", string(code_expr(prepare(owned_sum)))) == wide
    @test occursin("_vector_log1pexp", string(code_expr(prepare(foreign_sum)))) == wide
    @test occursin("_vector_rk_log1pexp", string(code_expr(prepare(softplus_total)))) == wide
    @test !occursin("_vector_", string(code_expr(prepare(kept_cells))))
    @test !occursin("_vector_", string(code_expr(prepare(scanned_cells))))
    x0 = [-0.2, 0.3, 40.0]; n0 = [1.0, 2.0, 3.0]
    weighted = prepare_nonallocating(weighted_softplus, n0, x0)
    @test occursin("_vector_log1pexp", string(code_expr(weighted))) == wide
    X0 = reshape([sin(i) * 20 for i in 1:12], 4, 3)
    @test prepare(scanned_cells)(X0) ≈ sum(log1pexp, X0)

    # One call that keeps the loop scalar makes the vector softplus the slower
    # one (a cell with Base's `log` beside it took twice as long), so such a
    # cell and such a fused broadcast keep the source's softplus.
    y0 = [true, false, true]; w0 = [0.5, 1.5, 2.5]
    @test occursin("_vector_rk_log1pexp", string(code_expr(prepare(weighted_cells)))) == wide
    @test !occursin("_vector_", string(code_expr(prepare(mixed_cells))))
    @test !occursin("_vector_", string(code_expr(prepare_nonallocating(fused_broadcast, x0, w0))))
    signed0 = ifelse.(y0, -x0, x0)
    @test prepare(weighted_cells)(y0, x0, w0) ≈ -sum(log1pexp.(signed0) .* w0)
    @test prepare(mixed_cells)(y0, x0, w0) ≈ sum(log.(w0) .- log1pexp.(signed0))
    @test prepare_nonallocating(fused_broadcast, x0, w0)(x0, w0) ≈ sum(log1pexp.(x0) .+ log.(w0))
    safe = ReactiveKernels._vector_safe_cell
    ref(m, n) = GlobalRef(m, n)
    @test safe(:(total = $(ref(Base, :+))(total, $(ref(ReactiveKernels, :rk_log1pexp))($(ref(Base, :-))(x)))))
    @test safe(:(v = $(ref(Base, :*))(__constants__[2], $(ref(Base, :ifelse))(b, x, y))))
    @test !safe(:(v = $(ref(Base, :log))(x)))                       # a call that stays scalar
    @test !safe(:(v = f(x)))                                        # a call it cannot name
    @test !safe(:(v = data[i]))                                     # an indexed read
    @test !safe(:(v = b ? x : y)) && !safe(:(b && $(ref(Base, :-))(x)))   # a lazy branch
    @test !safe(:(for j in 1:3; v = x; end))                        # a loop of its own
    operand = ReactiveKernels._vector_safe_broadcast
    negated = typeof(Base.Broadcast.broadcasted(-, x0))
    logged = typeof(Base.Broadcast.broadcasted(log, x0))
    @test operand((Vector{Float64},)) && operand((Vector{Float64}, Float64)) && operand((negated,))
    @test !operand((logged,)) && !operand((Vector{Any},))
    @test !operand((typeof(Base.Broadcast.broadcasted(+, x0, Base.Broadcast.broadcasted(log, x0))),))

    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    for enabled in (false, true), n in (0, 1, 7, 513)
        x = [sin(i) * 30 for i in 1:n]; y = [isodd(i) for i in 1:n]; w = [1.0 + i % 3 for i in 1:n]
        signed = ifelse.(y, -x, x)
        with_plate_reverse(enabled) do
            @test prepare(owned_sum)(y, x) ≈ -sum(log1pexp, signed; init = 0.0)
            @test prepare(foreign_sum)(x) ≈ sum(log1pexp, x; init = 0.0)
            @test prepare(kept_cells)(x) == rk_log1pexp.(x)
            @test prepare_nonallocating(weighted_softplus, w, x)(w, x) ≈ dot(w, log1pexp.(x))
            g = fill(99.0, n)
            op = prepare_ad(prepare(owned_sum), backend, y, x; active = :eta)
            value, gradient = ad_value_and_gradient!(op, g, y, x)
            @test value ≈ -sum(log1pexp, signed; init = 0.0)
            @test gradient ≈ -ifelse.(y, -1.0, 1.0) .* logistic.(signed)
            op = prepare_ad(prepare(foreign_sum), backend, x; active = :x)
            value, gradient = ad_value_and_gradient!(op, g, x)
            @test value ≈ sum(log1pexp, x; init = 0.0)
            @test gradient ≈ logistic.(x)
            op = prepare_ad(prepare_nonallocating(weighted_softplus, w, x), backend, w, x; active = :x)
            value, gradient = ad_value_and_gradient!(op, g, w, x)
            @test value ≈ dot(w, log1pexp.(x))
            @test gradient ≈ w .* logistic.(x)
        end
    end
    # The generated plate reverse reads a vector softplus call as the rule.
    @test last(only(plate_reverse_report(prepare(owned_sum), 2))) == "rule"
    @test last(only(plate_reverse_report(prepare(foreign_sum), 1))) == "rule"

    # A table of a softplus over indexed invariant values holds the softplus
    # itself, whichever softplus the cell's loop calls.
    formula = ReactiveKernels._indexed_formula
    for f in (GlobalRef(parentmodule(log1pexp), :log1pexp), GlobalRef(ReactiveKernels, :_vector_log1pexp),
              GlobalRef(ReactiveKernels, :_vector_rk_log1pexp))
        @test formula(Expr(:call, f, :x), :x) == (:log1pexp, :input)
    end
end
end
