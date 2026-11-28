module SharedSoftplusTests
using Test, ReactiveKernels, MutatingFunctions
using DifferentiationInterface: AutoEnzyme
using LogExpFunctions: log1pexp, logistic, loglogistic, log1mlogistic
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
    # The width is read from the loop this process compiles, not from the CPU.
    compiled = maximum((parse(Int, m.captures[1]) for m in eachmatch(r"<(\d+) x double>", _loop_llvm(vector_sum)));
                       init = 1)
    @test ReactiveKernels._vector_doubles() == compiled
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

# A summed plate over two vectors with the given cell, and the cell as a
# function of its two values.
macro softplus_cell(name, cell)
    esc(quote
        @kernel $name(x::Vector{Float64}, w::Vector{Float64}) = begin
            cells = plate(x, w) do v, z
                $cell
            end
            total::Float64 = sum(cells)
            return total
        end
        $(Symbol(name, :_cell))(v, z) = $cell
    end)
end
smooth_term(z) = muladd(z, z, 1.0) / (1.0 + abs(z))

# A softplus beside another term.
@softplus_cell beside_square log1pexp(v) + 0.5 * z^2
@softplus_cell beside_cube log1pexp(v) + 0.5 * z^3
@softplus_cell beside_fourth log1pexp(v) + 0.5 * z^4
@softplus_cell beside_float_power log1pexp(v) + 0.5 * z^2.0
@softplus_cell beside_root log1pexp(v) + sqrt(abs2(z) + 1.0)
@softplus_cell beside_bare_root log1pexp(v) + sqrt(z)
@softplus_cell beside_function log1pexp(v) + smooth_term(z)
@softplus_cell beside_constant log1pexp(v) + log(2.0) * z
@softplus_cell beside_log log1pexp(v) + log(abs2(z) + 1.0)
@softplus_cell beside_log_sum log1pexp(v) + log(exp(z) + exp(-z) + exp(2.0 * z))

# A softplus written out.
@softplus_cell spelled_log1p log1p(exp(v)) + 0.5 * abs2(z)
@softplus_cell spelled_log log(1 + exp(v)) + 0.5 * abs2(z)
@softplus_cell spelled_logistic -log(logistic(-v)) + 0.5 * abs2(z)
@softplus_cell spelled_log1m -log1p(-logistic(v)) + 0.5 * abs2(z)
@softplus_cell spelled_complement -log(1 - logistic(v)) + 0.5 * abs2(z)
@softplus_cell spelled_loglogistic -loglogistic(-v) + 0.5 * abs2(z)
@softplus_cell spelled_log1mlogistic -log1mlogistic(v) + 0.5 * abs2(z)
@softplus_cell spelled_sigmoid -log(1 / (1 + exp(-(-v)))) + 0.5 * abs2(z)
@softplus_cell spelled_sigmoid_complement -log1p(-(1 / (1 + exp(-v)))) + 0.5 * abs2(z)
@softplus_cell spelled_beside_square log1p(exp(v)) + 0.5 * z^2
@softplus_cell spelled_beside_log log1p(exp(v)) + log(abs2(z) + 1.0)
# Any element type: the twins hand every operand that is not a `Float64` to
# the function or helper the cell names.
@kernel spelled_any(x, w) = begin
    cells = plate(x, w) do v, z
        log1p(exp(v)) - log(logistic(z))
    end
    total = sum(cells)
    return total
end
@kernel weighted_loglogistic(n::Vector{Float64}, x::Vector{Float64}) = begin
    total::Float64 = dot(n, loglogistic.(x))
    return total
end
@kernel plain_fifth(x::Vector{Float64}) = begin
    cells = plate(x) do v
        0.5 * v^5
    end
    total::Float64 = sum(cells)
    return total
end

# A sum over each other term by itself, for what the compiler makes of it.
square_sum(x) = (s = 0.0; @inbounds @simd for i in eachindex(x); s += x[i]^2; end; s)
cube_sum(x) = (s = 0.0; @inbounds @simd for i in eachindex(x); s += x[i]^3; end; s)
fourth_sum(x) = (s = 0.0; @inbounds @simd for i in eachindex(x); s += x[i]^4; end; s)
float_power_sum(x) = (s = 0.0; @inbounds @simd for i in eachindex(x); s += x[i]^2.0; end; s)
root_sum(x) = (s = 0.0; @inbounds @simd for i in eachindex(x); s += sqrt(abs2(x[i]) + 1.0); end; s)
bare_root_sum(x) = (s = 0.0; @inbounds @simd for i in eachindex(x); s += sqrt(x[i]); end; s)
function_sum(x) = (s = 0.0; @inbounds @simd for i in eachindex(x); s += smooth_term(x[i]); end; s)
log_sum(x) = (s = 0.0; @inbounds @simd for i in eachindex(x); s += log(abs2(x[i]) + 1.0); end; s)
# The same square with the function `^` itself as the callee, which Julia does
# not lower as a literal power.
@eval object_square_sum(x) = (s = 0.0; @inbounds @simd for i in eachindex(x)
    s += $(Expr(:call, ^, :(x[i]), 2)); end; s)
# Vector code as the package reads it: vectors as wide as a softplus loop's
# and no call left.
function _vector_loop(f)
    code = _loop_llvm(f)
    any(l -> occursin(r"\b(?:call|invoke)\b", l) && !occursin("@llvm.", l), eachline(IOBuffer(code))) && return false
    doubles = ReactiveKernels._vector_doubles()
    doubles >= 2 && maximum((parse(Int, m.captures[1]) for m in eachmatch(r"<(\d+) x double>", code)); init = 1) >= doubles
end
_twins(kernel) = sort!(unique!([String(m.match) for m in eachmatch(r"_vector_\w+|_log_exp_sum", string(code_expr(prepare(kernel))))]))

@testset "a softplus cell is vector code beside every call the compiler vectorizes" begin
    wide = ReactiveKernels._vector_doubles() >= ReactiveKernels._VECTOR_SOFTPLUS_DOUBLES
    simd = ReactiveKernels._vector_doubles() >= 2
    # What the compiler does with each term on this CPU and this Julia. A
    # square is a product, a bare square root can throw, Base's logarithm is a
    # call; the rest is the compiler's to say.
    @test _vector_loop(square_sum) == simd
    @test !_vector_loop(bare_root_sum) && !_vector_loop(log_sum)
    for (kernel, loop) in ((beside_square, square_sum), (beside_cube, cube_sum), (beside_fourth, fourth_sum),
                           (beside_float_power, float_power_sum), (beside_root, root_sum),
                           (beside_bare_root, bare_root_sum), (beside_function, function_sum),
                           (beside_log, log_sum))
        @test ("_vector_log1pexp" in _twins(kernel)) == (wide && _vector_loop(loop))
    end
    # A call of literals alone is a constant of the loop.
    @test ("_vector_log1pexp" in _twins(beside_constant)) == wide
    # The vector logarithm and the vector softplus in one cell.
    @test _twins(beside_log_sum) == (wide ? ["_log_exp_sum", "_vector_log1pexp"] : String[])

    # The answer for one call, against the loop written by hand.
    compiles = ReactiveKernels._vector_compiles
    ref(m, n) = GlobalRef(m, n)
    @test compiles(:($(ref(Base, :^))(z, 2))) == _vector_loop(square_sum)
    @test compiles(:($(ref(Base, :^))(z, 4))) == _vector_loop(fourth_sum)
    @test compiles(:($(ref(Base, :^))(z, 2.0))) == _vector_loop(float_power_sum)
    @test compiles(Expr(:call, ^, :z, 2)) == _vector_loop(object_square_sum)
    @test compiles(:($(ref(Base, :sqrt))($(ref(Base, :+))($(ref(Base, :abs2))(z), 1.0)))) == _vector_loop(root_sum)
    @test compiles(:($(ref(@__MODULE__, :smooth_term))(z))) == _vector_loop(function_sum)
    @test !compiles(:($(ref(Base, :sqrt))(z)))                      # its domain check can throw
    @test !compiles(:($(ref(Base, :log))(z)))                       # a call that stays a call
    @test compiles(:($(ref(Base, :log))(2.0)))                      # a constant
    @test !compiles(:(f(z)))                                        # a callee it cannot name
    @test !compiles(:($(ref(Base, :abs2))(data[i])))                # an indexed read
    @test !compiles(:($(ref(Base, :abs2))(__constants__[2])))       # a table read of unknown type
    scaled = let c = 2.0; z -> c * z end
    @test !compiles(Expr(:call, scaled, :z))                        # a callable with a value of its own
    # One answer per call, whatever names it reads, and none asked for a cell
    # without a softplus.
    asked = length(ReactiveKernels._VECTOR_COMPILED)
    @test compiles(:($(ref(Base, :^))(other, 2))) == compiles(:($(ref(Base, :^))(z, 2)))
    @test prepare(plain_fifth)([0.5, -1.0]) ≈ 0.5 * (0.5^5 - 1.0)
    @test length(ReactiveKernels._VECTOR_COMPILED) == asked

    # Values on plates shorter than one vector and longer, and the error a
    # bare square root raises.
    for kernel_cell in ((beside_square, beside_square_cell), (beside_cube, beside_cube_cell),
                        (beside_fourth, beside_fourth_cell), (beside_float_power, beside_float_power_cell),
                        (beside_root, beside_root_cell), (beside_function, beside_function_cell),
                        (beside_constant, beside_constant_cell), (beside_log, beside_log_cell),
                        (beside_log_sum, beside_log_sum_cell)), n in (0, 1, 7, 513)
        kernel, cell = kernel_cell
        x = [sin(i) * 30 for i in 1:n]; w = [1.5 * cos(i) for i in 1:n]
        @test prepare(kernel)(x, w) ≈ sum(cell.(x, w); init = 0.0) rtol = 1e-13
    end
    @test prepare(beside_bare_root)([0.3, -2.0], [4.0, 9.0]) ≈ log1pexp(0.3) + log1pexp(-2.0) + 5.0
    @test_throws DomainError prepare(beside_bare_root)([0.3, -2.0], [4.0, -9.0])

    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    for enabled in (false, true)
        with_plate_reverse(enabled) do
            for (kernel, slope) in ((beside_square, identity), (beside_root, z -> z / sqrt(abs2(z) + 1.0)))
                x0 = [-0.2, 0.3, 40.0]; w0 = [0.5, -1.5, 2.5]
                by_x = prepare_ad(prepare(kernel), backend, x0, w0; active = :x)
                by_w = prepare_ad(prepare(kernel), backend, x0, w0; active = :w)
                for n in (0, 7, 513)
                    x = [sin(i) * 30 for i in 1:n]; w = [1.5 * cos(i) for i in 1:n]
                    g = fill(99.0, n)
                    @test ad_value_and_gradient!(by_x, g, x, w)[2] ≈ logistic.(x)
                    @test ad_value_and_gradient!(by_w, g, x, w)[2] ≈ slope.(w)
                end
            end
        end
    end
end

@testset "every spelling of a softplus is the vector softplus in a vector loop" begin
    RKS = ReactiveKernels
    wide = RKS._vector_doubles() >= RKS._VECTOR_SOFTPLUS_DOUBLES
    spellings = ((spelled_log1p, "_vector_log1p_of"),
                 (spelled_log, "_vector_log_one_plus_of"),
                 (spelled_logistic, "_vector_log_of"),
                 (spelled_log1m, "_vector_log1p_neg_of"),
                 (spelled_complement, "_vector_log_one_minus_of"),
                 (spelled_loglogistic, "_vector_loglogistic"),
                 (spelled_log1mlogistic, "_vector_log1mlogistic"),
                 (spelled_sigmoid, "_vector_log_sigmoid"),
                 (spelled_sigmoid_complement, "_vector_log1m_sigmoid"),
                 (spelled_beside_square, "_vector_log1p_of"))
    for (kernel, twin) in spellings
        @test _twins(kernel) == (wide ? [twin] : String[])
        for n in (0, 1, 7, 513)
            x = [sin(i) * 30 for i in 1:n]; w = [1.5 * cos(i) for i in 1:n]
            # The reference is the softplus: several of these spellings lose
            # digits when plain Julia evaluates them as written.
            @test prepare(kernel)(x, w) ≈ sum(log1pexp, x; init = 0.0) + sum(z -> 0.5 * abs2(z), w; init = 0.0) rtol = 1e-13
        end
    end
    # One call that keeps the loop scalar keeps the helper too.
    @test isempty(_twins(spelled_beside_log))
    @test prepare(spelled_beside_log)([0.3, -2.0], [1.0, 2.0]) ≈ sum(spelled_beside_log_cell.([0.3, -2.0], [1.0, 2.0]))

    # Each twin: the helper's value to rounding for a `Float64`, the helper
    # itself for any other operand and for any other function.
    grid = (range(-800.0, 800.0; length = 4001)..., -745.2, -745.0, -40.0, -37.0, -1.0, -0.0, 0.0, 1.0,
            33.9, 34.0, 40.0, 700.0, 1e300, -1e300, Inf, -Inf)
    pairs = ((x -> RKS._vector_log1p_of(exp, x), x -> RKS._rk_log1p_of(exp, x)),
             (x -> RKS._vector_log_one_plus_of(1, exp, x), x -> RKS._rk_log_one_plus_of(1, exp, x)),
             (x -> RKS._vector_log_of(logistic, x), x -> RKS._rk_log_of(logistic, x)),
             (x -> RKS._vector_log1p_neg_of(logistic, x), x -> RKS._rk_log1p_neg_of(logistic, x)),
             (x -> RKS._vector_log_one_minus_of(1, logistic, x), x -> RKS._rk_log_one_minus_of(1, logistic, x)),
             (x -> RKS._vector_log_sigmoid(1, 1, exp, x), x -> RKS._rk_log_sigmoid(1, 1, exp, x)),
             (x -> RKS._vector_log1m_sigmoid(1, 1, exp, x), x -> RKS._rk_log1m_sigmoid(1, 1, exp, x)),
             (RKS._vector_loglogistic, loglogistic), (RKS._vector_log1mlogistic, log1mlogistic))
    for (twin, helper) in pairs
        @test maximum(x -> _ulps(twin(x), helper(x)), grid) <= 1.0
        @test isnan(twin(NaN)) && twin(0.5) isa Float64
    end
    @test RKS._vector_log1p_of(exp, 0.5f0) === RKS._rk_log1p_of(exp, 0.5f0)
    @test RKS._vector_log1p_of(exp, 1 + 2im) === log1p(exp(1 + 2im))
    @test RKS._vector_log1p_of(abs, -0.5) === log1p(0.5)
    @test RKS._vector_log_of(logistic, big"0.5") == loglogistic(big"0.5")
    @test RKS._vector_log_of(exp, 0.5) === 0.5
    @test RKS._vector_log_one_plus_of(big"1", exp, 0.5) isa BigFloat
    @test RKS._vector_loglogistic(0.5f0) === loglogistic(0.5f0)
    @test RKS._vector_log1mlogistic(big"0.5") == log1mlogistic(big"0.5")
    for (T, tolerance) in ((Float32, 1f-5), (BigFloat, 1e-30))
        x = T[-0.2, 0.3, 4.0]; w = T[0.5, -1.5, 2.5]
        value = prepare(spelled_any)(x, w)
        @test value isa T
        @test value ≈ sum(log1pexp, x) - sum(loglogistic, w) rtol = tolerance
    end

    # A broadcast of a logistic logarithm that a non-allocating program
    # materializes by itself, as one of `log1pexp`.
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    n0 = [1.0, 2.0, 3.0]; x0 = [-0.2, 0.3, 40.0]
    weighted = prepare_nonallocating(weighted_loglogistic, n0, x0)
    @test occursin("_vector_loglogistic", string(code_expr(weighted))) == wide
    for n in (0, 7, 513)
        x = [sin(i) * 30 for i in 1:n]; w = [1.0 + i % 3 for i in 1:n]
        @test prepare_nonallocating(weighted_loglogistic, w, x)(w, x) ≈ dot(w, loglogistic.(x))
        g = fill(99.0, n)
        op = prepare_ad(prepare_nonallocating(weighted_loglogistic, w, x), backend, w, x; active = :x)
        value, gradient = ad_value_and_gradient!(op, g, w, x)
        @test value ≈ dot(w, loglogistic.(x))
        @test gradient ≈ w .* logistic.(.-x)
    end

    # Which helper calls have a twin: the ones whose method is a softplus.
    twin = RKS._vector_helper_twin
    e = GlobalRef(Base, :exp); l = GlobalRef(parentmodule(logistic), :logistic)
    @test twin(RKS._rk_log1p_of, Any[e, :x]) === RKS._vector_log1p_of
    @test twin(RKS._rk_log1p_of, Any[GlobalRef(Base, :abs), :x]) === nothing
    @test twin(RKS._rk_log_of, Any[l, :x]) === RKS._vector_log_of
    @test twin(RKS._rk_log_of, Any[e, :x]) === nothing                      # `log(exp(x))` is `x`
    @test twin(RKS._rk_log_one_plus_of, Any[1, e, :x]) === RKS._vector_log_one_plus_of
    @test twin(RKS._rk_log_one_plus_of, Any[:c, e, :x]) === nothing         # not a literal one
    @test twin(RKS._rk_log_sigmoid, Any[1, 1.0, e, :u]) === RKS._vector_log_sigmoid
    @test twin(RKS._rk_log_add_of, Any[e, :a, e, :b]) === nothing
    # With two doubles to a vector the softplus keeps its helper.
    written = Expr(:call, GlobalRef(RKS, :_rk_log1p_of), e, :x)
    @test RKS._vector_cell_forms(written, 2) === written
    @test RKS._vector_cell_forms(written, 8) == Expr(:call, GlobalRef(RKS, :_vector_log1p_of), e, :x)
    @test RKS._vectorize_cell(Expr(:block, written)) == Expr(:block, wide ? RKS._vector_cell_forms(written, 8) : written)

    for enabled in (false, true)
        with_plate_reverse(enabled) do
            for kernel in (spelled_log1p, spelled_logistic, spelled_loglogistic, spelled_sigmoid)
                x0 = [-0.2, 0.3, 40.0]; w0 = [0.5, -1.5, 2.5]
                by_x = prepare_ad(prepare(kernel), backend, x0, w0; active = :x)
                for n in (0, 7, 513)
                    x = [sin(i) * 30 for i in 1:n]; w = [1.5 * cos(i) for i in 1:n]
                    g = fill(99.0, n)
                    value, gradient = ad_value_and_gradient!(by_x, g, x, w)
                    @test value ≈ sum(log1pexp, x; init = 0.0) + sum(z -> 0.5 * abs2(z), w; init = 0.0)
                    @test gradient ≈ logistic.(x)
                end
            end
        end
    end
    # The generated plate reverse reads a twin as the method it selects.
    for kernel in (spelled_log1p, spelled_logistic, spelled_loglogistic)
        @test last(only(plate_reverse_report(prepare(kernel), 1))) == "rule"
    end
end
end
