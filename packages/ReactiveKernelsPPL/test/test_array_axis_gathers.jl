# Helpers and imports come from test_generator.jl and test_array_values.jl.
# These are ordinary Julia gathers: a column selection stays K×N until
# the authored adjoint makes it N×K for a matrix-vector product.
function _axis_gather_fixture(kind, G, n)
    axis = kind === :subset ? :(levels(g)[[3, 1]]) : :(levels(g))
    definition = kind === :derived ? :(C = 2 .* B) :
        kind === :adjoint ? :(C = B') : nothing
    value = kind === :adjoint ? :(C[g, 1] .+ C[g, :] * w) :
        kind === :derived ? :(C[1, g] .+ C[:, g]' * w) :
            :(B[1, g] .+ B[:, g]' * w)
    ast = quote
        B[1:2, $axis] .~ Normal.(0, 1)
        w[1:2] .~ Normal.(0, 1)
        $definition
        mu = $value
        y .~ Normal.(mu, 0.7)
    end
    filter!(!isnothing, ast.args)
    g = ["g$(mod1(2i + 1, G))" for i in 1:n]
    y = [sin(0.3i) for i in 1:n]
    bound = bind_data(lower_rkppl(ast, (:g, :y); conditioned = (:g, :y)),
        Dict(:g => g, :y => y))
    built = build_kernel(bound)
    return (; bound, built, g, y, kind)
end

function _axis_gather_oracle(fx, u)
    nt = constrain(fx.built.layout, u)
    levels = sort!(unique(fx.g))
    if fx.kind === :subset
        selected = levels[[3, 1]]
        codes = [something(findfirst(==(g), selected), 0) + 1 for g in fx.g]
        B = hcat(zeros(2, 1), nt.B)
    else
        codes = [findfirst(==(g), levels) for g in fx.g]
        B = fx.kind === :derived ? 2 .* nt.B : nt.B
    end
    mu = B[1, codes] .+ B[:, codes]' * nt.w
    return sum(logpdf.(Normal(), nt.B)) + sum(logpdf.(Normal(), nt.w)) +
        sum(logpdf.(Normal.(mu, 0.7), fx.y)) + logjac(fx.built.layout, u)
end

@testset "array gathers: either declared axis" begin
    for orientation in (:rows, :cols), read in (:scalar, :matvec),
            index in (:levels, :positions), derived in (false, true)
        axis = index === :levels ? :(levels(g)) : :(1:3)
        dims = orientation === :rows ? Any[axis, :(1:2)] : Any[:(1:2), axis]
        declaration = :($(Expr(:ref, :B, dims...)) .~ Normal.(0, 1))
        # Adjoint transfers declared axis metadata as well as values.
        definition = derived ? :(C = B') : nothing
        base = derived ? :C : :B
        rows = (orientation === :rows) != derived
        value = read === :scalar ?
            (rows ? :($base[g, 1]) : :($base[1, g])) :
            (rows ? :($base[g, :] * w) : :($base[:, g]' * w))
        ast = quote
            $declaration
            w[1:2] .~ Normal.(0, 1)
            $definition
            mu = $value
            y .~ Normal.(mu, 0.7)
        end
        filter!(!isnothing, ast.args)
        plan = lower_rkppl(ast, (:g, :y); conditioned = (:g, :y))
        g = index === :levels ? _av_g() : [2, 1, 3, 1, 2, 3, 1, 2]
        bound = bind_data(plan, Dict(:g => g, :y => _av_y()))
        built = build_kernel(bound)
        u = _av_point(built.layout.total)
        nt = constrain(built.layout, u)
        codes = index === :levels ?
            [findfirst(==(v), ["a", "b", "c"]) for v in g] : g
        A = derived ? nt.B' : nt.B
        mu = read === :scalar ?
            (rows ? A[codes, 1] : A[1, codes]) :
            (rows ? A[codes, :] * nt.w : A[:, codes]' * nt.w)
        @test _av_node(built, bound, :likelihood, u) ≈
            sum(logpdf.(Normal.(mu, 0.7), _av_y()))
        @test _av_node(built, bound, :prior, u) ≈
            sum(logpdf.(Normal(), nt.B)) + sum(logpdf.(Normal(), nt.w))
        _check_gradient(built.spec, bound, u)
    end
end

@testset "array gathers: eachcol centered multivariate prior" begin
    plan = lower_rkppl(:(begin
        L ~ LKJCholesky(2, 2.0)
        sd[1:2] .~ Exponential.(1)
        F = sd .* L
        eachcol(B[1:2, levels(g)]) .~ MvNormalCholesky([0.5, -0.25], F)
        w[1:2] .~ Normal.(0, 1)
        mu = B[1, g] .+ B[:, g]' * w
        y .~ Normal.(mu, 0.7)
    end), (:g, :y); conditioned = (:g, :y))
    bound = bind_data(plan, Dict(:g => _av_g(), :y => _av_y()))
    built = build_kernel(bound)
    u = _av_point(built.layout.total)
    nt = constrain(built.layout, u)
    codes = [findfirst(==(v), ["a", "b", "c"]) for v in _av_g()]
    mu = nt.B[1, codes] .+ nt.B[:, codes]' * nt.w
    F = nt.sd .* nt.L
    prior = logpdf(LKJCholesky(2, 2.0), Cholesky(LowerTriangular(nt.L))) +
        sum(logpdf.(Exponential(1), nt.sd)) + sum(logpdf.(Normal(), nt.w)) +
        sum(logpdf(MvNormal([0.5, -0.25], F * F'), b) for b in eachcol(nt.B))
    @test _av_node(built, bound, :prior, u) ≈ prior
    @test _av_node(built, bound, :likelihood, u) ≈
        sum(logpdf.(Normal.(mu, 0.7), _av_y()))
    _check_gradient(built.spec, bound, u)
end

@testset "array gathers: independent subset codes on both axes" begin
    plan = lower_rkppl(:(begin
        B[levels(h)[[3, 1]], levels(g)[[2, 3]]] .~ Normal.(0, 1)
        w[1:2] .~ Normal.(0, 1)
        mu = B[g, 1] .+ B[1, g] .+ B[:, g]' * w
        y .~ Normal.(mu, 0.7)
    end), (:g, :h, :y); conditioned = (:g, :h, :y))
    bound = bind_data(plan, Dict(:g => _av_g(), :h => reverse(_av_g()),
        :y => _av_y()))
    built = build_kernel(bound)
    u = _av_point(built.layout.total)
    nt = constrain(built.layout, u)
    rows = [something(findfirst(==(g), ["c", "a"]), 0) for g in _av_g()]
    cols = [something(findfirst(==(g), ["b", "c"]), 0) for g in _av_g()]
    rowpad = vcat(zeros(1, 2), nt.B)
    colpad = hcat(zeros(2, 1), nt.B)
    mu = rowpad[rows .+ 1, 1] .+ colpad[1, cols .+ 1] .+
        colpad[:, cols .+ 1]' * nt.w
    @test size(nt.B) == (2, 2)
    @test _av_node(built, bound, :likelihood, u) ≈
        sum(logpdf.(Normal.(mu, 0.7), _av_y()))
    _check_gradient(built.spec, bound, u)
end

@testset "array gathers: second-axis bound validation" begin
    positional = lower_rkppl(:(begin
        B[1:2, 1:3] .~ Normal.(0, 1)
        mu = B[1, g]
        y .~ Normal.(mu, 1)
    end), (:g, :y); conditioned = (:g, :y))
    for g in ([0, 1], [1, 4], [true, false], [1.0, 2.0], ["a", "b"])
        @test_throws ContractValidationError bind_data(positional,
            Dict(:g => g, :y => [0.1, 0.2]))
    end
    levels = lower_rkppl(:(begin
        B[1:2, levels(h)] .~ Normal.(0, 1)
        mu = B[1, g]
        y .~ Normal.(mu, 1)
    end), (:g, :h, :y); conditioned = (:g, :h, :y))
    @test_throws ContractValidationError bind_data(levels,
        Dict(:g => ["a", "unknown"], :h => ["a", "b"], :y => [0.1, 0.2]))
    @test_throws ContractValidationError lower_rkppl(:(begin
        B[levels(g), levels(h)] .~ Normal.(0, 1)
        mu = B[g, h]
        y .~ Normal.(mu, 1)
    end), (:g, :h, :y); conditioned = (:g, :h, :y))
end
