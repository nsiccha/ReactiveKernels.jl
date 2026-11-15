using DifferentiationInterface, Enzyme, ReactiveKernels, ReactiveKernelsPPL, Test
import SpecialFunctions
using Statistics: mean
include(joinpath(pkgdir(ReactiveKernels), "test", "jax_python.jl"))

# Ordinal thresholds the model computes from sampled values (snag
# rkppl-sampled-pe-bc377786): one vector shared by every observation
# (`Ref(cuts)` of a definition), one vector per group computed in a `@plate`
# cell (`Ref(cuts)` of a cell local), and one vector per observation
# (`cutrows[g]`). The oracle below is independent mathematics in 256 bits:
# no RK-PPL lowering, distribution adapter or derivative rule.

const _ct_links = (
    LogitLink = (F = z -> inv(1 + exp(-z)), CC = z -> inv(1 + exp(z))),
    ProbitLink = (F = z -> SpecialFunctions.erfc(-z / sqrt(big(2))) / 2,
        CC = z -> SpecialFunctions.erfc(z / sqrt(big(2))) / 2),
    CloglogLink = (F = z -> -expm1(-exp(z)), CC = z -> exp(-exp(z))))

# log P(y = k) for thresholds c, location eta and discrimination d.
function _ct_logmass(link, structure, c, eta, d, k)
    F, CC = _ct_links[link]
    K = length(c) + 1
    1 <= k <= K || return big(-Inf)
    z(j) = d * (big(c[j]) - eta)
    if structure === :Cumulative
        lo = k == 1 ? big(0) : F(z(k - 1))
        hi = k == K ? big(1) : F(z(k))
        # A level above eta takes its mass from the complement.
        mass = k > 1 && z(k - 1) > 0 ?
            CC(z(k - 1)) - (k == K ? big(0) : CC(z(k))) : hi - lo
        return mass > 0 ? log(mass) : big(-Inf)
    end
    lp = sum((log(CC(z(j))) for j in 1:k-1); init = big(0))
    return k == K ? lp : lp + log(F(z(k)))
end

_ct_stdnormal(v) = sum(t -> -(log(2 * big(pi)) + big(t)^2) / 2, v; init = big(0))
_ct_normal(x, s) = -(log(2 * big(pi)) + 2log(big(s)) + (big(x) / s)^2) / 2
_ct_centred(m, lg) = (s = cumsum(vcat(big(0), exp.(big.(lg)))); m .+ s .- mean(s))
_ct_sum(x::Real) = x
_ct_sum(xs) = sum(_ct_sum, xs; init = 0.0)

function _ct_fd(f, u)
    h = cbrt(eps(Float64))
    map(eachindex(u)) do i
        up, down = copy(u), copy(u)
        up[i] += h
        down[i] -= h
        (f(up) - f(down)) / (2h)
    end
end

function _ct_bound(ast, data)
    plan = lower_rkppl(ast, data; mod = @__MODULE__, conditioned = (:y,))
    bind_data(plan, data)
end

# Value, ordinary Enzyme Reverse gradient, pointwise densities and the
# likelihood against `oracle(nt) = (; prior, pointwise)`, read from the
# layout's own constrained values; the input vector is not written.
function _ct_check(bound, u, oracle)
    built = build_kernel(bound)
    @test built.layout.total == length(u)
    original = copy(u)
    sampler = prepare_sampler(built, bound, u;
        backend = AutoEnzyme(; mode = Enzyme.Reverse))
    total(v) = (o = oracle(constrain(built.layout, v));
        Float64(o.prior + _ct_sum(o.pointwise)))
    value, grad = sampler_value_and_gradient!(sampler, similar(u), u)
    @test value ≈ total(u) rtol = 1e-12
    @test grad ≈ _ct_fd(total, u) rtol = 2e-6 atol = 2e-8
    o = oracle(constrain(built.layout, u))
    pw = Base.invokelatest(prepare_query(built, bound, :pointwise), u)
    @test _ct_sum(pw.y) ≈ Float64(_ct_sum(o.pointwise)) rtol = 1e-12
    @test map(x -> Float64.(x), pw.y) ≈ map(x -> Float64.(x), o.pointwise) rtol = 1e-10
    @test u == original
    return (; built, sampler)
end

_ct_nodes(ex) = ex isa Expr ? 1 + sum(_ct_nodes, ex.args; init = 0) : 1
_ct_loops(ex) = ex isa Expr ? Int(ex.head === :for || ex.head === :while) +
    sum(_ct_loops, ex.args; init = 0) : 0
_ct_shape(sampler) = (ex = Base.remove_linenums!(deepcopy(code_expr(sampler.kernel)));
    (_ct_nodes(ex), _ct_loops(ex)))

function _ct_jax_check(built, bound, data)
    rk_jax_available() || return @test_skip false
    src = jax_source(built, bound)
    names = first.(only(f for f in src.functions if f.name == "prepare_data").inputs)
    args = Tuple(getproperty(data, Symbol(name)) for name in names)
    u = collect(range(-0.4, 0.5; length = built.layout.total))
    native = prepare_query(built, bound, :sampler)
    got = rk_jax_evaluate(src, :log_density, (u, args...); grad = true, prepare = :arguments)
    @test got.value ≈ native(u) rtol = 1e-11
    @test got.grad ≈ _ct_fd(native, u) rtol = 2e-6 atol = 1e-7
    return src
end

# Centred cutpoints, through Statistics' `mean` or as its own sum over length.
_ct_centre(centre) = centre === :mean ? :(cuts = m .+ s .- mean(s)) :
    :(cuts = m .+ s .- sum(s) / length(s))

# One centred vector of computed cutpoints, read by every observation.
function _ct_shared_ast(structure, link; discrimination = false, centre = :mean)
    response = discrimination ?
        :(Ordinal.($structure(), $link(), b .* x, Ref(cuts), exp.(gamma .* x))) :
        :(Ordinal.($structure(), $link(), b .* x, Ref(cuts)))
    quote
        m ~ Normal(0, 1)
        b ~ Normal(0, 1)
        gamma ~ Normal(0, 1)
        log_gap[1:n_gaps] .~ Normal.(0, 1)
        s = cumsum(vcat(0.0, exp.(log_gap)))
        $(_ct_centre(centre))
        y .~ $response
    end
end

function _ct_shared_oracle(data, structure, link; discrimination = false)
    nt -> begin
        c = _ct_centred(big(nt.m), nt.log_gap)
        pointwise = [Float64(_ct_logmass(link, structure, c, big(nt.b) * data.x[i],
            discrimination ? exp(big(nt.gamma) * data.x[i]) : big(1), data.y[i]))
            for i in eachindex(data.y)]
        (; prior = _ct_stdnormal((nt.m, nt.b, nt.gamma, nt.log_gap...)), pointwise)
    end
end

@testset "a computed cutpoint vector shared by every observation" begin
    data = (; y = [1, 3, 2, 4, 1, 2, 4, 3], n_gaps = 2,
        x = [-0.5, 0.2, 0.1, 0.9, -1.0, 0.3, 1.4, -0.2])
    u = [0.15, -0.4, 0.25, 0.3, -0.2]
    for structure in (:Cumulative, :StoppingRatio), link in keys(_ct_links),
            discrimination in (false, true)
        _ct_check(_ct_bound(_ct_shared_ast(structure, link; discrimination), data), u,
            _ct_shared_oracle(data, structure, link; discrimination))
    end
end

@testset "an inline computed cutpoint vector reads as its named twin" begin
    data = (; y = [1, 3, 2, 4, 1, 2], n_gaps = 2, x = [-0.5, 0.2, 0.1, 0.9, -1.0, 0.3])
    for structure in (:Cumulative, :StoppingRatio)
        ast = _ct_shared_ast(structure, :LogitLink)
        # `cuts = m .+ s .- mean(s); … Ref(cuts)` with the definition inlined.
        ast.args[end] = :(y .~ Ordinal.($structure(), LogitLink(), b .* x,
            Ref(m .+ s .- mean(s))))
        filter!(st -> !Meta.isexpr(st, :(=)) || st.args[1] !== :cuts, ast.args)
        _ct_check(_ct_bound(ast, data), [0.15, -0.4, 0.25, 0.3, -0.2],
            _ct_shared_oracle(data, structure, :LogitLink))
    end
end

@testset "one program for every computed threshold count" begin
    for structure in (:Cumulative, :StoppingRatio)
        ast = _ct_shared_ast(structure, :ProbitLink)
        three = (; y = [1, 3, 2, 3], x = [-0.5, 0.2, 0.1, 0.9], n_gaps = 1)
        five = (; y = [1, 5, 2, 4, 3, 5], x = [-0.5, 0.2, 0.1, 0.9, -1.0, 0.3], n_gaps = 3)
        plan = lower_rkppl(ast, three; mod = @__MODULE__, conditioned = (:y,))
        shapes = map((three, five)) do data
            u = collect(range(-0.3, 0.4; length = 3 + data.n_gaps))
            result = _ct_check(bind_data(plan, data), u,
                _ct_shared_oracle(data, structure, :ProbitLink))
            _ct_shape(result.sampler)
        end
        # Generated names differ between builds; the program does not.
        @test shapes[1] == shapes[2]
    end
end

@testset "computed cumulative cutpoints keep the lazy order guard" begin
    # A vector of two free scalars is not ordered at every point.
    ast(structure) = quote
        c1 ~ Normal(0, 1)
        c2 ~ Normal(0, 1)
        cuts = vcat(c1, c2)
        y .~ Ordinal.($structure(), LogitLink(), 0.0 .* x, Ref(cuts))
    end
    data = (; y = [1, 2, 3, 2], x = [0.1, -0.3, 0.2, 0.4])
    oracle(structure) = nt -> (; prior = _ct_stdnormal((nt.c1, nt.c2)),
        pointwise = [Float64(_ct_logmass(:LogitLink, structure, [nt.c1, nt.c2],
            big(0), big(1), k)) for k in data.y])
    for structure in (:Cumulative, :StoppingRatio)
        bound = _ct_bound(ast(structure), data)
        _ct_check(bound, [-0.6, 0.8], oracle(structure))
        built = build_kernel(bound)
        lik = Base.invokelatest(prepare_query(built, bound, :likelihood), [0.8, -0.6])
        # Decreasing cumulative cutpoints have no law; stopping-ratio
        # thresholds need not increase.
        if structure === :Cumulative
            @test lik == -Inf
        else
            @test lik ≈ Float64(sum(k -> _ct_logmass(:LogitLink, :StoppingRatio,
                [0.8, -0.6], big(0), big(1), k), data.y))
        end
    end
end

@testset "a level beyond a computed vector is outside its support" begin
    for structure in (:Cumulative, :StoppingRatio)
        data = (; y = [1, 2, 5], x = [0.1, -0.3, 0.2], n_gaps = 2)
        bound = _ct_bound(_ct_shared_ast(structure, :LogitLink), data)
        built = build_kernel(bound)
        u = [0.15, -0.4, 0.25, 0.3, -0.2]
        @test Base.invokelatest(prepare_query(built, bound, :likelihood), u) == -Inf
        pw = Base.invokelatest(prepare_query(built, bound, :pointwise), u)
        @test isfinite(pw.y[1]) && isfinite(pw.y[2]) && pw.y[3] == -Inf
        @test_throws ContractValidationError _ct_bound(_ct_shared_ast(structure, :LogitLink),
            merge(data, (; y = [1, 0, 2])))
    end
end

# The reporter's program: each group's centred cutpoints from population
# log-gaps and the group's partially pooled deviations, in the group's cell.
function _ct_cell_ast(structure, link; centre = :mean)
    quote
        m ~ Normal(0, 1)
        b ~ Normal(0, 1)
        lt ~ Normal(0, 1)
        log_gap[1:n_gaps] .~ Normal.(0, 1)
        @plate for e in eachindex(y)
            u[e] .~ Normal.(zeros(n_gaps), exp(lt))
            s = cumsum(vcat(0.0, exp.(log_gap .+ u[e])))
            $(_ct_centre(centre))
            y[e] .~ Ordinal.($structure(), $link(), b .* x[e], Ref(cuts))
        end
    end
end

function _ct_cell_oracle(data, structure, link)
    nt -> begin
        tau = exp(big(nt.lt))
        prior = _ct_stdnormal((nt.m, nt.b, nt.lt, nt.log_gap...)) +
            sum(sum(t -> _ct_normal(t, tau), ue; init = big(0)) for ue in nt.u; init = big(0))
        pointwise = map(eachindex(data.y)) do e
            c = _ct_centred(big(nt.m), nt.log_gap .+ nt.u[e])
            Float64[Float64(_ct_logmass(link, structure, c, big(nt.b) * data.x[e][i],
                big(1), data.y[e][i])) for i in eachindex(data.y[e])]
        end
        (; prior, pointwise)
    end
end

function _ct_cells(n; n_gaps = 3)
    sizes = [mod(3e, 4) for e in 1:n]          # ragged, with empty groups
    y = [[mod(5e + 2i, n_gaps + 1) + 1 for i in 1:sizes[e]] for e in 1:n]
    x = [[0.55i - 1.0 + 0.1e for i in 1:sizes[e]] for e in 1:n]
    (; y, x, n_gaps)
end

@testset "per-group cutpoints computed in a cell" begin
    for structure in (:Cumulative, :StoppingRatio), link in keys(_ct_links)
        data = _ct_cells(4)
        u = collect(range(-0.5, 0.45; length = 6 + 4 * data.n_gaps))
        _ct_check(_ct_bound(_ct_cell_ast(structure, link), data), u,
            _ct_cell_oracle(data, structure, link))
    end
end

_ct_plates(program) = [(e.kind, e.depth) for e in recipe_inventory(program)
    if e.kind !== :ordinary]

@testset "per-group computed cutpoints retain the group and observation plates" begin
    for structure in (:Cumulative, :StoppingRatio)
        shapes = map((4, 40)) do n
            data = _ct_cells(n)
            u = collect(range(-0.5, 0.45; length = 6 + n * data.n_gaps))
            bound = _ct_bound(_ct_cell_ast(structure, :ProbitLink), data)
            result = _ct_check(bound, u, _ct_cell_oracle(data, structure, :ProbitLink))
            likelihood = prepare_query(result.built, bound, :likelihood)
            (_ct_shape(result.sampler), _ct_plates(result.built.spec), _ct_plates(likelihood))
        end
        # One observation plate per group inside the group plate, whatever
        # the number of groups: nothing is emitted per group or entry.
        @test shapes[1] == shapes[2]
        @test (:plate, 1) in shapes[1][2]
        @test count(==((:plate, 1)), shapes[1][3]) == 1
    end
end

# One computed vector per observation, gathered from one vector per group.
_ct_group_cuts(m, lg, U) = [(s = cumsum(vcat(0.0, exp.(lg .+ U[g, :]))); m .+ s .- mean(s))
    for g in axes(U, 1)]

@testset "one computed threshold vector per observation" begin
    ast(structure) = quote
        m ~ Normal(0, 1)
        b ~ Normal(0, 1)
        log_gap[1:n_gaps] .~ Normal.(0, 1)
        U[1:n_groups, 1:n_gaps] .~ Normal.(0, 1)
        cutrows = _ct_group_cuts(m, log_gap, U)
        y .~ Ordinal.($structure(), CloglogLink(), b .* x, cutrows[g])
    end
    data = (; y = [1, 3, 2, 4, 1, 2, 4, 3], g = [1, 2, 1, 3, 2, 3, 3, 1],
        x = [-0.5, 0.2, 0.1, 0.9, -1.0, 0.3, 1.4, -0.2], n_gaps = 2, n_groups = 3)
    oracle(structure) = nt -> (; prior = _ct_stdnormal((nt.m, nt.b, nt.log_gap..., nt.U...)),
        pointwise = [Float64(_ct_logmass(:CloglogLink, structure,
            _ct_centred(big(nt.m), nt.log_gap .+ nt.U[data.g[i], :]),
            big(nt.b) * data.x[i], big(1), data.y[i])) for i in eachindex(data.y)])
    for structure in (:Cumulative, :StoppingRatio)
        u = collect(range(-0.5, 0.45; length = 4 + 3 * 2))
        _ct_check(_ct_bound(ast(structure), data), u, oracle(structure))
    end
end

@testset "computed thresholds in jax_source" begin
    # The thresholds need nothing of their own in JAX. Centring with
    # Statistics' `mean` there needs RK's Statistics translation
    # (`ReactiveKernelsStatisticsExt`); without it only `sum(s) / length(s)`
    # is checked, and the skip is recorded.
    translated = Base.get_extension(ReactiveKernels, :ReactiveKernelsStatisticsExt) !== nothing
    translated || @test_skip false
    for centre in (translated ? (:sum, :mean) : (:sum,)), structure in (:Cumulative, :StoppingRatio)
        texts = String[]
        for n in (3, 8)
            data = (; y = mod.(1:n, 4) .+ 1, x = collect(range(-1.0, 1.0; length = n)), n_gaps = 2)
            bound = _ct_bound(_ct_shared_ast(structure, :ProbitLink; discrimination = true,
                centre), data)
            src = _ct_jax_check(build_kernel(bound), bound, data)
            rk_jax_available() && push!(texts, src.text)
        end
        isempty(texts) || @test allequal(texts)
        for n in (3, 6)
            data = _ct_cells(n)
            bound = _ct_bound(_ct_cell_ast(structure, :ProbitLink; centre), data)
            _ct_jax_check(build_kernel(bound), bound, data)
        end
    end
end

# Range, censored and truncated evidence over thresholds computed in a cell
# (snag jax-source-of-an-6bfaf24a). Its stopping-ratio survival loops run to
# the vector's length, a padded row's length in JAX: lazy steps to that row's
# static limit, which `jax.grad` reverses. The second form is the one
# BayesianRegressionModels emits: the cell's latent thresholds are stored by
# group and read in a second plate.
function _ct_evidence(kind, structure, link, eta, cuts)
    D = :(Ordinal.($structure(), $link(), $eta, Ref($cuts)))
    kind === :interval ? :(interval_censored.($D, hi[e])) :
        kind === :censored ? :(censored.($D, 2, 3)) : :(truncated.($D, 1, 3))
end
function _ct_cell_evidence_ast(kind, structure, link)
    quote
        m ~ Normal(0, 1)
        b ~ Normal(0, 1)
        lt ~ Normal(0, 1)
        log_gap[1:n_gaps] .~ Normal.(0, 1)
        @plate for e in eachindex(y)
            u[e] .~ Normal.(zeros(n_gaps), exp(lt))
            s = cumsum(vcat(0.0, exp.(log_gap .+ u[e])))
            $(_ct_centre(:sum))
            y[e] .~ $(_ct_evidence(kind, structure, link, :(b .* x[e]), :cuts))
        end
    end
end
function _ct_stored_evidence_ast(kind, structure, link)
    quote
        b ~ Normal(0, 1)
        @plate for e in eachindex(y)
            t[e] .~ Normal.(zeros(n_gaps + 1), 1.0)
            t_cell = t[e]
            stored[e] = t_cell
        end
        @plate for e in eachindex(y)
            y[e] .~ $(_ct_evidence(kind, structure, link, :(b .* x[e]), :(stored[e])))
        end
    end
end
function _ct_evidence_cells(kind, n)
    data = _ct_cells(n)
    K = data.n_gaps + 2
    # Censored and truncated responses lie within their bounds.
    y = kind === :interval ? data.y :
        [clamp.(ye, kind === :censored ? 2 : 1, 3) for ye in data.y]
    hi = [[min(ye[i] + 1 + mod(i, 2), K) for i in eachindex(ye)] for ye in y]
    merge(data, (; y, hi))
end

@testset "evidence over computed thresholds in jax_source ($kind, $structure)" for
        kind in (:interval, :censored, :truncated), structure in (:StoppingRatio, :Cumulative)
    for form in (_ct_cell_evidence_ast, _ct_stored_evidence_ast)
        texts = String[]
        for n in (3, 6)
            data = _ct_evidence_cells(kind, n)
            bound = _ct_bound(form(kind, structure, :CloglogLink), data)
            built = build_kernel(bound)
            # The native gradient is finite where the JAX one is checked.
            u = collect(range(-0.4, 0.5; length = built.layout.total))
            sampler = prepare_sampler(built, bound, u; backend = AutoEnzyme(; mode = Enzyme.Reverse))
            value, grad = sampler_value_and_gradient!(sampler, similar(u), u)
            @test isfinite(value) && all(isfinite, grad)
            src = _ct_jax_check(built, bound, data)
            if rk_jax_available()
                @test !occursin("while_loop", src.text)
                push!(texts, src.text)
            end
        end
        # One program for every number of groups.
        isempty(texts) || @test allequal(texts)
    end
end
