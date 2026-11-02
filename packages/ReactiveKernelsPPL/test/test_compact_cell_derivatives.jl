using ReactiveKernels, ReactiveKernelsPPL, Distributions, Enzyme, Test, LinearAlgebra
using DifferentiationInterface: AutoEnzyme, SecondOrder
using SpecialFunctions: digamma

# Hessian-vector products and third-order contractions of ONE cell's log
# density over a compact selection of packed coordinates
# (`prepare_compact_cell_ad`, snag `build-brm-s-spar-454be53f`): the m × m
# Hessian block of a cell over the m coordinates it reads, from one call with
# the m unit directions, and `∇(vᵀ H v)` for a Laplace-marginal gradient.
# Helpers `_CQ_GROUPED`, `_cq_grouped_data`, `_cq_bound` and `_cq_central` come
# from `test_cell_query.jl`.
#
# The backends are plain static Enzyme: both Hessian compositions and
# reverse over forward over forward for third-order contractions.
# The compact overlay reads either a constant base entry or an active one;
# registered Enzyme 0.13.216 refuses that read under static activity
# (`EnzymeRuntimeActivityError`), and Enzyme `dogfood` repairs it from
# `21a909f7` (Enzyme snag `static-reverse-s-f9483e9c`). On registered Enzyme
# each testset records that gap as broken; drop the branch once a registered
# Enzyme carries the repair.
#
# Nested static modes keep no copy of the constant packed base on dogfood from
# `7fad557` (Enzyme snag `static-reverse-o-3d3debe4`). Under `--check-bounds=yes`
# the grouped decay-curve cell (an LKJ factor and `R = Z * F'`) still copies it
# once per call (Enzyme todo `0t6ufcp`); its cost testset records that as
# broken.

const _CCD_SECOND = SecondOrder(AutoEnzyme(; mode = Enzyme.Reverse),
                                AutoEnzyme(; mode = Enzyme.Forward))
const _CCD_FORWARD_REVERSE = SecondOrder(AutoEnzyme(; mode = Enzyme.Forward),
                                        AutoEnzyme(; mode = Enzyme.Reverse))
const _CCD_THIRD = SecondOrder(_CCD_SECOND, AutoEnzyme(; mode = Enzyme.Forward))

# Runs `f()`; returns `nothing` when registered Enzyme refuses the overlay's
# read, and rethrows anything else.
function _ccd_static(f)
    try
        f()
    catch err
        err isa Enzyme.Compiler.EnzymeRuntimeActivityError || rethrow()
        @test occursin("Mismatched activity", sprint(showerror, err))
        @test_broken false
        nothing
    end
end

_ccd_unit(k, m) = (e = zeros(m); e[k] = 1.0; e)

# A Poisson-log cell per group whose log rate is linear in the intercept, the
# slope and the group's effect, so the cell's derivatives over those
# coordinates are analytic: with c_j = (1, x_j, 1), H = -Σ exp(η_j) c_j c_jᵀ and
# D³ℓ[v, v, ·] = -Σ exp(η_j) (c_j ⋅ v)² c_j.
const _CCD_POISSON = quote
    a ~ Normal(0, 1)
    b ~ Normal(0, 1)
    z[levels(group)] .~ Normal.(0, 1)
    zg = z[group]
    @plate for i in eachindex(y)
        eta = a .+ b .* x[i] .+ zg[i]
        y[i] .~ Poisson.(exp.(eta))
    end
end

function _ccd_poisson_data(n)
    x = [collect(0.3 .* (1:(mod1(g, 4) - 1)) .- 0.5) for g in 1:n]
    y = [[mod(3g + j, 5) for j in eachindex(xg)] for (g, xg) in enumerate(x)]
    (; group = collect(1:n), x, y)       # labels sort as the coordinates z.1 … z.n
end

# The analytic value, gradient, Hessian and third-order contraction of cell `i`
# over the coordinates (a, b, z.i, extra...), the extras unread by the cell.
function _ccd_poisson_cell(data, a, b, zi, v, nextra)
    x, y = data.x, data.y
    m = 3 + nextra
    value, g, H, t = 0.0, zeros(m), zeros(m, m), zeros(m)
    for (xj, yj) in zip(x, y)
        eta = a + b * xj + zi
        rate = exp(eta)
        c = [1.0, xj, 1.0, zeros(nextra)...]
        value += logpdf(Poisson(rate), yj)
        g .+= (yj - rate) .* c
        H .-= rate .* (c * c')
        t .-= rate * dot(c, v)^2 .* c
    end
    value, g, H, t
end

@testset "compact cell derivatives: analytic Hessian blocks and third-order contractions" begin
    data = _ccd_poisson_data(6)
    bound, built = _cq_bound(_CCD_POISSON, data, (:y,))
    names = coordinate_names(built.layout)
    @test names == [:a; :b; [Symbol("z.", g) for g in 1:6]]
    u = collect(range(-0.3, 0.4; length = built.layout.total))
    # Cell i's selection: the shared intercept and slope, its own effect, and
    # the next group's effect, which the cell does not read.
    selection(i) = [1, 2, 2 + i, 3 + i]
    m = 4
    directions = ntuple(k -> _ccd_unit(k, m), m)
    v = [0.4, -0.7, 1.1, 0.3]
    h = prepare_compact_cell_ad(prepare_ad_hvp, built, bound, :y, u, selection(1),
        _CCD_SECOND, directions)
    hf = prepare_compact_cell_ad(prepare_ad_hvp, built, bound, :y, u, selection(1),
        _CCD_FORWARD_REVERSE, directions)
    t = prepare_compact_cell_ad(prepare_ad_third, built, bound, :y, u, selection(1),
        _CCD_THIRD, (v,))
    @test h isa CompactCellAD
    q = prepare_cell_query(built, bound, :y)
    base = copy(u)
    _ccd_static() do
        for i in (1, 3, 4, 5)          # groups 1 and 5 have no observation
            indices = selection(i)
            active = u[indices] .+ [0.05, -0.1, 0.2, 0.0]
            overlaid = copy(u)
            overlaid[indices] = active
            cell = (; x = data.x[i], y = data.y[i])
            value, g, H, third = _ccd_poisson_cell(cell, active[1], active[2],
                active[3], v, 1)
            gh, columns = ad_gradient_and_hvp(h, directions, base, active, indices, i)
            @test reduce(hcat, columns) ≈ H atol = 1e-12
            @test gh ≈ g atol = 1e-12
            @test q(overlaid, i) ≈ value atol = 1e-12
            # Forward over reverse batches the directions through the reverse
            # gradient. Preserve all mixed columns, including unused slots,
            # when reusing the preparation at another point and cell.
            gf, forward_columns = ad_gradient_and_hvp(hf, directions,
                base, active, indices, i)
            @test gf ≈ g atol = 1e-12
            @test reduce(hcat, forward_columns) ≈ H atol = 1e-12
            # The forward-over-reverse preparation fixes the direction tuple
            # type; change its values while retaining its four directions.
            mixed = (v, directions[2], directions[3], directions[4])
            @test first(ad_hvp(hf, mixed, base, active, indices, i)) ≈ H * v atol = 1e-12
            gt, (hv,), (tv,) = ad_gradient_hvp_and_third(t, (v,), base, active, indices, i)
            @test gt ≈ g atol = 1e-12
            @test hv ≈ H * v atol = 1e-12
            @test tv ≈ third atol = 1e-12
            @test (ad_third(t, (v,), base, active, indices, i))[1] ≈ third atol = 1e-12
            # In place, into caller buffers of the selection's length.
            gbuf = zeros(m)
            outs = ntuple(_ -> zeros(m), m)
            ad_gradient_and_hvp!(h, gbuf, outs, directions, base, active, indices, i)
            @test gbuf ≈ g atol = 1e-12
            @test reduce(hcat, outs) ≈ H atol = 1e-12
            fill!(gbuf, NaN)
            foreach(out -> fill!(out, NaN), outs)
            ad_gradient_and_hvp!(hf, gbuf, outs, directions, base, active, indices, i)
            @test gbuf ≈ g atol = 1e-12
            @test reduce(hcat, outs) ≈ H atol = 1e-12
            hout, tout = (zeros(m),), (zeros(m),)
            ad_gradient_hvp_and_third!(t, gbuf, hout, tout, (v,), base, active, indices, i)
            @test only(tout) ≈ third atol = 1e-12
            @test only(hout) ≈ H * v atol = 1e-12
            @test base == u
            @test indices == selection(i)
            @test active == u[indices] .+ [0.05, -0.1, 0.2, 0.0]
            @test directions == ntuple(k -> _ccd_unit(k, m), m)
            @test v == [0.4, -0.7, 1.1, 0.3]
        end
    end
    # The trailing arguments are the compact sampler's.
    @test_throws ReactiveKernelsPPL.ContractValidationError ad_hvp(h, directions,
        base, u[selection(1)], selection(1), 1.0)
    @test_throws ReactiveKernelsPPL.ContractValidationError ad_hvp(h, directions,
        base, u[selection(1)], [2, 1, 3, 4], 1)
    @test_throws ReactiveKernelsPPL.ContractValidationError ad_hvp(h, directions)
    @test data == _ccd_poisson_data(6)
end

# Independent negative-binomial derivatives in (intercept, log shape,
# selected group effect, unused group effect) coordinates. Differencing this
# gradient checks H*v without differentiating RK's generated logbeta rule.
function _ccd_negbin_gradient(x, y)
    mu, shape = exp(x[1] + x[3]), exp(x[2])
    eta_gradient = y - (y + shape) * mu / (shape + mu)
    shape_gradient = shape * (digamma(y + shape) - digamma(shape) +
        log(shape) - log(shape + mu) + 1 - (y + shape) / (shape + mu))
    [eta_gradient, shape_gradient, eta_gradient, 0.0]
end

@testset "compact cell derivatives: negative-binomial forward over reverse" begin
    # An ordinary elementwise observation reads a gathered group effect and
    # a shared shape. Its logbeta derivative graph contains digamma calls.
    # Forward over reverse could not link those calls before Enzyme dogfood
    # 474fd636 (snag forward-over-rev-188051dc); the primal stays unchanged.
    ast = quote
        a ~ Normal(0, 1)
        shape ~ Exponential(1)
        z[1:6] .~ Normal.(0, 1)
        mu = exp.(a .+ z[idx])
        y .~ NegativeBinomial2.(mu, shape)
    end
    data = (; idx = [1, 3, 2, 1], y = [0, 2, 5, 1])
    saved_data = deepcopy(data)
    bound, built = _cq_bound(ast, data, (:y,))
    @test coordinate_names(built.layout) == [:a; :shape; [Symbol("z.", g) for g in 1:6]]
    base = collect(range(-0.3, 0.4; length = built.layout.total))
    saved_base = copy(base)
    selection(i) = [1, 2, 2 + data.idx[i], 3 + data.idx[i]]
    directions = ([0.2, -0.1, 0.3, 0.7], [-0.4, 0.2, 0.1, -0.5])
    saved_directions = deepcopy(directions)
    q = prepare_cell_query(built, bound, :y)
    single = prepare_compact_cell_ad(prepare_ad_hvp, built, bound, :y,
        base, selection(1), _CCD_FORWARD_REVERSE, (directions[1],))
    batch = prepare_compact_cell_ad(prepare_ad_hvp, built, bound, :y,
        base, selection(1), _CCD_FORWARD_REVERSE, directions)
    _ccd_static() do
        for (i, offset) in ((1, 0.0), (2, 0.15), (3, -0.2), (1, 0.1))
            indices = selection(i)
            active = base[indices] .+ offset
            saved_active = copy(active)
            overlaid = copy(base)
            overlaid[indices] = active
            mu, shape = exp(active[1] + active[3]), exp(active[2])
            @test q(overlaid, i) ≈ logpdf(NegativeBinomial(shape, shape / (shape + mu)), data.y[i]) atol = 1e-12
            expected_g = _ccd_negbin_gradient(active, data.y[i])
            expected_hv = map(directions) do v
                h = 1e-5
                (_ccd_negbin_gradient(active + h * v, data.y[i]) -
                    _ccd_negbin_gradient(active - h * v, data.y[i])) / (2h)
            end
            g, products = ad_gradient_and_hvp(batch, directions, base, active, indices, i)
            @test g ≈ expected_g atol = 1e-11
            for (product, expected) in zip(products, expected_hv)
                @test product ≈ expected rtol = 2e-8 atol = 2e-9
                @test iszero(product[4])
            end
            @test only(ad_hvp(single, (directions[1],), base, active, indices, i)) ≈
                expected_hv[1] rtol = 2e-8 atol = 2e-9
            gbuf, buffers = fill(NaN, 4), (fill(NaN, 4), fill(NaN, 4))
            ad_gradient_and_hvp!(batch, gbuf, buffers, directions, base, active, indices, i)
            @test gbuf ≈ expected_g atol = 1e-11
            @test all(isapprox(got, expected; rtol = 2e-8, atol = 2e-9)
                for (got, expected) in zip(buffers, expected_hv))
            @test active == saved_active
            @test indices == selection(i)
        end
    end
    @test base == saved_base
    @test data == saved_data
    @test directions == saved_directions
end

@testset "compact cell derivatives: a nonlinear grouped cell against differences" begin
    data = _cq_grouped_data()
    bound, built = _cq_bound(_CQ_GROUPED, data, (:y,))
    u = collect(range(-0.4, 0.5; length = built.layout.total))
    names = coordinate_names(built.layout)
    block(i) = findall(n -> startswith(String(n), "Z.$i."), names)
    # The selection: three shared coordinates (a, bw, v0) and the group's block.
    selection(i) = sort!([2, 3, 4, block(i)...])
    m = 5
    directions = ntuple(k -> _ccd_unit(k, m), m)
    v = collect(range(0.3, -0.2; length = m))
    h = prepare_compact_cell_ad(prepare_ad_hvp, built, bound, :y, u, selection(1),
        _CCD_SECOND, directions)
    hf = prepare_compact_cell_ad(prepare_ad_hvp, built, bound, :y, u, selection(1),
        _CCD_FORWARD_REVERSE, directions)
    t = prepare_compact_cell_ad(prepare_ad_third, built, bound, :y, u, selection(1),
        _CCD_THIRD, (v,))
    dense = prepare_cell_sampler(built, bound, :y, u;
        backend = AutoEnzyme(; mode = Enzyme.Reverse))
    cell_gradient(x, i) = (g = similar(x); cell_value_and_gradient!(dense, g, x, i); g)
    base = copy(u)
    step = 1e-5
    _ccd_static() do
        for i in (4, 1, 2)
            indices = selection(i)
            active = u[indices] .+ 0.05
            overlaid = copy(u)
            overlaid[indices] = active
            g, columns = ad_gradient_and_hvp(h, directions, base, active, indices, i)
            H = reduce(hcat, columns)
            @test g ≈ cell_gradient(overlaid, i)[indices]
            @test H ≈ H' atol = 1e-10
            # Each column against central differences of the exact cell gradient.
            differences = reduce(hcat, [begin
                up = copy(overlaid); up[indices[k]] += step
                down = copy(overlaid); down[indices[k]] -= step
                (cell_gradient(up, i) .- cell_gradient(down, i))[indices] ./ (2step)
            end for k in 1:m])
            @test H ≈ differences rtol = 1e-6 atol = 1e-7
            gf, forward_columns = ad_gradient_and_hvp(hf, directions,
                base, active, indices, i)
            @test gf ≈ g
            @test reduce(hcat, forward_columns) ≈ differences rtol = 1e-6 atol = 1e-7
            # ∇(vᵀ H v) against central differences of the exact vᵀ H v.
            quadratic(x) = dot(only(last(ad_gradient_and_hvp(h, (v,), base,
                active .+ x, indices, i))), v)
            expected = [(quadratic(step .* _ccd_unit(k, m)) -
                         quadratic(-step .* _ccd_unit(k, m))) / (2step) for k in 1:m]
            gt, (hv,), (tv,) = ad_gradient_hvp_and_third(t, (v,), base, active, indices, i)
            @test gt ≈ g
            @test hv ≈ H * v
            @test tv ≈ expected rtol = 1e-6 atol = 1e-7
        end
    end
    @test data == _cq_grouped_data()
end

@testset "compact cell derivatives: a cell's cost does not grow with the number of groups" begin
    measured = map((10, 2_000)) do n
        data = _ccd_poisson_data(n)
        bound, built = _cq_bound(_CCD_POISSON, data, (:y,))
        u = collect(range(-0.3, 0.4; length = built.layout.total))
        indices = [1, 2, 2 + 3]
        m = length(indices)
        directions = ntuple(k -> _ccd_unit(k, m), m)
        v = [0.4, -0.7, 1.1]
        h = prepare_compact_cell_ad(prepare_ad_hvp, built, bound, :y, u, indices,
            _CCD_SECOND, directions)
        t = prepare_compact_cell_ad(prepare_ad_third, built, bound, :y, u, indices,
            _CCD_THIRD, (v,))
        active = u[indices]
        g, outs = zeros(m), ntuple(_ -> zeros(m), m)
        hout, tout = (zeros(m),), (zeros(m),)
        _ccd_static() do
            ad_gradient_and_hvp!(h, g, outs, directions, u, active, indices, 3)
            ad_gradient_hvp_and_third!(t, g, hout, tout, (v,), u, active, indices, 3)
            cell = (; x = data.x[3], y = data.y[3])
            _, _, H, third = _ccd_poisson_cell(cell, active..., v, 0)
            (; H = reduce(hcat, outs), expected = H, third = only(tout),
               expected_third = third,
               hvp_bytes = (@allocated ad_gradient_and_hvp!(h, g, outs, directions,
                   u, active, indices, 3)),
               third_bytes = (@allocated ad_gradient_hvp_and_third!(t, g, hout, tout,
                   (v,), u, active, indices, 3)))
        end
    end
    if all(!isnothing, measured)
        @test all(r -> r.H ≈ r.expected, measured)
        @test all(r -> r.third ≈ r.expected_third, measured)
        # The cell's work is cell-local: nested static modes keep no copy of
        # the constant packed base (above).
        @test measured[2].hvp_bytes <= measured[1].hvp_bytes + 1024
        @test measured[2].third_bytes <= measured[1].third_bytes + 1024
    end
end

# `_cq_grouped_data`'s shape at `n` groups of three observations each.
_ccd_grouped_data(n) = (; group = ["g$g" for g in 1:n],
    w = [0.1mod(g, 5) - 0.2 for g in 1:n], amount = fill(10.0, n),
    t = [[0.5, 1.0, 2.0] for _ in 1:n], y = [[1.8, 1.5, 1.1] .+ 0.01mod(g, 7) for g in 1:n])

@testset "compact cell derivatives: a nonlinear grouped cell's cost does not grow with the number of groups" begin
    measured = map((10, 2_000)) do n
        bound, built = _cq_bound(_CQ_GROUPED, _ccd_grouped_data(n), (:y,))
        u = collect(range(-0.4, 0.5; length = built.layout.total))
        names = coordinate_names(built.layout)
        indices = sort!([2, 3, 4, findall(s -> startswith(String(s), "Z.3."), names)...])
        m = length(indices)
        v = collect(range(0.3, -0.2; length = m))
        h = prepare_compact_cell_ad(prepare_ad_hvp, built, bound, :y, u, indices,
            _CCD_SECOND, (v,))
        t = prepare_compact_cell_ad(prepare_ad_third, built, bound, :y, u, indices,
            _CCD_THIRD, (v,))
        active = u[indices] .+ 0.05
        g, hout, tout = zeros(m), (zeros(m),), (zeros(m),)
        _ccd_static() do
            ad_gradient_and_hvp!(h, g, hout, (v,), u, active, indices, 3)
            ad_gradient_hvp_and_third!(t, g, hout, tout, (v,), u, active, indices, 3)
            (; hvp_bytes = (@allocated ad_gradient_and_hvp!(h, g, hout, (v,), u, active,
                   indices, 3)),
               third_bytes = (@allocated ad_gradient_hvp_and_third!(t, g, hout, tout, (v,),
                   u, active, indices, 3)))
        end
    end
    if all(!isnothing, measured)
        # Under `--check-bounds=yes` this cell still copies the packed base once
        # per call (Enzyme todo `0t6ufcp`, above).
        forced_bounds = Base.JLOptions().check_bounds == 1
        @test measured[2].hvp_bytes <= measured[1].hvp_bytes + 1024 broken = forced_bounds
        @test measured[2].third_bytes <= measured[1].third_bytes + 1024 broken = forced_bounds
    end
end
