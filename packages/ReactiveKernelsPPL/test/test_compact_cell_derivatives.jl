using ReactiveKernels, ReactiveKernelsPPL, Distributions, Enzyme, Test, LinearAlgebra
using DifferentiationInterface: AutoEnzyme, SecondOrder

# Hessian-vector products and third-order contractions of ONE cell's log
# density over a compact selection of packed coordinates
# (`prepare_compact_cell_ad`, snag `build-brm-s-spar-454be53f`): the m × m
# Hessian block of a cell over the m coordinates it reads, from one call with
# the m unit directions, and `∇(vᵀ H v)` for a Laplace-marginal gradient.
# Helpers `_CQ_GROUPED`, `_cq_grouped_data`, `_cq_bound` and `_cq_central` come
# from `test_cell_query.jl`.
#
# The backends are plain static Enzyme, reverse over forward (over forward).
# The compact overlay reads either a constant base entry or an active one;
# registered Enzyme 0.13.216 refuses that read under static activity
# (`EnzymeRuntimeActivityError`), and Enzyme `dogfood` repairs it from
# `21a909f7` (Enzyme snag `static-reverse-s-f9483e9c`). On registered Enzyme
# each testset records that gap as broken; drop the branch once a registered
# Enzyme carries the repair.

const _CCD_SECOND = SecondOrder(AutoEnzyme(; mode = Enzyme.Reverse),
                                AutoEnzyme(; mode = Enzyme.Forward))
const _CCD_THIRD = SecondOrder(_CCD_SECOND, AutoEnzyme(; mode = Enzyme.Forward))

# Runs `f()`; returns `nothing` when static Enzyme refuses the compact
# overlay's read (the registered-Enzyme gap above), and rethrows anything else.
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
            hout, tout = (zeros(m),), (zeros(m),)
            ad_gradient_hvp_and_third!(t, gbuf, hout, tout, (v,), base, active, indices, i)
            @test only(tout) ≈ third atol = 1e-12
            @test only(hout) ≈ H * v atol = 1e-12
            @test base == u
            @test indices == selection(i)
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
        # The cell's work is cell-local, but static reverse over forward copies
        # the constant packed base once per call on Enzyme dogfood, where
        # first-order static Reverse and Forward do not (Enzyme snag
        # `static-reverse-o-3d3debe4`). Drop `_broken` when it lands.
        @test_broken measured[2].hvp_bytes <= measured[1].hvp_bytes + 1024
        @test_broken measured[2].third_bytes <= measured[1].third_bytes + 1024
    end
end
