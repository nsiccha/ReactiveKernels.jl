using DifferentiationInterface, Distributions, Enzyme, ReactiveKernels, ReactiveKernelsPPL, Test

# A `@plate` output whose cells are vectors is one vector per cell at model
# scope, as the Julia loop leaves it (snag `summing-a-plate-eba2648d`, BRM
# decision `09e0gc5`): `sum(cells)` adds the cells elementwise, `cells[g]` is
# cell `g`'s vector and `cells[k]` the cells `k`. That holds whether the cell
# reads one array per index (`A[i]`) or builds its vector from scalar reads
# (`[b * a1[i], b * a2[i]]`), and a value computed from such reads is a
# location like any other vector.

const _VPO_A = [[1.0, 2.0], [3.0, -1.0], [0.5, 0.25], [-2.0, 7.0]]
const _VPO_A1 = first.(_VPO_A)
const _VPO_A2 = last.(_VPO_A)
const _VPO_K = [4, 1]
const _VPO_OBS = [4.0, 1.5]

_vpo_flatten(c) = eltype(c) <: Number ? c :
    isempty(c) ? eltype(eltype(c))[] : reduce(vcat, c)
_vpo_cells(b) = [b .* x for x in _VPO_A]
_vpo_query(built, bound, name, u) = Base.invokelatest(ReactiveKernels.prepare(built.spec;
    have = built.spec.have_names, want = (name,),
    bound = NamedTuple(k => v for (k, v) in bound.columns)), u)

# The density, its native Enzyme Reverse gradient and the named values, at
# two points, against the Julia loop's cells. Every location is `b` times a
# data vector, so its derivative is the location at `b = 1`.
function _vpo_check(bound, location; values = ())
    built = build_kernel(bound)
    sampler = prepare_sampler(built, bound, [0.3];
        backend = AutoEnzyme(; mode = Enzyme.Reverse))
    for b in (0.3, -1.2)
        n = location(_vpo_cells(b))
        dn = location(_vpo_cells(1.0))
        expected = logpdf(Normal(), b) + sum(logpdf.(Normal.(n, 1.0), _VPO_OBS))
        @test sampler([b]) ≈ expected rtol = 1e-12
        g = zeros(1)
        value, _ = sampler_value_and_gradient!(sampler, g, [b])
        @test value ≈ expected rtol = 1e-12
        @test g[1] ≈ -b + sum((_VPO_OBS .- n) .* dn) rtol = 1e-10
        for (name, f) in values
            @test _vpo_query(built, bound, name, [b]) ≈ f(_vpo_cells(b))
        end
    end
end

const _VPO_EMITTED = @rkppl begin
    b ~ Normal(0.0, 1.0)
    @plate for i = eachindex(A)
        cnt_cells[i] = [b * A[i][1], b * A[i][2]]
    end
    cnt = _vpo_flatten(cnt_cells)
    n = sum(cnt_cells)
    obs .~ Normal.(n, 1.0)
end

const _VPO_POSITIONS = @rkppl begin
    b ~ Normal(0.0, 1.0)
    @plate for i = eachindex(A)
        cnt_cells[i] = b .* A[i]
    end
    n = .+(cnt_cells[1], cnt_cells[2], cnt_cells[4])
    obs .~ Normal.(n, 1.0)
end

const _VPO_LITERAL = @rkppl begin
    b ~ Normal(0.0, 1.0)
    @plate for i = eachindex(a1)
        cnt_cells[i] = [b * a1[i], b * a2[i]]
    end
    n = sum(cnt_cells)
    obs .~ Normal.(n, 1.0)
end

const _VPO_SELECTED = @rkppl begin
    b ~ Normal(0.0, 1.0)
    @plate for i = eachindex(a1)
        cnt_cells[i] = [b * a1[i], b * a2[i]]
    end
    picked = sum(cnt_cells[k])
    third = cnt_cells[3]
    obs .~ Normal.(picked .+ third, 1.0)
end

@testset "a vector-valued @plate output read at model scope" begin
    # The program BRM emits for `cnt[s] = [b * A[s][1], b * A[s][2]]`, then
    # `n = sum(cnt)` (its retained plate's fused total).
    _vpo_check(_VPO_EMITTED(; A = _VPO_A) | (; obs = _VPO_OBS), sum;
        values = (:cnt_cells => identity, :n => sum, :cnt => c -> reduce(vcat, c)))
    # Positions read whole cells, never entries of the flat value.
    _vpo_check(_VPO_POSITIONS(; A = _VPO_A) | (; obs = _VPO_OBS),
        c -> c[1] .+ c[2] .+ c[4]; values = (:n => c -> c[1] .+ c[2] .+ c[4],))
    # A vector literal over scalar reads is one vector per cell, also from a
    # plan lowered from data names alone.
    data = (; a1 = _VPO_A1, a2 = _VPO_A2)
    _vpo_check(_VPO_LITERAL(; data...) | (; obs = _VPO_OBS), sum;
        values = (:cnt_cells => identity, :n => sum))
    names_only = lower_rkppl(_VPO_LITERAL.ast, (:a1, :a2, :obs);
        conditioned = (:obs,), mod = @__MODULE__)
    _vpo_check(bind_data(names_only, (; data..., obs = _VPO_OBS)), sum)
    # A data index selects cells, checked against the cells (4), not against
    # each cell's length (2), as the inline location the reporter wrote.
    _vpo_check(_VPO_SELECTED(; data..., k = _VPO_K) | (; obs = _VPO_OBS),
        c -> sum(c[_VPO_K]) .+ c[3];
        values = (:picked => c -> sum(c[_VPO_K]), :third => c -> c[3]))
    # refused: as Julia's indexing, a position outside the cells.
    outside = _VPO_SELECTED(; data..., k = [5, 1]) | (; obs = _VPO_OBS)
    @test_throws BoundsError Base.invokelatest(
        prepare_query(build_kernel(outside), outside, :sampler), [0.3])
    # refused: as `sum(Vector{Float64}[])`, a sum of no vectors has no zero.
    empty = _VPO_LITERAL(; a1 = Float64[], a2 = Float64[]) | (; obs = _VPO_OBS)
    @test_throws "zero(::Type{Vector{Float64}})" Base.invokelatest(
        prepare_query(build_kernel(empty), empty, :sampler), [0.3])
end

@testset "array literals in a @plate cell keep one array per index" begin
    # Each literal spelling builds the same two-entry vector per cell.
    for cell in (:([b * a1[i], b * a2[i]]), :([b * a1[i]; b * a2[i]]),
                 :([b * x for x in (a1[i], a2[i])]),
                 :(Base.vect(b * a1[i], b * a2[i])))
        ast = quote
            b ~ Normal(0.0, 1.0)
            @plate for i = eachindex(a1)
                cnt_cells[i] = $cell
            end
            n = sum(cnt_cells)
            obs .~ Normal.(n, 1.0)
        end
        data = (; a1 = _VPO_A1, a2 = _VPO_A2, obs = _VPO_OBS)
        bound = bind_data(lower_rkppl(ast, data; conditioned = (:obs,),
            mod = @__MODULE__), data)
        _vpo_check(bound, sum; values = (:cnt_cells => identity,))
    end
end
