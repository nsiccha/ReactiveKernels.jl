module JAXRaggedLaneTests
using ReactiveKernels, Test, Enzyme
include("jax_python.jl")

@kernel weighted_rows(w::Float64, x::Vector{Vector{Float64}},
                      mode::Vector{Vector{Float64}}) = begin
    totals = plate(x, mode) do row, flags
        values = plate(row, flags, w) do v, m, weight
            if m > 0 && v <= log(m)
                m == 1 ? exp(weight * v) : exp(-weight * v)
            else
                weight + v
            end
        end
        sum(values)
    end
    score = sum(totals)
end

@testset "prepared decisions on ragged rows retain the native broadcast domain" begin
    have = (:w, :x, :mode)
    emit(d) = jax_source(weighted_rows; have, data = d, functions = (score = :score,))
    mixed = (; x = [[2., -1.], Float64[], [-0.5]], mode = [[-1., 1.], Float64[], [2.]])
    reference = (; x = repeat(mixed.x, 2), mode = repeat(mixed.mode, 2))
    larger = (; x = repeat(reference.x, 7), mode = repeat(reference.mode, 7))
    src = emit(mixed)
    @test src.text == emit(larger).text
    @test occursin(" = _rk_ragged_lane_sum(", src.text)
    # Prepared lengths are proven once by `prepare_data`; the function runs
    # the checks and the layout branch only for data that fail the proof.
    @test occursin("= _rk_ragged_counts(", src.text)
    @test occursin(r"\n\s+if not _rk_ragged_counts_\w+:\n", src.text)
    live = emit((:mode,))
    @test occursin(" = _rk_ragged_lane_sum(", live.text)
    # Rows passed when the function runs cannot be proven beforehand.
    @test !occursin("_rk_ragged_counts", live.text)
    # A location's row lengths read through a prepared one-based row index,
    # as a plate passes them out (`lengths[index - 1]`), are proven from
    # both prepared arrays; anything else that is not prepared is not.
    let RK = ReactiveKernels
        ctx = RK._JaxContext(RK._JaxModule(), RK._JaxScope(), nothing, Main, Dict{Symbol,RK._PyVal}(), nothing)
        merge!(ctx.scope.prepared_keys, Dict("bound" => "obs_row_lengths", "loc_lengths" => "loc_rows_lengths",
                                             "which" => "obs_subject"))
        proof = RK._jax_ragged_counts_proof(ctx, ["bound", "loc_lengths[which - 1]"])
        @test (proof, :ragged_counts, ("obs_row_lengths", "loc_rows_lengths", "obs_subject")) in
              RK._jax_prepared_bounds(ctx.mod)
        @test RK._jax_ragged_counts_proof(ctx, ["bound", "loc_lengths[other - 1]"]) === nothing
        @test RK._jax_ragged_counts_proof(ctx, ["bound", "jnp.full((m.shape[0],), m.shape[1])"]) === nothing
    end
    native = prepare(weighted_rows; have, want = :score)
    if rk_jax_available()
        cases = [reference, larger, mixed, (; x=mixed.x, mode=[[2., -1.], Float64[], [1.]]),
                 (; x=[[1.], [-2.]], mode=[[-1., 1.], [2., 1., -1.]]),
                 (; x=[[1., 2.], [3.]], mode=[[-1.]]),
                 (; x=[Float64[]], mode=[[1.]]), (; x=[[1.]], mode=[Float64[]]),
                 (; x=Vector{Float64}[], mode=Vector{Float64}[])]
        w = 0.4
        @test_throws DimensionMismatch native(w, [[1., 2.]], [[1., 1., 1.]])
        expected = [(native(w, d.x, d.mode),
            Enzyme.autodiff(Enzyme.Reverse, native, Enzyme.Active, Enzyme.Active(w),
                            Enzyme.Const(d.x), Enzyme.Const(d.mode))[1][1]) for d in cases]
        mktempdir() do dir
            write(joinpath(dir, "fixed.py"), src)
            write(joinpath(dir, "live.py"), live)
            arguments = [_rk_py_input((d.x, d.mode, e...)) for (d,e) in zip(cases, expected)]
            program = "import numpy as np\ncases = [" * join(arguments, ", ") * "]\n"
            write(joinpath(dir, "check.py"), program * raw"""
import fixed, live, jax, numpy as np
import jax.numpy as jnp
from collections import Counter

def inventory(jaxpr, exp_widths=None):
    result = Counter()
    def count(item):
        if hasattr(item, "eqns"):
            for eq in item.eqns:
                result[eq.primitive.name] += 1
                if exp_widths is not None and eq.primitive.name == "exp":
                    exp_widths.append(np.prod(eq.outvars[0].aval.shape, dtype=int))
                count(eq.params)
        elif hasattr(item, "jaxpr"):
            count(item.jaxpr)
        elif isinstance(item, dict):
            for v in item.values(): count(v)
        elif isinstance(item, (list, tuple)):
            for v in item: count(v)
    count(jaxpr)
    return result

# A live guard selects a logarithm or a square root. Exercise both uniform
# arms and a mixed group, including outer chains whose choices disagree.
def guarded(x, center):
    return jax.lax.cond(x > center, lambda: jnp.log(x - center),
                        lambda: jnp.sqrt(center - x))

def grouped(xs, center):
    return live._rk_ragged_group(guarded, (xs, jnp.broadcast_to(center, xs.shape))).sum()

def scalar(xs, center):
    return jax.lax.map(lambda x: guarded(x, center), xs).sum()

def density_and_derivatives(f):
    return jax.jit(jax.vmap(jax.value_and_grad(f, (0, 1)), in_axes=(0, None)))

parallel_vg, scalar_vg = map(density_and_derivatives, (grouped, scalar))
positive = np.array([[1.5, 2., 3.], [4., 1.25, 2.25]])
for xs in (positive, -positive, positive * [[1.], [-1.]],
           positive * [[1., -1., 1.], [-1., 1., -1.]]):
    for a, b in zip(jax.tree.leaves(parallel_vg(xs, 0.5)), jax.tree.leaves(scalar_vg(xs, 0.5))):
        assert np.isfinite(a).all()
        np.testing.assert_allclose(a, b, rtol=2e-12, atol=2e-12)

# Both sides of the live guards occur in one prepared group. The log and
# square root have singular or undefined derivatives on the opposite side.
@jax.custom_jvp
def root(x):
    return jnp.sqrt(x)

@root.defjvp
def root_jvp(primals, tangents):
    y = root(primals[0])
    return y, tangents[0] / (2 * y)

def nested(x):
    return jax.lax.cond(x > 0,
        lambda: jax.lax.cond(x > 1, lambda: jnp.log(x - 1), lambda: root(1 - x)),
        lambda: root(-x))

def nested_sum(xs):
    return live._rk_ragged_group(nested, (xs,)).sum()

traces = []
for repeats in (1, 7):
    xs = np.tile([0.5, -4., 3.], repeats)
    value, grad = jax.jit(jax.value_and_grad(nested_sum))(xs)
    np.testing.assert_allclose(value, repeats * (np.sqrt(0.5) + 2 + np.log(2)))
    np.testing.assert_allclose(grad, np.tile([-0.5 / np.sqrt(0.5), -0.25, 0.5], repeats))
    traces.append(inventory(jax.make_jaxpr(jax.value_and_grad(nested_sum))(xs)))
for trace in traces:
    assert trace["scan"] == 0 and trace["sqrt"] > 0 and trace["log"] > 0, trace
for op in ("sqrt", "log"):
    assert traces[1][op] <= traces[0][op], (op, traces)

# The primal alone cannot establish that the derivative is safe to batch.
@jax.custom_jvp
def checked_root(x):
    return jnp.sqrt(x)

from jax.experimental import checkify

@checked_root.defjvp
def checked_root_jvp(primals, tangents):
    x, = primals
    checkify.check(x > 0, "root derivative requires positive input")
    y = checked_root(x)
    return y, tangents[0] / (2 * y)

def checked_sum(xs, cutoff):
    def cell(x):
        return jax.lax.cond(x > cutoff, checked_root, jnp.square, x)
    return live._rk_ragged_group(cell, (xs,)).sum()

checked = jax.jit(checkify.checkify(jax.value_and_grad(checked_sum)))
xs = np.array([-1., 4.])
err, (value, grad) = checked(xs, 0.)
err.throw()
np.testing.assert_allclose((value, *grad), (3., -2., 0.25))
err, _ = checked(xs, -2.)
assert "root derivative requires positive input" in err.get()

for module in (fixed, live):
    vg = jax.value_and_grad(module.score)
    compiled = jax.jit(vg)
    traces = []
    widths = []
    for rows, modes, value, gradient in cases:
        if module is live:
            padded, lengths, flat = live._rk_pad_rows(rows, np.float64)
            args = dict(x=padded, x_lengths=lengths, x_flat=flat, **live.prepare_data(modes))
        else:
            args = fixed.prepare_data(x=rows, mode=modes)
        got = compiled(0.4, **args)
        np.testing.assert_allclose(got, (value, gradient), rtol=2e-12, atol=2e-12)
        if len(traces) < 2:
            exp_widths = []
            traces.append(inventory(jax.make_jaxpr(vg)(0.4, **args), exp_widths))
            widths.append(max(exp_widths))
    # Repeating the rows grows the arrays carrying the cell's arithmetic.
    assert widths[1] == 7 * widths[0], widths
    for body in ("exp", "log", "cond", "scan"):
        assert traces[1][body] == traces[0][body], (body, traces)
    rows, modes = [[1., 2.]], [[1., 1., 1.]]
    if module is live:
        padded, lengths, flat = live._rk_pad_rows(rows, np.float64)
        args = dict(x=padded, x_lengths=lengths, x_flat=flat, **live.prepare_data(modes))
    else:
        args = fixed.prepare_data(x=rows, mode=modes)
    try:
        jax.block_until_ready(compiled(0.4, **args))
    except Exception as err:
        assert "incompatible ragged plate axes" in str(err), str(err)
    else:
        raise AssertionError("accepted incompatible row lengths")

# On a GPU every evaluated check and data-dependent layout branch copies its
# predicate to the host. Proven prepared lengths need neither; lengths that
# fail the proof keep both, so the error above still comes from the function.
def checks_and_branches(f, args):
    found = inventory(jax.make_jaxpr(f)(0.4, **args))
    return found["debug_callback"], found["cond"]
proofs = [k for k in fixed.PREPARED if k.startswith("_rk_ragged_counts")]
assert len(proofs) == 1, list(fixed.PREPARED)
vg = jax.value_and_grad(fixed.score)
for rows, modes, _, _ in cases[:3]:
    good = fixed.prepare_data(x=rows, mode=modes)
    assert good[proofs[0]] == 1, (rows, modes, good[proofs[0]])
    assert checks_and_branches(vg, good)[0] == 0, checks_and_branches(vg, good)
bad = fixed.prepare_data(x=[[1., 2.]], mode=[[1., 1., 1.]])
assert bad[proofs[0]] == 0, bad[proofs[0]]
callbacks, branches = checks_and_branches(vg, bad)
assert callbacks > 0, callbacks
assert branches > checks_and_branches(vg, good)[1], (branches, checks_and_branches(vg, good))

# The proof over a location's selected rows: a valid selection whose lengths
# match the data's (or have length one) proves; an out-of-range selector,
# incompatible rows, a negative length or a different row count do not.
counts = fixed._rk_ragged_counts
lengths, subject = np.array([2, 1, 0, 5]), np.array([1, 2, 3])
assert counts(np.array([2, 1, 0]), lengths, subject) == 1
assert counts(np.array([2, 1, 0]), lengths, None) == 0
assert counts(np.array([2, 1, 0]), lengths, np.array([1, 2, 5])) == 0
assert counts(np.array([2, 1, 0]), lengths, np.array([0, 2, 3])) == 0
assert counts(np.array([3, 1, 0]), lengths, subject) == 0
# A location row of length one broadcasts over its data row, either way round.
assert counts(np.array([2, 2, 0]), lengths, subject) == 1
assert counts(np.array([2, 1, 0]), np.array([1, 1, 1]), subject) == 1
assert counts(np.array([1, 1, 0]), np.array([2, 1, 0]), subject) == 0
assert counts(np.array([-1, 1, 0]), np.array([1, 1, 0]), subject) == 0
assert counts(np.array([2, 1, 0])) == 1 and counts(np.array([2, -1, 0])) == 0
""")
            @test success(pipeline(`$RK_JAX_PYTHON $(joinpath(dir, "check.py"))`, stderr=stderr))
        end
    else
        @test_skip "RK_JAX_PYTHON not set"
    end
end

# A location row another plate computed. The subject plate returns two rows
# per subject, the second the broadcast of two shared rows read at the
# subject's index; a plate projects that row and a third reads a subject's
# row for each data row. Its lengths are data, though no prepared array holds
# them.
@kernel located_rows(w::Float64, t::Vector{Vector{Float64}}, base::Vector{Vector{Float64}},
                     subject::Vector{Int}, which::Vector{Int},
                     y::Vector{Vector{Float64}}) = begin
    cells = plate(subject, Ref(t), Ref(base), w) do s, ts, bs, weight
        (bs[s] .* weight, weight .* ts[s] .+ bs[s])
    end
    locs = plate(subject, cells) do s, cell
        cell[2]
    end
    selected = plate(which, Ref(locs)) do i, rows
        rows[i]
    end
    totals = plate(y, selected) do obs, loc
        values = plate(obs, loc, w) do v, m, weight
            v > 0 ? exp(-weight * (v - m)^2) : weight + m
        end
        sum(values)
    end
    score = sum(totals)
end

# The computed rows read whole, one per data row.
@kernel located_whole(w::Float64, t::Vector{Vector{Float64}}, base::Vector{Vector{Float64}},
                      subject::Vector{Int}, y::Vector{Vector{Float64}}) = begin
    locs = plate(subject, Ref(t), Ref(base), w) do s, ts, bs, weight
        weight .* ts[s] .+ bs[s]
    end
    totals = plate(y, locs) do obs, loc
        values = plate(obs, loc, w) do v, m, weight
            v > 0 ? exp(-weight * (v - m)^2) : weight + m
        end
        sum(values)
    end
    score = sum(totals)
end

@testset "prepare_data proves the lengths of rows another plate computed" begin
    have = (:w, :t, :base, :subject, :which, :y)
    emit(d) = jax_source(located_rows; have, data = d, functions = (score = :score,))
    # Subject rows of lengths 3, 2 (a singleton `t` row under two `base`
    # entries), 2, 1 and 0.
    t = [[1., 2., 0.5], [0.5], [3., 1.], [2.], Float64[]]
    base = [[0.1, 0.2, 0.3], [0.3, 0.1], [0., 0.], [0.5], Float64[]]
    subject = collect(1:5)
    reference = (; t, base, subject, which = [1, 3, 3, 2],
                 y = [[1., -1., 2.], [0.5, 0.25], [-2., 1.], [4., -0.5]])
    larger = (; t, base, subject, which = repeat(reference.which, 7), y = repeat(reference.y, 7))
    src = emit(reference)
    @test src.text == emit(larger).text
    @test occursin(" = _rk_ragged_lane_sum(", src.text)
    # The sum reads the selected rows' lengths as the projected field's,
    # which the subject plate computes.
    @test occursin(r"jnp\.broadcast_arrays\(\w+, plate_lengths\w*\[1\]\[which - 1\]\)", src.text)
    # `prepare_data` builds those rows' layout from the prepared rows they
    # broadcast and proves the sum's lengths from it.
    @test occursin(r"= _rk_prepare_row_layout\(_rk_prepared\[\"base_lengths\"\], _rk_prepared\[\"subject\"\], _rk_prepared\[\"t_lengths\"\], _rk_prepared\[\"subject\"\]\)", src.text)
    @test occursin(r"= _rk_ragged_counts\(_rk_prepared\[\"\w+\"\], _rk_prepared\[\"_rk_row_layout_\w+\"\], _rk_prepared\[\"which\"\]\)", src.text)
    @test occursin(r"\n\s+if not _rk_ragged_counts_\w+:\n", src.text)
    # A module whose caller chooses the plates' layout proves them the same way.
    grouped = jax_source(located_rows; have, data = reference, functions = (score = :score,),
                         plate_layout = :groups)
    @test occursin(r"= _rk_ragged_counts\(_rk_prepared\[\"\w+\"\], _rk_prepared\[\"_rk_row_layout_\w+\"\], _rk_prepared\[\"which\"\]\)", grouped.text)
    @test occursin(r"\n\s+if not _rk_ragged_counts_\w+:\n", grouped.text)
    whole_have = (:w, :t, :base, :subject, :y)
    whole_data = (; t, base, subject = [1, 3, 3, 2], y = reference.y)
    whole = jax_source(located_whole; have = whole_have, data = whole_data, functions = (score = :score,))
    # Rows read whole are proven from the prepared rows themselves.
    @test occursin(r"= _rk_ragged_counts\(_rk_prepared\[\"\w+\"\], _rk_prepared\[\"base_lengths\"\], _rk_prepared\[\"subject\"\], _rk_prepared\[\"t_lengths\"\], _rk_prepared\[\"subject\"\]\)", whole.text)
    @test !occursin("_rk_prepare_row_layout", whole.text)
    @test occursin(r"\n\s+if not _rk_ragged_counts_\w+:\n", whole.text)
    let RK = ReactiveKernels
        ctx = RK._JaxContext(RK._JaxModule(), RK._JaxScope(), nothing, Main, Dict{Symbol,RK._PyVal}(), nothing)
        merge!(ctx.scope.prepared_keys, Dict("bound" => "obs_row_lengths", "which" => "obs_subject"))
        sources = ["base_lengths", "subject", "t_lengths", "subject"]
        ctx.mod.helper_names[(:row_sources, ctx.scope, "plate_lengths[1]")] = sources
        proof = RK._jax_ragged_counts_proof(ctx, ["bound", "plate_lengths[1][which - 1]"])
        bounds = RK._jax_prepared_bounds(ctx.mod)
        layout = only(name for (name, kind, _) in bounds if kind === :row_layout)
        @test (layout, :row_layout, Tuple(sources)) in bounds
        @test (proof, :ragged_counts, ("obs_row_lengths", layout, "obs_subject")) in bounds
        # Read whole, the rows' own prepared sources are the operands.
        whole_proof = RK._jax_ragged_counts_proof(ctx, ["bound", "plate_lengths[1]"])
        @test (whole_proof, :ragged_counts, ("obs_row_lengths", sources...)) in RK._jax_prepared_bounds(ctx.mod)
        # A selector that is not prepared, and rows no plate recorded, are not proven.
        @test RK._jax_ragged_counts_proof(ctx, ["bound", "plate_lengths[1][other - 1]"]) === nothing
        @test RK._jax_ragged_counts_proof(ctx, ["bound", "plate_lengths[0][which - 1]"]) === nothing
        @test RK._jax_ragged_counts_proof(ctx, ["bound", "plate_lengths[0]"]) === nothing
    end
    native = prepare(located_rows; have, want = :score)
    native_whole = prepare(located_whole; have = whole_have, want = :score)
    if rk_jax_available()
        w = 0.4
        cases = [reference, larger,
                 # A computed row of length one under a longer data row, an
                 # empty computed row under an empty data row, one computed
                 # row under every data row, and no rows.
                 (; t, base, subject, which = [4, 5, 2], y = [[1., -2., 3.], Float64[], [0.5, -0.5]]),
                 (; t, base, subject, which = [1], y = [[1., 2., 3.], [-1., 2., 0.5]]),
                 (; t, base, subject, which = Int[], y = Vector{Float64}[])]
        @test_throws DimensionMismatch native(w, t, base, subject, [1], [[1., 2.]])
        slope(f, args...) = Enzyme.autodiff(Enzyme.Reverse, f, Enzyme.Active, Enzyme.Active(w),
                                            map(Enzyme.Const, args)...)[1][1]
        expected = [(native(w, d.t, d.base, d.subject, d.which, d.y),
                     slope(native, d.t, d.base, d.subject, d.which, d.y)) for d in cases]
        whole_expected = (native_whole(w, values(whole_data)...), slope(native_whole, values(whole_data)...))
        mktempdir() do dir
            write(joinpath(dir, "located.py"), src)
            write(joinpath(dir, "whole.py"), whole)
            write(joinpath(dir, "grouped.py"), grouped)
            arguments = [_rk_py_input((d.t, d.base, d.subject, d.which, d.y, e...)) for (d, e) in zip(cases, expected)]
            program = "import numpy as np\ncases = [" * join(arguments, ", ") * "]\n" *
                      "whole_case = " * _rk_py_input((values(whole_data)..., whole_expected...)) * "\n"
            write(joinpath(dir, "check.py"), program * raw"""
import located, whole, grouped, jax
from collections import Counter

def inventory(jaxpr):
    result = Counter()
    def count(item):
        if hasattr(item, "eqns"):
            for eq in item.eqns:
                result[eq.primitive.name] += 1
                count(eq.params)
        elif hasattr(item, "jaxpr"):
            count(item.jaxpr)
        elif isinstance(item, dict):
            for v in item.values(): count(v)
        elif isinstance(item, (list, tuple)):
            for v in item: count(v)
    count(jaxpr)
    return result

def checks_and_branches(f, args):
    found = inventory(jax.make_jaxpr(f)(0.4, **args))
    return found["debug_callback"], found["cond"]

proofs = [k for k in located.PREPARED if k.startswith("_rk_ragged_counts")]
layouts = [k for k in located.PREPARED if k.startswith("_rk_row_layout")]
assert len(proofs) == 1 and len(layouts) == 1, list(located.PREPARED)
vg = jax.value_and_grad(located.score)
compiled = jax.jit(vg)
traces = []
for t, base, subject, which, y, value, gradient in cases:
    args = located.prepare_data(t=t, base=base, subject=subject, which=which, y=y)
    # Proven lengths run the grouped sum with no check and no layout branch.
    assert args[proofs[0]] == 1, (which, args[proofs[0]])
    assert checks_and_branches(vg, args)[0] == 0, checks_and_branches(vg, args)
    np.testing.assert_allclose(compiled(0.4, **args), (value, gradient), rtol=2e-12, atol=2e-12)
    traces.append(inventory(jax.make_jaxpr(vg)(0.4, **args)))
# Seven times the data rows repeat no body or control-flow region.
for body in ("exp", "cond", "scan", "while"):
    assert traces[1][body] == traces[0][body], (body, traces[0][body], traces[1][body])

t, base, subject = cases[0][:3]
good = located.prepare_data(t=t, base=base, subject=subject, which=cases[0][3], y=cases[0][4])
def unproven(**changed):
    args = located.prepare_data(**{**dict(t=t, base=base, subject=subject, which=[1], y=[[1., 2., 3.]]), **changed})
    assert args[proofs[0]] == 0, (changed, args[proofs[0]])
    return args
# A data row of two entries under a computed row of three keeps the check,
# which fails when the function runs, and the layout branch.
bad = unproven(y=[[1., 2.]])
callbacks, branches = checks_and_branches(vg, bad)
assert callbacks > 0 and branches > checks_and_branches(vg, good)[1], (callbacks, branches)
try:
    jax.block_until_ready(compiled(0.4, **bad))
except Exception as err:
    assert "incompatible ragged plate axes" in str(err), str(err)
else:
    raise AssertionError("accepted incompatible row lengths")
# So does a data row of one entry, which the computed row would widen, and
# a number of rows that does not broadcast.
unproven(y=[[1.]])
unproven(which=[1, 1, 1], y=[[1., 2., 3.], [1., 2., 3.]])
# Rows the subject plate cannot broadcast have no layout, and a row index
# outside the subjects selects none.
assert unproven(t=[[1., 2.]] + t[1:])[layouts[0]] is None
unproven(which=[6])
unproven(which=[0])

# The proof reads a plate's computed rows as their prepared layout.
counts = located._rk_ragged_counts
layout = located._rk_prepare_row_layout(np.array([3, 2, 2]), np.array([1, 2, 3]),
                                        np.array([3, 1, 2]), np.array([1, 2, 3]))
np.testing.assert_array_equal(layout[0], [3, 2, 2])
assert counts(np.array([2, 3, 2]), layout, np.array([2, 1, 3])) == 1
assert counts(np.array([2, 3, 2]), layout, np.array([2, 1, 4])) == 0
assert counts(np.array([3, 2, 2]), layout, None) == 1
assert counts(np.array([3, 2, 2]), None, None) == 0

# Plates run in the caller's groups read the same verdict.
grouped_proofs = [k for k in grouped.PREPARED if k.startswith("_rk_ragged_counts")]
assert len(grouped_proofs) == 1, list(grouped.PREPARED)
grouped_vg = jax.jit(jax.value_and_grad(grouped.score))
for t, base, subject, which, y, value, gradient in cases:
    prepared = grouped.prepare_data(t=t, base=base, subject=subject, which=which, y=y)
    assert prepared[grouped_proofs[0]] == 1, (which, prepared[grouped_proofs[0]])
    for groups in (1, 2, 3):
        args = grouped.prepare_plate_layout(prepared, groups)
        np.testing.assert_allclose(grouped_vg(0.4, **args), (value, gradient), rtol=2e-12, atol=2e-12)

*data, value, gradient = whole_case
args = whole.prepare_data(**dict(zip(("t", "base", "subject", "y"), data)))
proofs = [k for k in whole.PREPARED if k.startswith("_rk_ragged_counts")]
assert len(proofs) == 1 and args[proofs[0]] == 1, (list(whole.PREPARED), args.get(proofs[0]))
vg = jax.value_and_grad(whole.score)
assert checks_and_branches(vg, args)[0] == 0, checks_and_branches(vg, args)
np.testing.assert_allclose(jax.jit(vg)(0.4, **args), (value, gradient), rtol=2e-12, atol=2e-12)
""")
            @test success(pipeline(`$RK_JAX_PYTHON $(joinpath(dir, "check.py"))`, stderr=stderr))
        end
    else
        @test_skip "RK_JAX_PYTHON not set"
    end
end
end
