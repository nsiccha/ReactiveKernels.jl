using ReactiveKernels, Test, Enzyme
include(joinpath(@__DIR__, "jax_python.jl"))

@traceable function jax_group_rows(a::Float64, labels::Vector{Int}, rates::Vector{Float64})
    k = isempty(labels) ? 0 : maximum(labels)
    rows = Vector{Vector{Float64}}(undef, k)
    for i in eachindex(labels)
        t = labels[i]
        isassigned(rows, t) && continue
        rate = exp(rates[i])
        rows[t] = [a, rate, 1 + a * rate]
    end
    for t in 1:k
        isassigned(rows, t) || (rows[t] = zeros(3))
    end
    rows
end

@traceable function jax_rows_total(rows::Vector{Vector{Float64}})
    total = 0.0
    for t in eachindex(rows)
        total += sum(abs2, rows[t])
    end
    total
end

@kernel jax_slots_graph(a::Float64, labels::Vector{Int}, rates::Vector{Float64}) = begin
    rows = jax_group_rows(a, labels, rates)
    score = jax_rows_total(rows)
    parts = plate(rows) do row
        sum(abs2, row)
    end
    mapped = sum(parts)
    return (score, mapped)
end

@kernel jax_slots_ragged(a::Float64, labelsets::Vector{Vector{Int}}, ratesets::Vector{Vector{Float64}}) = begin
    parts = plate(labelsets, ratesets) do labels, rates
        rows = jax_group_rows(a, labels, rates)
        jax_rows_total(rows)
    end
    score = sum(parts)
    return score
end

# The integer labels are a data-only recipe inside the cell. Their bound
# must be prepared before a parameter-dependent helper allocates its rows.
@kernel jax_slots_derived(a::Float64, rawsets::Vector{Vector{Float64}}, ratesets::Vector{Vector{Float64}}) = begin
    parts = plate(rawsets, ratesets) do raw, rates
        labels = Int.(raw)
        rows = jax_group_rows(a, labels, rates)
        jax_rows_total(rows)
    end
    score = sum(parts)
    return score
end

@kernel jax_slots_ragged_plate(a::Float64, labelsets::Vector{Vector{Int}}, ratesets::Vector{Vector{Float64}}) = begin
    parts = plate(labelsets, ratesets) do labels, rates
        rows = jax_group_rows(a, labels, rates)
        terms = plate(rows) do row
            a * log(row[2])
        end
        sum(terms)
    end
    score = sum(parts)
    return score
end

# Named records retain the buffer's padded axis independently of field shapes.
@kernel jax_slots_record_plate(a::Float64, labelsets::Vector{Vector{Int}}, ratesets::Vector{Vector{Float64}}) = begin
    parts = plate(labelsets, ratesets) do labels, rates
        rows = jax_group_rows(a, labels, rates)
        records = plate(rows) do row
            (; score = a * log(row[2]))
        end
        terms = plate(records) do record
            record.score
        end
        sum(terms)
    end
    score = sum(parts)
    return score
end

@traceable function jax_partial_rows(a::Float64, i::Int, guarded::Bool)
    rows = Vector{Vector{Float64}}(undef, 2)
    alias = rows
    if a > 0
        alias[1] = [a, a * a]
    else
        rows[1] = [-a, a * a]
    end
    guarded && !isassigned(rows, i) && return 0.0
    sum(rows[i])
end
@kernel jax_partial_graph(a::Float64, i::Int, guarded::Bool) = begin
    score = jax_partial_rows(a, i, guarded)
    return score
end

jax_slot_loss(a, labels, rates) = jax_rows_total(jax_group_rows(a, labels, rates))
function jax_slots_log_loss(a, labelsets, ratesets)
    total = 0.0
    for (labels, rates) in zip(labelsets, ratesets)
        for row in jax_group_rows(a, labels, rates)
            total += a * log(row[2])
        end
    end
    total
end

@traceable function jax_stored_row_mutation(a::Float64)
    rows = Vector{Vector{Float64}}(undef, 1)
    row = [a, a]
    rows[1] = row
    row[1] = 2a
    sum(rows[1])
end
@traceable function jax_write_borrowed(rows::Vector{Vector{Float64}}, a::Float64)
    rows[1] = [a, a]
    sum(rows[1])
end
@traceable function jax_stored_row_view_mutation(a::Float64)
    rows = Vector{Vector{Float64}}(undef, 1)
    rows[1] = [a, a]
    row = view(rows[1], :)
    row[1] = 2a
    sum(rows[1])
end
@traceable function jax_stored_row_while_mutation(a::Float64)
    rows = Vector{Vector{Float64}}(undef, 1)
    row = [a, a]
    rows[1] = row
    i = 0
    while i < 2
        i += 1
        i == 1 && continue
        row[1] = 2a
    end
    sum(rows[1])
end
@traceable function jax_borrowed_mutation(a::Float64)
    rows = Vector{Vector{Float64}}(undef, 1)
    rows[1] = [a, a]
    jax_write_borrowed(rows, a)
end
@traceable function jax_write_row_argument(row::Vector{Float64}, a::Float64)
    row[1] = 2a
    nothing
end
@traceable function jax_borrowed_row_mutation(a::Float64)
    row = [a, a]
    jax_write_row_argument(row, a)
    rows = Vector{Vector{Float64}}(undef, 1)
    rows[1] = row
    sum(rows[1])
end
@traceable function jax_mismatched_rows(a::Float64)
    rows = Vector{Vector{Float64}}(undef, 2)
    rows[1] = [a]
    rows[2] = [a, a]
    sum(rows[1])
end
@traceable function jax_reduce_slots(a::Float64)
    rows = Vector{Vector{Float64}}(undef, 2)
    rows[1] = [a, a]
    sum(sum(rows))
end
for (name, helper) in ((:jax_row_alias_graph, :jax_stored_row_mutation),
                       (:jax_row_while_alias_graph, :jax_stored_row_while_mutation),
                       (:jax_row_view_graph, :jax_stored_row_view_mutation),
                       (:jax_borrowed_graph, :jax_borrowed_mutation),
                       (:jax_borrowed_row_graph, :jax_borrowed_row_mutation),
                       (:jax_shapes_graph, :jax_mismatched_rows),
                       (:jax_reduce_graph, :jax_reduce_slots))
    @eval @kernel $name(a::Float64) = begin
        score = $helper(a)
        return score
    end
end

@testset "nested-vector buffers with runtime assignment state" begin
    src = jax_source(jax_slots_graph; have=(:a, :labels, :rates), data=(:labels,),
        functions=(score=:score, mapped=:mapped))
    ragged = jax_source(jax_slots_ragged; have=(:a, :labelsets, :ratesets), data=(:labelsets,), functions=(score=:score,))
    derived = jax_source(jax_slots_derived; have=(:a, :rawsets, :ratesets), data=(:rawsets,), functions=(score=:score,))
    ragged_plate = jax_source(jax_slots_ragged_plate; have=(:a, :labelsets, :ratesets), data=(:labelsets,), functions=(score=:score,))
    record_plate = jax_source(jax_slots_record_plate; have=(:a, :labelsets, :ratesets), data=(:labelsets,), functions=(score=:score,))
    partial = jax_source(jax_partial_graph; have=(:a, :i, :guarded), functions=(score=:score,))
    @test occursin("jax.lax.fori_loop", string(src))
    @test occursin("jax.lax.cond", string(src))
    @test string(jax_source(jax_slots_graph; have=(:a, :labels, :rates),
        data=(labels=[1, 2],), functions=(score=:score, mapped=:mapped))) ==
        string(jax_source(jax_slots_graph; have=(:a, :labels, :rates),
        data=(labels=[1, 9, 1, 4],), functions=(score=:score, mapped=:mapped)))
    cases = [Int[], [1], [3, 1, 3], [7, 2, 7, 2], repeat([4, 1, 4], 23)]
    a = 0.4
    expected = Float64[]
    gradients = Float64[]
    rate_gradients = Vector{Float64}[]
    native = prepare(jax_slots_graph; have=(:a, :labels, :rates), want=(:score, :mapped))
    for labels in cases
        rates = collect(range(-0.2, step=0.02, length=length(labels)))
        v = jax_rows_total(jax_group_rows(a, labels, rates))
        push!(expected, v)
        drates = zeros(length(rates))
        da = Enzyme.autodiff(Enzyme.Reverse, jax_slot_loss, Enzyme.Active,
            Enzyme.Active(a), Enzyme.Const(labels), Enzyme.Duplicated(rates, drates))[1][1]
        push!(gradients, da)
        push!(rate_gradients, drates)
        @test da ≈ only(rk_fd_gradient(x -> jax_slot_loss(x[1], labels, rates), [a]))
        @test all(isapprox.(native(a, labels, rates), (v, v)))
    end
    @test jax_partial_rows(a, 1, true) ≈ a + a*a
    @test jax_partial_rows(a, 2, true) == 0.0
    @test_throws UndefRefError jax_partial_rows(a, 2, false)
    @test_throws BoundsError jax_partial_rows(a, 3, false)
    log_labels = [Int[], [1], [1, 2, 1]]
    log_rates = [Float64[], [-0.2], [-0.2, 0.1, 0.8]]
    @test jax_slots_log_loss(a, log_labels, log_rates) ≈ -0.12
    record_native = prepare(jax_slots_record_plate; have=(:a, :labelsets, :ratesets), want=:score)
    @test record_native(a, log_labels, log_rates) ≈ -0.12
    @test Enzyme.autodiff(Enzyme.Reverse, jax_slots_log_loss, Enzyme.Active,
        Enzyme.Active(a), Enzyme.Const(log_labels), Enzyme.Const(log_rates))[1][1] ≈ -0.3
    for graph in (jax_row_alias_graph, jax_row_while_alias_graph, jax_row_view_graph, jax_borrowed_graph,
                  jax_borrowed_row_graph, jax_shapes_graph, jax_reduce_graph)
        @test_throws JAXEmissionError jax_source(graph; have=(:a,), functions=(score=:score,))
    end
    if rk_jax_available()
        mktempdir() do dir
            write(joinpath(dir, "slots.py"), src)
            write(joinpath(dir, "ragged.py"), ragged)
            write(joinpath(dir, "derived.py"), derived)
            write(joinpath(dir, "ragged_plate.py"), ragged_plate)
            write(joinpath(dir, "record_plate.py"), record_plate)
            write(joinpath(dir, "partial.py"), partial)
            # Keep the generated file for diagnosing a failing Python run.
            get(ENV, "RK_SLOT_SOURCE", "") == "" || write(ENV["RK_SLOT_SOURCE"], src)
            code = "cases = " * _rk_py(cases) * "\nexpected = " * _rk_py(expected) *
                "\ngradients = " * _rk_py(gradients) * "\nrate_gradients = " * _rk_py(rate_gradients) * "\n" * raw"""
import jax
import numpy as np
import slots, ragged, derived, ragged_plate, record_plate, partial
for labels, value, gradient, drates in zip(cases, expected, gradients, rate_gradients):
    labels = np.asarray(labels, dtype=np.int64)
    rates = -0.2 + 0.02*np.arange(labels.size)
    prepared = slots.prepare_data(labels=labels)
    for f in (slots.score, slots.mapped):
        np.testing.assert_allclose(jax.jit(f)(0.4, rates, **prepared), value, rtol=1e-10)
        np.testing.assert_allclose(jax.jit(jax.grad(f))(0.4, rates, **prepared), gradient, rtol=1e-8)
        np.testing.assert_allclose(jax.jit(jax.grad(f, argnums=1))(0.4, rates, **prepared), drates, rtol=1e-8, atol=1e-12)
labelsets = [np.asarray(x, dtype=np.int64) for x in cases]
ratesets = [-0.2 + 0.02*np.arange(len(x)) for x in cases]
for module, data in ((ragged, dict(labelsets=labelsets)),
                     (derived, dict(rawsets=[x.astype(float) for x in labelsets]))):
    prepared = module.prepare_data(**data)
    rates, prepared['ratesets_lengths'], prepared['ratesets_flat'] = module._rk_pad_rows(ratesets, np.float64)
    np.testing.assert_allclose(jax.jit(module.score)(0.4, rates, **prepared), sum(expected), rtol=1e-10)
    np.testing.assert_allclose(jax.jit(jax.grad(module.score))(0.4, rates, **prepared), sum(gradients), rtol=1e-8)
    actual = jax.jit(jax.grad(module.score, argnums=1))(0.4, rates, **prepared)
    for row, expected_row in zip(actual, rate_gradients):
        np.testing.assert_allclose(row[:len(expected_row)], expected_row, rtol=1e-8, atol=1e-12)
        np.testing.assert_allclose(row[len(expected_row):], 0., atol=1e-12)
prepared = ragged.prepare_data(labelsets=[np.array([], dtype=np.int64)] * 3)
rates, prepared['ratesets_lengths'], prepared['ratesets_flat'] = ragged._rk_pad_rows([np.array([])] * 3, np.float64)
np.testing.assert_allclose(jax.jit(ragged.score)(0.4, rates, **prepared), 0.)
np.testing.assert_allclose(jax.jit(jax.grad(ragged.score))(0.4, rates, **prepared), 0.)
# A zero-length buffer inside a nonempty outer plate must not differentiate
# its unused storage through log(0), including the shared scalar argument.
for module in (ragged_plate, record_plate):
    prepared = module.prepare_data(labelsets=[np.array(x, dtype=np.int64) for x in ([], [1], [1,2,1])])
    rates, prepared['ratesets_lengths'], prepared['ratesets_flat'] = module._rk_pad_rows(
        [np.array([]), np.array([-.2]), np.array([-.2,.1,.8])], np.float64)
    np.testing.assert_allclose(jax.jit(module.score)(.4, rates, **prepared), -.12, atol=1e-12)
    np.testing.assert_allclose(jax.jit(jax.grad(module.score))(.4, rates, **prepared), -.3, atol=1e-12)
for a in (-0.4, 0.4):
    np.testing.assert_allclose(jax.jit(partial.score)(a, 1, False), abs(a)+a*a)
    for i in (-1, 0, 2, 3):
        np.testing.assert_allclose(jax.jit(partial.score)(a, i, True), 0.)
        np.testing.assert_allclose(jax.jit(jax.grad(partial.score))(a, i, True), 0.)
for i, error in ((2, 'UndefRefError'), (3, 'BoundsError')):
    try:
        jax.jit(partial.score)(0.4, i, False).block_until_ready()
    except Exception as exc:
        assert error in str(exc), str(exc)
    else:
        raise AssertionError('missing '+error)
prepared = slots.prepare_data(labels=np.array([1, 1]))
for f in (slots.score, slots.mapped):
    np.testing.assert_allclose(jax.jit(jax.grad(f, argnums=1))(0.4, np.array([0.1, np.nan]), **prepared)[1], 0.)
def loops(tree):
    tree = getattr(tree, 'jaxpr', tree)
    count = 0
    for eqn in tree.eqns:
        count += eqn.primitive.name in ('scan', 'while')
        for v in eqn.params.values():
            for child in v if isinstance(v, (tuple, list)) else (v,):
                if hasattr(child, 'jaxpr') or hasattr(child, 'eqns'):
                    count += loops(child)
    return count
counts = []
for n in (3, 70):
    prepared = slots.prepare_data(labels=np.arange(1,n+1))
    counts.append([loops(jax.make_jaxpr(f)(0.4, np.zeros(n), **prepared))
                   for f in (slots.score, jax.grad(slots.score))])
assert counts[0] == counts[1] and min(counts[0]) >= 3, counts
print('retained loop regions at lengths 3 and 70:', counts)
print('slot buffers: empty, duplicate, sparse and growing labels passed')
"""
            write(joinpath(dir, "check.py"), code)
            if haskey(ENV, "RK_SLOT_DIR")
                mkpath(ENV["RK_SLOT_DIR"])
                for name in ("slots.py", "ragged.py", "derived.py", "ragged_plate.py", "record_plate.py", "partial.py", "check.py")
                    cp(joinpath(dir, name), joinpath(ENV["RK_SLOT_DIR"], name); force=true)
                end
            end
            @test success(pipeline(`$RK_JAX_PYTHON $(joinpath(dir, "check.py"))`; stdout=stdout, stderr=stderr))
        end
    else
        @test_skip false
    end
end
