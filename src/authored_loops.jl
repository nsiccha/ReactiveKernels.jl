# Loop syntax for authored plates and scans (user decision `05cugvn`): a
# statement `@plate for i in R … end` of a `@kernel` body desugars onto the
# authored `plate(...) do` operation, so it lowers exactly as the call form.
#
# - Each cell is one iteration of the loop.
# - A read `X[i]` at exactly the loop index zips `X` as a plate argument; every
#   other read is a closure capture (`sigma`, `x[idx[i]]`, `x[i - 1]`). An
#   array the cell also reads whole stays a capture, read at `i` by a gather.
# - `out[i] = expr` (optionally `out[i]::T = expr`) names an output of the
#   loop; each output becomes its own `out = plate(...) do … end` over the cell
#   statements it reads (`_kernel_plate_loop_slice`), exactly the call form
#   written for that output alone: it captures, zips and checks only what those
#   statements read, so a cell of one output computes nothing that only a
#   sibling output reads (snag `sibling-only-pla-2b3df69e`).
# - A zipped array has exactly the loop's indices. When that is not
#   syntactically guaranteed (`@plate for i in eachindex(y)` reading only `y[i]`),
#   the loop's domain is passed as a plate argument through
#   `_plate_loop_axes`, which checks every zipped array against it, so a
#   singleton is never silently repeated across the domain.

"""
    @plate for i in R
        …
        out[i] = expr
    end

Loop syntax for an authored plate inside a `@kernel` body: each iteration is
one cell, `X[i]` at the loop index zips `X`, other values are read as closures,
and `out[i] = …` names the plate's output `out`. It prepares the same kernel as
the equivalent `out = plate(...) do … end`. It is recognized only as a
statement of a `@kernel` body.
"""
macro plate(args...)
    throw(ArgumentError(
        "`@plate for i in R … end` is @kernel authoring syntax: write it as a " *
        "statement of a @kernel body."))
end

const _PLATE_MACRO = Symbol("@plate")

_kernel_loop_macro(mod, name, target::Symbol, macro_value) =
    mod === nothing ? name === target :
    _kernel_resolve_binding(mod, name) === macro_value

_kernel_is_plate_loop(mod, ex) =
    ex isa Expr && ex.head === :macrocall && length(ex.args) >= 3 &&
    _kernel_loop_macro(mod, ex.args[1], _PLATE_MACRO, var"@plate")

# Desugar every `@plate for` statement, including those nested in another
# loop's cells or in a `plate(...) do` / `scan(...) do` body, before any cell
# computes its captures.
function _kernel_desugar_loops(statements, mod)
    # Bodies without loop syntax are returned as they are.
    any(st -> _kernel_has_loop_macro(st, mod), statements) || return statements
    result = Any[]
    line = nothing
    for statement in statements
        statement isa LineNumberNode && (line = statement)
        if _kernel_is_plate_loop(mod, statement)
            statement.args[2] isa LineNumberNode && (line = statement.args[2])
            append!(result, _kernel_plate_loop_statements(statement, mod, line))
        elseif _kernel_is_scan_loop(mod, statement)
            statement.args[2] isa LineNumberNode && (line = statement.args[2])
            append!(result, _kernel_scan_loop_statements(statement, mod, line))
        else
            push!(result, _kernel_desugar_nested_loops(statement, mod))
        end
    end
    result
end

_kernel_has_loop_macro(ex, mod) =
    ex isa Expr && ex.head !== :quote &&
    (_kernel_is_plate_loop(mod, ex) || _kernel_is_scan_loop(mod, ex) ||
     any(arg -> _kernel_has_loop_macro(arg, mod), ex.args))

function _kernel_desugar_nested_loops(ex, mod)
    _kernel_has_loop_macro(ex, mod) || return ex
    ex.head === :quote && return ex
    if ex.head === :do && length(ex.args) == 2 &&
       ex.args[2] isa Expr && ex.args[2].head === :(->)
        lambda = ex.args[2]
        body = lambda.args[2]
        statements = body isa Expr && body.head === :block ? body.args : Any[body]
        new_body = Expr(:block, _kernel_desugar_loops(statements, mod)...)
        return Expr(:do, _kernel_desugar_nested_loops(ex.args[1], mod),
                    Expr(:(->), lambda.args[1], new_body))
    end
    Expr(ex.head, (_kernel_desugar_nested_loops(arg, mod) for arg in ex.args)...)
end

function _kernel_plate_loop_statements(statement::Expr, mod, line)
    loop = statement.args[end]
    loop isa Expr && loop.head === :for || throw(ArgumentError(
        "`@plate` takes a `for` loop: `@plate for i in R … end`"))
    spec, body = loop.args
    spec isa Expr && spec.head === :(=) && length(spec.args) == 2 &&
        spec.args[1] isa Symbol || throw(ArgumentError(
            "`@plate for` takes one loop variable: `@plate for i in R … end`"))
    index, domain = spec.args
    statements = body isa Expr && body.head === :block ? body.args : Any[body]

    taken = Set{Symbol}()
    _kernel_loop_symbols!(taken, loop)

    # Outputs. An output is written once, at the loop index.
    outputs = Tuple{Symbol,Any,Symbol}[]          # (name, element type, cell local)
    for st in statements
        target = _kernel_plate_loop_output(st, index)
        target === nothing && continue
        name, T, _ = target
        any(o -> o[1] === name, outputs) && throw(ArgumentError(
            "`@plate for` writes `$name[$index]` twice"))
        push!(outputs, (name, T, _kernel_loop_fresh!(taken, name, index)))
    end
    isempty(outputs) && throw(ArgumentError(
        "`@plate for $index in …` must write at least one output, " *
        "`out[$index] = …`, at the loop index"))

    mapping = Dict{Symbol,Symbol}(name => local_name for (name, _, local_name) in outputs)
    output_names = Set(first.(outputs))
    # Each output is its own plate over the statements it reads, minting its
    # zipped arguments' names as that output's call form alone would.
    Any[_kernel_plate_loop_plate(
            _kernel_plate_loop_slice(statements, index, name, mod), index, domain,
            name, local_name, copy(mapping), output_names, copy(taken), mod, line)
        for (name, _, local_name) in outputs]
end

# A cell-local name for `base` at loop index `index` that the loop does not use.
function _kernel_loop_fresh!(taken, base, index)
    name = Symbol(base, :_, index)
    while name in taken
        name = Symbol(name, :_)
    end
    push!(taken, name)
    name
end

# The statements of a `@plate for` body that output `name` reads: its own write
# and, to a fixpoint, every statement that writes a name a kept statement reads
# (any symbol it holds, so a cell local shadowing an outer value counts). A
# statement that writes no name, such as a check, is kept for every output.
function _kernel_plate_loop_slice(statements, index, name, mod)
    writes = Any[_kernel_loop_statement_writes(st, index, mod) for st in statements]
    keep = Bool[w === nothing for w in writes]
    needed = Set{Symbol}((name,))
    foreach(j -> _kernel_loop_symbols!(needed, statements[j]), findall(keep))
    changed = true
    while changed
        changed = false
        for j in eachindex(statements)
            keep[j] && continue
            isdisjoint(writes[j], needed) && continue
            keep[j] = true
            _kernel_loop_symbols!(needed, statements[j])
            changed = true
        end
    end
    statements[keep]
end

# The names a statement of a `@plate for` body writes: `out` for an output
# `out[i] = …`, the targets of an assignment, the arrays a nested `@plate for`
# or `@scan` writes by index. `nothing` when it writes no name it can name.
function _kernel_loop_statement_writes(st, index, mod)
    st isa Expr || return nothing
    if st.head === :(=) && length(st.args) == 2
        target = _kernel_plate_loop_output(st, index)
        target === nothing || return Set{Symbol}((target[1],))
        names = _kernel_assignment_names!(Set{Symbol}(), st.args[1])
        return isempty(names) ? nothing : names
    end
    if _kernel_is_plate_loop(mod, st) || _kernel_is_scan_loop(mod, st)
        names = _kernel_loop_indexed_writes!(Set{Symbol}(), st)
        return isempty(names) ? nothing : names
    end
    nothing
end

function _kernel_loop_indexed_writes!(names, ex)
    ex isa Expr && ex.head !== :quote || return names
    if ex.head === :(=) && length(ex.args) == 2
        lhs = ex.args[1]
        Meta.isexpr(lhs, :(::), 2) && (lhs = lhs.args[1])
        Meta.isexpr(lhs, :ref) && lhs.args[1] isa Symbol && push!(names, lhs.args[1])
    end
    foreach(arg -> _kernel_loop_indexed_writes!(names, arg), ex.args)
    names
end

# The `out = plate(...) do … end` of output `name` over its statements `own`.
function _kernel_plate_loop_plate(own, index, domain, name, local_name, mapping,
                                  output_names, taken, mod, line)
    fresh(base) = _kernel_loop_fresh!(taken, base, index)
    locals = Set{Symbol}()
    for st in own
        _kernel_plate_loop_output(st, index) === nothing && st isa Expr &&
            st.head === :(=) && _kernel_assignment_names!(locals, st.args[1])
    end
    # An array the cell also reads whole is captured; its reads at the index
    # are gathers from that capture rather than a second, zipped operand.
    whole = Set{Symbol}()
    foreach(st -> _kernel_loop_whole_reads!(whole, st, index), own)
    union!(locals, whole)
    zipped = Symbol[]
    cell = Any[]
    for st in own
        target = _kernel_plate_loop_output(st, index)
        if target === nothing
            push!(cell, _kernel_loop_rewrite(st, index, mapping, zipped, locals, fresh))
        else
            written, T, rhs = target
            rewritten = _kernel_loop_rewrite(rhs, index, mapping, zipped, locals, fresh)
            lhs = T === nothing ? mapping[written] : Expr(:(::), mapping[written], T)
            push!(cell, Expr(:(=), lhs, rewritten))
        end
    end
    filter!(x -> !(x in output_names), zipped)
    cell = _kernel_desugar_loops(cell, mod)

    # The loop's domain is a plate argument unless the cells only read the one
    # array the domain iterates.
    uses_index = any(st -> _kernel_loop_uses(st, index), cell)
    single = !uses_index && length(zipped) == 1 && _kernel_is_eachindex_of(domain, zipped[1])
    operands = Any[]
    formals = Symbol[]
    if !single
        checked = if isempty(zipped) || _kernel_eachindex_covers(domain, zipped)
            domain
        else
            Expr(:call, GlobalRef(@__MODULE__, :_plate_loop_axes),
                 Expr(:call, GlobalRef(Base, :Val), QuoteNode(Tuple(zipped))),
                 domain, zipped...)
        end
        push!(operands, checked)
        push!(formals, index)
    end
    append!(operands, zipped)
    append!(formals, (mapping[x] for x in zipped))

    plate_call = Expr(:call, GlobalRef(@__MODULE__, :plate), operands...)
    cell_body = Expr(:block, (line === nothing ? () : (line,))..., cell..., local_name)
    Expr(:(=), name, Expr(:do, plate_call, Expr(:(->), Expr(:tuple, formals...), cell_body)))
end

# `out[i] = rhs` or `out[i]::T = rhs` at exactly the loop index.
function _kernel_plate_loop_output(st, index)
    st isa Expr && st.head === :(=) && length(st.args) == 2 || return nothing
    lhs, rhs = st.args
    T = nothing
    if lhs isa Expr && lhs.head === :(::) && length(lhs.args) == 2
        lhs, T = lhs.args
    end
    lhs isa Expr && lhs.head === :ref || return nothing
    length(lhs.args) == 2 && lhs.args[1] isa Symbol && lhs.args[2] === index ||
        throw(ArgumentError(
            "`@plate for $index in …` writes outputs at exactly the loop index " *
            "(`out[$index] = …`); got `$(lhs)`"))
    (lhs.args[1], T, rhs)
end

# Rewrite reads at the loop index: an output read `out[i]` becomes its cell
# local, an array read `X[i]` becomes X's zipped plate argument. Scopes that
# rebind the loop variable are left alone.
function _kernel_loop_rewrite(ex, index, mapping, zipped, locals, fresh)
    ex isa Expr || return ex
    ex.head === :quote && return ex
    if ex.head === :ref && length(ex.args) == 2 && ex.args[2] === index &&
       ex.args[1] isa Symbol
        array = ex.args[1]
        haskey(mapping, array) && return mapping[array]
        if !(array in locals) && array !== index
            mapping[array] = fresh(array)
            push!(zipped, array)
            return mapping[array]
        end
        return ex
    end
    if _kernel_loop_rebinds(ex, index)
        return _kernel_loop_rewrite_outer_parts(ex, index, mapping, zipped, locals, fresh)
    end
    Expr(ex.head, (_kernel_loop_rewrite(arg, index, mapping, zipped, locals, fresh)
                   for arg in ex.args)...)
end

# Within a scope that rebinds the loop variable only its iterator expressions
# belong to the enclosing cell.
function _kernel_loop_rewrite_outer_parts(ex, index, mapping, zipped, locals, fresh)
    rewrite(x) = _kernel_loop_rewrite(x, index, mapping, zipped, locals, fresh)
    if ex.head in (:generator, :comprehension) || ex.head === :for
        return Expr(ex.head, (
            arg isa Expr && arg.head === :(=) && length(arg.args) == 2 ?
                Expr(:(=), arg.args[1], rewrite(arg.args[2])) : arg
            for arg in ex.args)...)
    end
    ex
end

function _kernel_loop_rebinds(ex::Expr, index)
    names = Set{Symbol}()
    if ex.head === :for || ex.head === :generator || ex.head === :comprehension
        specs = ex.head === :for ? Any[ex.args[1]] : ex.args[2:end]
        for spec in specs
            for iterator in _kernel_iterators(spec)
                iterator isa Expr && iterator.head in (:(=), :in) &&
                    _kernel_bound_names!(names, iterator.args[1])
            end
        end
    elseif ex.head === :(->)
        _kernel_bound_names!(names, ex.args[1])
    elseif ex.head === :let
        for binding in ex.args[1:(end - 1)]
            binding isa Expr && binding.head === :(=) ?
                _kernel_bound_names!(names, binding.args[1]) :
                _kernel_bound_names!(names, binding)
        end
    elseif ex.head === :macrocall && length(ex.args) >= 3 && ex.args[end] isa Expr &&
           ex.args[end].head === :for
        return _kernel_loop_rebinds(ex.args[end], index)
    end
    index in names
end

# Whether the loop variable is still read after the rewrite.
function _kernel_loop_uses(ex, index)
    ex === index && return true
    ex isa Expr || return false
    ex.head === :quote && return false
    if _kernel_loop_rebinds(ex, index)
        if ex.head in (:generator, :comprehension, :for)
            return any(arg -> arg isa Expr && arg.head === :(=) &&
                              _kernel_loop_uses(arg.args[2], index), ex.args)
        end
        return false
    end
    any(arg -> _kernel_loop_uses(arg, index), ex.args)
end

# Names a loop body reads other than at the loop index (`X` in `X[i]`).
function _kernel_loop_whole_reads!(names, ex, index)
    if ex isa Symbol
        push!(names, ex)
    elseif ex isa Expr && ex.head !== :quote
        if ex.head === :ref && length(ex.args) == 2 && ex.args[1] isa Symbol &&
           ex.args[2] === index
            return names
        end
        if ex.head === :(=) && length(ex.args) == 2
            lhs = ex.args[1]
            lhs isa Expr && lhs.head === :(::) && (lhs = lhs.args[1])
            lhs isa Expr && lhs.head === :ref && length(lhs.args) == 2 &&
                lhs.args[2] === index && return _kernel_loop_whole_reads!(names, ex.args[2], index)
        end
        foreach(arg -> _kernel_loop_whole_reads!(names, arg, index), ex.args)
    end
    names
end

function _kernel_loop_symbols!(names, ex)
    ex isa Symbol && push!(names, ex)
    ex isa Expr && foreach(arg -> _kernel_loop_symbols!(names, arg), ex.args)
    names
end

_kernel_is_eachindex_of(domain, array) =
    domain isa Expr && domain.head === :call && length(domain.args) == 2 &&
    domain.args[1] === :eachindex && domain.args[2] === array

# `eachindex(A, B, …)` itself requires its arguments to share indices, so it
# checks every zipped array it names.
_kernel_eachindex_covers(domain, zipped) =
    domain isa Expr && domain.head === :call && length(domain.args) >= 2 &&
    domain.args[1] === :eachindex && all(x -> x in domain.args[2:end], zipped)

"""
    _plate_loop_axes(Val(names), domain, arrays...)

Return `domain` after checking that every array a `@plate for` loop reads at
its index has exactly the loop's indices, so the arrays zip with the domain
cell by cell and none is repeated across it.
"""
function _plate_loop_axes(::Val{names}, domain, arrays...) where {names}
    for (name, array) in zip(names, arrays)
        eachindex(array) == domain || throw(DimensionMismatch(
            "`@plate for` reads `$name` at the loop index, so its indices " *
            "$(eachindex(array)) must equal the loop's $(domain)"))
    end
    domain
end

"""
    @scan begin
        a[1] = seed
        for t in 2:T
            a[t] = f(a[t - 1], x[t], t)
        end
    end

Loop syntax for an authored scan inside a `@kernel` body. Every array the loop
writes at `t` is carried: it is seeded at `1..m` before the loop (the same depth
for every carried array), each step writes it once, and the step reads earlier
values through literal lags `a[t - k]` (`1 ≤ k ≤ m`), the current value `a[t]`
after its write, and any earlier entry whose index provably lies in
`1:(t - 1)`: `a[s]` or `a[t - s]` under a generator `for s in 1:(t - 1)`, or a
window `a[(t - k):(t - 1)]`. Other values are read as closures. The result `a`
is the whole trajectory `[a[1], …, a[T]]`. One carried array with one seed
prepares the same kernel as `a = scan(2:T; init = seed, include_init = true) do
a_prev, t … end`; several carried arrays share one scan and expose named
output columns. History reads use seed-prefixed buffers, one per carried
array, so mutually dependent earlier entries are supported in the same scan
(`docs/src/scan.md`, "Reading earlier entries").
"""
macro scan(args...)
    throw(ArgumentError(
        "`@scan begin … end` is @kernel authoring syntax: write it as a statement " *
        "of a @kernel body."))
end

const _SCAN_MACRO = Symbol("@scan")

_kernel_is_scan_loop(mod, ex) =
    ex isa Expr && ex.head === :macrocall && length(ex.args) >= 3 &&
    _kernel_loop_macro(mod, ex.args[1], _SCAN_MACRO, var"@scan")

function _kernel_scan_loop_statements(statement::Expr, mod, line)
    block = statement.args[end]
    items = block isa Expr && block.head === :block ?
        Any[s for s in block.args if !(s isa LineNumberNode)] : Any[block]
    (isempty(items) || !(items[end] isa Expr && items[end].head === :for)) &&
        throw(ArgumentError(
            "`@scan begin … end` holds the seeds `a[1] = …` followed by one " *
            "`for t in R … end` loop"))
    loop = items[end]
    spec, body = loop.args
    spec isa Expr && spec.head === :(=) && length(spec.args) == 2 &&
        spec.args[1] isa Symbol || throw(ArgumentError(
            "`@scan`'s loop takes one loop variable: `for t in R … end`"))
    index, domain = spec.args

    # Seeds `a[k] = expr` at literal 1..m, the same depth for every array.
    seeds = Dict{Symbol,Dict{Int,Any}}()
    carried = Symbol[]
    for st in items[1:(end - 1)]
        target = _kernel_scan_seed(st)
        target === nothing && throw(ArgumentError(
            "`@scan` takes only seeds `a[k] = …` (a literal index) before its loop; got `$st`"))
        name, k, rhs = target
        haskey(seeds, name) || (seeds[name] = Dict{Int,Any}(); push!(carried, name))
        haskey(seeds[name], k) && throw(ArgumentError("`@scan` seeds `$name[$k]` twice"))
        seeds[name][k] = rhs
    end
    isempty(carried) && throw(ArgumentError(
        "`@scan` needs at least one seeded array `a[1] = …` before its loop"))
    depth = maximum(keys(seeds[first(carried)]))
    for name in carried
        sort!(collect(keys(seeds[name]))) == collect(1:depth) || throw(ArgumentError(
            "`@scan` seeds every carried array at 1..$depth; `$name` has " *
            "$(sort!(collect(keys(seeds[name]))))"))
    end

    taken = Set{Symbol}()
    _kernel_loop_symbols!(taken, statement)
    fresh(base) = begin
        name = base
        while name in taken
            name = Symbol(name, :_)
        end
        push!(taken, name)
        name
    end

    # Seed values are named once; a seed may read earlier seeds of any array.
    seed_names = Dict{Tuple{Symbol,Int},Symbol}()
    for name in carried, k in 1:depth
        seed_names[(name, k)] = gensym(Symbol(name, :_seed, k))
    end
    result = Any[]
    for name in carried, k in 1:depth
        rhs = _kernel_scan_seed_reads(seeds[name][k], seed_names, k)
        push!(result, Expr(:(=), seed_names[(name, k)], rhs))
    end

    statements = body isa Expr && body.head === :block ? body.args : Any[body]
    writes = Symbol[]
    for st in statements
        target = _kernel_scan_write(st, index)
        target === nothing && continue
        name = target[1]
        name in carried || throw(ArgumentError(
            "`@scan` writes `$name[$index]`, which has no seed before the loop"))
        name in writes && throw(ArgumentError("`@scan` writes `$name[$index]` twice"))
        push!(writes, name)
    end
    for name in carried
        name in writes || throw(ArgumentError(
            "`@scan` seeds `$name` but its loop never writes `$name[$index]`"))
    end

    summaries = _kernel_scan_statement_summaries(statements, index)
    reads = _history_statement_reads(carried, index, last.(summaries))
    _history_has_reads(reads) && return append!(result,
        _kernel_scan_history_folds(statements, summaries, reads, index, domain,
            carried, depth, seed_names, fresh, mod, line))

    scalar = length(carried) == 1 && depth == 1
    carry = scalar ? fresh(Symbol(only(carried), :_prev)) : fresh(:carry)
    lag(name, k) = scalar ? carry :
        Expr(:., carry, QuoteNode(Symbol(name, :_lag, k)))
    step_locals = Dict{Symbol,Symbol}(name => fresh(Symbol(name, :_, index)) for name in carried)
    written = Set{Symbol}()
    step = Any[]
    for st in statements
        target = _kernel_scan_write(st, index)
        if target === nothing
            push!(step, _kernel_scan_rewrite(st, index, carried, depth, lag, step_locals, written))
        else
            name, T, rhs = target
            rewritten = _kernel_scan_rewrite(rhs, index, carried, depth, lag, step_locals, written)
            push!(written, name)
            lhs = T === nothing ? step_locals[name] : Expr(:(::), step_locals[name], T)
            push!(step, Expr(:(=), lhs, rewritten))
        end
    end
    step = _kernel_desugar_loops(step, mod)
    lines = line === nothing ? () : (line,)
    scan_ref = GlobalRef(@__MODULE__, :scan)
    if scalar
        name = only(carried)
        call = Expr(:call, scan_ref,
                    Expr(:parameters, Expr(:kw, :init, seed_names[(name, 1)]),
                         Expr(:kw, :include_init, true)),
                    domain)
        step_body = Expr(:block, lines..., step...,
                         Expr(:tuple, step_locals[name], step_locals[name]))
        push!(result, Expr(:(=), name,
                           Expr(:do, call, Expr(:(->), Expr(:tuple, carry, index), step_body))))
        return result
    end
    init = Expr(:tuple, Expr(:parameters, (
        Expr(:kw, Symbol(name, :_lag, k), seed_names[(name, depth - k + 1)])
        for name in carried for k in 1:depth)...))
    next = Expr(:tuple, Expr(:parameters, (
        Expr(:kw, Symbol(name, :_lag, k), k == 1 ? step_locals[name] : lag(name, k - 1))
        for name in carried for k in 1:depth)...))
    steps = gensym(:scan_steps)
    one_output = length(carried) == 1
    output = one_output ? step_locals[only(carried)] :
        _scan_named_tuple_expr(carried, name -> step_locals[name])
    call = Expr(:call, scan_ref, Expr(:parameters, Expr(:kw, :init, init)), domain)
    step_body = Expr(:block, lines..., step..., Expr(:tuple, next, output))
    push!(result, Expr(:(=), steps,
                       Expr(:do, call, Expr(:(->), Expr(:tuple, carry, index), step_body))))
    for name in carried
        column = one_output ? steps : gensym(Symbol(name, :_steps))
        one_output || push!(result, Expr(:(=), column, _scan_column_expr(steps, name)))
        push!(result, Expr(:(=), name, Expr(:call, GlobalRef(Base, :vcat),
                                            Expr(:vect, (seed_names[(name, k)] for k in 1:depth)...),
                                            column)))
    end
    result
end

function _kernel_scan_seed(st)
    st isa Expr && st.head === :(=) && length(st.args) == 2 || return nothing
    lhs, rhs = st.args
    lhs isa Expr && lhs.head === :ref && length(lhs.args) == 2 &&
        lhs.args[1] isa Symbol && lhs.args[2] isa Int && lhs.args[2] >= 1 || return nothing
    (lhs.args[1], lhs.args[2], rhs)
end

# A seed reads earlier seeds by literal index.
function _kernel_scan_seed_reads(ex, seed_names, k)
    ex isa Expr || return ex
    ex.head === :quote && return ex
    if ex.head === :ref && length(ex.args) == 2 && ex.args[1] isa Symbol && ex.args[2] isa Int
        key = (ex.args[1], ex.args[2])
        if haskey(seed_names, key)
            key[2] < k || throw(ArgumentError(
                "`@scan`'s seed $k reads `$(ex.args[1])[$(ex.args[2])]`, which is not an earlier seed"))
            return seed_names[key]
        end
    end
    Expr(ex.head, (_kernel_scan_seed_reads(arg, seed_names, k) for arg in ex.args)...)
end

function _kernel_scan_write(st, index)
    st isa Expr && st.head === :(=) && length(st.args) == 2 || return nothing
    lhs, rhs = st.args
    T = nothing
    if lhs isa Expr && lhs.head === :(::) && length(lhs.args) == 2
        lhs, T = lhs.args
    end
    lhs isa Expr && lhs.head === :ref && length(lhs.args) == 2 &&
        lhs.args[1] isa Symbol && lhs.args[2] === index || return nothing
    (lhs.args[1], T, rhs)
end

# Rewrite carried reads in a step: `a[t - k]` reads the carry, `a[t]` the value
# this step already wrote. With a history context `ctx`, earlier entries of
# every carried array read its column of the shared trajectory buffers.
function _kernel_scan_rewrite(ex, index, carried, depth, lag, step_locals, written;
                              ctx = nothing, env = Dict{Symbol,Any}())
    ex isa Expr || return ex
    ex.head === :quote && return ex
    rewrite(x, e = env) = _kernel_scan_rewrite(x, index, carried, depth, lag,
        step_locals, written; ctx, env = e)
    if ex.head === :ref && ex.args[1] isa Symbol && ex.args[1] in carried
        name = ex.args[1]
        length(ex.args) == 2 || throw(ArgumentError(
            "`@scan` reads its carried array `$name` with one index"))
        position = ex.args[2]
        if position === index
            name in written || throw(ArgumentError(
                "`@scan` reads `$name[$index]` before the step writes it"))
            return step_locals[name]
        elseif position isa Expr && position.head === :call && length(position.args) == 3 &&
               position.args[1] === :- && position.args[2] === index &&
               position.args[3] isa Int && 1 <= position.args[3] <= depth
            return lag(name, position.args[3])
        elseif ctx !== nothing &&
               _history_is_earlier(position, index, ctx.lo, env)
            history = ctx.columns[name]
            return _history_is_slice(position) ?
                Expr(:call, GlobalRef(Base, :view), history, rewrite(position)) :
                Expr(:ref, history, rewrite(position))
        end
        throw(ArgumentError(
            "`@scan` reads `$(ex)`: a carried array is read at `$index` after its write, " *
            "at a literal lag `$name[$index - k]` with 1 ≤ k ≤ $depth, or at an " *
            "earlier entry whose index lies in `1:($index - 1)` at every step " *
            "(`$name[s]` or `$name[$index - s]` under `for s in 1:($index - 1)`, " *
            "or a slice `$name[($index - k):($index - 1)]`)"))
    end
    _kernel_loop_rebinds(ex, index) && return ex
    if ex.head === :generator && _history_generator_clauses(ex) !== nothing
        return _history_map_generator((x, e) -> rewrite(x, e), (x, e) -> rewrite(x, e),
            ex, index, env)
    end
    Expr(ex.head, (rewrite(arg) for arg in ex.args)...)
end

# Each statement of a scan body in the shared fold-planning form
# (`_history_statement_reads`), paired with its position in `statements`. A
# statement that neither writes a carried array nor assigns a name is kept in
# every fold.
function _kernel_scan_statement_summaries(statements, index)
    out = Tuple{Int,Any}[]
    for (i, st) in enumerate(statements)
        st isa Expr || continue
        target = _kernel_scan_write(st, index)
        summary = if target !== nothing
            (; writes = target[1], defines = Set{Symbol}(), sampled = false,
               reads = Any[target[3]])
        elseif st.head === :(=) && length(st.args) == 2
            (; writes = nothing, defines = _kernel_assignment_names!(Set{Symbol}(), st.args[1]),
               sampled = false, reads = Any[st.args[2]])
        else
            (; writes = nothing, defines = Set{Symbol}(), sampled = false, reads = Any[st])
        end
        push!(out, (i, summary))
    end
    out
end

# A history-reading recurrence owns all its trajectory columns in one scan.
# Each step runs once in source order and sees every column's earlier entries.
function _kernel_scan_history_folds(statements, summaries, reads, index, domain,
                                    carried, depth, seed_names, fresh, mod, line)
    lo = depth + 1
    (domain isa Expr && domain.head === :call && length(domain.args) == 3 &&
     domain.args[1] === :(:) && domain.args[2] == lo) || throw(ArgumentError(
        "`@scan` reads earlier entries of its carried arrays, so its loop must start " *
        "one past the seeds, written `for $index in $lo:…`; got `$domain`"))
    for (_, summary) in summaries, r in summary.reads
        _history_foreach_read(r, carried, index) do name, position, env
            position === index && return nothing
            _history_lag_form(position, index, env) !== nothing && return nothing
            _history_is_earlier(position, index, lo, env) || throw(ArgumentError(
                "`@scan` reads `$name[$position]`, whose index is not provably before " *
                "`$index`: an earlier entry's index must lie in `1:($index - 1)` at " *
                "every step (`$name[s]` or `$name[$index - s]` under " *
                "`for s in 1:($index - 1)`, or a slice `$name[($index - k):($index - 1)]`)"))
        end
    end
    scalar = length(carried) == 1
    carry, history = fresh(:carry), fresh(:history)
    lag(a, k) = Expr(:., carry, QuoteNode(Symbol(a, :_lag, k)))
    step_locals = Dict(a => fresh(Symbol(a, :_, index)) for a in carried)
    columns = Dict(a => scalar ? history : Expr(:., history, QuoteNode(a)) for a in carried)
    ctx = (; columns, lo)
    written = Set{Symbol}()
    step = Any[]
    for st in statements
        target = _kernel_scan_write(st, index)
        if target === nothing
            push!(step, _kernel_scan_rewrite(st, index, carried, depth, lag,
                                             step_locals, written; ctx))
        else
            a, T, rhs = target
            rewritten = _kernel_scan_rewrite(rhs, index, carried, depth, lag,
                                             step_locals, written; ctx)
            push!(written, a)
            lhs = T === nothing ? step_locals[a] : Expr(:(::), step_locals[a], T)
            push!(step, Expr(:(=), lhs, rewritten))
        end
    end
    step = _kernel_desugar_loops(step, mod)
    init = Expr(:tuple, Expr(:parameters, (
        Expr(:kw, Symbol(a, :_lag, k), seed_names[(a, depth - k + 1)])
        for a in carried for k in 1:depth)...))
    next = Expr(:tuple, Expr(:parameters, (
        Expr(:kw, Symbol(a, :_lag, k), k == 1 ? step_locals[a] : lag(a, k - 1))
        for a in carried for k in 1:depth)...))
    seed(a) = Expr(:vect, (seed_names[(a, k)] for k in 1:depth)...)
    seeds = scalar ? seed(only(carried)) : _scan_named_tuple_expr(carried, seed)
    output = scalar ? step_locals[only(carried)] :
        _scan_named_tuple_expr(carried, a -> step_locals[a])
    call = Expr(:call, GlobalRef(@__MODULE__, :scan),
        Expr(:parameters, Expr(:kw, :init, init), Expr(:kw, :history, seeds)), domain)
    lines = line === nothing ? () : (line,)
    step_body = Expr(:block, lines..., step..., Expr(:tuple, next, output))
    fold = Expr(:do, call, Expr(:(->), Expr(:tuple, carry, index, history), step_body))
    result = Any[]
    trajectories = scalar ? only(carried) : gensym(:scan_trajectories)
    push!(result, Expr(:(=), trajectories, fold))
    if !scalar
        for a in carried
            push!(result, Expr(:(=), a, Expr(:., trajectories, QuoteNode(a))))
        end
    end
    result
end
