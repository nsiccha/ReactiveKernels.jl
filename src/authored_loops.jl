# Loop syntax for authored plates and scans (user decision `05cugvn`): a
# statement `@plate for i in R … end` of a `@kernel` body desugars onto the
# authored `plate(...) do` operation, so it lowers exactly as the call form.
#
# - Each cell is one iteration of the loop.
# - A read `X[i]` at exactly the loop index zips `X` as a plate argument; every
#   other read is a closure capture (`sigma`, `x[idx[i]]`, `x[i - 1]`). An
#   array the cell also reads whole stays a capture, read at `i` by a gather.
# - `out[i] = expr` (optionally `out[i]::T = expr`) names an output of the
#   loop; each output becomes its own `out = plate(...) do … end` over the same
#   cell statements. The planner selects only the statements an output needs.
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
    fresh(base) = begin
        name = Symbol(base, :_, index)
        while name in taken
            name = Symbol(name, :_)
        end
        push!(taken, name)
        name
    end

    # Outputs and cell locals. An output is written once, at the loop index.
    outputs = Tuple{Symbol,Any,Symbol}[]          # (name, element type, cell local)
    locals = Set{Symbol}()
    for st in statements
        target = _kernel_plate_loop_output(st, index)
        if target === nothing
            st isa Expr && st.head === :(=) && _kernel_assignment_names!(locals, st.args[1])
        else
            name, T, _ = target
            any(o -> o[1] === name, outputs) && throw(ArgumentError(
                "`@plate for` writes `$name[$index]` twice"))
            push!(outputs, (name, T, fresh(name)))
        end
    end
    isempty(outputs) && throw(ArgumentError(
        "`@plate for $index in …` must write at least one output, " *
        "`out[$index] = …`, at the loop index"))

    mapping = Dict{Symbol,Symbol}(name => local_name for (name, _, local_name) in outputs)
    output_names = Set(first.(outputs))
    # An array the cell also reads whole is captured; its reads at the index
    # are gathers from that capture rather than a second, zipped operand.
    whole = Set{Symbol}()
    foreach(st -> _kernel_loop_whole_reads!(whole, st, index), statements)
    union!(locals, whole)
    zipped = Symbol[]
    cell = Any[]
    for st in statements
        target = _kernel_plate_loop_output(st, index)
        if target === nothing
            push!(cell, _kernel_loop_rewrite(st, index, mapping, zipped, locals, fresh))
        else
            name, T, rhs = target
            local_name = mapping[name]
            rewritten = _kernel_loop_rewrite(rhs, index, mapping, zipped, locals, fresh)
            lhs = T === nothing ? local_name : Expr(:(::), local_name, T)
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
    lambda_formals = Expr(:tuple, formals...)
    result = Any[]
    for (name, _, local_name) in outputs
        cell_body = Expr(:block, (line === nothing ? () : (line,))..., cell..., local_name)
        push!(result, Expr(:(=), name,
                           Expr(:do, plate_call, Expr(:(->), lambda_formals, cell_body))))
    end
    result
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
values through literal lags `a[t - k]` (`1 ≤ k ≤ m`) and the current value
`a[t]` after its write. Other values are read as closures. The result `a` is the
whole trajectory `[a[1], …, a[T]]`. One carried array with one seed prepares the
same kernel as `a = scan(2:T; init = seed, include_init = true) do a_prev, t …
end`; several carried arrays are folded by one scan each.
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
    for name in carried
        steps = gensym(Symbol(name, :_steps))
        call = Expr(:call, scan_ref, Expr(:parameters, Expr(:kw, :init, init)), domain)
        step_body = Expr(:block, lines..., step..., Expr(:tuple, next, step_locals[name]))
        push!(result, Expr(:(=), steps,
                           Expr(:do, call, Expr(:(->), Expr(:tuple, carry, index), step_body))))
        push!(result, Expr(:(=), name, Expr(:call, GlobalRef(Base, :vcat),
                                            Expr(:vect, (seed_names[(name, k)] for k in 1:depth)...),
                                            steps)))
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
# this step already wrote. Any other read of a carried array is refused.
function _kernel_scan_rewrite(ex, index, carried, depth, lag, step_locals, written)
    ex isa Expr || return ex
    ex.head === :quote && return ex
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
        end
        throw(ArgumentError(
            "`@scan` reads `$(ex)`: a carried array is read at `$index` after its write " *
            "or at a literal lag `$name[$index - k]` with 1 ≤ k ≤ $depth"))
    end
    _kernel_loop_rebinds(ex, index) && return ex
    Expr(ex.head, (_kernel_scan_rewrite(arg, index, carried, depth, lag, step_locals, written)
                   for arg in ex.args)...)
end
