# Scheduling of an existing generated reverse over bound contiguous-index runs.
# The preserved flat arm supplies the derivative graph; the primal keeps its
# original grouped reduction. No derivative formulas are authored here.
struct _PRIndexRun
    metadata::Symbol
    operands::Vector{Symbol}
    guard::Expr
end

function _pr_index_run_names(x)
    x isa GlobalRef && x.mod === Base && return x.name
    x isa Expr || return x
    Meta.isexpr(x, :., 2) && x.args[1] === :Base && x.args[2] isa QuoteNode &&
        return x.args[2].value
    Expr(x.head, (_pr_index_run_names(a) for a in x.args if !(a isa LineNumberNode))...)
end

# Recognize only the compiler's run/flat pair, including its scalar-total guard,
# seed and final accumulation. An unrelated loop or partial sum stays subject
# to the ordinary generator's eligibility checks.
function _pr_index_run_arm(ex)
    Meta.isexpr(ex, :if, 3) || return nothing
    run = ex.args[2]
    Meta.isexpr(run, :for, 2) || return nothing
    binding = run.args[1]
    Meta.isexpr(binding, :(=), 2) || return nothing
    range = binding.args[2]
    Meta.isexpr(range, :call, 2) && _pr_index_run_names(range.args[1]) === :eachindex ||
        return nothing
    field = range.args[2]
    Meta.isexpr(field, :call, 3) && _pr_index_run_names(field.args[1]) === :getfield &&
        field.args[3] === 1 || return nothing
    metadata = field.args[2]
    metadata isa Symbol || return nothing
    flat = _index_run_simd_plate(ex.args[3])
    flat === nothing && return nothing
    axes, _, cell = flat
    additions = Expr[]
    _index_run_visit(cell) do node
        Meta.isexpr(node, :(=), 2) &&
            _index_run_call(node.args[2], GlobalRef(@__MODULE__, :_plate_total_add), 2) &&
            node.args[2].args[2] === node.args[1] && push!(additions, node)
    end
    length(additions) == 1 || return nothing
    total = only(additions).args[1]
    total isa Symbol || return nothing
    guard = :(length($axes) == 1 && Base.only($axes) == Base.OneTo(getfield($metadata, 3)) &&
              $total isa Number)
    condition = _pr_index_run_names(ex.args[1])
    (condition == _pr_index_run_names(guard) ||
     condition == _pr_index_run_names(:(getfield($metadata, 4) && $guard))) || return nothing
    seeds = Symbol[]
    finals = Any[]
    low = Ref{Union{Nothing,Symbol}}(nothing)
    _index_run_visit(run.args[2]) do node
        Meta.isexpr(node, :(=), 2) || return
        lhs, rhs = node.args
        _index_run_call(rhs, GlobalRef(@__MODULE__, :_plate_run_seed), 1) &&
            rhs.args[2] === total && lhs isa Symbol && push!(seeds, lhs)
        lhs === total && _index_run_call(rhs, GlobalRef(@__MODULE__, :_plate_total_add), 2) &&
            rhs.args[2] === total && push!(finals, rhs.args[3])
        if lhs isa Symbol && Meta.isexpr(rhs, :ref, 2)
            f = rhs.args[1]
            Meta.isexpr(f, :call, 3) && _pr_index_run_names(f.args[1]) === :getfield &&
                f.args[2] === metadata && f.args[3] === 1 && (low[] = lhs)
        end
    end
    length(seeds) == length(finals) == 1 && only(seeds) === only(finals) || return nothing
    low[] === nothing && return nothing
    operands = Symbol[]
    _index_run_visit(run.args[2]) do node
        _index_run_read(node) || return
        operand, index = node.args[2:3]
        operand isa Symbol && Meta.isexpr(index, :call, 2) &&
            _pr_is_ref(index.args[1], Base, :CartesianIndex) && index.args[2] === low[] || return
        operand in operands || push!(operands, operand)
    end
    isempty(operands) && return nothing
    loop = _pr_cell_loop(ex.args[3])
    loop === nothing && return nothing
    (ex.args[3], loop[2], _PRIndexRun(metadata, operands, ex.args[1]))
end

# A reverse scheduled by runs reads the run metadata, a tuple holding the two
# index vectors, inside the backend's rule for the plate. A backend extension
# stores a reason here when its rule boundary cannot hand a rule such an
# unchanged tuple. The plate then keeps the backend's ordinary reverse, as it
# did before runs were scheduled, and `plate_reverse_report` gives the reason.
const _PR_INDEX_RUN_REFUSAL = Ref{Union{Nothing,String}}(nothing)

function _pr_index_run_source(ast)
    runs = Dict{Symbol,_PRIndexRun}()
    source = _index_run_replace(ast) do ex
        arm = _pr_index_run_arm(ex)
        arm === nothing && return ex
        refusal = _PR_INDEX_RUN_REFUSAL[]
        refusal === nothing || _pr_fail(refusal)
        flat, linear, run = arm
        haskey(runs, linear) && _pr_fail("ambiguous index-run cell binding")
        runs[linear] = run
        deepcopy(flat)
    end
    source, runs
end

@inline _pr_run_overlap(a::AbstractArray, b::AbstractArray) = Base.mightalias(a, b)
@inline _pr_run_overlap(a, b) = false
@inline _pr_run_shadow_overlap(a, b) = _pr_run_overlap(a, b)
@inline function _pr_run_shadow_overlap(a::SubArray, b::SubArray)
    _pr_gather_value(typeof(a)) && _pr_gather_value(typeof(b)) ||
        return Base.mightalias(a, b)
    # The one-pass adapter gathers through disjoint slices of one tape. Base's
    # outer-view check sees the shared tape even when those slices are disjoint.
    # Writes stay in the parents; index arrays are read and must not overlap
    # either write region. Sharing an index array alone is harmless.
    Base.mightalias(parent(a), parent(b)) ||
        Base.mightalias(parent(a), parentindices(b)[1]) ||
        Base.mightalias(parent(b), parentindices(a)[1])
end
@inline _pr_run_shadow_matches(::Nothing, value) = true
@inline _pr_run_shadow_matches(shadow, value) = false
@inline function _pr_run_shadow_matches(shadow::SubArray, value::SubArray)
    _pr_gather_value(typeof(shadow)) && _pr_gather_value(typeof(value)) &&
        parentindices(shadow)[1] === parentindices(value)[1]
end
@inline function _pr_run_disjoint(values::Tuple, shadows::Tuple)
    all(s -> all(v -> !_pr_run_overlap(s, v), values), shadows) || return false
    # Tuple arity is the program's structural operand count, never data size.
    all(i -> all(j -> i == j || !_pr_run_shadow_overlap(shadows[i], shadows[j]),
                 eachindex(shadows)), eachindex(shadows))
end

function _pr_index_run_schedule(reverse, runs, prepared, shadow, params, batched)
    isempty(runs) && return reverse
    _index_run_replace(reverse) do ex
        Meta.isexpr(ex, :for, 2) && Meta.isexpr(ex.args[1], :(=), 2) || return ex
        linear, cells = ex.args[1].args
        haskey(runs, linear) && cells isa Symbol && startswith(string(cells), "_pr_cells") ||
            return ex
        spec = runs[linear]
        all(op -> haskey(prepared, op), spec.operands) ||
            _pr_fail("unmapped index-run operand")
        metadata = spec.metadata
        # Names depend on compiler bindings, not on run counts or fresh gensyms.
        run, lo, hi = (Symbol(cells, suffix) for suffix in (:_run, :_run_lo, :_run_hi))
        cached = Dict(op => Symbol(cells, :_read_, i) for (i, op) in enumerate(spec.operands))
        affected = unique([prepared[op] for op in spec.operands if haskey(shadow, prepared[op])])
        sums = Dict(shadow[p] => Symbol(cells, :_sum_, i) for (i, p) in enumerate(affected))
        reads = Ref(0)
        used = Set{Symbol}()
        cell = _index_run_replace(ex.args[2]) do node
            if Meta.isexpr(node, :call, 3) && haskey(cached, node.args[2]) &&
                    (_index_run_read(node) || _pr_is_ref(node.args[1], @__MODULE__, :_pr_direct_read))
                reads[] += 1
                return cached[node.args[2]]
            elseif Meta.isexpr(node, :call, 4) &&
                    _pr_is_ref(node.args[1], @__MODULE__, :_pr_shadow_add!) &&
                    haskey(sums, node.args[2]) && node.args[3] === linear
                push!(used, node.args[2])
                sumvar = sums[node.args[2]]
                return :($sumvar = $sumvar + $(node.args[4]))
            end
            node
        end
        reads[] > 0 || _pr_fail("index-run reverse has no matching read")
        filter!(p -> shadow[p] in used, affected)
        inner = Expr(:macrocall, GlobalRef(Base, Symbol("@simd")), LineNumberNode(0),
                     Expr(:for, :($linear = $lo:$hi), cell))
        initial = Any[:($(cached[op]) = $(GlobalRef(Base.Broadcast, :_broadcast_getindex))(
            $op, Base.CartesianIndex($lo))) for op in spec.operands]
        append!(initial, [:($(sums[shadow[p]]) = 0.0) for p in affected])
        final = Any[:($(GlobalRef(@__MODULE__, :_pr_shadow_add!))(
            $(shadow[p]), $lo, $(sums[shadow[p]]))) for p in affected]
        runbody = quote
            $lo = @inbounds getfield($metadata, 1)[$run]
            $hi = @inbounds getfield($metadata, 2)[$run]
            $(initial...)
            $inner
            $(final...)
        end
        grouped = Expr(:for, :($run = eachindex(getfield($metadata, 1))), runbody)
        guard = deepcopy(spec.guard)
        for p in affected
            guard = :($guard && $(GlobalRef(@__MODULE__, :_pr_run_shadow_matches))($(shadow[p]), $p))
        end
        # Delayed writes need both the original gather map and disjoint buffers.
        # All other operands/maps retain their generated per-cell reads/writes.
        guard = :($guard && $(GlobalRef(@__MODULE__, :_pr_run_disjoint))(
            $(Expr(:tuple, params...)), $(Expr(:tuple, (shadow[p] for p in batched)...))))
        # Only the generated cell loop is replaced. Whole-buffer scaling and
        # value reconstruction remain after all runs, where the generator put them.
        Expr(:if, guard, grouped, ex)
    end
end
