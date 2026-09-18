# `@rkppl` authoring surface: StanBlocks-close model blocks lowering to
# data-free StructuralPlans.
#
# The shape mirrors StanBlocks `@slic` (block AST capture, `~` density
# statements, deterministic `=`, `model(; data...)` binding) under the
# standing constraints: Distributions.jl constructors (never Stan lowercase),
# immutable single-assignment top level, no control flow, no `target`,
# `@plate`/`@scan` reserved. Broadcasting is EXPLICIT (no implied
# vectorization anywhere): vector math is dotted (`mu = a .+ b .* x` —
# undotted scalar/vector `+` is a `MethodError` in Julia too), and vector
# responses use the Turing dotted tilde (`y .~ Normal.(mu, sigma)`).
# `=` binds values with Julia semantics; predictor locations inline
# deterministic structure and recover affine terms, so naming a
# subexpression never changes legality. Response-vs-parameter
# classification needs data names (a bare `lhs ~ Normal(...)` is
# shape-identical either way — the eight-schools indistinguishability), so
# the macro only captures; lowering runs at bind time with `data_names`.
# The BRM emitter targets the same `lower_rkppl(ast, data_names)` entry
# point with ASTs (no text seam).

"""
    SurfaceLoweringError <: Exception

Thrown by [`lower_rkppl`](@ref) and the `@rkppl` macro forms when a model
block is not slice-1 lowerable. Every message names the offending statement
and the fix; IR-level defects behind a well-formed surface are lowering bugs
and surface as [`ContractValidationError`](@ref) from the trailing
`validate_structure` call instead.
"""
struct SurfaceLoweringError <: Exception
    message::String
end
Base.showerror(io::IO, e::SurfaceLoweringError) =
    print(io, "SurfaceLoweringError: ", e.message)

_sfail(msg) = throw(SurfaceLoweringError(msg))

"""
    RKPPLModel

A captured `@rkppl` block (AST with call-site line numbers), not yet
lowered. Call it with data keywords to lower and bind:
`model(; y, x, g)::StructuralPlan` (bound). Mirrors StanBlocks `SlicModel`.
"""
struct RKPPLModel
    ast::Expr
end

function _check_body_shape(body)
    body isa Expr && body.head === :block && return nothing
    if Meta.isexpr(body, :(=)) && Meta.isexpr(first(body.args), :call)
        _sfail("named submodel definitions (`@rkppl sm(args...) = ...`) " *
               "need composition semantics (planned); the use-site form " *
               "`y ~ sm(...)` is locked for that slice")
    end
    return _sfail("@rkppl takes a `begin ... end` block " *
                  "(or caller-scope data plus a block)")
end

"""Capture a model block (§ surface.jl for the admitted vocabulary)."""
macro rkppl(body)
    _check_body_shape(body)
    return Expr(:call, RKPPLModel, Meta.quot(body))
end

"""Capture a model block and immediately lower+bind caller-scope data
(a `NamedTuple` or dict of columns)."""
macro rkppl(data, body)
    _check_body_shape(body)
    q = Meta.quot(body)
    return esc(:($(_bind_immediate)($(RKPPLModel)($q), $data)))
end

function (m::RKPPLModel)(; kwargs...)
    cols = Dict{Symbol,AbstractVector}()
    for (k, v) in kwargs
        cols[k] = _check_col(k, v)
    end
    return _bind_model(m, cols)
end

function _bind_immediate(m::RKPPLModel, data)
    cols = if data isa NamedTuple
        Dict{Symbol,AbstractVector}(k => _check_col(k, v) for (k, v) in pairs(data))
    elseif data isa AbstractDict
        Dict{Symbol,AbstractVector}(
            _dict_key(k) => _check_col(k, v) for (k, v) in data)
    else
        _sfail("@rkppl data must be a NamedTuple or dict of columns, " *
               "got $(typeof(data))")
    end
    return _bind_model(m, cols)
end

_dict_key(k::Symbol) = k
_dict_key(k::AbstractString) = Symbol(k)
_dict_key(k) = _sfail("data column keys must be Symbols, got $(repr(k))")

_check_col(k, v) = v isa AbstractVector ? v :
    _sfail("data column $k must be an AbstractVector, got $(typeof(v))")

function _bind_model(m::RKPPLModel, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(m.ast, keys(cols))
    return bind_data(plan, cols)
end

"""
    lower_rkppl(ast, data_names) -> StructuralPlan

Lower a captured `@rkppl` block AST to a data-free (unbound) plan.
`data_names` classifies every `~` / `.~` LHS: data under `.~` is a
response, non-data under `~` is a prior (sampled parameter or, when the
name sits in a predictor coefficient position, a population prior); the
crossed spellings fail closed (`~` is scalar-only, `.~` broadcasts over
data). Runs `validate_structure` before returning. The BRM emitter calls
this entry point directly with ASTs.
"""
function lower_rkppl(ast, data_names)::StructuralPlan
    data = Set{Symbol}()
    for n in data_names
        n isa Symbol || _sfail("data names must be Symbols, got $(repr(n))")
        push!(data, n)
    end
    ast isa Expr && ast.head === :block ||
        _sfail("lower_rkppl takes a `begin ... end` block AST")
    sample, det = _partition_statements(ast, data)
    detmap = Dict{Symbol,Any}(nm => rhs for (nm, rhs) in det)
    prior_names = Set{Symbol}(s.lhs for s in sample if s.lhs ∉ data)
    normal_priors = Set{Symbol}(s.lhs for s in sample
        if s.lhs ∉ data && _is_normal_call(s.rhs))
    # Shape every definition (data-free: data ⇒ vector, sampled ⇒ scalar,
    # det-refs recurse with memo; cycles error downstream), then
    # canonicalize each RHS in dependency order (Julia-valid undotted
    # scalar-array ops take dotted-canonical form; Julia-invalid vector
    # combinations fail here naming the definition). Everything downstream
    # sees canonical RHSs.
    detshape = _def_shapes(det, data, detmap)
    canonmap = Dict{Symbol,Any}()
    for nm in _det_topo_order(det, detmap)
        rhs = detmap[nm]
        _reject_unknown_calls("definition `$nm = $(repr(rhs))`", rhs)
        canonmap[nm] = _canonical_expr(rhs, data, detshape,
            "definition `$nm = $(repr(rhs))`")
    end
    # Structural definitions inline into predictors: anything transitively
    # referencing a coefficient candidate (Normal-priored or free name).
    # All other vector definitions stay symbolic as named locals.
    structural = _structural_defs(det, data, canonmap, normal_priors,
        prior_names)
    vecdefs = Set{Symbol}(nm for (nm, _) in det if detshape[nm] === :vector)
    taken = union(data, Set{Symbol}(nm for (nm, _) in det), prior_names)
    ctx = (; data, detmap = canonmap, prior_names, normal_priors, detshape,
        vecdefs, structural, absorbed = Set{Symbol}(), synth = Ref(0),
        synth_derived = VectorAssignmentSpec[], taken)
    responses = LikelihoodSpec[]
    predictors = PredictorSpec[]
    pred_idx = Dict{Symbol,Int}()
    coefuse = Dict{Symbol,Vector{Tuple{Symbol,Symbol,Int}}}()
    for s in sample
        if s.broadcast
            s.lhs in data || _sfail("`.~` broadcasts over a data column " *
                                    "— $(s.lhs) is not data (scalar " *
                                    "parameters use `~`)")
            push!(responses,
                _lower_response(s.lhs, s.rhs, s.range, ctx, predictors,
                    pred_idx, coefuse))
        elseif s.lhs in data
            _sfail("$(s.lhs) is data — vector responses broadcast with " *
                   "`.~` (`$(s.lhs) .~ Normal.(mu, sigma)`); `~` is " *
                   "scalar-only")
        end
    end
    priors = _lower_coefficient_priors(sample, coefuse, predictors)
    params, paramsyms = _lower_parameters(sample, coefuse, ctx)
    used_locs = Set{Symbol}(r.predictor for r in responses)
    skip = _absorbed_skip(det, canonmap, responses, paramsyms, ctx.absorbed,
        used_locs)
    assigns = AssignmentSpec[]
    derived = VectorAssignmentSpec[]
    for (nm, _) in det
        nm in skip && continue
        rhs = canonmap[nm]
        if detshape[nm] === :vector
            push!(derived, _lower_vector_assignment(nm, rhs, coefuse))
        else
            push!(assigns, _lower_assignment(nm, rhs, coefuse))
        end
    end
    append!(derived, ctx.synth_derived)
    for d in ctx.synth_derived
        for s in _value_symbols(d.expr)
            haskey(coefuse, s) && _sfail("$s is a predictor coefficient " *
                                        "and cannot appear in an extracted " *
                                        "column (predictor $(d.label))")
        end
    end
    plan = StructuralPlan(responses, predictors, priors, params, assigns,
        Dict{Symbol,AbstractVector}(), 0; derived = derived)
    validate_structure(plan)
    return plan
end

# Shape inference + canonicalization (data-free, Julia-truthful). Shapes:
# data columns are vectors, sampled/det names resolve by position and memo,
# dotted forms are vectors, reductions are scalars, undotted scalar-array
# combinations follow Julia exactly (`2*v`, `v*2`, `v/2`, `-v` are vectors;
# `a+x`, `x*z`, `s/x`, `x^2`, `x>1`, `log(x)` are `:invalid` — Julia
# `MethodError`s, reported with the dotted fix). Canonicalization rewrites
# the Julia-valid undotted scalar-array ops (`*`, `/`) to dotted-canonical
# form (Base implements them by broadcast — behavior-preserving); every
# other head passes through. Unknown call heads do not shape-route here
# (the vocabulary screen rejects them first); their shape follows their
# arguments so the downstream error names the function.
function _def_shapes(det, data::Set{Symbol}, detmap)
    memo = Dict{Symbol,Symbol}()
    for (nm, _) in det
        memo[nm] = _shape_of(detmap[nm], data, detmap, memo, Set{Symbol}())
    end
    return memo
end

function _shape_of(ex, data, detmap, memo, active::Set{Symbol})
    ex isa Symbol || return _shape_of_expr(ex, data, detmap, memo, active)
    ex in data && return :vector
    haskey(detmap, ex) || return :scalar
    haskey(memo, ex) && return memo[ex]
    ex in active && return :scalar  # cyclic: errors downstream
    push!(active, ex)
    sh = _shape_of(detmap[ex], data, detmap, memo, active)
    pop!(active)
    memo[ex] = sh
    return sh
end

function _shape_of_expr(ex, data, detmap, memo, active)
    ex isa LineNumberNode && return :scalar
    ex isa Expr || return :scalar
    head = ex.head
    head === :. && return :vector
    head === :ref && return :scalar
    head === :call || return :scalar  # exotic heads: downstream rejects
    isempty(ex.args) && return :scalar
    fn = ex.args[1]
    fn isa Symbol || return :scalar  # anonymous calls: downstream rejects
    fn in REDUCTION_FNS && return :scalar
    fn in ELEMENTWISE_OPS && return :vector
    argshapes = [_shape_of(a, data, detmap, memo, active)
        for a in ex.args[2:end]]
    return _shape_of_call(fn, argshapes)
end

# Single rule table for undotted `:call` shapes over argument shapes.
function _shape_of_call(fn::Symbol, argshapes::Vector{Symbol})
    :invalid in argshapes && return :invalid
    nvec = count(==(:vector), argshapes)
    if fn === :+ || fn === :-
        length(argshapes) == 1 && return only(argshapes)
        return nvec == 0 ? :scalar : :invalid
    elseif fn === :*
        length(argshapes) == 2 || return nvec == 0 ? :scalar : :invalid
        nvec == 0 && return :scalar
        nvec == 1 && return :vector
        return :invalid
    elseif fn === :/
        length(argshapes) == 2 || return nvec == 0 ? :scalar : :invalid
        nvec == 0 && return :scalar
        argshapes[1] === :vector && argshapes[2] === :scalar && return :vector
        return :invalid
    elseif fn === :(===) || fn === :(!==)
        return :scalar  # egal is defined over arrays (Bool result)
    elseif fn === :^ || _is_plain_comparison(fn) || fn === :ifelse ||
            fn in ASSIGNMENT_FNS
        return nvec == 0 ? :scalar : :invalid
    end
    return nvec == 0 ? :scalar : :vector  # unknown heads: follow the args
end

_is_plain_comparison(fn::Symbol) =
    fn === :< || fn === :> || fn === :(==) || fn === :(!=) ||
    fn === :(<=) || fn === :(>=)

function _canonical_expr(ex, data, detshape, where)
    ex isa Symbol && return ex
    ex isa Expr || return ex
    ex.head === :parameters && _sfail("$where takes positional " *
                                      "arguments only (no keywords)")
    ex.head === :call || return ex
    isempty(ex.args) && return ex
    fn = ex.args[1]
    fn isa Symbol || return ex
    fn in REDUCTION_FNS && return ex  # args validated downstream
    args = [_canonical_expr(a, data, detshape, where) for a in ex.args[2:end]]
    argshapes = [_canon_shape(a, data, detshape) for a in args]
    if :invalid in argshapes
        return Expr(ex.head, ex.args[1], args...)  # broken ref: raises at its own def
    end
    _shape_of_call(fn, argshapes) === :invalid &&
        _sfail(_julia_mismatch_msg(fn, where, ex))
    if (fn === :* || fn === :/) && length(args) == 2
        if fn === :* && (argshapes[1] === :vector) != (argshapes[2] === :vector)
            return Expr(:call, :.*, args...)
        elseif fn === :/ && argshapes[1] === :vector &&
                argshapes[2] === :scalar
            return Expr(:call, :./, args...)
        end
    end
    if fn === :- && length(args) == 1 && argshapes[1] === :vector
        # Unary minus over vectors has no derived-column vocabulary slot;
        # the dotted form is equivalent and admitted.
        return Expr(:call, :.-, args...)
    end
    return Expr(ex.head, ex.args[1], args...)
end

function _canon_shape(ex, data, detshape)
    ex isa Symbol || return _canon_shape_expr(ex, data, detshape)
    ex in data && return :vector
    return get(detshape, ex, :scalar)
end

function _canon_shape_expr(ex, data, detshape)
    ex isa Expr || return :scalar
    head = ex.head
    head === :. && return :vector
    head === :ref && return :scalar
    head === :call || return :scalar
    isempty(ex.args) && return :scalar
    fn = ex.args[1]
    fn isa Symbol || return :scalar
    fn in REDUCTION_FNS && return :scalar
    fn in ELEMENTWISE_OPS && return :vector
    return _shape_of_call(fn,
        [_canon_shape(a, data, detshape) for a in ex.args[2:end]])
end

function _julia_mismatch_msg(fn::Symbol, where, ex)
    fix = if fn === :+ || fn === :-
        "as in Julia, `$fn` over a vector needs dots " *
        "(`a .+ b .* x`); implied broadcasting is not in the surface"
    elseif fn === :* && length(ex.args) != 3
        "n-ary `*` with a vector operand does not lower — parenthesize " *
        "binary products"
    elseif fn === :*
        "as in Julia, `*` over two vectors is not elementwise — write " *
        "the interaction dotted (`x .* z`)"
    elseif fn === :/
        "as in Julia, `/` over a vector needs dots (`x ./ s`)"
    elseif fn === :^
        "as in Julia, `^` over a vector needs dots (`x .^ 2`)"
    elseif _is_plain_comparison(fn)
        "as in Julia, `$fn` over a vector needs dots (`x .$fn y`)"
    elseif fn === :ifelse
        "use elementwise `ifelse.(condition, x, y)`"
    else
        "as in Julia, `$fn` over a vector needs dots (`$fn.(x)`)"
    end
    return "$where combines vectors without dots: `$(repr(ex))` — $fix"
end

_is_normal_call(rhs) =
    rhs isa Expr && rhs.head === :call && !isempty(rhs.args) &&
    rhs.args[1] === :Normal

# Dependency order over deterministic definitions (callee before caller;
# cycles and unknown refs keep source order — cycles error downstream).
function _det_topo_order(det, detmap)
    detkeys = Set{Symbol}(nm for (nm, _) in det)
    order = Symbol[]
    done = Set{Symbol}()
    active = Set{Symbol}()
    function visit(nm)
        nm in done && return nothing
        nm in active && return nothing
        push!(active, nm)
        for s in _symbols_in(detmap[nm])
            s in detkeys && s != nm && visit(s)
        end
        pop!(active)
        push!(done, nm)
        push!(order, nm)
        return nothing
    end
    for (nm, _) in det
        visit(nm)
    end
    return order
end

# Known value-level call heads (excluded from free-name detection so a
# definition like `m = log` never reads as referencing data).
const _KNOWN_VALUE_FNS = union(Set{Symbol}(ASSIGNMENT_FNS),
    Set{Symbol}(ELEMENTWISE_OPS), Set{Symbol}(ELEMENTWISE_FNS),
    Set{Symbol}((:ifelse, :treatment)))

# Structural definitions: anything transitively referencing a coefficient
# candidate (a Normal-priored sampled name or a free name — data, det, and
# other sampled names excluded). Structural definitions inline into
# predictors; every other definition keeps its binding as a kernel local.
function _structural_defs(det, data, canonmap, normal_priors, prior_names)
    detkeys = Set{Symbol}(nm for (nm, _) in det)
    structural = Set{Symbol}()
    for (nm, _) in det
        refs = _value_symbols(canonmap[nm])
        if any(s -> s in normal_priors ||
                _is_free_name(s, data, detkeys, prior_names), refs)
            push!(structural, nm)
        end
    end
    changed = true
    while changed
        changed = false
        for (nm, _) in det
            nm in structural && continue
            if any(s -> s in structural, _value_symbols(canonmap[nm]))
                push!(structural, nm)
                changed = true
            end
        end
    end
    return structural
end

function _is_free_name(s::Symbol, data, detkeys, prior_names)
    s in data && return false
    s in detkeys && return false
    s in prior_names && return false
    s in _KNOWN_VALUE_FNS && return false
    return true
end

# Top-down unknown-head screen (vocabulary layer runs before shape and
# structure layers, so the error names the function, not its fallout).
# Factor references are skipped (the factor machinery owns that position).
# Heads that only lower inside `.~` responses, never as values (`exp` is
# excluded: undotted it is admitted scalar math, dotted admitted
# elementwise math — only the Poisson-link position is response-only, and
# that position never reaches this screen).
const _RESPONSE_ONLY_FNS =
    (:weighted, :truncated, :censored, :interval_censored, :logistic)
const _DIST_VALUE_FNS =
    (:Normal, :Cauchy, :Exponential, :Gamma, :LogNormal, :Beta,
        :InverseGamma, :Bernoulli, :Poisson, :HalfNormal, :HalfCauchy, :Flat)

function _reject_unknown_calls(where, rhs)
    rhs isa Expr || return nothing
    rhs.head === :ref && return nothing
    if rhs.head === :call && !isempty(rhs.args)
        fn = rhs.args[1]
        if fn isa Symbol && fn ∉ ELEMENTWISE_OPS && fn ∉ ASSIGNMENT_FNS
            startswith(string(fn), ".") && _sfail(
                "$where uses dotted operator `$fn`, which is not in " *
                "the slice-1 elementwise vocabulary")
            fn === :treatment && _sfail(
                "$where calls `treatment`, which only lowers as a " *
                "factor index (`c[treatment(g, ref)]`)")
            fn === :ifelse && _sfail(
                "$where calls undotted `ifelse` — use elementwise " *
                "`ifelse.(condition, x, y)`")
            fn in _RESPONSE_ONLY_FNS && _sfail(
                "$where calls `$fn`, which only lowers in `.~` " *
                "responses, not as a value")
            fn in _DIST_VALUE_FNS && _sfail(
                "$where calls `$fn`, which lowers only under `~`/`.~`, " *
                "not as a value")
            _sfail("$where calls `$fn`, which is not in the slice-1 " *
                   "value vocabulary — arbitrary Julia functions are " *
                   "planned (no-@deffun-ceremony direction) but need " *
                   "IR/contract growth")
        end
    elseif rhs.head === :.
        length(rhs.args) == 2 && rhs.args[1] isa Symbol &&
            rhs.args[2] isa Expr && rhs.args[2].head === :tuple ||
            return nothing  # malformed dotted: downstream rejects
        f = rhs.args[1]
        # (`exp.` is both the Poisson link and admitted elementwise math,
        # so only `logistic.` guides here.)
        f === :logistic && _sfail(
            "$where calls `logistic.`, which only lowers as a `.~` link " *
            "(`Bernoulli.(logistic.(eta))`)")
        f === :ifelse || f in ELEMENTWISE_FNS || _sfail(
            "$where calls `$f.`, which is not in the slice-1 value " *
            "vocabulary — arbitrary Julia functions are planned " *
            "(no-@deffun-ceremony direction) but need IR/contract growth")
    end
    for a in rhs.args
        _reject_unknown_calls(where, a)
    end
    return nothing
end

# Emission skip set: locations never emit; absorbed definitions emit only
# while a non-skipped definition (or a response scale / parameter
# argument) still names them. Fixpoint: skipping cascades through
# absorbed-only reference chains (chained intermediates vanish entirely).
function _absorbed_skip(det, canonmap, responses, paramsyms, absorbed,
        used_locs)
    skip = Set{Symbol}(used_locs)
    while true
        refs = Set{Symbol}(paramsyms)
        for r in responses
            r.scale isa Symbol && push!(refs, r.scale)
        end
        for (nm, _) in det
            nm in skip && continue
            union!(refs, _value_symbols(canonmap[nm]))
        end
        newskip = union(skip, Set{Symbol}(nm for (nm, _) in det
            if nm in absorbed && nm ∉ refs))
        newskip == skip && return skip
        skip = newskip
    end
end

# Top-level statements: skip line numbers and one leading docstring-to-be
# (ignored in slice 1); everything else must be an Expr.
function _partition_statements(ast::Expr, data::Set{Symbol})
    sample = SampleStmt[]
    det = Pair{Symbol,Any}[]
    seen = Set{Symbol}()
    seelines = Dict{Symbol,Int}()
    seen_doc = false
    line = 0
    for arg in ast.args
        if arg isa LineNumberNode
            line = arg.line
            continue
        end
        if arg isa String && !seen_doc && isempty(sample) && isempty(det)
            seen_doc = true
            continue
        end
        arg isa Expr || _sfail("stray literal $(repr(arg)) at model level " *
                               "(only `~`, `.~`, `=` and reserved macros lower)")
        arg.head === :block &&
            _sfail("nested `begin` blocks do not lower — flatten the block")
        st = _unwrap_trivia(arg)
        if _is_sample(st) || _is_broadcast_sample(st)
            bc = _is_broadcast_sample(st)
            tilde = bc ? "`.~`" : "`~`"
            lhs, rng = _sample_lhs(st.args[2], bc, tilde)
            _claim!(seen, seelines, lhs, line)
            _reject_target(st.args[3], lhs)
            push!(sample, SampleStmt(lhs, st.args[3], bc, rng))
        elseif st.head === :(=) && length(st.args) == 2 && st.args[1] isa Symbol
            lhs = st.args[1]
            lhs === :target && _sfail("no `target` in rkppl models " *
                                      "(density comes only from `~`)")
            lhs in data && _sfail("$lhs is bound data and cannot be redefined")
            _claim!(seen, seelines, lhs, line)
            _reject_target(st.args[2], lhs)
            push!(det, lhs => st.args[2])
        else
            _reject_statement(st)
        end
    end
    return sample, det
end

function _claim!(seen::Set{Symbol}, seelines::Dict{Symbol,Int}, nm::Symbol,
        line::Int)
    if nm in seen
        first = get(seelines, nm, 0)
        at = first > 0 ? " (first at line $first)" : ""
        _sfail("single assignment: $nm is defined twice at model level$at")
    end
    push!(seen, nm)
    seelines[nm] = line
    return nothing
end

_is_sample(st::Expr) =
    st.head === :call && length(st.args) == 3 && st.args[1] === :~

_is_broadcast_sample(st::Expr) =
    st.head === :call && length(st.args) == 3 && st.args[1] === :.~

# Sampling-statement LHS: a bare Symbol, or (`.~` only) a one-dimensional
# range ref `y[R]`. Returns `(column, range)` with `range === nothing`
# for bare and self-covering (`eachindex`/`axes`) forms.
_sample_lhs(lhs::Symbol, bc, tilde) = (lhs, nothing)
function _sample_lhs(lhs, bc, tilde)
    lhs isa Expr || _sfail("$tilde left-hand side must be a bare Symbol " *
                           "or a range ref (`y[1:N]`, `y[eachindex(y)]`), " *
                           "got $(repr(lhs))")
    lhs.head === :. && _sfail("dotted left-hand side $(repr(lhs)) does " *
                              "not lower (nested targets are out of scope)")
    if lhs.head !== :ref || length(lhs.args) != 2 || !(lhs.args[1] isa Symbol)
        _sfail("$tilde left-hand side must be a bare Symbol or a " *
               "one-dimensional range ref (`y[1:N]`), got $(repr(lhs))")
    end
    bc || _sfail("sliced response `$(lhs.args[1])[...]` is a vector — " *
                 "use `.~`, not `~`")
    col = lhs.args[1]
    return col, _lower_lhs_range(col, lhs.args[2])
end

function _lower_lhs_range(col::Symbol, r)
    # Literal `1:N`: structural cover check now (start 1, non-empty);
    # `N == n_obs` is verified at bind (the range rides the plan).
    if r isa Expr && r.head === :call && length(r.args) == 3 && r.args[1] === :(:)
        lo, hi = r.args[2], r.args[3]
        lo === 1 || _sfail("response $col range must start at 1 " *
                           "(got $(repr(r))) — ranges cover eachindex exactly")
        hi isa Integer || _sfail("response $col range endpoint is " *
                                 "unbound ($(repr(hi))) — no `n` is bound in " *
                                 "the surface; write `eachindex($col)` or a " *
                                 "literal `1:N`")
        hi >= 1 || _sfail("response $col range $(repr(r)) is empty")
        return UnitRange(1, Int(hi))
    end
    # Self-covering forms: the column's own full index set, by construction.
    if r isa Expr && r.head === :call && !isempty(r.args) && r.args[1] === :eachindex
        length(r.args) == 2 && r.args[2] isa Symbol || _sfail(
            "response $col range takes `eachindex($col)` — " *
            "got $(repr(r))")
        r.args[2] === col || _sfail("response $col range covers " *
                                    "$(r.args[2]), not $col — ranges cover " *
                                    "their own column exactly " *
                                    "(`eachindex($col)`)")
        return nothing
    end
    if r isa Expr && r.head === :call && !isempty(r.args) && r.args[1] === :axes
        length(r.args) == 3 && r.args[2] isa Symbol && r.args[3] == 1 || _sfail(
            "response $col range takes `axes($col, 1)` — got $(repr(r))")
        r.args[2] === col || _sfail("response $col range covers " *
                                    "$(r.args[2]), not $col (`axes($col, 1)`)")
        return nothing
    end
    r === :(:) && _sfail("whole-column `$col[:]` is not admitted — " *
                            "write the range out (`$col[eachindex($col)]`)")
    (r isa Symbol || r isa Integer) &&
        _sfail("scalar cell index `$col[$(r)]` at top level does not " *
               "lower — per-cell refs live in `@plate` (slice B); " *
               "broadcast with `.~` or `eachindex`")
    return _sfail("response $col range must be `1:N`, `eachindex($col)`, " *
                  "or `axes($col, 1)` — got $(repr(r))")
end

"""One `~` / `.~` statement: scalar (`~`) or elementwise (`.~`) density.
`range` carries a literal `y[1:N]` response range (`nothing` = whole
column: bare LHS, `eachindex`, `axes`)."""
struct SampleStmt
    lhs::Symbol
    rhs::Any
    broadcast::Bool
    range::Union{Nothing,UnitRange{Int}}
end
SampleStmt(lhs::Symbol, rhs, broadcast::Bool) =
    SampleStmt(lhs, rhs, broadcast, nothing)

_is_doc_macro(m) =
    m === Symbol("@doc") || (m isa GlobalRef && m.name === Symbol("@doc"))

function _unwrap_trivia(st::Expr)
    while st.head === :macrocall
        m = st.args[1]
        if _is_doc_macro(m)
            # A leading string before a statement parses as `@doc "str" stmt`;
            # docstrings are unsupported in slice 1, so drop and lower the rest.
            length(st.args) >= 4 && st.args[3] isa String &&
                st.args[end] isa Expr ||
                _sfail("only `@doc \"...\" <statement>` docstring form " *
                       "unwraps here")
            st = st.args[end]
            continue
        end
        (m === Symbol("@plate") || m === Symbol("@scan")) && _sfail(
            "$m needs per-cell/sequential IR support beyond slice 1 " *
            "(shapes copied from StanBlocks; vector responses broadcast " *
            "with `.~` — `y .~ Normal.(mu, sigma)` over data)")
        m === Symbol("@.") && _sfail("explicit broadcast (`@.`) does not " *
                                     "lower — write the dots out " *
                                     "(`mu = a .+ b .* x`)")
        (m in (Symbol("@views"), Symbol("@inbounds"), Symbol("@simd"),
            Symbol("@fastmath"))) || _sfail("cannot expand macro $m at " *
                                            "lowering (user-macro transparency is a planned gap)")
        length(st.args) == 3 && st.args[3] isa Expr ||
            _sfail("$m wraps a non-statement here")
        st = st.args[3]
    end
    return st
end

function _reject_target(rhs, lhs)
    _symbols_in(rhs, Set{Symbol}([:target])) &&
        _sfail("no `target` in rkppl models (density comes only from " *
               "`~`/`.~`; statement for $lhs mentions `target`)")
    return nothing
end

# Value-position symbols (call heads and dotted function names excluded):
# coefficient-discipline scans must not mistake `log` in `log.(z)` for a
# coefficient named `log`.
function _value_symbols(ex)
    out = Set{Symbol}()
    _value_symbols!(out, ex)
    return out
end
function _value_symbols!(out::Set{Symbol}, ex)
    ex isa Symbol && return push!(out, ex)
    ex isa Expr || return nothing
    if ex.head === :call && !isempty(ex.args)
        for a in ex.args[2:end]
            _value_symbols!(out, a)
        end
        return nothing
    end
    if ex.head === :.
        length(ex.args) == 2 && ex.args[2] isa Expr &&
            ex.args[2].head === :tuple || return nothing
        for a in ex.args[2].args
            _value_symbols!(out, a)
        end
        return nothing
    end
    for a in ex.args
        _value_symbols!(out, a)
    end
    return nothing
end

# Collect symbols; with `want` nonempty, short-circuit true on first hit.
function _symbols_in(ex, want::Set{Symbol})
    ex isa Symbol && return ex in want
    ex isa Expr || return false
    for a in ex.args
        _symbols_in(a, want) && return true
    end
    return false
end
function _symbols_in(ex::Union{Expr,Symbol,Real})
    out = Set{Symbol}()
    _symbols_collect!(out, ex)
    return out
end
function _symbols_collect!(out::Set{Symbol}, ex)
    ex isa Symbol && return push!(out, ex)
    ex isa Expr || return nothing
    for a in ex.args
        _symbols_collect!(out, a)
    end
    return nothing
end

function _reject_statement(st::Expr)
    head = st.head
    if head === :call && !isempty(st.args) && st.args[1] === :~
        return _sfail("`~` takes `lhs ~ distribution`, got $(repr(st))")
    elseif head === :call && !isempty(st.args) && st.args[1] === :.~
        return _sfail("`.~` takes `lhs .~ distribution`, got $(repr(st))")
    elseif head === :call
        fn = isempty(st.args) ? "?" : repr(first(st.args))
        return _sfail("bare call `$fn(...)` at model level does nothing — " *
                      "did you mean `y = ...`, `y ~ ...` or `y .~ ...`?")
    elseif head === :(=)
        return _sfail("model assignment binds a bare Symbol only, got " *
                      "$(repr(st)) (index/function assignment needs the " *
                      "arbitrary-Julia-function direction — planned)")
    elseif head in (:for, :while)
        return _sfail("control flow is not allowed at model level " *
                      "(StanBlocks rule) — use an elementwise form; " *
                      "deterministic-function escape is planned")
    elseif head in (:if, :elseif, :comprehension, :generator,
        :typed_comprehension, :&&, :||)
        return _sfail("branching/comprehensions are not allowed at model " *
                      "level (StanBlocks rule) — use an elementwise form")
    elseif head === :(::)
        return _sfail("type/shape annotations (`$(repr(st))`) are a later " *
                      "slice (prior-predictive shapes) — write bare names")
    elseif head in (:+=, :-=, :*=, :/=)
        return _sfail("no mutating assignment at model level " *
                      "(everything top-level is immutable)")
    elseif head in (:function, :macro, :return, :local, :global, :struct,
        :module, :import, :using, :export)
        return _sfail("`$head` does not lower at model level")
    end
    return _sfail("unsupported model statement $(repr(st)) " *
                  "(only `~`, `.~`, `=` and reserved macros lower)")
end

# Plain positional call args (keyword calls never lower: positional only).
function _plain_args(rhs::Expr, what)
    for a in rhs.args[2:end]
        a isa Expr && a.head === :parameters &&
            _sfail("$what takes positional arguments only (no keywords)")
    end
    return rhs.args[2:end]
end

function _lower_response(lhs, rhs, range, ctx, predictors, pred_idx, coefuse)
    call = _dot2call_response(lhs, rhs)
    weights, call = _peel_weighted(lhs, call, ctx)
    evidence, call = _peel_evidence(lhs, call, ctx)
    family, lik_link, pred_link, loc, scale =
        _lower_response_base(lhs, call, ctx)
    pname = _lower_location(lhs, loc, pred_link, ctx, predictors, pred_idx,
        coefuse)
    return LikelihoodSpec(family, lik_link, lhs, pname, scale, weights,
        evidence, Symbol(lhs, "_resp"), range)
end

# `.~` takes a dotted distribution object (`Normal.(mu, sigma)`); convert
# the distribution spine to call form and reuse the peeling machinery
# (which reports the same object-form errors, now against dotted input).
# Only spine positions convert (nested objects, links); locations, scales,
# bounds, and weights pass through untouched (inline predictor dots are
# the analysis's business, not the peeler's).
const _DOT_WRAPPERS = (:weighted, :truncated, :censored, :interval_censored)

function _dot2call_response(lhs, rhs)
    rhs isa Expr && rhs.head === :. ||
        return _dot2call_object_error(lhs, rhs)
    length(rhs.args) == 2 && rhs.args[1] isa Symbol &&
        rhs.args[2] isa Expr && rhs.args[2].head === :tuple ||
        _sfail("response $lhs: malformed dotted object $(repr(rhs)) " *
               "(`Normal.(mu, sigma)` — no field access, no keywords)")
    targs = rhs.args[2].args
    any(a -> a isa Expr && a.head === :parameters, targs) && _sfail(
        "response $lhs: dotted objects take positional arguments " *
        "only (no keywords)")
    f = rhs.args[1]
    return Expr(:call, f, _dot2call_spine_args(lhs, f, targs)...)
end

function _dot2call_object_error(lhs, rhs)
    if rhs isa Expr && rhs.head === :call && !isempty(rhs.args) &&
            rhs.args[1] isa Symbol && rhs.args[1] in
            (:Normal, :Bernoulli, :Poisson, :weighted, :truncated, :censored,
                :interval_censored)
        _sfail("response $lhs: broadcast the object " *
               "(`$(rhs.args[1]).(...)` — `.~` is elementwise)")
    end
    return _sfail("response $lhs needs a dotted distribution object " *
                  "(`Normal.(mu, sigma)`), got $(repr(rhs))")
end

function _dot2call_spine_args(lhs, f, targs)
    return Any[_dot2call_spine_arg(lhs, f, i, a)
        for (i, a) in enumerate(targs)]
end

function _dot2call_spine_arg(lhs, f, i, a)
    if f in _DOT_WRAPPERS && i == 1
        return _dot2call_nested_object(lhs, a)
    elseif (f === :Bernoulli || f === :Poisson) && i == 1
        return _dot2call_nested_link(lhs, a, f)
    end
    return a
end

function _dot2call_nested_object(lhs, a)
    a isa Expr && a.head === :. && length(a.args) == 2 &&
        a.args[1] isa Symbol && a.args[2] isa Expr &&
        a.args[2].head === :tuple ||
        _sfail("response $lhs: broadcast the object " *
               "(`Normal.(...)` — `.~` is elementwise at every level)")
    targs = a.args[2].args
    any(x -> x isa Expr && x.head === :parameters, targs) && _sfail(
        "response $lhs: dotted objects take positional arguments " *
        "only (no keywords)")
    f = a.args[1]
    return Expr(:call, f, _dot2call_spine_args(lhs, f, targs)...)
end

function _dot2call_nested_link(lhs, a, base)
    want = base === :Bernoulli ? :logistic : :exp
    a isa Expr && a.head === :. && length(a.args) == 2 &&
        a.args[1] isa Symbol && a.args[2] isa Expr &&
        a.args[2].head === :tuple ||
        _sfail("response $lhs: broadcast the link " *
               "(`$want.(eta)` — `.~` is elementwise at every level)")
    targs = a.args[2].args
    any(x -> x isa Expr && x.head === :parameters, targs) && _sfail(
        "response $lhs: dotted objects take positional arguments " *
        "only (no keywords)")
    return Expr(:call, a.args[1], targs...)
end

function _peel_weighted(lhs, rhs::Expr, ctx)
    rhs.args[1] === :weighted || return nothing, rhs
    args = _plain_args(rhs, "`weighted`")
    length(args) == 2 || _sfail("`weighted` takes the object form " *
                                "`weighted.(Normal.(mu, sigma), w)`")
    dist, w = args
    dist isa Expr && dist.head === :call || _sfail(
        "`weighted` wraps a distribution object " *
        "(`weighted.(Normal.(mu, sigma), w)`), got $(repr(dist))")
    w isa Symbol && w in ctx.vecdefs && _sfail(
        "`weighted` weights column $w is derived — slice-1 binds weights " *
        "raw (derived weights need shape metadata — planned)")
    w isa Symbol && w in ctx.data || _sfail("`weighted` weights must be a " *
                                            "bare data column, got $(repr(w))")
    return w, dist
end

function _peel_evidence(lhs, rhs::Expr, ctx)
    head = rhs.args[1]
    head === :truncated || head === :censored ||
        head === :interval_censored || return ResponseEvidence(:none, nothing, nothing), rhs
    if head === :interval_censored
        # Object + upper only: the response itself IS the lower endpoint
        # (the IR carries lower = nothing), so no `lo` argument exists.
        args = _plain_args(rhs, "`interval_censored`")
        length(args) == 2 || _sfail("`interval_censored` takes " *
                                    "`interval_censored.(Normal.(mu, sigma), hi)`")
        dist, hi = args
        dist isa Expr && dist.head === :call || _sfail(
            "`interval_censored` wraps a distribution object, got " *
            "$(repr(dist))")
        ev = ResponseEvidence(:interval_censored, nothing,
            _bound(lhs, hi, :upper, ctx))
        return ev, dist
    end
    args = _plain_args(rhs, "`$head`")
    length(args) == 3 || _sfail("use the Distributions.jl object form " *
                                "`$head.(Normal.(...), lo, hi)`, got $(repr(rhs))")
    obj, lo, hi = args
    obj isa Expr && obj.head === :call || _sfail(
        "`$head` wraps a distribution object " *
        "(`$head.(Normal.(...), lo, hi)`), got $(repr(obj))")
    ev = ResponseEvidence(head, _bound(lhs, lo, :lower, ctx),
        _bound(lhs, hi, :upper, ctx))
    return ev, obj
end

# Bounds are literals or data columns; ±Inf normalizes to a missing side
# (nothing): `truncated(d, -Inf, hi)` is the Distributions.jl upper-only
# spelling and the IR carries one-sided bounds as nothing. Crossed
# infinities are degenerate. Both the `-Inf` Symbol/call spellings and
# actual ±Inf Reals (emitter-built ASTs) normalize.
function _bound(lhs, b, side::Symbol, ctx)
    if b isa Real
        isinf(b) || return b
        return _bound_infinite(lhs, b, side)
    end
    b === :Inf && return _bound_infinite(lhs, Inf, side)
    if b isa Expr && b.head === :call && length(b.args) == 2 &&
            b.args[1] === :- && b.args[2] === :Inf
        return _bound_infinite(lhs, -Inf, side)
    end
    b isa Symbol && b in ctx.data && return b
    b isa Symbol && b in ctx.vecdefs && _sfail(
        "response $lhs bound $b is a derived column — slice-1 binds " *
        "evidence bounds raw (derived bounds need shape metadata — planned)")
    return _sfail("response $lhs bound $(repr(b)) must be a literal or a " *
                  "data column")
end

function _bound_infinite(lhs, v::Real, side::Symbol)
    if (side === :lower && v < 0) || (side === :upper && v > 0)
        return nothing
    end
    return _sfail("response $lhs $side bound " *
                  "$(v > 0 ? "+Inf" : "-Inf") is degenerate (one-sided " *
                  "$side is $(side === :lower ? "-Inf" : "+Inf"))")
end

const _RESPONSE_BASE_MSG =
    "response distribution must be `Normal.(mu, sigma)`, " *
    "`Bernoulli.(logistic.(eta))` or `Poisson.(exp.(eta))`"

function _lower_response_base(lhs, rhs::Expr, ctx)
    rhs.head === :call || _sfail("response $lhs: $_RESPONSE_BASE_MSG; " *
                                 "got $(repr(rhs))")
    fam = rhs.args[1]
    fam === :weighted &&
        _sfail("`weighted.(...)` goes outermost: " *
               "`y .~ weighted.(Normal.(mu, sigma), w)`")
    fam in (:Normal, :Bernoulli, :Poisson) ||
        return _lower_response_base_error(lhs, rhs, fam)
    args = _plain_args(rhs, "`$fam`")
    if fam === :Normal
        length(args) == 2 || _sfail("response $lhs: `Normal` takes " *
                                    "`Normal.(mu, sigma)`")
        return GaussianFam, IdentityLink, IdentityLink, args[1],
        _lower_scale(lhs, args[2], ctx)
    elseif fam === :Bernoulli
        length(args) == 1 || _sfail("response $lhs: `Bernoulli` takes " *
                                    "`Bernoulli.(logistic.(eta))`")
        return BernoulliLogitFam, LogitLink, IdentityLink,
        _lower_link_arg(lhs, args[1], :logistic), nothing
    else
        length(args) == 1 || _sfail("response $lhs: `Poisson` takes " *
                                    "`Poisson.(exp.(eta))`")
        return PoissonLogFam, LogLink, LogLink,
        _lower_link_arg(lhs, args[1], :exp), nothing
    end
end

function _lower_response_base_error(lhs, rhs, fam)
    fam in (:normal, :bernoulli, :poisson) && _sfail(
        "response $lhs: use Distributions.jl constructors " *
        "(`Normal`, not `normal`)")
    fam === :BernoulliLogit && _sfail("response $lhs: write " *
                                      "`Bernoulli.(logistic.(eta))`")
    fam === :PoissonLog && _sfail("response $lhs: write " *
                                  "`Poisson.(exp.(eta))`")
    return _sfail("response $lhs: unknown distribution `$(repr(fam))` " *
                  "(admitted: Normal, Bernoulli, Poisson). " *
                  "If `$fam` is a submodel, `y ~ sm(...)` calls need " *
                  "composition semantics (planned).")
end

function _lower_link_arg(lhs, arg, wrap)
    arg isa Expr && arg.head === :call && !isempty(arg.args) &&
        arg.args[1] === wrap ||
        _sfail("response $lhs: write `$wrap` around the predictor " *
               "(`Bernoulli.(logistic.(eta))` / `Poisson.(exp.(eta))`)")
    args = _plain_args(arg, "`$wrap`")
    length(args) == 1 ||
        _sfail("response $lhs: `$wrap` takes exactly the predictor")
    return args[1]
end

function _lower_scale(lhs, s, ctx)
    s isa Real && return s
    s === :Inf && return Inf
    if s isa Symbol
        (s in ctx.data || s in ctx.vecdefs) && _sfail(
            "response $lhs scale $s varies by observation — " *
            "per-observation scales need plate plumbing (planned)")
        return s
    end
    return _sfail("response $lhs scale must be a bare parameter/assignment " *
                  "name or a literal (bind expressions via an assignment " *
                  "first), got $(repr(s))")
end

function _lower_location(lhs, loc, pred_link, ctx, predictors, pred_idx,
        coefuse)
    if loc isa Symbol
        haskey(ctx.detmap, loc) ||
            return _lower_location_symbol_error(lhs, loc, ctx)
        pname = loc
        if haskey(pred_idx, pname)
            pred = predictors[pred_idx[pname]]
            pred.link === pred_link || _sfail(
                "predictor $pname is shared by responses needing links " *
                "$(pred.link) and $pred_link — one link per predictor")
            return pname
        end
        terms, uses = _analyze_predictor(loc, ctx.detmap[loc], ctx, lhs)
    elseif loc isa Number
        _sfail("response $lhs location is a literal — use an intercept-only " *
               "predictor (`eta = a`)")
    else
        pname = Symbol(lhs, "_eta")
        haskey(ctx.detmap, pname) && _sfail(
            "derived predictor name $pname collides with your definition — " *
            "rename yours")
        terms, uses = _analyze_predictor(pname, loc, ctx, lhs)
    end
    _record_coefuses!(coefuse, pname, uses, lhs)
    push!(predictors, PredictorSpec(pname, pred_link, terms, pname))
    pred_idx[pname] = length(predictors)
    return pname
end

function _lower_location_symbol_error(lhs, loc, ctx)
    loc in ctx.data && _sfail("response $lhs location is the data column " *
                              "$loc — locations must be predictors with " *
                              "estimated coefficients (wrap: " *
                              "`eta = a .+ b .* $loc`)")
    loc in ctx.prior_names && _sfail(
        "response $lhs location is the bare parameter $loc " *
        "(per-observation latents need per-cell IR support — planned; " *
        "population-GLM responses take predictors)")
    return _sfail("response $lhs location $loc is not a predictor " *
                  "definition (`$loc = ...` affine in data)")
end

function _record_coefuses!(coefuse, pname, uses, lhs)
    for (name, addr, sign) in uses
        bucket = get!(coefuse, name, Tuple{Symbol,Symbol,Int}[])
        for (p2, _, _) in bucket
            p2 === pname && _sfail("response $lhs: coefficient $name is " *
                                   "used twice in predictor $pname")
        end
        push!(bucket, (pname, addr, sign))
    end
    return nothing
end

# Predictor analysis: inline deterministic structure (scalars always;
# vectors only when structural — coefficient-holding), canonicalize the
# expanded location, split the affine sum, classify each summand. Returns
# (terms, uses) with uses :: Vector{(coef name, addressee, sign)}.
# Anonymous non-affine vector substructure auto-extracts to synthetic
# derived locals, so naming a subexpression never changes legality.
function _analyze_predictor(pname, rhs, ctx, lhs)
    where = "predictor $pname"
    expanded = _inline_structure(rhs, ctx, Set{Symbol}([pname]), where)
    _reject_unknown_calls(where, expanded)
    canon = _canonical_expr(expanded, ctx.data, ctx.detshape, where)
    out = Tuple{Int,Any}[]
    _collect_signed!(out, canon, 1, pname)
    terms = TermSpec[]
    uses = Tuple{Symbol,Symbol,Int}[]
    addr_owner = Dict{Symbol,Symbol}()
    for (sign, core) in out
        term, use = _classify_summand(pname, core, sign, ctx)
        if use !== nothing
            name, addr, _ = use
            haskey(addr_owner, addr) && _sfail(
                "predictor $pname: column $addr has two coefficients " *
                "$(addr_owner[addr]) and $name — one coefficient per column")
            addr_owner[addr] = name
            push!(uses, use)
        end
        push!(terms, term)
    end
    isempty(uses) && _sfail("predictor $pname has no estimated " *
                            "coefficients (offsets only) — add an intercept " *
                            "or coefficient")
    return terms, uses
end

function _inline_structure(ex, ctx, visited::Set{Symbol}, where)
    ex isa Symbol || return _inline_structure_expr(ex, ctx, visited, where)
    haskey(ctx.detmap, ex) || return ex
    if ex in ctx.structural || ctx.detshape[ex] !== :vector
        ex in visited && _sfail("$where: cyclic definition through $ex")
        push!(ctx.absorbed, ex)
        push!(visited, ex)
        out = _inline_structure(ctx.detmap[ex], ctx, visited, where)
        pop!(visited)
        return out
    end
    return ex
end
function _inline_structure_expr(ex, ctx, visited, where)
    ex isa Expr || return ex
    return Expr(ex.head, (_inline_structure(a, ctx, visited, where)
                          for a in ex.args)...)
end

function _collect_signed!(out, ex, sign::Int, pname)
    if ex isa Expr && ex.head === :call && !isempty(ex.args)
        fn = ex.args[1]
        if fn === :+ || fn === :.+
            if length(ex.args) == 2
                return _collect_signed!(out, ex.args[2], sign, pname)
            end
            for a in ex.args[2:end]
                _collect_signed!(out, a, sign, pname)
            end
            return nothing
        elseif fn === :- || fn === :.-
            if length(ex.args) == 2
                return _collect_signed!(out, ex.args[2], -sign, pname)
            elseif length(ex.args) == 3
                _collect_signed!(out, ex.args[2], sign, pname)
                return _collect_signed!(out, ex.args[3], -sign, pname)
            end
        end
    end
    push!(out, (sign, ex))
    return nothing
end

function _classify_summand(pname, core, sign::Int, ctx)
    core isa Number && _sfail("predictor $pname: literal $core is not a " *
                              "term — fold constants into an intercept " *
                              "coefficient with an offset prior location")
    core isa Symbol && return _classify_symbol(pname, core, sign, ctx)
    core isa Expr || _sfail("predictor $pname: $(repr(core)) is not affine " *
                            "in data (terms are bare coefficients, " *
                            "`coefficient .* column`, " *
                            "`coefficients[column]`, and bare columns)")
    head = core.head
    head === :ref && return _classify_ref(pname, core, sign, ctx)
    if head === :call && !isempty(core.args) && core.args[1] === :.*
        return _classify_product(pname, core, sign, ctx)
    end
    head === :macrocall && _sfail("predictor $pname: macros do not lower " *
                                  "inside predictor expressions")
    if _canon_shape(core, ctx.data, ctx.detshape) === :vector
        return _extract_summand(pname, core, sign, ctx)
    end
    detkeys = Set{Symbol}(keys(ctx.detmap))
    coefrefs = Symbol[s for s in _value_symbols(core)
        if s in ctx.normal_priors ||
            _is_free_name(s, ctx.data, detkeys, ctx.prior_names)]
    length(coefrefs) > 1 && _sfail("predictor $pname: $(repr(core)) is " *
                                   "nonlinear in coefficients")
    length(coefrefs) == 1 && _sfail(
        "predictor $pname: $(repr(core)) computes over the coefficient " *
        "$(only(coefrefs)) — computed coefficients are not in slice 1")
    return _sfail("predictor $pname: $(repr(core)) is a scalar, not a " *
                  "term — scalar parameters are not identified " *
                  "separately from the intercept (bind the value to a " *
                  "column, `w = s .+ x`, or fold it into an intercept " *
                  "prior location)")
end

function _classify_symbol(pname, core::Symbol, sign::Int, ctx)
    (core in ctx.data || core in ctx.vecdefs) &&
        return TermSpec(OffsetTerm, [core], NamedTuple(),
        core, Symbol(core, "_off")), nothing
    haskey(ctx.detmap, core) && _sfail("predictor $pname: $core is a " *
                                       "computed scalar, not a sampled " *
                                       "coefficient (computed coefficients " *
                                       "are not in slice 1)")
    core in ctx.prior_names && core ∉ ctx.normal_priors &&
        _sfail("predictor $pname: $core is a scalar parameter, not a " *
               "term — scalar parameters are not identified separately " *
               "from the intercept (bind the value to a column, " *
               "`w = $core .+ x`, or fold it into an intercept prior " *
               "location)")
    return TermSpec(InterceptTerm, ColumnRef[], NamedTuple(), :Intercept,
        :intercept), (core, :Intercept, sign)
end

# Anonymous non-affine vector substructure becomes a synthetic derived
# local (offset term over it); the sign folds into the extracted column.
function _extract_summand(pname, core::Expr, sign::Int, ctx)
    e = sign < 0 ? Expr(:call, :.-, core) : core
    nm = _extract_column(pname, e, ctx)
    return TermSpec(OffsetTerm, [nm], NamedTuple(), nm,
        Symbol(nm, "_off")), nothing
end

function _extract_column(pname, e::Expr, ctx)
    while true
        ctx.synth[] += 1
        nm = Symbol(:_rkppl_synth_, ctx.synth[])
        nm in ctx.taken && continue
        push!(ctx.taken, nm)
        push!(ctx.synth_derived, VectorAssignmentSpec(nm, e, nm))
        return nm
    end
end

function _classify_product(pname, core::Expr, sign::Int, ctx)
    inner = sign
    coefs = Symbol[]
    values = Any[]
    stripped = Any[]
    for f in core.args[2:end]
        s, g = _strip_sign(f)
        inner *= s
        push!(stripped, g)
        if g isa Symbol
            k = _summand_kind(g, ctx)
            if k === :coef
                push!(coefs, g)
            elseif k === :data || k === :local
                push!(values, g)
            elseif k === :number
                _sfail("predictor $pname: literal scaling in " *
                       "$(repr(core)) is not a term — scale the column " *
                       "or the prior instead")
            elseif k === :param
                _sfail("predictor $pname: $(repr(core)) scales by the " *
                       "parameter $g — computed coefficients are not " *
                       "in slice 1")
            else
                _sfail("predictor $pname: computed assignments do not " *
                       "lower as coefficients (computed coefficients " *
                       "are not in slice 1)")
            end
        elseif g isa Number
            _sfail("predictor $pname: literal scaling in $(repr(core)) " *
                   "is not a term — scale the column or the prior instead")
        elseif _canon_shape(g, ctx.data, ctx.detshape) === :vector
            push!(values, g)
        else
            _sfail("predictor $pname: $(repr(core)) scales by the " *
                   "computed scalar $(repr(g)) — computed coefficients " *
                   "are not in slice 1")
        end
    end
    length(coefs) > 1 && _sfail("predictor $pname: $(repr(core)) is " *
                                "nonlinear in coefficients")
    if isempty(coefs)
        # Pure value product (interaction): extract whole (signs folded
        # in), offset term over it.
        e = length(stripped) == 1 ? only(stripped) :
            Expr(:call, :.*, stripped...)
        inner < 0 && (e = Expr(:call, :.-, e))
        if e isa Symbol
            return TermSpec(OffsetTerm, [e], NamedTuple(), e,
                Symbol(e, "_off")), nothing
        end
        nm = _extract_column(pname, e, ctx)
        return TermSpec(OffsetTerm, [nm], NamedTuple(), nm,
            Symbol(nm, "_off")), nothing
    end
    coef = only(coefs)
    if length(values) == 1 && values[1] isa Symbol
        col = values[1]
        return TermSpec(ContinuousTerm, [col], NamedTuple(), col,
            Symbol(col, "_term")), (coef, col, inner)
    end
    if isempty(values)
        # Degenerate `.*(coef)`: the coefficient stands alone.
        return TermSpec(InterceptTerm, ColumnRef[], NamedTuple(),
            :Intercept, :intercept), (coef, :Intercept, inner)
    end
    # Coefficient times interaction: extract the value part, scale it.
    rest = length(values) == 1 ? values[1] :
        Expr(:call, :.*, values...)
    col = _extract_column(pname, rest, ctx)
    return TermSpec(ContinuousTerm, [col], NamedTuple(), col,
        Symbol(col, "_term")), (coef, col, inner)
end

function _strip_sign(ex)
    sign = 1
    while ex isa Expr && ex.head === :call && length(ex.args) == 2 &&
            (ex.args[1] === :- || ex.args[1] === :.-)
        sign = -sign
        ex = ex.args[2]
    end
    return sign, ex
end

function _summand_kind(s::Symbol, ctx)
    s in ctx.data && return :data
    s in ctx.vecdefs && return :local
    haskey(ctx.detmap, s) && return :det
    s in ctx.prior_names && s ∉ ctx.normal_priors && return :param
    return :coef
end
function _summand_kind(n::Number, ctx)
    return :number
end
function _summand_kind(_, _)
    return :other
end

function _classify_ref(pname, core::Expr, sign::Int, ctx)
    length(core.args) == 2 || _sfail("predictor $pname: factor indexing " *
                                     "takes `coefficients[group]` exactly, " *
                                     "got $(repr(core))")
    base, idx = core.args
    base isa Symbol || _sfail("predictor $pname: factor base must be a " *
                              "bare coefficient vector, got $(repr(base))")
    base in ctx.data && _sfail("predictor $pname: $base is data — " *
                               "precompute data-indexed columns")
    haskey(ctx.detmap, base) && _sfail("predictor $pname: $base is a " *
                                       "computed assignment, not a sampled " *
                                       "coefficient vector")
    col, ref = _factor_index(pname, idx, ctx)
    return TermSpec(FactorTerm, [col], (contrasts = :treatment, ref = ref),
        col, Symbol(col, "_term")), (base, col, sign)
end

# Bare `c[g]` is treatment/ref-1 sugar; `c[treatment(g, ref)]` pins the
# reference level (R `contr.treatment` tradition). The `treatment` head is
# AST vocabulary only — it is never called.
function _factor_index(pname, idx, ctx)
    idx isa Symbol || return _treatment_index(pname, idx, ctx)
    idx in ctx.vecdefs && _sfail("predictor $pname: factor over the " *
                                 "derived column $idx needs pre-evaluation " *
                                 "level knowledge — factors take raw " *
                                 "grouping columns in slice 1")
    idx in ctx.data || _sfail("predictor $pname: factor index $idx must " *
                              "be a data column")
    return idx, 1
end

function _treatment_index(pname, idx, ctx)
    idx isa Expr && idx.head === :call && !isempty(idx.args) &&
        idx.args[1] === :treatment || _sfail(
            "predictor $pname: factor index must be a bare data column or " *
            "`treatment(group, ref)`, got $(repr(idx))")
    args = _plain_args(idx, "`treatment`")
    (length(args) == 1 || length(args) == 2) ||
        _sfail("predictor $pname: `treatment` takes " *
               "`treatment(group[, ref])`")
    g = args[1]
    g isa Symbol && g in ctx.vecdefs && _sfail(
        "predictor $pname: factor over the derived column $g needs " *
        "pre-evaluation level knowledge — factors take raw grouping " *
        "columns in slice 1")
    g isa Symbol && g in ctx.data || _sfail(
        "predictor $pname: `treatment` group must be a bare data column, " *
        "got $(repr(g))")
    length(args) == 1 && return g, 1
    ref = args[2]
    ref isa Integer && !(ref isa Bool) && ref >= 1 || _sfail(
        "predictor $pname: `treatment` ref must be a literal 1-based level " *
        "index, got $(repr(ref))")
    return g, Int(ref)
end

# Coefficient priors: recovered by name from `coef ~ Normal(lit, lit)`
# statements; missing priors default to Normal(0, 1) (emitter convention).
# Plan order follows predictors, addressees in term order.
function _lower_coefficient_priors(sample, coefuse, predictors)
    stated = Dict{Symbol,Any}()
    for s in sample
        haskey(coefuse, s.lhs) && (stated[s.lhs] = s.rhs)
    end
    for (name, uses) in coefuse
        preds = unique!(map(first, copy(uses)))
        length(preds) > 1 && _sfail("coefficient $name is shared across " *
                                    "predictors $(join(preds, ", ")) — " *
                                    "coefficient blocks are per-predictor, " *
                                    "rename or duplicate it")
        addrs = unique!(map(u -> u[2], copy(uses)))
        length(addrs) > 1 && _sfail("coefficient $name is used on two " *
                                    "columns ($(join(addrs, ", "))) — one " *
                                    "coefficient per column")
    end
    priors = PopulationPrior[]
    for pred in predictors
        for t in pred.terms
            t.kind === OffsetTerm && continue
            addr = t.kind === InterceptTerm ? :Intercept : only(t.columns)
            use = _find_use(coefuse, pred.name, addr)
            use === nothing && _sfail("internal: no coefficient use for " *
                                      "($(pred.name), $addr)")
            name = use[1]
            sign = use[3]
            if !haskey(stated, name)
                push!(priors, PopulationPrior(pred.name, addr, 0.0, 1.0))
                continue
            end
            rhs = stated[name]
            loc, scale = _coefficient_normal(name, rhs, pred.name, addr)
            push!(priors, PopulationPrior(pred.name, addr, sign * loc, scale))
        end
    end
    return priors
end

function _find_use(coefuse, pname, addr)
    for (name, uses) in coefuse
        for (p2, a2, s2) in uses
            p2 === pname && a2 === addr && return (name, a2, s2)
        end
    end
    return nothing
end

function _coefficient_normal(name, rhs, pname, addr)
    rhs isa Expr && rhs.head === :call && rhs.args[1] === :Normal ||
        _sfail("coefficient $name of predictor $pname needs a " *
               "`Normal(literal, literal)` prior, got $(repr(rhs))")
    args = _plain_args(rhs, "coefficient prior")
    length(args) == 2 || _sfail("coefficient $name of predictor $pname " *
                                "needs `Normal(location, scale)`")
    loc, scale = args
    (loc isa Real || loc === :Inf) ||
        _sfail("coefficient $name prior location must be a literal " *
               "(hierarchical coefficient priors are not in slice 1)")
    (scale isa Real || scale === :Inf) ||
        _sfail("coefficient $name prior scale must be a literal")
    locv = loc === :Inf ? Inf : Float64(loc)
    scalev = scale === :Inf ? Inf : Float64(scale)
    return locv, scalev
end

const _PARAM_FAMILIES = Dict{Symbol,Symbol}(
    :Normal => :normal, :Cauchy => :cauchy,
    :Exponential => :exponential, :Gamma => :gamma,
    :LogNormal => :lognormal, :Beta => :beta,
    :InverseGamma => :inverse_gamma,
)

function _lower_parameters(sample, coefuse, ctx)
    params = SampledParameter[]
    syms = Set{Symbol}()
    for s in sample
        s.lhs in ctx.data && continue
        haskey(coefuse, s.lhs) && continue
        p = _lower_parameter(s.lhs, s.rhs, coefuse)
        push!(params, p)
        for v in values(p.args)
            v isa Symbol && push!(syms, v)
        end
    end
    return params, syms
end

function _lower_parameter(lhs, rhs, coefuse)
    rhs isa Expr && rhs.head === :call && !isempty(rhs.args) || _sfail(
        "parameter $lhs needs a distribution call, got $(repr(rhs))")
    fam = rhs.args[1]
    fam === :Flat && return _lower_flat(lhs, rhs)
    fam === :flat && _sfail("parameter $lhs: use `Flat()` (Turing-style " *
                            "improper uniform), not Stan-style `flat()`")
    fam === :truncated && return _lower_truncated_param(lhs, rhs, coefuse)
    fam in (:HalfNormal, :HalfCauchy) &&
        return _lower_half_param(lhs, rhs, fam, coefuse)
    fam === :positive && _sfail("parameter $lhs: write `HalfNormal(s)` " *
                                "or `truncated(Normal(0, s), 0, Inf)`")
    fam in (:weighted, :censored, :interval_censored) &&
        _sfail("`$fam` applies to responses only (parameter $lhs)")
    fam in (:normal, :cauchy, :exponential, :gamma, :lognormal, :beta,
        :inverse_gamma, :halfnormal, :halfcauchy) &&
        _sfail("parameter $lhs: use Distributions.jl constructors " *
               "(`Normal`, not `normal`)")
    haskey(_PARAM_FAMILIES, fam) ||
        _sfail("parameter $lhs: unknown distribution `$(repr(fam))` " *
               "(admitted: Normal, Cauchy, Exponential, Gamma, LogNormal, " *
               "Beta, InverseGamma, HalfNormal, HalfCauchy, Flat, " *
               "truncated). " *
               "If `$fam` is a submodel, `y ~ sm(...)` calls need " *
               "composition semantics (planned).")
    args = _plain_args(rhs, "`$fam`")
    vals = [_lower_param_arg(lhs, a, coefuse) for a in args]
    argkeys = ntuple(i -> Symbol(:arg, i), length(vals))
    return SampledParameter(lhs, _PARAM_FAMILIES[fam],
        NamedTuple{argkeys}(Tuple(vals)), nothing, lhs)
end

function _lower_param_arg(lhs, a, coefuse)
    a isa Real && return a
    a === :Inf && return Inf
    a isa Symbol || _sfail("parameter $lhs argument $(repr(a)) must be a " *
                           "literal or a parameter/assignment name (bind " *
                           "expressions via an assignment first)")
    haskey(coefuse, a) && _sfail("$a is a predictor coefficient and cannot " *
                                 "also be a parameter argument (parameter $lhs)")
    return a
end

function _lower_flat(lhs, rhs)
    args = _plain_args(rhs, "`Flat`")
    isempty(args) ||
        _sfail("parameter $lhs: `Flat()` takes no arguments in slice 1")
    return SampledParameter(lhs, :flat, NamedTuple(), nothing, lhs)
end

# `HalfNormal(s)` / `HalfCauchy(s)` lower to the `:positive` support
# override with synthesized literal-zero location (exact +log(2)).
function _lower_half_param(lhs, rhs, fam, coefuse)
    args = _plain_args(rhs, "`$fam`")
    length(args) == 1 ||
        _sfail("parameter $lhs: `$fam` takes exactly the scale")
    scale = _lower_param_arg(lhs, args[1], coefuse)
    base = fam === :HalfNormal ? :normal : :cauchy
    return SampledParameter(lhs, base, (arg1 = 0, arg2 = scale), :positive,
        lhs)
end

# Slice-1 parameter truncation is half-Normal/half-Cauchy only, matching the
# `:positive` support override (exact +log(2) by symmetry at literal 0).
function _lower_truncated_param(lhs, rhs, coefuse)
    args = _plain_args(rhs, "`truncated`")
    length(args) == 3 || _sfail("parameter $lhs: use the Distributions.jl " *
                                "object form `truncated(Normal(0, s), 0, Inf)`")
    obj, lo, hi = args
    obj isa Expr && obj.head === :call || _sfail(
        "parameter $lhs: `truncated` wraps a distribution object, got " *
        "$(repr(obj))")
    fam = obj.args[1]
    fam in (:Normal, :Cauchy) || _sfail(
        "parameter $lhs: slice-1 truncation is half-Normal/half-Cauchy " *
        "only (`truncated(Normal(0, s), 0, Inf)`)")
    oargs = _plain_args(obj, "`$fam`")
    length(oargs) == 2 || _sfail("parameter $lhs: `$fam` takes two arguments")
    oargs[1] isa Real && oargs[1] == 0 || _sfail(
        "parameter $lhs: half-truncation needs literal zero location")
    lo isa Real && lo == 0 || _sfail("parameter $lhs: half-truncation " *
                                     "lower bound must be literal 0")
    (hi === :Inf || (hi isa Real && isinf(hi) && hi > 0)) ||
        _sfail("parameter $lhs: half-truncation upper bound must be Inf")
    vals = [_lower_param_arg(lhs, a, coefuse) for a in oargs]
    return SampledParameter(lhs, _PARAM_FAMILIES[fam],
        (arg1 = vals[1], arg2 = vals[2]), :positive, lhs)
end

function _lower_assignment(nm, rhs, coefuse)
    for s in _value_symbols(rhs)
        haskey(coefuse, s) && _sfail("$s is a predictor coefficient and " *
                                     "cannot also be referenced by assignment " *
                                     "$nm")
    end
    rhs isa Expr || rhs isa Symbol || rhs isa Real ||
        _sfail("assignment $nm must be an expression, name or literal, " *
               "got $(repr(rhs))")
    return AssignmentSpec(nm, rhs, nm)
end

# Vector shape rules belong to validation (contract v3); lowering checks
# coefficient discipline and types the node (vocabulary and Julia-shape
# screens ran at the definition pre-pass).
function _lower_vector_assignment(nm, rhs, coefuse)
    for s in _value_symbols(rhs)
        haskey(coefuse, s) && _sfail("$s is a predictor coefficient and " *
                                     "cannot also be referenced by derived " *
                                     "column $nm")
    end
    rhs isa Expr || rhs isa Symbol ||
        _sfail("derived column $nm must be an expression or column alias, " *
               "got $(repr(rhs))")
    return VectorAssignmentSpec(nm, rhs, nm)
end
