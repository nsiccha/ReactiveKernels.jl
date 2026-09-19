# `@rkppl` authoring surface: StanBlocks-close model blocks lowering to
# data-free StructuralPlans.
#
# The shape mirrors StanBlocks `@slic` (block AST capture, `~` density
# statements, deterministic `=`, `model(; data...)` binding) under the
# standing constraints: Distributions.jl constructors (never Stan lowercase),
# immutable single-assignment top level, no control flow, no `target`,
# `@plate` observations + deterministic cells (desugar; sampled cells
# deferred), `@scan` reserved. Broadcasting is EXPLICIT (no implied
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
    mod::Module
end

"""
    RKPPLSubmodel(name, argnames, body, mod)

A captured reusable submodel definition (`@rkppl sm(a, b) = begin … end`),
mirroring StanBlocks `@slic f(args…)=body`. `argnames` are the positional
inputs bound by name at the use site; `body` is the block AST — density /
deterministic `=` statements followed by a trailing RETURN expression whose
value binds to the use-site LHS; `mod` is the defining module (for symbol
resolution). Invoked as `latent ~ sm(a, b)` and expanded inline by
[`lower_rkppl`](@ref) (see `_expand_submodels`): the submodel's own `~`/`=`
names are namespaced under the LHS (`latent_…`) and spliced into the parent
plan, so a submodel lowers exactly like a hand-inlined model — transparent and
reusable, never an opaque node.
"""
struct RKPPLSubmodel
    name::Symbol
    argnames::Vector{Symbol}
    body::Expr
    mod::Module
end

"""True for the submodel-definition head `sm(args...) = begin ... end`."""
_is_submodel_def(body) =
    Meta.isexpr(body, :(=), 2) && Meta.isexpr(first(body.args), :call)

function _check_body_shape(body)
    body isa Expr && body.head === :block && return nothing
    return _sfail("@rkppl takes a `begin ... end` block, a submodel " *
                  "definition (`sm(args...) = begin ... end`), or " *
                  "caller-scope data plus a block")
end

"""Build the `sm = RKPPLSubmodel(...)` binding from a submodel definition."""
function _submodel_def_expr(def::Expr, mod::Module)
    call, body = def.args[1], def.args[2]
    name = call.args[1]
    name isa Symbol || _sfail("submodel name must be a bare Symbol, got " *
                              "$(repr(name))")
    argnames = call.args[2:end]
    for a in argnames
        a isa Symbol || _sfail("submodel `$name` positional arguments must " *
                               "be bare Symbols, got $(repr(a))")
    end
    body isa Expr && body.head === :block ||
        _sfail("submodel `$name` body must be a `begin ... end` block")
    argvec = Expr(:vect, [QuoteNode(a) for a in argnames]...)
    return esc(:($name = $(RKPPLSubmodel)($(QuoteNode(name)), $argvec,
                                          $(Meta.quot(body)), $mod)))
end

"""Capture a model block, or define a reusable submodel
(`@rkppl sm(args...) = begin ... end`; see [`RKPPLSubmodel`](@ref))."""
macro rkppl(body)
    _is_submodel_def(body) && return _submodel_def_expr(body, __module__)
    _check_body_shape(body)
    return Expr(:call, RKPPLModel, Meta.quot(body), __module__)
end

"""Capture a model block and immediately lower+bind caller-scope data
(a `NamedTuple` or dict of columns)."""
macro rkppl(data, body)
    _is_submodel_def(body) && _sfail("a submodel definition takes no data " *
        "(write `@rkppl sm(args...) = begin ... end`); data binds at the " *
        "use site")
    _check_body_shape(body)
    q = Meta.quot(body)
    return esc(:($(_bind_immediate)($(RKPPLModel)($q, $(__module__)), $data)))
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
    plan = lower_rkppl(m.ast, keys(cols); mod = m.mod)
    return bind_data(plan, cols)
end

"""
    lower_rkppl(ast, data_names; mod=Main) -> StructuralPlan

Lower a captured `@rkppl` block AST to a data-free (unbound) plan.
`data_names` classifies every `~` / `.~` LHS: data under `.~` is a
response, non-data under `~` is a prior (sampled parameter or, when the
name sits in a predictor coefficient position, a population prior); the
crossed spellings fail closed (`~` is scalar-only, `.~` broadcasts over
data). Runs `validate_structure` before returning. The BRM emitter calls
this entry point directly with ASTs.

`mod` is the module against which `latent ~ sm(args...)` call heads are
resolved to [`RKPPLSubmodel`](@ref)s; a resolving call is expanded inline
before partitioning (see `_expand_submodels`). Non-submodel call heads
(distributions, unknown names) are untouched and screened as before, so the
default `mod=Main` keeps every non-submodel model unchanged.
"""
function lower_rkppl(ast, data_names; mod::Module = Main)::StructuralPlan
    data = Set{Symbol}()
    for n in data_names
        n isa Symbol || _sfail("data names must be Symbols, got $(repr(n))")
        push!(data, n)
    end
    ast isa Expr && ast.head === :block ||
        _sfail("lower_rkppl takes a `begin ... end` block AST")
    ast = _expand_submodels(ast, data, mod)
    sample, det, plate_ctx, plate_specs, scans = _partition_statements(ast, data)
    plate_names = Set{Symbol}(nm for (nm, _, _, _) in plate_specs)
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
    taken = union(data, Set{Symbol}(nm for (nm, _) in det), prior_names,
        plate_names)
    ctx = (; data, detmap = canonmap, prior_names, normal_priors, detshape,
        vecdefs, structural, plate_names, absorbed = Set{Symbol}(),
        synth = Ref(0), synth_derived = VectorAssignmentSpec[], taken,
        scan_states = Set{Symbol}(s.state for s in scans))
    responses = LikelihoodSpec[]
    predictors = PredictorSpec[]
    pred_idx = Dict{Symbol,Int}()
    coefuse = Dict{Symbol,Vector{Tuple{Symbol,Symbol,Int}}}()
    for s in sample
        if s.broadcast
            # Broadcast coefficient priors lower with their factor term.
            s.levels !== nothing && continue
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
    priors, levelmaps = _lower_coefficient_priors(sample, coefuse, predictors)
    params, paramsyms = _lower_parameters(sample, coefuse, ctx)
    plate_parameters = PlateParameter[
        _lower_plate_parameter(nm, rhs, rng, coefuse)
        for (nm, rhs, rng, _) in plate_specs]
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
    _check_plate_bares(plate_ctx, data, Set{Symbol}(p.name for p in predictors),
        Set{Symbol}(d.name for d in derived), _factor_coefs(coefuse, predictors),
        plate_names)
    plan = StructuralPlan(responses, predictors, priors, params, assigns,
        Dict{Symbol,AbstractVector}(), 0; derived = derived,
        levelmaps = levelmaps, plate_parameters = plate_parameters, scans = scans)
    validate_structure(plan)
    return plan
end

# A per-cell latent declaration reuses the scalar-parameter distribution
# parsing (family, args, HalfNormal/truncated → :positive) and rides the
# plate's range.
function _lower_plate_parameter(name::Symbol, rhs, range, coefuse)
    sp = _lower_parameter(name, rhs, coefuse)
    return PlateParameter(sp.name, sp.family, sp.args, sp.support_override,
        range, sp.label)
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
    Set{Symbol}((:ifelse,)))

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
                "$where calls `treatment`, which was removed " *
                "(BRM-specific contrasts)")
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
    scans = ScanSpec[]
    seen = Set{Symbol}()
    seelines = Dict{Symbol,Int}()
    seen_doc = false
    line = 0
    args, plate_ctx, plate_params = _expand_plates(ast.args, data)
    for arg in args
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
        # `@scan begin <setup>; for … end end` — a sequential-recurrence block.
        # Parsed into a `ScanSpec` here; its carried state is claimed as a
        # model-level latent name.
        if arg.head === :macrocall && arg.args[1] === Symbol("@scan")
            (length(arg.args) >= 3 && arg.args[end] isa Expr) ||
                _sfail("@scan takes a `begin … end` block")
            sp = parse_scan_block(arg.args[end])
            _claim!(seen, seelines, sp.state, line)
            push!(scans, sp)
            continue
        end
        arg.head === :block &&
            _sfail("nested `begin` blocks do not lower — flatten the block")
        st = _unwrap_trivia(arg)
        if _is_sample(st) || _is_broadcast_sample(st)
            bc = _is_broadcast_sample(st)
            tilde = bc ? "`.~`" : "`~`"
            lhs, rng, levs = _sample_lhs(st.args[2], bc, tilde, data)
            _claim!(seen, seelines, lhs, line)
            _reject_target(st.args[3], lhs)
            push!(sample, SampleStmt(lhs, st.args[3], bc, rng, levs))
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
    return sample, det, plate_ctx, plate_params, scans
end

# Pre-pass: expand top-level `@plate for i in R ... end` blocks into
# spliced top-level statements (desugar slice: observations +
# deterministic cells; per-cell sampled arrays deferred). Spliced
# statements carry the plate's line for claim messages. Also returns the
# plate context: `(lhs, line, bare-symbols)` per spliced statement for
# the post-analysis whole-vector check.
function _expand_plates(args, data::Set{Symbol})
    expanded = Any[]
    ctx = Tuple{Symbol,Int,Set{Symbol}}[]
    params = Tuple{Symbol,Any,Union{Nothing,UnitRange{Int}},Int}[]
    line = 0
    for arg in args
        if arg isa LineNumberNode
            line = arg.line
            push!(expanded, arg)
            continue
        end
        if arg isa Expr && arg.head === :macrocall && !isempty(arg.args) &&
                arg.args[1] === Symbol("@plate")
            pl = line
            if length(arg.args) >= 2 && arg.args[2] isa LineNumberNode
                pl = arg.args[2].line
            end
            stmts, stx, prm = _desugar_plate(arg, pl, data)
            for st in stmts
                pl > 0 && push!(expanded, LineNumberNode(pl))
                push!(expanded, st)
            end
            append!(ctx, stx)
            append!(params, prm)
            continue
        end
        push!(expanded, arg)
    end
    return expanded, ctx, params
end

function _desugar_plate(st::Expr, line::Int, data::Set{Symbol})
    (length(st.args) == 3 && st.args[3] isa Expr &&
        st.args[3].head === :for) ||
        _sfail("`@plate` takes `@plate for i in R ... end` exactly")
    loop = st.args[3]
    asg = loop.args[1]
    asg isa Expr && asg.head === :block &&
        _sfail("`@plate` takes one loop variable — multi-index " *
               "`@plate for i in …, j in …` does not lower. Crossed/nested " *
               "group effects use factor terms (`c[levels(g)]`, slice-C " *
               "`FactorTerm`+`LevelMap`); a shared-prior multi-index grid " *
               "equals a single-index plate over `n_obs`; multi-index " *
               "observations need an N-D response this model class lacks")
    (asg isa Expr && asg.head === :(=) && length(asg.args) == 2 &&
        asg.args[1] isa Symbol) ||
        _sfail("`@plate` loop must be `for i in R`")
    ivar = asg.args[1]
    R = asg.args[2]
    rkind = _plate_range_kind(R)
    body = loop.args[2]
    cells = Any[a for a in body.args if !(a isa LineNumberNode)]
    isempty(cells) && _sfail("`@plate` body is empty")
    # Names bound by `=` anywhere in this plate (cell locals, excluded
    # from the bare-vector check even on forward reference) — a bare LHS
    # (`t = ...`) or an i-indexed LHS (`theta[i] = ...`).
    plate_defs = Set{Symbol}()
    for c in cells
        c isa Expr && c.head === :(=) && length(c.args) == 2 || continue
        lc = c.args[1]
        if lc isa Symbol
            push!(plate_defs, lc)
        elseif lc isa Expr && lc.head === :ref && length(lc.args) == 2 &&
                lc.args[1] isa Symbol && lc.args[2] === ivar
            push!(plate_defs, lc.args[1])
        end
    end
    out = Expr[]
    ctx = Tuple{Symbol,Int,Set{Symbol}}[]
    params = Tuple{Symbol,Any,Union{Nothing,UnitRange{Int}},Int}[]
    for c in cells
        push!(out, _desugar_cell(c, ivar, rkind, line, data, plate_defs, ctx,
            params)...)
    end
    return out, ctx, params
end

# Plate ranges mirror the response ranges: literal `a:b` (validated via
# the desugared `y[a:b]` form), `eachindex(v)`, `axes(v, 1)`.
function _plate_range_kind(R)
    R isa Expr && R.head === :call && !isempty(R.args) || return _sfail(
        "`@plate` range must be `1:N`, `eachindex(v)`, or `axes(v, 1)` — " *
        "got $(repr(R)) (values-iteration is planned)")
    R.args[1] === :(:) && return (:coloncall, R)
    R.args[1] === :eachindex && length(R.args) == 2 &&
        R.args[2] isa Symbol && return (:eachindex, R.args[2])
    R.args[1] === :axes && length(R.args) == 3 && R.args[2] isa Symbol &&
        R.args[3] == 1 && return (:axes, R.args[2])
    return _sfail("`@plate` range must be `1:N`, `eachindex(v)`, or " *
                  "`axes(v, 1)` — got $(repr(R))")
end

# One cell statement → spliced top-level statement(s). Observations
# (`y[i] ~ OBJ`) become `y .~ OBJ.` (or `y[a:b] .~ OBJ.` under a literal
# range); deterministic cells strip to top level (visible model-wide —
# documented looseness: desugared locals leak like any top-level det).
function _desugar_cell(c, ivar, rkind, line, data, plate_defs, ctx, params)
    c isa Expr || _sfail("cells hold `~` observations and `=` " *
                         "assignments only")
    if c.head === :macrocall && !isempty(c.args) &&
            c.args[1] === Symbol("@plate")
        _sfail("nested `@plate` blocks are not a StanBlocks form — use " *
               "factor/levels for crossed effects (`c[levels(g)]`), `@scan` " *
               "for sequential recurrence")
    end
    if _is_broadcast_sample(c)
        _sfail("cells are scalar (`~`); broadcast (`.~`) at top level")
    end
    if _is_sample(c)
        return _desugar_cell_sample(c, ivar, rkind, line, data, plate_defs,
            ctx, params)
    end
    if c.head === :(=) && length(c.args) == 2
        # A deterministic cell binds a bare local (`t = expr`) or an i-indexed
        # column (`theta[i] = f(...[i])`); both strip `[i]` refs to a
        # whole-vector top-level assignment.
        lc = c.args[1]
        col = if lc isa Symbol
            lc
        elseif lc isa Expr && lc.head === :ref && length(lc.args) == 2 &&
                lc.args[1] isa Symbol && lc.args[2] === ivar
            lc.args[1]
        else
            _sfail("cell assignment LHS is a bare local (`t = ...`) or an " *
                   "`$ivar`-indexed column (`theta[$ivar] = ...`), got " *
                   "$(repr(lc))")
        end
        col in data && _sfail("cell assignment `$col = ...` redefines " *
                              "bound data")
        bares = _cell_bares(c.args[2], ivar)
        setdiff!(bares, plate_defs)
        push!(ctx, (col, line, bares))
        return [Expr(:(=), col, _strip_cell(c.args[2], ivar))]
    end
    return _sfail("cells hold `~` observations and `=` assignments only")
end

# A cell `~` statement is EITHER a per-cell latent parameter declaration
# (`theta[i] ~ Normal(mu, tau)` — a non-data indexed LHS, scalar undotted
# distribution, shared-scalar args) or an observation on a sliced data column
# (`y[i] ~ Normal.(mu[i], s)` — dotted object). Returns the spliced top-level
# statement(s); a per-cell parameter records its spec in `params` and emits no
# top-level statement.
function _desugar_cell_sample(c, ivar, rkind, line, data, plate_defs, ctx, params)
    lhs = c.args[2]
    lhs isa Symbol && _sfail("bare per-cell sample `$lhs ~ ...` does not " *
                             "lower — index the latent (`$lhs[$ivar] ~ ...`) " *
                             "for a per-cell parameter, or write a shared " *
                             "prior outside the plate")
    (lhs isa Expr && lhs.head === :ref && length(lhs.args) == 2 &&
        lhs.args[1] isa Symbol && lhs.args[2] === ivar) ||
        _cell_lhs_error(lhs, ivar)
    col = lhs.args[1]
    obj = c.args[3]
    if col ∉ data
        # Per-cell latent PARAMETER: a scalar (undotted) distribution. Its args
        # are shared across cells (a captured scalar) or per-cell (`eta[$ivar]`,
        # a varying prior mean/scale); the `[$ivar]` strip and the bare-vector
        # check enforce the index discipline, exactly like an observation cell.
        obj isa Expr && obj.head === :. && _sfail(
            "per-cell latent `$col[$ivar] ~ ...` takes a scalar (undotted) " *
            "distribution (`$col[$ivar] ~ Normal(mu, tau)` / " *
            "`$col[$ivar] ~ Normal(eta[$ivar], tau)`), got the dotted $(repr(obj))")
        bares = _cell_bares(obj, ivar)
        setdiff!(bares, plate_defs)
        push!(ctx, (col, line, bares))
        push!(params, (col, _strip_cell(obj, ivar), _plate_param_range(col, rkind),
            line))
        return Expr[]
    end
    bares = _cell_bares(obj, ivar)
    setdiff!(bares, plate_defs)
    push!(ctx, (col, line, bares))
    # Observation cells mirror top-level spelling exactly (dots as written —
    # the desugar strips refs, never invents dots): the object must already
    # be dotted, pairing scalar `~` (the cell) with a pre-dotted object.
    obj isa Expr && obj.head === :. || _sfail(
        "cell objects are dotted distribution calls " *
        "(`y[$ivar] ~ Normal.(mu[$ivar], s)`), got $(repr(obj))")
    obj = _strip_cell(obj, ivar)
    if rkind[1] === :coloncall
        # Literal ranges validate through the slice-A `y[a:b]` path
        # (start-1, literal endpoints, bind-time cover check).
        return Expr[Expr(:call, :.~, Expr(:ref, col, rkind[2]), obj)]
    end
    rcol = rkind[2]
    rcol === col || _sfail("plate over `$(rkind[1])($rcol)` cannot " *
                           "sample `$col[$ivar]` (one response column " *
                           "per range)")
    return Expr[Expr(:call, :.~, col, obj)]
end

# The per-cell latent's size follows the plate range: a literal `1:N` rides as
# a UnitRange (validated to cover 1:n_obs at bind), eachindex/axes ⇒ n_obs.
_plate_param_range(name::Symbol, rkind) =
    rkind[1] === :coloncall ? _lower_lhs_range(name, rkind[2]) : nothing

function _cell_lhs_error(lhs, ivar)
    lhs isa Expr && lhs.head === :ref || return _sfail(
        "cell responses sample `name[$ivar]` exactly — got $(repr(lhs))")
    length(lhs.args) == 2 || return _sfail(
        "one-dimensional cell refs only (`v[$ivar]`)")
    lhs.args[1] isa Symbol || return _sfail(
        "cell refs index a bare column (`v[$ivar]`)")
    idx = lhs.args[2]
    idx isa Expr && idx.head === :ref && return _sfail(
        "factor indexing inside plates is not in slice B")
    return _sfail("cell reads index the loop variable exactly " *
                  "(`v[$ivar]`) — got $(repr(lhs)) (cross-index reads " *
                  "need `@scan`, reserved)")
end

# Value-position symbols of a cell expression, validating the loop-variable
# discipline on the way: refs are exactly `v[i]`, and `i` appears only as
# an index. Function heads and kw names are positions, not refs.
function _cell_bares(ex, ivar)
    bares = Set{Symbol}()
    _cell_bares!(ex, ivar, bares)
    return bares
end

function _cell_bares!(ex::Symbol, ivar, bares)
    ex === ivar && _sfail("loop variable `$ivar` appears only as an " *
                          "index (`v[$ivar]`)")
    push!(bares, ex)
    return nothing
end
_cell_bares!(ex, ivar, bares) = nothing
function _cell_bares!(ex::Expr, ivar, bares)
    if ex.head === :ref
        (length(ex.args) == 2 && ex.args[1] isa Symbol &&
            ex.args[2] === ivar) || _cell_lhs_error(ex, ivar)
        return nothing
    end
    if ex.head === :call
        for a in ex.args[2:end]
            _cell_bares!(a, ivar, bares)
        end
        return nothing
    end
    if ex.head === :.
        start = length(ex.args) >= 1 && ex.args[1] isa Symbol ? 2 : 1
        for a in ex.args[start:end]
            _cell_bares!(a, ivar, bares)
        end
        return nothing
    end
    if ex.head === :kw
        for a in ex.args[2:end]
            _cell_bares!(a, ivar, bares)
        end
        return nothing
    end
    for a in ex.args
        _cell_bares!(a, ivar, bares)
    end
    return nothing
end

# Strip exact `v[i]` refs to whole columns (validation ran first).
_strip_cell(ex, ivar) = ex
_strip_cell(s::Symbol, ivar) = s
function _strip_cell(ex::Expr, ivar)
    ex.head === :ref && return ex.args[1]
    return Expr(ex.head, (_strip_cell(a, ivar) for a in ex.args)...)
end

# Post-analysis whole-vector check: bare cell symbols denoting vectors
# (data, predictors, derived, factor coefs) needed `[i]`. Runs after the
# main loop, when predictors and derived are known.
function _check_plate_bares(ctx, data, prednames, derivedkeys, factorcoefs,
        plate_names = Set{Symbol}())
    isempty(ctx) && return nothing
    # A per-cell latent VECTOR read bare in an observation cell also needs
    # `[i]` (it is a vector like data/predictors/derived).
    vecs = union(data, prednames, derivedkeys, factorcoefs, plate_names)
    for (lhs, line, bares) in ctx
        at = line > 0 ? " (line $line)" : ""
        for b in sort!(collect(bares))
            b in vecs && _sfail("`@plate`$at: bare `$b` reads a whole " *
                                "vector in a cell — index it (`$b[i]`)")
        end
    end
    return nothing
end

function _factor_coefs(coefuse, predictors)
    out = Set{Symbol}()
    for pred in predictors, t in pred.terms
        t.kind === FactorTerm || continue
        for (nm, uses) in coefuse, u in uses
            u[1] === pred.name && u[2] in t.columns && push!(out, nm)
        end
    end
    return out
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

# Sampling-statement LHS: a bare Symbol, a one-dimensional range ref
# `y[R]` (`.~` only), or a levels ref `c[levels(g)]` / `c[levels(g)][S]`
# (`.~` only). Returns `(column, range, levels)` with at most one of
# `range` / `levels` set.
_sample_lhs(lhs::Symbol, bc, tilde, data) = (lhs, nothing, nothing)
function _sample_lhs(lhs, bc, tilde, data)
    lhs isa Expr || _sfail("$tilde left-hand side must be a bare Symbol, " *
                           "a range ref (`y[1:N]`), or a levels ref " *
                           "(`c[levels(g)]`), got $(repr(lhs))")
    lhs.head === :. && _sfail("dotted left-hand side $(repr(lhs)) does " *
                              "not lower (nested targets are out of scope)")
    lhs.head === :ref && length(lhs.args) == 2 || _sfail(
        "$tilde left-hand side must be a bare Symbol or a one-dimensional " *
        "ref (`y[1:N]`, `c[levels(g)]`), got $(repr(lhs))")
    target, index = lhs.args
    # Chained outside subset (`c[levels(g)][2:end]`): one way to write it —
    # the subset goes inside (`c[levels(g)[2:end]]`).
    target isa Expr && _sfail("$tilde subset goes inside the levels " *
                              "expression (`c[levels(g)[2:end]]`), got " *
                              "$(repr(lhs))")
    target isa Symbol || _sfail("$tilde left-hand side must be a bare " *
                                "Symbol or a one-dimensional ref, got " *
                                "$(repr(lhs))")
    # `c[levels(g)[S]]`: subset selection over the levels.
    if index isa Expr && index.head === :ref
        bc || _sfail("sized prior `$(target)[levels(...)[...]]` is a " *
                     "vector — use `.~`, not `~`")
        target in data && _sfail("`levels` sizes coefficient priors, not " *
                                 "responses ($target is data)")
        gcol, sub = _levels_subset_index(target, index, data)
        return target, nothing, (gcol, sub)
    end
    if _is_levels_call(index)
        bc || _sfail("sized prior `$(target)[levels(...)]` is a vector — " *
                     "use `.~`, not `~`")
        target in data && _sfail("`levels` sizes coefficient priors, not " *
                                 "responses ($target is data)")
        gcol = _levels_column(target, index, data)
        return target, nothing, (gcol, Colon())
    end
    bc || _sfail("sliced response `$target[...]` is a vector — " *
                 "use `.~`, not `~`")
    return target, _lower_lhs_range(target, index), nothing
end

# The `levels(g)[S]` index of a subset prior: returns `(g, subset)`.
function _levels_subset_index(col::Symbol, index::Expr, data::Set{Symbol})
    length(index.args) == 2 && _is_levels_call(index.args[1]) ||
        _sfail("coefficient $col: subsets select over `levels` " *
               "(`$col[levels(g)[2:end]]`), got $(repr(index))")
    gcol = _levels_column(col, index.args[1], data)
    return gcol, _lower_levels_subset(col, gcol, index.args[2])
end

_is_levels_call(x) =
    x isa Expr && x.head === :call && !isempty(x.args) &&
    x.args[1] isa Symbol && x.args[1] in (:levels, :unique, :sort)

function _levels_column(col::Symbol, call::Expr, data::Set{Symbol})
    fn = call.args[1]
    fn === :levels || _sfail("coefficient $col: write `levels(...)`, not " *
                             "`$fn(...)` (the levels function is `levels`)")
    length(call.args) == 2 ||
        _sfail("coefficient $col: `levels` takes exactly one grouping " *
               "column, got $(repr(call))")
    gcol = call.args[2]
    gcol isa Symbol || _sfail("coefficient $col: `levels` takes a bare " *
                              "grouping column, got $(repr(gcol))")
    gcol in data || _sfail("coefficient $col: `levels($gcol)` needs a " *
                           "data grouping column — $gcol is not data")
    return gcol
end

# Subset selections over `levels(g)`: `2:end`, literal `a:b`, or literal
# `[i, j]` (full cover is the bare `c[levels(g)]` — no `[:]` sugar).
# Returns the LevelMap subset value.
function _lower_levels_subset(col::Symbol, gcol::Symbol, s)
    if s isa Expr && s.head === :call && !isempty(s.args) && s.args[1] === :(:)
        length(s.args) == 3 || _sfail("coefficient $col: level subsets " *
                                      "are `a:b` or `a:end`, got $(repr(s))")
        lo, hi = s.args[2], s.args[3]
        lo isa Integer && !(lo isa Bool) && lo >= 1 || _sfail(
            "coefficient $col: subset must start at a literal 1-based " *
            "position, got $(repr(lo))")
        hi isa Integer && !(hi isa Bool) && return _subset_range(col, lo, hi)
        hi === :end && return (Int(lo), :end)
        return _sfail("coefficient $col: subset endpoint must be a " *
                      "literal or `end`, got $(repr(hi))")
    end
    if s isa Expr && s.head === :vect
        all(a -> a isa Integer && !(a isa Bool) && a >= 1, s.args) ||
            _sfail("coefficient $col: level index lists take literal " *
                   "1-based positions, got $(repr(s))")
        !isempty(s.args) ||
            _sfail("coefficient $col: level index list is empty")
        return Int.(s.args)
    end
    return _sfail("coefficient $col: level subsets are `[2:end]`, " *
                  "`[a:b]`, or `[[i, j]]`, got $(repr(s))")
end

function _subset_range(col::Symbol, lo::Integer, hi::Integer)
    lo <= hi || _sfail("coefficient $col: level range $lo:$hi is empty")
    return UnitRange(Int(lo), Int(hi))
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
column: bare LHS, `eachindex`, `axes`). `levels` carries a
`(grouping column, subset)` pair for `c[levels(g)]` broadcast priors
(`nothing` otherwise)."""
struct SampleStmt
    lhs::Symbol
    rhs::Any
    broadcast::Bool
    range::Union{Nothing,UnitRange{Int}}
    levels::Any
end
SampleStmt(lhs::Symbol, rhs, broadcast::Bool) =
    SampleStmt(lhs, rhs, broadcast, nothing, nothing)

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
        m === Symbol("@plate") && _sfail(
            "`@plate` is only admitted at model top level (nested plates " *
            "do not lower)")
        m === Symbol("@scan") && _sfail(
            "`@scan` must be a top-level model statement (a `@scan begin … end` " *
            "block), not nested inside another macro or statement")
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

# ── Submodel expansion (StanBlocks-style reusable submodels) ─────────────
# Runs before partitioning. Every plain-`~` statement whose RHS call head
# resolves in `mod` to an `RKPPLSubmodel` is rewritten into inline statements;
# the submodel's positional args bind to the call arguments and its own `~`/`=`
# names are namespaced under the LHS, so the result lowers exactly like a
# hand-inlined model (the submodel is transparent). Two kinds, read from the
# submodel's RETURN:
#   • latent (`latent ~ sm(a,b)`, `latent` NOT data) — returns a VALUE
#     expression; every local is namespaced (`latent_…`) and a trailing
#     `latent = <return>` binds the LHS.
#   • observation stream (`y ~ sm(a,b)`, `y` a data column) — returns a bare
#     `slot` that is the LHS of an internal `slot .~ family.(...)` response;
#     the slot maps to the data column (`y .~ family.(y_…)`), the rest is
#     namespaced under `y`, and there is NO trailing binding (the response IS
#     the binding).
#
# Admitting plain `~` on a data LHS is a SHAPE-COMPATIBILITY rule (dots follow
# the callee, not the LHS), NOT a submodel type-exception: a data column is
# vector-shaped, so it accepts only a VECTORIZED callee whose logpdf consumes
# the whole column — a stream submodel. A scalar callee (`y ~ Normal(...)`)
# stays incompatible and is rejected downstream (`.~` required), and a latent
# submodel on a data LHS is likewise incompatible. Two shapes are deliberately
# left for later and NOT handled here: a scalar-observation LHS + scalar callee
# (plain `~`, once scalar data columns exist) and an elementwise per-row
# submodel broadcast as `y .~ sm.(x, g)`.
#
# A `~` whose head is a distribution or unknown name passes through untouched —
# the ordinary vocabulary/response screen still applies downstream.

"""Return the [`RKPPLSubmodel`](@ref) a call RHS names in `mod`, or nothing."""
function _resolve_submodel(rhs, mod::Module)
    rhs isa Expr && rhs.head === :call && !isempty(rhs.args) || return nothing
    head = rhs.args[1]
    head isa Symbol || return nothing
    isdefined(mod, head) || return nothing
    val = getfield(mod, head)
    return val isa RKPPLSubmodel ? val : nothing
end

# A top-level statement that is a plain `~` whose RHS resolves to a submodel
# (pure predicate; the latent/stream compatibility check happens on expansion).
function _stmt_is_submodel_call(arg, mod::Module)
    arg isa Expr || return false
    st = try
        _unwrap_trivia(arg)
    catch
        return false
    end
    st isa Expr && _is_sample(st) || return false
    _resolve_submodel(st.args[3], mod) === nothing && return false
    return st.args[2] isa Symbol
end

function _expand_submodels(ast::Expr, data::Set{Symbol}, mod::Module)
    any(_stmt_is_submodel_call(a, mod) || _plate_has_submodel_cell(a, mod)
        for a in ast.args) || return ast
    out = Any[]
    for arg in ast.args
        if _stmt_is_submodel_call(arg, mod)
            st = _unwrap_trivia(arg)
            append!(out, _expand_one_submodel(st.args[2], st.args[3], mod, data))
        elseif _plate_has_submodel_cell(arg, mod)
            push!(out, _expand_plate_cell_submodels(arg, mod, data))
        else
            push!(out, arg)
        end
    end
    return Expr(:block, out...)
end

_ns(lhs::Symbol, nm::Symbol) = Symbol(lhs, :_, nm)

# Split a submodel body into (statements, return-expression).
function _submodel_body_parts(sm::RKPPLSubmodel)
    items = Any[a for a in sm.body.args if !(a isa LineNumberNode)]
    isempty(items) && _sfail("submodel `$(sm.name)` has an empty body")
    ret = last(items)
    stmts = Any[_unwrap_trivia(st) for st in items[1:end-1]]
    for st in stmts
        (st isa Expr && (_is_sample(st) || _is_broadcast_sample(st) ||
            (st.head === :(=) && length(st.args) == 2 &&
             st.args[1] isa Symbol))) || _sfail(
            "submodel `$(sm.name)`: statement `$(repr(st))` is not a `~`/`=` " *
            "form (submodels are straight-line; the body must end in a return " *
            "expression)")
    end
    (ret isa Expr && (_is_sample(ret) || _is_broadcast_sample(ret) ||
        (ret.head === :(=) && length(ret.args) == 2))) && _sfail(
        "submodel `$(sm.name)` must end in a RETURN expression bound to the " *
        "use-site LHS (a bare value, not a `~`/`=` statement)")
    return stmts, ret
end

_stmt_lhs(st::Expr) =
    (_is_sample(st) || _is_broadcast_sample(st)) ? st.args[2] : st.args[1]

# Reject a nested submodel call (single-level submodels this slice).
function _reject_nested_submodel(sm::RKPPLSubmodel, st::Expr)
    rhs = (_is_sample(st) || _is_broadcast_sample(st)) ? st.args[3] : st.args[2]
    _resolve_submodel(rhs, sm.mod) === nothing || _sfail(
        "submodel `$(sm.name)` calls another submodel — nested submodels are " *
        "a follow-up slice; inline it for now")
end

# Substitute Symbols per `map` everywhere except inside QuoteNodes.
_subst(ex::Symbol, map::AbstractDict) = get(map, ex, ex)
_subst(ex::QuoteNode, ::AbstractDict) = ex
_subst(ex, ::AbstractDict) = ex
_subst(ex::Expr, map::AbstractDict) =
    Expr(ex.head, Any[_subst(a, map) for a in ex.args]...)

# A stream submodel returns a bare `slot` Symbol that is the LHS of exactly one
# internal `.~` response statement. Returns that statement, or `nothing` for a
# latent submodel (value return / non-response slot).
function _stream_response(sm::RKPPLSubmodel, stmts, ret)
    ret isa Symbol || return nothing
    hits = findall(st -> _is_broadcast_sample(st) && st.args[2] === ret, stmts)
    isempty(hits) && return nothing
    length(hits) == 1 || _sfail("stream submodel `$(sm.name)`: return `$ret` " *
        "names more than one `.~` response")
    return stmts[first(hits)]
end

function _expand_one_submodel(lhs::Symbol, callexpr::Expr, mod::Module,
                              data::Set{Symbol})
    sm = _resolve_submodel(callexpr, mod)::RKPPLSubmodel
    callargs = callexpr.args[2:end]
    length(callargs) == length(sm.argnames) || _sfail(
        "submodel `$(sm.name)` expects $(length(sm.argnames)) argument(s) " *
        "$(Tuple(sm.argnames)), got $(length(callargs)) at `$lhs ~ " *
        "$(sm.name)(...)`")
    stmts, ret = _submodel_body_parts(sm)
    stream = _stream_response(sm, stmts, ret) !== nothing
    # Shape compatibility (dots follow the callee): a data column is
    # vector-shaped, so it admits only a vectorized (stream) callee; a non-data
    # LHS binds a latent value.
    if lhs in data
        stream || _sfail("`$lhs` is a data column, so `$(sm.name)` must be an " *
            "observation-stream submodel: end its body by returning a `slot` " *
            "that is the LHS of an internal `slot .~ family.(...)` response. " *
            "`$(sm.name)` returns a value (latent) — use a non-data LHS.")
    else
        stream && _sfail("`$(sm.name)` is an observation-stream submodel (it " *
            "returns the `.~` response slot `$ret`); bind it to a DATA column " *
            "(`<data> ~ $(sm.name)(...)`), not the non-data name `$lhs`.")
    end
    argset = Set{Symbol}(sm.argnames)
    submap = Dict{Symbol,Any}()
    for (a, v) in zip(sm.argnames, callargs)
        submap[a] = v
    end
    for st in stmts
        nm = _stmt_lhs(st)
        nm in argset && _sfail("submodel `$(sm.name)`: `$nm` is both an " *
            "argument and a local statement — rename the local")
        haskey(submap, nm) && _sfail("submodel `$(sm.name)`: `$nm` is " *
            "assigned twice")
        # The stream response slot binds to the data LHS; all other locals are
        # namespaced under the LHS.
        submap[nm] = (stream && nm === ret) ? lhs : _ns(lhs, nm)
    end
    out = Any[]
    for st in stmts
        _reject_nested_submodel(sm, st)
        push!(out, _subst(st, submap))
    end
    # Latent: bind the LHS to the return value. Stream: the response IS the
    # binding (the data LHS is already bound), so no trailing assignment.
    stream || push!(out, Expr(:(=), lhs, _subst(ret, submap)))
    return out
end

# ── Per-cell submodel promotion inside `@plate` ──────────────────────────
# A plate cell may embed a submodel, `col[i] ~ sm(args…)`, mirroring StanBlocks'
# per-cell submodel promotion (a `plate` do-block embedding `lhs ~ submodel(…)`).
# Runs here, before partitioning, alongside the top-level submodel expansion:
# each submodel-call cell is inlined into `i`-indexed cell statements, so the
# submodel's own `~`/`=` names promote PER CELL (namespaced under `col`) and its
# return binds to `col[i]`. The rewritten plate then flows through the ordinary
# plate desugar, so a per-cell submodel lowers exactly like the hand-inlined
# per-cell program (transparent) — a centered latent (`col[i] ~ dist`), a
# non-centered transform (`z[i] ~ dist; col[i] = f(z[i])`), or a per-cell
# observation stream (`col` a data column). No new IR: the inlined statements
# reduce to the per-cell parameter / derived-cell / observation shapes the
# primitive already supports.

# A plate cell that is a per-cell submodel call `col[i] ~ sm(…)`: returns
# `(colref, callexpr)` with `colref === Expr(:ref, col, i)`, else nothing. Only a
# scalar `~` whose LHS is the loop-variable-indexed column and whose RHS head
# resolves to a submodel qualifies; every other cell passes through untouched to
# the ordinary cell desugar.
function _cell_submodel_call(c, ivar::Symbol, mod::Module)
    c isa Expr && _is_sample(c) || return nothing
    lhs = c.args[2]
    (lhs isa Expr && lhs.head === :ref && length(lhs.args) == 2 &&
        lhs.args[1] isa Symbol && lhs.args[2] === ivar) || return nothing
    _resolve_submodel(c.args[3], mod) === nothing && return nothing
    return (lhs, c.args[3])
end

# True for a top-level `@plate for i in R … end` macrocall with at least one
# per-cell submodel-call cell. Only peeks for submodel cells (the plate's exact
# structural validation stays in `_desugar_plate`); a malformed plate returns
# false here and errors later with the canonical message.
function _plate_has_submodel_cell(arg, mod::Module)
    arg isa Expr && arg.head === :macrocall && !isempty(arg.args) &&
        arg.args[1] === Symbol("@plate") || return false
    loop = arg.args[end]
    (loop isa Expr && loop.head === :for && length(loop.args) == 2) || return false
    asg = loop.args[1]
    (asg isa Expr && asg.head === :(=) && length(asg.args) == 2 &&
        asg.args[1] isa Symbol) || return false
    body = loop.args[2]
    body isa Expr && body.head === :block || return false
    return any(_cell_submodel_call(c, asg.args[1], mod) !== nothing
        for c in body.args)
end

# Rewrite a `@plate` block, replacing each per-cell submodel-call cell with the
# submodel's inlined `i`-indexed cell statements; other cells pass through.
function _expand_plate_cell_submodels(pl::Expr, mod::Module, data::Set{Symbol})
    loop = pl.args[end]::Expr
    asg = loop.args[1]::Expr
    ivar = asg.args[1]::Symbol
    body = loop.args[2]::Expr
    cells = Any[]
    for c in body.args
        call = _cell_submodel_call(c, ivar, mod)
        call === nothing ? push!(cells, c) :
            append!(cells, _expand_cell_submodel(call[1], call[2], ivar, mod, data))
    end
    newloop = Expr(:for, asg, Expr(:block, cells...))
    return Expr(:macrocall, pl.args[1:end-1]..., newloop)
end

_is_dotted_obj(st::Expr) =
    _is_sample(st) && st.args[3] isa Expr && st.args[3].head === :.

# Inline one per-cell submodel call `col[i] ~ sm(callargs…)` into a sequence of
# `i`-indexed cell statements. The submodel's own `~`/`=` names are namespaced
# under `col` and indexed per cell (`col_<nm>[i]`); its positional args bind to
# the call arguments (written per-cell by the user, e.g. `x[i]`). The RETURN
# selects what binds to `col[i]`:
#   • a bare Symbol naming exactly one internal statement → that statement's LHS
#     binds DIRECTLY to `col[i]` (a clean centered latent `col[i] ~ dist(…)`, a
#     non-centered derived `col[i] = …`, or an observation slot `col[i] ~
#     dist.(…)` on a data column), no trailing binding;
#   • any other return → every internal name namespaces and a trailing
#     `col[i] = <return>` binds the cell (a compound latent transform).
# Shape compatibility (dots follow the callee, exactly as at top level): a data
# column admits only an observation-slot submodel; a non-data column binds a
# latent. The dotted/undotted distinction is finally enforced by the cell
# desugar the inlined statements flow through.
function _expand_cell_submodel(colref::Expr, callexpr::Expr, ivar::Symbol,
                               mod::Module, data::Set{Symbol})
    col = colref.args[1]::Symbol
    sm = _resolve_submodel(callexpr, mod)::RKPPLSubmodel
    callargs = callexpr.args[2:end]
    length(callargs) == length(sm.argnames) || _sfail(
        "submodel `$(sm.name)` expects $(length(sm.argnames)) argument(s) " *
        "$(Tuple(sm.argnames)), got $(length(callargs)) at `$col[$ivar] ~ " *
        "$(sm.name)(...)`")
    stmts, ret = _submodel_body_parts(sm)
    # The direct-bound slot: a bare-Symbol return naming exactly one internal
    # statement, whose LHS binds to `col[i]`.
    slot = nothing
    if ret isa Symbol
        hits = findall(st -> _stmt_lhs(st) === ret, stmts)
        length(hits) > 1 && _sfail("submodel `$(sm.name)`: return `$ret` names " *
            "more than one statement")
        isempty(hits) || (slot = first(hits))
    end
    if col in data
        (slot !== nothing && _is_dotted_obj(stmts[slot])) || _sfail(
            "`$col` is a data column, so `$(sm.name)` must be a per-cell " *
            "observation submodel: return a `slot` that is the LHS of an " *
            "internal `slot ~ family.(...)` dotted response — `$(sm.name)` " *
            "returns " *
            (slot === nothing ? "a value" :
             _is_sample(stmts[slot]) ? "a scalar (latent) distribution" :
             "a derived assignment") *
            ", so use a non-data LHS.")
    elseif slot !== nothing && _is_dotted_obj(stmts[slot])
        _sfail("`$(sm.name)` is a per-cell observation submodel (its return " *
            "slot `$ret` is a `~ family.(...)` dotted response); bind it to a " *
            "DATA column (`<data>[$ivar] ~ $(sm.name)(...)`), not the non-data " *
            "name `$col`.")
    end
    # Build the substitution: args → call args; each internal name → its indexed
    # namespaced ref, except the direct-bound slot → `col[i]`.
    argset = Set{Symbol}(sm.argnames)
    submap = Dict{Symbol,Any}()
    for (a, v) in zip(sm.argnames, callargs)
        submap[a] = v
    end
    for (k, st) in enumerate(stmts)
        nm = _stmt_lhs(st)
        nm in argset && _sfail("submodel `$(sm.name)`: `$nm` is both an " *
            "argument and a local statement — rename the local")
        haskey(submap, nm) && _sfail("submodel `$(sm.name)`: `$nm` is " *
            "assigned twice")
        submap[nm] = k == slot ? Expr(:ref, col, ivar) :
            Expr(:ref, _ns(col, nm), ivar)
    end
    out = Any[]
    for st in stmts
        _reject_nested_submodel(sm, st)
        push!(out, _subst(st, submap))
    end
    # Compound-return latent: bind `col[i]` to the substituted return value.
    slot === nothing && push!(out, Expr(:(=), Expr(:ref, col, ivar),
        _subst(ret, submap)))
    return out
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
    elseif head === :macrocall && !isempty(st.args) &&
            st.args[1] === Symbol("@plate")
        return _sfail("docstrings on `@plate` blocks do not lower")
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
    family, lik_link, pred_link, loc, scale, trials =
        _lower_response_base(lhs, call, ctx)
    pname = _lower_location(lhs, loc, pred_link, ctx, predictors, pred_idx,
        coefuse)
    return LikelihoodSpec(family, lik_link, lhs, pname, scale, weights,
        evidence, Symbol(lhs, "_resp"), trials, range)
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
            (:Normal, :Bernoulli, :Poisson, :Binomial, :NegativeBinomial2,
                :Gamma, :weighted, :truncated, :censored,
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
    elseif f === :Binomial && i == 2
        return _dot2call_nested_link(lhs, a, f)
    elseif f === :NegativeBinomial2 && i == 1
        return _dot2call_nested_link(lhs, a, f)
    end
    # Gamma position 2 (`exp.(eta) ./ alpha`) passes through; the
    # response branch matches the `./` structure (link + alpha identity).
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
    want = (base === :Bernoulli || base === :Binomial) ? :logistic : :exp
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
    "`Bernoulli.(logistic.(eta))`, `Poisson.(exp.(eta))`, " *
    "`Binomial.(n, logistic.(mu))`, `NegativeBinomial2.(exp.(eta), phi)` " *
    "or `Gamma.(alpha, exp.(eta) ./ alpha)`"

function _lower_response_base(lhs, rhs::Expr, ctx)
    rhs.head === :call || _sfail("response $lhs: $_RESPONSE_BASE_MSG; " *
                                 "got $(repr(rhs))")
    fam = rhs.args[1]
    fam === :weighted &&
        _sfail("`weighted.(...)` goes outermost: " *
               "`y .~ weighted.(Normal.(mu, sigma), w)`")
    fam in (:Normal, :Bernoulli, :Poisson, :Binomial, :NegativeBinomial2,
        :Gamma) || return _lower_response_base_error(lhs, rhs, fam)
    args = _plain_args(rhs, "`$fam`")
    if fam === :Normal
        length(args) == 2 || _sfail("response $lhs: `Normal` takes " *
                                    "`Normal.(mu, sigma)`")
        return GaussianFam, IdentityLink, IdentityLink, args[1],
        _lower_scale(lhs, args[2], ctx), nothing
    elseif fam === :Bernoulli
        length(args) == 1 || _sfail("response $lhs: `Bernoulli` takes " *
                                    "`Bernoulli.(logistic.(eta))`")
        return BernoulliLogitFam, LogitLink, IdentityLink,
        _lower_link_arg(lhs, args[1], :logistic), nothing, nothing
    elseif fam === :Binomial
        length(args) == 2 || _sfail("response $lhs: `Binomial` takes " *
                                    "`Binomial.(n, logistic.(mu))`")
        return BinomialLogitFam, LogitLink, IdentityLink,
        _lower_link_arg(lhs, args[2], :logistic), nothing,
        _lower_trials(lhs, args[1], ctx)
    elseif fam === :NegativeBinomial2
        length(args) == 2 || _sfail("response $lhs: `NegativeBinomial2` takes " *
                                    "`NegativeBinomial2.(exp.(eta), phi)`")
        return NegativeBinomial2Fam, LogLink, LogLink,
        _lower_link_arg(lhs, args[1], :exp),
        _lower_scale(lhs, args[2], ctx), nothing
    elseif fam === :Gamma
        loc, scale = _lower_gamma_args(lhs, args, ctx)
        return GammaLogFam, LogLink, LogLink, loc, scale, nothing
    else
        length(args) == 1 || _sfail("response $lhs: `Poisson` takes " *
                                    "`Poisson.(exp.(eta))`")
        return PoissonLogFam, LogLink, LogLink,
        _lower_link_arg(lhs, args[1], :exp), nothing, nothing
    end
end

function _lower_trials(lhs, t, ctx)
    t isa Bool && _sfail("response $lhs trials must be an Int data " *
                         "column or Int literal, got Bool")
    t isa Integer && return Int(t)
    t isa Real && _sfail("response $lhs trials must be an Int data " *
                         "column or Int literal, got $(repr(t))")
    if t isa Symbol
        t in ctx.data && return t
        t in ctx.vecdefs && _sfail(
            "response $lhs trials column $t is derived — slice-1 binds " *
            "trials raw (derived trials need shape metadata — planned)")
        return _sfail("response $lhs trials $(repr(t)) must be an Int " *
                      "data column or Int literal")
    end
    return _sfail("response $lhs trials must be an Int data column or " *
                  "Int literal, got $(repr(t))")
end

function _lower_gamma_args(lhs, args, ctx)
    length(args) == 2 || _sfail("response $lhs: `Gamma` takes " *
                                "`Gamma.(alpha, exp.(eta) ./ alpha)`")
    a1, div = args
    div isa Expr && div.head === :call && length(div.args) == 3 &&
        div.args[1] === Symbol("./") ||
        _sfail("response $lhs: `Gamma` takes " *
               "`Gamma.(alpha, exp.(eta) ./ alpha)`")
    loc = _lower_link_arg(lhs,
        _dot2call_nested_link(lhs, div.args[2], :Gamma), :exp)
    a2 = div.args[3]
    _same_gamma_alpha(a1, a2) || _sfail(
        "response $lhs: both `Gamma` positions must name the same alpha " *
        "(got $(repr(a1)) and $(repr(a2)))")
    return loc, _lower_scale(lhs, a1, ctx)
end

_same_gamma_alpha(a, b) =
    a isa Symbol && b isa Symbol ? a === b :
    a isa Real && b isa Real ? a == b : false

function _lower_response_base_error(lhs, rhs, fam)
    fam in (:normal, :bernoulli, :poisson, :binomial, :gamma) && _sfail(
        "response $lhs: use Distributions.jl constructors " *
        "(`Normal`, not `normal`)")
    fam === :negative_binomial2 && _sfail("response $lhs: use " *
                                          "`NegativeBinomial2` (the response " *
                                          "spelling, not the kernel endpoint)")
    fam === :BernoulliLogit && _sfail("response $lhs: write " *
                                      "`Bernoulli.(logistic.(eta))`")
    fam === :PoissonLog && _sfail("response $lhs: write " *
                                  "`Poisson.(exp.(eta))`")
    fam === :BinomialLogit && _sfail("response $lhs: write " *
                                     "`Binomial.(n, logistic.(mu))`")
    fam === :NegativeBinomial2Log && _sfail("response $lhs: write " *
        "`NegativeBinomial2.(exp.(eta), phi)`")
    fam === :GammaLog && _sfail("response $lhs: write " *
                                "`Gamma.(alpha, exp.(eta) ./ alpha)`")
    return _sfail("response $lhs: unknown distribution `$(repr(fam))` " *
                  "(admitted: Normal, Bernoulli, Poisson, Binomial, " *
                  "NegativeBinomial2, Gamma). When `$fam` is a " *
                  "defined RKPPLSubmodel, a latent uses `latent ~ $fam(...)` " *
                  "and an observation stream uses plain `$lhs ~ $fam(...)` " *
                  "(the whole-column vectorized callee); an elementwise " *
                  "per-row submodel broadcast `$lhs .~ $fam.(...)` is a " *
                  "follow-up slice.")
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
    # A per-cell latent VECTOR is the whole location via a LatentTerm predictor
    # (`lp = theta`, identity design; the latent's prior lives on its
    # PlateParameter, so no coefficient use is recorded). Two spellings: a bare
    # latent (`y[i] ~ Normal.(theta[i], s)`) or a deterministic transform of one
    # (`theta[i] = mu .+ tau .* z[i]` then `y[i] ~ Normal.(theta[i], s)` — the
    # non-centered / latent-transform shape, emitted as a derived column and
    # referenced directly). A derived location with NO latent stays a design
    # predictor; mixed latent+fixed BARE-expression locations are a later slice.
    if loc isa Symbol && loc in ctx.plate_names
        return _latent_predictor!(lhs, loc, pred_link, ctx, predictors, pred_idx)
    end
    if loc isa Symbol && haskey(ctx.detmap, loc) && _derived_reads_latent(loc, ctx)
        return _latent_predictor!(lhs, loc, pred_link, ctx, predictors, pred_idx)
    end
    if loc isa Symbol
        # A scan-state latent vector is a direct per-observation location: the
        # response mean IS the carried state (no linear predictor). Admitted
        # family/link is checked in `_validate_responses` (Gaussian-identity, v1).
        loc in ctx.scan_states && return loc
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
        _reject_plate_in_predictor(lhs, loc, ctx)
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

# Build the LatentTerm location predictor over `col` (a plate parameter or a
# derived column that reads one); the generator emits `lp = col`.
function _latent_predictor!(lhs, col, pred_link, ctx, predictors, pred_idx)
    pname = Symbol(lhs, "_loc")
    (haskey(ctx.detmap, pname) || pname in ctx.plate_names) && _sfail(
        "latent-location predictor name $pname collides with your " *
        "definition — rename it")
    term = TermSpec(LatentTerm, [col], NamedTuple(), col, Symbol(col, "_lat"))
    push!(predictors, PredictorSpec(pname, pred_link, [term], pname))
    pred_idx[pname] = length(predictors)
    return pname
end

# Does a derived column transitively read a per-cell latent (plate parameter)?
# If so, its value is a latent transform (e.g. non-centered `mu .+ tau .* z`),
# not a design predictor — it stays a derived column used directly as the LP.
function _derived_reads_latent(name::Symbol, ctx)
    haskey(ctx.detmap, name) || return name in ctx.plate_names
    seen = Set{Symbol}((name,))
    stack = collect(_value_symbols(ctx.detmap[name]))
    while !isempty(stack)
        s = pop!(stack)
        s in seen && continue
        push!(seen, s)
        s in ctx.plate_names && return true
        haskey(ctx.detmap, s) && append!(stack, _value_symbols(ctx.detmap[s]))
    end
    return false
end

function _lower_location_symbol_error(lhs, loc, ctx)
    loc in ctx.data && _sfail("response $lhs location is the data column " *
                              "$loc — locations must be predictors with " *
                              "estimated coefficients (wrap: " *
                              "`eta = a .+ b .* $loc`)")
    loc in ctx.prior_names && _sfail(
        "response $lhs location is the bare scalar parameter $loc — a " *
        "per-observation latent is a per-cell parameter (`@plate for i ...; " *
        "$loc[i] ~ Normal(mu, tau); y[i] ~ Normal.($loc[i], s); end`); a " *
        "scalar parameter cannot vary per observation")
    return _sfail("response $lhs location $loc is not a predictor " *
                  "definition (`$loc = ...` affine in data)")
end

# A per-cell latent is a bare whole location in slice-1; a latent buried in a
# predictor expression (mixed latent + fixed effects) is a later increment.
function _reject_plate_in_predictor(lhs, loc, ctx)
    for s in _value_symbols(loc)
        s in ctx.plate_names && _sfail(
            "response $lhs location $(repr(loc)) combines the per-cell latent " *
            "$s with other predictor structure — a latent is a bare location " *
            "in slice-1 (`y[i] ~ Normal.($s[i], s)`); mixed latent + " *
            "fixed-effect predictors are planned")
    end
    return nothing
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
    col = _factor_index(pname, idx, ctx)
    return TermSpec(FactorTerm, [col], NamedTuple(), col,
        Symbol(col, "_term")), (base, col, sign)
end

# Factor use is always bare `c[g]` over a raw data grouping column; the
# coefficient's `c[levels(g)]` broadcast prior sizes the full-rank block.
# The `treatment(g, ref)` vocabulary was BRM-specific contrasts machinery
# and is removed (F2: full-rank, no reference dropping).
function _factor_index(pname, idx, ctx)
    idx isa Symbol || _sfail("predictor $pname: factor index must be a " *
                             "bare data column (`c[g]`) — " *
                             _treatment_removed(idx))
    idx in ctx.vecdefs && _sfail("predictor $pname: factor over the " *
                                 "derived column $idx needs pre-evaluation " *
                                 "level knowledge — factors take raw " *
                                 "grouping columns in slice 1")
    idx in ctx.data || _sfail("predictor $pname: factor index $idx must " *
                              "be a data column")
    return idx
end

function _treatment_removed(idx)
    idx isa Expr && idx.head === :call && !isempty(idx.args) &&
        idx.args[1] === :treatment &&
        return "`treatment` was removed (BRM-specific contrasts); size " *
               "the vector with a broadcast prior " *
               "(`c[levels(g)] .~ ...`) and index a subset for " *
               "identified models"
    return "got $(repr(idx))"
end

# Coefficient priors: recovered by name from `coef ~ Normal(lit, lit)`
# statements; missing scalar priors default to Normal(0, 1) (emitter
# convention). Factor coefficients instead take broadcast priors
# (`c[levels(g)] .~ Normal.(lit, lit)`), which also size the block —
# required, never defaulted — and each one emits its LevelMap.
# Plan order follows predictors, addressees in term order.
function _lower_coefficient_priors(sample, coefuse, predictors)
    stated = Dict{Symbol,Any}()
    for s in sample
        haskey(coefuse, s.lhs) && (stated[s.lhs] = s)
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
    levelmaps = LevelMap[]
    for pred in predictors
        for t in pred.terms
            # Offsets carry no coefficient; latent terms carry a PlateParameter
            # whose prior lives on the plate parameter, not as a coefficient.
            (t.kind === OffsetTerm || t.kind === LatentTerm) && continue
            addr = t.kind === InterceptTerm ? :Intercept : only(t.columns)
            use = _find_use(coefuse, pred.name, addr)
            use === nothing && _sfail("internal: no coefficient use for " *
                                      "($(pred.name), $addr)")
            name = use[1]
            sign = use[3]
            if t.kind === FactorTerm
                push!(priors, _lower_factor_prior(pred, t, name, sign,
                    stated, levelmaps))
                continue
            end
            if !haskey(stated, name)
                push!(priors, PopulationPrior(pred.name, addr, 0.0, 1.0))
                continue
            end
            s = stated[name]
            s.levels !== nothing && _sfail("coefficient $name takes a " *
                                           "scalar prior (`$name ~ Normal`), " *
                                           "not a levels prior — it is used " *
                                           "as $(t.kind), not a factor")
            loc, scale = _coefficient_normal(name, s.rhs, pred.name, addr)
            push!(priors, PopulationPrior(pred.name, addr, sign * loc, scale))
        end
    end
    _check_identified(predictors, levelmaps)
    return priors, levelmaps
end

function _lower_factor_prior(pred, t, name, sign, stated, levelmaps)
    col = only(t.columns)
    haskey(stated, name) || _sfail("factor coefficient $name over $col " *
                                   "needs an explicit broadcast prior " *
                                   "(`$name[levels($col)] .~ Normal.(0, 1)`) " *
                                   "— the prior sizes the coefficient vector")
    s = stated[name]
    s.levels === nothing && _sfail("coefficient $name is vector-valued " *
                                   "(factor over $col) — scalar priors " *
                                   "cannot size it; write " *
                                   "`$name[levels($col)] .~ Normal.(0, 1)`")
    gcol, subset = s.levels
    gcol === col || _sfail("coefficient $name: levels column $gcol " *
                           "differs from use column $col")
    loc, scale = _coefficient_broadcast_normal(name, s.rhs, pred.name, col)
    push!(levelmaps, LevelMap(pred.name, col, [], :levels, subset))
    return PopulationPrior(pred.name, col, sign * loc, scale)
end

# Dotted coefficient priors peel to one shared (location, scale): broadcast
# args must be literals (per-level priors are not in slice 1).
function _coefficient_broadcast_normal(name, rhs, pname, col)
    rhs isa Expr && rhs.head === :. && length(rhs.args) == 2 &&
        rhs.args[1] === :Normal && rhs.args[2] isa Expr &&
        rhs.args[2].head === :tuple || _sfail(
            "coefficient $name of predictor $pname needs a broadcast " *
            "`Normal.(literal, literal)` prior, got $(repr(rhs))")
    args = rhs.args[2].args
    length(args) == 2 || _sfail("coefficient $name of predictor $pname " *
                                "needs `Normal.(location, scale)`")
    loc, scale = args
    loc isa Real || _sfail("coefficient $name prior location must be a " *
                           "literal (per-level priors are not in slice 1)")
    scale isa Real || _sfail("coefficient $name prior scale must be a " *
                             "literal (per-level priors are not in slice 1)")
    return Float64(loc), Float64(scale)
end

# Surface-side identifiability gate (the contract validator repeats it for
# hand-built plans): intercept + full-cover factor is unidentified.
function _check_identified(predictors, levelmaps)
    for pred in predictors
        any(t -> t.kind === InterceptTerm, pred.terms) || continue
        for t in pred.terms
            t.kind === FactorTerm || continue
            m = _find_levelmap(levelmaps, pred.name, only(t.columns))
            m !== nothing && m.subset === Colon() && _sfail(
                "predictor $(pred.name) is unidentified: intercept + " *
                "full-cover factor over $(only(t.columns)) (drop the " *
                "intercept or index a strict subset of levels)")
        end
    end
    return nothing
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
        s.levels !== nothing && _sfail("levels prior `$(s.lhs)[...]` is " *
                                       "never used in a predictor — size " *
                                       "only vectors the model indexes")
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
               "truncated). If `$fam` is meant as a submodel, define it with " *
               "`@rkppl $fam(args...) = begin ... end` and make it visible in " *
               "the lowering module (`mod=`).")
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
