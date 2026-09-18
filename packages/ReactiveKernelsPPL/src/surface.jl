# `@rkppl` authoring surface: StanBlocks-close model blocks lowering to
# data-free StructuralPlans.
#
# The shape mirrors StanBlocks `@slic` (block AST capture, `~` density
# statements, deterministic `=`, `model(; data...)` binding) under the
# standing constraints: Distributions.jl constructors (never Stan lowercase),
# immutable single-assignment top level, no control flow, no `target`,
# `@plate`/`@scan` reserved. Response-vs-parameter classification needs data
# names (a bare `lhs ~ Normal(...)` is shape-identical either way — the
# eight-schools indistinguishability), so the macro only captures; lowering
# runs at bind time with `data_names`. The BRM emitter targets the same
# `lower_rkppl(ast, data_names)` entry point with ASTs (no text seam).

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
`data_names` classifies every `~` LHS: in-data is a response, otherwise a
prior (sampled parameter or, when the name sits in a predictor coefficient
position, a population prior). Runs `validate_structure` before returning.
The BRM emitter calls this entry point directly with ASTs.
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
    prior_names = Set{Symbol}(lhs for (lhs, _) in sample if lhs ∉ data)
    ctx = (; data, detmap, prior_names, consumed = Set{Symbol}())
    responses = LikelihoodSpec[]
    predictors = PredictorSpec[]
    pred_idx = Dict{Symbol,Int}()
    coefuse = Dict{Symbol,Vector{Tuple{Symbol,Symbol,Int}}}()
    for (lhs, rhs) in sample
        lhs ∉ data && continue
        push!(responses,
            _lower_response(lhs, rhs, ctx, predictors, pred_idx, coefuse))
    end
    priors = _lower_coefficient_priors(sample, coefuse, predictors)
    params = _lower_parameters(sample, coefuse, ctx)
    used_locs = Set{Symbol}()
    for r in responses
        push!(used_locs, r.predictor)
    end
    assigns = AssignmentSpec[]
    for (nm, rhs) in det
        nm in used_locs && continue
        nm in ctx.consumed && _sfail("$nm is already inlined into a " *
                                     "predictor — remove the standalone `$nm = ...` " *
                                     "definition (row-varying values live only in " *
                                     "predictors in slice 1)")
        push!(assigns, _lower_assignment(nm, rhs, coefuse))
    end
    plan = StructuralPlan(responses, predictors, priors, params, assigns,
        Dict{Symbol,AbstractVector}(), 0)
    validate_structure(plan)
    return plan
end

# Top-level statements: skip line numbers and one leading docstring-to-be
# (ignored in slice 1); everything else must be an Expr.
function _partition_statements(ast::Expr, data::Set{Symbol})
    sample = Pair{Symbol,Any}[]
    det = Pair{Symbol,Any}[]
    seen = Set{Symbol}()
    seen_doc = false
    for arg in ast.args
        arg isa LineNumberNode && continue
        if arg isa String && !seen_doc && isempty(sample) && isempty(det)
            seen_doc = true
            continue
        end
        arg isa Expr || _sfail("stray literal $(repr(arg)) at model level " *
                               "(only `~`, `=` and reserved macros lower)")
        arg.head === :block &&
            _sfail("nested `begin` blocks do not lower — flatten the block")
        st = _unwrap_trivia(arg)
        if _is_sample(st)
            lhs = st.args[2]
            lhs isa Symbol || _sfail("`~` left-hand side must be a bare " *
                                     "Symbol, got $(repr(lhs))")
            _claim!(seen, lhs)
            _reject_target(st.args[3], lhs)
            push!(sample, lhs => st.args[3])
        elseif st.head === :(=) && length(st.args) == 2 && st.args[1] isa Symbol
            lhs = st.args[1]
            lhs === :target && _sfail("no `target` in rkppl models " *
                                      "(density comes only from `~`)")
            lhs in data && _sfail("$lhs is bound data and cannot be redefined")
            _claim!(seen, lhs)
            _reject_target(st.args[2], lhs)
            push!(det, lhs => st.args[2])
        else
            _reject_statement(st)
        end
    end
    return sample, det
end

_claim!(seen::Set{Symbol}, nm::Symbol) =
    nm in seen ?
    _sfail("single assignment: $nm is defined twice at model level") :
    push!(seen, nm)

_is_sample(st::Expr) =
    st.head === :call && length(st.args) == 3 && st.args[1] === :~

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
            "(shapes copied from StanBlocks; likelihood plates are implicit " *
            "— just write `y ~ Normal(mu, sigma)` over data)")
        m === Symbol("@.") && _sfail("explicit broadcast (`@.`) spelling is " *
                                     "undecided — write the predictor undotted " *
                                     "(`mu = a + b*x + c[g]`)")
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
        _sfail("no `target` in rkppl models (density comes only from `~`; " *
               "statement for $lhs mentions `target`)")
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
    elseif head === :call
        fn = isempty(st.args) ? "?" : repr(first(st.args))
        return _sfail("bare call `$fn(...)` at model level does nothing — " *
                      "did you mean `y = ...` or `y ~ ...`?")
    elseif head === :(=)
        return _sfail("model assignment binds a bare Symbol only, got " *
                      "$(repr(st)) (index/function assignment needs the " *
                      "arbitrary-Julia-function direction — planned)")
    elseif head in (:for, :while)
        return _sfail("control flow is not allowed at model level " *
                      "(StanBlocks rule) — use a vectorized form; " *
                      "deterministic-function escape is planned")
    elseif head in (:if, :elseif, :comprehension, :generator,
        :typed_comprehension, :&&, :||)
        return _sfail("branching/comprehensions are not allowed at model " *
                      "level (StanBlocks rule) — use a vectorized form")
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
                  "(only `~`, `=` and reserved macros lower)")
end

# Plain positional call args (keyword calls never lower: positional only).
function _plain_args(rhs::Expr, what)
    for a in rhs.args[2:end]
        a isa Expr && a.head === :parameters &&
            _sfail("$what takes positional arguments only (no keywords)")
    end
    return rhs.args[2:end]
end

function _lower_response(lhs, rhs, ctx, predictors, pred_idx, coefuse)
    rhs isa Expr && rhs.head === :call && !isempty(rhs.args) || _sfail(
        "response $lhs needs a distribution call, got $(repr(rhs))")
    weights, rhs = _peel_weighted(lhs, rhs, ctx)
    evidence, rhs = _peel_evidence(lhs, rhs, ctx)
    family, lik_link, pred_link, loc, scale =
        _lower_response_base(lhs, rhs, ctx)
    pname = _lower_location(lhs, loc, pred_link, ctx, predictors, pred_idx,
        coefuse)
    return LikelihoodSpec(family, lik_link, lhs, pname, scale, weights,
        evidence, Symbol(lhs, "_resp"))
end

function _peel_weighted(lhs, rhs::Expr, ctx)
    rhs.args[1] === :weighted || return nothing, rhs
    args = _plain_args(rhs, "`weighted`")
    length(args) == 2 || _sfail("`weighted` takes the object form " *
                                "`weighted(Normal(mu, sigma), w)`")
    dist, w = args
    dist isa Expr && dist.head === :call || _sfail(
        "`weighted` wraps a distribution object " *
        "(`weighted(Normal(mu, sigma), w)`), got $(repr(dist))")
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
                                    "`interval_censored(Normal(mu, sigma), hi)`")
        dist, hi = args
        dist isa Expr && dist.head === :call || _sfail(
            "`interval_censored` wraps a distribution object, got " *
            "$(repr(dist))")
        ev = ResponseEvidence(:interval_censored, nothing, _bound(lhs, hi, ctx))
        return ev, dist
    end
    args = _plain_args(rhs, "`$head`")
    length(args) == 3 || _sfail("use the Distributions.jl object form " *
                                "`$head(Normal(...), lo, hi)`, got $(repr(rhs))")
    obj, lo, hi = args
    obj isa Expr && obj.head === :call || _sfail(
        "`$head` wraps a distribution object " *
        "(`$head(Normal(...), lo, hi)`), got $(repr(obj))")
    ev = ResponseEvidence(head, _bound(lhs, lo, ctx), _bound(lhs, hi, ctx))
    return ev, obj
end

function _bound(lhs, b, ctx)
    b isa Real && return b
    b === :Inf && return Inf
    b isa Symbol && b in ctx.data && return b
    return _sfail("response $lhs bound $(repr(b)) must be a literal or a " *
                  "data column")
end

const _RESPONSE_BASE_MSG =
    "response distribution must be `Normal(mu, sigma)`, " *
    "`Bernoulli(logistic(eta))` or `Poisson(exp(eta))`"

function _lower_response_base(lhs, rhs::Expr, ctx)
    rhs.head === :call || _sfail("response $lhs: $_RESPONSE_BASE_MSG; " *
                                 "got $(repr(rhs))")
    fam = rhs.args[1]
    fam === :weighted &&
        _sfail("`weighted(...)` goes outermost: " *
               "`y ~ weighted(Normal(mu, sigma), w)`")
    fam in (:Normal, :Bernoulli, :Poisson) ||
        return _lower_response_base_error(lhs, rhs, fam)
    args = _plain_args(rhs, "`$fam`")
    if fam === :Normal
        length(args) == 2 || _sfail("response $lhs: `Normal` takes " *
                                    "`Normal(mu, sigma)`")
        return GaussianFam, IdentityLink, IdentityLink, args[1],
        _lower_scale(lhs, args[2])
    elseif fam === :Bernoulli
        length(args) == 1 || _sfail("response $lhs: `Bernoulli` takes " *
                                    "`Bernoulli(logistic(eta))`")
        return BernoulliLogitFam, LogitLink, IdentityLink,
        _lower_link_arg(lhs, args[1], :logistic), nothing
    else
        length(args) == 1 || _sfail("response $lhs: `Poisson` takes " *
                                    "`Poisson(exp(eta))`")
        return PoissonLogFam, LogLink, LogLink,
        _lower_link_arg(lhs, args[1], :exp), nothing
    end
end

function _lower_response_base_error(lhs, rhs, fam)
    fam in (:normal, :bernoulli, :poisson) && _sfail(
        "response $lhs: use Distributions.jl constructors " *
        "(`Normal`, not `normal`)")
    fam === :BernoulliLogit && _sfail("response $lhs: write " *
                                      "`Bernoulli(logistic(eta))`")
    fam === :PoissonLog && _sfail("response $lhs: write `Poisson(exp(eta))`")
    return _sfail("response $lhs: unknown distribution `$(repr(fam))` " *
                  "(admitted: Normal, Bernoulli, Poisson). " *
                  "If `$fam` is a submodel, `y ~ sm(...)` calls need " *
                  "composition semantics (planned).")
end

function _lower_link_arg(lhs, arg, wrap)
    arg isa Expr && arg.head === :call && !isempty(arg.args) &&
        arg.args[1] === wrap ||
        _sfail("response $lhs: write `$wrap` around the predictor " *
               "(`Bernoulli(logistic(eta))` / `Poisson(exp(eta))`)")
    args = _plain_args(arg, "`$wrap`")
    length(args) == 1 ||
        _sfail("response $lhs: `$wrap` takes exactly the predictor")
    return args[1]
end

function _lower_scale(lhs, s)
    s isa Real && return s
    s === :Inf && return Inf
    s isa Symbol && return s
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
                              "estimated coefficients (wrap: `eta = a + b*$loc`)")
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

# Predictor analysis: expand deterministic refs (inlining), split the affine
# sum, classify each summand. Returns (terms, uses) with
# uses :: Vector{(coef name, addressee, sign)}.
function _analyze_predictor(pname, rhs, ctx, lhs)
    expanded = _expand_dets(rhs, ctx, Set{Symbol}([pname]), pname)
    out = Tuple{Int,Any}[]
    _collect_signed!(out, expanded, 1, pname)
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

function _expand_dets(ex, ctx, visited::Set{Symbol}, pname)
    ex isa Symbol || return _expand_expr(ex, ctx, visited, pname)
    haskey(ctx.detmap, ex) || return ex
    ex in visited && _sfail("predictor $pname: cyclic definition through $ex")
    push!(ctx.consumed, ex)
    push!(visited, ex)
    out = _expand_dets(ctx.detmap[ex], ctx, visited, pname)
    pop!(visited)
    return out
end
function _expand_expr(ex, ctx, visited, pname)
    ex isa Expr || return ex
    return Expr(ex.head, (_expand_dets(a, ctx, visited, pname)
                          for a in ex.args)...)
end

function _collect_signed!(out, ex, sign::Int, pname)
    if ex isa Expr && ex.head === :call && !isempty(ex.args)
        fn = ex.args[1]
        fn isa Symbol && startswith(string(fn), ".") &&
            _sfail("predictor $pname: dotted/broadcast spelling is " *
                   "undecided — write the predictor undotted " *
                   "(`mu = a + b*x + c[g]`)")
        if fn === :+
            length(ex.args) == 2 && return _collect_signed!(out, ex.args[2], sign, pname)
            for a in ex.args[2:end]
                _collect_signed!(out, a, sign, pname)
            end
            return nothing
        elseif fn === :-
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
                            "`coefficient * column`, `coefficients[column]`, " *
                            "and bare columns)")
    head = core.head
    head === :ref && return _classify_ref(pname, core, sign, ctx)
    if head === :call && !isempty(core.args) && core.args[1] === :*
        return _classify_product(pname, core, sign, ctx)
    end
    head === :macrocall && _sfail("predictor $pname: macros do not lower " *
                                  "inside predictor expressions")
    head === :. && _sfail("predictor $pname: dotted/broadcast spelling is " *
                          "undecided — write the predictor undotted")
    return _sfail("predictor $pname: $(repr(core)) is not affine in data " *
                  "(terms are bare coefficients, `coefficient * column`, " *
                  "`coefficients[column]`, and bare columns)")
end

function _classify_symbol(pname, core::Symbol, sign::Int, ctx)
    core in ctx.data && return TermSpec(OffsetTerm, [core], NamedTuple(),
        core, Symbol(core, "_off")), nothing
    haskey(ctx.detmap, core) && _sfail("predictor $pname: $core is a " *
                                       "computed assignment, not a sampled " *
                                       "coefficient (computed coefficients " *
                                       "are not in slice 1)")
    return TermSpec(InterceptTerm, ColumnRef[], NamedTuple(), :Intercept,
        :intercept), (core, :Intercept, sign)
end

function _classify_product(pname, core::Expr, sign::Int, ctx)
    length(core.args) == 3 || _sfail("predictor $pname: only binary " *
                                     "`coefficient * column` products lower " *
                                     "in slice 1, got $(repr(core))")
    s1, a = _strip_sign(core.args[2])
    s2, b = _strip_sign(core.args[3])
    inner = sign * s1 * s2
    a isa Symbol || _sfail("predictor $pname: $(repr(core)) is not affine " *
                           "in data")
    b isa Symbol || _sfail("predictor $pname: $(repr(core)) is not affine " *
                           "in data")
    kinds = (_summand_kind(a, ctx), _summand_kind(b, ctx))
    if kinds == (:data, :data)
        _sfail("predictor $pname: $a * $b multiplies two data columns — " *
               "precompute the interaction column")
    elseif kinds == (:coef, :coef)
        _sfail("predictor $pname: $a * $b is nonlinear in coefficients")
    elseif kinds == (:number, :number)
        _sfail("predictor $pname: literal $a * $b is not a term — fold " *
               "constants into an intercept coefficient")
    elseif :number in kinds
        _sfail("predictor $pname: literal scaling in $(repr(core)) is not " *
               "a term — scale the column or the prior instead")
    elseif :det in kinds
        _sfail("predictor $pname: computed assignments do not lower as " *
               "coefficients (computed coefficients are not in slice 1)")
    end
    coef, col = kinds[1] === :coef ? (a, b) : (b, a)
    return TermSpec(ContinuousTerm, [col], NamedTuple(), col,
        Symbol(col, "_term")), (coef, col, inner)
end

function _strip_sign(ex)
    sign = 1
    while ex isa Expr && ex.head === :call && length(ex.args) == 2 &&
            ex.args[1] === :-
        sign = -sign
        ex = ex.args[2]
    end
    return sign, ex
end

function _summand_kind(s::Symbol, ctx)
    s in ctx.data && return :data
    haskey(ctx.detmap, s) && return :det
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
    idx isa Symbol || _sfail("predictor $pname: factor index must be a " *
                             "bare data column, got $(repr(idx))")
    base in ctx.data && _sfail("predictor $pname: $base is data — " *
                               "precompute data-indexed columns")
    idx in ctx.data || _sfail("predictor $pname: factor index $idx must " *
                              "be a data column")
    haskey(ctx.detmap, base) && _sfail("predictor $pname: $base is a " *
                                       "computed assignment, not a sampled " *
                                       "coefficient vector")
    return TermSpec(FactorTerm, [idx], (contrasts = :treatment, ref = 1),
        idx, Symbol(idx, "_term")), (base, idx, sign)
end

# Coefficient priors: recovered by name from `coef ~ Normal(lit, lit)`
# statements; missing priors default to Normal(0, 1) (emitter convention).
# Plan order follows predictors, addressees in term order.
function _lower_coefficient_priors(sample, coefuse, predictors)
    stated = Dict{Symbol,Any}()
    for (lhs, rhs) in sample
        haskey(coefuse, lhs) && (stated[lhs] = rhs)
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
    for (lhs, rhs) in sample
        lhs in ctx.data && continue
        haskey(coefuse, lhs) && continue
        push!(params, _lower_parameter(lhs, rhs, coefuse))
    end
    return params
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
    for s in _symbols_in(rhs)
        haskey(coefuse, s) && _sfail("$s is a predictor coefficient and " *
                                     "cannot also be referenced by assignment " *
                                     "$nm")
    end
    _reject_assignment_calls(nm, rhs)
    rhs isa Expr || rhs isa Symbol || rhs isa Real ||
        _sfail("assignment $nm must be an expression, name or literal, " *
               "got $(repr(rhs))")
    return AssignmentSpec(nm, rhs, nm)
end

# Unknown call heads fail here (with the planned-direction pointer) so the
# allowlist error never has to guess; argument shapes stay with validation.
function _reject_assignment_calls(nm, rhs)
    rhs isa Expr || return nothing
    if rhs.head === :call && !isempty(rhs.args)
        fn = rhs.args[1]
        fn isa Symbol && startswith(string(fn), ".") &&
            _sfail("assignment $nm: dotted/broadcast spelling is " *
                   "undecided — slice-1 assignments are scalar")
        fn isa Symbol && fn ∉ ASSIGNMENT_FNS && _sfail(
            "assignment $nm calls `$fn`, which is not in the slice-1 " *
            "assignment vocabulary — arbitrary Julia functions are planned " *
            "(no-@deffun-ceremony direction) but need IR/contract growth")
    end
    for a in rhs.args
        _reject_assignment_calls(nm, a)
    end
    return nothing
end
