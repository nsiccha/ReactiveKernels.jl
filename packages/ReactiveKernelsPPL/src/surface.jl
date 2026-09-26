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
`fixed` carries `Base.merge` NamedTuple fixes (name => data column); explicit
call kwargs win over it.
"""
struct RKPPLModel
    ast::Expr
    mod::Module
    fixed::Dict{Symbol,ColumnData}
end
RKPPLModel(ast, mod) = RKPPLModel(ast, mod, Dict{Symbol,ColumnData}())

"""
    RKPPLSubmodel(name, argnames, body, mod)

A captured reusable submodel definition (`@rkppl sm(a, b) = begin … end`),
mirroring StanBlocks `@slic f(args…)=body`. `argnames` are the positional
inputs bound by name at the use site; `body` is the block AST — density /
deterministic `=` statements followed by a trailing RETURN expression; `mod`
is the defining module (for symbol resolution). A trailing `return x` and a
bare trailing `x` are equivalent spellings. The RETURN selects the submodel
kind: a bare trailing Symbol that is the LHS of an internal
`slot .~ family.(...)` response is a RESPONSE POINTER, not a value — it maps
to the data column at an observation-stream use site (`y ~ sm(...)`), so a
stream body carries no trailing binding; any other return is a latent VALUE
bound to a non-data LHS (`latent = <return>`). Invoked as
`latent ~ sm(a, b)` and expanded inline by [`lower_rkppl`](@ref) (see
`_expand_submodels`): the submodel's own `~`/`=` names are namespaced under
the LHS (`latent_…`) and spliced into the parent plan, so a submodel lowers
exactly like a hand-inlined model — transparent and reusable, never an opaque
node.

A fused stream def (design + coefficients inside, Stan
`bernoulli_logit_glm`-style) keeps the response shell but NOT the predictor
name: its affine local namespaces under the data LHS (`y_mu`), and an inline
compound location synthesizes (`y_eta`) — both move the lowered predictor and
its `<predictor>_coef` block. To factor design + coefficients into a def
WITHOUT moving names, return the affine from a latent def and bind it at a
named use site (`mu ~ affine_def(X, b)`, then `y .~ family.(...mu...)`): the
use-site LHS names the predictor, and the plan is identical to the
hand-written decomposed program.

Alternatively, a stream use site pins the lowered predictor name directly
(`y ~ fused_def(X, b; predictor = mu)`): the response's predictor is `mu`
whether the def locates it by a local or inline, and the plan matches the
decomposed program. A pin claims a fresh predictor name (once — a second
claim fails); it renames one response's predictor, so the pinned location
cannot lower under another name. Pins apply to single-predictor responses
(a multi-eta categorical or a predictorless simplex response fails closed)
at top-level stream calls (a per-cell pin fails closed).
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
(`@rkppl sm(args...) = begin ... end`; see [`RKPPLSubmodel`](@ref)).

Design-matrix vocabulary (standard-Julia value semantics throughout):
bind the matrix once (`X = hcat(1, x1, x2)` — the intercept `1` plus
bare data/derived columns), use it only as a predictor matmul
(`mu = X * b`), and size the coefficient vector with an axes prior
(`b[axes(X, 2)] .~ Normal.(loc, scale)` — scalar args share over
elements, literal `[…]` vectors go per element). Unstated vectors
default to `Normal(0, 1)` per element; under `r2d2(mu, R2, phi)` a
stated vector becomes per-element share-0 overrides and an unstated
one joins the simplex. Every other matrix position (scales, prior
arguments, indexing, arithmetic outside a matmul) fails loudly
naming the spelling. Emitter guidance: matrix and affine spellings
of one model evaluate bit-identically with identical coefficient
labels — emit the matrix form wherever the consumer wants
Stan-shaped fused structure (design matrix + coefficients), the
affine form otherwise."""
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
    cols = Dict{Symbol,ColumnData}()
    for (k, v) in m.fixed
        cols[k] = v
    end
    for (k, v) in kwargs
        cols[k] = _check_col(k, v)
    end
    return _bind_model(m, cols)
end

function _bind_immediate(m::RKPPLModel, data)
    cols = if data isa NamedTuple
        Dict{Symbol,ColumnData}(k => _check_col(k, v) for (k, v) in pairs(data))
    elseif data isa AbstractDict
        Dict{Symbol,ColumnData}(
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

_check_col(k, v) = v isa ColumnData ? v :
    _sfail("data column $k must be a vector or matrix, got $(typeof(v))")

function _bind_model(m::RKPPLModel, cols::Dict{Symbol,ColumnData})
    plan = lower_rkppl(m.ast, keys(cols); mod = m.mod)
    return bind_data(plan, cols)
end

# ── Model merge (StanBlocks-style program composition) ─────────────────────
# `Base.merge(m, override)` splices override statements into a copy of the
# base model's captured block AST, keyed on the bare LHS name — the
# RK-submodel analogue of StanBlocks `Base.merge` (stanblocks-use §2).
# Syntactic only: every semantic gate (roles, shapes, vocabulary) stays in
# `lower_rkppl`, so a merged model lowers through the one pipeline and a
# spliced statement automatically keeps the base's structural role (the RK
# analogue of SB declaration inheritance — re-derived, never carried).

"""
    Base.merge(m::RKPPLModel, override::Expr) -> RKPPLModel
    Base.merge(m::RKPPLModel, fix::NamedTuple) -> RKPPLModel
    Base.merge(m::RKPPLModel, parts...) -> RKPPLModel

Compose model programs over shared submodel blocks (StanBlocks `Base.merge`
analogue). Each form returns a NEW model; the base (AST and fixed dict) is
unchanged.

Statement splice: `override` is one `~` / `.~` / `name = ...` statement or a
`quote ... end` block of them. An override whose bare LHS names a base
top-level statement replaces it in place; a fresh LHS appends (completing an
incomplete base). A submodel use-site is a statement, so merge rewrites which
shared block a program calls (`y ~ stream_a(...)` to `y ~ stream_b(...)`)
without forking the shared def. Multi-part calls fold left, so fix-vs-splice
conflicts resolve to the LATER part (write the fix last).

NamedTuple fix: each `name = value` removes the matching base statement and
stores `value` (a vector or matrix) as model data, bound at the call —
explicit call kwargs win over it (SB easily-rebound data).

Fail-closed: non-statement overrides, non-bare LHS, duplicate base LHS,
fixes naming no statement, and non-vector fix values throw
`SurfaceLoweringError`. Ranged / levels / joint LHS (`y[R]`,
`c[levels(g)]`, `[y1, y2]`) are unmatchable (indexed overrides deferred);
`@plate` / `@scan` cells are invisible to the top-level matcher, so a
colliding append fails at lowering through the single-assignment gate.
"""
function Base.merge(m::RKPPLModel, override::Expr)
    out = Any[a for a in m.ast.args]
    idx, dups, blocked = _merge_base_index(out)
    for raw in _merge_override_stmts(override)
        st = _merge_unwrap_override(raw)
        lhs = _merge_override_lhs(st)
        lhs in dups && _sfail("merge override `$lhs` matches more than " *
                              "one base-model statement (the base is " *
                              "broken — lowering would reject it)")
        lhs in blocked && _sfail("merge override `$lhs` names a ranged, " *
                                 "levels, or joint LHS (indexed overrides " *
                                 "are a deferred slice)")
        if haskey(idx, lhs)
            out[idx[lhs]] = raw
        else
            push!(out, raw)
        end
    end
    return RKPPLModel(Expr(:block, out...), m.mod, copy(m.fixed))
end

function Base.merge(m::RKPPLModel, fix::NamedTuple)
    isempty(fix) && _sfail("merge with an empty NamedTuple fixes nothing")
    out = Any[a for a in m.ast.args]
    idx, dups, blocked = _merge_base_index(out)
    new_fixed = copy(m.fixed)
    drop = Set{Int}()
    for (nm, val) in pairs(fix)
        nm in dups && _sfail("merge fix `$nm` matches more than one " *
                             "base-model statement (the base is broken — " *
                             "lowering would reject it)")
        nm in blocked && _sfail("merge fix `$nm` names a ranged, levels, " *
                                "or joint LHS (indexed overrides are a " *
                                "deferred slice)")
        haskey(idx, nm) || _sfail("merge fix `$nm` matches no base-model " *
                                  "statement (a fixed name must name a " *
                                  "`~` / `.~` / `=` statement to remove)")
        val isa ColumnData || _sfail("merge fix `$nm` must be a " *
            "vector or matrix (data-backed; got $(typeof(val)))")
        push!(drop, idx[nm])
        new_fixed[nm] = val
    end
    kept = Any[a for (i, a) in enumerate(out) if i ∉ drop]
    return RKPPLModel(Expr(:block, kept...), m.mod, new_fixed)
end

function Base.merge(m::RKPPLModel, first, rest...)
    if isempty(rest)
        _sfail("merge takes a quoted statement/block or a NamedTuple of " *
               "fixed values, got $(repr(first))")
    end
    out = Base.merge(m, first)
    for part in rest
        out = Base.merge(out, part)
    end
    return out
end

# Index base top-level statements by bare LHS name: `idx` (name => position),
# `dups` (broken-base duplicates), `blocked` (ref stems / joint outcomes —
# indexed overrides deferred). Plate/scan blocks and unparseable statements
# are invisible (lowering owns their gates).
function _merge_base_index(args)
    idx = Dict{Symbol,Int}()
    dups = Set{Symbol}()
    blocked = Set{Symbol}()
    for (i, arg) in pairs(args)
        arg isa Expr || continue
        st = try
            _unwrap_trivia(arg)
        catch
            continue
        end
        if _is_sample(st) || _is_broadcast_sample(st)
            _merge_index_lhs!(idx, dups, blocked, st.args[2], i)
        elseif st.head === :(=) && length(st.args) == 2 &&
                st.args[1] isa Symbol
            _merge_claim!(idx, dups, st.args[1], i)
        end
    end
    return idx, dups, blocked
end

_merge_claim!(idx::Dict{Symbol,Int}, dups::Set{Symbol}, nm::Symbol, i::Int) =
    haskey(idx, nm) ? push!(dups, nm) : (idx[nm] = i)

function _merge_index_lhs!(idx, dups, blocked, lhs, i::Int)
    lhs isa Symbol && return _merge_claim!(idx, dups, lhs, i)
    lhs isa Expr || return nothing
    if lhs.head === :ref && !isempty(lhs.args) && lhs.args[1] isa Symbol
        push!(blocked, lhs.args[1])
    elseif lhs.head === :vect
        for o in lhs.args
            o isa Symbol && push!(blocked, o)
        end
    end
    return nothing
end

function _merge_override_stmts(override::Expr)
    override.head === :block || return Any[override]
    return Any[a for a in override.args if !(a isa LineNumberNode)]
end

function _merge_unwrap_override(raw)
    raw isa Expr || _sfail("merge override must be a `~`, `.~` or " *
                           "`name = ...` statement, got $(repr(raw))")
    try
        return _unwrap_trivia(raw)
    catch
        _sfail("merge override $(repr(raw)) is not a splicing statement")
    end
end

function _merge_override_lhs(st::Expr)
    ok = ((_is_sample(st) || _is_broadcast_sample(st)) &&
        st.args[2] isa Symbol) ||
        (st.head === :(=) && length(st.args) == 2 && st.args[1] isa Symbol)
    ok || _sfail("merge override must be a `~`, `.~` or `name = ...` " *
                 "statement with a bare Symbol LHS, got $(repr(st))")
    return _stmt_lhs(st)::Symbol
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
    ast, pins = _expand_submodels(ast, data, mod)
    sample, det, plate_ctx, plate_specs, scans, bases, vectors,
    hbases, kplates, kstmts, schedules, event_lps, r2d2decls, joints,
    varying_draws, varying_pending, glms = _partition_statements(ast, data)
    # Varying bindings (draws + contributions): contributions compose
    # only as direct predictor summands, never inside definitions.
    varying_names = Set{Symbol}()
    varying_contribs = Set{Symbol}()
    for p in varying_pending
        push!(varying_names, p.contrib)
        push!(varying_names, p.draws_lhs)
        push!(varying_contribs, p.contrib)
    end
    varying_draws_names = Set{Symbol}(p.draws_lhs for p in varying_pending)
    plate_names = Set{Symbol}(nm for (nm, _, _, _) in plate_specs)
    detmap = Dict{Symbol,Any}(nm => rhs for (nm, rhs) in det)
    prior_names = Set{Symbol}(s.lhs for s in sample if s.lhs ∉ data)
    # Sampled names usable as predictor coefficients: Normal-priored
    # scalars plus per-coefficient `~ Horseshoe()` scalars (the
    # horseshoe triple synthesis in `_lower_horseshoe_priors`).
    coef_priors = Set{Symbol}(s.lhs for s in sample
        if s.lhs ∉ data &&
            (_is_normal_call(s.rhs) || _is_horseshoe_call(s.rhs)))
    # Simplex parameters (`s ~ Dirichlet(...)`): the only names a
    # monotonic term accepts as its increments (checked during response
    # lowering, before `_lower_parameters` runs).
    dirichlet_names = Set{Symbol}(s.lhs for s in sample
        if s.lhs ∉ data && _is_dirichlet_call(s.rhs))
    # Covariance-factor declarations (`L ~ LKJCovarianceFactor(...)`): the
    # only stems a joint response accepts as its factor (checked during
    # joint lowering, before `_lower_parameters` runs).
    factor_names = Set{Symbol}(s.lhs for s in sample
        if s.lhs ∉ data && _is_lkj_factor_call(s.rhs))
    # Dar trajectory parameters: persistence (`truncated(Normal(mu, s),
    # 0, 1)`) and scale (`HalfNormal(s)` / `truncated(Normal(0, s), 0,
    # Inf)`) — the only names a `dar()` call accepts (checked during
    # response lowering, before `_lower_parameters` runs; the contract
    # re-checks for hand-built plans).
    dar_beta_names = Set{Symbol}(s.lhs for s in sample
        if s.lhs ∉ data && _is_dar_beta_rhs(s.rhs))
    dar_sigma_names = Set{Symbol}(s.lhs for s in sample
        if s.lhs ∉ data && _is_dar_sigma_rhs(s.rhs))
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
    # Design matrices leave `det` for the plan-level table (validated
    # here; `detshape` keeps their `:matrix` shape for downstream
    # honesty, but nothing inlines or assigns them).
    det, matrices = _extract_matrices(det, detmap, detshape, data,
        prior_names, plate_names, scans)
    for m in matrices
        delete!(canonmap, m.name)
    end
    # Structural definitions inline into predictors: anything transitively
    # referencing a coefficient candidate (coef-priored or free name).
    # All other vector definitions stay symbolic as named locals.
    structural = _structural_defs(det, data, canonmap, coef_priors,
        prior_names, plate_names)
    vecdefs = Set{Symbol}(nm for (nm, _) in det if detshape[nm] === :vector)
    taken = union(data, Set{Symbol}(nm for (nm, _) in det), prior_names,
        plate_names)
    ctx = (; data, detmap = canonmap, prior_names, coef_priors, detshape,
        vecdefs, structural, plate_names, absorbed = Set{Symbol}(),
        predictor_pins = pins, pins_used = Set{Symbol}(),
        pin_owner = Dict{Symbol,Symbol}(),
        pin_source = Dict{Symbol,Tuple{Symbol,Symbol}}(),
        synth = Ref(0), synth_derived = VectorAssignmentSpec[], taken,
        scan_states = Set{Symbol}(s.state for s in scans),
        scan_coefs = Set{Symbol}(),
        varying_draws = Dict{Symbol,VaryingDraws}(
            d.label => d for d in varying_draws),
        varying_pending = varying_pending,
        varying_contribs = varying_contribs,
        varying_draws_names = varying_draws_names,
        varying_use = Dict{Symbol,Symbol}(),
        implicit_vectors = VectorParameter[],
        splines = Dict{Symbol,SplineBasis}(b.id => b for b in bases),
        spline_uses = Dict{Symbol,Symbol}(),
        hsgps = Dict{Symbol,HSGPBasis}(b.id => b for b in hbases),
        hsgp_uses = Dict{Symbol,Symbol}(),
        dirichlet_names = dirichlet_names,
        mo_uses = Dict{Symbol,Symbol}(),
        matrices = Dict{Symbol,DesignMatrix}(m.name => m for m in matrices),
        matrices_used = Set{Symbol}(),
        coefvecs = Dict{Symbol,Symbol}(s.lhs => s.matrix for s in sample
            if s.matrix !== nothing),
        dar_beta_names = dar_beta_names,
        dar_sigma_names = dar_sigma_names,
        dar_states = Set{Symbol}(),
        dar_coefs = Set{Symbol}(),
        dar_specs = DarSpec[])
    responses = LikelihoodSpec[]
    predictors = PredictorSpec[]
    pred_idx = Dict{Symbol,Int}()
    coefuse = Dict{Symbol,Vector{Tuple{Symbol,Symbol,Int}}}()
    glmuse = Dict{Symbol,Tuple{Symbol,Symbol}}()
    for s in sample
        if s.broadcast
            # Broadcast coefficient priors lower with their factor term.
            s.levels !== nothing && continue
            # Matrix coefficient priors lower with their matrix term.
            s.matrix !== nothing && continue
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
    # Joint correlated-outcomes responses lower after the broadcast
    # responses (same predictor interning/coefficient recording, before
    # coefficient priors resolve).
    for j in joints
        push!(responses,
            _lower_joint_response(j, factor_names, ctx, predictors, pred_idx,
                coefuse))
    end
    # Grouped kernel statements lower here (LP-arg predictor interning
    # needs the response-loop table; their coefuses join before
    # coefficient priors resolve). LP definitions absorb like response
    # locations (see `used_locs` below).
    kernel_lp_predictors = Set{Symbol}()
    for ks in kstmts
        kp = _lower_plate_stmt(ks.st, ks.line, data, ctx, predictors,
            pred_idx, coefuse, schedules, event_lps)
        for (p, _) in kp.lp_args
            push!(kernel_lp_predictors, p)
        end
        push!(kplates, kp)
    end
    # GLM-object responses lower after the joints (no predictor
    # interning — the object owns eta; coefficient recording goes to
    # the separate GLM-use table, before coefficient priors resolve).
    for g in glms
        push!(responses,
            _lower_glm_response(g, sample, prior_names, ctx, coefuse, glmuse))
    end
    # Every use-site pin must name a lowered predictor: a pin the response
    # never claimed is a silent no-op, never a skip.
    for (rlhs, pin) in ctx.predictor_pins
        rlhs in ctx.pins_used || _sfail(
            "response $rlhs pins predictor $pin, but the response lowered " *
            "no predictor (nothing to pin — simplex responses and " *
            "scan-state locations build none)")
    end
    for c in ctx.scan_coefs
        haskey(coefuse, c) && _sfail("$c is both a predictor coefficient " *
            "and a scan coefficient — scan coefficients are sampled " *
            "scalars, not population coefficients (rename one)")
    end
    # Varying slices finalize once every predictor is interned: each
    # contribution resolves to the single predictor that uses it (target
    # inferred from the single use — never declared twice), in-graph
    # `r_<target>_<suffix>` labels claim, and each draws block's slice
    # ranges prove exact-once partition of 1:K.
    varying_slices = _finalize_varying_slices(ctx)
    r2d2set = Set{Symbol}(d.predictor for d in r2d2decls)
    hsset = _horseshoe_predictors(sample, coefuse, predictors, r2d2set)
    for c in ctx.dar_coefs
        haskey(coefuse, c) && _sfail("$c is both a predictor coefficient " *
            "and a dar trajectory parameter — dar parameters are sampled " *
            "scalars, not population coefficients (rename one)")
    end
    for c in ctx.dar_states
        haskey(coefuse, c) && _sfail("$c is a dar trajectory state — it " *
            "splices via its `dar()` call, not as a coefficient (rename one)")
    end
    priors, levelmaps = _lower_coefficient_priors(sample, coefuse, predictors,
        ctx.matrices, r2d2set, hsset)
    for (beta, (label, X)) in glmuse
        append!(priors, _lower_glm_beta_priors(label, beta, X, sample,
            ctx.matrices))
    end
    r2d2s, taus = _lower_r2d2_priors(r2d2decls, sample, coefuse, predictors,
        levelmaps, taken, ctx.matrices)
    hses, hsparams = _lower_horseshoe_priors(sample, coefuse, predictors,
        hsset, taken)
    params, paramsyms, dirichlets = _lower_parameters(sample, coefuse, ctx, glmuse)
    append!(params, taus)
    append!(params, hsparams)
    plate_parameters = PlateParameter[
        _lower_plate_parameter(nm, rhs, rng, coefuse, ctx.matrices)
        for (nm, rhs, rng, _) in plate_specs]
    used_locs = Set{Symbol}()
    for r in responses
        push!(used_locs, r.predictor)
        union!(used_locs, r.extra_predictors)
        # A scale predictor's definition is absorbed like a location's —
        # never also a derived column.
        r.scale isa ScalePredictorRef &&
            push!(used_locs, r.scale.predictor)
        # Mixture component predictors absorb like locations (non-predictor
        # slot names are not definitions — `_absorbed_skip` ignores them).
        if r.family === MixtureFam
            for l in r.mixture_locs
                l isa Symbol && push!(used_locs, l)
            end
            for s in r.mixture_scales
                s isa ScalePredictorRef && push!(used_locs, s.predictor)
            end
        end
    end
    # Kernel LP definitions absorb exactly like response locations.
    union!(used_locs, kernel_lp_predictors)
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
    _validate_varying_margins(varying_draws, data, derived, detshape,
        used_locs)
    # Varying bindings compose only as direct predictor summands:
    # non-predictor definitions referencing one fail here (predictor
    # definitions are the one admitted position; inlined aliases fail
    # at `_inline_structure` with the use site).
    for (nm, rhs) in det
        nm in used_locs && continue
        if _uses_varying_contrib(rhs, varying_names)
            _sfail("definition `$nm = $(repr(rhs))` references a varying " *
                  "binding, which lowers only as a direct predictor " *
                  "summand (`mu = a .+ r`), not inside definitions — " *
                  "slice draws explicitly " *
                  "(`r ~ varying_slice(d, ...)`) and use the slice")
        end
    end
    _check_plate_bares(plate_ctx, data, Set{Symbol}(p.name for p in predictors),
        Set{Symbol}(d.name for d in derived), _factor_coefs(coefuse, predictors),
        plate_names)
    for m in matrices
        m.name in ctx.matrices_used || _sfail(
            "design matrix `$(m.name)` is never used in a predictor " *
            "matmul — drop it or add the use (`mu = $(m.name) * b`)")
    end
    plan = StructuralPlan(responses, predictors, priors, params, assigns,
        Dict{Symbol,AbstractVector}(), 0; derived = derived,
        levelmaps = levelmaps, plate_parameters = plate_parameters, scans = scans,
        dar_paths = ctx.dar_specs,
        varying_draws = varying_draws,
        varying_slices = varying_slices,
        vector_parameters = vcat(ctx.implicit_vectors, dirichlets),
        spline_bases = bases, spline_vectors = vectors, hsgp_bases = hbases,
        kernel_plates = kplates, r2d2_priors = r2d2s, horseshoe_priors = hses,
        matrices = matrices, event_lps = event_lps)
    validate_structure(plan)
    return plan
end

# A per-cell latent declaration reuses the scalar-parameter distribution
# parsing (family, args, HalfNormal/truncated → :positive) and rides the
# plate's range.
function _lower_plate_parameter(name::Symbol, rhs, range, coefuse, matrices)
    sp = _lower_parameter(name, rhs, coefuse, matrices)
    return PlateParameter(sp.name, sp.family, sp.args, sp.support_override,
        range, sp.label)
end

# Shape inference + canonicalization (data-free, Julia-truthful). Shapes:
# data columns are vectors, sampled/det names resolve by position and memo,
# dotted forms are vectors, reductions are scalars, `hcat` is a matrix,
# undotted scalar-array combinations follow Julia exactly (`2*v`, `v*2`,
# `v/2`, `-v` are vectors; `a+x`, `x*z`, `s/x`, `x^2`, `x>1`, `log(x)`
# are `:invalid` — Julia `MethodError`s, reported with the dotted fix).
# Canonicalization rewrites the Julia-valid undotted scalar-array ops
# (`*`, `/`) to dotted-canonical form (Base implements them by broadcast —
# behavior-preserving); matrices never dotted-rewrite (`X * b` keeps its
# shape for predictor classification); every other head passes through.
# Unknown call heads do not shape-route here (the vocabulary screen
# rejects them first); their shape follows their arguments so the
# downstream error names the function.
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
    nmat = count(==(:matrix), argshapes)
    if fn === :hcat
        # Design-matrix construction is always matrix-shaped (arg
        # validation — intercept `1` plus vector columns — lives in
        # matrix-def extraction, not here).
        return :matrix
    elseif nmat > 0 && fn !== :*
        # Slice D1 admits matrices only in predictor matmuls (`mu =
        # X * b`); every other matrix operation fails closed at the
        # mismatch message below.
        return :invalid
    elseif fn === :+ || fn === :-
        length(argshapes) == 1 && return only(argshapes)
        return nvec == 0 ? :scalar : :invalid
    elseif fn === :*
        if nmat > 0
            # Matmul shapes: matrix×matrix → matrix, matrix×vector →
            # vector. Coefficient vectors are scalar-shaped (name role,
            # not shape — the affine free-name precedent), so
            # matrix×scalar ALSO shapes `:vector` here: predictor
            # classification owns `X * b` and rejects true scalar
            # multis (`X * 2`) while naming the spelling. vector×matrix is
            # invalid (no row vectors in slice D1). Coefficient-length
            # checks live in classification, which sees declarations.
            length(argshapes) == 2 || return :invalid
            a, b = argshapes
            a === :matrix && b === :matrix && return :matrix
            a === :matrix && b === :vector && return :vector
            a === :matrix && b === :scalar && return :vector
            a === :scalar && b === :matrix && return :matrix
            return :invalid
        end
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
    elseif fn in VECTOR_FNS
        return :vector
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
        _sfail(_julia_mismatch_msg(fn, where, ex, argshapes))
    if (fn === :* || fn === :/) && length(args) == 2
        # Matrices never dotted-rewrite: `X * b` keeps its shape for
        # predictor classification (which checks the coefficient
        # declaration); anything else with a matrix operand already
        # failed above.
        if fn === :* && :matrix ∉ argshapes &&
                (argshapes[1] === :vector) != (argshapes[2] === :vector)
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

function _julia_mismatch_msg(fn::Symbol, where, ex, argshapes)
    if :matrix in argshapes
        fix = "matrices lower only in predictor matmuls (`mu = X * b`) " *
            "in slice D1"
        if fn === :- && length(ex.args) == 2
            fix *= "; negate dotted (`.-(X * b)`) to distribute the " *
                "sign over the coefficients"
        end
        return "$where combines a matrix outside a matmul: " *
            "`$(repr(ex))` — $fix"
    end
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

_is_horseshoe_call(rhs) =
    rhs isa Expr && rhs.head === :call && !isempty(rhs.args) &&
    rhs.args[1] === :Horseshoe

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
# candidate (a coef-priored sampled name or a free name — data, det,
# per-cell latents, and other sampled names excluded). Structural
# definitions inline into predictors; every other definition keeps its
# binding as a kernel local.
function _structural_defs(det, data, canonmap, coef_priors, prior_names,
        plate_names)
    detkeys = Set{Symbol}(nm for (nm, _) in det)
    structural = Set{Symbol}()
    for (nm, _) in det
        refs = _value_symbols(canonmap[nm])
        if any(s -> s in coef_priors ||
                _is_free_name(s, data, detkeys, prior_names, plate_names), refs)
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

function _is_free_name(s::Symbol, data, detkeys, prior_names, plate_names)
    s in data && return false
    s in detkeys && return false
    s in prior_names && return false
    s in plate_names && return false
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
        :InverseGamma, :Bernoulli, :Poisson, :HalfNormal, :HalfCauchy, :Flat,
        :Dirichlet, :CategoricalLogit, :OrderedLogistic, :Ordinal,
        :Multinomial, :Categorical)

function _reject_unknown_calls(where, rhs)
    rhs isa Expr || return nothing
    rhs.head === :ref && return nothing
    if rhs.head === :call && !isempty(rhs.args)
        fn = rhs.args[1]
        # `hcat` passes the vocabulary screen everywhere (args still
        # recurse below): matrix definitions validate at extraction;
        # strays fail there (definitions) or at predictor analysis
        # (responses) with bind-to-a-name guidance.
        if fn isa Symbol && fn ∉ ELEMENTWISE_OPS && fn ∉ ASSIGNMENT_FNS &&
                fn ∉ VECTOR_FNS && fn !== :spline &&
                fn !== :hsgp && fn !== :mo && fn !== :mo1 &&
                fn !== :hcat && fn !== :dar
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
            fn in (:varying_effect, :varying_draws, :varying_slice) &&
                _sfail("$where calls `$fn` as a value — varying " *
                "statements lower only under `~` (`r ~ $fn(...)`)")
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
            for s in r.mixture_scales
                s isa Symbol && push!(refs, s)
            end
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
# Declared grouping levels (`levels=["c", "a", "b", "d"]`): a literal
# vector in DECLARED numbering order (SB `CA.levels` for categorical
# groupings) — extra entries are unobserved prior-only levels. Elements
# are literal-embeddable scalars: numbers (Bool rides Real), strings,
# chars, or QUOTED symbols (`:a` — a bare name is not a level value).
# Shape (non-empty, duplicate-free) is checked here with line context;
# the contract re-checks for hand-built plans.
function _lower_grouping_levels(v, where)
    v isa Expr && v.head === :vect ||
        _sfail("$where levels takes a literal level vector " *
              "(`levels=[\"b\", \"a\"]`), got $(repr(v))")
    levels = Any[]
    for e in v.args
        if e isa QuoteNode && e.value isa Symbol
            push!(levels, e.value)
        elseif e isa Union{Real,String,Char}
            push!(levels, e)
        elseif e isa Symbol
            _sfail("$where level $(repr(e)) is a bare name — levels are " *
                  "literal values: quote symbols (`:$e`) or write strings " *
                  "(`\"$e\"`)")
        else
            _sfail("$where level $(repr(e)) is not literal-embeddable " *
                  "(numbers, strings, chars, or quoted symbols only)")
        end
    end
    isempty(levels) &&
        _sfail("$where levels declares zero grouping levels " *
              "(`levels=[]` carries no groups)")
    length(unique(levels)) == length(levels) ||
        _sfail("$where levels declares duplicate grouping levels " *
              "($(repr(levels)))")
    return levels
end

# Varying-effect margin elements: `1` (intercept), a bare data column or
# vector-shaped derived local (continuous Z), or an explicit `dummy(c, k)`
# indicator (margins live on the shared draws; targets live on slices).
function _lower_varying_margin_elem(e, data::Set{Symbol},
        detnames::Set{Symbol}, where)
    e isa Integer && !(e isa Bool) ||
        return _lower_varying_margin_symbol(e, data, detnames, where)
    e == 1 ||
        _sfail("$where margin integer must be exactly `1` (intercept); " *
              "for slopes write the bare column or derived local")
    return VaryingMargin(:Intercept, VaryingZRecipe(:ones, :none, nothing))
end

function _lower_varying_margin_symbol(e, data::Set{Symbol},
        detnames::Set{Symbol}, where)
    e isa Symbol || return _lower_varying_margin_dummy(e, data, where)
    e in data || e in detnames ||
        _sfail("$where margin `$e` is neither bound data nor a model " *
              "definition (margins are `1`, bare data columns, " *
              "vector-shaped derived locals, or `dummy(c, k)`)")
    return VaryingMargin(e, VaryingZRecipe(:column, e, nothing))
end

function _lower_varying_margin_dummy(e, data::Set{Symbol}, where)
    e isa Expr && e.head === :call && length(e.args) == 3 &&
        e.args[1] === :dummy ||
        _sfail("$where margin $(repr(e)) is not admitted (margins are " *
              "`1`, bare data columns, vector-shaped derived locals, or " *
              "`dummy(c, k)` — bind an inline expression via an assignment " *
              "first, e.g. `w = x .* z` then `[1, w]`)")
    c, k = e.args[2], e.args[3]
    c isa Symbol ||
        _sfail("$where `dummy` column must be a bare data column, got " *
              "$(repr(c))")
    c in data ||
        _sfail("$where `dummy` column `$c` is not data (`dummy` needs a " *
              "raw column — level membership needs bound values)")
    (k isa Integer && !(k isa Bool)) || k isa AbstractString ||
        _sfail("$where `dummy` level must be an Int value or string, got " *
              "$(repr(k))")
    return VaryingMargin(Symbol(string(c) * "_dummy_" * string(k)),
        VaryingZRecipe(:dummy, c, k))
end

_is_varying_call(rhs) =
    rhs isa Expr && rhs.head === :call && !isempty(rhs.args) &&
    rhs.args[1] isa Symbol &&
    rhs.args[1] in (:varying_effect, :varying_draws, :varying_slice)

_varying_head(rhs::Expr) = rhs.args[1]::Symbol

# A draws block shared by fused and split forms:
# `r ~ varying_effect(g, [margins...]; eta, levels)` binds a contribution
# over an anonymous draws block; `d ~ varying_draws(g, [margins...];
# eta, levels)` binds the draws for explicit `varying_slice` consumers.
# Lowers directly to VaryingDraws IR. Continuous margins reference data
# columns or vector-shaped derived locals; the partition-time gate
# admits data-or-defined names (forward references work) and
# `_validate_varying_margins` proves vector shape after lowering.
# Claims the draws label up front so user definitions can never
# collide with in-graph names (K=1 scale/xi, correlated L/tau/z).
function _lower_varying_draws_block(lhs::Symbol, call::Expr, line::Int,
        data::Set{Symbol}, detnames::Set{Symbol}, seen::Set{Symbol},
        seelines::Dict{Symbol,Int}, used_suffixes::Set{String})
    head = _varying_head(call)
    where = line > 0 ? "$head `$lhs` (line $line)" : "$head `$lhs`"
    pos = Any[]
    eta = 1.0
    eta_given = false
    levels = nothing
    for a in call.args[2:end]
        if a isa Expr && a.head === :parameters
            for kw in a.args
                kw isa Expr && kw.head === :kw && length(kw.args) == 2 ||
                    _sfail("$where takes keywords `eta`/`levels` only")
                kw.args[1] === :eta || kw.args[1] === :levels ||
                    _sfail("$where takes keywords `eta`/`levels` only, got " *
                          "`$(kw.args[1])`")
                if kw.args[1] === :eta
                    v = kw.args[2]
                    v isa Real && !(v isa Bool) ||
                        _sfail("$where eta must be a numeric literal, got $(repr(v))")
                    eta = Float64(v)
                    eta_given = true
                else
                    levels = _lower_grouping_levels(kw.args[2], where)
                end
            end
        else
            push!(pos, a)
        end
    end
    length(pos) == 3 && pos[1] isa QuoteNode && pos[1].value isa Symbol &&
        _sfail("$where takes `(group, [margins...])` — no id position " *
              "(independent blocks on one grouping disambiguate " *
              "by binding name, not labels)")
    length(pos) == 2 ||
        _sfail("$where takes `(group, [margins...])` positionally, got " *
              "$(length(pos)) positional argument(s)")
    group, vec = pos
    group isa Symbol ||
        _sfail("$where grouping must be a bare data column, got $(repr(group))")
    group in data ||
        _sfail("$where grouping `$group` is not data")
    vec isa Expr && vec.head === :vect ||
        _sfail("$where margins must be a vector (`[1, x]`), even for one " *
              "margin")
    isempty(vec.args) &&
        _sfail("$where margin list is empty")
    margins = VaryingMargin[
        _lower_varying_margin_elem(e, data, detnames, where)
        for e in vec.args]
    K = length(margins)
    kind = if K == 1 && !eta_given && _is_ones_vmargin(first(margins))
        :intercept1
    elseif K == 1 && !eta_given
        :slope1
    else
        :correlated
    end
    if kind !== :correlated
        eta = NaN
    elseif !(eta > 0)
        _sfail("$where eta must be positive, got $eta")
    end
    suffix = string(group)
    if suffix in used_suffixes
        suffix = string(group) * "_" * string(lhs)
        suffix in used_suffixes &&
            _sfail("$where in-graph suffix `$suffix` collides (two draws " *
                  "blocks share grouping and binding stem — rename a binding)")
    end
    push!(used_suffixes, suffix)
    label = Symbol("draws_" * suffix)
    _claim!(seen, seelines, label, line)
    d = VaryingDraws(group, kind, margins, eta, label, suffix, levels)
    if kind === :intercept1 || kind === :slope1
        for nm in _varying_k1_names(d)
            _claim!(seen, seelines, nm, line)
        end
    else
        for nm in _varying_corr_names(d)
            _claim!(seen, seelines, nm, line)
        end
        # The derived draws `b_<suffix>` live in `constrain` output only
        # (never sampled, never in-graph) — claimed so a user definition
        # can never shadow them there.
        _claim!(seen, seelines, Symbol("b_" * suffix), line)
    end
    return d
end

_is_ones_vmargin(m::VaryingMargin) =
    m.z.kind === :ones && m.coefficient === :Intercept

# One target application of shared draws:
# `r ~ varying_slice(d, cols)` with `cols` an Int column or a `lo:hi`
# UnitRange over the draws block's margins. Columns are explicit and
# validated against the draws width here; the partition (exact-once
# coverage of 1:K) is checked once all slices are in.
function _lower_varying_slice(lhs::Symbol, call::Expr, line::Int,
        draws_by_lhs::Dict{Symbol,VaryingDraws})
    where = line > 0 ? "varying_slice `$lhs` (line $line)" :
        "varying_slice `$lhs`"
    args = call.args[2:end]
    (length(args) == 2 && !any(a -> a isa Expr && a.head === :parameters,
        args)) ||
        _sfail("$where takes `(draws, cols)` positionally (no keywords)")
    dref, colsel = args
    dref isa Symbol ||
        _sfail("$where names a draws block by its binding, got " *
              "$(repr(dref))")
    haskey(draws_by_lhs, dref) ||
        _sfail("$where names unknown draws `$dref` (bind one first: " *
              "`$dref ~ varying_draws(group, [margins...])`)")
    d = draws_by_lhs[dref]
    K = length(d.margins)
    cols = _lower_varying_columns(colsel, K, where)
    return (contrib = lhs, draws = d.label, draws_lhs = dref,
        columns = cols, line = line)
end

function _lower_varying_columns(colsel, K::Int, where)
    if colsel isa Integer && !(colsel isa Bool)
        c = Int(colsel)
        (1 <= c <= K) ||
            _sfail("$where selects column $c outside 1:$K")
        return c:c
    end
    colsel isa Expr && colsel.head === :call && length(colsel.args) == 3 &&
        colsel.args[1] === :(:) &&
        colsel.args[2] isa Integer && !(colsel.args[2] isa Bool) &&
        colsel.args[3] isa Integer && !(colsel.args[3] isa Bool) ||
        _sfail("$where selects an Int column or a `lo:hi` range, got " *
              "$(repr(colsel))")
    lo, hi = Int(colsel.args[2]), Int(colsel.args[3])
    (1 <= lo <= hi <= K) ||
        _sfail("$where selects $lo:$hi outside 1:$K")
    return lo:hi
end

# A varying contribution referenced by name anywhere in an expression
# (QuoteNodes are not references — `spline(:s_x)` must not match a
# contribution named `s_x`). Contribs compose only as direct additive
# predictor summands; every other position fails closed naming this.
_uses_varying_contrib(ex::Symbol, names::Set{Symbol}) = ex in names
_uses_varying_contrib(::QuoteNode, ::Set{Symbol}) = false
_uses_varying_contrib(ex::Expr, names::Set{Symbol}) =
    any(a -> _uses_varying_contrib(a, names), ex.args)
_uses_varying_contrib(::Any, ::Set{Symbol}) = false

function _finalize_varying_slices(ctx)
    slices = VaryingSlice[]
    for p in ctx.varying_pending
        target = get(ctx.varying_use, p.contrib, nothing)
        target === nothing &&
            _sfail("varying contribution `$(p.contrib)` (line $(p.line)) " *
                  "is never used in a predictor — every bound " *
                  "contribution feeds exactly one predictor " *
                  "(`mu = a .+ $(p.contrib)`); drop it or use it")
        d = ctx.varying_draws[p.draws]
        rlabel = Symbol("r_", target, "_", d.suffix)
        rlabel in ctx.taken &&
            _sfail("implicit `$rlabel` collides with your definition — " *
                  "rename yours")
        push!(ctx.taken, rlabel)
        push!(slices, VaryingSlice(p.draws, p.columns, target))
    end
    # Exact-once partition of 1:K per draws, in slice order (columns
    # are explicit, so order is free — sort, then prove contiguity).
    for (label, d) in ctx.varying_draws
        K = length(d.margins)
        own = [s for s in slices if s.draws === label]
        targets = [s.target for s in own]
        length(unique(targets)) == length(targets) ||
            _sfail("draws $label feeds a predictor twice (one slice per " *
                  "(draws, target) — fuse the column ranges)")
        lo = 1
        for s in sort!(own; by = s -> first(s.columns))
            first(s.columns) == lo ||
                _sfail("draws $label slice for `$(s.target)` starts at " *
                      "$(first(s.columns)), want $lo (slices partition " *
                      "1:$K exactly once — every margin consumed once)")
            lo = last(s.columns) + 1
        end
        lo - 1 == K ||
            _sfail("draws $label slices cover $(lo - 1) of $K margins " *
                  "(unconsumed margins sample dead parameters — slice " *
                  "them or drop them from the draws)")
    end
    return slices
end

# Post-lowering margin proof: the partition-time gate admits
# data-or-defined names (shapes don't exist yet), so every `:column`
# margin proves here that it is bound data or an EMITTED
# vector-shaped derived local.
function _validate_varying_margins(draws::Vector{VaryingDraws},
        data::Set{Symbol}, derived::Vector{VectorAssignmentSpec},
        detshape, used_locs::Set{Symbol})
    emitted = Set{Symbol}(d.name for d in derived)
    for d in draws
        for m in d.margins
            m.z.kind === :column || continue
            c = m.z.column
            c in data && continue
            c in emitted && continue
            shape = get(detshape, c, :unknown)
            if shape === :scalar
                _sfail("draws $(d.label) margin `$c` is a scalar model " *
                       "definition — margins need vector-shaped (n_obs) " *
                       "derived locals (bind `w = x .* z`, then list `[w]`)")
            elseif shape === :vector && c in used_locs
                _sfail("draws $(d.label) margin `$c` is the predictor " *
                       "location `$c`, which inlines into the predictor " *
                       "and emits no Z column — bind the interaction as " *
                       "its own derived local (`w = ...`, then list `[w]`)")
            elseif shape === :vector
                _sfail("draws $(d.label) margin `$c` is absorbed into " *
                       "its predictor (predictor structure, not a " *
                       "standalone column) and emits no Z column — Z " *
                       "columns must be data-only derivations (`w = x .* z`)")
            else
                _sfail("draws $(d.label) margin `$c` is neither bound " *
                       "data nor a vector-shaped derived local")
            end
        end
    end
    return nothing
end

_contains_spline(ex) = ex isa Expr &&
    (_is_spline_call(ex) || any(_contains_spline, ex.args))

_is_spline_call(ex) =
    ex isa Expr && ex.head === :call && !isempty(ex.args) &&
    ex.args[1] === :spline

_contains_hsgp(ex) = ex isa Expr &&
    (_is_hsgp_call(ex) || any(_contains_hsgp, ex.args))

_is_hsgp_call(ex) =
    ex isa Expr && ex.head === :call && !isempty(ex.args) &&
    ex.args[1] === :hsgp

_contains_mo(ex) = ex isa Expr &&
    (_is_mo_call(ex) || any(_contains_mo, ex.args))

_is_mo_call(ex) =
    ex isa Expr && ex.head === :call && !isempty(ex.args) &&
    ex.args[1] === :mo

_contains_mo1(ex) = ex isa Expr &&
    (_is_mo1_call(ex) || any(_contains_mo1, ex.args))

_is_mo1_call(ex) =
    ex isa Expr && ex.head === :call && !isempty(ex.args) &&
    ex.args[1] === :mo1

_contains_dar(ex) = ex isa Expr &&
    (_is_dar_call(ex) || any(_contains_dar, ex.args))

_is_dar_call(ex) =
    ex isa Expr && ex.head === :call && !isempty(ex.args) &&
    ex.args[1] === :dar

# A dar-persistence RHS: `truncated(Normal(mu, s), 0, 1)` exactly (SB's
# `beta ~ normal(0.5, 0.2; lower=0, upper=1)`; location/scale ride free,
# bounds are literal). Arity/shape details stay with `_lower_parameter`;
# this is the use-site screen, like `dirichlet_names` for `mo()`.
_is_dar_beta_rhs(rhs) =
    rhs isa Expr && rhs.head === :call && length(rhs.args) == 4 &&
    rhs.args[1] === :truncated && _is_normal_call(rhs.args[2]) &&
    _dar_bound_eq(rhs.args[3], 0.0) && _dar_bound_eq(rhs.args[4], 1.0)

# A dar-scale RHS: `HalfNormal(s)` or `truncated(Normal(0, s), 0, Inf)`
# (SB's `sigma ~ normal(0, 0.2; lower=0)`; the zero-location detail
# stays with `_lower_parameter`).
_is_dar_sigma_rhs(rhs) =
    rhs isa Expr && rhs.head === :call && !isempty(rhs.args) &&
    ((rhs.args[1] === :HalfNormal ||
      (length(rhs.args) == 4 && rhs.args[1] === :truncated &&
       _is_normal_call(rhs.args[2]) &&
       _dar_bound_eq(rhs.args[3], 0.0) && _is_dar_inf(rhs.args[4]))))

_dar_bound_eq(b, v::Float64) = b isa Real && Float64(b) == v

_is_dar_inf(b) = b === :Inf || (b isa Real && isinf(Float64(b)) && Float64(b) > 0)

# A bare `spline_basis(:id, x...; kind=..., k=...)` call declares one
# spline basis (the first bare-call statement: declarations do work at
# lowering — they build IR + claim the generated names — so the "bare
# call does nothing" rejection does not apply). Quoted id, bare raw
# axes (1 → :tps, 2 → :t2 when `kind` is omitted), literal `k`
# (default 10 / (5, 5)). Lowers directly to SplineBasis IR + the fully
# determined SplineVector set (contract `_spline_*` rules); claims the
# basis label, vector names, and materialized basis-column names up
# front so user definitions can never collide with bind/graph names.
_is_basis_stmt(st) =
    st isa Expr && st.head === :call && !isempty(st.args) &&
    st.args[1] === :spline_basis

# A bare `r2d2(mu, R2, phi[, tau])` call declares a flat R2D2 variance
# decomposition over one predictor (same bare-call-declaration shape as
# `spline_basis`): positional predictor + R2/phi parameter names, plus
# an optional tau (sampled-parameter name or positive literal; omitted
# synthesizes a half-standard-Normal `r2d2_<pred>_tau_bsv`).
_is_r2d2_stmt(st) =
    st isa Expr && st.head === :call && !isempty(st.args) &&
    st.args[1] === :r2d2

function _lower_r2d2_decl(st::Expr, line::Int)
    where = line > 0 ? "r2d2 (line $line)" : "r2d2"
    args = [a for a in st.args[2:end]
        if !(a isa Expr && a.head === :parameters)]
    any(a -> a isa Expr && a.head === :parameters, st.args[2:end]) &&
        _sfail("$where takes positional args only " *
               "(`r2d2(mu, R2, phi[, tau])`), no keywords")
    length(args) == 3 || length(args) == 4 ||
        _sfail("$where takes `(predictor, R2, phi[, tau])` — " *
               "$(length(args)) positional args, got $(repr(st))")
    pred, r2, phi = args[1:3]
    pred isa Symbol || _sfail("$where predictor must be a bare " *
                              "predictor name, got $(repr(pred))")
    r2 isa Symbol || _sfail("$where R2 must be a bare scalar-Beta " *
                            "parameter name, got $(repr(r2))")
    phi isa Symbol || _sfail("$where phi must be a bare " *
                             "simplex-parameter name, got $(repr(phi))")
    tau = length(args) == 4 ? args[4] : nothing
    tau === nothing || tau isa Symbol || tau isa Real ||
        _sfail("$where tau must be a sampled-parameter name or a " *
               "positive literal, got $(repr(tau))")
    return (predictor = pred, r2 = r2, phi = phi, tau = tau, line = line)
end

function _lower_basis(st::Expr, line::Int, data::Set{Symbol},
        seen::Set{Symbol}, seelines::Dict{Symbol,Int},
        bases::Vector{SplineBasis})
    where = line > 0 ? "spline basis (line $line)" : "spline basis"
    pos = Any[]
    kind = nothing
    kind_given = false
    k = nothing
    for a in st.args[2:end]
        if a isa Expr && a.head === :parameters
            for kw in a.args
                kw isa Expr && kw.head === :kw && length(kw.args) == 2 ||
                    _sfail("$where takes keywords `kind`/`k` only")
                key = kw.args[1]
                key === :kind || key === :k ||
                    _sfail("$where takes keywords `kind`/`k` only, got " *
                          "`$key`")
                if key === :kind
                    v = kw.args[2]
                    v isa QuoteNode && v.value isa Symbol ||
                        _sfail("$where quotes its kind: got $(repr(v)) — " *
                              "write `kind=:tps` or `kind=:t2`")
                    v.value === :tps || v.value === :t2 ||
                        _sfail("$where kind must be `:tps` or `:t2`, got " *
                              "$(repr(v.value))")
                    kind = v.value
                    kind_given = true
                else
                    k = _lower_basis_k(kw.args[2], where)
                end
            end
        else
            push!(pos, a)
        end
    end
    length(pos) >= 2 ||
        _sfail("$where takes `(:id, axis...)` positionally, got " *
              "($(join(repr.(pos), ", ")))")
    id, axes = pos[1], pos[2:end]
    id isa QuoteNode && id.value isa Symbol ||
        _sfail("$where quotes its basis id: got $(repr(id)) — write " *
              "`spline_basis(:id, x)` (bare names are data columns)")
    id = id.value
    any(b -> b.id === id, bases) &&
        _sfail("$where duplicates basis :$id (one declaration per id)")
    for c in axes
        c isa Symbol ||
            _sfail("$where axes must be bare data columns, got $(repr(c))")
        c in data || _sfail("$where axis `$c` is not data")
    end
    length(axes) == 1 || length(axes) == 2 ||
        _sfail("$where takes one axis (`s(x)`) or two (`t2(x, z)`), got " *
              "$(length(axes))")
    kind === nothing && (kind = length(axes) == 1 ? :tps : :t2)
    if kind_given
        want = kind === :tps ? 1 : 2
        spell = want == 1 ? "one axis (`s(x)`)" : "two axes (`t2(x, z)`)"
        length(axes) == want ||
            _sfail("$where kind=:$kind takes $spell, got $(length(axes))")
    end
    if k === nothing
        k = kind === :tps ? 10 : (5, 5)
    elseif kind === :tps
        k isa Int ||
            _sfail("$where kind=:tps takes an integer `k`, got $(repr(k))")
    else
        k isa Tuple{Int,Int} ||
            _sfail("$where kind=:t2 takes a `(k1, k2)` integer tuple `k`, " *
                  "got $(repr(k))")
    end
    blocks = [SplineBasisBlock(n, w, Symbol[])
              for (n, w) in _spline_blocks(kind, k)]
    label = Symbol("spline_", id)
    _claim!(seen, seelines, label, line)
    vectors = SplineVector[]
    for (vname, vfamily, vargs, vsupport, vwidth) in
            _spline_vector_specs(id, kind, k)
        _claim!(seen, seelines, vname, line)
        push!(vectors, SplineVector(vname, vfamily, vargs, vsupport,
            vwidth, id, vname))
    end
    for (_, cols) in _spline_basis_columns(id, kind, k), c in cols
        _claim!(seen, seelines, c, line)
    end
    return SplineBasis(id, kind, Vector{Symbol}(axes), k, blocks, label),
        vectors
end

function _lower_basis_k(v, where)
    v isa Integer && !(v isa Bool) ||
        (v isa Expr && v.head === :tuple) ||
        _sfail("$where `k` must be a literal (an integer for `s`, a " *
              "`(k1, k2)` integer tuple for `t2`), got $(repr(v))")
    if v isa Expr
        length(v.args) == 2 ||
            _sfail("$where `k` tuple takes exactly two entries, got " *
                  "$(repr(v))")
        all(e -> e isa Integer && !(e isa Bool), v.args) ||
            _sfail("$where `k` tuple entries must be integer literals, " *
                  "got $(repr(v))")
        all(e -> e > 2, v.args) ||
            _sfail("$where `k` entries must exceed 2, got $(repr(v))")
        return (Int(v.args[1]), Int(v.args[2]))
    end
    v > 2 || _sfail("$where `k` must exceed 2, got $v")
    return Int(v)
end

# A bare `hsgp_basis(:id, x...; k=..., c=..., iso=...)` call declares
# one HSGP basis (same bare-call-declaration shape as `spline_basis`).
# Quoted id, bare raw axes (any count ≥ 1), literal `k` (positive
# integer or per-axis tuple, default 20), literal `c` (real > 1 or
# per-axis tuple, default 1.5), literal `iso` Bool (default true).
# Lowers directly to HSGPBasis IR (fits fill at bind); claims the
# basis label and the sampled names up front so user definitions can
# never collide with Stage-B graph names (summand labels ride the
# spline precedent — unclaimed, predictor-derived).
_is_hsgp_basis_stmt(st) =
    st isa Expr && st.head === :call && !isempty(st.args) &&
    st.args[1] === :hsgp_basis

function _lower_hsgp_basis(st::Expr, line::Int, data::Set{Symbol},
        seen::Set{Symbol}, seelines::Dict{Symbol,Int},
        hbases::Vector{HSGPBasis})
    where = line > 0 ? "hsgp basis (line $line)" : "hsgp basis"
    pos = Any[]
    k = nothing
    c = nothing
    iso = true
    for a in st.args[2:end]
        if a isa Expr && a.head === :parameters
            for kw in a.args
                kw isa Expr && kw.head === :kw && length(kw.args) == 2 ||
                    _sfail("$where takes keywords `k`/`c`/`iso` only")
                key = kw.args[1]
                key === :k || key === :c || key === :iso ||
                    _sfail("$where takes keywords `k`/`c`/`iso` only, got " *
                          "`$key`")
                if key === :k
                    k = _lower_hsgp_k(kw.args[2], where)
                elseif key === :c
                    c = _lower_hsgp_c(kw.args[2], where)
                else
                    v = kw.args[2]
                    v isa Bool ||
                        _sfail("$where `iso` must be a Bool literal, got " *
                              "$(repr(v))")
                    iso = v
                end
            end
        else
            push!(pos, a)
        end
    end
    length(pos) >= 2 ||
        _sfail("$where takes `(:id, axis...)` positionally, got " *
              "($(join(repr.(pos), ", ")))")
    id, axes = pos[1], pos[2:end]
    id isa QuoteNode && id.value isa Symbol ||
        _sfail("$where quotes its basis id: got $(repr(id)) — write " *
              "`hsgp_basis(:id, x)` (bare names are data columns)")
    id = id.value
    any(b -> b.id === id, hbases) &&
        _sfail("$where duplicates basis :$id (one declaration per id)")
    for ax in axes
        ax isa Symbol ||
            _sfail("$where axes must be bare data columns, got $(repr(ax))")
        ax in data || _sfail("$where axis `$ax` is not data")
    end
    length(axes) == length(unique(axes)) ||
        _sfail("$where axis columns must be distinct, got $axes")
    d = length(axes)
    k = k === nothing ? fill(20, d) : _hsgp_broadcast_opt(k, d, where, :k)
    c = c === nothing ? fill(1.5, d) : _hsgp_broadcast_opt(c, d, where, :c)
    label = Symbol("hsgp_", id)
    _claim!(seen, seelines, label, line)
    hb = HSGPBasis(id, Vector{Symbol}(axes), k, c, iso,
        Tuple{Float64,Float64}[], label)
    for nm in _hsgp_all_names(hb)
        _claim!(seen, seelines, nm, line)
    end
    return hb
end

function _lower_hsgp_k(v, where)
    v isa Integer && !(v isa Bool) ||
        (v isa Expr && v.head === :tuple) ||
        _sfail("$where `k` must be a literal (a positive integer or a " *
              "per-axis integer tuple), got $(repr(v))")
    if v isa Expr
        all(e -> e isa Integer && !(e isa Bool), v.args) ||
            _sfail("$where `k` tuple entries must be integer literals, " *
                  "got $(repr(v))")
        all(e -> e >= 1, v.args) ||
            _sfail("$where `k` entries must be positive, got $(repr(v))")
        return Int[e for e in v.args]
    end
    v >= 1 || _sfail("$where `k` must be positive, got $v")
    return Int(v)
end

function _lower_hsgp_c(v, where)
    v isa Real && !(v isa Bool) ||
        (v isa Expr && v.head === :tuple) ||
        _sfail("$where `c` must be a literal (a real > 1 or a per-axis " *
              "tuple), got $(repr(v))")
    if v isa Expr
        all(e -> e isa Real && !(e isa Bool), v.args) ||
            _sfail("$where `c` tuple entries must be real literals, " *
                  "got $(repr(v))")
        all(e -> isfinite(Float64(e)) && Float64(e) > 1, v.args) ||
            _sfail("$where `c` entries must be finite and exceed 1, " *
                  "got $(repr(v))")
        return Float64[e for e in v.args]
    end
    isfinite(Float64(v)) && Float64(v) > 1 ||
        _sfail("$where `c` must be finite and exceed 1, got $(repr(v))")
    return Float64(v)
end

# Scalar broadcasts per axis (SB `_brm_axis_option` shape); tuples must
# match the axis count exactly.
function _hsgp_broadcast_opt(v, d::Int, where, key::Symbol)
    v isa Vector && length(v) == d && return v
    v isa Vector &&
        _sfail("$where `$key` tuple takes one entry per axis ($d), got " *
              "$(length(v))")
    T = key === :k ? Int : Float64
    return fill(T(v), d)
end

# Panel-kernel plate statement:
#   `result ~ plate(cols...; subjects=N) do slices... <cell> end`
# The cell is a REAL subgraph (assignments + exactly one dotted `.~`
# observation + a trailing collected name) — NOT desugared to flat
# top-level statements. `subjects` is an integer literal or a dims-key
# name resolved at bind; slice columns must be bound data.
function _is_kernel_plate_stmt(st::Expr)
    (_is_sample(st) || _is_broadcast_sample(st)) || return false
    rhs = st.args[3]
    rhs isa Expr && rhs.head === :do || return false
    isempty(rhs.args) && return false
    call = rhs.args[1]
    return call isa Expr && call.head === :call && !isempty(call.args) &&
        call.args[1] === :plate
end

function _lower_kernel_plate(st::Expr, line::Int, data::Set{Symbol},
        seen::Set{Symbol}, seelines::Dict{Symbol,Int})
    where = line > 0 ? "kernel plate (line $line)" : "kernel plate"
    bc = _is_broadcast_sample(st)
    bc && _sfail("$where carries a scalar `~` (the collected result " *
                 "name), not `.~`")
    result = st.args[2]
    result isa Symbol ||
        _sfail("$where LHS must be a bare Symbol (the collected " *
               "per-subject result name), got $(repr(result))")
    doex = st.args[3]
    length(doex.args) >= 2 && doex.args[2] isa Expr ||
        _sfail("$where needs `plate(cols...; subjects=N) do slices... " *
               "cell end`")
    call, lam = doex.args[1], doex.args[2]
    # `plate(cols...; subjects=N)`: exactly the subjects kwarg, then ≥1
    # bare data columns.
    subj = nothing
    cols = Symbol[]
    for arg in call.args[2:end]
        if arg isa Expr && arg.head === :parameters
            for kw in arg.args
                kw isa Expr && kw.head === :kw && length(kw.args) == 2 &&
                    kw.args[1] === :subjects ||
                    _sfail("$where takes exactly one keyword " *
                           "`subjects=N`, got $(repr(kw))")
                subj !== nothing &&
                    _sfail("$where repeats `subjects=`")
                subj = kw.args[2]
            end
        elseif arg isa Symbol
            push!(cols, arg)
        else
            _sfail("$where plate inputs must be bare data columns, got " *
                   "$(repr(arg))")
        end
    end
    subj === nothing &&
        _sfail("$where needs `subjects=N` (an integer literal or a " *
               "dims-key name bound at bind)")
    subjects = if subj isa Int
        subj > 0 ||
            _sfail("$where subject count must be positive, got $subj")
        subj
    elseif subj isa Symbol
        subj
    else
        _sfail("$where `subjects` must be an integer literal or a " *
               "dims-key name, got $(repr(subj))")
    end
    isempty(cols) &&
        _sfail("$where takes at least one slice column")
    for c in cols
        c in data ||
            _sfail("$where slice column `$c` is not bound data " *
                   "(responses enter the cell as slices)")
    end
    # `do slices... cell end`: plain-Symbol params, one per column.
    lam.head === :-> && length(lam.args) == 2 ||
        _sfail("$where `do` block must be `do slices... cell end`")
    ptuple, body = lam.args[1], lam.args[2]
    params = if ptuple isa Symbol
        Symbol[ptuple]
    elseif ptuple isa Expr && ptuple.head === :tuple &&
            all(p -> p isa Symbol, ptuple.args)
        Symbol[ptuple.args...]
    else
        _sfail("$where cell params must be plain names (one per slice " *
               "column)")
    end
    length(params) == length(cols) ||
        _sfail("$where has $(length(params)) cell params for " *
               "$(length(cols)) slice columns (one param per column)")
    body isa Expr && body.head === :block ||
        _sfail("$where cell must be a `begin ... end`-style block")
    # Cell: assignments + exactly one `.~` + trailing collected name.
    assignments = Pair{Symbol,Any}[]
    obs_stmt = nothing
    collected = nothing
    cell = Any[s for s in body.args if !(s isa LineNumberNode)]
    isempty(cell) && _sfail("$where cell is empty (need assignments, " *
                            "one `.~` observation, and a collected name)")
    for (k, s) in enumerate(cell)
        last_stmt = k == length(cell)
        if s isa Symbol
            last_stmt ||
                _sfail("$where cell names a bare `$s` mid-cell — only " *
                       "the trailing statement may be a bare name (the " *
                       "collected result)")
            collected = s
        elseif s isa Expr && s.head === :(=) && length(s.args) == 2 &&
                s.args[1] isa Symbol
            push!(assignments, s.args[1] => s.args[2])
        elseif s isa Expr && s.head === :call && length(s.args) == 3 &&
                (s.args[1] === :.~ || s.args[1] === :~)
            s.args[1] === :~ &&
                _sfail("$where in-cell observation broadcasts " *
                       "(`yy .~ Normal.(mu, sigma)`); scalar `~` over " *
                       "vectors is rejected per the explicit-dots ruling")
            obs_stmt !== nothing &&
                _sfail("$where cell has more than one `.~` observation " *
                       "(panel v1 admits exactly one)")
            obs_stmt = s
        else
            _sfail("$where cell statements are `name = ...`, one " *
                   "`yy .~ Normal.(mu, sigma)`, and a trailing collected " *
                   "name — got $(repr(s))")
        end
    end
    obs_stmt === nothing &&
        _sfail("$where cell has no `.~` observation (panel v1 needs " *
               "exactly one in-cell likelihood)")
    collected === nothing &&
        _sfail("$where cell must end with a collected result name (a " *
               "bare cell name)")
    obs = _lower_kernel_obs(obs_stmt, params, where)
    local_names = union(Set{Symbol}(params),
        Set{Symbol}(nm for (nm, _) in assignments))
    collected in local_names ||
        _sfail("$where collected result `$collected` is not a cell " *
               "name (slice param or cell-local assignment)")
    # Cell names become flat model-scope locals at codegen: claim them
    # alongside the result (later model statements reusing them fail as
    # redefinitions, and vice versa).
    _claim!(seen, seelines, result, line)
    for nm in params
        _claim!(seen, seelines, nm, line)
    end
    for (nm, _) in assignments
        _claim!(seen, seelines, nm, line)
    end
    slices = Tuple{Symbol,Symbol,Symbol}[(c, p, :unknown)
        for (c, p) in zip(cols, params)]
    return KernelPlate(result, subjects, nothing, slices, assignments,
        obs, collected, result)
end

# In-cell observation families: surface head => (family enum, total arg
# count). Location-first, scale second positional, the rest `params`.
# Panel admits `Normal` only; grouped admits the joint families too.
const _KERNEL_OBS_FAMILIES = Dict{Symbol,Tuple{Any,Int}}(
    :Normal => (GaussianFam, 2),
    :CensoredAddpropnormal => (CensoredAddpropnormalFam, 4),
    :TgiCategory => (TgiCategoryFam, 7),
    :TgiResponse => (TgiResponseFam, 6),
    :TgiCensored => (TgiCensoredFam, 3))

# One in-cell observation: `yy .~ Fam.(args...)` with a slice-param
# response and name-or-literal args (panel: Gaussian only, exactly one
# obs; grouped: the joint families too, a list; plate: like grouped
# but the response names its data column directly). Inline scale/location
# expressions are NOT admitted — complex values spell via a
# pre-assignment (julianic delta, one line).
function _lower_kernel_obs(stmt::Expr, params::Vector{Symbol}, where,
        form::String = "panel v1")
    resp = stmt.args[2]
    plate = form == "plate v1"
    resp isa Symbol ||
        _sfail(plate ? "$where obs response must name a response " *
               "data column, got $(repr(resp))" :
               "$where obs response must be a bare slice param, got " *
               "$(repr(resp))")
    resp in params ||
        _sfail(plate ? "$where obs response `$resp` is not an observed " *
               "response column" :
               "$where obs response `$resp` is not a slice param " *
               "(responses enter the cell as slices)")
    dist = stmt.args[3]
    (dist isa Expr && dist.head === :.) ||
        _sfail("$where obs broadcasts (`yy .~ Normal.(mu, sigma)`), " *
               "got $(repr(dist))")
    length(dist.args) == 2 && dist.args[1] isa Symbol &&
        dist.args[2] isa Expr && dist.args[2].head === :tuple ||
        _sfail("$where obs takes `yy .~ Fam.(args...)`, got " *
               "$(repr(dist))")
    head = dist.args[1]
    grouped = form != "panel v1"
    if !grouped && head !== :Normal
        _sfail("$where $form admits a Gaussian in-cell observation " *
               "only, got `$head.(...)`")
    end
    spec = get(_KERNEL_OBS_FAMILIES, head, nothing)
    spec === nothing &&
        _sfail("$where $form admits in-cell observations " *
               "`Normal.(...)`, `CensoredAddpropnormal.(...)`, " *
               "`TgiCategory.(...)`, `TgiResponse.(...)`, " *
               "`TgiCensored.(...)` only, got `$head.(...)`")
    fam, arity = spec
    dargs = dist.args[2].args
    length(dargs) == arity ||
        _sfail("$where `$head.(...)` takes exactly $arity arguments, " *
               "got $(length(dargs))")
    for (i, ref) in enumerate(dargs)
        nm = i == 1 ? :location : i == 2 ? :scale : :params
        ref isa Symbol || (ref isa Number && !(ref isa Bool)) ||
            _sfail("$where obs $nm must be a cell/model name or a " *
                   "numeric literal, got $(repr(ref))")
    end
    return (response = resp, family = fam, location = dargs[1],
        scale = dargs[2], params = Tuple(dargs[3:end]))
end

# Grouped-kernel statement (plate form, SB `@plate for` verbatim modulo
# two documented deviations):
#   `@plate <result> for <s> in 1:<N> <cell> end`
# Everything resolves lexically — no argument lists: `.~` LHSs name
# response data columns directly, outer LP definitions are referenced
# by name (interned as subject-level predictors at late lowering),
# schedules/event-LPs enter by separate declarations as before. The
# cell is assignments (cell calls + gathers + arithmetic) + ONE OR
# MORE dotted `.~` observations (one per response axis) + a trailing
# collected name.
#
# Deviations from SB (both forced by the KernelPlate IR): (1) the
# result name rides the header (`@plate pk_loc for ...`) — the IR
# needs a label, dims-key root, and collected alias; (2) the loop
# variable is a declarative axis binder and may go unused — the cell
# is vectorized over the plate (whole-column gathers + per-subject
# unrolled cell calls), not scalar-per-cell, so there is no per-cell
# index to use.
function _is_plate_stmt(st::Expr)
    st.head === :macrocall && length(st.args) == 4 || return false
    st.args[1] === Symbol("@plate") || return false
    st.args[3] isa Symbol || return false
    loop = st.args[4]
    loop isa Expr && loop.head === :for || return false
    return true
end

# Removed grouped form (`result ~ kernel(...) do ... end`, decision
# 0tgodim): matched only to fail with the pointer, never lowered.
function _is_removed_kernel_stmt(st::Expr)
    (_is_sample(st) || _is_broadcast_sample(st)) || return false
    rhs = st.args[3]
    rhs isa Expr && rhs.head === :do || return false
    isempty(rhs.args) && return false
    call = rhs.args[1]
    return call isa Expr && call.head === :call && !isempty(call.args) &&
        call.args[1] === :kernel
end

# Defensive pre-claim of a plate statement's syntactic names (result +
# loop variable + assignment LHSs): late lowering owns every
# rejection, so anything unparseable here is skipped silently and fails
# there with the precise message.
function _claim_plate_stmt_names(st::Expr, line::Int, seen::Set{Symbol},
        seelines::Dict{Symbol,Int})
    _claim!(seen, seelines, st.args[3], line)
    loop = st.args[4]
    loop isa Expr && loop.head === :for && length(loop.args) == 2 ||
        return nothing
    head, body = loop.args[1], loop.args[2]
    if head isa Expr && head.head === :(=) && length(head.args) == 2 &&
            head.args[1] isa Symbol
        _claim!(seen, seelines, head.args[1], line)
    end
    body isa Expr && body.head === :block || return nothing
    for s in body.args
        s isa Expr && s.head === :(=) && length(s.args) == 2 &&
            s.args[1] isa Symbol &&
            _claim!(seen, seelines, s.args[1], line)
    end
    return nothing
end

_is_schedule_decl_rhs(rhs) =
    rhs isa Expr && rhs.head === :call && !isempty(rhs.args) &&
    rhs.args[1] === :linear_pk_schedule

# `name = linear_pk_schedule(obs = (subj, time), dose = (subj, time,
# amount), ecg = (subj, time), tgi = (subj, time))`: raw obs/dose DATA
# columns the bind-time recipe builds the op stream from (D5a), plus
# optional extra read axes (R1: the joint model's ECG + tumor rows join
# the stream as read-only points). Keywords take `;`-style or bare
# form.
function _lower_schedule_decl(lhs::Symbol, rhs::Expr, line::Int,
        data::Set{Symbol})
    where = line > 0 ? "schedule `$lhs` (line $line)" : "schedule `$lhs`"
    kws = Any[]
    for arg in rhs.args[2:end]
        if arg isa Expr && arg.head === :parameters
            append!(kws, arg.args)
        else
            push!(kws, arg)
        end
    end
    obs_spec, dose_spec, ecg_spec, tgi_spec = nothing, nothing, nothing, nothing
    for kw in kws
        kw isa Expr && kw.head === :kw && length(kw.args) == 2 ||
            _sfail("$where takes `obs=(subj, time), dose=(subj, time, " *
                   "amount)` keywords only, got $(repr(kw))")
        key, val = kw.args[1], kw.args[2]
        key === :obs || key === :dose || key === :ecg || key === :tgi ||
            _sfail("$where takes `obs=`/`dose=`/`ecg=`/`tgi=` keywords " *
                   "only, got `$key=`")
        want = key === :dose ? 3 : 2
        cols = _schedule_column_tuple(val, where, key, want, data)
        if key === :obs
            obs_spec === nothing || _sfail("$where repeats `obs=`")
            obs_spec = cols
        elseif key === :dose
            dose_spec === nothing || _sfail("$where repeats `dose=`")
            dose_spec = cols
        elseif key === :ecg
            ecg_spec === nothing || _sfail("$where repeats `ecg=`")
            ecg_spec = cols
        else
            tgi_spec === nothing || _sfail("$where repeats `tgi=`")
            tgi_spec = cols
        end
    end
    obs_spec === nothing && _sfail("$where needs `obs=(subj, time)`")
    dose_spec === nothing && _sfail("$where needs `dose=(subj, time, amount)`")
    ecg = ecg_spec === nothing ? nothing : (ecg_spec[1], ecg_spec[2])
    tgi = tgi_spec === nothing ? nothing : (tgi_spec[1], tgi_spec[2])
    return LinearPKScheduleSpec(lhs, obs_spec[1], obs_spec[2], dose_spec[1],
        dose_spec[2], dose_spec[3], ecg, tgi)
end

_is_event_lp_decl_rhs(rhs) =
    rhs isa Expr && rhs.head === :call && !isempty(rhs.args) &&
    rhs.args[1] === :linear_pk_log_f

# `log_F = linear_pk_log_f(sched; k = 5, c = 1.5)`: the default-V2
# event-axis bioavailability LP (SB `log_F ~ 0 + op_log_dose +
# hsgp(op_log_dose; k = 5)`) over one declared schedule's op stream.
# The LHS is the fixed seam name (the per-subject expansion slices
# the cell-call arg by NAME); the schedule resolves late (any
# statement order — the contract proves declaration); k/c are
# validated literals with the V2 defaults. Keywords take `;`-style
# or bare form (the schedule-decl precedent).
function _lower_event_lp_decl(lhs::Symbol, rhs::Expr, line::Int,
        data::Set{Symbol})
    where = line > 0 ? "event-LP `$lhs` (line $line)" : "event-LP `$lhs`"
    lhs === EVENT_LP_NAME ||
        _sfail("$where: the event-LP provider name is fixed to " *
               "`$(EVENT_LP_NAME)` (SB seam name — the cell call and " *
               "the per-subject expansion address it by name)")
    pos = Any[]
    kws = Any[]
    for arg in rhs.args[2:end]
        if arg isa Expr && arg.head === :parameters
            append!(kws, arg.args)
        elseif arg isa Expr && arg.head === :kw
            push!(kws, arg)
        else
            push!(pos, arg)
        end
    end
    length(pos) == 1 && pos[1] isa Symbol ||
        _sfail("$where takes one schedule handle plus `k=`/`c=` " *
               "keywords, got $(repr(rhs))")
    sched = pos[1]
    sched in data &&
        _sfail("$where takes a declared schedule handle (got bound " *
               "data `$sched` — pass the `linear_pk_schedule` " *
               "declaration's name)")
    k, c = 5, 1.5
    for kw in kws
        kw isa Expr && kw.head === :kw && length(kw.args) == 2 ||
            _sfail("$where takes `k=`/`c=` keywords only, got $(repr(kw))")
        key, val = kw.args[1], kw.args[2]
        if key === :k
            val isa Integer && !(val isa Bool) ||
                _sfail("$where `k` is a positive-integer literal " *
                       "(V2: k = 5, got $(repr(val)))")
            k = Int(val)
        elseif key === :c
            val isa Real ||
                _sfail("$where `c` is a numeric literal exceeding 1 " *
                       "(V2: c = 1.5, got $(repr(val)))")
            c = Float64(val)
        else
            _sfail("$where takes `k=`/`c=` keywords only, got `$key=`")
        end
    end
    k >= 2 ||
        _sfail("$where `k` must be at least 2 (V2: k = 5, got $k)")
    isfinite(c) && c > 1 ||
        _sfail("$where `c` must be finite and exceed 1 (V2: c = 1.5, " *
               "got $(repr(c)))")
    return LinearPKEventLPSpec(lhs, sched, k, c, nothing,
        Symbol(:event_lp_, lhs))
end

# A `:sym`-tuple of bound data columns (parsed `:sym` is a QuoteNode —
# the `hsgp_basis(:id, ...)` precedent unwraps the same way).
function _schedule_column_tuple(val, where, key::Symbol, want::Int,
        data::Set{Symbol})
    val isa Expr && val.head === :tuple && length(val.args) == want ||
        _sfail("$where `$key` is a $want-tuple of bound data " *
               "columns, got $(repr(val))")
    cols = Symbol[]
    for v in val.args
        c = v isa QuoteNode && v.value isa Symbol ? v.value : v
        c isa Symbol ||
            _sfail("$where `$key` is a $want-tuple of bound data " *
                   "columns, got $(repr(val))")
        c in data ||
            _sfail("$where `$key` column `$c` is not bound data")
        push!(cols, c)
    end
    return cols
end

# Late lowering of a plate statement (see `_is_plate_stmt`): parses
# the header/cell, discovers responses (`.~` LHS data columns) and LP
# references (outer definitions used in-cell) lexically, interns the LP
# predictors via the response-location path (subject-level by use),
# resolves schedule references against the declared schedules, and
# assembles the grouped KernelPlate. Unused schedule declarations fail
# closed. Produces IR identical in kind to the removed kernel-do form —
# slices are `(column, column)` and LP cell params are the outer
# definition names — so contract and generator are untouched.
function _lower_plate_stmt(st::Expr, line::Int, data::Set{Symbol}, ctx,
        predictors, pred_idx, coefuse, schedules::Vector{LinearPKScheduleSpec},
        event_lps::Vector{LinearPKEventLPSpec})
    where = line > 0 ? "plate (line $line)" : "plate"
    result = st.args[3]
    result isa Symbol ||
        _sfail("$where result must be a bare Symbol (the collected " *
               "result name), got $(repr(result))")
    result in data &&
        _sfail("$where result `$result` is bound data and cannot " *
               "collect a plate")
    loop = st.args[4]
    loop isa Expr && loop.head === :for && length(loop.args) == 2 ||
        _sfail("$where needs `@plate <result> for <s> in 1:<N> cell end`")
    head, body = loop.args[1], loop.args[2]
    head isa Expr && head.head === :(=) && length(head.args) == 2 &&
        head.args[1] isa Symbol ||
        _sfail("$where loop binds one axis variable " *
               "(`for <s> in 1:<N>`), got $(repr(head))")
    loopvar = head.args[1]
    rng = head.args[2]
    rng isa Expr && rng.head === :call && length(rng.args) == 3 &&
        rng.args[1] === :(:) && rng.args[2] == 1 ||
        _sfail("$where range is `1:<N>` (an integer literal or a " *
               "dims-key name bound at bind), got $(repr(rng))")
    subj = rng.args[3]
    subjects = if subj isa Int
        subj > 0 ||
            _sfail("$where subject count must be positive, got $subj")
        subj
    elseif subj isa Symbol
        subj
    else
        _sfail("$where `1:<N>` takes an integer literal or a dims-key " *
               "name, got $(repr(subj))")
    end
    body isa Expr && body.head === :block ||
        _sfail("$where cell must be a `begin ... end`-style block")
    # Cell: assignments + one or more `.~` + trailing collected name.
    assignments = Pair{Symbol,Any}[]
    obs_stmts = Expr[]
    collected = nothing
    cell = Any[s for s in body.args if !(s isa LineNumberNode)]
    isempty(cell) && _sfail("$where cell is empty (need assignments, " *
                            "`.~` observations, and a collected name)")
    for (k, s) in enumerate(cell)
        last_stmt = k == length(cell)
        if s isa Symbol
            last_stmt ||
                _sfail("$where cell names a bare `$s` mid-cell — only " *
                       "the trailing statement may be a bare name (the " *
                       "collected result)")
            collected = s
        elseif s isa Expr && s.head === :(=) && length(s.args) == 2 &&
                s.args[1] isa Symbol
            push!(assignments, s.args[1] => s.args[2])
        elseif s isa Expr && s.head === :call && length(s.args) == 3 &&
                (s.args[1] === :.~ || s.args[1] === :~)
            s.args[1] === :~ &&
                _sfail("$where in-cell observation broadcasts " *
                       "(`yy .~ Normal.(mu, sigma)`); scalar `~` over " *
                       "vectors is rejected per the explicit-dots ruling")
            push!(obs_stmts, s)
        else
            _sfail("$where cell statements are `name = ...`, one or more " *
                   "`yy .~ Normal.(mu, sigma)`, and a trailing collected " *
                   "name — got $(repr(s))")
        end
    end
    isempty(obs_stmts) &&
        _sfail("$where cell has no `.~` observation (need at least one " *
               "in-cell likelihood)")
    collected === nothing &&
        _sfail("$where cell must end with a collected result name (a " *
               "bare cell name)")
    # Lexical discovery: `.~` LHSs are response data columns; free cell
    # names resolving to outer definitions are LP references (first
    # appearance order — deterministic). Assignment LHSs must not shadow
    # outer names (SB: writes to outside-bound names are rejected).
    cell_locals = Set{Symbol}(nm for (nm, _) in assignments)
    # Shadowing outer definitions, the result, or the loop variable fails
    # at claim time ("defined twice"); bound data is unclaimed, so the
    # data shadow fails here (SB write rule).
    for nm in cell_locals
        nm in data &&
            _sfail("$where cell local `$nm` shadows a bound data column " *
                   "(rename the local — responses name their columns directly)")
    end
    resps = Symbol[]
    for s in obs_stmts
        lhs = s.args[2]
        lhs isa Symbol && lhs in data ||
            _sfail("$where obs response must name a response data " *
                   "column, got $(repr(lhs))")
        lhs in resps &&
            _sfail("$where response `$lhs` is observed twice (one " *
                   "`.~` per response axis)")
        push!(resps, lhs)
    end
    sched_decl = Set{Symbol}(s.name for s in schedules)
    elp_decl = Set{Symbol}(el.name for el in event_lps)
    lpraws = Symbol[]
    extras = Symbol[]
    lpseen = Set{Symbol}()
    for ex in Iterators.flatten(((rhs for (_, rhs) in assignments),
            (s.args[3] for s in obs_stmts)))
        for nm in _plate_value_names(ex)
            (nm in cell_locals || nm === loopvar ||
                nm in sched_decl || nm in elp_decl || nm in lpseen) &&
                continue
            if nm in data
                # Value-position data reads ride auto-slices (the old
                # extra positional inputs); index maps never surface
                # here (gather indices are not values).
                nm in resps || push!(extras, nm)
                push!(lpseen, nm)
                continue
            end
            haskey(ctx.detmap, nm) || continue
            push!(lpraws, nm)
            push!(lpseen, nm)
        end
    end
    isempty(lpraws) &&
        _sfail("$where references no LP definition (LP values gather " *
               "per subject in-cell — reference an outer LP by name)")
    # LP interning via the response-location path (the late-lowering
    # reason): each LP reference names a definition, interned as an
    # identity-link predictor whose coefs join `coefuse` like response
    # locations. Subject level follows from exclusive kernel use. The
    # cell param IS the outer name (lexical, no aliasing).
    lp_args = Tuple{Symbol,Symbol}[]
    for raw in lpraws
        pname = try
            _lower_location(result, raw, IdentityLink, ctx, predictors,
                pred_idx, coefuse)
        catch err
            err isa SurfaceLoweringError && _sfail("$where LP `$raw`: " *
                "$(err.message)")
            rethrow()
        end
        push!(lp_args, (pname, raw))
    end
    obses = KernelObs[_lower_kernel_obs(s, resps, where, "plate v1")
        for s in obs_stmts]
    local_names = union(Set{Symbol}(resps), Set{Symbol}(lpraws),
        Set{Symbol}(nm for (nm, _) in assignments))
    collected in local_names ||
        _sfail("$where collected result `$collected` is not a cell " *
               "name (response column, LP reference, or cell-local assignment)")
    # Schedule references: the cell's call-first-args and gather roots
    # must name declared schedules; unused declarations fail closed.
    schednames = _kernel_cell_schedule_refs(assignments)
    declared = Set{Symbol}(s.name for s in schedules)
    for s in schednames
        s in declared ||
            _sfail("$where references schedule `$s`, which is not " *
                   "declared (`$s = linear_pk_schedule(...)`)")
    end
    for s in schedules
        s.name in schednames ||
            _sfail("$where leaves schedule `$(s.name)` unused (declared " *
                   "schedules must feed the cell — typo'd schedule name?)")
    end
    # Event-LP references: a 7-arg call's second arg must name a
    # declared event-LP; unused declarations fail closed (the
    # schedule precedent — unfed LPs would sample dead parameters).
    elpnames = _kernel_cell_event_lp_refs(assignments)
    declared_elp = Set{Symbol}(el.name for el in event_lps)
    for e in elpnames
        e in declared_elp ||
            _sfail("$where references event-LP `$e`, which is not " *
                   "declared (`$e = linear_pk_log_f(sched; k = 5)`)")
    end
    for el in event_lps
        el.name in elpnames ||
            _sfail("$where leaves event-LP `$(el.name)` unused " *
                   "(declared event-LPs must feed a 7-arg cell call — " *
                   "typo'd event-LP name?)")
    end
    used = [s for s in schedules if s.name in schednames]
    slices = Tuple{Symbol,Symbol,Symbol}[(r, r, :unknown)
        for r in Iterators.flatten((resps, extras))]
    return KernelPlate(result, subjects, nothing, slices, assignments,
        obses, collected, result, lp_args, used)
end

# Value-position names of a plate-cell expression in first-appearance
# order: gather indices (`v[map]`) contribute nothing (maps are
# positions, never values), as do function heads (`f(...)`, `f.(...)`),
# property tags (`x.tag`), and literals; everything else reads
# lexically. Lenient collection — the contract cell walker owns precise
# shape rejection; unknown names fail there.
function _plate_value_names(ex)
    out = Symbol[]
    _collect_plate_value_names!(out, ex)
    return out
end

function _collect_plate_value_names!(out::Vector{Symbol}, ex)
    ex isa Symbol && (push!(out, ex); return nothing)
    ex isa Expr || return nothing
    if ex.head === :ref && length(ex.args) == 2
        _collect_plate_value_names!(out, ex.args[1])
        return nothing
    end
    if ex.head === :call && !isempty(ex.args)
        for a in ex.args[2:end]
            _collect_plate_value_names!(out, a)
        end
        return nothing
    end
    if ex.head === :.
        if length(ex.args) >= 2 && ex.args[2] isa QuoteNode
            # Getproperty `x.tag`: the object reads; the tag is static.
            _collect_plate_value_names!(out, ex.args[1])
        else
            # Broadcast `f.(args...)`: the head is static.
            for a in ex.args[2:end]
                _collect_plate_value_names!(out, a)
            end
        end
        return nothing
    end
    for a in ex.args
        a isa QuoteNode && continue
        _collect_plate_value_names!(out, a)
    end
    return nothing
end

# Schedule handles a grouped cell references (call first-args of CELL_FNS
# calls + gather roots): lenient collection — malformed shapes belong to
# the contract cell walker, which fails with the precise message.
function _kernel_cell_schedule_refs(assignments::Vector{Pair{Symbol,Any}})
    refs = Set{Symbol}()
    for (_, ex) in assignments
        _collect_schedule_refs!(refs, ex)
    end
    return refs
end

function _collect_schedule_refs!(refs::Set{Symbol}, ex)
    ex isa Expr || return nothing
    if ex.head === :call && !isempty(ex.args) && ex.args[1] isa Symbol &&
            ex.args[1] in CELL_FNS && length(ex.args) >= 2 &&
            ex.args[2] isa Symbol
        push!(refs, ex.args[2])
    end
    if ex.head === :ref && length(ex.args) == 2
        idx = ex.args[2]
        idx isa Expr && idx.head === :. && length(idx.args) == 2 &&
            idx.args[1] isa Symbol && push!(refs, idx.args[1])
    end
    for a in ex.args
        _collect_schedule_refs!(refs, a)
    end
    return nothing
end

# Event-LP names a grouped cell references (second args of 7-arg
# CELL_FNS calls): lenient collection — the contract cell walker
# owns precise shape rejection.
function _kernel_cell_event_lp_refs(assignments::Vector{Pair{Symbol,Any}})
    refs = Set{Symbol}()
    for (_, ex) in assignments
        _collect_event_lp_refs!(refs, ex)
    end
    return refs
end

function _collect_event_lp_refs!(refs::Set{Symbol}, ex)
    ex isa Expr || return nothing
    if ex.head === :call && length(ex.args) == 8 &&
            ex.args[1] isa Symbol && ex.args[1] in CELL_FNS &&
            ex.args[3] isa Symbol
        push!(refs, ex.args[3])
    end
    for a in ex.args
        _collect_event_lp_refs!(refs, a)
    end
    return nothing
end

function _partition_statements(ast::Expr, data::Set{Symbol})
    sample = SampleStmt[]
    det = Pair{Symbol,Any}[]
    scans = ScanSpec[]
    bases = SplineBasis[]
    vectors = SplineVector[]
    hbases = HSGPBasis[]
    kplates = KernelPlate[]
    kstmts = NamedTuple[]
    schedules = LinearPKScheduleSpec[]
    event_lps = LinearPKEventLPSpec[]
    r2d2decls = NamedTuple[]
    joints = JointSampleStmt[]
    glms = GLMSampleStmt[]
    varying_raw = NamedTuple[]
    level_bindings = Dict{Symbol,Tuple{Symbol,Any}}()
    seen = Set{Symbol}()
    seelines = Dict{Symbol,Int}()
    seen_doc = false
    line = 0
    args, plate_ctx, plate_params = _expand_plates(ast.args, data)
    # Defined names for the varying partition-time gate: draws blocks
    # lower in statement order, before shapes exist, so margins admit
    # data-or-defined names here (forward references work) and prove
    # vector shape after lowering (`_validate_varying_margins`). The
    # scan never throws — the main loop below owns every rejection.
    detnames = Set{Symbol}()
    for arg in args
        arg isa Expr || continue
        st = try
            _unwrap_trivia(arg)
        catch
            continue
        end
        st.head === :(=) && length(st.args) == 2 && st.args[1] isa Symbol &&
            push!(detnames, st.args[1])
    end
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
        if _is_basis_stmt(st)
            b, vs = _lower_basis(st, line, data, seen, seelines, bases)
            push!(bases, b)
            append!(vectors, vs)
            continue
        end
        if _is_hsgp_basis_stmt(st)
            hb = _lower_hsgp_basis(st, line, data, seen, seelines, hbases)
            push!(hbases, hb)
            continue
        end
        if _is_kernel_plate_stmt(st)
            kp = _lower_kernel_plate(st, line, data, seen, seelines)
            push!(kplates, kp)
            continue
        end
        if _is_plate_stmt(st)
            # Plates lower LATE (LP interning needs the response-loop
            # predictor table): claim the syntactic names now
            # (defensive — late lowering owns every rejection) and
            # stash the statement.
            _claim_plate_stmt_names(st, line, seen, seelines)
            push!(kstmts, (st = st, line = line))
            continue
        end
        if _is_removed_kernel_stmt(st)
            _sfail("grouped `kernel(...) do ... end` was removed " *
                   "(decision 0tgodim) — spell " *
                   "`@plate <result> for <s> in 1:<N> ... end`")
        end
        if _is_r2d2_stmt(st)
            push!(r2d2decls, _lower_r2d2_decl(st, line))
            continue
        end
        if _is_sample(st) || _is_broadcast_sample(st)
            bc = _is_broadcast_sample(st)
            tilde = bc ? "`.~`" : "`~`"
            # A vector LHS is the joint correlated-outcomes form (plain `~`
            # only — row-grouped, never broadcast).
            if st.args[2] isa Expr && st.args[2].head === :vect
                bc && _sfail("joint responses use `~`, not `.~` " *
                             "(`[y1, y2] ~ MvNormalCholesky([mu1, mu2], L)` " *
                             "— row-grouped, never broadcast)")
                j = _parse_joint_stmt(st, line, data)
                for o in j.outcomes
                    _claim!(seen, seelines, o, line)
                end
                push!(joints, j)
                continue
            end
            if !bc && st.args[3] isa Expr && st.args[3].head === :call &&
                    !isempty(st.args[3].args) &&
                    st.args[3].args[1] isa Symbol &&
                    st.args[3].args[1] in _GLM_HEADS
                g = _parse_glm_stmt(st, line, data)
                _claim!(seen, seelines, g.response, line)
                push!(glms, g)
                continue
            end
            if bc && st.args[3] isa Expr && st.args[3].head === :call &&
                    !isempty(st.args[3].args) &&
                    st.args[3].args[1] isa Symbol &&
                    st.args[3].args[1] in _GLM_HEADS
                _sfail("GLM-object heads use whole-data `~`, not `.~` " *
                       "(`$(st.args[3].args[1])(X, alpha, beta)` — the " *
                       "object owns eta over the whole column)")
            end
            lhs, rng, levs, mat = _sample_lhs(st.args[2], bc, tilde, data,
                level_bindings)
            lhs === :dummy &&
                _sfail("`dummy` is reserved (margin surface) and cannot " *
                       "be sampled")
            lhs in (:spline, :spline_basis) &&
                _sfail("`$lhs` is reserved (spline surface) and cannot be " *
                       "sampled")
            lhs in (:hsgp, :hsgp_basis) &&
                _sfail("`$lhs` is reserved (hsgp surface) and cannot be " *
                       "sampled")
            lhs === :r2d2 &&
                _sfail("`r2d2` is reserved (r2d2 surface) and cannot be " *
                       "sampled")
            _claim!(seen, seelines, lhs, line)
            if _is_varying_call(st.args[3])
                head = _varying_head(st.args[3])
                bc && _sfail("`$lhs` uses `.~` — `$head` statements bind " *
                             "with `~` (one binding per statement)")
                st.args[2] isa Symbol ||
                    _sfail("`$head` left-hand side must be a bare Symbol " *
                           "(one binding per statement)")
                lhs in data &&
                    _sfail("`$lhs` is bound data and cannot bind a " *
                           "`$head` statement")
                push!(varying_raw, (st = st, lhs = lhs, line = line))
                continue
            end
            _reject_target(st.args[3], lhs)
            push!(sample, SampleStmt(lhs, st.args[3], bc, rng, levs, mat))
        elseif st.head === :(=) && length(st.args) == 2 && st.args[1] isa Symbol
            lhs = st.args[1]
            lhs === :target && _sfail("no `target` in rkppl models " *
                                      "(density comes only from `~`)")
            lhs === :dummy &&
                _sfail("`dummy` is reserved (margin surface) and cannot " *
                       "be redefined")
            lhs in (:spline, :spline_basis) &&
                _sfail("`$lhs` is reserved (spline surface) and cannot be " *
                       "redefined")
            lhs in (:hsgp, :hsgp_basis) &&
                _sfail("`$lhs` is reserved (hsgp surface) and cannot be " *
                       "redefined")
            lhs === :r2d2 &&
                _sfail("`r2d2` is reserved (r2d2 surface) and cannot be " *
                       "redefined")
            lhs in data && _sfail("$lhs is bound data and cannot be redefined")
            _claim!(seen, seelines, lhs, line)
            if _is_schedule_decl_rhs(st.args[2])
                push!(schedules, _lower_schedule_decl(lhs, st.args[2], line,
                    data))
                continue
            end
            if _is_event_lp_decl_rhs(st.args[2])
                push!(event_lps, _lower_event_lp_decl(lhs, st.args[2], line,
                    data))
                continue
            end
            if _is_levels_binding_rhs(st.args[2])
                push!(level_bindings,
                    lhs => _lower_levels_binding(lhs, st.args[2], data))
                continue
            end
            _reject_target(st.args[2], lhs)
            push!(det, lhs => st.args[2])
        else
            _reject_statement(st)
        end
    end
    # Varying statements lower draws blocks first (statement order),
    # then slices — slices may precede their draws textually (forward
    # references resolve against lowered draws).
    varying_draws = VaryingDraws[]
    draws_by_lhs = Dict{Symbol,VaryingDraws}()
    draws_lines = Dict{Symbol,Int}()
    used_suffixes = Set{String}()
    for v in varying_raw
        _varying_head(v.st.args[3]) === :varying_slice && continue
        d = _lower_varying_draws_block(v.lhs, v.st.args[3], v.line, data,
            detnames, seen, seelines, used_suffixes)
        push!(varying_draws, d)
        draws_by_lhs[v.lhs] = d
        draws_lines[d.label] = v.line
    end
    varying_pending = NamedTuple[]
    for v in varying_raw
        head = _varying_head(v.st.args[3])
        if head === :varying_effect
            d = draws_by_lhs[v.lhs]
            K = length(d.margins)
            push!(varying_pending, (contrib = v.lhs, draws = d.label,
                draws_lhs = v.lhs, columns = 1:K, line = v.line))
        elseif head === :varying_slice
            push!(varying_pending,
                _lower_varying_slice(v.lhs, v.st.args[3], v.line,
                    draws_by_lhs))
        end
    end
    # Same-group draws share one per-group encoder: both-declared
    # differing levels fail here with lines (bind-derived levels agree
    # by construction; mixed declared/derived agreement is a bind-time
    # check on values).
    levels_by_group = Dict{Symbol,Tuple{Any,Int}}()
    for d in varying_draws
        d.levels === nothing && continue
        if haskey(levels_by_group, d.group)
            prev, pline = levels_by_group[d.group]
            prev == d.levels ||
                _sfail("draws $(d.label) declares grouping levels " *
                      "$(repr(d.levels)) but same-group draws already " *
                      "declared $(repr(prev)) (line $pline) — one " *
                      "grouping, one numbering")
        else
            levels_by_group[d.group] = (d.levels, draws_lines[d.label])
        end
    end
    return sample, det, plate_ctx, plate_params, scans, bases,
        vectors, hbases, kplates, kstmts, schedules, event_lps, r2d2decls,
        joints, varying_draws, varying_pending, glms
end

# ── Design-matrix extraction (slice D1) ─────────────────────────────
# `X = hcat(1, x, ...)` definitions leave `det` for the plan-level
# `matrices` table (the ranef-bucket/spline-basis precedent: special
# statements lower to plan tables, not kernel assignments). The generator
# emits each matrix once (`X = Float64.(hcat(...))`); predictor matmuls
# (`mu = X * b`) reference it by name. Columns are the intercept `1`
# plus bare data/derived-data columns — latent, scan, parameter, and
# nested-matrix columns fail closed here; duplicate columns and double
# intercepts fail in contract validation. Only named definitions are
# admitted: inline `hcat` binds to a name first (a synth-naming
# follow-up, the `_extract_column` precedent).

_is_hcat_def(rhs) =
    rhs isa Expr && rhs.head === :call && !isempty(rhs.args) &&
    rhs.args[1] === :hcat

# Any `hcat` call under `ex` (QuoteNodes opaque — a quoted `:hcat` is
# not a call).
_find_hcat(ex) = _find_hcat!(ex, Ref(false))[]

function _find_hcat!(ex, found)
    found[] && return found
    ex isa Expr || return found
    if _is_hcat_def(ex)
        found[] = true
        return found
    end
    for a in ex.args
        _find_hcat!(a, found)
    end
    return found
end

# A surviving reference to a matrix name (outside QuoteNodes and
# outside `matrix * name` matmul-shaped nodes, which predictor
# classification owns — and rejects unless the name is a coefficient
# vector). Non-symbol right-hand sides and matrix right-hand sides are
# NOT skipped: they cannot be matmuls, so the use is a stray here.
function _find_matrix_use(ex, matrices)
    ex isa Symbol && return ex in matrices ? ex : nothing
    ex isa Expr || return nothing
    if ex.head === :call && length(ex.args) == 3 && ex.args[1] === :* &&
            ex.args[2] isa Symbol && ex.args[2] in matrices &&
            ex.args[3] isa Symbol && ex.args[3] ∉ matrices
        return nothing
    end
    for a in ex.args
        hit = _find_matrix_use(a, matrices)
        hit === nothing || return hit
    end
    return nothing
end

# Whether `ex` encloses a `matrix * _` node of any right-hand side
# (literal scalings like `2 * (X * b)` reach the nonlinearity
# fallthrough — this names the matmul instead).
function _contains_matmul(ex, matrices)
    ex isa Expr || return false
    if ex.head === :call && length(ex.args) == 3 && ex.args[1] === :* &&
            ex.args[2] isa Symbol && ex.args[2] in matrices
        return true
    end
    return any(a -> _contains_matmul(a, matrices), ex.args)
end

# A composed matmul names its fix by composition: literal scaling
# names the prior, parameter scaling names slice-1, anything else
# names the bare-summand spelling.
function _matmul_composition_error(pname, core, ctx)
    _has_number(core) && _sfail(
        "predictor $pname: literal scaling of a matmul is not a term — " *
        "scale the coefficient prior instead")
    for s in _value_symbols(core)
        _summand_kind(s, ctx) === :param && _sfail(
            "predictor $pname: $(repr(core)) scales a matmul by the " *
            "parameter $s — computed coefficients are not in slice 1")
    end
    return _sfail("predictor $pname: $(repr(core)) composes a matmul " *
                  "outside a term — a matmul is a complete summand " *
                  "(`mu = ... .+ X * b`)")
end

function _has_number(ex)
    ex isa Number && return true
    ex isa Expr || return false
    return any(_has_number, ex.args)
end

function _extract_matrices(det, detmap, detshape, data, prior_names,
        plate_names, scans)
    matrices = DesignMatrix[]
    kept = Pair{Symbol,Any}[]
    for (nm, _) in det
        rhs = detmap[nm]
        _is_hcat_def(rhs) || (push!(kept, nm => rhs); continue)
        args = rhs.args[2:end]
        isempty(args) && _sfail("design matrix `$nm = $(repr(rhs))` " *
                                "calls `hcat` with no columns — a design " *
                                "matrix needs at least one " *
                                "(`$nm = hcat(1, x, ...)`)")
        cols = Union{Nothing,Symbol}[]
        for a in args
            if a isa Number && !(a isa Bool) && a == 1
                push!(cols, nothing)
            elseif a isa Symbol
                push!(cols, _matrix_column(nm, a, detshape, detmap, data,
                    prior_names, plate_names, scans))
            else
                _sfail("design matrix `$nm` has a non-column argument " *
                       "$(repr(a)) — columns are the intercept `1` or " *
                       "bare data/derived columns " *
                       "(`$nm = hcat(1, x, ...)`); bind richer " *
                       "expressions to a name first")
            end
        end
        push!(matrices, DesignMatrix(nm, cols, nm))
    end
    # Strays in the remaining definitions: inline `hcat` binds to a name;
    # matrix names lower only in predictor matmuls.
    matnames = Set{Symbol}(m.name for m in matrices)
    for (nm, rhs) in kept
        _find_hcat(rhs) && _sfail("definition `$nm` calls `hcat` outside " *
                                  "a matrix definition — bind the matrix " *
                                  "to a name first (`X = hcat(1, x, ...)`)")
        hit = _find_matrix_use(rhs, matnames)
        hit === nothing || _sfail("definition `$nm` uses design matrix " *
                                  "`$hit` outside a predictor matmul — a " *
                                  "design matrix lowers only as " *
                                  "`mu = $hit * b`")
    end
    return kept, matrices
end

function _matrix_column(nm, c, detshape, detmap, data, prior_names,
        plate_names, scans)
    get(detshape, c, :scalar) === :matrix &&
        _sfail("design matrix `$nm` nests matrix `$c` — nested hcat is " *
               "not in slice D1 (flatten it)")
    c in data && return c
    if haskey(detmap, c)
        detshape[c] === :vector && return c
        _sfail("design matrix `$nm` over `$c`, which is scalar — " *
               "matrices take the intercept `1` plus vector columns")
    end
    c in prior_names && _sfail("design matrix `$nm` over sampled " *
                               "parameter `$c` is not in slice D1 " *
                               "(data/derived columns only)")
    c in plate_names && _sfail("design matrix `$nm` over the latent " *
                               "vector `$c` is not in slice D1 (the me " *
                               "mirror stays affine)")
    any(s -> s.state === c, scans) && _sfail("design matrix `$nm` over " *
                                             "scan state `$c` is not in " *
                                             "slice D1 (data/derived " *
                                             "columns only)")
    _sfail("design matrix `$nm` over unknown name `$c` — columns are " *
           "the intercept `1` or bare data/derived columns")
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
                arg.args[1] === Symbol("@plate") && length(arg.args) == 3
            # Bare `@plate for ...` (3-arg macrocall) desugars here; the
            # 4-arg kernel form (`@plate <result> for ...`) passes
            # through to `_is_plate_stmt` dispatch below.
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
# `y[R]` (`.~` only), a levels ref `c[levels(g)]` / `c[levels(g)][S]`
# (`.~` only), or a matrix-sized ref `b[axes(X, 2)]` (`.~` only).
# Returns `(column, range, levels, matrix)` with at most one of `range` /
# `levels` / `matrix` set (`matrix` is the sizing design matrix).
_sample_lhs(lhs::Symbol, bc, tilde, data,
        level_bindings = Dict{Symbol,Tuple{Symbol,Any}}()) =
    (lhs, nothing, nothing, nothing)
function _sample_lhs(lhs, bc, tilde, data, level_bindings)
    lhs isa Expr || _sfail("$tilde left-hand side must be a bare Symbol, " *
                           "a range ref (`y[1:N]`), a levels ref " *
                           "(`c[levels(g)]`), or a matrix-sized ref " *
                           "(`b[axes(X, 2)]`), got $(repr(lhs))")
    lhs.head === :. && _sfail("dotted left-hand side $(repr(lhs)) does " *
                              "not lower (nested targets are out of scope)")
    lhs.head === :ref && length(lhs.args) == 2 || _sfail(
        "$tilde left-hand side must be a bare Symbol or a one-dimensional " *
        "ref (`y[1:N]`, `c[levels(g)]`, `b[axes(X, 2)]`), got $(repr(lhs))")
    target, index = lhs.args
    # Chained outside subset (`c[levels(g)][2:end]`): one way to write it —
    # the subset goes inside (`c[levels(g)[2:end]]`).
    target isa Expr && _sfail("$tilde subset goes inside the levels " *
                              "expression (`c[levels(g)[2:end]]`), got " *
                              "$(repr(lhs))")
    target isa Symbol || _sfail("$tilde left-hand side must be a bare " *
                                "Symbol or a one-dimensional ref, got " *
                                "$(repr(lhs))")
    # `b[axes(X, 2)]`: coefficient-vector sizing over a design matrix
    # (the `axes(col, 1)` response-range precedent, second dimension).
    # Matrix existence is checked at prior lowering (defs may follow).
    if _is_axes2_call(index)
        bc || _sfail("sized prior `$(target)[axes(...)]` is a vector — " *
                     "use `.~`, not `~`")
        target in data && _sfail("`axes` sizes coefficient priors, not " *
                                 "responses ($target is data)")
        return target, nothing, nothing, index.args[2]
    end
    if index isa Expr && index.head === :call && !isempty(index.args) &&
            index.args[1] === :axes &&
            !(length(index.args) == 3 && index.args[2] === target &&
                index.args[3] == 1) && target ∉ data
        _sfail("coefficient vector $target takes `axes(X, 2)` over its " *
               "design matrix — got $(repr(index)) (response ranges take " *
               "`axes($target, 1)`; `size` is not a sizing form)")
    end
    # `c[levels(g)[S]]`: subset selection over the levels.
    if index isa Expr && index.head === :ref
        bc || _sfail("sized prior `$(target)[levels(...)[...]]` is a " *
                     "vector — use `.~`, not `~`")
        target in data && _sfail("`levels` sizes coefficient priors, not " *
                                 "responses ($target is data)")
        gcol, sub = _levels_subset_index(target, index, data)
        return target, nothing, (gcol, sub), nothing
    end
    if _is_levels_call(index)
        bc || _sfail("sized prior `$(target)[levels(...)]` is a vector — " *
                     "use `.~`, not `~`")
        target in data && _sfail("`levels` sizes coefficient priors, not " *
                                 "responses ($target is data)")
        gcol = _levels_column(target, index, data)
        return target, nothing, (gcol, Colon()), nothing
    end
    if index isa Symbol
        # Data targets keep the pre-levels path verbatim (per-cell refs
        # belong in `@plate`; only coefficient targets see levels logic).
        target in data &&
            return target, _lower_lhs_range(target, index), nothing, nothing
        bc || _sfail("sized prior `$(target)[$(index)]` is a vector — " *
                     "use `.~`, not `~`")
        haskey(level_bindings, index) || _sfail(
            "coefficient $target: index name $index must be a bound " *
            "levels-subset declared as `$index = levels(group)` or " *
            "`$index = levels(group)[subset]`, got $(repr(index))")
        return target, nothing, level_bindings[index], nothing
    end
    bc || _sfail("sliced response `$target[...]` is a vector — " *
                 "use `.~`, not `~`")
    return target, _lower_lhs_range(target, index), nothing, nothing
end

_is_axes2_call(index) =
    index isa Expr && index.head === :call && length(index.args) == 3 &&
    index.args[1] === :axes && index.args[2] isa Symbol &&
    index.args[3] == 2

# Joint-response statement: `[y1, y2] ~ MvNormalCholesky([mu1, mu2], L)`
# (SB's joint form). Outcomes are bare distinct data symbols; means are
# one location expression per outcome; the factor names an
# `LKJCovarianceFactor` declaration. Width/factor linkage checks belong
# to `_lower_joint_response` + contract validation.
function _parse_joint_stmt(st::Expr, line::Int, data::Set{Symbol})
    outs = st.args[2].args
    isempty(outs) && _sfail("joint response `[]` is empty — name the " *
                            "outcome data columns " *
                            "(`[y1, y2] ~ MvNormalCholesky([mu1, mu2], L)`)")
    for o in outs
        o isa Symbol || _sfail("joint outcomes are bare data columns, " *
                               "got $(repr(o))")
        o in data || _sfail("joint outcome $o is not data (joint " *
                            "responses observe data columns — $o is not " *
                            "among the bound data names)")
    end
    length(unique(outs)) == length(outs) ||
        _sfail("joint outcomes repeat a column " *
               "($(join(outs, ", ")))")
    rhs = st.args[3]
    rhs isa Expr && rhs.head === :call && !isempty(rhs.args) &&
        rhs.args[1] === :MvNormalCholesky ||
        _sfail("a `[y1, y2]` response takes " *
               "`MvNormalCholesky([mu1, mu2], L)`, got $(repr(rhs))")
    args = _plain_args(rhs, "`MvNormalCholesky`")
    length(args) == 2 || _sfail("`MvNormalCholesky` takes ([means], " *
                                "factor) " *
                                "(`[y1, y2] ~ MvNormalCholesky([mu1, mu2], L)`), " *
                                "got $(length(args)) arguments")
    means, factor = args
    means isa Expr && means.head === :vect &&
        length(means.args) == length(outs) ||
        _sfail("joint response [$(join(outs, ", "))] has " *
               "$(length(outs)) outcomes but $(repr(means)) means " *
               "(one mean per outcome: `[mu1, mu2]`)")
    factor isa Symbol || _sfail("joint factor $(repr(factor)) must name " *
                                "an `LKJCovarianceFactor` declaration " *
                                "(`L ~ LKJCovarianceFactor(K, Exponential(1.0), eta)` " *
                                "in the model)")
    _reject_target(rhs, Symbol(join(outs, "_")))
    return JointSampleStmt(Vector{Symbol}(outs), Vector{Any}(means.args),
        factor, line)
end

# The `levels(g)[S]` index of a subset prior: returns `(g, subset)`.
function _levels_subset_index(col::Symbol, index::Expr, data::Set{Symbol})
    length(index.args) == 2 && _is_levels_call(index.args[1]) ||
        _sfail("coefficient $col: subsets select over `levels` " *
               "(`$col[levels(g)[2:end]]`), got $(repr(index))")
    gcol = _levels_column(col, index.args[1], data)
    return gcol, _lower_levels_subset(col, gcol, index.args[2])
end

# A model-level levels binding: exactly the inline levels grammar under a
# name (`sel = levels(g)` or `sel = levels(g)[subset]`). The stored value is
# the same `(grouping, subset)` pair carried by a `SampleStmt`, so reuse in
# coefficient indices follows the ordinary inline lowering path.
_is_levels_binding_rhs(rhs) =
    _is_levels_call(rhs) ||
    rhs isa Expr && rhs.head === :ref && length(rhs.args) == 2 &&
        _is_levels_call(rhs.args[1])

function _lower_levels_binding(name::Symbol, rhs, data::Set{Symbol})
    if rhs isa Expr && rhs.head === :ref && length(rhs.args) == 2
        call, subset = rhs.args
    else
        call, subset = rhs, nothing
    end
    gcol = _levels_column(name, call, data, "levels binding $name")
    subset === nothing && return (gcol, Colon())
    return gcol, _lower_levels_subset(name, gcol, subset)
end

_is_levels_call(x) =
    x isa Expr && x.head === :call && !isempty(x.args) &&
    x.args[1] isa Symbol && x.args[1] in (:levels, :unique, :sort)

function _levels_column(col::Symbol, call::Expr, data::Set{Symbol},
        where::AbstractString = "coefficient $col")
    fn = call.args[1]
    fn === :levels || _sfail("$where: write `levels(...)`, not " *
                             "`$fn(...)` (the levels function is `levels`)")
    length(call.args) == 2 ||
        _sfail("$where: `levels` takes exactly one grouping " *
               "column, got $(repr(call))")
    gcol = call.args[2]
    gcol isa Symbol || _sfail("$where: `levels` takes a bare " *
                              "grouping column, got $(repr(gcol))")
    gcol in data || _sfail("$where: `levels($gcol)` needs a " *
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
(`nothing` otherwise). `matrix` carries the sizing design matrix for
`b[axes(X, 2)]` coefficient-vector priors (`nothing` otherwise)."""
struct SampleStmt
    lhs::Symbol
    rhs::Any
    broadcast::Bool
    range::Union{Nothing,UnitRange{Int}}
    levels::Any
    matrix::Union{Nothing,Symbol}
end
SampleStmt(lhs::Symbol, rhs, broadcast::Bool) =
    SampleStmt(lhs, rhs, broadcast, nothing, nothing, nothing)

"""One joint `[y1, y2] ~ MvNormalCholesky([mu1, mu2], L)` statement: K
outcome columns, K mean expressions, and the factor stem (an
`LKJCovarianceFactor` declaration elsewhere in the model). Plain `~`
only — the likelihood groups rows, never broadcasts."""
struct JointSampleStmt
    outcomes::Vector{Symbol}
    means::Vector{Any}
    factor::Symbol
    line::Int
end

"""One whole-data GLM-object statement
(`y ~ NormalIDGLM(X, alpha, beta, sigma)`): the response column, the
head family, the intercept-free design-matrix name, the scalar
intercept and coefficient-vector parameters, and the Normal sigma
(parameter name or positive literal, `nothing` otherwise). Plain `~`
only — the object owns eta over the whole column, never broadcasts."""
struct GLMSampleStmt
    response::Symbol
    head::Symbol
    matrix::Symbol
    alpha::Symbol
    beta::Symbol
    sigma::Any
    line::Int
end

const _GLM_HEADS = (:NormalIDGLM, :BernoulliLogitGLM, :PoissonLogGLM)

function _parse_glm_stmt(st::Expr, line::Int, data::Set{Symbol})
    lhs, rhs = st.args[2], st.args[3]
    head = rhs.args[1]
    lhs isa Symbol || _sfail("`$head` left-hand side must be a bare " *
                             "data column, got $(repr(lhs))")
    lhs in data || _sfail("`$head` responds over data — $lhs is not " *
                          "a data column")
    args = _plain_args(rhs, "`$head`")
    want = head === :NormalIDGLM ? 4 : 3
    usage = head === :NormalIDGLM ? "`$head(X, alpha, beta, sigma)`" :
        "`$head(X, alpha, beta)`"
    length(args) == want || _sfail("response $lhs: `$head` takes " *
                                   "$usage, got $(length(args)) arguments")
    X, alpha, beta = args[1], args[2], args[3]
    X isa Symbol || _sfail("response $lhs: `$head` design matrix " *
                           "must be a name (`X = hcat(...)`), got $(repr(X))")
    alpha isa Symbol || _sfail("response $lhs: `$head` intercept " *
                               "must be a parameter name, got $(repr(alpha))")
    beta isa Symbol || _sfail("response $lhs: `$head` coefficients " *
                              "must be a parameter name, got $(repr(beta))")
    sigma = nothing
    if head === :NormalIDGLM
        sigma = args[4]
        (sigma isa Symbol || sigma isa Real) || _sfail(
            "response $lhs: `$head` sigma must be a parameter name or " *
            "a positive literal, got $(repr(sigma))")
    end
    return GLMSampleStmt(lhs, head, X, alpha, beta, sigma, line)
end

_is_doc_macro(m) =
    m === Symbol("@doc") || (m isa GlobalRef && m.name === Symbol("@doc"))

function _unwrap_trivia(st::Expr)
    while st.head === :macrocall
        # The 4-arg kernel form (`@plate <result> for ...`) is a
        # top-level statement in its own right — pass it through to
        # `_is_plate_stmt` dispatch (only the bare desugared form nests
        # illegally below).
        if st.args[1] === Symbol("@plate") && length(st.args) == 4
            return st
        end
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
# submodel's RETURN (a trailing `return x` unwraps to `x` in
# `_submodel_body_parts`, so both spellings lower identically):
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
    pins = Dict{Symbol,Symbol}()
    any(_stmt_is_submodel_call(a, mod) || _plate_has_submodel_cell(a, mod)
        for a in ast.args) || return ast, pins
    out = Any[]
    for arg in ast.args
        if _stmt_is_submodel_call(arg, mod)
            st = _unwrap_trivia(arg)
            append!(out, _expand_one_submodel(st.args[2], st.args[3], mod,
                data, pins))
        elseif _plate_has_submodel_cell(arg, mod)
            push!(out, _expand_plate_cell_submodels(arg, mod, data))
        else
            push!(out, arg)
        end
    end
    return Expr(:block, out...), pins
end

_ns(lhs::Symbol, nm::Symbol) = Symbol(lhs, :_, nm)

# Split a submodel body into (statements, return-expression). A trailing
# explicit `return x` unwraps to `x`, so it lowers identically to the implicit
# trailing-expression form on every path (stream + latent, top-level +
# per-cell): all of them read `ret` from here.
function _submodel_body_parts(sm::RKPPLSubmodel)
    items = Any[a for a in sm.body.args if !(a isa LineNumberNode)]
    isempty(items) && _sfail("submodel `$(sm.name)` has an empty body")
    ret = last(items)
    if Meta.isexpr(ret, :return)
        # A bare `return` parses as `Expr(:return, nothing)` (length 1, value
        # `nothing` — indistinguishable from an explicit `return nothing`, and
        # equally unbindable), so the arity check alone cannot fail it closed.
        unwrap = length(ret.args) == 1 ? ret.args[1] : nothing
        unwrap === nothing && _sfail(
            "submodel `$(sm.name)` must end in a RETURN expression bound to " *
            "the use-site LHS — a bare `return` returns nothing")
        ret = unwrap
    end
    stmts = Any[_unwrap_trivia(st) for st in items[1:end-1]]
    for st in stmts
        Meta.isexpr(st, :return) && _sfail(
            "submodel `$(sm.name)`: `return` is only admitted as the " *
            "trailing expression (submodels are straight-line; no early return)")
        (st isa Expr && (_is_sample(st) || _is_broadcast_sample(st)) &&
            _is_varying_call(st.args[3])) && _sfail(
            "submodel `$(sm.name)`: varying statements lower only at " *
            "top level (submodel-body extension is future work)")
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

# Peel a `predictor = name` use-site pin from a submodel call: returns
# (positional-callargs, pin-or-nothing). `predictor` is the only admitted
# keyword and only as a bare Symbol; anything else fails naming the call.
function _peel_predictor_pin(callexpr::Expr, sm::RKPPLSubmodel)
    posargs = Any[]
    pin = nothing
    for a in callexpr.args[2:end]
        if a isa Expr && a.head === :parameters
            for kw in a.args
                kw isa Expr && kw.head === :kw && length(kw.args) == 2 ||
                    _sfail("submodel `$(sm.name)`: malformed keyword " *
                           "$(repr(a)) (submodel calls take " *
                           "`predictor = name` only)")
                k = kw.args[1]
                k === :predictor ||
                    _sfail("submodel `$(sm.name)` takes keyword " *
                           "`predictor` only, got `$k`")
                pin === nothing ||
                    _sfail("submodel `$(sm.name)`: duplicate `predictor =` " *
                           "(`$pin` and `$(kw.args[2])` — one pin per call)")
                v = kw.args[2]
                v isa Symbol ||
                    _sfail("submodel `$(sm.name)`: `predictor` takes a bare " *
                           "predictor name (a Symbol), got $(repr(v))")
                pin = v
            end
        else
            push!(posargs, a)
        end
    end
    return posargs, pin
end

function _expand_one_submodel(lhs::Symbol, callexpr::Expr, mod::Module,
                              data::Set{Symbol}, pins::Dict{Symbol,Symbol})
    sm = _resolve_submodel(callexpr, mod)::RKPPLSubmodel
    callargs, pin = _peel_predictor_pin(callexpr, sm)
    length(callargs) == length(sm.argnames) || _sfail(
        "submodel `$(sm.name)` expects $(length(sm.argnames)) argument(s) " *
        "$(Tuple(sm.argnames)), got $(length(callargs)) at `$lhs ~ " *
        "$(sm.name)(...)`")
    stmts, ret = _submodel_body_parts(sm)
    stream = _stream_response(sm, stmts, ret) !== nothing
    if pin !== nothing && !stream
        _sfail("`predictor = $pin` names a response predictor, but " *
               "`$(sm.name)` is a latent submodel (value return) — bind a " *
               "stream submodel to data (`y ~ sm(...; predictor = ...)`)")
    end
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
    if pin !== nothing
        haskey(pins, lhs) && _sfail("response $lhs pins two predictors " *
            "($(pins[lhs]) and $pin) — one `predictor =` per response")
        pins[lhs] = pin
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
    callargs, pin = _peel_predictor_pin(callexpr, sm)
    pin === nothing || _sfail("`predictor = $pin` is top-level-only " *
        "(`y ~ sm(...; predictor = ...)`); per-cell predictors lower " *
        "through the plate path")
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
    dotted = _desugar_fused_head(lhs, rhs)
    call = _dot2call_response(lhs, dotted)
    weights, call = _peel_weighted(lhs, call, ctx)
    evidence, call = _peel_evidence(lhs, call, ctx)
    call.args[1] === :MixtureModel && return _lower_mixture_response(lhs,
        call, range, weights, evidence, ctx, predictors, pred_idx, coefuse)
    if call.args[1] in (:CategoricalLogit, :OrderedLogistic, :Ordinal,
            :Multinomial, :Categorical)
        return _lower_leveled_response(lhs, call, range, weights, evidence,
            ctx, predictors, pred_idx, coefuse)
    end
    family, lik_link, pred_link, loc, scale_raw, trials, nu_raw, zi_raw =
        _lower_response_base(lhs, call, ctx)
    pname = _lower_location(lhs, loc, pred_link, ctx, predictors, pred_idx,
        coefuse)
    scale = _lower_scale_use(lhs, scale_raw, ctx, predictors, pred_idx,
        coefuse)
    nu = _lower_nu_use(lhs, nu_raw, ctx)
    zi = _lower_zi_use(lhs, zi_raw, ctx)
    return LikelihoodSpec(family, lik_link, lhs, pname, scale, weights,
        evidence, Symbol(lhs, "_resp"), trials, range; nu = nu, zi = zi)
end

# Run `thunk()`; on a surface failure, attribute it to mixture component
# `k` (base messages already quote the response + repr — this adds the
# index without doubling the response prefix).
function _mixture_component_context(thunk, lhs, k::Int)
    try
        return thunk()
    catch e
        e isa SurfaceLoweringError || rethrow()
        detail = e.message
        prefix = "response $lhs: "
        startswith(detail, prefix) &&
            (detail = detail[length(prefix)+1:end])
        _sfail("response $lhs: mixture component $k: $detail")
    end
end

# `y .~ MixtureModel.([C1, ..., CK], w)` — K same-family univariate
# components + mixing weights (SB `MixtureModel` mirror). Each component
# lowers through the single-family base spelling (decomposed twin:
# predictors wrapped, params/literals bare); locations route to predictors
# (link-space) or scalar slots (sampled params / literals,
# constrained-scale); weights are a literal vector or a simplex name.
function _lower_mixture_response(lhs, call, range, weights, evidence, ctx,
        predictors, pred_idx, coefuse)
    label = Symbol(lhs, "_resp")
    weights === nothing || _sfail("response $lhs: mixture responses take " *
        "no frequency weights (v1 — `weighted.(...)` over a mixture is a " *
        "follow-up)")
    evidence.kind === :none || _sfail("response $lhs: mixture responses " *
        "take no censoring/truncation evidence (v1)")
    range === nothing || _sfail("response $lhs: mixture responses take no " *
        "range (v1 — mixtures cover the whole column)")
    args = _plain_args(call, "`MixtureModel`")
    length(args) == 2 || _sfail("response $lhs: `MixtureModel` takes " *
        "`MixtureModel.([C1, ..., CK], w)` (a component vector + weights)")
    comps, wraw = args
    comps isa Expr && comps.head === :vect || _sfail("response $lhs: " *
        "`MixtureModel` components ride a vector literal " *
        "(`[Normal.(mu1, s1), Normal.(mu2, s2)]`), got $(repr(comps))")
    K = length(comps.args)
    K >= 1 || _sfail("response $lhs: `MixtureModel` needs ≥ 1 component")
    fams = LikelihoodFamily[]
    llinks = LinkFunction[]
    plinks = LinkFunction[]
    locs_raw = Any[]
    scales_raw = Any[]
    trials_raw = Any[]
    wrappeds = Bool[]
    for (k, c) in enumerate(comps.args)
        c isa Expr && c.head === :. && length(c.args) == 2 &&
            c.args[1] isa Symbol || _sfail("response $lhs: mixture " *
            "component $k is not a distribution call " *
            "(`Normal.(mu, sigma)`), got $(repr(c))")
        lowered = _mixture_component_context(lhs, k) do
            compcall = _mixture_dot2call(lhs, _desugar_fused_head(lhs, c))
            _lower_mixture_component(lhs, compcall, ctx)
        end
        fam, ll, pl, loc, sc, tr, wrapped = lowered
        push!(fams, fam)
        push!(llinks, ll)
        push!(plinks, pl)
        push!(locs_raw, loc)
        push!(scales_raw, sc)
        push!(trials_raw, tr)
        push!(wrappeds, wrapped)
    end
    f = fams[1]
    all(==(f), fams) || _sfail("response $lhs: mixture components must " *
        "share one family (found $(join(unique!(string.(fams)), ", "))); " *
        "heterogeneous mixtures are not supported)")
    f in _MIXTURE_V1_FAMILIES || _sfail("response $lhs: mixture " *
        "components over $f are not admitted in v1 (admitted: Normal, " *
        "Bernoulli-logit, Poisson-log, Binomial-logit, " *
        "NegativeBinomial2-log, Gamma-log, Beta-logit)")
    ll, pl = llinks[1], plinks[1]
    trials = nothing
    if f === BinomialLogitFam
        t1 = trials_raw[1]
        all(t -> isequal(t, t1), trials_raw) || _sfail("response $lhs: " *
            "mixture Binomial components must share one identical " *
            "trial-count expression (SB rule — share one column)")
        trials = t1
    end
    loc_uses = Union{Symbol,Real}[
        _lower_mixture_loc(lhs, k, loc, pl, wrappeds[k], ctx, predictors,
            pred_idx, coefuse) for (k, loc) in enumerate(locs_raw)]
    scale_uses = Union{Nothing,Symbol,Real,ScalePredictorRef}[]
    for (k, sc) in enumerate(scales_raw)
        lowered_sc = _mixture_component_context(lhs, k) do
            _lower_scale_use(lhs, sc, ctx, predictors, pred_idx, coefuse)
        end
        push!(scale_uses, lowered_sc)
    end
    w = _lower_mixture_weights(lhs, wraw)
    prednames = Set{Symbol}(p.name for p in predictors)
    anchor = _mixture_anchor(lhs, loc_uses, scale_uses, w, prednames, ctx)
    return LikelihoodSpec(MixtureFam, ll, lhs, anchor, nothing, weights,
        evidence, label, trials, range; mixture_family = f,
        mixture_locs = loc_uses, mixture_scales = scale_uses,
        mixture_weights = w)
end

# Dotted→call conversion for one mixture component: like
# `_dot2call_response`, but nested link positions tolerate bare means (a
# bare Symbol/Real passes through; dotted links convert exactly as in
# single-family). Bare detection happens downstream in
# `_lower_mixture_component`.
function _mixture_dot2call(lhs, rhs::Expr)
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
    return Expr(:call, f,
        (_mixture_spine_arg(lhs, f, i, a) for (i, a) in enumerate(targs))...)
end

function _mixture_spine_arg(lhs, f, i, a)
    link_pos = ((f === :Bernoulli || f === :Poisson) && i == 1) ||
        (f === :Binomial && i == 2) ||
        (f === :NegativeBinomial2 && i == 1) ||
        (f === :ZeroInflatedPoisson && i == 1)
    link_pos || return a
    a isa Expr && a.head === :. || return a # A bare mean passes through.
    return _dot2call_nested_link(lhs, a, f)
end

# One mixture component: the single-family base spelling, plus bare
# params/literals (constrained-scale, no link inversion). Wrapped
# positions route through `_lower_response_base` (identical behavior +
# messages); bare positions construct directly. Normal (identity link)
# always routes through base with `wrapped` marking the predictor-bound
# shape instead. Returns the base leading tuple plus `wrapped`
# (trailing auxiliary slots are dropped — mixture components carry
# their own).
function _lower_mixture_component(lhs, compcall::Expr, ctx)
    head = compcall.args[1]
    head isa Symbol || _sfail("response $lhs: malformed mixture " *
        "component $(repr(compcall))")
    if head === :Normal
        # Identity link: link-space is constrained-space, so predictors,
        # params, and literals share the position — `wrapped` marks the
        # predictor-bound shape (definition or inline expression), which
        # is what the location strictness actually gates on.
        fam, ll, pl, loc, sc, tr = _lower_response_base(lhs, compcall, ctx)
        w = !(loc isa Real || (loc isa Symbol && loc in ctx.prior_names))
        return fam, ll, pl, loc, sc, tr, w
    elseif head === :Bernoulli
        args = _plain_args(compcall, "`Bernoulli`")
        if length(args) == 1 && (args[1] isa Symbol || args[1] isa Real)
            return BernoulliLogitFam, LogitLink, IdentityLink, args[1],
            nothing, nothing, false
        end
    elseif head === :Poisson
        args = _plain_args(compcall, "`Poisson`")
        if length(args) == 1 && (args[1] isa Symbol || args[1] isa Real)
            return PoissonLogFam, LogLink, LogLink, args[1], nothing,
            nothing, false
        end
    elseif head === :Binomial
        args = _plain_args(compcall, "`Binomial`")
        if length(args) == 2 && (args[2] isa Symbol || args[2] isa Real)
            return BinomialLogitFam, LogitLink, IdentityLink, args[2],
            nothing, _lower_trials(lhs, args[1], ctx), false
        end
    elseif head === :NegativeBinomial2
        args = _plain_args(compcall, "`NegativeBinomial2`")
        if length(args) == 2 && (args[1] isa Symbol || args[1] isa Real)
            return NegativeBinomial2Fam, LogLink, LogLink, args[1], args[2],
            nothing, false
        end
    elseif head === :Gamma
        bare = _match_bare_gamma(lhs, compcall)
        bare !== nothing &&
            return GammaLogFam, LogLink, LogLink, bare[1], bare[2],
            nothing, false
    elseif head === :Beta
        bare = _match_bare_beta(lhs, compcall)
        bare !== nothing &&
            return BetaLogitFam, LogitLink, IdentityLink, bare[1], bare[2],
            nothing, false
    end
    fam, ll, pl, loc, sc, tr = _lower_response_base(lhs, compcall, ctx)
    return fam, ll, pl, loc, sc, tr, true
end

# A bare-mean Gamma shape (`Gamma.(alpha, mean ./ alpha)` with a bare
# `mean`): `(mean, alpha)` or `nothing` (wrapped means and malformed
# shapes fall through to the strict base path).
function _match_bare_gamma(lhs, compcall::Expr)
    args = _plain_args(compcall, "`Gamma`")
    length(args) == 2 || return nothing
    a1, div = args
    div isa Expr && div.head === :call && length(div.args) == 3 &&
        div.args[1] === Symbol("./") || return nothing
    X, a = div.args[2], div.args[3]
    (X isa Symbol || X isa Real) || return nothing
    _same_aux(a1, a) || return nothing
    return X, a1
end

# A bare-mean Beta shape (`Beta.(mu .* k, (1 .- mu) .* k)` with a bare
# `mu`): `(mu, kappa)` or `nothing` (wrapped means and malformed shapes
# fall through to the strict base path).
function _match_bare_beta(lhs, compcall::Expr)
    args = _plain_args(compcall, "`Beta`")
    length(args) == 2 || return nothing
    a1, a2 = args
    a1 isa Expr && a1.head === :call && length(a1.args) == 3 &&
        a1.args[1] === Symbol(".*") || return nothing
    a2 isa Expr && a2.head === :call && length(a2.args) == 3 &&
        a2.args[1] === Symbol(".*") || return nothing
    c = a2.args[2]
    c isa Expr && c.head === :call && length(c.args) == 3 &&
        c.args[1] === Symbol(".-") && c.args[2] == 1 || return nothing
    m1, k1 = a1.args[2], a1.args[3]
    m2, k2 = c.args[3], a2.args[3]
    m1 == m2 || return nothing
    (m1 isa Symbol || m1 isa Real) || return nothing
    _same_aux(k1, k2) || return nothing
    return m1, k1
end

# One mixture location: a vector predictor definition (or inline affine)
# lowers to a predictor (link-space; shared definitions intern by name
# like CategoricalLogit etas); a sampled scalar parameter or numeric
# literal rides scalar (constrained-scale). Link wrappers apply to
# predictors only (a wrapped param/literal fails); bare predictors fail
# (link-space predictors wrap). Scan/latent/matrix/data columns fail
# closed (data wraps in an offset-only predictor first).
function _lower_mixture_loc(lhs, k::Int, loc, pred_link, wrapped::Bool, ctx,
        predictors, pred_idx, coefuse)
    if loc isa Bool
        _sfail("response $lhs: mixture component $k location is Boolean " *
               "— locations are numeric")
    elseif loc isa Real
        wrapped && _sfail("response $lhs: mixture component $k wraps a " *
            "literal in a link function — link wrappers apply to " *
            "predictors (spell literals constrained-scale)")
        return loc
    elseif loc isa Symbol
        loc in ctx.scan_states && _sfail("response $lhs: mixture " *
            "component $k location is a scan state — scan-state mixture " *
            "locations are a follow-up")
        loc in ctx.plate_names && _sfail("response $lhs: mixture " *
            "component $k location is a per-cell latent — latent mixture " *
            "locations are a follow-up")
        haskey(ctx.matrices, loc) && _sfail("response $lhs: mixture " *
            "component $k location $loc is a design matrix — locations " *
            "are predictors, sampled parameters, or literals")
        # A stated-prior alias reads like the name itself (a sampled
        # parameter), so it never routes here — only undeclared
        # intercept-only defs (the SB `mu ~ 1` mirror) qualify.
        if loc in ctx.vecdefs && haskey(ctx.detmap, loc) ||
                _is_scalar_coef_def(loc, ctx, false)
            wrapped || _sfail("response $lhs: mixture component $k " *
                "location $loc is a predictor — link-space predictors " *
                "wrap (`Poisson.(exp.(eta))`); bare slots are sampled " *
                "parameters or literals")
            _derived_reads_latent(loc, ctx) &&
                !_is_design_shaped(loc, ctx) && _sfail("response $lhs: " *
                "mixture component $k location reads a latent — latent " *
                "mixture locations are a follow-up")
            return _lower_location(lhs, loc, pred_link, ctx, predictors,
                pred_idx, coefuse; synth = Symbol(lhs, "_mix_", k, "_eta"))
        end
        haskey(ctx.detmap, loc) && _sfail("response $lhs: mixture " *
            "component $k location $loc is a scalar definition — v1 " *
            "locations are predictors, sampled parameters, or literals")
        if loc in ctx.prior_names
            wrapped && _sfail("response $lhs: mixture component $k " *
                "wraps the sampled parameter $loc in a link function — " *
                "link wrappers apply to predictors (spell sampled " *
                "parameters bare, constrained-scale)")
            return loc
        end
        loc in ctx.data && _sfail("response $lhs: mixture component $k " *
            "location is the data column $loc — wrap it in a predictor " *
            "(`eta = a .+ b .* $loc`, or offset-only `mu = $loc`)")
        _sfail("response $lhs: mixture component $k location $loc is not " *
               "a predictor definition, sampled parameter, or literal")
    else
        wrapped || _sfail("response $lhs: mixture component $k location " *
            "is an inline expression — inline locations lower as " *
            "predictors, so link-space expressions wrap " *
            "(`Poisson.(exp.(eta))`); bare slots are sampled parameters " *
            "or literals")
        # An inline link-space expression: a synthetic predictor, the
        # CategoricalLogit-eta precedent.
        return _lower_location(lhs, loc, pred_link, ctx, predictors,
            pred_idx, coefuse; synth = Symbol(lhs, "_mix_", k, "_eta"))
    end
end

# Mixture weights: a literal numeric vector or a simplex parameter name
# (length/sum/concentration checked at contract — structural, normative).
function _lower_mixture_weights(lhs, wraw)
    wraw isa Symbol && return wraw
    wraw isa Expr && wraw.head === :vect || _sfail("response $lhs: " *
        "mixture weights are a literal vector (`[0.4, 0.6]`) or a " *
        "simplex parameter name, got $(repr(wraw))")
    for (j, e) in enumerate(wraw.args)
        e isa Real && !(e isa Bool) || _sfail("response $lhs: mixture " *
            "weight $j is not a numeric literal (got $(repr(e)))")
    end
    return Float64.(wraw.args)
end

# The mixture anchor (the non-nullable `predictor` slot): first location
# predictor, else first scale predictor, else the weights simplex name,
# else the first location/scale parameter name — the BRM struct order,
# verbatim. Fully-fixed mixtures fail closed before anchoring.
function _mixture_anchor(lhs, loc_uses, scale_uses, w, prednames, ctx)
    for loc in loc_uses
        loc isa Symbol && loc in prednames && return loc
    end
    for s in scale_uses
        s isa ScalePredictorRef && return s.predictor
    end
    w isa Symbol && return w
    for loc in loc_uses
        loc isa Symbol && loc in ctx.prior_names && return loc
    end
    for s in scale_uses
        s isa Symbol && s in ctx.prior_names && return s
    end
    for s in scale_uses
        s isa Symbol && return s # A data-column scale: opaque anchor, never resolved.
    end
    _sfail("response $lhs: fully-fixed mixture (all literals) is a " *
        "constant density with no plan — leave at least one slot free " *
        "or drop the response")
end

const _LEVELED_FAMS =
    (:CategoricalLogit, :OrderedLogistic, :Ordinal, :Multinomial, :Categorical)

function _lower_leveled_response(lhs, call, range, weights, evidence, ctx,
        predictors, pred_idx, coefuse)
    fam = call.args[1]
    label = Symbol(lhs, "_resp")
    if fam === :CategoricalLogit
        return _lower_categorical_logit_response(lhs, call, range, weights,
            evidence, label, ctx, predictors, pred_idx, coefuse)
    elseif fam === :OrderedLogistic
        return _lower_ordered_logistic_response(lhs, call, range, weights,
            evidence, label, ctx, predictors, pred_idx, coefuse)
    elseif fam === :Ordinal
        return _lower_ordinal_response(lhs, call, range, weights, evidence,
            label, ctx, predictors, pred_idx, coefuse)
    elseif fam === :Multinomial
        return _lower_multinomial_response(lhs, call, range, weights,
            evidence, label, ctx)
    else
        return _lower_categorical_response(lhs, call, range, weights,
            evidence, label, ctx)
    end
end

# Reference-coded multi-logit categorical:
# `y .~ CategoricalLogit.(eta_2, ..., eta_K)` — K−1 linear-predictor
# etas (class 1 is the implicit zero reference). Each eta lowers as an
# ordinary location (named definitions intern by name; inline etas
# synthesize indexed predictors).
function _lower_categorical_logit_response(lhs, call, range, weights,
        evidence, label, ctx, predictors, pred_idx, coefuse)
    args = _plain_args(call, "`CategoricalLogit`")
    isempty(args) && _sfail("response $lhs: `CategoricalLogit` takes the " *
                            "K−1 non-reference etas " *
                            "(`y .~ CategoricalLogit.(eta_2, eta_3)` for K=3)")
    pnames = Symbol[_lower_location(lhs, a, IdentityLink, ctx, predictors,
        pred_idx, coefuse; synth = Symbol(lhs, "_eta_", j))
        for (j, a) in enumerate(args)]
    return LikelihoodSpec(CategoricalLogitFam, LogitLink, lhs, pnames[1],
        nothing, weights, evidence, label, nothing, range;
        extra_predictors = pnames[2:end])
end

# Allocate an implicit SB-mirroring vector parameter (`y_cutpoints` /
# `y_thresholds`): std-normal elementwise prior, size inferred at bind.
# Loud on collision with a user definition.
function _implicit_vector!(ctx, name::Symbol, family::Symbol, lhs::Symbol)
    name in ctx.taken && _sfail(
        "implicit $family parameter $name for response $lhs collides " *
        "with your definition — rename yours")
    push!(ctx.taken, name)
    push!(ctx.implicit_vectors,
        VectorParameter(name, family, (arg1 = 0.0, arg2 = 1.0), nothing, name))
    return name
end

# Cumulative-logit ordinal: `y .~ OrderedLogistic.(eta)` + implicit
# ordered cutpoints (SB's `y_cutpoints::ordered[K-1] ~ std_normal()`).
function _lower_ordered_logistic_response(lhs, call, range, weights,
        evidence, label, ctx, predictors, pred_idx, coefuse)
    args = _plain_args(call, "`OrderedLogistic`")
    length(args) == 1 || _sfail("response $lhs: `OrderedLogistic` takes " *
                                "`y .~ OrderedLogistic.(eta)`")
    pname = _lower_location(lhs, args[1], IdentityLink, ctx, predictors,
        pred_idx, coefuse)
    cut = _implicit_vector!(ctx, Symbol(lhs, :_cutpoints), :ordered_normal, lhs)
    return LikelihoodSpec(OrderedLogisticFam, LogitLink, lhs, pname,
        nothing, weights, evidence, label, nothing, range; thresholds = cut)
end

# Ordinal tags (SB's typed composition, mirrored): structure
# `Cumulative()`/`StoppingRatio()`, link
# `LogitLink()`/`ProbitLink()`/`CloglogLink()` — nullary calls.
function _ordinal_tag(lhs, arg, kinds::Tuple{Vararg{Symbol}}, what::String)
    arg isa Expr && arg.head === :call && length(arg.args) == 1 &&
        arg.args[1] isa Symbol && arg.args[1] in kinds ||
        _sfail("response $lhs: ordinal $what is " *
               join(("`$k()`" for k in kinds), "/") * ", got $(repr(arg))")
    return arg.args[1]
end

const _ORDINAL_LINKS = Dict{Symbol,LinkFunction}(
    :LogitLink => LogitLink,
    :ProbitLink => ProbitLink,
    :CloglogLink => CloglogLink,
)

# General typed ordinal: `y .~ Ordinal.(Cumulative(), LogitLink(), eta)`
# (+ implicit thresholds — ordered iff cumulative). Discrimination and
# per-threshold design are plan-level only (the BRM emitter's path):
# the dotted object takes exactly three positionals.
function _lower_ordinal_response(lhs, call, range, weights, evidence,
        label, ctx, predictors, pred_idx, coefuse)
    args = _plain_args(call, "`Ordinal`")
    length(args) == 3 || _sfail("response $lhs: `Ordinal` takes " *
                                "`y .~ Ordinal.(Cumulative(), LogitLink(), eta)` " *
                                "(structure, link, eta — discrimination and " *
                                "per-threshold design are plan-level only)")
    structure = _ordinal_tag(lhs, args[1], (:Cumulative, :StoppingRatio),
        "structure")
    linktag = _ordinal_tag(lhs, args[2],
        (:LogitLink, :ProbitLink, :CloglogLink), "link")
    pname = _lower_location(lhs, args[3], IdentityLink, ctx, predictors,
        pred_idx, coefuse)
    vfam = structure === :Cumulative ? :ordered_normal : :vector_normal
    thresh = _implicit_vector!(ctx, Symbol(lhs, :_thresholds), vfam, lhs)
    structure_sym = structure === :Cumulative ? :cumulative : :stopping
    return LikelihoodSpec(OrdinalFam, _ORDINAL_LINKS[linktag], lhs, pname,
        nothing, weights, evidence, label, nothing, range;
        thresholds = thresh, ordinal_structure = structure_sym)
end

# Shared-simplex multinomial: `c1 .~ Multinomial.(N, s, c2, ..., cK)` —
# the lead count column (LHS) plus the K−1 tail count columns, trials N
# (Int literal or column), and the simplex parameter `s`
# (`s ~ Dirichlet(...)` elsewhere in the model).
function _lower_multinomial_response(lhs, call, range, weights, evidence,
        label, ctx)
    args = _plain_args(call, "`Multinomial`")
    length(args) >= 2 || _sfail("response $lhs: `Multinomial` takes " *
                                "`c1 .~ Multinomial.(N, s, c2, ..., cK)` " *
                                "(trials, simplex, tail count columns)")
    trials = _lower_trials(lhs, args[1], ctx)
    s = args[2]
    s isa Symbol || _sfail("response $lhs: multinomial probs $s must be " *
                           "a simplex parameter name " *
                           "(`s ~ Dirichlet(...)` in the model)")
    haskey(ctx.matrices, s) && _sfail("response $lhs: multinomial probs " *
                                      "$s is a design matrix — probs are a " *
                                      "simplex parameter " *
                                      "(`s ~ Dirichlet(...)` in the model)")
    tail = args[3:end]
    for c in tail
        c isa Symbol || _sfail("response $lhs: multinomial tail column " *
                               "$(repr(c)) must be a data column name")
    end
    return LikelihoodSpec(MultinomialFam, IdentityLink, lhs, s,
        nothing, weights, evidence, label, trials, range;
        count_columns = Vector{Symbol}(tail))
end

# Plain categorical over simplex probabilities: `y .~ Categorical.(s)`.
function _lower_categorical_response(lhs, call, range, weights, evidence,
        label, ctx)
    args = _plain_args(call, "`Categorical`")
    length(args) == 1 || _sfail("response $lhs: `Categorical` takes " *
                                "`y .~ Categorical.(s)` (a simplex parameter)")
    s = only(args)
    s isa Symbol || _sfail("response $lhs: categorical probs $s must be " *
                           "a simplex parameter name " *
                           "(`s ~ Dirichlet(...)` in the model)")
    haskey(ctx.matrices, s) && _sfail("response $lhs: categorical probs " *
                                      "$s is a design matrix — probs are a " *
                                      "simplex parameter " *
                                      "(`s ~ Dirichlet(...)` in the model)")
    return LikelihoodSpec(CategoricalFam, IdentityLink, lhs, s,
        nothing, weights, evidence, label, nothing, range)
end

# Joint correlated-outcomes response:
# `[y1, y2] ~ MvNormalCholesky([mu1, mu2], L)`. Each mean lowers as an
# ordinary identity-link location (own predictor per outcome, named
# definitions interned by name, inline means synthesized per outcome);
# the factor stem resolves to its two `LKJCovarianceFactor` pieces.
function _lower_joint_response(j::JointSampleStmt, factor_names::Set{Symbol},
        ctx, predictors, pred_idx, coefuse)
    tag = "[$(join(j.outcomes, ", "))]"
    j.factor in factor_names || _sfail(
        "joint response $tag factor $(j.factor) must name an " *
        "`LKJCovarianceFactor` declaration " *
        "(`L ~ LKJCovarianceFactor(K, Exponential(1.0), eta)` in the model)")
    pnames = Symbol[_lower_location(o, m, IdentityLink, ctx, predictors,
        pred_idx, coefuse; synth = Symbol(o, "_joint_", k))
        for (k, (o, m)) in enumerate(zip(j.outcomes, j.means))]
    scales, corr = _lkj_factor_names(j.factor)
    label = Symbol(join(j.outcomes, "_") * "_resp")
    return LikelihoodSpec(MvNormalCholeskyFam, IdentityLink, j.outcomes[1],
        pnames[1], nothing, nothing, ResponseEvidence(:none, nothing, nothing),
        label, nothing, nothing; extra_responses = j.outcomes[2:end],
        extra_predictors = pnames[2:end], factor_scales = scales,
        factor_corr = corr)
end

function _lower_glm_response(g::GLMSampleStmt, sample, prior_names::Set{Symbol},
        ctx, coefuse, glmuse::Dict{Symbol,Tuple{Symbol,Symbol}})
    fam, link = g.head === :NormalIDGLM ? (NormalIDGLMFam, IdentityLink) :
        g.head === :BernoulliLogitGLM ? (BernoulliLogitGLMFam, LogitLink) :
        (PoissonLogGLMFam, LogLink)
    m = get(ctx.matrices, g.matrix, nothing)
    m === nothing && _sfail("response $(g.response): `$(g.head)` " *
                            "design matrix $(g.matrix) is not a bound " *
                            "design matrix (`$(g.matrix) = hcat(...)`)")
    any(c -> c === nothing, m.columns) && _sfail(
        "response $(g.response): `$(g.head)` design matrix $(g.matrix) " *
        "has an intercept-ones position — pass intercept-free X and a " *
        "separate alpha")
    for (nm, role) in ((g.alpha, "intercept"), (g.beta, "coefficients"))
        haskey(coefuse, nm) && _sfail(
            "response $(g.response): `$(g.head)` $role $nm is also a " *
            "predictor coefficient — one use per name")
    end
    g.alpha in prior_names || _sfail(
        "response $(g.response): `$(g.head)` intercept $(g.alpha) " *
        "needs a prior statement (`$(g.alpha) ~ Normal(0, 10)`)")
    for s in sample
        s.lhs === g.beta || continue
        if s.matrix === nothing
            _sfail("response $(g.response): `$(g.head)` coefficient " *
                   "vector $(g.beta) needs a broadcast prior " *
                   "(`$(g.beta)[axes($(g.matrix), 2)] .~ Normal.(...)`)")
        end
        s.matrix === g.matrix || _sfail(
            "response $(g.response): `$(g.head)` coefficient prior " *
            "sizes $(s.matrix), not the response matrix $(g.matrix)")
    end
    sigma = g.sigma
    if sigma isa Symbol
        sigma in prior_names || _sfail(
            "response $(g.response): `$(g.head)` sigma $sigma needs " *
            "a prior statement or a positive literal")
        haskey(coefuse, sigma) && _sfail(
            "response $(g.response): `$(g.head)` sigma $sigma is also " *
            "a predictor coefficient — one use per name")
    end
    label = Symbol(g.response, "_resp")
    push!(ctx.matrices_used, g.matrix)
    glmuse[g.beta] = (label, g.matrix)
    return LikelihoodSpec(fam, link, g.response, g.matrix, sigma, nothing,
        ResponseEvidence(:none, nothing, nothing), label, nothing, nothing;
        glm_alpha = g.alpha, glm_beta = g.beta)
end

# A GLM coefficient vector: K per-element PopulationPriors over the
# response matrix columns, addressed by response label. Unstated
# vectors default to K× Normal(0, 1) (the matrix-prior emitter
# convention — the width is static, so no declaration is needed to
# size them). Stated vectors take `b[axes(X, 2)] .~ Normal.(loc,
# scale)` with scalar (shared) or length-K literal-vector
# (per-element) args.
function _lower_glm_beta_priors(label::Symbol, beta::Symbol, X::Symbol,
        sample, matrices::Dict{Symbol,DesignMatrix})
    m = get(matrices, X, nothing)
    m === nothing && _sfail("internal: GLM prior over unknown matrix $X")
    cols = Symbol[c for c in m.columns if c !== nothing]
    K = length(cols)
    stated = nothing
    for s in sample
        s.lhs === beta || continue
        stated = s
    end
    stated === nothing && return PopulationPrior[
        PopulationPrior(label, c, 0.0, 1.0) for c in cols]
    locs, scales = _coefficient_matrix_normal(beta, stated.rhs, label, K)
    return PopulationPrior[PopulationPrior(label, c, l, sc)
        for (c, l, sc) in zip(cols, locs, scales)]
end

# `.~` takes a dotted distribution object (`Normal.(mu, sigma)`); convert
# the distribution spine to call form and reuse the peeling machinery
# (which reports the same object-form errors, now against dotted input).
# Only spine positions convert (nested objects, links); locations, scales,
# bounds, and weights pass through untouched (inline predictor dots are
# the analysis's business, not the peeler's).
const _DOT_WRAPPERS = (:weighted, :truncated, :censored, :interval_censored)

_dotted_obj(f::Symbol, args...) = Expr(:., f, Expr(:tuple, args...))

# Fused family-name response heads (`BernoulliLogit.(eta)`,
# `PoissonLog.(eta)`, `BinomialLogit.(n, mu)`,
# `NegativeBinomial2Log.(eta, phi)`, `GammaLog.(alpha, eta)`,
# `BetaLogit.(mu, kappa)`): rewrite to the decomposed spelling BEFORE
# spine conversion, so HAVE recovery is shared by construction — a fused
# response lowers to the identical plan as its decomposed twin. Arity is
# validated with fused-spelled errors first (a wrong-arity fused head
# never reports a confusing decomposed message); keyword arguments fall
# through untouched to the canonical positional-only error. Malformed
# shapes likewise pass through to the canonical shape errors. Recurses
# into the object position of the `weighted.`/`truncated.`/`censored.`/
# `interval_censored.` wrappers (the first tuple arg in every wrapper).
function _desugar_fused_head(lhs, rhs)
    rhs isa Expr && rhs.head === :. || return rhs
    length(rhs.args) == 2 && rhs.args[1] isa Symbol &&
        rhs.args[2] isa Expr && rhs.args[2].head === :tuple || return rhs
    f = rhs.args[1]
    targs = rhs.args[2].args
    if f in _DOT_WRAPPERS
        isempty(targs) && return rhs
        newfirst = _desugar_fused_head(lhs, targs[1])
        newfirst === targs[1] && return rhs
        return Expr(:., f, Expr(:tuple, newfirst, targs[2:end]...))
    end
    newrhs = _desugar_fused_base(lhs, f, targs)
    newrhs === nothing && return rhs
    return newrhs
end

function _desugar_fused_base(lhs, f, targs)
    f in (:BernoulliLogit, :PoissonLog, :BinomialLogit,
        :NegativeBinomial2Log, :GammaLog, :BetaLogit) || return nothing
    any(a -> a isa Expr && a.head === :parameters, targs) && return nothing
    if f === :BernoulliLogit
        length(targs) == 1 ||
            _sfail("response $lhs: `BernoulliLogit` takes " *
                   "`BernoulliLogit.(eta)`")
        return _dotted_obj(:Bernoulli, _dotted_obj(:logistic, targs[1]))
    elseif f === :PoissonLog
        length(targs) == 1 ||
            _sfail("response $lhs: `PoissonLog` takes `PoissonLog.(eta)`")
        return _dotted_obj(:Poisson, _dotted_obj(:exp, targs[1]))
    elseif f === :BinomialLogit
        length(targs) == 2 ||
            _sfail("response $lhs: `BinomialLogit` takes " *
                   "`BinomialLogit.(n, mu)`")
        return _dotted_obj(:Binomial, targs[1],
            _dotted_obj(:logistic, targs[2]))
    elseif f === :NegativeBinomial2Log
        length(targs) == 2 ||
            _sfail("response $lhs: `NegativeBinomial2Log` takes " *
                   "`NegativeBinomial2Log.(eta, phi)`")
        return _dotted_obj(:NegativeBinomial2, _dotted_obj(:exp, targs[1]),
            targs[2])
    elseif f === :GammaLog
        length(targs) == 2 ||
            _sfail("response $lhs: `GammaLog` takes `GammaLog.(alpha, eta)`")
        return _dotted_obj(:Gamma, targs[1],
            Expr(:call, Symbol("./"), _dotted_obj(:exp, targs[2]), targs[1]))
    else
        length(targs) == 2 ||
            _sfail("response $lhs: `BetaLogit` takes `BetaLogit.(mu, kappa)`")
        m1 = _dotted_obj(:logistic, targs[1])
        m2 = _dotted_obj(:logistic, targs[1])
        return _dotted_obj(:Beta,
            Expr(:call, Symbol(".*"), m1, targs[2]),
            Expr(:call, Symbol(".*"),
                Expr(:call, Symbol(".-"), 1, m2), targs[2]))
    end
end

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
                :Gamma, :Beta, :ZeroInflatedPoisson, :BernoulliLogit,
                :PoissonLog, :BinomialLogit,
                :NegativeBinomial2Log, :GammaLog, :BetaLogit,
                :CategoricalLogit, :OrderedLogistic, :Ordinal,
                :Multinomial, :Categorical, :weighted, :truncated, :censored,
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
    elseif f === :HurdlePoisson && i == 1
        return _dot2call_nested_link(lhs, a, f)
    elseif f === :ZeroInflatedPoisson && i == 1
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
    want = (base === :Bernoulli || base === :Binomial) ? "logistic/probit/cloglog" :
        base === :Beta ? "logistic" : "exp"
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
    "`StudentT.(nu, mu, sigma)`, " *
    "`Bernoulli.(logistic.(eta))` (or `probit`/`cloglog` for the link), " *
    "`Poisson.(exp.(eta))`, `Binomial.(n, logistic.(mu))` (or " *
    "`probit`/`cloglog` for the link), " *
    "`NegativeBinomial2.(exp.(eta), phi)`, " *
    "`HurdlePoisson.(exp.(eta), p_zero)`, " *
    "`ZeroInflatedPoisson.(exp.(eta), zi)`, " *
    "`Gamma.(alpha, exp.(eta) ./ alpha)`, " *
    "`Beta.(logistic.(mu) .* kappa, (1 .- logistic.(mu)) .* kappa)`, " *
    "`CategoricalLogit.(eta_2, ..., eta_K)`, `OrderedLogistic.(eta)`, " *
    "`Ordinal.(Cumulative(), LogitLink(), eta)`, " *
    "`Multinomial.(N, s, c2, ..., cK)`, or `Categorical.(s)` " *
    "(or the fused heads `BernoulliLogit.(eta)`, `PoissonLog.(eta)`, " *
    "`BinomialLogit.(n, mu)`, `NegativeBinomial2Log.(eta, phi)`, " *
    "`GammaLog.(alpha, eta)`, `BetaLogit.(mu, kappa)`, which lower " *
    "identically to their decomposed spellings)"

function _lower_response_base(lhs, rhs::Expr, ctx)
    rhs.head === :call || _sfail("response $lhs: $_RESPONSE_BASE_MSG; " *
                                 "got $(repr(rhs))")
    fam = rhs.args[1]
    fam === :weighted &&
        _sfail("`weighted.(...)` goes outermost: " *
               "`y .~ weighted.(Normal.(mu, sigma), w)`")
    fam in (:Normal, :StudentT, :Bernoulli, :Poisson, :Binomial,
        :NegativeBinomial2, :Gamma, :Beta, :HurdlePoisson,
        :ZeroInflatedPoisson) ||
        return _lower_response_base_error(lhs, rhs, fam)
    args = _plain_args(rhs, "`$fam`")
    if fam === :Normal
        length(args) == 2 || _sfail("response $lhs: `Normal` takes " *
                                    "`Normal.(mu, sigma)`")
        return GaussianFam, IdentityLink, IdentityLink, args[1], args[2],
        nothing, nothing, nothing
    elseif fam === :StudentT
        length(args) == 3 || _sfail("response $lhs: `StudentT` takes " *
                                    "`StudentT.(nu, mu, sigma)`")
        return StudentTFam, IdentityLink, IdentityLink, args[2], args[3],
        nothing, args[1], nothing
    elseif fam === :Bernoulli
        length(args) == 1 || _sfail("response $lhs: `Bernoulli` takes " *
                                    "`Bernoulli.(logistic.(eta))` (or `probit`/`cloglog` for the link)")
        f, l, loc = _lower_bernoulli_link(lhs, args[1])
        return f, l, IdentityLink, loc, nothing, nothing, nothing, nothing
    elseif fam === :Binomial
        length(args) == 2 || _sfail("response $lhs: `Binomial` takes " *
                                    "`Binomial.(n, logistic.(mu))` (or `probit`/`cloglog` for the link)")
        f, l, loc = _lower_binomial_link(lhs, args[2])
        return f, l, IdentityLink, loc, nothing,
        _lower_trials(lhs, args[1], ctx), nothing, nothing
    elseif fam === :NegativeBinomial2
        length(args) == 2 || _sfail("response $lhs: `NegativeBinomial2` takes " *
                                    "`NegativeBinomial2.(exp.(eta), phi)`")
        return NegativeBinomial2Fam, LogLink, LogLink,
        _lower_link_arg(lhs, args[1], :exp), args[2], nothing, nothing,
        nothing
    elseif fam === :Gamma
        loc, scale = _lower_gamma_args(lhs, args, ctx)
        return GammaLogFam, LogLink, LogLink, loc, scale, nothing, nothing,
        nothing
    elseif fam === :Beta
        loc, scale = _lower_beta_args(lhs, args, ctx)
        return BetaLogitFam, LogitLink, IdentityLink, loc, scale, nothing,
        nothing, nothing
    elseif fam === :HurdlePoisson
        length(args) == 2 || _sfail("response $lhs: `HurdlePoisson` takes " *
                                    "`HurdlePoisson.(exp.(eta), p_zero)`")
        return HurdlePoissonFam, LogLink, LogLink,
        _lower_link_arg(lhs, args[1], :exp), args[2], nothing, nothing,
        nothing
    elseif fam === :ZeroInflatedPoisson
        length(args) == 2 || _sfail("response $lhs: `ZeroInflatedPoisson` takes " *
                                    "`ZeroInflatedPoisson.(exp.(eta), zi)`")
        return ZeroInflatedPoissonFam, LogLink, LogLink,
        _lower_link_arg(lhs, args[1], :exp), nothing, nothing, nothing,
        args[2]
    else
        length(args) == 1 || _sfail("response $lhs: `Poisson` takes " *
                                    "`Poisson.(exp.(eta))`")
        return PoissonLogFam, LogLink, LogLink,
        _lower_link_arg(lhs, args[1], :exp), nothing, nothing, nothing,
        nothing
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
    _same_aux(a1, a2) || _sfail(
        "response $lhs: both `Gamma` positions must name the same alpha " *
        "(got $(repr(a1)) and $(repr(a2)))")
    return loc, a1
end

# Both auxiliary positions name the same use: bare names, equal literals,
# or structurally equal scale-predictor spellings (bare or one
# `exp.`/`logistic.` wrapper over the same predictor). Anything else —
# mixed wrappers, distinct predictors — is a mismatch, never a merge.
function _same_aux(a, b)
    ka = _aux_key(a)
    kb = _aux_key(b)
    return ka !== nothing && ka == kb
end

function _aux_key(a)
    a isa Symbol && return (:bare, a)
    a isa Real && return (:lit, a)
    if a isa Expr && a.head === :. && length(a.args) == 2 &&
            a.args[1] isa Symbol && a.args[1] in (:exp, :logistic) &&
            a.args[2] isa Expr && a.args[2].head === :tuple &&
            length(a.args[2].args) == 1 && a.args[2].args[1] isa Symbol
        return (:wrap, a.args[1], a.args[2].args[1])
    end
    return nothing
end

const _BETA_MSG = "`Beta.(logistic.(mu) .* kappa, (1 .- logistic.(mu)) .* kappa)`"

# Beta mean-concentration shape: both positions share the SAME mu expression
# (structurally) and the SAME kappa (name or literal, Gamma-precedent check);
# mu link is logistic only in slice 2. Arguments arrive in `:call` form
# (dotted operators parse as calls); the nested `logistic.(mu)` stays dotted
# and converts explicitly, mirroring `_lower_gamma_args`.
function _lower_beta_args(lhs, args, ctx)
    length(args) == 2 ||
        _sfail("response $lhs: `Beta` takes $_BETA_MSG")
    a1, a2 = args
    a1 isa Expr && a1.head === :call && length(a1.args) == 3 &&
        a1.args[1] === Symbol(".*") ||
        _sfail("response $lhs: `Beta` first position is `mu .* kappa` " *
               "(`$_BETA_MSG`), got $(repr(a1))")
    a2 isa Expr && a2.head === :call && length(a2.args) == 3 &&
        a2.args[1] === Symbol(".*") ||
        _sfail("response $lhs: `Beta` second position is " *
               "`(1 .- mu) .* kappa` (`$_BETA_MSG`), got $(repr(a2))")
    c = a2.args[2]
    c isa Expr && c.head === :call && length(c.args) == 3 &&
        c.args[1] === Symbol(".-") && c.args[2] == 1 ||
        _sfail("response $lhs: `Beta` second position is " *
               "`(1 .- mu) .* kappa` (`$_BETA_MSG`), got $(repr(a2))")
    m1, k1 = a1.args[2], a1.args[3]
    m2, k2 = c.args[3], a2.args[3]
    m1 == m2 || _sfail("response $lhs: both `Beta` positions must share " *
                       "the same mu expression " *
                       "(got $(repr(m1)) and $(repr(m2)))")
    _same_aux(k1, k2) || _sfail(
        "response $lhs: both `Beta` positions must name the same kappa " *
        "(got $(repr(k1)) and $(repr(k2)))")
    loc = _lower_link_arg(lhs, _dot2call_nested_link(lhs, m1, :Beta), :logistic)
    return loc, k1
end

function _lower_response_base_error(lhs, rhs, fam)
    fam in (:normal, :bernoulli, :poisson, :binomial, :gamma, :beta) && _sfail(
        "response $lhs: use Distributions.jl constructors " *
        "(`Normal`, not `normal`)")
    fam === :negative_binomial2 && _sfail("response $lhs: use " *
                                          "`NegativeBinomial2` (the response " *
                                          "spelling, not the kernel endpoint)")
    fam === :student_t && _sfail("response $lhs: use " *
                                 "`StudentT` (the response " *
                                 "spelling, not the kernel endpoint)")
    fam === :hurdle_poisson && _sfail("response $lhs: use " *
                                      "`HurdlePoisson` (the response " *
                                      "spelling, not the kernel endpoint)")
    fam === :zero_inflated_poisson && _sfail("response $lhs: use " *
                                 "`ZeroInflatedPoisson` (the response " *
                                 "spelling, not the kernel endpoint)")
    fam === :OrderedLogit && _sfail("response $lhs: unknown distribution " *
                                    "`:OrderedLogit` (write " *
                                    "`OrderedLogistic.(eta)`)")
    fam === :MvNormalCholesky && _sfail(
        "response $lhs: `MvNormalCholesky` is joint-only " *
        "(`[y1, y2] ~ MvNormalCholesky([mu1, mu2], L)` with plain `~` — " *
        "row-grouped, never broadcast)")
    return _sfail("response $lhs: unknown distribution `$(repr(fam))` " *
                  "(admitted: Normal, StudentT, Bernoulli, Poisson, Binomial, " *
                  "NegativeBinomial2, HurdlePoisson, Gamma, Beta, " *
                  "ZeroInflatedPoisson, BernoulliLogit, " *
                  "PoissonLog, BinomialLogit, NegativeBinomial2Log, " *
                  "GammaLog, BetaLogit, CategoricalLogit, " *
                  "OrderedLogistic, Ordinal, Multinomial, Categorical, " *
                  "MixtureModel). " *
                  "When `$fam` is a defined RKPPLSubmodel, a latent uses " *
                  "`latent ~ $fam(...)` and an observation stream uses plain " *
                  "`$lhs ~ $fam(...)` (the whole-column vectorized callee); " *
                  "an elementwise per-row submodel broadcast " *
                  "`$lhs .~ $fam.(...)` is a follow-up slice.")
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

# Bernoulli/Binomial link dispatch (logit + slice-2 probit/cloglog). The
# wrapper arrives converted to `:call` by `_dot2call_nested_link`; unknown
# wrappers fail here (the spine converter is wrapper-generic).
const _BERNOULLI_LINKS = Dict{Symbol,Tuple{LikelihoodFamily,LinkFunction}}(
    :logistic => (BernoulliLogitFam, LogitLink),
    :probit => (BernoulliProbitFam, ProbitLink),
    :cloglog => (BernoulliCloglogFam, CloglogLink),
)
const _BINOMIAL_LINKS = Dict{Symbol,Tuple{LikelihoodFamily,LinkFunction}}(
    :logistic => (BinomialLogitFam, LogitLink),
    :probit => (BinomialProbitFam, ProbitLink),
    :cloglog => (BinomialCloglogFam, CloglogLink),
)

function _lower_bernoulli_link(lhs, arg)
    arg isa Expr && arg.head === :call && !isempty(arg.args) &&
        haskey(_BERNOULLI_LINKS, arg.args[1]) ||
        _sfail("response $lhs: `Bernoulli` takes a link wrapper " *
               "(`logistic.(eta)`, `probit.(eta)`, or `cloglog.(eta)`), " *
               "got $(repr(arg))")
    fam, link = _BERNOULLI_LINKS[arg.args[1]]
    return fam, link, _lower_link_arg(lhs, arg, arg.args[1])
end

function _lower_binomial_link(lhs, arg)
    arg isa Expr && arg.head === :call && !isempty(arg.args) &&
        haskey(_BINOMIAL_LINKS, arg.args[1]) ||
        _sfail("response $lhs: `Binomial` probability takes a link wrapper " *
               "(`logistic.(mu)`, `probit.(mu)`, or `cloglog.(mu)`), " *
               "got $(repr(arg))")
    fam, link = _BINOMIAL_LINKS[arg.args[1]]
    return fam, link, _lower_link_arg(lhs, arg, arg.args[1])
end

function _lower_scale(lhs, s, ctx)
    s isa Real && return s
    s === :Inf && return Inf
    if s isa Symbol
        haskey(ctx.matrices, s) && _sfail(
            "response $lhs scale $s is a design matrix — scales are " *
            "scalar (a parameter/assignment name, a per-observation data " *
            "column, or a literal)")
        # A per-observation scale is a RAW data column (the eight-schools known
        # SE `se[i]`): it threads through the response plate per cell exactly
        # like a per-obs weight column (the generator's `_thread_ref!`
        # broadcasts a scalar param and iterates a per-obs column). A DERIVED
        # column scale still needs shape metadata the plate cannot yet size,
        # so keep it rejected with an actionable message.
        s in ctx.vecdefs && _sfail(
            "response $lhs scale $s is a derived column — a per-observation " *
            "scale must be a raw data column (bind it raw) or a scalar " *
            "parameter/assignment name (planned: derived-column scales)")
        return s
    end
    return _sfail("response $lhs scale must be a bare parameter/assignment " *
                  "name, a per-observation data column, or a literal (bind " *
                  "expressions via an assignment first), got $(repr(s))")
end

# Student nu use-site lowering: a bare parameter/assignment name or a
# literal, scalar only — no per-observation columns (a column name fails
# at the contract's unknown-name gate), no predictor-fed nu (a modeled
# nu is deferred), no expressions (bind via an assignment first).
function _lower_nu_use(lhs, s, ctx)
    s === nothing && return nothing
    s isa Real && return s
    if s isa Symbol
        # Bare use: stated-prior aliases are scalar nu, like a bare
        # scale — only undeclared/vector/factor defs count as
        # predictor-fed here.
        _is_scale_predictor_def(s, ctx, false) && _sfail(
            "response $lhs nu $s is a predictor definition — " *
            "predictor-fed nu (a modeled df) is deferred: use a scalar " *
            "nu (parameter or literal)")
        return s
    end
    return _sfail("response $lhs nu must be a bare parameter/assignment " *
                  "name or a literal (bind expressions via an assignment " *
                  "first), got $(repr(s))")
end

# ZIP zi use-site lowering: a bare parameter/assignment name or a
# literal, scalar only — no per-observation columns (a column name fails
# at the contract's unknown-name gate), no predictor-fed zi (a modeled
# zi submodel is deferred), no expressions (bind via an assignment
# first).
function _lower_zi_use(lhs, s, ctx)
    s === nothing && return nothing
    s isa Real && return s
    if s isa Symbol
        # Bare use: stated-prior aliases are scalar zi, like a bare
        # scale — only undeclared/vector/factor defs count as
        # predictor-fed here.
        _is_scale_predictor_def(s, ctx, false) && _sfail(
            "response $lhs zi $s is a predictor definition — " *
            "predictor-fed zi (a modeled zi submodel) is deferred: use " *
            "a scalar zi (parameter or literal)")
        return s
    end
    return _sfail("response $lhs zi must be a bare parameter/assignment " *
                  "name or a literal (bind expressions via an assignment " *
                  "first), got $(repr(s))")
end

# Scale use-site lowering (Gaussian sigma, NB2 phi, Gamma alpha, Beta
# kappa, Student sigma, hurdle p_zero): a scalar scale
# (parameter/assignment name, raw per-observation data column, literal)
# passes through `_lower_scale` untouched; a
# predictor definition feeds the scale slot — bare for an identity-link
# scale (`Normal.(mu, sigma)`), or under one dotted link wrapper
# (`Normal.(mu, exp.(sigma))` for log, `logistic.(sigma)` for logit).
# The wrapper arrives unconverted (the scale position passes the spine
# converter through), so it matches here in dotted `Expr(:., ...)` form.
# Undotted wrappers fail closed (scalar `exp(log_sigma)` use-site
# wrappers are deferred — the LP link spells the transform instead), as
# do wrappers over anything but a predictor definition.
function _lower_scale_use(lhs, s, ctx, predictors, pred_idx, coefuse)
    # Scaleless families (Bernoulli/Poisson/Binomial) carry `nothing`
    # through untouched.
    s === nothing && return nothing
    if s isa Expr && s.head === :.
        return _lower_scale_wrapped(lhs, s, ctx, predictors, pred_idx, coefuse)
    end
    if s isa Expr && s.head === :call && !isempty(s.args) &&
            s.args[1] isa Symbol && s.args[1] in (:exp, :logistic)
        return _sfail("response $lhs scale wraps `$(s.args[1])` undotted " *
                      "(`$(repr(s))`) — scale link wrappers broadcast " *
                      "(`$(s.args[1]).(predictor)` over a predictor " *
                      "definition); scalar `exp(log_sigma)` use-site " *
                      "wrappers are deferred (spell the transform as the " *
                      "predictor's link instead)")
    end
    # A bare scale keeps the scalar meaning: an alias over a stated prior
    # lowers like the name itself (a parameter), never as a predictor.
    if s isa Symbol && _is_scale_predictor_def(s, ctx, false)
        pname = _lower_scale_predictor(lhs, s, IdentityLink, ctx, predictors,
            pred_idx, coefuse)
        return ScalePredictorRef(pname, IdentityLink)
    end
    return _lower_scale(lhs, s, ctx)
end

# A bare scale name feeds the predictor slot when it is a per-observation
# definition that is not latent-backed: vector-shaped definitions plus
# bare `coefficients[group]` factor-index refs (ref-shaped, hence
# scalar-shaped in `detshape`, but per-observation at runtime — the
# factor-term predictor spelling) plus scalar `name = coef` intercept-
# only definitions (the SB `log(sigma) ~ 1` mirror — the design carries
# the `ones(n)` intercept block, so the LP still evaluates per cell).
# Per-cell latents (plate names and derived columns reading one) stay
# on the scalar path — a plate parameter already threads per cell as
# a name, and a latent transform is not an affine predictor. Other
# scalar assignments, data gathers (`x[g]`), and literal indexing
# (`v[1]`) likewise stay scalar-path, exactly as before.
# `allow_stated` gates aliases over stated priors: a link-wrapped use
# admits stated-Normal coefficients (links exist only over predictors,
# so the scalar path cannot spell the use at all), while a bare use
# keeps them scalar — naming a stated name must not re-bucket it.
_is_scale_predictor_def(s::Symbol, ctx, allow_stated::Bool) =
    haskey(ctx.detmap, s) && !(s in ctx.plate_names) &&
    !_derived_reads_latent(s, ctx) &&
    (get(ctx.detshape, s, :scalar) === :vector ||
        _is_factor_index_def(ctx.detmap[s], ctx) ||
        _is_scalar_coef_def(s, ctx, allow_stated))

# A scalar definition spelling an intercept-only predictor (`name =
# coef` over a bare coefficient): admitted to the scale-predictor slot
# exactly when `_classify_symbol` would take the RHS as an
# InterceptTerm — anything it rejects or routes elsewhere (computed
# scalars, data, latents, varying draws, matrices) stays on the scalar
# path, so alias chains (`s2 = s`), literal scales, and data-column
# scales keep today's behavior bit-for-bit. Stated priors stay scalar
# on a bare use (`allow_stated == false`): a direct stated name is a
# parameter (`_lower_scale`), so its alias must be one too — routing
# the alias to analysis would re-bucket the same prior by spelling.
# Under a link wrapper (`allow_stated == true`) stated coef-prior names
# route to analysis like the location path's stated intercept priors
# (`a ~ Normal(0, 5)` over `eta = a .+ b .* x`): the wrapper has no
# scalar meaning, so the predictor path is the only spelling.
function _is_scalar_coef_def(s::Symbol, ctx, allow_stated::Bool)
    haskey(ctx.detmap, s) || return false
    s in ctx.plate_names && return false
    _derived_reads_latent(s, ctx) && return false
    get(ctx.detshape, s, :scalar) === :scalar || return false
    rhs = ctx.detmap[s]
    rhs isa Symbol || return false
    rhs in ctx.data && return false
    rhs in ctx.vecdefs && return false
    haskey(ctx.detmap, rhs) && return false
    rhs in ctx.prior_names && (rhs ∉ ctx.coef_priors || !allow_stated) &&
        return false
    rhs in ctx.plate_names && return false
    rhs in ctx.scan_states && return false
    rhs in ctx.varying_contribs && return false
    rhs in ctx.varying_draws_names && return false
    haskey(ctx.matrices, rhs) && return false
    return true
end

function _is_factor_index_def(rhs, ctx)
    rhs isa Expr || return false
    rhs.head === :ref || return false
    length(rhs.args) == 2 || return false
    base, idx = rhs.args
    base isa Symbol || return false
    idx isa Symbol || return false
    base in ctx.data && return false
    return idx in ctx.data
end

function _lower_scale_wrapped(lhs, s, ctx, predictors, pred_idx, coefuse)
    f = length(s.args) >= 1 ? s.args[1] : nothing
    targs = length(s.args) == 2 && s.args[2] isa Expr &&
            s.args[2].head === :tuple ? s.args[2].args : Any[]
    if !(f isa Symbol && f in (:exp, :logistic)) || length(targs) != 1
        return _sfail("response $lhs scale $(repr(s)) is not an admitted " *
                      "scale use — write a bare parameter/assignment name, " *
                      "a per-observation data column, a literal, a bare " *
                      "predictor definition, or one `exp.`/`logistic.` " *
                      "wrapper over a predictor definition")
    end
    inner = only(targs)
    inner isa Symbol && _is_scale_predictor_def(inner, ctx, true) || return _sfail(
        "response $lhs scale $(repr(s)): `$f.` wraps a predictor " *
        "definition (`$f.(predictor)` with `predictor = ...` affine in " *
        "data) — got $(repr(inner))")
    link = f === :exp ? LogLink : LogitLink
    pname = _lower_scale_predictor(lhs, inner, link, ctx, predictors,
        pred_idx, coefuse)
    return ScalePredictorRef(pname, link)
end

# Analyze (or intern) a scale predictor: exactly the location-predictor
# treatment (`_lower_location`'s named-definition arm) under the use-site
# link — affine analysis, coefficient-use recording, one link per
# predictor. Family admission (Gaussian/NB2/Gamma/Student/hurdle; Beta
# deferred) is the contract's gate (`_validate_scale_predictor`), so
# hand-built plans get the same rule.
function _lower_scale_predictor(lhs, name::Symbol, link, ctx, predictors,
        pred_idx, coefuse)
    haskey(pred_idx, name) || haskey(ctx.detmap, name) ||
        return _lower_scale_predictor_error(lhs, name, ctx)
    if haskey(pred_idx, name)
        pred = predictors[pred_idx[name]]
        pred.link === link || _sfail(
            "predictor $name is shared by slots needing links " *
            "$(pred.link) and $link — one link per predictor")
        return name
    end
    terms, uses = _analyze_predictor(name, ctx.detmap[name], ctx, lhs)
    _record_coefuses!(coefuse, name, uses, lhs)
    push!(predictors, PredictorSpec(name, link, terms, name))
    pred_idx[name] = length(predictors)
    return name
end

function _lower_scale_predictor_error(lhs, name, ctx)
    name in ctx.data && _sfail("response $lhs scale predictor $name is a " *
                              "data column, not a predictor definition " *
                              "(`$name = ...` affine in data)")
    name in ctx.prior_names && _sfail(
        "response $lhs scale predictor $name is a scalar parameter — a " *
        "predictor-fed scale is a per-observation definition " *
        "(`$name = ...` affine in data)")
    return _sfail("response $lhs scale predictor $name is not a predictor " *
                  "definition (`$name = ...` affine in data)")
end

# Claim a use-site predictor pin: the pin names a FRESH predictor, claimed
# once. Records the claiming response (first claim wins; a second claim of
# the same name reports its owner) and marks the response's pin used.
function _claim_pin!(lhs, pin, ctx, pred_idx)
    pin in ctx.taken && _sfail("response $lhs pins predictor $pin, but " *
        "`$pin` is already taken (a pin names a fresh predictor — rename one)")
    if haskey(pred_idx, pin)
        owner = get(ctx.pin_owner, pin, nothing)
        owner === nothing && _sfail("response $lhs pins predictor $pin, " *
            "but `$pin` is already a synthesized predictor (a pin names a " *
            "fresh predictor — rename it)")
        _sfail("response $lhs pins predictor $pin, but it is already " *
               "pinned by response $owner")
    end
    push!(ctx.pins_used, lhs)
    ctx.pin_owner[pin] = lhs
    return nothing
end

function _lower_location(lhs, loc, pred_link, ctx, predictors, pred_idx,
        coefuse; synth::Union{Nothing,Symbol} = nothing)
    # A per-cell latent VECTOR is the whole location via a LatentTerm predictor
    # (`lp = theta`, identity design; the latent's prior lives on its
    # PlateParameter, so no coefficient use is recorded). Two spellings: a bare
    # latent (`y[i] ~ Normal.(theta[i], s)`) or a deterministic transform of one
    # (`theta[i] = mu .+ tau .* z[i]` then `y[i] ~ Normal.(theta[i], s)` — the
    # non-centered / latent-transform shape, emitted as a derived column and
    # referenced directly). A derived location with NO latent stays a design
    # predictor; a latent-reading definition WITH coefficient structure is a
    # design predictor too (`b .* theta` classifies as a ContinuousTerm over
    # the latent — the SB `me` mirror), while a bare latent inside a larger
    # expression fails in `_classify_symbol`.
    if loc isa Symbol && loc in ctx.plate_names
        return _latent_predictor!(lhs, loc, pred_link, ctx, predictors, pred_idx)
    end
    if loc isa Symbol && haskey(ctx.detmap, loc) && _derived_reads_latent(loc, ctx) &&
            !_is_design_shaped(loc, ctx)
        return _latent_predictor!(lhs, loc, pred_link, ctx, predictors, pred_idx)
    end
    if loc isa Symbol
        haskey(ctx.matrices, loc) && _sfail(
            "response $lhs location $loc is a design matrix — locations " *
            "are predictors (`mu = $loc * b`), data, or inline affine " *
            "expressions")
        # A scan-state latent vector is a direct per-observation location: the
        # response mean IS the carried state (no linear predictor). Admitted
        # family/link is checked in `_validate_responses` (Gaussian-identity, v1).
        # A pin over a scan state claims nothing (no predictor is built) and
        # falls through to the unconsumed-pin error.
        loc in ctx.scan_states && return loc
        haskey(ctx.detmap, loc) ||
            return _lower_location_symbol_error(lhs, loc, ctx)
        pin = get(ctx.predictor_pins, lhs, nothing)
        if pin !== nothing && pin !== loc
            # A pin renames one response's predictor — the pinned location
            # cannot also lower under another name (either direction fails:
            # an already-interned location, or one pinned away before).
            haskey(pred_idx, loc) && _sfail(
                "response $lhs pins predictor $pin, but its location " *
                "`$loc` already lowers as its own predictor (a pin renames " *
                "one response's predictor — it cannot fork a shared " *
                "definition)")
            if haskey(ctx.pin_source, loc)
                prior, owner = ctx.pin_source[loc]
                _sfail("response $lhs pins predictor $pin, but its " *
                       "location `$loc` is already pinned as `$prior` by " *
                       "response $owner (a pin renames one response's " *
                       "predictor — it cannot fork a shared definition)")
            end
            _claim_pin!(lhs, pin, ctx, pred_idx)
            ctx.pin_source[loc] = (pin, lhs)
            # The source definition vanishes like any absorbed location (it
            # is referenced nowhere else — any other reader fails above).
            push!(ctx.absorbed, loc)
            pname = pin
        else
            pin !== nothing && push!(ctx.pins_used, lhs)
            if haskey(ctx.pin_source, loc)
                prior, owner = ctx.pin_source[loc]
                _sfail("response $lhs reads `$loc`, but `$loc` is pinned " *
                       "as predictor `$prior` by response $owner (a pin " *
                       "renames one response's predictor — it cannot fork " *
                       "a shared definition)")
            end
            pname = loc
            if haskey(pred_idx, pname)
                pred = predictors[pred_idx[pname]]
                pred.link === pred_link || _sfail(
                    "predictor $pname is shared by responses needing links " *
                    "$(pred.link) and $pred_link — one link per predictor")
                return pname
            end
        end
        terms, uses = _analyze_predictor(pname, ctx.detmap[loc], ctx, lhs)
    elseif loc isa Number
        _sfail("response $lhs location is a literal — use an intercept-only " *
               "predictor (`eta = a`)")
    else
        # Per-cell latents classify inline like data columns: `b .* x_true`
        # is a ContinuousTerm over the latent (the SB `me` mirror); a bare
        # latent fails in `_classify_symbol`, never silently.
        # Multi-eta responses (CategoricalLogit) index their synthetic
        # predictors; the single-eta default keeps its established name.
        pin = get(ctx.predictor_pins, lhs, nothing)
        if pin !== nothing
            synth === nothing || _sfail(
                "response $lhs pins predictor $pin, but $lhs needs one " *
                "predictor per index (multi-predictor response) — a pin " *
                "names exactly one predictor")
            _claim_pin!(lhs, pin, ctx, pred_idx)
            pname = pin
        else
            pname = synth === nothing ? Symbol(lhs, "_eta") : synth
            (haskey(ctx.detmap, pname) || haskey(pred_idx, pname)) && _sfail(
                "derived predictor name $pname collides with your definition — " *
                "rename yours")
        end
        terms, uses = _analyze_predictor(pname, loc, ctx, lhs)
    end
    _record_coefuses!(coefuse, pname, uses, lhs)
    push!(predictors, PredictorSpec(pname, pred_link, terms, pname))
    pred_idx[pname] = length(predictors)
    return pname
end

# Build the LatentTerm location predictor over `col` (a plate parameter or a
# derived column that reads one); the generator emits `lp = col`. A pin names
# it like any design predictor; latent predictors stay per-response (no
# interning here, pinned or not).
function _latent_predictor!(lhs, col, pred_link, ctx, predictors, pred_idx)
    pin = get(ctx.predictor_pins, lhs, nothing)
    if pin !== nothing
        _claim_pin!(lhs, pin, ctx, pred_idx)
        pname = pin
    else
        pname = Symbol(lhs, "_loc")
        (haskey(ctx.detmap, pname) || pname in ctx.plate_names) && _sfail(
            "latent-location predictor name $pname collides with your " *
            "definition — rename it")
    end
    term = TermSpec(LatentTerm, [col], NamedTuple(), col, Symbol(col, "_lat"))
    push!(predictors, PredictorSpec(pname, pred_link, [term], pname))
    pred_idx[pname] = length(predictors)
    return pname
end

# A latent-reading definition is a DESIGN predictor (not a latent transform)
# when it has coefficient structure (a coef-priored or free coefficient
# candidate), reads no scalar parameter (a non-coefficient sampled name
# like the non-centered `tau` — any such read marks a latent transform),
# and scales every latent it mentions by a coefficient (the SB `me`
# `a .+ b .* x_true` shape). Anything else latent-reading stays a
# LatentTerm location (the conservative pre-me behavior).
_is_design_shaped(name::Symbol, ctx) =
    name in ctx.structural && !_det_reads_param(name, ctx) &&
    _latent_uses_scaled(name, ctx)

# Every per-cell latent mention in the (inlined) definition is scaled by a
# coefficient (`b .* theta`): the summand split mirrors `_analyze_predictor`
# (inlining is idempotent — re-running it there re-absorbs the same names).
function _latent_uses_scaled(name::Symbol, ctx)
    rhs = _inline_structure(ctx.detmap[name], ctx, Set{Symbol}([name]),
        "predictor $name")
    out = Tuple{Int,Any}[]
    _collect_signed!(out, rhs, 1, name)
    for (_, core) in out
        _summand_latent_scaled(core, ctx) || return false
    end
    return true
end

# A summand is latent-scaled when no per-cell latent appears in it except
# as a factor of a dotted product with a coefficient (`b .* theta`, with
# data/local/vector factors alongside at most): bare latents, latents
# under any other operator, and parameter/computed scalings mark a latent
# transform instead.
function _summand_latent_scaled(core, ctx)
    core isa Symbol && return core ∉ ctx.plate_names
    core isa Expr || return true
    if core.head === :call && !isempty(core.args) && core.args[1] === :.*
        any(f -> f isa Symbol && f in ctx.plate_names,
            core.args[2:end]) || return true
        coefs = 0
        for f in core.args[2:end]
            if f isa Symbol
                k = _summand_kind(f, ctx)
                (k === :coef || k === :latent || k === :data ||
                    k === :local) || return false
                k === :coef && (coefs += 1)
            elseif f isa Number
                return false
            else
                _canon_shape(f, ctx.data, ctx.detshape) === :vector ||
                    return false
                any(s -> s in ctx.plate_names, _value_symbols(f)) &&
                    return false
            end
        end
        return coefs >= 1
    end
    return all(s -> s ∉ ctx.plate_names, _value_symbols(core))
end

# Does a definition transitively read a scalar parameter (a sampled name
# that is NOT a coefficient-prior candidate)?
function _det_reads_param(name::Symbol, ctx)
    seen = Set{Symbol}((name,))
    stack = collect(_value_symbols(ctx.detmap[name]))
    while !isempty(stack)
        s = pop!(stack)
        s in seen && continue
        push!(seen, s)
        s in ctx.prior_names && s ∉ ctx.coef_priors && return true
        haskey(ctx.detmap, s) && append!(stack, _value_symbols(ctx.detmap[s]))
    end
    return false
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
    loc in ctx.varying_contribs && _sfail(
        "response $lhs location is the varying contribution $loc — " *
        "locations must be predictors with estimated coefficients " *
        "(bind: `mu = a .+ $loc`)")
    loc in ctx.varying_draws_names && loc ∉ ctx.varying_contribs &&
        _sfail("response $lhs location is the varying draws block $loc " *
              "— slice it (`r ~ varying_slice($loc, ...)`) and bind the " *
              "slice in a predictor with estimated coefficients")
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

function _record_coefuses!(coefuse, pname, uses, lhs)
    for (name, addr, sign) in uses
        entries = get!(coefuse, name, Tuple{Symbol,Symbol,Int}[])
        for (p2, _, _) in entries
            p2 === pname && _sfail("response $lhs: coefficient $name is " *
                                   "used twice in predictor $pname")
        end
        push!(entries, (pname, addr, sign))
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
    _find_hcat(expanded) && _sfail(
        "predictor $pname calls `hcat` outside a matrix definition — " *
        "bind the matrix to a name first (`X = hcat(1, x, ...)`)")
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
            # Matrix uses claim every element addressee (one coefficient
            # per column, across matrix and affine terms alike).
            addrs = addr in keys(ctx.matrices) ?
                _matrix_element_addressees(ctx.matrices[addr]) : (addr,)
            for a in addrs
                haskey(addr_owner, a) && _sfail(
                    "predictor $pname: column $a has two coefficients " *
                    "$(addr_owner[a]) and $name — one coefficient per column")
                addr_owner[a] = name
            end
            push!(uses, use)
        end
        push!(terms, term)
    end
    # Zero-coefficient predictors: bare-data affines (a non-empty all-
    # offset summand list) and beta-free monotonic (`mo1`) predictors are
    # admitted — both evaluate the likelihood over a coefficient-free LP
    # (data for offsets, the increment-simplex contrast for `mo1`) with an
    # empty coefficient layout. Any other coefficient-free shape
    # (latent/effect/spline/hsgp/scan-only, or an empty summand list)
    # stays fail-closed: a scan summand needs a sibling coefficient
    # (SB's `ar` always pairs with an intercept).
    if isempty(uses) && !(!isempty(terms) &&
            all(t -> t.kind === OffsetTerm ||
                t.kind === MonotonicSummandTerm, terms))
        _sfail("predictor $pname has no estimated coefficients — add an " *
               "intercept or coefficient (bare-data offset affines and " *
               "beta-free `mo1()` predictors are the only " *
               "coefficient-free shapes)")
    end
    return terms, uses
end

function _inline_structure(ex, ctx, visited::Set{Symbol}, where)
    ex isa Symbol || return _inline_structure_expr(ex, ctx, visited, where)
    haskey(ctx.detmap, ex) || return ex
    # Summand atoms never hide in definitions: an inlined alias would
    # silently become a direct summand, bypassing the lowering screens
    # (scalar defs always inline; structural vector defs inline too).
    _uses_varying_contrib(ctx.detmap[ex], ctx.varying_contribs) &&
        _sfail("definition `$ex` (inlined into $where) references a " *
        "varying contribution, which lowers only as a direct predictor " *
        "summand (`mu = a .+ b .* x .+ r`), not inside definitions")
    _contains_spline(ctx.detmap[ex]) && _sfail("definition `$ex` (inlined " *
        "into $where) calls `spline()`, which lowers only as a direct " *
        "predictor summand (`mu = a .+ b .* x .+ spline(:s_x)`), not " *
        "inside definitions")
    _contains_hsgp(ctx.detmap[ex]) && _sfail("definition `$ex` (inlined " *
        "into $where) calls `hsgp()`, which lowers only as a direct " *
        "predictor summand (`mu = a .+ b .* x .+ hsgp(:h_x)`), not " *
        "inside definitions")
    _contains_mo(ctx.detmap[ex]) && _sfail("definition `$ex` (inlined " *
        "into $where) calls `mo()`, which lowers only in a predictor " *
        "(`mu = a .+ b .* mo(c, s)`), not inside definitions")
    _contains_mo1(ctx.detmap[ex]) && _sfail("definition `$ex` (inlined " *
        "into $where) calls `mo1()`, which lowers only as a direct " *
        "predictor summand (`mu = a .+ mo1(c, s)`), not " *
        "inside definitions")
    _contains_dar(ctx.detmap[ex]) && _sfail("definition `$ex` (inlined " *
        "into $where) calls `dar()`, which lowers only as a direct " *
        "predictor summand (`mu = a .+ dar(beta, sigma)`), not " *
        "inside definitions")
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
    # Bare contributions route via `_classify_symbol` above; any other
    # expression mentioning one fails here (additive-only, never nested).
    if _uses_varying_contrib(core, ctx.varying_contribs)
        _sfail("predictor $pname: varying contributions lower only as " *
              "direct additive summands (`mu = a .+ b .* x .+ r`), not " *
              "nested in $(repr(core))")
    end
    if _contains_spline(core)
        _is_spline_call(core) ||
            _sfail("predictor $pname: `spline()` summands lower only as " *
                  "direct additive summands " *
                  "(`mu = a .+ b .* x .+ spline(:s_x)`), not nested in " *
                  "$(repr(core))")
        return _classify_spline(pname, core, sign, ctx)
    end
    if _contains_hsgp(core)
        _is_hsgp_call(core) ||
            _sfail("predictor $pname: `hsgp()` summands lower only as " *
                  "direct additive summands " *
                  "(`mu = a .+ b .* x .+ hsgp(:h_x)`), not nested in " *
                  "$(repr(core))")
        return _classify_hsgp(pname, core, sign, ctx)
    end
    if _contains_dar(core)
        _is_dar_call(core) ||
            _sfail("predictor $pname: `dar()` summands lower only as " *
                  "direct additive summands " *
                  "(`mu = a .+ dar(beta, sigma)`), not nested in " *
                  "$(repr(core))")
        return _classify_dar(pname, core, sign, ctx)
    end
    if _contains_scan(core, ctx.scan_states)
        _is_scan_product(core) ||
            _sfail("predictor $pname: scan states lower only as direct " *
                  "scaled summands (`mu = a .+ b .* u`), not nested in " *
                  "$(repr(core))")
        return _classify_scan(pname, core, sign, ctx)
    end
    if _contains_mo1(core)
        _is_mo1_call(core) ||
            _sfail("predictor $pname: `mo1()` summands lower only as " *
                  "direct additive summands " *
                  "(`mu = a .+ mo1(c, s)`), not nested in " *
                  "$(repr(core))")
        return _classify_mo1(pname, core, sign, ctx)
    end
    if _contains_mo(core)
        # `mo()` lowers only scaled by one free coefficient — the product
        # arm below routes to `_classify_mo_product`; any other shape
        # fails here naming the spelling.
        core isa Expr && core.head === :call && !isempty(core.args) &&
            core.args[1] === :.* ||
            _sfail("predictor $pname: `mo()` takes a free coefficient " *
                  "(`mu = a .+ b .* mo(c, s)`); beta-free monotonic " *
                  "summands spell `mo1(c, s)`")
    end
    # Design matrices lower as `X * b` matmuls (the arm validates the
    # right-hand side); every other matrix mention fails below (the
    # `_find_matrix_use` walk skips matmul-shaped nodes, so a hit is
    # always a stray).
    if core isa Expr && core.head === :call && length(core.args) == 3 &&
            core.args[1] === :* && core.args[2] isa Symbol &&
            core.args[2] in keys(ctx.matrices)
        return _classify_matmul(pname, core, sign, ctx)
    end
    hit = _find_matrix_use(core, keys(ctx.matrices))
    if hit !== nothing
        _sfail("predictor $pname: design matrix `$hit` lowers only in " *
               "a predictor matmul (`mu = $hit * b`)")
    end
    # A matmul node composed under anything but a bare additive
    # summand (products, links, extraction): the matmul arm above
    # owns direct matmuls, so anything reaching here miscomposes.
    if _contains_matmul(core, keys(ctx.matrices))
        _matmul_composition_error(pname, core, ctx)
    end
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
        if s in ctx.coef_priors ||
            _is_free_name(s, ctx.data, detkeys, ctx.prior_names, ctx.plate_names)]
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

# An `X * b` matmul: the design matrix's K columns take one coefficient
# vector — declared (`b[axes(X, 2)] .~ ...`, width-checked against the
# use matrix) or free (implicit, width follows the use). Coefficient-ness
# is by name role, not shape (the affine free-name precedent): data,
# computed, sampled, latent, scan, and matrix names all fail naming the
# spelling. Records `(b, X, sign)`; element priors lower per use-matrix
# column.
function _classify_matmul(pname, core::Expr, sign::Int, ctx)
    X, r = core.args[2], core.args[3]
    m = ctx.matrices[X]
    K = length(m.columns)
    need = "`$X * $r` needs a bare $K-element coefficient vector " *
        "(`mu = $X * b` with `b[axes($X, 2)] .~ ...`)"
    r isa Symbol || _sfail("predictor $pname: $need, got $(repr(r))")
    if haskey(ctx.coefvecs, r)
        S = ctx.coefvecs[r]
        Smat = get(ctx.matrices, S, nothing)
        Smat === nothing && _sfail("predictor $pname: coefficient " *
                                   "vector `$r` is sized by `$S`, which is " *
                                   "not a design matrix " *
                                   "(`$S = hcat(1, x, ...)`)")
        length(Smat.columns) == K || _sfail(
            "predictor $pname: coefficient vector `$r` has " *
            "$(length(Smat.columns)) elements (sized by `$S`) but matrix " *
            "`$X` has $K columns")
    else
        role = r in ctx.data ? "a data column" :
            haskey(ctx.detmap, r) ? "a computed assignment" :
            r in ctx.prior_names ? "a sampled name" :
            r in ctx.plate_names ? "a latent vector" :
            r in ctx.scan_states ? "a scan state" :
            r in keys(ctx.matrices) ? "a design matrix" : nothing
        role === nothing || _sfail("predictor $pname: $need — `$r` is $role")
    end
    push!(ctx.matrices_used, X)
    data_cols = Symbol[c for c in m.columns if c !== nothing]
    return TermSpec(MatrixTerm, data_cols, (matrix = X,), X,
        Symbol(X, "_term")), (r, X, sign)
end

# A `spline(:id)` summand: the named basis's direct summand in the
# enclosing predictor (SB's `X*b + Z*(sd*z)` shape). Additive only, one
# target per smooth (a second use fails here so the surface error names
# both predictors; the contract re-checks for hand-built plans).
function _classify_spline(pname, core::Expr, sign::Int, ctx)
    where = "predictor $pname"
    args = core.args[2:end]
    length(args) == 1 ||
        _sfail("$where spline summand takes `spline(:id)` exactly, got " *
              "$(repr(core))")
    id = only(args)
    id isa QuoteNode && id.value isa Symbol ||
        _sfail("$where quotes its spline id: got $(repr(id)) — write " *
              "`spline(:id)`")
    id = id.value
    sign > 0 ||
        _sfail("$where negates a `spline()` summand — summands are " *
              "additive only (write `.+ spline(:$id)`)")
    haskey(ctx.splines, id) ||
        _sfail("$where uses unknown spline :$id — declare it with " *
              "`spline_basis(:$id, x)`")
    haskey(ctx.spline_uses, id) &&
        _sfail("$where reuses spline :$id, which already feeds predictor " *
              "`$(ctx.spline_uses[id])` (one target per smooth)")
    ctx.spline_uses[id] = pname
    label = Symbol("spline_", pname, "_", id)
    return TermSpec(SplineSummandTerm, ColumnRef[], (spline_id = id,),
        label, label), nothing
end

# An `hsgp(:id)` summand: the named basis's direct summand in the
# enclosing predictor (SB's `PHI * (sqrt_spd .* beta)` shape, Stage B).
# Additive only, one target per basis (a second use fails here so the
# surface error names both predictors; the contract re-checks for
# hand-built plans).
function _classify_hsgp(pname, core::Expr, sign::Int, ctx)
    where = "predictor $pname"
    args = core.args[2:end]
    length(args) == 1 ||
        _sfail("$where hsgp summand takes `hsgp(:id)` exactly, got " *
              "$(repr(core))")
    id = only(args)
    id isa QuoteNode && id.value isa Symbol ||
        _sfail("$where quotes its hsgp id: got $(repr(id)) — write " *
              "`hsgp(:id)`")
    id = id.value
    sign > 0 ||
        _sfail("$where negates an `hsgp()` summand — summands are " *
              "additive only (write `.+ hsgp(:$id)`)")
    haskey(ctx.hsgps, id) ||
        _sfail("$where uses unknown hsgp :$id — declare it with " *
              "`hsgp_basis(:$id, x)`")
    haskey(ctx.hsgp_uses, id) &&
        _sfail("$where reuses hsgp :$id, which already feeds predictor " *
              "`$(ctx.hsgp_uses[id])` (one target per basis)")
    ctx.hsgp_uses[id] = pname
    label = Symbol("hsgp_", pname, "_", id)
    return TermSpec(HSGPSummandTerm, ColumnRef[], (hsgp_id = id,),
        label, label), nothing
end

# A scan state read anywhere in a summand (bare Symbol leaves; quoted ids
# such as `spline(:id)` are not reads).
_contains_scan(ex, states) = ex isa Expr && _contains_scan_go(ex, states)

_contains_scan_go(ex::Expr, states) =
    any(a -> _contains_scan_arg(a, states), ex.args)

_contains_scan_arg(a, states) =
    a isa Symbol ? a in states :
    a isa QuoteNode ? false :
    a isa Expr ? _contains_scan_go(a, states) : false

_is_scan_product(core) =
    core isa Expr && core.head === :call && !isempty(core.args) &&
    core.args[1] === :.*

# A `coef .* state` scaled scan summand (SB's `ar` latent path with its
# free beta): exactly two factors, one the scan state, the other a bare
# sampled scalar. Additive only. The coefficient records in `scan_coefs`
# (checked disjoint from predictor coefficients after lowering) and lowers
# to a `SampledParameter`, never a population prior.
function _classify_scan(pname, core::Expr, sign::Int, ctx)
    where = "predictor $pname"
    factors = core.args[2:end]
    length(factors) == 2 ||
        _sfail("$where scales a scan state with $(length(factors)) " *
              "factors — write `coef .* state` exactly, got $(repr(core))")
    stripped = [_strip_sign(f) for f in factors]
    inner = sign * prod(first, stripped)
    inner > 0 ||
        _sfail("$where negates a scan summand — summands are additive " *
              "only (write `.+ b .* u`)")
    states = [g for (_, g) in stripped if g isa Symbol && g in ctx.scan_states]
    if isempty(states)
        _sfail("$where nests a scan read inside $(repr(core)) — scan " *
              "states lower only as direct scaled summands " *
              "(`mu = a .+ b .* u`)")
    end
    length(states) == 1 ||
        _sfail("$where combines two scan states in $(repr(core)) — " *
              "products of latents are nonlinear (one `coef .* state` " *
              "summand per scan read)")
    others = [g for (_, g) in stripped if !(g isa Symbol && g in ctx.scan_states)]
    coef = only(others)
    coef isa Symbol ||
        _sfail("$where scales scan state :$(only(states)) by " *
              "$(repr(coef)) — scan coefficients are bare sampled " *
              "scalars (`b .* u`)")
    coef in ctx.data &&
        _sfail("$where scales scan state :$(only(states)) by the data " *
              "column $coef — data-varying (interaction) scalings are planned")
    (coef in ctx.vecdefs || coef in ctx.plate_names) &&
        _sfail("$where scales scan state :$(only(states)) by $coef, " *
              "which is vector-valued — scan coefficients are scalars")
    haskey(ctx.detmap, coef) &&
        _sfail("$where scales scan state :$(only(states)) by the " *
              "computed scalar $coef — computed coefficients are not " *
              "in slice 1")
    coef in ctx.prior_names ||
        _sfail("$where scales scan state :$(only(states)) by $coef, " *
              "which has no `~` statement — scan coefficients are " *
              "sampled scalars (`$coef ~ Normal(0, 1)`)")
    push!(ctx.scan_coefs, coef)
    label = Symbol("scan_", pname, "_", only(states))
    return TermSpec(ScanSummandTerm, ColumnRef[],
        (scan_id = only(states), coef = coef), label, label), nothing
end

# Shared `mo(col, s)` / `mo1(col, s)` argument screen: two bare names —
# the index column (a raw data column of integer level codes 1..K, bound
# by the emitter — SB's `<c>_idx`) and the increments simplex (a
# `~ Dirichlet(...)` parameter). Shape-verified here so both classifiers
# fail with the use-site spelling.
function _mo_args(pname, core::Expr, ctx, head::Symbol)
    where = "predictor $pname"
    args = core.args[2:end]
    length(args) == 2 ||
        _sfail("$where `$head()` takes `(index column, increments)` " *
              "exactly (`$head(c, s)`), got $(repr(core))")
    col, incr = args
    col isa Symbol ||
        _sfail("$where `$head()` index must be a bare data column of " *
              "level codes, got $(repr(col))")
    col in ctx.data ||
        _sfail("$where `$head()` index $col must be a data column " *
              "(the emitter binds integer codes 1..K)")
    incr isa Symbol ||
        _sfail("$where `$head()` increments must name a " *
              "`~ Dirichlet(...)` simplex parameter, got $(repr(incr))")
    incr in ctx.dirichlet_names ||
        _sfail("$where `$head()` increments $incr is not a " *
              "`~ Dirichlet(...)` simplex parameter")
    haskey(ctx.mo_uses, incr) &&
        _sfail("$where reuses increments $incr, which already feed " *
              "predictor `$(ctx.mo_uses[incr])` (one monotonic term per " *
              "simplex)")
    ctx.mo_uses[incr] = pname
    return col, incr
end

# A `coef .* mo(col, s)` product: the only `mo()` shape (SB's free-beta
# monotonic column). Exactly two factors — one coefficient, one `mo()`
# call — in either order; anything else (unscaled, multi-scaled,
# interacting, `mo1()`-carrying) fails naming the spelling.
function _classify_mo_product(pname, core::Expr, sign::Int, ctx)
    where = "predictor $pname"
    inner = sign
    coef = nothing
    mocall = nothing
    for f in core.args[2:end]
        s, g = _strip_sign(f)
        inner *= s
        if _is_mo_call(g)
            mocall === nothing ||
                _sfail("$where combines two `mo()` calls — one " *
                      "monotonic column per product " *
                      "(`b .* mo(c, s)`)")
            mocall = g
        elseif _contains_mo(g) || _contains_mo1(g)
            _sfail("$where nests `$(repr(g))` — `mo()` lowers only as " *
                  "`coef .* mo(col, s)`")
        elseif g isa Symbol && _summand_kind(g, ctx) === :coef
            coef === nothing ||
                _sfail("$where scales `mo()` by two coefficients — " *
                      "one coefficient per column")
            coef = g
        else
            _sfail("$where combines `mo()` with $(repr(g)) — `mo()` " *
                  "lowers only as `coef .* mo(col, s)`")
        end
    end
    mocall === nothing && _sfail("internal: mo product without an `mo()` call")
    coef === nothing &&
        _sfail("$where `mo()` takes a free coefficient " *
              "(`b .* mo(c, s)`); beta-free monotonic summands spell " *
              "`mo1(c, s)`")
    col, incr = _mo_args(pname, mocall, ctx, :mo)
    label = Symbol("mo_", pname, "_", col)
    return TermSpec(MonotonicTerm, [col], (increments = incr,), col,
        label), (coef, col, inner)
end

# A `mo1(col, s)` summand: the named increments' contrast as a direct
# beta-free summand (SB's `mo1(c)` shape). Additive only, one term per
# simplex (SB allocates one submodel per term; the contract re-checks
# for hand-built plans).
function _classify_mo1(pname, core::Expr, sign::Int, ctx)
    where = "predictor $pname"
    sign > 0 ||
        _sfail("$where negates a `mo1()` summand — summands are " *
              "additive only (write `.+ mo1(c, s)`)")
    col, incr = _mo_args(pname, core, ctx, :mo1)
    label = Symbol("mo1_", pname, "_", col)
    return TermSpec(MonotonicSummandTerm, [col], (increments = incr,),
        label, label), nothing
end

# A `dar(beta, sigma)` summand: the zero-started differenced-AR(1)
# trajectory as a direct beta-free summand (SB's `dar(time)` shape —
# the formula intercept is the initial level, the `mo1` splice shape).
# Exactly two bare sampled scalars: a truncated-`[0, 1]`-Normal
# persistence and a positive-Normal scale. Additive only; one `dar()`
# call per predictor in v1. The state synthesizes as `dar_<pname>` and
# claims the name up front (the `_implicit_vector!` precedent). Both
# parameters record in `dar_coefs` (checked disjoint from predictor
# coefficients after lowering) and lower to `SampledParameter`s, never
# population priors.
function _classify_dar(pname, core::Expr, sign::Int, ctx)
    where = "predictor $pname"
    sign > 0 ||
        _sfail("$where negates a `dar()` summand — summands are " *
              "additive only (write `.+ dar(beta, sigma)`)")
    args = core.args[2:end]
    length(args) == 2 ||
        _sfail("$where `dar()` takes `(persistence, scale)` exactly " *
              "(`dar(beta, sigma)`), got $(repr(core))")
    beta, sigma = args
    beta isa Symbol ||
        _sfail("$where `dar()` persistence must be a bare sampled " *
              "parameter, got $(repr(beta))")
    sigma isa Symbol ||
        _sfail("$where `dar()` scale must be a bare sampled parameter, " *
              "got $(repr(sigma))")
    beta === sigma &&
        _sfail("$where `dar()` persistence and scale must be distinct " *
              "parameters (SB samples `beta` and `sigma` separately), " *
              "got :$beta twice")
    beta in ctx.prior_names ||
        _sfail("$where `dar()` persistence $beta has no `~` statement — " *
              "dar parameters are sampled scalars " *
              "(`$beta ~ truncated(Normal(0.5, 0.2), 0, 1)`)")
    beta in ctx.dar_beta_names ||
        _sfail("$where `dar()` persistence $beta must be " *
              "`truncated(Normal(mu, s), 0, 1)` (SB's `beta ~ " *
              "normal(0.5, 0.2; lower=0, upper=1)`)")
    sigma in ctx.prior_names ||
        _sfail("$where `dar()` scale $sigma has no `~` statement — dar " *
              "parameters are sampled scalars (`$sigma ~ HalfNormal(0.2)`)")
    sigma in ctx.dar_sigma_names ||
        _sfail("$where `dar()` scale $sigma must be `HalfNormal(s)` or " *
              "`truncated(Normal(0, s), 0, Inf)` (SB's `sigma ~ " *
              "normal(0, 0.2; lower=0)`)")
    state = Symbol(:dar_, pname)
    state in ctx.dar_states &&
        _sfail("$where calls `dar()` twice — one dar summand per " *
              "predictor in v1")
    state in ctx.taken &&
        _sfail("$where dar state $state collides with your definition — " *
              "rename yours")
    push!(ctx.taken, state)
    push!(ctx.dar_states, state)
    push!(ctx.dar_coefs, beta)
    push!(ctx.dar_coefs, sigma)
    push!(ctx.dar_specs, DarSpec(state, beta, sigma, state))
    return TermSpec(DarSummandTerm, ColumnRef[], (dar_id = state,),
        state, state), nothing
end

function _classify_symbol(pname, core::Symbol, sign::Int, ctx)
    if core in ctx.varying_contribs
        sign < 0 && _sfail("predictor $pname: varying contribution " *
                           "`$core` is additive-only (write `.+ $core`)")
        haskey(ctx.varying_use, core) &&
            _sfail("predictor $pname: varying contribution `$core` is " *
                  "already used in predictor $(ctx.varying_use[core]) " *
                  "(one contribution feeds exactly one predictor)")
        ctx.varying_use[core] = pname
        plabel = only(p.draws for p in ctx.varying_pending
            if p.contrib === core)
        d = ctx.varying_draws[plabel]
        rlabel = Symbol("r_", pname, "_", d.suffix)
        return TermSpec(VaryingEffectTerm, ColumnRef[d.group],
            (draws = plabel,), rlabel, rlabel), nothing
    end
    core in ctx.varying_draws_names && core ∉ ctx.varying_contribs &&
        _sfail("predictor $pname: `$core` is a varying draws block, not " *
              "a per-observation value — slice it " *
              "(`r ~ varying_slice($core, ...)`) and use the slice")
    # (Design matrices never reach here as bare summands: named
    # definitions screen them at extraction, inline locations at the
    # location arm, and dotted compositions at canonicalization.)
    (core in ctx.data || core in ctx.vecdefs) &&
        return TermSpec(OffsetTerm, [core], NamedTuple(),
        core, Symbol(core, "_off")), nothing
    core in ctx.plate_names && _sfail(
        "predictor $pname: bare latent $core is not a term — scale it " *
        "by a coefficient (`b .* $core`, the SB `me` mirror)")
    core in ctx.scan_states && _sfail("predictor $pname: $core is a bare " *
        "scan state — LP use needs a sampled coefficient (`b .* $core` " *
        "in an additive position); a bare scan state is only a direct " *
        "response location (`y .~ Normal.($core, s)`)")
    haskey(ctx.detmap, core) && _sfail("predictor $pname: $core is a " *
                                       "computed scalar, not a sampled " *
                                       "coefficient (computed coefficients " *
                                       "are not in slice 1)")
    core in ctx.prior_names && core ∉ ctx.coef_priors &&
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
    if any(f -> _contains_mo(f) || _contains_mo1(f), core.args[2:end])
        return _classify_mo_product(pname, core, sign, ctx)
    end
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
            elseif k === :data || k === :local || k === :latent
                # A per-cell latent scales like a data column (`b .* x_true`
                # — the SB `me` mirror): the ContinuousTerm below names the
                # latent vector and the coefficient stays free.
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
    s in ctx.prior_names && s ∉ ctx.coef_priors && return :param
    s in ctx.plate_names && return :latent
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
function _lower_coefficient_priors(sample, coefuse, predictors,
        matrices::Dict{Symbol,DesignMatrix},
        r2d2::Set{Symbol} = Set{Symbol}(),
        hs::Set{Symbol} = Set{Symbol}())
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
        # R2D2 predictors carry their prior mass in the R2D2Prior
        # (overrides included) — _lower_r2d2_priors, not here. Horseshoe
        # predictors likewise — _lower_horseshoe_priors, not here.
        pred.name in r2d2 && continue
        pred.name in hs && continue
        for t in pred.terms
            # Offsets carry no coefficient; latent terms carry a PlateParameter
            # whose prior lives on the plate parameter, not as a coefficient;
            # effect terms carry a VaryingDraws, whose geometry is
            # self-priored; spline summands carry SplineVectors,
            # self-priored likewise; hsgp summands carry an HSGPBasis,
            # self-priored likewise; and monotonic summands (mo1) carry
            # an increment simplex, also self-priored.
            (t.kind === OffsetTerm || t.kind === LatentTerm ||
                t.kind === VaryingEffectTerm ||
                t.kind === SplineSummandTerm ||
                t.kind === HSGPSummandTerm ||
                t.kind === ScanSummandTerm ||
                t.kind === MonotonicSummandTerm ||
                t.kind === DarSummandTerm) && continue
            if t.kind === MatrixTerm
                append!(priors, _lower_matrix_priors(pred, t, coefuse,
                    stated, matrices))
                continue
            end
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
    _check_identified(predictors, levelmaps, matrices)
    return priors, levelmaps
end

# A matrix coefficient vector: K per-element PopulationPriors over the
# use-matrix columns (`:Intercept` at intercept positions). Unstated
# vectors default to K× Normal(0, 1) (emitter convention — the width is
# static, so no declaration is needed to size them). Stated vectors take
# `b[axes(S, 2)] .~ Normal.(loc, scale)` with scalar (shared) or
# length-K literal-vector (per-element) args — real broadcast semantics.
function _lower_matrix_priors(pred, t, coefuse, stated, matrices)
    X = t.options.matrix
    m = get(matrices, X, nothing)
    m === nothing && _sfail("internal: matrix term over unknown matrix $X")
    use = _find_use(coefuse, pred.name, X)
    use === nothing && _sfail("internal: no coefficient use for " *
                              "($(pred.name), $X)")
    name, _, sign = use
    elems = _matrix_element_addressees(m)
    K = length(elems)
    haskey(stated, name) || return PopulationPrior[
        PopulationPrior(pred.name, e, 0.0, 1.0) for e in elems]
    s = stated[name]
    # Reachable only with a matrix marker sized for this use:
    # classification accepts a use only for declared vectors (width
    # checked against the use matrix) or free names (unstated,
    # defaulted above).
    s.matrix === nothing && _sfail("internal: matrix prior for $name " *
                                   "lost its sizing matrix")
    locs, scales = _coefficient_matrix_normal(name, s.rhs, pred.name, K)
    return PopulationPrior[PopulationPrior(pred.name, e, sign * l, sc)
        for (e, l, sc) in zip(elems, locs, scales)]
end

# Dotted matrix priors peel to K (location, scale) pairs: each arg is a
# Real (shared over elements) or a literal K-vector (per-element) —
# Julia broadcast semantics over `Normal.(loc, scale)`.
function _coefficient_matrix_normal(name, rhs, pname, K)
    rhs isa Expr && rhs.head === :. && length(rhs.args) == 2 &&
        rhs.args[1] === :Normal && rhs.args[2] isa Expr &&
        rhs.args[2].head === :tuple || _sfail(
            "coefficient $name of predictor $pname needs a broadcast " *
            "`Normal.(location, scale)` prior, got $(repr(rhs))")
    args = rhs.args[2].args
    length(args) == 2 || _sfail("coefficient $name of predictor $pname " *
                                "needs `Normal.(location, scale)`")
    return _matrix_prior_arg(name, args[1], pname, K, "location"),
        _matrix_prior_arg(name, args[2], pname, K, "scale")
end

function _matrix_prior_arg(name, a, pname, K, role)
    a isa Real && return fill(Float64(a), K)
    a === :Inf && return fill(Inf, K)
    if a isa Expr && a.head === :vect
        length(a.args) == K || _sfail(
            "coefficient $name prior $role has $(length(a.args)) " *
            "elements for $K columns — one per column")
        all(x -> x isa Real || x === :Inf, a.args) || _sfail(
            "coefficient $name prior $role elements must be literals " *
            "(hierarchical coefficient priors are not in slice 1)")
        return Float64[x === :Inf ? Inf : x for x in a.args]
    end
    _sfail("coefficient $name prior $role must be a literal or a " *
           "literal $K-vector (hierarchical coefficient priors are not " *
           "in slice 1), got $(repr(a))")
end

# Horseshoe predictors: any predictor with a stated scalar `~ Horseshoe()`
# coefficient prior. Stated beside an `r2d2(...)` declaration, or outside
# an intercept/continuous term, it fails here (one structured prior per
# predictor; the flat slice covers scalar coefficients only).
function _horseshoe_predictors(sample, coefuse, predictors,
        r2d2::Set{Symbol})
    by_pred = Dict{Symbol,PredictorSpec}(p.name => p for p in predictors)
    out = Set{Symbol}()
    for s in sample
        haskey(coefuse, s.lhs) || continue
        _is_horseshoe_call(s.rhs) || continue
        s.levels === nothing && s.matrix === nothing || _sfail(
            "coefficient $(s.lhs) takes a scalar `Horseshoe(...)` prior " *
            "— sized (levels/matrix) horseshoe priors are not in slice 1")
        for (pname, addr, _) in coefuse[s.lhs]
            pred = get(by_pred, pname, nothing)
            pred === nothing && continue
            pname in r2d2 && _sfail(
                "predictor $pname carries both `r2d2(...)` and a " *
                "`~ Horseshoe()` coefficient prior — one structured " *
                "prior per predictor")
            scalar = any(pred.terms) do t
                a = t.kind === InterceptTerm ? :Intercept :
                    t.kind === ContinuousTerm ? only(t.columns) : nothing
                a === addr
            end
            scalar || _sfail(
                "coefficient $(s.lhs) of predictor $pname takes " *
                "`Horseshoe(...)` outside an intercept/continuous term " *
                "— the flat slice covers scalar coefficients only")
            push!(out, pname)
        end
    end
    return out
end

# SB `Horseshoe(...)` keyword contract: keywords `local_scale` /
# `global_scale` only (no positionals), positive finite literal scales
# defaulting to 1.0 (Bools rejected — SB's numeric-constant rule).
function _coefficient_horseshoe(name, rhs, pname, addr)
    where = "coefficient $name of predictor $pname"
    kws = Any[]
    for a in rhs.args[2:end]
        if a isa Expr && a.head === :parameters
            append!(kws, a.args)
        elseif a isa Expr && a.head === :kw
            push!(kws, a)
        else
            _sfail("$where `Horseshoe(...)` takes no positional " *
                   "arguments — use `local_scale=` and/or `global_scale=`")
        end
    end
    ls, gs = 1.0, 1.0
    for kw in kws
        kw isa Expr && kw.head === :kw && length(kw.args) == 2 ||
            _sfail("$where `Horseshoe(...)` takes keywords " *
                   "`local_scale`/`global_scale` only")
        key, v = kw.args
        key === :local_scale || key === :global_scale || _sfail(
            "$where `Horseshoe(...)` takes keywords " *
            "`local_scale`/`global_scale` only, got `$key`")
        v isa Bool && _sfail("$where `Horseshoe($key=...)` needs a " *
                             "numeric literal scale, got `$v`")
        v isa Real && isfinite(v) && v > 0 || _sfail(
            "$where `Horseshoe($key=...)` must be finite and " *
            "strictly positive, got $(repr(v))")
        if key === :local_scale
            ls = Float64(v)
        else
            gs = Float64(v)
        end
    end
    return ls, gs
end

# Horseshoe predictors to IR: one HorseshoePrior per stated `~ Horseshoe()`
# addressee plus its synthesized triple; stated-Normal (or unstated)
# scalar addressees ride Normal scalars (the mixed-predictor
# coordinate). Anything but intercept/continuous/offset terms fails
# closed (the flat slice).
function _lower_horseshoe_priors(sample, coefuse, predictors,
        hs::Set{Symbol}, taken::Set{Symbol})
    stated = Dict{Symbol,Any}()
    for s in sample
        haskey(coefuse, s.lhs) && (stated[s.lhs] = s)
    end
    out = HorseshoePrior[]
    params = SampledParameter[]
    for pred in predictors
        pred.name in hs || continue
        for t in pred.terms
            (t.kind === InterceptTerm || t.kind === ContinuousTerm ||
                t.kind === OffsetTerm) || _sfail(
                "horseshoe over predictor $(pred.name) meets a " *
                "$(t.kind) term — the flat slice covers " *
                "intercept/continuous coefficients only")
            t.kind === OffsetTerm && continue
            addr = t.kind === InterceptTerm ? :Intercept : only(t.columns)
            use = _find_use(coefuse, pred.name, addr)
            use === nothing && _sfail("internal: no coefficient use for " *
                                      "($(pred.name), $addr)")
            name, _, sign = use
            if !haskey(stated, name)
                push!(params, _horseshoe_normal_param(pred.name, addr,
                    0.0, 1.0, taken))
                continue
            end
            s = stated[name]
            if _is_horseshoe_call(s.rhs)
                ls, gs = _coefficient_horseshoe(name, s.rhs, pred.name,
                    addr)
                push!(out, HorseshoePrior(pred.name, addr, ls, gs, sign))
                for (nm, fam, args, ov) in (
                        (horseshoe_raw_name(pred.name, addr), :normal,
                            (arg1 = 0, arg2 = 1), nothing),
                        (horseshoe_lambda_name(pred.name, addr), :cauchy,
                            (arg1 = 0, arg2 = ls), :positive_stan),
                        (horseshoe_tau_name(pred.name, addr), :cauchy,
                            (arg1 = 0, arg2 = gs), :positive_stan))
                    nm in taken && _sfail(
                        "horseshoe over $(pred.name): synthesized $nm " *
                        "collides with a model name — rename yours")
                    push!(taken, nm)
                    push!(params, SampledParameter(nm, fam, args, ov, nm))
                end
                continue
            end
            s.levels !== nothing && _sfail("coefficient $name takes a " *
                                           "scalar prior (`$name ~ Normal`), " *
                                           "not a levels prior — it is used " *
                                           "as $(t.kind), not a factor")
            loc, scale = _coefficient_normal(name, s.rhs, pred.name, addr)
            push!(params, _horseshoe_normal_param(pred.name, addr,
                sign * loc, scale, taken))
        end
    end
    return out, params
end

# One mixed-predictor Normal coordinate (the PopulationPrior convention:
# the sign rides the location, the scalar IS the signed coordinate).
function _horseshoe_normal_param(pname, addr, loc, scale,
        taken::Set{Symbol})
    nm = horseshoe_normal_name(pname, addr)
    nm in taken && _sfail(
        "horseshoe over $pname: synthesized $nm collides with a " *
        "model name — rename yours")
    push!(taken, nm)
    return SampledParameter(nm, :normal, (arg1 = loc, arg2 = scale),
        nothing, nm)
end

# R2D2 declarations to IR: one R2D2Prior per declared predictor.
# Stated Normal coefficient priors become share-0 overrides (scalar
# via _coefficient_normal, factor blocks via the broadcast form);
# unstated columns join the simplex (factors take a full-cover
# LevelMap — the identified check fires exactly when that collides
# with an intercept, same as the PopulationPrior path). Omitted tau
# synthesizes a half-standard-Normal parameter.
function _lower_r2d2_priors(decls, sample, coefuse, predictors, levelmaps,
        taken, matrices::Dict{Symbol,DesignMatrix})
    stated = Dict{Symbol,Any}()
    for s in sample
        haskey(coefuse, s.lhs) && (stated[s.lhs] = s)
    end
    by_pred = Dict{Symbol,PredictorSpec}(p.name => p for p in predictors)
    seen = Set{Symbol}()
    out = R2D2Prior[]
    taus = SampledParameter[]
    r2d2preds = PredictorSpec[]
    for d in decls
        haskey(by_pred, d.predictor) || _sfail(
            "r2d2 over unknown predictor $(d.predictor) " *
            "(predictors come from response linear predictors)")
        d.predictor in seen && _sfail(
            "duplicate r2d2 declaration for predictor $(d.predictor) " *
            "(one per predictor)")
        push!(seen, d.predictor)
        pred = by_pred[d.predictor]
        push!(r2d2preds, pred)
        overrides = Dict{Symbol,Tuple{Float64,Float64}}()
        for t in pred.terms
            (t.kind === OffsetTerm || t.kind === LatentTerm ||
                t.kind === VaryingEffectTerm ||
                t.kind === SplineSummandTerm ||
                t.kind === HSGPSummandTerm ||
                t.kind === ScanSummandTerm ||
                t.kind === MonotonicSummandTerm) && continue
            if t.kind === MatrixTerm
                for (e, ov) in _lower_r2d2_matrix(pred, t, coefuse,
                        stated, matrices)
                    overrides[e] = ov
                end
                continue
            end
            addr = t.kind === InterceptTerm ? :Intercept : only(t.columns)
            use = _find_use(coefuse, pred.name, addr)
            use === nothing && _sfail("internal: no coefficient use for " *
                                      "($(pred.name), $addr)")
            t.kind === MonotonicTerm && _sfail(
                "r2d2 over predictor $(pred.name): monotonic columns " *
                "are not in the flat slice (the mo contrast is " *
                "parameter-derived, so no data variance exists)")
            name = use[1]
            sign = use[3]
            if t.kind === FactorTerm
                ov = _lower_r2d2_factor(pred, t, name, sign, stated,
                    levelmaps)
                ov === nothing || (overrides[addr] = ov)
                continue
            end
            haskey(stated, name) || continue
            s = stated[name]
            s.levels !== nothing && _sfail("coefficient $name takes a " *
                                           "scalar prior (`$name ~ Normal`), " *
                                           "not a levels prior — it is used " *
                                           "as $(t.kind), not a factor")
            loc, scale = _coefficient_normal(name, s.rhs, pred.name, addr)
            overrides[addr] = (sign * loc, scale)
        end
        tau = d.tau
        if tau === nothing
            tau = Symbol(:r2d2_, d.predictor, :_tau_bsv)
            tau in taken && _sfail(
                "r2d2 over $(d.predictor): synthesized tau $tau " *
                "collides with a model name — pass tau explicitly " *
                "(`r2d2($(d.predictor), $(d.r2), $(d.phi), mytau)` " *
                "with `mytau ~ HalfNormal(1)`)")
            push!(taus, SampledParameter(tau, :normal, (arg1 = 0, arg2 = 1),
                :positive, tau))
        end
        push!(out, R2D2Prior(d.predictor, d.r2, d.phi, tau, overrides))
    end
    _check_identified(r2d2preds, levelmaps, matrices)
    return out, taus
end

# An R2D2 matrix: a stated broadcast prior becomes per-element share-0
# overrides; an unstated vector joins the simplex (the intercept
# element takes share 0 by default, as on the scalar path).
function _lower_r2d2_matrix(pred, t, coefuse, stated, matrices)
    X = t.options.matrix
    m = get(matrices, X, nothing)
    m === nothing && _sfail("internal: matrix term over unknown matrix $X")
    use = _find_use(coefuse, pred.name, X)
    use === nothing && _sfail("internal: no coefficient use for " *
                              "($(pred.name), $X)")
    name, _, sign = use
    elems = _matrix_element_addressees(m)
    K = length(elems)
    haskey(stated, name) || return Tuple{Symbol,Tuple{Float64,Float64}}[]
    s = stated[name]
    s.matrix === nothing && _sfail("internal: matrix prior for $name " *
                                   "lost its sizing matrix")
    locs, scales = _coefficient_matrix_normal(name, s.rhs, pred.name, K)
    return [(e, (sign * l, sc)) for (e, l, sc) in zip(elems, locs, scales)]
end

# An R2D2 factor: a stated broadcast prior becomes a share-0 override
# (with its levels subset, as on the PopulationPrior path); an
# unstated factor joins the simplex under a full-cover LevelMap.
function _lower_r2d2_factor(pred, t, name, sign, stated, levelmaps)
    col = only(t.columns)
    haskey(stated, name) || begin
        push!(levelmaps, LevelMap(pred.name, col, [], :levels, :))
        return nothing
    end
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
    return (sign * loc, scale)
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
# Matrix intercepts count (any intercept position in any matrix term).
function _check_identified(predictors, levelmaps, matrices)
    for pred in predictors
        has_intercept = any(t -> t.kind === InterceptTerm, pred.terms)
        if !has_intercept
            for t in pred.terms
                t.kind === MatrixTerm || continue
                m = get(matrices, t.options.matrix, nothing)
                m !== nothing && any(isnothing, m.columns) &&
                    (has_intercept = true; break)
            end
        end
        has_intercept || continue
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

function _lower_parameters(sample, coefuse, ctx, glmuse)
    params = SampledParameter[]
    syms = Set{Symbol}()
    vectors = VectorParameter[]
    for s in sample
        s.lhs in ctx.data && continue
        if _is_dirichlet_call(s.rhs)
            haskey(coefuse, s.lhs) && _sfail(
                "$(s.lhs) is a predictor coefficient and cannot also be " *
                "a Dirichlet parameter")
            push!(vectors, _lower_dirichlet(s.lhs, s.rhs))
            continue
        end
        if _is_lkj_factor_call(s.rhs)
            haskey(coefuse, s.lhs) && _sfail(
                "$(s.lhs) is a predictor coefficient and cannot also be " *
                "an LKJCovarianceFactor")
            sc, cr = _lower_lkj_factor(s.lhs, s.rhs, coefuse, ctx)
            push!(vectors, sc)
            push!(vectors, cr)
            sc.args.arg1 isa Symbol && push!(syms, sc.args.arg1)
            continue
        end
        haskey(coefuse, s.lhs) && continue
        haskey(glmuse, s.lhs) && continue
        s.levels !== nothing && _sfail("levels prior `$(s.lhs)[...]` is " *
                                       "never used in a predictor — size " *
                                       "only vectors the model indexes")
        s.matrix !== nothing && _sfail("matrix prior `$(s.lhs)[...]` is " *
                                       "never used in a predictor — use " *
                                       "it in a matmul (`mu = X * " *
                                       "$(s.lhs)`) or drop it")
        p = _lower_parameter(s.lhs, s.rhs, coefuse, ctx.matrices)
        push!(params, p)
        for v in values(p.args)
            v isa Symbol && push!(syms, v)
        end
    end
    return params, syms, vectors
end

_is_dirichlet_call(rhs) =
    rhs isa Expr && rhs.head === :call && !isempty(rhs.args) &&
    rhs.args[1] === :Dirichlet

# Simplex parameter: `s ~ Dirichlet(alpha)` (a literal concentration
# vector) or symmetric `s ~ Dirichlet(K, a)` — SB's two constructor
# forms, resolved here to a frozen concentration vector (concentrations
# are hyperparameters: data columns and sampled/model-dependent args
# fail closed — planned).
function _lower_dirichlet(lhs, rhs)
    args = _plain_args(rhs, "`Dirichlet`")
    alpha = if length(args) == 1
        a = only(args)
        a isa Expr && a.head === :vect && !isempty(a.args) &&
            all(x -> x isa Real, a.args) ||
            _sfail("parameter $lhs: `Dirichlet(alpha)` takes a literal " *
                   "concentration vector (`Dirichlet([1.0, 2.0])`) or " *
                   "symmetric `Dirichlet(K, a)`, got $(repr(a))")
        Vector{Float64}(a.args)
    elseif length(args) == 2
        K, a = args
        (K isa Integer && K >= 1 && a isa Real) ||
            _sfail("parameter $lhs: symmetric `Dirichlet(K, a)` takes a " *
                   "positive integer dimension and a real concentration, got " *
                   "($(repr(K)), $(repr(a)))")
        fill(Float64(a), Int(K))
    else
        _sfail("parameter $lhs: `Dirichlet` takes a concentration vector " *
               "`Dirichlet(alpha)` or symmetric `Dirichlet(K, a)`, got " *
               "$(length(args)) arguments")
    end
    return VectorParameter(lhs, :simplex_dirichlet, (arg1 = alpha,), nothing,
        lhs)
end

_is_lkj_factor_call(rhs) =
    rhs isa Expr && rhs.head === :call && !isempty(rhs.args) &&
    rhs.args[1] === :LKJCovarianceFactor

# SB's derived factor-piece names for a stem `L`: `L_scales` (the positive
# scale vector) and `L_L_corr` (the LKJ Cholesky factor). Single source for
# the factor allocator and the joint-response linker.
_lkj_factor_names(stem::Symbol) =
    (Symbol(stem, :_scales), Symbol(stem, :_L_corr))

# SB's covariance-factor declaration, decomposed:
# `L ~ LKJCovarianceFactor(K, Exponential(θ), eta)` allocates the factor's
# two plan nodes — the positive scales vector and the LKJ Cholesky factor
# (SB's `target_scales` / `target_L_corr`) — which the joint response
# links explicitly. The `L` factor itself materializes in-graph as
# `diag_pre_multiply(scales, L_corr)`; the stem binds no plan node.
# Scale priors are Exponential-only in this slice (SB's default;
# sampled-θ hyperparameters ride the scalar-prior shape).
function _lower_lkj_factor(lhs, rhs, coefuse, ctx)
    args = _plain_args(rhs, "`LKJCovarianceFactor`")
    length(args) == 3 || _sfail("parameter $lhs: " *
                                "`LKJCovarianceFactor` takes (K, scale prior, " *
                                "shape) " *
                                "(`L ~ LKJCovarianceFactor(2, Exponential(1.0), 2.0)`), " *
                                "got $(length(args)) arguments")
    K, prior, shape = args
    (K isa Integer && !(K isa Bool) && K >= 1) ||
        _sfail("parameter $lhs: `LKJCovarianceFactor` needs an integer " *
               "dimension K ≥ 1, got $(repr(K))")
    prior isa Expr && prior.head === :call && !isempty(prior.args) &&
        prior.args[1] === :Exponential ||
        _sfail("parameter $lhs: joint-factor scale prior is " *
               "`Exponential(θ)` in this slice (SB's default), got " *
               "$(repr(prior))")
    pargs = _plain_args(prior, "`Exponential`")
    length(pargs) == 1 || _sfail("parameter $lhs: `Exponential` takes " *
                                 "exactly the scale")
    theta = _lower_param_arg(lhs, only(pargs), coefuse, ctx.matrices)
    (shape isa Real && isfinite(shape) && shape > 0) ||
        _sfail("parameter $lhs: LKJ shape must be a finite positive " *
               "literal (a hyperparameter), got $(repr(shape))")
    scales, corr = _lkj_factor_names(lhs)
    for nm in (scales, corr)
        nm in ctx.taken && _sfail(
            "implicit factor piece $nm for $lhs collides " *
            "with your definition — rename yours")
        push!(ctx.taken, nm)
    end
    Ki = Int(K)
    return (VectorParameter(scales, :positive_exponential, (arg1 = theta,),
            Ki, lhs),
        VectorParameter(corr, :cholesky_corr_lkj, (arg1 = Float64(shape),),
            Ki, lhs))
end

function _lower_parameter(lhs, rhs, coefuse, matrices)
    rhs isa Expr && rhs.head === :call && !isempty(rhs.args) || _sfail(
        "parameter $lhs needs a distribution call, got $(repr(rhs))")
    fam = rhs.args[1]
    fam === :Flat && return _lower_flat(lhs, rhs)
    fam === :flat && _sfail("parameter $lhs: use `Flat()` (Turing-style " *
                            "improper uniform), not Stan-style `flat()`")
    fam === :truncated && return _lower_truncated_param(lhs, rhs, coefuse,
        matrices)
    fam in (:HalfNormal, :HalfCauchy) &&
        return _lower_half_param(lhs, rhs, fam, coefuse, matrices)
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
               "Dirichlet, LKJCovarianceFactor, truncated). If `$fam` is " *
               "meant as a submodel, define it with " *
               "`@rkppl $fam(args...) = begin ... end` and " *
               "make it visible in the lowering module (`mod=`).")
    args = _plain_args(rhs, "`$fam`")
    vals = [_lower_param_arg(lhs, a, coefuse, matrices) for a in args]
    argkeys = ntuple(i -> Symbol(:arg, i), length(vals))
    return SampledParameter(lhs, _PARAM_FAMILIES[fam],
        NamedTuple{argkeys}(Tuple(vals)), nothing, lhs)
end

function _lower_param_arg(lhs, a, coefuse, matrices)
    a isa Real && return a
    a === :Inf && return Inf
    a isa Symbol || _sfail("parameter $lhs argument $(repr(a)) must be a " *
                           "literal or a parameter/assignment name (bind " *
                           "expressions via an assignment first)")
    haskey(coefuse, a) && _sfail("$a is a predictor coefficient and cannot " *
                                 "also be a parameter argument (parameter $lhs)")
    haskey(matrices, a) && _sfail("parameter $lhs argument $a is a design " *
                                  "matrix — prior arguments are scalar " *
                                  "(a literal or a parameter/assignment name)")
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
function _lower_half_param(lhs, rhs, fam, coefuse, matrices)
    args = _plain_args(rhs, "`$fam`")
    length(args) == 1 ||
        _sfail("parameter $lhs: `$fam` takes exactly the scale")
    scale = _lower_param_arg(lhs, args[1], coefuse, matrices)
    base = fam === :HalfNormal ? :normal : :cauchy
    return SampledParameter(lhs, base, (arg1 = 0, arg2 = scale), :positive,
        lhs)
end

# A literal truncation bound: `Inf`/`-Inf` (as the `:Inf` symbol, a signed
# `:Inf` call (`-Inf`/`+Inf` AST), or a Real infinity) or a finite Real; a
# name/expression is rejected (bounds are constant, independent of the cell
# position).
function _truncation_bound(lhs, b)
    b === :Inf && return Inf
    b isa Real && return Float64(b)
    if b isa Expr && b.head === :call && length(b.args) == 2 && b.args[2] === :Inf
        b.args[1] === :- && return -Inf
        b.args[1] === :+ && return Inf
    end
    return _sfail(
        "parameter $lhs: truncation bounds must be literals " *
        "(a finite Real or ±Inf), got $(repr(b))")
end

# Parameter truncation lowers to a support override: `truncated(Normal(0, s), 0,
# Inf)` (a half at a literal-zero location) → `:positive` (exact +log(2)); an
# upper-only `truncated(Normal(mu, s), -Inf, hi)` → `(:upper, hi)` (Stan's
# upper-bound kernel `x = hi - exp(u)`, bare-`u` Jacobian, NO truncation
# renormalizer — Normal-only, any location); and a two-sided FINITE
# `truncated(Normal(mu, s), lo, hi)` → `(:interval, lo, hi)` (an
# affine-logistic constrained transform with the exact -log(cdf(hi)-cdf(lo))
# renormalization; Normal-only, any location). Bounds are literals.
function _lower_truncated_param(lhs, rhs, coefuse, matrices)
    args = _plain_args(rhs, "`truncated`")
    length(args) == 3 || _sfail("parameter $lhs: use the Distributions.jl " *
                                "object form `truncated(Normal(mu, s), lo, hi)`")
    obj, lo_a, hi_a = args
    obj isa Expr && obj.head === :call || _sfail(
        "parameter $lhs: `truncated` wraps a distribution object, got " *
        "$(repr(obj))")
    fam = obj.args[1]
    fam in (:Normal, :Cauchy) || _sfail(
        "parameter $lhs: slice-1 truncation wraps Normal/Cauchy " *
        "(`truncated(Normal(mu, s), lo, hi)`); got $fam")
    oargs = _plain_args(obj, "`$fam`")
    length(oargs) == 2 || _sfail("parameter $lhs: `$fam` takes two arguments")
    vals = [_lower_param_arg(lhs, a, coefuse, matrices) for a in oargs]
    base = _PARAM_FAMILIES[fam]
    lo = _truncation_bound(lhs, lo_a)
    hi = _truncation_bound(lhs, hi_a)
    lo < hi || _sfail("parameter $lhs: truncation needs lower < upper, " *
                      "got ($lo, $hi)")
    # Half-truncation [0, Inf) at a literal-zero location → the exact +log(2) case.
    if lo == 0 && isinf(hi) && hi > 0
        (oargs[1] isa Real && oargs[1] == 0) || _sfail(
            "parameter $lhs: a `truncated(_, 0, Inf)` half needs a literal zero " *
            "location (use `HalfNormal(s)`); a non-zero location needs finite " *
            "bounds (`truncated(Normal(mu, s), lo, hi)`)")
        return SampledParameter(lhs, base,
            (arg1 = vals[1], arg2 = vals[2]), :positive, lhs)
    end
    # Upper-only truncation → Stan's upper-bound kernel (Normal-only in
    # slice 1). A finite LOWER-only bound stays rejected (no one-sided
    # lower support exists on the scalar path yet).
    if isinf(lo) && lo < 0 && isfinite(hi)
        fam === :Normal || _sfail(
            "parameter $lhs: an upper-only truncation is Normal-only in slice 1 " *
            "(`truncated(Normal(mu, s), -Inf, hi)`); got $fam")
        return SampledParameter(lhs, base,
            (arg1 = vals[1], arg2 = vals[2]), (:upper, hi), lhs)
    end
    # Two-sided FINITE interval → affine-logistic transform + truncated-Normal
    # renormalization (Normal-only in slice 1).
    (isfinite(lo) && isfinite(hi)) || _sfail(
        "parameter $lhs: one-sided truncation is slice-1 only as `[0, Inf)` at a " *
        "zero location (`HalfNormal(s)`) or upper-only " *
        "(`truncated(Normal(mu, s), -Inf, hi)`); use finite bounds " *
        "(`truncated(Normal(mu, s), lo, hi)`) otherwise")
    fam === :Normal || _sfail(
        "parameter $lhs: a finite truncated interval is Normal-only in slice 1 " *
        "(`truncated(Normal(mu, s), lo, hi)`); got $fam")
    return SampledParameter(lhs, base,
        (arg1 = vals[1], arg2 = vals[2]), (:interval, lo, hi), lhs)
end

function _lower_assignment(nm, rhs, coefuse)
    _contains_spline(rhs) && _sfail("assignment `$nm` calls `spline()`, " *
        "which lowers only as a direct predictor summand " *
        "(`mu = a .+ b .* x .+ spline(:s_x)`), not inside definitions")
    _contains_hsgp(rhs) && _sfail("assignment `$nm` calls `hsgp()`, " *
        "which lowers only as a direct predictor summand " *
        "(`mu = a .+ b .* x .+ hsgp(:h_x)`), not inside definitions")
    _contains_mo(rhs) && _sfail("assignment `$nm` calls `mo()`, " *
        "which lowers only in a predictor (`mu = a .+ b .* mo(c, s)`), " *
        "not inside definitions")
    _contains_mo1(rhs) && _sfail("assignment `$nm` calls `mo1()`, " *
        "which lowers only as a direct predictor summand " *
        "(`mu = a .+ mo1(c, s)`), not inside definitions")
    _contains_dar(rhs) && _sfail("assignment `$nm` calls `dar()`, " *
        "which lowers only as a direct predictor summand " *
        "(`mu = a .+ dar(beta, sigma)`), not inside definitions")
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
    _contains_spline(rhs) && _sfail("derived column `$nm` calls `spline()`, " *
        "which lowers only as a direct predictor summand " *
        "(`mu = a .+ b .* x .+ spline(:s_x)`), not inside definitions")
    _contains_hsgp(rhs) && _sfail("derived column `$nm` calls `hsgp()`, " *
        "which lowers only as a direct predictor summand " *
        "(`mu = a .+ b .* x .+ hsgp(:h_x)`), not inside definitions")
    _contains_mo(rhs) && _sfail("derived column `$nm` calls `mo()`, " *
        "which lowers only in a predictor (`mu = a .+ b .* mo(c, s)`), " *
        "not inside definitions")
    _contains_mo1(rhs) && _sfail("derived column `$nm` calls `mo1()`, " *
        "which lowers only as a direct predictor summand " *
        "(`mu = a .+ mo1(c, s)`), not inside definitions")
    _contains_dar(rhs) && _sfail("derived column `$nm` calls `dar()`, " *
        "which lowers only as a direct predictor summand " *
        "(`mu = a .+ dar(beta, sigma)`), not inside definitions")
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
