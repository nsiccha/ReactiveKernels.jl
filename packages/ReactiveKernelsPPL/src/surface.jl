# `@rkppl` authoring surface: StanBlocks-close model blocks lowering to
# data-free StructuralPlans.
#
# The shape mirrors StanBlocks `@slic` (block AST capture, `~` density
# statements, deterministic `=`, `model(; data...)` binding) under the
# standing constraints: Distributions.jl constructors (never Stan lowercase),
# immutable single-assignment top level, no control flow, no `target`,
# `@plate for i in R` cells that mean one iteration of that Julia loop
# (observations, per-cell latents, per-cell submodels, cell locals; the
# desugar evaluates every iteration at once), `@scan` sequential
# recurrences. Broadcasting is EXPLICIT at top level (no implied
# vectorization; a plate cell is scalar Julia, so its values need no dots):
# vector math is dotted (`mu = a .+ b .* x` —
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
    SurfaceLoweringError(message::AbstractString) =
        new(_author_names(message))
end
Base.showerror(io::IO, e::SurfaceLoweringError) =
    print(io, "SurfaceLoweringError: ", e.message)

_sfail(msg) = throw(SurfaceLoweringError(msg))

"""
    RKPPLModel

A captured `@rkppl` block. `model(; x, g)` binds inputs and pins any named
declaration. Observe values explicitly: `model(; x, g) | (; y)`, or
`condition(model(; x, g); y)`, returns a bound `StructuralPlan`.
`fixed` stores merge pins, `rewrites` stores scoped statement replacements,
and `conditioned` stores explicit observations. Operations return new models.
"""
struct RKPPLModel
    ast::Expr
    mod::Module
    fixed::Dict{Symbol,ColumnData}
    rewrites::Vector{Expr}
    conditioned::Dict{Symbol,ColumnData}
end
RKPPLModel(ast, mod) = RKPPLModel(ast, mod, Dict{Symbol,ColumnData}(), Expr[], Dict{Symbol,ColumnData}())
RKPPLModel(ast, mod, fixed) = RKPPLModel(ast, mod, fixed, Expr[])
RKPPLModel(ast, mod, fixed, rewrites) = RKPPLModel(ast, mod, fixed, rewrites, Dict{Symbol,ColumnData}())

"""Inputs bound to a captured program, awaiting explicit observations.
Use `binding | (; y)` or `condition(binding; y)` to lower and bind it."""
struct RKPPLBoundModel
    model::RKPPLModel
    data::Dict{Symbol,ColumnData}
end

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
`_expand_submodels`): the submodel owns a lexical namespace under the LHS
and its mathematics is spliced into the parent plan, so a submodel lowers
like a hand-inlined model — transparent and reusable, never an opaque node.

A body holds any statement a top-level program can: `~` / `.~` / `=`
statements, indexed and sized priors (`c[levels(g)] .~ Normal.(0, s)`,
`b[axes(X, 2)] .~ …`), `@plate` and `@scan` blocks,
and calls to other submodels. Scope access is static:

- every name the body binds — a `~` / `.~` / `=` left-hand side (the base
  name of an indexed one), a `@plate` result, cell or loop variable, a
  `@scan` carried state, step local or loop variable, a `do`-block argument
  — receives a private identifier; its author path is retained separately.
  `z.b` reads local `b` of call `z`. Index expressions keep their shape;
- each argument is replaced by the call's argument expression; a body may
  observe (`~` / `.~`) an argument bound to a data column, and binding an
  argument name any other way fails;
- function names, keyword names and every name the body does not bind are
  left as written (a free name refers to the calling program);
- a nested call expands after the enclosing body is substituted, so paths
  compose (`z ~ outer(…)` → `w ~ inner(…)` → `b` gives `z.w.b`); it resolves
  in the module that defined the enclosing submodel. Recursion fails.

Bare `z` denotes the actual return value, including in Julia function
arguments. `z.b` is local-first: when `b` is absent from the scope it reads
`getproperty(z, :b)` on that return value. Explicit `getproperty(z, :b)`
always reads the return value. Ordinary caller assignments cannot write locals;
explicit `merge` operations replace their declarations. Locals
never create unqualified caller bindings. `a.b_c`, `a_b.c`, and caller
`a_b_c` coexist. A per-cell call (`col[i] ~ sm(…)` inside `@plate for i …`)
admits scalar `~` / `=` statements over bare names; `col[i].b` reads local
`b` at cell `i`, and `col.b` is the array of those locals. Loop variables
remain scoped to their loop and cannot be read outside it.

[`constrain`](@ref) and [`restore_draws`](@ref) return nested NamedTuples
of sampled locals (`nt.z.b`, `nt.z.w.b`); [`coordinate_names`](@ref) uses
dotted author paths. These draw containers omit deterministic locals,
observed slots and return values. [`unconstrain`](@ref) accepts the same
nested containers. Rebuild prepared layouts and draw mappings when
migrating from flattened submodel names.

A fused stream def (design + coefficients inside, Stan
`bernoulli_logit_glm`-style) keeps the response shell but NOT the predictor
name: its affine local belongs to the data LHS's scope (`y.mu`), and an inline
compound location synthesizes (`y_eta`) — both move the lowered predictor.
Ordinary parameter names follow their declarations; legacy construct-owned
coefficient blocks follow the predictor. To factor design + coefficients into a def
WITHOUT moving names, return the affine from a latent def and bind it at a
named use site (`mu ~ affine_def(X, b)`, then `y .~ family.(...mu...)`): the
use-site LHS names the predictor, and the plan is identical to the
hand-written decomposed program.
"""
struct RKPPLSubmodel
    name::Symbol
    argnames::Vector{Symbol}
    body::Expr
    mod::Module
    kwdefaults::Vector{Pair{Symbol,Any}}
    rewrites::Vector{Expr}
    fixed::Dict{Symbol,ColumnData}
end
RKPPLSubmodel(name, args, body, mod) =
    RKPPLSubmodel(name, args, body, mod, Pair{Symbol,Any}[])
RKPPLSubmodel(name, args, body, mod, kwdefaults) =
    RKPPLSubmodel(name, args, body, mod, kwdefaults, Expr[], Dict{Symbol,ColumnData}())
_submodel_args(sm::RKPPLSubmodel) = [sm.argnames; first.(sm.kwdefaults)]
sampling_fragment(sm::RKPPLSubmodel) = sm

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
    argnames = Any[]
    kwdefaults = Pair{Symbol,Any}[]
    for a in call.args[2:end]
        if Meta.isexpr(a, :parameters)
            for kw in a.args
                Meta.isexpr(kw, :kw, 2) && kw.args[1] isa Symbol ||
                    _sfail("submodel `$name` keyword arguments need named defaults")
                push!(kwdefaults, kw.args[1] => kw.args[2])
            end
        else
            push!(argnames, a)
        end
    end
    for a in argnames
        a isa Symbol || _sfail("submodel `$name` positional arguments must " *
                               "be bare Symbols, got $(repr(a))")
    end
    allnames = [argnames; first.(kwdefaults)]
    length(unique(allnames)) == length(allnames) ||
        _sfail("submodel `$name` has duplicate argument names")
    body isa Expr && body.head === :block ||
        _sfail("submodel `$name` body must be a `begin ... end` block")
    argvec = Expr(:vect, [QuoteNode(a) for a in argnames]...)
    kwvec = Expr(:vect, [:( $(QuoteNode(k)) => $(Meta.quot(v)) )
        for (k, v) in kwdefaults]...)
    return esc(:($name = $(RKPPLSubmodel)($(QuoteNode(name)), $argvec,
                                          $(Meta.quot(body)), $mod, $kwvec)))
end

"""Capture a model block, or define a reusable submodel
(`@rkppl sm(args...) = begin ... end`; see [`RKPPLSubmodel`](@ref)).

Design-matrix vocabulary (standard-Julia value semantics throughout):
bind the matrix once (`X = hcat(ones(length(x1)), x1, x2)` — the intercept `ones(length(x))` plus
bare data/derived columns), use it only as a predictor matmul
(`mu = X * b`), and size the coefficient vector with an axes prior
(`b[axes(X, 2)] .~ Normal.(loc, scale)` — scalar args share over
elements, literal `[…]` vectors go per element). Declarations are
strict: an undeclared coefficient name or vector fails naming the
prior to write — nothing is defaulted. Every other matrix position (scales, prior
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

"""Capture a model block and bind caller-scope inputs and pins
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
    idx, _, blocked = _merge_base_index(m.ast.args)
    fixes = (; (k => v for (k, v) in kwargs if haskey(idx, k) ||
        k in blocked || occursin('.', String(k)))...)
    program = isempty(fixes) ? m : Base.merge(m, fixes)
    binding = RKPPLBoundModel(program, cols)
    return isempty(program.conditioned) ? binding : bind_data(binding)
end

function condition(m::RKPPLModel; kwargs...)
    observations = merge(copy(m.conditioned),
        Dict{Symbol,ColumnData}(k => _check_col(k, v) for (k, v) in kwargs))
    return RKPPLModel(m.ast, m.mod, copy(m.fixed), copy(m.rewrites), observations)
end
condition(m::RKPPLBoundModel; kwargs...) =
    bind_data(RKPPLBoundModel(condition(m.model; kwargs...), copy(m.data)))
Base.:|(m::Union{RKPPLModel,RKPPLBoundModel}, data::NamedTuple) = condition(m; data...)
function condition(plan::StructuralPlan; kwargs...)
    aliases = Dict(Symbol(join(path, ".")) => nm for (nm, path) in
        _scope_name_paths(plan.submodel_scopes))
    names = Set(p.name for ps in (plan.parameters, plan.array_parameters,
        plan.vector_parameters, plan.plate_parameters) for p in ps)
    computed = _bound_module_data_names(plan)
    union!(computed, _derived_response_names(plan))
    columns = Dict{Symbol,ColumnData}(k => v for (k,v) in plan.columns if k ∉ computed)
    # Bound response storage is numeric. Restore its host-side missing entries
    # before rebinding so replacement data gets a fresh presence mask, while
    # responses left unchanged keep their original absence pattern.
    responses = Set(r.response for r in plan.responses)
    union!(responses, (s.array for s in _observed_selections(plan, plan.columns)))
    for name in responses
        maskname = _observed_mask_name(name)
        haskey(plan.columns, maskname) || continue
        if haskey(columns, name)
            value, present = columns[name], plan.columns[maskname]
            restored = Array{Union{Missing,eltype(value)}}(undef, size(value))
            map!((v, p) -> p ? v : missing, restored, value, present)
            columns[name] = restored
        end
        delete!(columns, maskname)
    end
    observing = copy(plan.conditioned)
    for (key, value) in kwargs
        name = get(aliases, key, key)
        if name in names
            input = _conditioned_input(name)
            name ∉ observing && (haskey(columns, input) || input in names ||
                any(a -> a.name === input, plan.assignments) ||
                any(d -> d.name === input, plan.derived)) &&
                _sfail("condition `$key` needs internal input `$input`, already used by the model")
            push!(observing, name)
            columns[input] = _check_col(key, value)
        elseif haskey(columns, name)
            columns[name] = _check_col(key, value)
        else
            _sfail("condition `$key` matches no sampling statement")
        end
    end
    return bind_data(plan, columns; conditioned = observing)
end
Base.:|(plan::StructuralPlan, data::NamedTuple) = condition(plan; data...)
bind_data(m::RKPPLBoundModel) = _bind_model(m.model, m.data)
build_kernel(m::RKPPLBoundModel) = build_kernel(bind_data(m))
build_kernel(m::RKPPLModel) = build_kernel(m())

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
    return m(; cols...)
end

_dict_key(k::Symbol) = k
_dict_key(k::AbstractString) = Symbol(k)
_dict_key(k) = _sfail("data column keys must be Symbols, got $(repr(k))")

_check_col(k, v) = v isa _SuppliedColumn ? v :
    _sfail("data value $k must be a number or an array, got $(typeof(v))")

function _bind_model(m::RKPPLModel, cols::Dict{Symbol,ColumnData})
    plan, values = _lower_model(m, cols)
    return bind_data(plan, values)
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

NamedTuple fix (a pin): each `name = value` removes the matching base
statement and stores `value` — a number or an array of any shape — as model
data, bound at the call; explicit call kwargs win over it (SB
easily-rebound data). A pinned number reads as a model-level value
(`merge(m, (; tau = 0.3))` reads `tau` exactly as `tau = 0.3` would).

Sized left-hand sides match their complete declaration, including its axes.
A bare name replaces the whole declaration. Scoped names use author paths
(`z.w.tau`); keyword pins use `var"z.w.tau" = value`. `merge(submodel, …)`
derives a reusable variant, including replacements and pins in nested calls.
Partial array writes, duplicate matches, and unknown pins fail explicitly.
Joint responses match their whole vector left-hand side; pins must supply
every member. Members cannot be rewritten separately;
`@plate` / `@scan` cells are invisible to the top-level matcher, so a
colliding append fails at lowering through the single-assignment gate.
"""
function Base.merge(m::RKPPLModel, override::Expr)
    out = Any[a for a in m.ast.args]
    idx, dups, blocked = _merge_base_index(out)
    rewrites = copy(m.rewrites)
    fixed = copy(m.fixed)
    for raw in _merge_override_stmts(override)
        idx, dups, blocked = _merge_base_index(out)
        st = _merge_unwrap_override(raw)
        lhs = _merge_override_lhs(st)
        if lhs isa Expr && lhs.head === :vect
            matches = findall(out) do a
                a isa Expr && !_is_block_macro(a) || return false
                st = _unwrap_trivia(a)
                return (_is_sample(st) || _is_broadcast_sample(st)) &&
                    isequal(_stmt_lhs(st), lhs)
            end
            length(matches) <= 1 || _sfail("merge joint target matches multiple statements")
            isempty(matches) ? push!(out, raw) : (out[only(matches)] = raw)
            foreach(n -> delete!(fixed, n), lhs.args)
            continue
        end
        if _merge_path(lhs) !== nothing
            push!(rewrites, st)
            delete!(fixed, Symbol(join(_merge_path(lhs), ".")))
            continue
        end
        nm = _merge_stem(lhs)
        nm in dups && _sfail("merge override `$lhs` matches more than " *
                              "one base-model statement (the base is " *
                              "broken — lowering would reject it)")
        nm in blocked && _sfail("merge override `$lhs` names one member of a joint " *
                                 "statement; replace its complete left-hand side")
        if haskey(idx, nm)
            _merge_check_lhs(lhs, _stmt_lhs(_merge_unwrap_override(out[idx[nm]])))
            out[idx[nm]] = raw
        else
            lhs isa Symbol || _sfail("merge indexed override `$lhs` matches no declaration")
            push!(out, raw)
        end
        delete!(fixed, nm)
    end
    return RKPPLModel(Expr(:block, out...), m.mod, fixed, rewrites, copy(m.conditioned))
end

Base.merge(m::RKPPLModel, fix::NamedTuple) = _merge_fixes(m, fix)

# An author's dotted key names a scoped declaration (`var"z.w.tau"`) and waits
# for expansion. Expansion pins a local by its plan identifier, which is
# spelled with the same dots, so it passes `scoped_keys = false`.
function _merge_fixes(m::RKPPLModel, fix::NamedTuple; scoped_keys::Bool = true)
    isempty(fix) && _sfail("merge with an empty NamedTuple fixes nothing")
    out = Any[a for a in m.ast.args]
    idx, dups, blocked = _merge_base_index(out)
    new_fixed = copy(m.fixed)
    observations = copy(m.conditioned)
    drop = Set{Int}()
    for (nm, val) in pairs(fix)
        delete!(observations, nm)
        if scoped_keys && occursin('.', String(nm))
            new_fixed[nm] = _check_col(nm, val)
            continue
        end
        nm in dups && _sfail("merge fix `$nm` matches more than one " *
                             "base-model statement (the base is broken — " *
                             "lowering would reject it)")
        if nm in blocked
            for (i, raw) in pairs(out)
                raw isa Expr && !_is_block_macro(raw) || continue
                st = _unwrap_trivia(raw)
                (_is_sample(st) || _is_broadcast_sample(st)) || continue
                lhs = _stmt_lhs(st)
                lhs isa Expr && lhs.head === :vect && nm in lhs.args || continue
                all(n -> haskey(fix, n), lhs.args) ||
                    _sfail("merge pin of a joint response must supply every member of $(repr(lhs))")
                push!(drop, i)
            end
            new_fixed[nm] = _check_col(nm, val)
            continue
        end
        (haskey(idx, nm) || haskey(new_fixed, nm)) || _sfail("merge fix `$nm` matches no base-model " *
                                  "statement (a fixed name must name a " *
                                  "`~` / `.~` / `=` statement to remove)")
        val isa _SuppliedColumn || _sfail("merge fix `$nm` must be a " *
            "number or an array (a data value; got $(typeof(val)))")
        haskey(idx, nm) && push!(drop, idx[nm])
        new_fixed[nm] = val
    end
    kept = Any[a for (i, a) in enumerate(out) if i ∉ drop]
    return RKPPLModel(Expr(:block, kept...), m.mod, new_fixed, copy(m.rewrites), observations)
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
# `dups` (broken-base duplicates), `blocked` (joint outcomes).
# Plate/scan blocks and unparseable statements
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
    stem = _merge_stem(lhs)
    if stem isa Symbol
        _merge_claim!(idx, dups, stem, i)
    elseif lhs.head === :vect
        for o in lhs.args
            o isa Symbol && push!(blocked, o)
        end
    end
    return nothing
end

function _merge_override_stmts(override::Expr)
    override.head === :block || return Any[override]
    return Any[b for a in override.args if !(a isa LineNumberNode)
        for b in (a isa Expr ? _merge_override_stmts(a) : Any[a])]
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
    ok = _is_sample(st) || _is_broadcast_sample(st) ||
        (st.head === :(=) && length(st.args) == 2)
    ok || _sfail("merge override must be a `~`, `.~` or `name = ...` " *
                 "statement with a bare Symbol LHS, got $(repr(st))")
    lhs = _stmt_lhs(st)
    lhs isa Expr && lhs.head === :vect && all(a -> a isa Symbol, lhs.args) && return lhs
    (_merge_stem(lhs) isa Symbol || _merge_path(lhs) !== nothing) ||
        _sfail("merge override must name a declaration, got $(repr(lhs))")
    return lhs
end

_merge_stem(lhs::Symbol) = lhs
function _merge_stem(lhs::Expr)
    lhs.head === :ref && return _merge_stem(first(lhs.args))
    lhs.head === :call && length(lhs.args) == 2 &&
        first(lhs.args) in (:eachrow, :eachcol) && return _merge_stem(lhs.args[2])
    return nothing
end
_merge_stem(lhs) = nothing
_merge_path(lhs::Symbol) = nothing
function _merge_path(lhs::Expr)
    lhs.head === :ref && return _merge_path(first(lhs.args))
    lhs.head === :call && length(lhs.args) == 2 &&
        first(lhs.args) in (:eachrow, :eachcol) && return _merge_path(lhs.args[2])
    if lhs.head === :. && length(lhs.args) == 2 && lhs.args[2] isa QuoteNode
        parent = first(lhs.args)
        path = parent isa Symbol ? (parent,) : _merge_path(parent)
        path === nothing || return (path..., lhs.args[2].value)
    end
    return nothing
end
_merge_path(lhs) = nothing

function _merge_check_lhs(override, base)
    override isa Symbol && return nothing # replacing the whole declaration
    isequal(override, base) && return nothing
    _sfail("merge indexed override $(repr(override)) must match the complete " *
        "left-hand side $(repr(base)); a partial write is not a statement replacement")
end

function Base.merge(sm::RKPPLSubmodel, part::Union{Expr,NamedTuple}, rest...)
    stmts, ret = _submodel_body_parts(sm)
    model = Base.merge(RKPPLModel(Expr(:block, stmts...), sm.mod,
        copy(sm.fixed), copy(sm.rewrites)), part, rest...)
    # Pins become pure data-only definitions in the variant's own scope.
    for (nm, value) in model.fixed
        occursin('.', String(nm)) && continue
        helper = value isa AbstractArray ? :_bound_array_value : :_bound_value
        push!(model.ast.args, Expr(:(=), nm,
            Expr(:call, GlobalRef(@__MODULE__, helper), QuoteNode(value))))
    end
    return RKPPLSubmodel(sm.name, copy(sm.argnames),
        Expr(:block, model.ast.args..., ret), sm.mod, copy(sm.kwdefaults), copy(model.rewrites),
        Dict(k => v for (k, v) in model.fixed if occursin('.', String(k))))
end

# Julia 1.12 also versions global bindings. A builder may eval a named
# KernelSpec or import a submodel immediately before lowering; resolve the
# complete program in one latest-world scope, without a caller-side barrier.
_lower_model(m::RKPPLModel, supplied; conditioned = keys(m.conditioned)) =
    Base.invokelatest(_lower_model_latest, m, supplied; conditioned)

function _lower_model_latest(m::RKPPLModel, supplied; conditioned = keys(m.conditioned))
    values = merge(copy(m.fixed), Dict{Symbol,ColumnData}(pairs(supplied)), m.conditioned)
    observing = Set{Symbol}(_conditioned_names(conditioned))
    scoped = Dict(k => pop!(values, k) for k in collect(keys(values))
        if occursin('.', String(k)) && k ∉ observing)
    scoped_observed = Dict(k => pop!(values, k) for k in collect(keys(values))
        if occursin('.', String(k)) && k in observing)
    data = Set(keys(values))
    ast = Expr(:block, _desugar_destructuring(m.ast.args)...)
    scoped_values = Dict{Symbol,ColumnData}()
    ast, scopes = _expand_submodels(ast, data, m.mod; with_scopes = true,
        rewrites = m.rewrites, fixes = scoped, bound_values = scoped_values)
    merge!(values, scoped_values)
    used = Set{Symbol}()
    _all_symbols!(used, ast)
    ctx = _ScopeExpansion(scopes, Dict(s.binding => s for s in scopes),
        _scope_name_paths(scopes), used)
    expanded = RKPPLModel(ast, m.mod)
    for (path, value) in scoped_observed
        parts = Symbol.(split(String(path), '.'))
        nm = _resolve_scope_properties(foldl((a,b) -> Expr(:., a, QuoteNode(b)), parts), ctx)
        nm isa Symbol || _sfail("condition `$path` matches no scoped statement")
        values[nm] = value
        push!(observing, nm)
        push!(data, nm)
    end
    observed_parameters = _conditioned_declarations(expanded.ast, observing)
    setdiff!(data, observed_parameters)
    # Response LHSs are data only through the explicit conditioning set.
    _check_observed_inputs(expanded.ast, data, observing)
    ast, data = _scalar_value_definitions(expanded.ast, data,
        Set{Symbol}(k for (k, v) in values if v isa Number && k ∉ observed_parameters);
        arrays = Set{Symbol}(k for (k, v) in values if v isa AbstractArray &&
            (haskey(m.fixed, k) || haskey(scoped_values, k))))
    ast = _indexed_observation_definitions(ast, data)
    plan = _lower_rkppl_once(ast, data, m.mod;
        submodel_scopes = scopes, conditioned = observed_parameters)
    for name in observed_parameters
        values[_conditioned_input(name)] = pop!(values, name)
    end
    return plan, values
end

lower_rkppl(m::RKPPLModel, data::NamedTuple; conditioned=keys(m.conditioned)) = first(_lower_model(m, data; conditioned))
lower_rkppl(m::RKPPLModel, data::AbstractDict; conditioned=keys(m.conditioned)) = first(_lower_model(m, data; conditioned))

_conditioned_names(data::Union{NamedTuple,AbstractDict}) = keys(data)
_conditioned_names(names) = names

function _conditioned_declarations(ast, names)
    out = Set{Symbol}()
    definitions = Dict{Symbol,Any}()
    for st in ast.args
        st isa Expr && st.head === :(=) && first(st.args) isa Symbol || continue
        definitions[first(st.args)] = st.args[2]
    end
    for raw in ast.args
        raw isa Expr || continue
        st = _is_block_macro(raw) ? raw : _unwrap_trivia(raw)
        (_is_sample(st) || _is_broadcast_sample(st)) || continue
        lhs = _stmt_lhs(st)
        name = _merge_stem(lhs)
        name in names || continue
        # Whole-data GLM `~` declares a response, not a parameter. Like a
        # broadcast response, its observed value keeps its observation axis
        # instead of becoming an internal conditioned-parameter input.
        _is_glm_call(last(st.args)) && continue
        # A positional slice of an observed stream remains a response.
        # Sized level/matrix axes and matrix declarations retain array metadata.
        array = lhs isa Expr && ((lhs.head === :ref &&
            (length(lhs.args) > 2 || _is_levels_call(lhs.args[2]) ||
                (lhs.args[2] isa Symbol && haskey(definitions, lhs.args[2]) &&
                    _is_levels_call(definitions[lhs.args[2]])) ||
                any(ast.args) do other
                    other isa Expr && !_is_block_macro(other) || return false
                    other = _unwrap_trivia(other)
                    (_is_sample(other) || _is_broadcast_sample(other) ||
                        other.head === :(=)) || return false
                    return other !== st && _mentions_symbol(last(other.args), name)
                end ||
                (_is_axes2_call(lhs.args[2]) && lhs.args[2].args[2] !== name))) ||
            (lhs.head === :call && first(lhs.args) in (:eachrow, :eachcol)))
        (_is_sample(st) || array) && push!(out, name)
    end
    return out
end

function _check_observed_inputs(ast, inputs, observing)
    ast isa Expr || return nothing
    if _is_sample(ast) || _is_broadcast_sample(ast)
        names = _collect_lhs_binders!(Set{Symbol}(), _stmt_lhs(ast))
        for name in names
            name in inputs && name ∉ observing && _sfail("$name names a sampling statement; " *
                "observe it with `|` / `condition` or the `conditioned` lowering keyword, " *
                "or remove its statement with a merge pin")
        end
    end
    foreach(arg -> _check_observed_inputs(arg, inputs, observing), ast.args)
    return nothing
end

# A number bound to a data name has no observation axis: it lowers as the
# model-level definition `s = _bound_value(_rkppl_value_s)`, a data-only
# call `bind_data` evaluates once (functions as values), so `s` reads
# exactly as the definition `s = <number>` would while its value binds at
# the call. A name some statement already binds (a `~`/`.~` left-hand
# side, a definition, a loop variable) keeps its existing reading: a
# number bound to a response is one observation, and binding data to a
# sampled or defined name keeps failing as an overlap.
function _scalar_value_definitions(ast::Expr, data::Set{Symbol},
        scalars::Set{Symbol}; arrays::Set{Symbol} = Set{Symbol}())
    isempty(scalars) && isempty(arrays) && return ast, data
    bound = Set{Symbol}()
    _statement_lhs_names!(bound, ast)
    defs = Any[]
    data = copy(data)
    for s in sort!(collect(union(scalars, arrays)))
        s in bound && continue
        input = _bound_value_input(s)
        (input in data || _mentions_symbol(ast, input)) &&
            _sfail("data value $s: the internal name $input is already " *
                   "used by the model or its data")
        delete!(data, s)
        push!(data, input)
        helper = s in arrays ? :_bound_array_value : :_bound_value
        push!(defs, Expr(:(=), s,
            Expr(:call, GlobalRef(@__MODULE__, helper), input)))
    end
    isempty(defs) && return ast, data
    return Expr(:block, ast.args..., defs...), data
end

function _statement_lhs_names!(out::Set{Symbol}, ex)
    ex isa Expr || return out
    if ex.head === :call && length(ex.args) == 3 &&
            ex.args[1] in (:~, :.~)
        _lhs_base_names!(out, ex.args[2])
    elseif ex.head === :(=) && length(ex.args) == 2
        _lhs_base_names!(out, ex.args[1])
    end
    for a in ex.args
        _statement_lhs_names!(out, a)
    end
    return out
end

function _lhs_base_names!(out::Set{Symbol}, lhs)
    if lhs isa Symbol
        push!(out, lhs)
    elseif lhs isa Expr && lhs.head === :ref && !isempty(lhs.args)
        _lhs_base_names!(out, lhs.args[1])
    elseif lhs isa Expr && lhs.head in (:vect, :tuple)
        foreach(a -> _lhs_base_names!(out, a), lhs.args)
    end
    return out
end

_mentions_symbol(ex, s::Symbol) = ex === s ||
    (ex isa Expr && any(a -> _mentions_symbol(a, s), ex.args))

"""
    lower_rkppl(ast, data_names; mod=Main) -> StructuralPlan
    lower_rkppl(ast, data; mod=Main) -> StructuralPlan

Lower a captured `@rkppl` block AST to a data-free (unbound) plan.
`data_names` (or the keys of `data`, a dict or `NamedTuple` of values)
declares supplied values. Pass their observation roles separately with
`conditioned = (:y, :tau)`. The keyword accepts names or the keys of a
NamedTuple/dict. A conditioned sampling declaration contributes likelihood
at its constrained value, with no sampled coordinate or transform Jacobian.
Unconditioned supplied names must be inputs; remove a sampling declaration
with a merge pin to supply it as an input instead. Runs `validate_structure`
before returning. The BRM emitter calls this entry point directly with ASTs.

`mod` is the module against which `latent ~ sm(args...)` call heads are
resolved to [`RKPPLSubmodel`](@ref)s; a resolving call is expanded inline
before partitioning (see `_expand_submodels`), and a call inside a submodel
body resolves against that submodel's defining module.

Functions as values: an `=` definition may call any function visible in
`mod` (a submodel body: in its own defining module). Call heads outside the
built-in value vocabulary resolve to `GlobalRef`s at lowering, and an
undefined one fails here naming it. An undotted call takes and returns whole
Julia values; a dotted call `f.(...)` is elementwise, observation
aligned exactly when an argument is; `v[c]` with an observation index is a
gather. A data-only definition calling such a function is evaluated once by
[`bind_data`](@ref) and bound as data; one that a parameter-dependent
expression consumes (named or inline) is evaluated once by preparation and
never differentiated. Any other call runs in the generated kernel under
generic AD (an RK-owned derivative rule, when the callee is one, is used by
Enzyme). A parameter-dependent undotted call may read observation columns
and return a scalar or an observation-length vector for use as a location
or predictor operand. Julia checks its concrete shape when evaluated.
A data column read only inside whole-value calls, with no per-observation
consumer, is a model-level data input of any length at bind.

Input ownership and concurrency: `lower_rkppl` is concurrency-safe and
takes no lock, so independent lowerings run concurrently. Neither `ast` nor
the submodel bodies reachable through `mod` is mutated, so one AST may be
lowered repeatedly and shared across tasks. Submodel and function resolution
only READ `mod` bindings (`isdefined` / `getfield` — no eval, no
registration), so definitions in distinct private modules never collide: one
fresh `Module` per lowering, each holding its own `@rkppl name(args...) = ...`
defs, is sufficient for concurrent independent lowerings.

Lowering resolves these bindings in a package-owned latest-world scope,
including kernels and imports just created by `Core.eval` in a builder.
No caller-side `invokelatest` is needed for lowering on Julia 1.12 or later.

Declared coefficients are ordinary named parameters. A scalar
`b ~ Fam(...)` keeps its name, prior and transform when an affine term
reads it as `b .* x`. A sized declaration such as `c[levels(g)]` or
`b[axes(X, 2)]` remains an array parameter when read as `c[g]` or `X * b`.
Affine optimization records references to those values and may pack their
reads for a matrix multiply. Other predictors, assignments and prior
arguments can read the same declaration without changing its coordinates
or adding another prior.

Data are values of any shape. Given the values (`data`), lowering reads
each one's shape: a number has no observation axis, so it lowers as a
model-level value — exactly as the definition `s = 2.0` would read, except
that `bind_data` binds the number and the kernel takes it as a typed
scalar argument (`y .~ Normal.(mu, s)` broadcasts it, as Julia does).
Every `@rkppl` entry point (the model call, `@rkppl data`, `merge` pins)
lowers with the values. Given names only, binding a number to an ordinary
broadcast argument also keeps Julia's scalar broadcasting. A number bound
to a response is one observation.
"""
lower_rkppl(ast, data::NamedTuple; mod::Module = Main, conditioned=()) =
    lower_rkppl(ast, Dict{Symbol,Any}(pairs(data)); mod, conditioned)

function lower_rkppl(ast, data::AbstractDict; mod::Module = Main, conditioned=())
    names = Symbol[]
    scalars = Set{Symbol}()
    for (k, v) in data
        k isa Symbol || _sfail("data names must be Symbols, got $(repr(k))")
        push!(names, k)
        v isa Number && push!(scalars, k)
    end
    return _lower_rkppl(ast, names, scalars, mod; conditioned)
end

lower_rkppl(ast, data_names; mod::Module = Main, conditioned=()) =
    _lower_rkppl(ast, data_names, Set{Symbol}(), mod; conditioned)

_lower_rkppl(ast, data_names, scalars::Set{Symbol}, mod::Module; conditioned=())::StructuralPlan =
    Base.invokelatest(_lower_rkppl_latest, ast, data_names, scalars, mod; conditioned)

function _lower_rkppl_latest(ast, data_names, scalars::Set{Symbol},
        mod::Module; conditioned=())::StructuralPlan
    data = Set{Symbol}()
    for n in data_names
        n isa Symbol || _sfail("data names must be Symbols, got $(repr(n))")
        push!(data, n)
    end
    ast isa Expr && ast.head === :block ||
        _sfail("lower_rkppl takes a `begin ... end` block AST")
    ast = Expr(:block, _desugar_destructuring(ast.args)...)
    observing = Set{Symbol}(_conditioned_names(conditioned))
    # Submodel streams need their observation name during expansion.
    primitive = Set(n for n in _conditioned_declarations(ast, observing)
        if !any(a -> a isa Expr && _stmt_is_submodel_call(a, mod) &&
            _merge_stem(_stmt_lhs(_unwrap_trivia(a))) === n, ast.args))
    setdiff!(data, primitive)
    ast, scopes = _expand_submodels(ast, data, mod; with_scopes = true)
    observed_parameters = _conditioned_declarations(ast, observing)
    setdiff!(data, observed_parameters)
    _check_observed_inputs(ast, data, observing)
    ast, data = _scalar_value_definitions(ast, data, setdiff(scalars, observed_parameters))
    ast = _indexed_observation_definitions(ast, data)
    return _lower_rkppl_once(ast, data, mod;
        submodel_scopes = scopes, conditioned = observed_parameters)

end

# A data-indexed observation is an ordinary gather followed by an
# observation of its result. Keep the RHS as authored: it explicitly
# selects every likelihood input that needs selection too.
function _indexed_observation_definitions(ast::Expr, data::Set{Symbol})
    taken = union(data, _expr_names(ast))
    out = Any[]
    for st in ast.args
        if st isa Expr && _is_broadcast_sample(st)
            lhs = st.args[2]
            if lhs isa Expr && lhs.head === :ref && length(lhs.args) == 2 &&
                    lhs.args[1] in data &&
                    ((lhs.args[2] isa Symbol && lhs.args[2] in data) ||
                     (_mentions_symbol(st.args[3], :MixtureModel) &&
                      _literal_row_range(lhs.args[2])))
                k = 1
                name = Symbol(_OBSERVED_GATHER_PREFIX, lhs.args[1], :_, k)
                while name in taken
                    k += 1
                    name = Symbol(_OBSERVED_GATHER_PREFIX, lhs.args[1], :_, k)
                end
                push!(taken, name)
                push!(out, Expr(:(=), name, lhs))
                push!(out, Expr(:call, st.args[1], name, st.args[3]))
                continue
            end
        end
        push!(out, st)
    end
    return Expr(:block, out...)
end

function _literal_row_range(ex)
    ex isa Expr && ex.head === :call && length(ex.args) == 3 || return false
    fn = ex.args[1]
    colon = fn === :(:) || (fn isa GlobalRef &&
        getfield(fn.mod, fn.name) === getfield(Base, :(:)))
    return colon && all(x -> x isa Integer, ex.args[2:end])
end

# Ordinary parameters may have any number of readers. Legacy
# construct-owned coefficients retain their ownership checks.
function _check_owned_coefficient(name::Symbol, ctx, msg)
    name in ctx.ordinary_parameters && return nothing
    return _sfail(msg)
end

function _ordinary_sampling(s)
    s.dims !== nothing && return true
    s.matrix !== nothing && return true
    s.levels !== nothing && return true
    s.broadcast && return false
    rhs = s.rhs
    return rhs isa Expr && rhs.head === :call &&
        rhs.args[1] in union(keys(_PARAM_FAMILIES),
            (:HalfNormal, :HalfCauchy, :Flat, :truncated, :restricted)) || _external_rhs(rhs)
end

function _lower_rkppl_once(ast, data::Set{Symbol}, mod::Module;
        submodel_scopes::Vector{SubmodelScope} = SubmodelScope[],
        conditioned::Set{Symbol} = Set{Symbol}(),
        value_defs::Set{Symbol} = Set{Symbol}(),
        shared_defs::Set{Symbol} = Set{Symbol}())
    input_data = data
    ast = _observation_call_definitions(ast, data, mod)
    ast = _gathered_value_definitions(ast, data)
    for name in conditioned
        input = _conditioned_input(name)
        (input in data || _mentions_symbol(ast, input)) &&
            _sfail("condition `$name` needs internal input `$input`, already used by the model or its data")
    end
    sample, det, plate_ctx, plate_specs, scans, joints, glms, cell_broadcasts =
        _partition_statements(ast, data)
    sample, det = _rewrite_plate_rows(sample, det)
    # Latent plates over values read their cell count as bound data.
    extents = Set{Symbol}(_plate_extent_input(nm) for (nm, rhs, _, _) in plate_specs
        if _mentions_symbol(rhs, _plate_extent_input(nm)))
    for input in extents
        (input in data || _mentions_symbol(ast, input)) && _sfail(
            "`@plate` needs internal input `$input`, already used by the model or its data")
    end
    data = union(data, extents)
    # Definitions resolve in the model module; lexical model names shadow
    # functions without any domain-specific schedule extraction.
    model_names = union(data, Set{Symbol}(nm for (nm, _) in det),
        Set{Symbol}(s.lhs for s in sample),
        Set{Symbol}(nm for (nm, _, _, _) in plate_specs),
        Set{Symbol}(st for s in scans for st in s.states))
    _check_plate_prior_reads(plate_specs, model_names)
    det = Pair{Symbol,Any}[nm => _resolve_module_calls(rhs, mod, model_names,
        "definition `$nm = $(repr(rhs))`") for (nm, rhs) in det]
    # Prior expression arguments hoist to synthetic definitions, resolved
    # exactly like a definition holding the same expression.
    det = _hoist_prior_args!(sample, det, data, plate_specs;
        resolve = a -> _resolve_module_calls(a, mod, model_names,
            "prior argument `$(repr(a))`"))
    det = _hoist_data_gather_indices!(sample, det, data,
        union(model_names, _expr_names(ast));
        resolve = a -> _resolve_module_calls(a, mod, model_names,
            "gather index `$(repr(a))`"))
    # A data-only module value a response reads per observation is that
    # observation column or matrix (user decision `0z5bsqi`, prong
    # `in_model`): it lowers as data, computed once by `bind_data`, which
    # checks its length or row count. Read whole, it stays model-level.
    rawdata = data
    sample, det, matrix_values = _route_hcat_values(sample, det, data, glms;
        resolve = a -> _resolve_module_calls(a, mod, model_names, "matrix value"))
    aligned = _aligned_module_data(sample, det, data)
    aligned_defs = Pair{Symbol,Any}[p for p in det if first(p) in aligned]
    if !isempty(aligned)
        data = union(data, aligned)
        det = Pair{Symbol,Any}[p for p in det if first(p) ∉ aligned]
    end
    plate_names = Set{Symbol}(nm for (nm, _, _, _) in plate_specs)
    detmap = Dict{Symbol,Any}(nm => rhs for (nm, rhs) in det)
    detnames_all = Set{Symbol}(nm for (nm, _) in det)
    for s in sample
        s.lhs in conditioned && !s.broadcast && s.dims === nothing || continue
        rhs = s.rhs
        rhs isa Expr && rhs.head === :call && first(rhs.args) in
            union(keys(_PARAM_FAMILIES), (:HalfNormal, :HalfCauchy)) || continue
        for argument in rhs.args[2:end]
            _shape_of(argument, data, detmap, Dict{Symbol,Symbol}(), Set{Symbol}()) === :scalar ||
                _sfail("conditioned scalar $(s.lhs) has a vector-valued distribution argument; " *
                    "observe a vector with `.~`")
        end
    end
    # Derived responses (`.~` over a deterministic definition, either
    # order): data-like everywhere downstream — excluded from the sampled
    # name table (mixture/coef-prior/param positions keep treating them
    # as data) and skipped by parameter lowering.
    derived_response_names = Set{Symbol}(s.lhs for s in sample
        if s.broadcast && s.levels === nothing && s.matrix === nothing &&
            s.lhs ∉ data && s.lhs in detnames_all)
    sampled_names = Set{Symbol}(s.lhs for s in sample if s.lhs ∉ data)
    prior_names = setdiff(sampled_names, derived_response_names)
    # Declarations determine value/prior semantics. Affine recognition
    # may use these values regardless of their prior or other readers.
    ordinary_parameters = Set{Symbol}(s.lhs for s in sample
        if s.lhs ∉ data && _ordinary_sampling(s))
    scalar_parameters = Set{Symbol}(s.lhs for s in sample
        if s.lhs in ordinary_parameters && !s.broadcast &&
            s.dims === nothing)
    coef_priors = scalar_parameters
    # Simplex parameters (`s ~ Dirichlet(...)`): the only names a
    # monotonic term accepts as its increments (checked during response
    # lowering, before `_lower_parameters` runs).
    dirichlet_names = Set{Symbol}(s.lhs for s in sample
        if s.lhs ∉ data && s.slices === nothing && _is_dirichlet_call(s.rhs))
    # Ordered vectors (`c ~ Ordered(Normal(0, 1), K)`): the only names a
    # cumulative ordinal response takes as explicit cutpoints (checked
    # during response lowering, before `_lower_parameters` runs).
    ordered_names = Set{Symbol}(s.lhs for s in sample
        if s.lhs ∉ data && !s.broadcast && s.slices === nothing &&
            _is_ordered_call(s.rhs))
    # An `hcat` matrix the program reads as a value (`var.(eachcol(X))`,
    # `X * v` over a computed or declared-array vector, a coefficient
    # prior the matrix term cannot carry) lowers exactly like a bound data
    # matrix `X`: its definition leaves `det`, its name joins `data`, and
    # `bind_data` builds it from its columns (`_value_design_matrices`).
    caller_data = data
    value_mats = _hcat_value_reads(det, detmap, sample, data, glms,
        union(dirichlet_names, ordered_names),
        union(plate_names, Set{Symbol}(st for s in scans for st in s.states)))
    value_mat_defs = Pair{Symbol,Any}[p for p in det if first(p) in value_mats]
    if !isempty(value_mats)
        det = Pair{Symbol,Any}[p for p in det if first(p) ∉ value_mats]
        for nm in value_mats
            delete!(detmap, nm)
        end
        data = union(data, value_mats)
    end
    # Declaration roles for predictor classification: sized positional
    # and two-axis arrays, LKJ Cholesky factors, and vectors sized by a
    # bound data matrix (not an `hcat` definition) always take the array
    # route. One-axis `c[levels(g)]` / `b[axes(X, 2)]` over an `hcat`
    # matrix stay coefficient-capable (`c[g]`, `X * b`), independently of
    # their whole-value array shape below. `value_arrays` are never
    # coefficients: a bare `z[g]` summand gathers them.
    # A one-axis `z[levels(g)]` that a definition reads as a whole value
    # (`b = sd .* z`) also takes the array role (`array_decls`,
    # `value_arrays`): a levels coefficient is only ever consumed indexed
    # by its group (`c[g]`), never whole.
    whole_reads = Set{Symbol}()
    for (_, rhs) in det
        _whole_name_reads!(whole_reads, rhs)
    end
    levels_values = Set{Symbol}(s.lhs for s in sample if s.lhs ∉ data &&
        s.dims === nothing && s.levels !== nothing && s.lhs in whole_reads)
    hcat_defs = Set{Symbol}(nm for (nm, rhs) in det if _is_hcat_def(rhs))
    array_decls = Set{Symbol}(s.lhs for s in sample if s.lhs ∉ data &&
        (s.dims !== nothing || s.lhs in levels_values ||
            (s.broadcast && s.matrix !== nothing && s.matrix ∉ hcat_defs) ||
            (!s.broadcast && _is_lkj_cholesky_call(s.rhs))))
    value_arrays = Set{Symbol}(s.lhs for s in sample if s.lhs ∉ data &&
        (s.dims !== nothing || s.lhs in levels_values ||
            (!s.broadcast && _is_lkj_cholesky_call(s.rhs))))
    # A whole-read levels declaration (`levels_values`) is an array, never
    # a factor coefficient an alias (`th = c[g]`) can stand for.
    factor_decls = Set{Symbol}(s.lhs for s in sample if s.lhs ∉ data &&
        s.broadcast && s.levels !== nothing && s.dims === nothing &&
        s.lhs ∉ levels_values)
    _check_definition_levels_axes(sample, det, data)
    # Every array-capable declaration (sized `.~`, LKJ, a `Dirichlet`
    # simplex or `Ordered` vector value): an expression that READS one by
    # index (`L[2, 1] .* x`, `tau .* z[g]`, `phi[1] .* x`) is a value, never
    # an affine sub-predictor over coefficients.
    sized_decls = union(array_decls, dirichlet_names, ordered_names,
        Set{Symbol}(s.lhs for s in sample
            if s.lhs ∉ data && s.broadcast &&
                (s.levels !== nothing || s.matrix !== nothing)))
    # Shape every definition (data-free: data ⇒ vector, sampled ⇒ scalar,
    # det-refs recurse with memo; cycles error downstream), then
    # canonicalize each RHS in dependency order (Julia-valid undotted
    # scalar-array ops take dotted-canonical form; Julia-invalid vector
    # combinations fail here naming the definition). Everything downstream
    # sees canonical RHSs.
    # Every sized declaration is an array when read as a whole value,
    # including coefficient-capable `z[levels(g)]` / `b[axes(X, 2)]`.
    # Shape is independent of whether predictor lowering later consumes
    # an indexed/matmul use as a coefficient. Keep `array_decls` and
    # `value_arrays` separate: they control those coefficient use sites.
    shape_env = _ShapeEnv(
        union(plate_names, Set{Symbol}(st for s in scans for st in s.states)),
        sized_decls)
    # A data column the definitions read only as whole values (module-call
    # arguments, gathered values) is a model-level data input at bind, so
    # a parameter-dependent call may take it. Confirm the whole-value
    # classification once the plan includes every consumer.
    # An array prior consumes its arguments as whole values too. Seed that
    # context before shaping definitions, then confirm it on the final plan.
    prior_defs = Pair{Symbol,Any}[Symbol(:_ppl_prior_input_, s.lhs) => s.rhs
        for s in sample if s.lhs in sized_decls]
    whole = _whole_value_data(vcat(det, prior_defs), data,
        _statement_names(ast, union(data, Set{Symbol}(keys(detmap)));
            whole_priors = sized_decls))
    # Whole-value data compose with array parameters before canonicalization,
    # exactly like a module call's model-level result. Their concrete shape
    # remains Julia's business at bind/evaluation, never an observation axis.
    shape_env = _ShapeEnv(shape_env.aligned, union(shape_env.values, whole),
        matrix_values)
    detshape = _def_shapes(det, data, detmap; arrays = sized_decls,
        env = shape_env)
    # A `.~` response is observed data. A definition reading only data that
    # it observes is that observation value, whatever shape lowering can
    # prove: an undotted module call's result is model-level here
    # (`y = f(raw)`). `bind_data` evaluates the definition once and
    # validates its value as the response; a number is one observation, as
    # when bound as data.
    for nm in derived_response_names
        get(detshape, nm, nothing) === :scalar || continue
        rhs = detmap[nm]
        !isempty(_value_symbols(rhs)) && _data_only(rhs, data, detmap) &&
            (detshape[nm] = :vector)
    end
    canonmap = Dict{Symbol,Any}()
    for nm in _det_topo_order(det, detmap)
        rhs = detmap[nm]
        _reject_unknown_calls("definition `$nm = $(repr(rhs))`", rhs;
            composed_maps = true)
        canonmap[nm] = _canonical_expr(rhs, data, detmap, detshape, shape_env,
            "definition `$nm = $(repr(rhs))`")
    end
    declaration_dependencies = _declaration_dependencies(sample, plate_specs,
        union(data, derived_response_names), canonmap)
    declaration_data = Set{Symbol}(nm for nm in declaration_dependencies
        if haskey(canonmap, nm) && _data_only(canonmap[nm], data, canonmap))
    # Design matrices leave `det` for the plan-level table (validated
    # here; `detshape` keeps their `:matrix` shape for downstream
    # honesty, but nothing inlines or assigns them).
    det, matrices = _extract_matrices(det, detmap, detshape, data,
        prior_names, plate_names, scans)
    for m in matrices
        delete!(canonmap, m.name)
    end
    value_matrices = _value_design_matrices(value_mat_defs, caller_data)
    # Structural definitions inline into predictors: anything transitively
    # referencing a coefficient candidate (coef-priored or free name).
    # All other vector definitions stay symbolic as named locals.
    structural = _structural_defs(det, data, canonmap, coef_priors,
        prior_names, plate_names)
    setdiff!(structural, value_defs)
    vecdefs = Set{Symbol}(nm for (nm, _) in det if detshape[nm] === :vector)
    taken = union(data, Set{Symbol}(nm for (nm, _) in det), prior_names,
        plate_names)
    # Interned predictors by name (shared with `ctx`: a composition over a
    # definition already interned as a predictor reads its LP node).
    pred_idx = Dict{Symbol,Int}()
    ctx = (; data, mod, conditioned, value_defs, shared_defs,
        detmap = canonmap,
        prior_names, coef_priors,
        ordinary_parameters,
        detshape, shape_env, declaration_data,
        vecdefs, structural, derived_responses = derived_response_names,
        pred_idx,
        plate_names, absorbed = Set{Symbol}(),
        synth = Ref(0), synth_derived = VectorAssignmentSpec[], taken,
        synth_assigns = AssignmentSpec[],
        negated = Dict{Symbol,Symbol}(),
        leaf_exprs = Dict{Any,Symbol}(),
        # Definitions a reader folded into itself (`_inlined_computed_defs`).
        inlined = Set{Symbol}(),
        # Inline factor references inside compositions (`sg .* z[g]`)
        # intern as synthetic sub-predictors, one per `(base, index)`,
        # exactly like the named alias `zg = z[g]`.
        scan_states = Set{Symbol}(st for s in scans for st in s.states),
        scan_coefs = Set{Symbol}(),
        dirichlet_names = dirichlet_names,
        ordered_names = ordered_names,
        array_dims = Dict{Symbol,Vector{Any}}(s.lhs => s.dims for s in sample
            if s.lhs ∉ data && s.dims !== nothing),
        threshold_uses = Dict{Symbol,NamedTuple{(:response, :ordered),
            Tuple{Symbol,Bool}}}(),
        vector_params = union(dirichlet_names, ordered_names,
            Set{Symbol}(s.lhs for s in sample
                if s.levels !== nothing || s.matrix !== nothing)),
        matrices = Dict{Symbol,DesignMatrix}(m.name => m for m in matrices),
        matrices_used = Set{Symbol}(),
        coefvecs = Dict{Symbol,Symbol}(s.lhs => s.matrix for s in sample
            if s.matrix !== nothing),
        array_decls = array_decls,
        value_arrays = value_arrays,
        factor_decls = factor_decls,
        factor_axes = Dict{Symbol,Any}(s.lhs => s.levels for s in sample
            if s.levels !== nothing),
        sized_decls = sized_decls)
    # Declarations own their priors and coordinates before affine analysis.
    # Sized positional declarations wait for response-specific validation.
    declarations = [s for s in sample if
        s.lhs in ordinary_parameters && s.dims === nothing]
    declared_names = Set(s.lhs for s in declarations)
    declared_params, declared_syms, declared_vectors, declared_arrays =
        _lower_parameters(declarations,
            Dict{Symbol,Vector{Tuple{Symbol,Symbol,Int}}}(), ctx,
            Dict{Symbol,Tuple{Symbol,Symbol}}())
    responses = LikelihoodSpec[]
    external_observations = SampledParameter[]
    predictors = PredictorSpec[]
    coefuse = Dict{Symbol,Vector{Tuple{Symbol,Symbol,Int}}}()
    glmuse = Dict{Symbol,Tuple{Symbol,Symbol}}()
    for s in sample
        # Declared arrays lower with the parameters.
        s.dims !== nothing && continue
        _external_rhs(s.rhs) && s.lhs in conditioned && continue
        if _external_rhs(s.rhs) && s.broadcast && (s.lhs in data || s.lhs in ctx.vecdefs)
            args = _external_parameter(s.lhs, s.rhs, (), mod, ctx.taken;
                observed=true, broadcast=true)
            push!(external_observations, SampledParameter(s.lhs, :external, args, nothing, s.lhs))
            continue
        end
        if s.broadcast
            # Broadcast coefficient priors lower with their factor term.
            s.levels !== nothing && continue
            # Matrix coefficient priors lower with their matrix term.
            s.matrix !== nothing && continue
            if s.lhs in data
                push!(responses,
                    _lower_response(s.lhs, s.rhs, s.range, ctx, predictors,
                        pred_idx, coefuse; count_columns = s.count_columns))
            elseif s.lhs in ctx.vecdefs
                _gate_derived_response!(s.lhs, ctx)
                push!(responses,
                    _lower_response(s.lhs, s.rhs, s.range, ctx, predictors,
                        pred_idx, coefuse; count_columns = s.count_columns))
            else
                _sfail(_broadcast_lhs_msg(s.lhs, detshape))
            end
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
            _lower_joint_response(j, ctx, predictors, pred_idx,
                coefuse))
    end
    # GLM-object responses lower after the joints (no predictor
    # interning — the object owns eta; coefficient recording goes to
    # the separate GLM-use table, before coefficient priors resolve).
    for g in glms
        push!(responses,
            _lower_glm_response(g, sample, prior_names, ctx, coefuse, glmuse))
    end
    _check_coefficient_uses(coefuse, ctx)
    for c in ctx.scan_coefs
        haskey(coefuse, c) && _check_owned_coefficient(c, ctx, "$c is both a " *
            "predictor coefficient and a scan coefficient — scan " *
            "coefficients are sampled scalars, not population " *
            "coefficients (rename one)")
    end
    # Affine uses reference ordinary declarations without taking their priors.
    owned_coefs = Dict(k => v for (k, v) in coefuse if k ∉ ordinary_parameters)
    _check_coefficient_uses(coefuse, ctx)
    # Hyperparameter names the broadcast peelers admit at the surface:
    # sampled scalar names (derived responses excluded — they are
    # vectors) and scalar definitions. Anything else (data,
    # coefficients, vectors, unknown names) fails here with the legacy
    # spelling error — the contract refines roles (location vs scale
    # support) for the admitted names.
    hyper_names = union(prior_names,
        Set{Symbol}(nm for (nm, _) in det if detshape[nm] === :scalar))
    priors, levelmaps = _lower_coefficient_priors(sample, coefuse, predictors,
        ctx.matrices, hyper_names, ctx)
    for (beta, (label, X)) in glmuse
        beta in ordinary_parameters && continue
        append!(priors, _lower_glm_beta_priors(label, beta, X, sample,
            ctx.matrices, hyper_names))
    end
    params, paramsyms, dirichlets, arrays =
        _lower_parameters([s for s in sample if s.lhs ∉ declared_names],
            owned_coefs, ctx, glmuse)
    prepend!(params, declared_params)
    union!(paramsyms, declared_syms)
    prepend!(dirichlets, declared_vectors)
    prepend!(arrays, declared_arrays)
    # Declaration order is independent of which values the optimizer used.
    declaration_order = Dict(s.lhs => i for (i, s) in enumerate(sample))
    sort!(arrays; by = p -> declaration_order[p.name])
    # Coefficient-prior hyperparameters read their names too (an inlined
    # scalar definition that is also a prior scale must still emit).
    for pr in priors, v in (pr.location, pr.scale)
        v isa Symbol && push!(paramsyms, v)
    end
    plate_parameters = PlateParameter[
        _lower_plate_parameter(nm, rhs, rng, coefuse, ctx.matrices, ctx)
        for (nm, rhs, rng, _) in plate_specs]
    used_locs = Set{Symbol}()
    # Composed sub-predictors are locations too (interned LP nodes).
    for p in predictors, t in p.terms
        t.kind === ComposedTerm && union!(used_locs, t.options.subs)
    end
    for r in responses
        push!(used_locs, r.predictor)
        union!(used_locs, r.extra_predictors)
        # A scale predictor's definition is absorbed like a location's —
        # never also a derived column. A predictor-fed nu or ZIP zi
        # absorbs the same way.
        r.scale isa ScalePredictorRef &&
            push!(used_locs, r.scale.predictor)
        r.nu isa ScalePredictorRef &&
            push!(used_locs, r.nu.predictor)
        r.zi isa ScalePredictorRef &&
            push!(used_locs, r.zi.predictor)
        r.discrimination isa ScalePredictorRef &&
            push!(used_locs, r.discrimination.predictor)
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
    skip = _absorbed_skip(det, canonmap, responses, paramsyms, ctx.absorbed,
        used_locs)
    # Optimizing a definition as a predictor must retain its value for
    # other consumers, including reductions, extracted leaves and the
    # retained recurrence body. Keep their definition dependencies too.
    needed = copy(declaration_dependencies)
    for observation in external_observations
        push!(needed, observation.name)
        union!(needed, _value_symbols(observation.args.rhs))
    end
    for p in arrays, d in p.dims
        union!(needed, _value_symbols(d))
    end
    for p in params, arg in values(p.args)
        union!(needed, _value_symbols(arg))
    end
    for a in (ctx.synth_assigns..., ctx.synth_derived...)
        union!(needed, _value_symbols(a.expr))
    end
    for (nm, _) in det
        nm in skip || union!(needed, _value_symbols(canonmap[nm]))
    end
    for s in scans, st in (s.setup..., s.step...)
        for ex in (st.kind === :sample ? st.args : (st.expr,))
            union!(needed, _value_symbols(ex))
        end
    end
    pending = collect(needed)
    kept = Set{Symbol}()
    while !isempty(pending)
        nm = pop!(pending)
        (nm in kept || !haskey(canonmap, nm)) && continue
        push!(kept, nm)
        delete!(skip, nm)
        append!(pending, _value_symbols(canonmap[nm]))
    end
    # A predictor optimization cannot also claim a definition retained by
    # an ordinary value reader. Replan those names as values; the set only
    # grows, so every retry releases at least one conflicting definition.
    retained = setdiff(intersect(kept, Set(p.name for p in predictors)),
        value_defs)
    if !isempty(retained)
        return _lower_rkppl_once(ast, input_data, mod;
            submodel_scopes, conditioned, value_defs = union(value_defs, retained),
            shared_defs)
    end
    # A predictor location inlines the definitions it reads. As in Julia,
    # a computed definition is instead one named value that its readers
    # read, however many there are (user decision `0fbe312`): a
    # per-observation definition replans as a retained value (every reader
    # reads the named column), a model-level one as a shared computation.
    # Both sets only grow too.
    shared = _inlined_computed_defs(ctx, canonmap, data)
    if !isempty(shared)
        columns = Set{Symbol}(nm for nm in shared
            if get(ctx.detshape, nm, :scalar) === :vector)
        return _lower_rkppl_once(ast, input_data, mod;
            submodel_scopes, conditioned,
            value_defs = union(value_defs, columns),
            shared_defs = union(shared_defs, setdiff(shared, columns)))
    end
    assigns = AssignmentSpec[]
    derived = VectorAssignmentSpec[]
    # Axes of every declared array an array-cell plate may read.
    pc_dims = Dict{Symbol,Vector{Any}}()
    for s in sample
        s.lhs in data && continue
        if s.dims !== nothing
            pc_dims[s.lhs] = s.dims
        elseif s.levels !== nothing
            gcol, subset = s.levels
            pc_dims[s.lhs] = Any[subset === Colon() ? :(levels($gcol)) :
                Expr(:call, :levels, gcol, QuoteNode(subset))]
        end
    end
    for (nm, rhs) in det
        axis = _plate_column_axis(rhs)
        axis === nothing || (pc_dims[nm] = Any[:(levels($axis))])
    end
    for (i, p) in enumerate(predictors)
        terms = TermSpec[if t.kind === ComposedTerm &&
                _is_plate_column_call(t.options.tree)
            TermSpec(t.kind, t.columns, merge(t.options,
                (; tree = _plate_column_expr(p.name, t.options.tree, pc_dims, data))),
                t.addressee, t.label)
        else
            t
        end for t in p.terms]
        predictors[i] = PredictorSpec(p.name, p.link, terms, p.label)
    end
    for (nm, _) in det
        nm in skip && continue
        rhs = canonmap[nm]
        _is_plate_column_call(rhs) &&
            (rhs = _plate_column_expr(nm, rhs, pc_dims, data))
        if detshape[nm] === :vector
            push!(derived, _lower_vector_assignment(nm, rhs, coefuse))
        else
            push!(assigns, _lower_assignment(nm, rhs, coefuse))
        end
    end
    for (nm, rhs) in aligned_defs
        push!(derived, VectorAssignmentSpec(nm, rhs, nm))
    end
    for d in ctx.synth_derived
        rhs = _is_plate_column_call(d.expr) ?
            _plate_column_expr(d.name, d.expr, pc_dims, data) : d.expr
        push!(derived, VectorAssignmentSpec(d.name, rhs, d.label))
    end
    append!(assigns, ctx.synth_assigns)
    _check_coefficient_readers(coefuse, ctx, predictors, priors, params,
        plate_parameters, assigns, derived, responses)
    _check_plate_bares(plate_ctx, data, Set{Symbol}(p.name for p in predictors),
        Set{Symbol}(d.name for d in derived), _factor_coefs(coefuse, predictors),
        plate_names)
    for m in matrices
        used_axis = any(p -> any(d -> _is_axis_dim(d) && d.args[2] === m.name,
            p.dims), arrays)
        m.name in ctx.matrices_used || used_axis || _sfail(
            "design matrix `$(m.name)` is never used in a predictor " *
            "matmul or array axis — drop it or add the use (`mu = $(m.name) * b`)")
    end
    indexed_observations = intersect(Set{Symbol}(first(c) for c in plate_ctx),
        Set{Symbol}(r.response for r in responses))
    # A caller-owned sampling RHS observed in a dotted `@plate` cell
    # (`y[i] .~ LogDensity.(f, loc[i], s)`).
    indexed_external = intersect(Set{Symbol}(first(c) for c in plate_ctx),
        Set{Symbol}(p.name for p in external_observations))
    # Explicit loops select each operand at the authored indices. Keep a
    # literal response iterator on that same gather path as eachindex/axes;
    # the older UnitRange response contract only checked full-column cover.
    responses = [r.range isa UnitRange && r.response in indexed_observations ?
        _with(r; range=Expr(:ref, r.response,
            Expr(:call, :(:), first(r.range), last(r.range)))) : r for r in responses]
    named_values = _absorbed_named_values(det, canonmap, skip, predictors)
    plan = StructuralPlan(responses, predictors, priors, params, assigns,
        Dict{Symbol,AbstractVector}(), 0; derived = derived,
        levelmaps = levelmaps, plate_parameters = plate_parameters, scans = scans,
        vector_parameters = dirichlets,
        matrices = vcat(matrices, value_matrices),
        array_parameters = arrays, submodel_scopes = submodel_scopes, conditioned, external_observations,
        indexed_observations,
        cell_broadcasts = Dict{Symbol,Vector{Symbol}}(k => v for (k, v) in cell_broadcasts
            if k in indexed_observations || k in indexed_external),
        named_values)
    _confirm_whole_value_data(plan, rawdata; whole)
    validate_structure(plan)
    return plan
end

# A definition optimized into a predictor (a response location or scale, a
# composition's sub-predictor) is that predictor's node, under its own name
# (user decision `1c0jiwz`). A pure alias absorbed into the value it names
# (`w = th` over a plate latent, `w = u` over a scan state) emits no
# statement of its own, so it reads the alias chain's target (user decision
# `0fbe312`; "all intermediate quantities" are part of the RK graph).
function _absorbed_named_values(det, canonmap, skip, predictors)
    pnames = Set{Symbol}(p.name for p in predictors)
    absorbed(nm) = nm in skip && haskey(canonmap, nm)
    named = Pair{Symbol,Symbol}[]
    for (nm, _) in det
        absorbed(nm) && nm ∉ pnames || continue
        target = nm
        seen = Set{Symbol}()
        while target ∉ pnames && absorbed(target) && target ∉ seen &&
                canonmap[target] isa Symbol
            push!(seen, target)
            target = canonmap[target]
        end
        (target in pnames || !absorbed(target)) && push!(named, nm => target)
    end
    return named
end


# ── Destructuring and in-model data values ───────────────────────────
# `(a, b) = rhs` (standard Julia destructuring): each name binds its
# element, `a = _destructure_element(rhs, 1)`, `b = _destructure_element(rhs, 2)`,
# Julia's iteration semantics (an extra element is dropped, a missing one is
# a `BoundsError`). A data-only `rhs` is evaluated once by `bind_data`
# (identical calls share one evaluation), so `(Xf, Zp) = tps_basis(x; k = 4)`
# fits the basis once. The generated kernel reassembles the element reads of
# one `rhs` into a single destructuring statement (`_assignment_statements`),
# so a parameter-dependent `rhs` also runs once, and a function-shaped
# `KernelSpec` call is destructured at its WANT boundary, as RK splices it.
_is_destructuring(st) =
    st isa Expr && st.head === :(=) && length(st.args) == 2 &&
    Meta.isexpr(st.args[1], :tuple) && !isempty(st.args[1].args) &&
    all(a -> a isa Symbol, st.args[1].args)

function _desugar_destructuring(stmts)
    out = Any[]
    for st in stmts
        if _is_destructuring(st)
            lhs, rhs = st.args
            for (i, nm) in enumerate(lhs.args)
                push!(out, Expr(:(=), nm, Expr(:call,
                    GlobalRef(@__MODULE__, :_destructure_element),
                    rhs isa Expr ? copy(rhs) : rhs, i)))
            end
        else
            push!(out, st)
        end
    end
    return out
end

# Element `i` of `(a, b, …) = value`: what Julia's lowering binds through
# `Base.indexed_iterate`. Tuples, named tuples and arrays read it directly.
@inline _destructure_element(value::Union{Tuple,NamedTuple}, i::Int) =
    getfield(value, i)
@inline _destructure_element(value::AbstractArray, i::Int) =
    value[firstindex(value) + i - 1]
function _destructure_element(value, i::Int)
    item, state = Base.indexed_iterate(value, 1)
    for k in 2:i
        item, state = Base.indexed_iterate(value, k, state)
    end
    return item
end

# The `(rhs, i)` of a destructured element read, or `nothing`.
_destructured_read(ex) = nothing
function _destructured_read(ex::Expr)
    ex.head === :call && length(ex.args) == 3 &&
        ex.args[1] == GlobalRef(@__MODULE__, :_destructure_element) &&
        ex.args[3] isa Int || return nothing
    return (ex.args[2], ex.args[3])
end

# Data-only module values (`B = tps_basis(x; k = 4)`, `z = f(x)`: an
# undotted module call reading only data) that a response needs per
# observation, found from every `.~` response's distribution arguments
# through definitions. The arguments broadcast against the response, so a
# model-level value fits there (`Normal.(mu, s)`). One is needed per
# observation only where a model-level value never fits: the matrix of a
# data product (`B * w`), a term of a sum (`a .+ z`; a predictor's terms
# are per observation), or a factor of such a term whose other operands
# are all model-level (`a .+ b .* z`). Beside a per-observation operand
# (`m .* x`) a model-level value broadcasts and stays model-level.
# Module-call arguments, reductions, gathered values and the vector of a
# data product are whole reads. A number bound as data (`_bound_value`)
# is known at lowering and never needs a column.
function _aligned_module_data(sample, det, data)
    detmap = Dict{Symbol,Any}(det)
    cands = Set{Symbol}(nm for (nm, rhs) in det
        if _is_module_value_call(rhs) && !_is_bound_value_call(rhs) &&
            _data_only(rhs, data, detmap))
    aligned = Set{Symbol}()
    isempty(cands) && return aligned
    visited = Set{Tuple{Symbol,Bool}}()
    for s in sample
        s.broadcast && s.levels === nothing && s.matrix === nothing &&
            s.dims === nothing || continue
        (s.lhs in data || haskey(detmap, s.lhs)) || continue
        # Each distribution argument broadcasts against the response.
        rhs = s.rhs
        slots = _is_dotted_call(rhs) ? rhs.args[2].args :
            (rhs isa Expr && rhs.head === :call ? rhs.args[2:end] : Any[rhs])
        for a in slots
            _column_reads!(aligned, a, cands, detmap, visited, data, false)
        end
    end
    return aligned
end

_is_module_value_call(ex) =
    ex isa Expr && ex.head === :call && !isempty(ex.args) &&
    ex.args[1] isa GlobalRef

const _WHOLE_READ_FNS = (:size, :axes, :levels, :eachindex, :Ref)
const _ADDITIVE_OPS = (:+, :-, :.+, :.-)

# An inline data-only index expression is the same bind-time value as its named
# definition. Give it a hygienic name before observation-axis analysis,
# which then routes it through the existing data-definition materializer.
# Only ordinary value compositions are visited: a lazy arm or loop keeps
# its authored evaluation context, just as in `_split_data_calls!`.
function _hoist_data_gather_indices!(sample, det, data, taken; resolve = identity)
    detmap = Dict{Symbol,Any}(det)
    extra = Pair{Symbol,Any}[]
    names = Dict{Any,Symbol}()
    function index_name(ex)
        ex isa Expr || return ex
        # Literal ranges already carry their vector shape. Hoisting one as an
        # opaque module call loses that shape before observation validation.
        _literal_row_range(ex) && return ex
        value = resolve(ex)
        (_contains_module_call(value) || value.head === :ref) &&
            _data_only(value, data, detmap) || return ex
        return get!(names, value) do
            k = length(names) + 1
            nm = Symbol(:_rkppl_index_, k)
            while nm in taken
                k += 1
                nm = Symbol(:_rkppl_index_, k)
            end
            push!(taken, nm)
            push!(extra, nm => value)
            detmap[nm] = value
            nm
        end
    end
    function walk(ex)
        ex isa Expr || return ex
        _is_plate_column_expr(ex) && return ex
        if ex.head === :ref
            return Expr(:ref, walk(ex.args[1]),
                (index_name(walk(a)) for a in ex.args[2:end])...)
        elseif ex.head === :call
            return Expr(:call, ex.args[1], Any[walk(a) for a in ex.args[2:end]]...)
        elseif ex.head === :. && length(ex.args) == 2 &&
                Meta.isexpr(ex.args[2], :tuple)
            return Expr(:., ex.args[1],
                Expr(:tuple, Any[walk(a) for a in (ex.args[2]::Expr).args]...))
        elseif ex.head === :kw && length(ex.args) == 2
            return Expr(:kw, ex.args[1], walk(ex.args[2]))
        elseif ex.head in (:parameters, :tuple, :vect)
            return Expr(ex.head, Any[walk(a) for a in ex.args]...)
        end
        return ex
    end
    out = Pair{Symbol,Any}[nm => walk(rhs) for (nm, rhs) in det]
    for (i, s) in enumerate(sample)
        rhs = walk(s.rhs)
        sample[i] = SampleStmt(s.lhs, rhs, s.broadcast, s.range, s.levels,
            s.matrix, s.dims, s.slices, s.count_columns)
    end
    return append!(out, extra)
end

# Record the data-only module values `ex` needs as observation columns.
# `need`: `ex` itself must carry the observations (a term of a sum); in a
# distribution argument, or beside a per-observation operand, it need not,
# though the matrix of a data product always must (a model-level product
# never mixes with a column).
function _column_reads!(aligned, ex, cands, detmap, visited, data,
        need::Bool = true)
    walk(a, nd = need) =
        _column_reads!(aligned, a, cands, detmap, visited, data, nd)
    if ex isa Symbol
        if ex in cands
            need && push!(aligned, ex)
        elseif haskey(detmap, ex) && !((ex, need) in visited)
            push!(visited, (ex, need))
            walk(detmap[ex])
        end
        return nothing
    end
    ex isa Expr || return nothing
    if ex.head === :ref
        # A gather `v[c]`: the value whole, the index per observation.
        foreach(a -> walk(a, true), ex.args[2:end])
        return nothing
    end
    dotted = _is_dotted_call(ex)
    ex.head === :call || dotted || (foreach(walk, ex.args); return nothing)
    fn = ex.args[1]
    fn isa GlobalRef && !dotted && return nothing
    fn isa Symbol && (fn in REDUCTION_FNS || fn in _WHOLE_READ_FNS) &&
        return nothing
    args = dotted ? ex.args[2].args : ex.args[2:end]
    if fn === :* && !dotted && length(args) == 2
        # A data matrix times a vector: rows from the matrix, the vector
        # whole. (`s * z`, a scalar times a column, reads both.)
        walk(args[1], true)
        _symbols_in(args[1], union(cands, data)) || walk(args[2])
        return nothing
    end
    # Each term of a sum carries the observations itself.
    fn in _ADDITIVE_OPS && return foreach(a -> walk(a, true), args)
    # A product (or other elementwise call) carries them when any operand
    # does; then the others may be model-level and broadcast.
    any(a -> _reads_obs(a, data, detmap, Set{Symbol}()), args) &&
        return foreach(a -> walk(a, false), args)
    foreach(walk, args)
    return nothing
end

# Does `ex` read data per observation (outside whole reads)?
function _reads_obs(ex, data, detmap, active::Set{Symbol})
    if ex isa Symbol
        ex in data && return true
        (haskey(detmap, ex) && !(ex in active)) || return false
        push!(active, ex)
        r = _reads_obs(detmap[ex], data, detmap, active)
        delete!(active, ex)
        return r
    end
    ex isa Expr || return false
    ex.head === :ref && return any(a -> _reads_obs(a, data, detmap, active),
        ex.args[2:end])
    dotted = _is_dotted_call(ex)
    if ex.head === :call || dotted
        fn = ex.args[1]
        fn isa GlobalRef && !dotted && return false
        fn isa Symbol && (fn in REDUCTION_FNS || fn in _WHOLE_READ_FNS) &&
            return false
        args = dotted ? ex.args[2].args : ex.args[2:end]
        return any(a -> _reads_obs(a, data, detmap, active), args)
    end
    return any(a -> _reads_obs(a, data, detmap, active), ex.args)
end

# A per-cell latent declaration reuses the scalar-parameter distribution
# parsing (family, args, HalfNormal/truncated → :positive) and rides the
# plate's range.
function _lower_plate_parameter(name::Symbol, rhs, range, coefuse, matrices, ctx)
    if _external_rhs(rhs)
        args = _external_parameter(name, rhs, (), ctx.mod, ctx.taken;
            observed=name in ctx.conditioned, broadcast=true)
        return PlateParameter(name, :external, args, nothing, range, name)
    end
    sp = _lower_parameter(name, rhs, coefuse, matrices)
    return PlateParameter(sp.name, sp.family, sp.args, sp.support_override,
        range, sp.label)
end

# Shape inference + canonicalization (data-free, Julia-truthful). Shapes:
# data columns are vectors, sampled/det names resolve by position and memo,
# dotted forms follow their operands' observation axis or model-level
# provenance, reductions are scalars, `hcat` is a matrix,
# undotted scalar-array combinations follow Julia exactly (`2*v`, `v*2`,
# `v/2`, `-v` are vectors; `a+x`, `x*z`, `s/x`, `x^2`, `x>1`, `log(x)`
# are `:invalid` — Julia `MethodError`s, reported with the dotted fix).
# Canonicalization rewrites the Julia-valid undotted scalar-array ops
# (`*`, `/`) to dotted-canonical form (Base implements them by broadcast —
# behavior-preserving); matrices never dotted-rewrite (`X * b` keeps its
# shape for predictor classification); every other head passes through.
# Module calls return model-level values; the remaining built-in call
# heads follow their argument shapes, after the vocabulary screen.
# Shape context beyond data and definitions (functions as values):
# `aligned` holds the non-data names that carry the observation axis
# (per-cell latents, scan states, varying bindings); `values` the
# model-level array parameters (Dirichlet simplexes) and whole-value data
# inputs definitions may compute with. Whole data have unknown Julia
# shape, like an undotted module call result (`:scalar` in this analysis).
# `matrices` retains the rank of ordinary hcat values after data-only
# definitions leave the definition table, preventing dotted matmul rewrites.
struct _ShapeEnv
    aligned::Set{Symbol}
    values::Set{Symbol}
    matrices::Set{Symbol}
end
_ShapeEnv(aligned, values) = _ShapeEnv(aligned, values, Set{Symbol}())
const _NO_SHAPE_ENV = _ShapeEnv(Set{Symbol}(), Set{Symbol}())

function _def_shapes(det, data::Set{Symbol}, detmap;
        arrays::Set{Symbol} = Set{Symbol}(), env::_ShapeEnv = _NO_SHAPE_ENV)
    memo = Dict{Symbol,Symbol}(a => :array for a in arrays)
    for (nm, _) in det
        memo[nm] = _shape_of(detmap[nm], data, detmap, memo, Set{Symbol}(),
            env)
    end
    return memo
end

function _shape_of(ex, data, detmap, memo, active::Set{Symbol},
        env::_ShapeEnv = _NO_SHAPE_ENV)
    ex isa Symbol ||
        return _shape_of_expr(ex, data, detmap, memo, active, env)
    ex in env.matrices && return :matrix
    ex in data && return ex in env.values ? :scalar : :vector
    haskey(detmap, ex) || return get(memo, ex, :scalar)
    haskey(memo, ex) && return memo[ex]
    ex in active && return :scalar  # cyclic: errors downstream
    push!(active, ex)
    sh = _shape_of(detmap[ex], data, detmap, memo, active, env)
    delete!(active, ex)
    memo[ex] = sh
    return sh
end

function _shape_of_expr(ex, data, detmap, memo, active,
        env::_ShapeEnv = _NO_SHAPE_ENV)
    ex isa LineNumberNode && return :scalar
    ex isa Expr || return :scalar
    shape(a) = _shape_of(a, data, detmap, memo, active, env)
    head = ex.head
    head === :vect && return :array
    if head === :.
        # Over a declared array value the elementwise array rules apply
        # (`sd .* z` stays an array; `z .+ x` checks Julia's actual axes).
        if _is_dotted_call(ex)
            argsh = [shape(a) for a in ex.args[2].args]
            (:array in argsh || :invalid in argsh) &&
                return _elementwise_shape(argsh)
        end
        if _is_dotted_call(ex) && ex.args[1] isa GlobalRef
            # A module function broadcast is elementwise over its
            # arguments: observation-aligned exactly when one of them is.
            return _obs_axis(ex, data, detmap, memo, active, env) ?
                :vector : :scalar
        end
        return _broadcast_shape(ex, data, detmap, memo, active, env)
    end
    if head === :ref
        base = ex.args[1]
        base in data && length(ex.args) == 2 &&
            _literal_row_range(ex.args[2]) && return :vector
        slice = _observation_slice_shape(ex, data, detmap, memo, active, env)
        slice === nothing || return slice
        _is_array_value_ref(ex, data, detmap, memo, active, env) &&
            return _array_ref_shape(ex, data, detmap, memo, active, env)
        _is_gather(ex, data, detmap, env) || return :scalar
        return _obs_axis(ex, data, detmap, memo, active, env) ?
            :vector : :scalar
    end
    head === Symbol("'") && return shape(ex.args[1]) in (:array, :matrix) ?
        shape(ex.args[1]) : :scalar
    head === :call || return :scalar  # exotic heads: downstream rejects
    isempty(ex.args) && return :scalar
    fn = ex.args[1]
    # Collecting a one-dimensional index iterator preserves its vector
    # domain. It is not an opaque scalar module-call result.
    if fn === GlobalRef(Base, :collect) && length(ex.args) == 2 &&
            Meta.isexpr(ex.args[2], :call)
        iterator = ex.args[2].args[1]
        index_iterator = iterator in (:eachindex, :axes, :(:)) ||
            (iterator isa GlobalRef && getfield(iterator.mod, iterator.name) in
                (Base.eachindex, Base.axes, getfield(Base, :(:))))
        index_iterator && return :vector
    end
    fn === GlobalRef(Base, :hcat) && return length(ex.args) > 1 ? :matrix : :array
    fn isa GlobalRef && fn.mod === (@__MODULE__) && fn.name === :_bound_array_value &&
        return :array
    fn isa Symbol || return :scalar  # module/anonymous calls: model-level
    fn in REDUCTION_FNS && return :scalar
    fn === :_ppl_plate_column && return _plate_column_axis(ex) === nothing ?
        :vector : :array
    argshapes = [shape(a) for a in ex.args[2:end]]
    if fn in ELEMENTWISE_OPS
        (:array in argshapes || :invalid in argshapes) &&
            return _elementwise_shape(argshapes)
        return _broadcast_shape(ex, data, detmap, memo, active, env)
    end
    return _shape_of_call(fn, argshapes)
end

# Elementwise results keep their operands' Julia shape: an observation
# column contributes its row axis, arrays retain their axes, and broadcasts
# over scalars remain scalar.
function _elementwise_shape(argshapes)
    :invalid in argshapes && return :invalid
    # Julia determines broadcast compatibility from the actual axes.
    # A declared array can share the observation axis of another operand.
    :matrix in argshapes && return :matrix
    :vector in argshapes && return :vector
    :array in argshapes && return :array
    return :scalar
end

# `A[:, j]` over a value whose rows are the observations (a bound data
# matrix, a product `X * B` of a data matrix and a declared or returned
# coefficient matrix) keeps that axis, as in Julia: the leading `:`
# selects every row. The other indices are literal positions or `:`, so
# `P[:, 1]` is one value per observation and `P[:, :]` stays a matrix.
function _observation_slice_shape(ex::Expr, data, detmap, memo, active, env)
    base = ex.args[1]
    base isa Union{Symbol,Expr} && length(ex.args) >= 3 &&
        ex.args[2] === :(:) || return nothing
    rest = ex.args[3:end]
    all(_is_position, rest) || return nothing
    _shape_of(base, data, detmap, memo, active, env) === :vector ||
        return nothing
    _obs_axis(base, data, detmap, memo, active, env) || return nothing
    return any(i -> i === :(:), rest) ? :matrix : :vector
end

# A read `A[...]` of an array value: a declared array, an array-valued
# definition (`b = z * M`) or a module call's result.
_is_array_value_ref(ex::Expr, data, detmap, memo, active, env) =
    ex.args[1] isa Symbol && length(ex.args) >= 2 &&
    (_shape_of(ex.args[1], data, detmap, memo, active, env) === :array ||
        _model_valued(ex.args[1], detmap, env, Set{Symbol}()))

# The shape of a read of an array value. A data index gathers per
# observation on either axis, also when the data are otherwise read only
# whole; so does an index carrying the observation axis. A definition's
# shape and the axis an enclosing broadcast inherits both read this one
# rule, so naming a gather never changes its shape: `rg = b[g, 1]` and
# `(b[g, 1] .+ b[h, 1]) ./ 2` are per observation alike.
_array_ref_shape(ex::Expr, data, detmap, memo, active, env) =
    _ref_shape(ex, i -> (i isa Symbol && i in data) ||
        _obs_axis(i, data, detmap, memo, active, env))

# Reads of an array (`z[g]` per observation, `phi[1]` scalar, `L[:, 1]`
# array); every other ref keeps the slice-1 scalar shape. Predictor
# coefficient roles classify by use site, independently of this shape.
function _ref_shape(ex, gathers)
    idx = ex.args[2:end]
    isempty(idx) && return :scalar
    if any(gathers, idx)
        # Per-observation gather: scalar selection or an oriented matrix.
        any(i -> i === :(:), idx) && return :matrix
        return :vector
    end
    any(i -> i === :(:), idx) && return :array
    return :scalar
end

# A broadcast has an observation axis exactly when an operand has one.
# Broadcasting scalar values remains scalar, as in Julia.
function _broadcast_shape(ex, data, detmap, memo, active, env)
    _obs_axis(ex, data, detmap, memo, active, env) && return :vector
    return :scalar
end

# `A[i]` with one index over a value (data, a definition, a call result,
# or a model-level array parameter) is a gather: it carries the
# observation axis exactly when its index does. Its predictor role is
# decided separately (`z[g]` may be a factor coefficient there).
_is_gather(ex::Expr, data, detmap, env::_ShapeEnv) =
    ex.head === :ref && length(ex.args) == 2 &&
    (ex.args[1] isa Expr || ex.args[1] in data ||
        haskey(detmap, ex.args[1]) || ex.args[1] in env.values ||
        ex.args[1] in env.aligned)

# Does `ex` carry the observation axis? Undotted module calls and
# reductions take whole values and return model-level ones; a gather
# follows its index.
function _obs_axis(ex, data, detmap, memo, active, env)
    if ex isa Symbol
        ex in data && return !(ex in env.values)
        ex in env.aligned && return true
        haskey(detmap, ex) || return false
        return _shape_of(ex, data, detmap, memo, active, env) in
            (:vector, :matrix)
    end
    ex isa Expr || return false
    if ex.head === :call && !isempty(ex.args)
        fn = ex.args[1]
        fn isa Symbol || return false
        fn in REDUCTION_FNS && return false
        fn === :_ppl_plate_column && return _plate_column_axis(ex) === nothing
        return any(a -> _obs_axis(a, data, detmap, memo, active, env),
            ex.args[2:end])
    elseif _is_dotted_call(ex)
        return any(a -> _obs_axis(a, data, detmap, memo, active, env),
            ex.args[2].args)
    elseif ex.head === :ref
        _observation_slice_shape(ex, data, detmap, memo, active, env) ===
            nothing || return true
        # A gather from an array value follows its observation index on
        # either axis, including definitions such as `b = z * M`.
        array_base = _is_array_value_ref(ex, data, detmap, memo, active, env)
        array_base || _is_gather(ex, data, detmap, env) || return false
        any(_literal_row_range, ex.args[2:end]) && return true
        array_base && return _array_ref_shape(ex, data, detmap, memo,
            active, env) in (:vector, :matrix)
        return any(i -> _obs_axis(i, data, detmap, memo, active, env),
            ex.args[2:end])
    elseif ex.head === Symbol("'")
        return _obs_axis(ex.args[1], data, detmap, memo, active, env)
    end
    return false
end

# Does `ex` compute with a model-level value beyond scalars — a module
# call result or an array parameter, directly or through definitions?
function _model_valued(ex, detmap, env, active::Set{Symbol})
    ex isa GlobalRef && return true
    if ex isa Symbol
        ex in env.values && return true
        haskey(detmap, ex) || return false
        ex in active && _sfail("cyclic definition through $ex")
        push!(active, ex)
        r = _model_valued(detmap[ex], detmap, env, active)
        delete!(active, ex)
        return r
    end
    ex isa Expr || return false
    # Walk every dependency even when the callable head already proves
    # this is a value, so retaining names still rejects cyclic definitions.
    valued = false
    for a in ex.args
        valued |= _model_valued(a, detmap, env, active)::Bool
    end
    return valued
end

# Single rule table for undotted `:call` shapes over argument shapes.
function _shape_of_call(fn::Symbol, argshapes::Vector{Symbol})
    :invalid in argshapes && return :invalid
    fn in ELEMENTWISE_OPS && return _elementwise_shape(argshapes)
    :array in argshapes && return _array_call_shape(fn, argshapes)
    nvec = count(==(:vector), argshapes)
    nmat = count(==(:matrix), argshapes)
    if fn === :hcat
        # Design-matrix construction is always matrix-shaped (arg
        # validation — intercept `ones(length(x))` plus vector columns — lives in
        # matrix-def extraction, not here).
        return :matrix
    elseif nmat > 0 && fn in (:+, :-)
        return nmat == length(argshapes) ? :matrix : :invalid
    elseif nmat > 0 && fn === :/
        return argshapes == [:matrix, :scalar] ? :matrix : :invalid
    elseif nmat > 0 && fn !== :*
        # Slice D1 admits matrices only in predictor matmuls (`mu =
        # X * b`); every other matrix operation fails closed at the
        # mismatch message below.
        return :invalid
    elseif fn === :+ || fn === :-
        length(argshapes) == 1 && return only(argshapes)
        return nvec == 0 ? :scalar :
            nvec == length(argshapes) ? :vector : :invalid
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

# Undotted calls over array values follow Julia: `*` is the matrix
# product (a data matrix times an array is per-observation — `B * w`;
# arrays and scalars scale), `+`/`-` add equal-shaped arrays, `/`
# divides by a scalar; elementwise math is dotted. Other heads follow
# their arguments (per-observation if any is, array otherwise).
function _array_call_shape(fn::Symbol, argshapes::Vector{Symbol})
    fn in ELEMENTWISE_OPS && return _elementwise_shape(argshapes)
    if fn === :*
        length(argshapes) == 2 || return :invalid
        a, b = argshapes
        (a === :matrix || a === :vector) && b === :array && return :vector
        a in (:scalar, :array) && b in (:scalar, :array) && return :array
        return :invalid
    elseif fn === :+ || fn === :-
        all(==(:array), argshapes) && return :array
        return :invalid
    elseif fn === :/
        length(argshapes) == 2 && argshapes[1] === :array &&
            argshapes[2] === :scalar && return :array
        return :invalid
    elseif fn === :^ || _is_plain_comparison(fn) || fn === :ifelse ||
            fn in ASSIGNMENT_FNS
        return :invalid
    end
    :vector in argshapes && return :vector
    return :array
end

_is_plain_comparison(fn::Symbol) =
    fn === :< || fn === :> || fn === :(==) || fn === :(!=) ||
    fn === :(<=) || fn === :(>=)

# `strict = false` (broadcast arguments, response bounds) only performs the
# Julia-valid dotted rewrites below: a shape this analysis cannot admit is
# left exactly as written for its own downstream check, never refused here.
function _canonical_expr(ex, data, detmap, detshape, env, where;
        strict::Bool = true)
    ex isa Symbol && return ex
    ex isa Expr || return ex
    if ex.head === :parameters
        strict || return ex
        _sfail("$where takes positional arguments only (no keywords)")
    end
    if _is_dotted_call(ex)
        # Julia evaluates each broadcast argument whole before
        # broadcasting, so `exp.(-cl .* t)` reads `-cl` exactly as the
        # definition `nc = -cl` does: its arguments take the same dotted
        # rewrites. Keywords pass through untouched.
        args = Any[Meta.isexpr(a, (:parameters, :kw)) ? a :
            _canonical_expr(a, data, detmap, detshape, env, where;
                strict = false)
            for a in ex.args[2].args]
        return Expr(:., ex.args[1], Expr(:tuple, args...))
    end
    ex.head === :call || return ex
    isempty(ex.args) && return ex
    fn = ex.args[1]
    fn isa Symbol || return ex
    fn in REDUCTION_FNS && return ex  # args validated downstream
    fn === :_ppl_plate_column && return ex  # an RK plate (array cells)
    args = Any[_canonical_expr(a, data, detmap, detshape, env, where; strict)
        for a in ex.args[2:end]]
    argshapes = [_canon_shape(a, data, detmap, detshape, env) for a in args]
    if :invalid in argshapes
        return Expr(ex.head, ex.args[1], args...)  # broken ref: raises at its own def
    end
    if _shape_of_call(fn, argshapes) === :invalid
        strict || return Expr(ex.head, ex.args[1], args...)
        _sfail(_julia_mismatch_msg(fn, where, ex, argshapes))
    end
    if fn === :* && length(args) > 2 && :matrix ∉ argshapes &&
            :array ∉ argshapes && :vector in argshapes
        # Keep the scalar products in Julia order, then normalize only
        # the scalar-vector multiplication to its broadcast equivalent.
        return _canonical_expr(Expr(:call, :*,
            Expr(:call, :*, args[1:end-1]...), args[end]), data, detmap,
            detshape, env, where; strict)
    end
    if fn in (:+, :-) && length(args) >= 2 && all(==(:vector), argshapes)
        return Expr(:call, fn === :+ ? :.+ : :.-, args...)
    end
    if (fn === :* || fn === :/) && length(args) == 2
        # Matrices never dotted-rewrite: `X * b` keeps its shape for
        # predictor classification (which checks the coefficient
        # declaration); anything else with a matrix operand already
        # failed above.
        if fn === :* && :matrix ∉ argshapes && :array ∉ argshapes &&
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

# Canonicalization and predictor classification use the same definition
# context and rules as inference. A shape alone cannot distinguish a
# scalar from a module call's model-level value: both are `:scalar`, but
# only the latter keeps a built-in broadcast model-level. Re-running a
# separate dotted => vector classifier here used to lose that provenance.
_canon_shape(ex, data, detmap, detshape, env) =
    _shape_of(ex, data, detmap, detshape, Set{Symbol}(), env)
_canon_shape(ex, ctx) =
    _canon_shape(ex, ctx.data, ctx.detmap, ctx.detshape, ctx.shape_env)

function _julia_mismatch_msg(fn::Symbol, where, ex, argshapes)
    if :array in argshapes
        :vector in argshapes && return "$where combines an array value " *
            "with a per-observation column: `$(repr(ex))` — read the " *
            "array per observation (`z[g]`, `B * w` with a data matrix " *
            "`B`) or by position (`z[1]`)"
        return "$where combines an array value as Julia does not: " *
            "`$(repr(ex))` — elementwise math over arrays is dotted " *
            "(`sd .* z`, `exp.(z)`, `a .+ z`), `*` is the matrix product " *
            "(`B * w` with a data matrix `B`), and `+`/`-` add arrays of " *
            "one shape"
    end
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

_is_coef_prior_call(rhs) =
    rhs isa Expr && rhs.head === :call && !isempty(rhs.args) &&
    rhs.args[1] isa Symbol && haskey(_COEF_FAMILIES, rhs.args[1])


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
        delete!(active, nm)
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
    # Index syntax (`z[g, :]`, `v[end]`) is not a name.
    (s === :(:) || s === :end || s === :begin) && return false
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

# `composed_maps`: the definition-level screen runs before composition
# analysis, so a definition that will inline into a composed tree
# (`bump = logistic.(xi) .* tm`) may carry the composed elementwise maps;
# `logistic.` still fails with the link guidance wherever it reaches the
# predictor analysis (`_analyze_predictor` re-screens strictly).
function _reject_unknown_calls(where, rhs; composed_maps::Bool = false)
    rhs isa Expr || return nothing
    rhs.head === :ref && return nothing
    if rhs.head === :call && !isempty(rhs.args)
        fn = rhs.args[1]
        # `hcat` passes the vocabulary screen everywhere (args still
        # recurse below): matrix definitions validate at extraction;
        # strays fail there (definitions) or at predictor analysis
        # (responses) with bind-to-a-name guidance.
        fn === :_ppl_plate_column && return nothing  # array-cell plate
        if fn isa Symbol && fn ∉ ELEMENTWISE_OPS && fn ∉ ASSIGNMENT_FNS &&
                fn !== :hcat
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
            _sfail("$where calls `$fn`, which is neither a built-in " *
                   "value function nor defined in the model module — " *
                   "define the function there (or import it) before " *
                   "lowering")
        end
    elseif rhs.head === :.
        if _is_dotted_call(rhs) && rhs.args[1] isa GlobalRef
            # A module function broadcast: its arguments still screen.
            for a in rhs.args[2].args
                _reject_unknown_calls(where, a; composed_maps)
            end
            return nothing
        end
        length(rhs.args) == 2 && rhs.args[1] isa Symbol &&
            rhs.args[2] isa Expr && rhs.args[2].head === :tuple ||
            return nothing  # malformed dotted: downstream rejects
        f = rhs.args[1]
        if !(f === :ifelse || f in ELEMENTWISE_FNS ||
                (composed_maps && f in _COMPOSED_UNARY))
            f in _COMPOSED_UNARY && _sfail("$where calls `$f.`, which " *
                "is not in the slice-1 value vocabulary")
            _sfail("$where calls `$f.`, which is neither a built-in " *
                "value function nor defined in the model module — " *
                "define the function there (or import it) before lowering")
        end
    end
    for a in rhs.args
        _reject_unknown_calls(where, a; composed_maps)
    end
    return nothing
end

# ── Functions as values ───────────────────────────────────────────────
# An `=` definition may call any function visible in the model's module.
# Call heads the built-in value vocabulary owns keep their built-in
# meaning, and names with dedicated guidance keep that guidance (the
# screen above still explains them). Every other head resolves to a
# `GlobalRef` in the defining module — the model's, or a submodel's own —
# checked to exist at lowering, read-only (`isdefined`/`getfield`, no
# eval). A `GlobalRef` call is plain Julia over whole values: undotted it
# yields a model-level value (no observation axis), dotted it broadcasts
# elementwise. Data-only definitions that call one are evaluated once by
# `bind_data`; the rest run in the generated kernel under generic AD.

const _CONSTRUCT_VALUE_HEADS = (:hcat,)

_builtin_value_head(fn::Symbol) =
    fn in ELEMENTWISE_OPS || fn in ASSIGNMENT_FNS ||
    fn in REDUCTION_FNS || fn in _CONSTRUCT_VALUE_HEADS ||
    startswith(string(fn), ".") || fn === :treatment || fn === :ifelse ||
    fn in _RESPONSE_ONLY_FNS

_builtin_dotted_head(f::Symbol) =
    f === :ifelse || f in ELEMENTWISE_FNS || f in _COMPOSED_UNARY

# A dotted built-in scalar function (`var.(cols)`, `tanh.(v)`) broadcasts
# the vocabulary's own binding — the one undotted `var(x)` means.
_vocabulary_ref(f::Symbol) = GlobalRef(PPLGeneratedModels, f)

function _resolve_module_path(ex, mod::Module, where)
    m = if ex isa Symbol
        isdefined(mod, ex) ? getfield(mod, ex) : nothing
    elseif ex isa Expr && ex.head === :. && length(ex.args) == 2 &&
            ex.args[2] isa QuoteNode && ex.args[2].value isa Symbol
        parent = _resolve_module_path(ex.args[1], mod, where)
        s = ex.args[2].value
        isdefined(parent, s) ? getfield(parent, s) : nothing
    else
        nothing
    end
    m isa Module || _sfail("$where qualifies a call with `$(repr(ex))`, " *
                           "which is not a module visible in " *
                           "`$(nameof(mod))`")
    return m
end

const _RETIRED_CONSTRUCT_MODELS = Dict(
    :varying_effect => :varying_coefs,
    :varying_draws => :varying_coefs,
    :varying_slice => :varying_coefs,
    :mm => :varying_coefs,
    :gr => :varying_stratified,
    :dummy => :varying_coefs,
    :spline => :penalized_smooth,
    :mo => :monotonic,
    :mo1 => :monotonic,
    :dar => :differenced_ar1,
    :hsgp => :hsgp_effect)

const _RETIRED_MODEL_HEADS = (
    :ordered_logistic, :penalized_smooth, :t2_smooth, :hsgp_effect, :hsgp_periodic_effect, :hsgp_grouped_effect, :monotonic, :differenced_ar1, :r2d2_coefs, :horseshoe_coefs, :varying_coefs, :varying_coefs_correlated, :varying_coefs_centered, :varying_coefs_centered_correlated, :varying_stratified, :varying_stratified_correlated)

const _RETIRED_PK_HEADS = (:linear_pk_schedule, :linear_pk_read_locs,
    :linear_pk_read_locs_auc, :build_linear_pk_schedule,
    :linear_pk_event_log_f, :linear_pk_op_log_dose,
    :linear_pk_system_3, :linear_pk_propagate_3, :linear_pk_add_dose_3,
    :linear_pk_add_regular_doses_3, :transit_twocmt_unit,
    :transit_twocmt_unit_response, :transit_twocmt_rule,
    :prepare_transit_twocmt_rule,
    :QT_COUPLING_SPINES, :QT_OBS_FAMILIES, :admit_qt_spine,
    :admit_qt_obs_family, :qt_loc_assignment, :qt_obs_statement,
    :pk_obs_statement, :validate_qt_joint_prep, :TGIOptions, :tgi_options,
    :TGI_OBSERVATIONS, :TGI_STRUCTURES, :TGI_THRESHOLDS, :TGI_MEASURES,
    :TGI_TIME_SCALE_H, :TGI_LOG_PR, :TGI_LOG_PD, :TGI_RECIST_LOG_PR,
    :TGI_RECIST_LOG_PD, :tgi_measure_dim, :tgi_threshold_scale,
    :tgi_fixed_cutpoints, :tgi_estimated_cutpoints, :tgi_uses_nadir,
    :tgi_ratio_loglinear, :tgi_ratio_resistant, :tgi_log_survival,
    :tgi_running_nadir, :tgi_nadir_scan_expr, :tgi_segmented_nadir,
    :tgi_inv_logit, :tgi_normal_lcdf, :tgi_log_diff_exp,
    :tgi_interval_logprob, :tgi_report_logprob, :tgi_category_lpmf,
    :tgi_category_lpmfs, :tgi_response_lpmf, :tgi_response_lpmfs,
    :tgi_censored_lpdf, :tgi_censored_lpdfs, :tgi_category_stmts,
    :tgi_response_stmts, :tgi_censored_stmts, :TGI_CELL_FUNCTIONS)

const _RETIRED_PREPARATION_HEADS = (:tps_basis, :cr_basis, :t2_basis,
    :hsgp_basis, :hsgp_sqrt_spd, :hsgp_rho_floors,
    :hsgp_periodic_basis, :hsgp_matern_sqrt_spd, :hsgp_grouped_sqrt_spd,
    :hsgp_periodic_sqrt_spd, :hsgp_periodic_grouped_sqrt_spd,
    :hsgp_periodic_rho_floor, :gp_exp_quad_cov, :gp_periodic_cov, :gp_chol_latent)

function _module_binding(m::Module, s::Symbol, where, shown)
    if !isdefined(m, s)
        s in _RETIRED_PREPARATION_HEADS && _sfail("$where: `$shown` is statistical preparation owned by BayesianRegressionModels.StatisticalPreparation; import the helper into the model module and call it as an ordinary function")
        s in _RETIRED_PK_HEADS && _sfail("$where: `$shown` is downstream PK code; import its replacement from RKPPLBench and call it as an ordinary function")
        haskey(_RETIRED_CONSTRUCT_MODELS, s) && _sfail("$where: the implicit `$shown` " *
            "construct is retired; write ordinary priors and values, or use " *
            "`BayesianRegressionModels.rkppl_model(:$(_RETIRED_CONSTRUCT_MODELS[s]))`")
        s in _RETIRED_MODEL_HEADS && _sfail("$where: the statistical model `$shown` " *
            "is owned by BayesianRegressionModels; obtain its ordinary body with " *
            "`BayesianRegressionModels.rkppl_model(:$s)` and bind it in the model module")
        s === :linear_pk_log_f && _sfail("$where: the PK-specific model `$shown` " *
            "belongs to downstream RKPPLBench; import the consumer-owned submodel " *
            "into the model module")
        s === :Horseshoe && _sfail("$where: the implicit `Horseshoe` " *
            "coefficient shortcut is retired; declare its priors and " *
            "coefficient product explicitly, or use the BRM-owned " *
            "`BayesianRegressionModels.rkppl_model(:horseshoe_coefs)`")
        s === :LKJCovarianceFactor && _sfail("$where: the implicit " *
            "`LKJCovarianceFactor` shortcut is retired; declare " *
            "`sd[1:K] .~ Exponential.(scale)` (or your scale prior), " *
            "`C ~ LKJCholesky(K, eta)` and `F = sd .* C` explicitly")
        _sfail("$where calls `$shown`, which is not defined " *
            "in module `$(nameof(m))` — define the function there (or import " *
            "it) before lowering")
    end
    v = getfield(m, s)
    v isa Module && _sfail("$where calls `$shown`, which is a module, not " *
                           "a function")
    v isa RKPPLSubmodel && _sfail("$where calls submodel `$shown` with " *
        "`=` — a submodel binds with `~` (`x ~ $shown(...)`)")
    return GlobalRef(m, s)
end

# A built-in call head's meaning: the head itself, or the vocabulary
# binding a dotted built-in broadcasts; `nothing` for any other name.
function _builtin_call_head(fn::Symbol; dotted::Bool = false)
    if dotted
        _builtin_dotted_head(fn) && return fn
        (fn in ASSIGNMENT_FNS && !(fn in ELEMENTWISE_OPS)) &&
            return _vocabulary_ref(fn)
        return nothing
    end
    return _builtin_value_head(fn) ? fn : nothing
end

# Resolve a call head; built-in heads come back unchanged.
function _resolve_call_head(fn, mod::Module, names::Set{Symbol}, where;
        dotted::Bool = false)
    fn isa GlobalRef && return fn
    if fn isa Symbol
        builtin = _builtin_call_head(fn; dotted)
        builtin === nothing || return builtin
        fn in names && _sfail("$where calls `$fn`, which is a model value, " *
                              "not a function")
        return _module_binding(mod, fn, where, fn)
    end
    if fn isa Expr && fn.head === :. && length(fn.args) == 2 &&
            fn.args[2] isa QuoteNode && fn.args[2].value isa Symbol
        m = _resolve_module_path(fn.args[1], mod, where)
        return _module_binding(m, fn.args[2].value, where, repr(fn))
    end
    return fn  # anonymous or exotic heads: the screen rejects them
end

# A bare function name passed to a module function (`map(abs2, v)`) is the
# function value, as in Julia; model names shadow it.
function _resolve_function_arg(a, mod::Module, names::Set{Symbol}, where)
    if a isa Symbol
        (a in names || !isdefined(mod, a)) && return a
        getfield(mod, a) isa Union{Function,KernelSpec} || return a
        return GlobalRef(mod, a)
    elseif a isa Expr && a.head === :kw && length(a.args) == 2
        return Expr(:kw, a.args[1],
            _resolve_function_arg(a.args[2], mod, names, where))
    elseif a isa Expr && a.head === :parameters
        return Expr(:parameters, Any[_resolve_function_arg(p, mod, names,
            where) for p in a.args]...)
    elseif a isa Expr && a.head === :. && length(a.args) == 2 &&
            a.args[2] isa QuoteNode && a.args[2].value isa Symbol
        a.args[1] isa Symbol && a.args[1] in names && return a
        m = _resolve_module_path(a.args[1], mod, where)
        return _module_binding(m, a.args[2].value, where, repr(a))
    end
    return a
end

_is_dotted_call(ex) =
    ex isa Expr && ex.head === :. && length(ex.args) == 2 &&
    ex.args[2] isa Expr && ex.args[2].head === :tuple

# `nothing` and `missing` keep Julia's meaning: Base's values, whether
# written as names or interpolated by an AST emitter. A model name shadows
# the name, as any Julia binding would.
_resolve_module_calls(::Nothing, ::Module, ::Set{Symbol}, where) =
    GlobalRef(Base, :nothing)
_resolve_module_calls(::Missing, ::Module, ::Set{Symbol}, where) =
    GlobalRef(Base, :missing)

function _resolve_module_calls(ex, mod::Module, names::Set{Symbol}, where)
    ex in (:pi, :π) && ex ∉ names && return Float64(pi)
    ex in (:nothing, :missing) && ex ∉ names && return GlobalRef(Base, ex)
    ex isa Expr || return ex
    ex.head === :quote && return ex
    # The quoted cell becomes executable in the generated kernel's module.
    # Resolve its callable heads here, retaining lexical cell-local names.
    if _is_plate_column_call(ex)
        spec = ex.args[2]
        body = spec.args[2].value
        locals = Set{Symbol}(st.args[1] for st in body.args
            if Meta.isexpr(st, :(=), 2) && st.args[1] isa Symbol)
        cell_names = union(names, locals, Set([spec.args[1].value]))
        resolved = copy(spec.args)
        for i in (2, 3)
            resolved[i] = QuoteNode(_resolve_module_calls(spec.args[i].value,
                mod, cell_names, where))
        end
        return Expr(:call, ex.args[1], Expr(:tuple, resolved...), ex.args[3:end]...)
    end
    if ex.head === :call && !isempty(ex.args)
        head = _resolve_call_head(ex.args[1], mod, names, where)
        args = Any[_resolve_module_calls(a, mod, names, where)
            for a in ex.args[2:end]]
        if head isa GlobalRef
            args = Any[_resolve_function_arg(a, mod, names, where)
                for a in args]
        end
        return Expr(:call, head, args...)
    elseif _is_dotted_call(ex)
        head = _resolve_call_head(ex.args[1], mod, names, where;
            dotted = true)
        args = Any[_resolve_module_calls(a, mod, names, where)
            for a in ex.args[2].args]
        if head isa GlobalRef
            args = Any[_resolve_function_arg(a, mod, names, where)
                for a in args]
        end
        return Expr(:., head, Expr(:tuple, args...))
    end
    return Expr(ex.head, Any[_resolve_module_calls(a, mod, names, where)
        for a in ex.args]...)
end

_contains_module_call(ex) =
    ex isa GlobalRef ||
    (ex isa Expr && any(_contains_module_call, ex.args))

# ── Module calls inside sampling arguments ────────────────────────────
# A built-in sampling RHS spells a distribution — families, wrappers,
# `Ref`/`eachrow`/`eachcol` sharing, a mixture's `vcat.` component list,
# probability links (and their legacy misspellings), ordinal structure
# and link tags — around ordinary Julia values. A module call written
# inline among those values means what its named definition means:
# naming a subexpression never changes legality or the density
# (rkppl-use §2, §9).
const _RHS_SPELLING_HEADS = (:Ref, :eachrow, :eachcol, :Cumulative,
    :StoppingRatio)

_rhs_spelling_head(fn, dotted::Bool) = _builtin_rhs_head(fn) ||
    (fn isa Symbol && (haskey(_BERNOULLI_LINKS, fn) ||
        haskey(_LEGACY_INVERSE_LINKS, fn) || haskey(_ORDINAL_LINKS, fn))) ||
    (dotted ? fn === :vcat : fn in _RHS_SPELLING_HEADS)

# Apply `f` to each value argument of a built-in sampling RHS.
_map_rhs_values(f, ex) = _map_rhs_arguments((value, _, _, _) -> f(value), ex)

# Apply `f(value, head, position, nargs)` to each value argument of a
# built-in sampling RHS: positional argument `position` of the `nargs` that
# `head` spells (`Normal.(mu, s)`: `s` is 2 of 2 of `:Normal`), or the
# keyword's name as `position`.
function _map_rhs_arguments(f, ex, head = nothing, position = 0, nargs = 0)
    if ex isa Expr && ex.head === :call && !isempty(ex.args) &&
            _rhs_spelling_head(ex.args[1], false)
        return Expr(:call, ex.args[1],
            _map_rhs_positions(f, ex.args[2:end], _rhs_head_symbol(ex.args[1]))...)
    elseif _is_dotted_call(ex) && _rhs_spelling_head(ex.args[1], true)
        return Expr(:., ex.args[1], Expr(:tuple,
            _map_rhs_positions(f, ex.args[2].args, _rhs_head_symbol(ex.args[1]))...))
    elseif ex isa Expr && ex.head === :parameters
        return Expr(:parameters,
            Any[_map_rhs_arguments(f, a, head, position, nargs) for a in ex.args]...)
    elseif ex isa Expr && ex.head === :kw && length(ex.args) == 2
        return Expr(:kw, ex.args[1],
            _map_rhs_arguments(f, ex.args[2], head, ex.args[1], nargs))
    end
    return f(ex, head, position, nargs)
end

function _map_rhs_positions(f, args, head)
    nargs = count(a -> !Meta.isexpr(a, :parameters), args)
    out, position = Any[], 0
    for a in args
        Meta.isexpr(a, :parameters) ||
            (position += 1)
        push!(out, _map_rhs_arguments(f, a, head,
            Meta.isexpr(a, :parameters) ? 0 : position, nargs))
    end
    return out
end

_rhs_head_symbol(fn::Symbol) = fn
_rhs_head_symbol(fn::GlobalRef) = fn.name
_rhs_head_symbol(fn::Expr) =
    Meta.isexpr(fn, :., 2) && fn.args[2] isa QuoteNode ? fn.args[2].value : nothing
_rhs_head_symbol(fn) = nothing

# A call `_resolve_call_head` resolves in the model module rather than
# keeping as a built-in: resolution names an undefined function or a
# model value used as one exactly as for the named definition.
function _module_call(ex)
    dotted = _is_dotted_call(ex)
    dotted || Meta.isexpr(ex, :call) && !isempty(ex.args) || return false
    return _module_call_head(ex.args[1], dotted)
end
_module_call_head(fn::GlobalRef, dotted) = fn.mod !== PPLGeneratedModels
# A range keeps its literal shape in place (`_literal_row_range`).
_module_call_head(fn::Symbol, dotted) =
    fn !== :(:) && _builtin_call_head(fn; dotted) === nothing
_module_call_head(fn::Expr, dotted) =
    fn.head === :. && length(fn.args) == 2 && fn.args[2] isa QuoteNode
_module_call_head(fn, dotted) = false

_has_module_call(ex) = ex isa Expr && ex.head !== :quote &&
    (_module_call(ex) || any(_has_module_call, ex.args))

# Heads whose operands Julia evaluates conditionally or repeatedly: a
# module call inside one stays inside, so `bind` takes the whole node.
const _DEFERRED_VALUE_HEADS = (:if, :elseif, :&&, :||, :comprehension,
    :generator, :->, :do, :let, :block)

# Replace each module call in the value `ex` by `bind(call)`. An index
# endpoint (`end`, `begin`) belongs to its enclosing reference, so a call
# reading one stays there.
function _bind_module_calls(bind, ex)
    _has_module_call(ex) || return ex
    if !_mentions_symbol(ex, :end) && !_mentions_symbol(ex, :begin)
        _module_call(ex) && return bind(ex)
        ex.head in _DEFERRED_VALUE_HEADS && return bind(ex)
    end
    return Expr(ex.head, Any[_bind_module_calls(bind, a) for a in ex.args]...)
end

# The names a statement's left-hand side observes or declares, including
# the count columns of `eachrow(hcat(c1, c2))`.
function _sampled_lhs_names!(out::Set{Symbol}, lhs)
    Meta.isexpr(lhs, :call) && !isempty(lhs.args) &&
        lhs.args[1] in (:eachrow, :eachcol, :hcat) ||
        return _lhs_base_names!(out, lhs)
    foreach(a -> _sampled_lhs_names!(out, a), lhs.args[2:end])
    return out
end

# An observation's inline module call is the value its named definition
# holds (`w = f(b .* x)` then `y .~ Normal.(w, s)`), so name it: the
# definition then takes exactly the named spelling's path — bind-time
# evaluation when it reads only data, whole-value data inputs, predictor
# and value analysis. A call that is a whole distribution argument is named
# after the response and that argument's role (`y .~ Normal.(mu, f(x))`
# names `y_scale`); any other after the response and its function
# (`y_f`). Identical calls share one name. Prior arguments have their own
# hoist (`_hoist_prior_args!`); external RHSs resolve their arguments
# themselves.
function _observation_call_definitions(ast::Expr, data::Set{Symbol},
        mod::Module)
    defined = Set{Symbol}(st.args[1] for st in ast.args
        if Meta.isexpr(st, :(=), 2) && st.args[1] isa Symbol)
    names = _statement_lhs_names!(union(data, defined), ast)
    taken = union(data, _expr_names(ast))
    named = Dict{Any,Symbol}()
    out = Any[]
    function name!(call, stem::Symbol, role)
        get!(named, call) do
            value = _resolve_module_calls(call, mod, names,
                "response $stem argument `$(repr(call))`")
            base = Symbol(stem, :_, something(role, _call_label(call)))
            nm, k = base, 1
            while nm in taken || nm in _synthesized_response_names(stem)
                k += 1
                nm = Symbol(base, :_, k)
            end
            push!(taken, nm)
            push!(out, Expr(:(=), nm, value))
            nm
        end
    end
    for st in ast.args
        if st isa Expr && (_is_sample(st) || _is_broadcast_sample(st)) &&
                !_external_rhs(st.args[3])
            observed = _sampled_lhs_names!(Set{Symbol}(), st.args[2])
            if any(in(data), observed) ||
                    (_is_broadcast_sample(st) && any(in(defined), observed))
                stem = first(sort!(collect(observed)))
                rhs = _map_rhs_arguments(st.args[3]) do value, head, position, nargs
                    _bind_module_calls(value) do call
                        role = call === value ?
                            _argument_role(head, position, nargs) : nothing
                        name!(call, stem, role)
                    end
                end
                rhs == st.args[3] ||
                    (st = Expr(:call, st.args[1], st.args[2], rhs))
            end
        end
        push!(out, st)
    end
    return Expr(:block, out...)
end

# A whole argument's role: its keyword's name, or its positional role.
_argument_role(head, position::Symbol, nargs) = position
_argument_role(head, position::Int, nargs) =
    _distribution_role(head, position, nargs)

# A computed argument's label when its position has no role: its function's
# name (`f` for `f(x)`, `M.f(x)` or `f.(x)`), or `call` for a deferred
# expression holding one (`if`, a comprehension, ...).
function _call_label(ex)
    fn = Meta.isexpr(ex, :call) ? ex.args[1] :
        _is_dotted_call(ex) ? ex.args[1] : nothing
    return something(_rhs_head_symbol(fn), :call)
end

# The names the response lowering synthesizes for response `stem`
# (`y_eta`, `y_loc`, ...; `_lower_location`, `_latent_predictor!`,
# `_lower_argument_predictor!`, the `_resp` likelihood label). A computed
# argument never takes one.
_synthesized_response_names(stem::Symbol) =
    (Symbol(stem, :_eta), Symbol(stem, :_loc), Symbol(stem, :_value),
     Symbol(stem, :_resp), Symbol(stem, :_disc))

# A gather from an inline value is the gather from its named definition:
# `loc = a .+ (z .* tau)[1, g]` reads as `r = z .* tau; loc = a .+ r[1, g]`
# (naming a subexpression never changes legality or the density). So name
# the gathered value: its definition then takes the named spelling's path —
# the axes it inherits from declared arrays, level lookup on a `levels(g)`
# axis, one value per observation. A data-only value already folds with its
# gather at binding. Lazy arms and `@plate` / `@scan` bodies keep their
# evaluation context. Identical values share one name. Runs after
# `_observation_call_definitions`, so a module call in a response argument
# is already a name here.
function _gathered_value_definitions(ast::Expr, data::Set{Symbol})
    detmap = Dict{Symbol,Any}(st.args[1] => st.args[2] for st in ast.args
        if Meta.isexpr(st, :(=), 2) && st.args[1] isa Symbol)
    taken = union(data, _expr_names(ast))
    named = Dict{Any,Symbol}()
    out = Any[]
    function name!(value)
        get!(named, value) do
            k = 1
            nm = Symbol(:_rkppl_gathered_, k)
            while nm in taken
                k += 1
                nm = Symbol(:_rkppl_gathered_, k)
            end
            push!(taken, nm)
            push!(out, Expr(:(=), nm, value))
            nm
        end
    end
    function walk(ex)
        ex isa Expr || return ex
        ex.head in (:macrocall, :quote) && return ex
        ex.head in _DEFERRED_VALUE_HEADS && return ex
        args = Any[walk(a) for a in ex.args]
        if ex.head === :ref && length(args) >= 2 &&
                _gathered_value(args[1]) && !_data_only(args[1], data, detmap)
            args[1] = name!(args[1])
        end
        return Expr(ex.head, args...)
    end
    defined = Set{Symbol}(keys(detmap))
    for st in ast.args
        if Meta.isexpr(st, :(=), 2) && st.args[1] isa Symbol
            st = Expr(:(=), st.args[1], walk(st.args[2]))
        elseif st isa Expr && (_is_sample(st) || _is_broadcast_sample(st)) &&
                !_external_rhs(st.args[3])
            observed = _sampled_lhs_names!(Set{Symbol}(), st.args[2])
            if any(in(data), observed) ||
                    (_is_broadcast_sample(st) && any(in(defined), observed))
                rhs = _map_rhs_values(walk, st.args[3])
                rhs == st.args[3] ||
                    (st = Expr(:call, st.args[1], st.args[2], rhs))
            end
        end
        push!(out, st)
    end
    return Expr(:block, out...)
end

# A computed value that a gather may read: a call, a broadcast or an
# adjoint. Names, literals, property paths and chained reads stay as written.
_gathered_value(ex) = ex isa Expr &&
    (ex.head === :call || ex.head === Symbol("'") || _is_dotted_call(ex))

# Data-only: every value the expression reads is data or a data-only
# definition (literals and function values aside).
function _data_only(ex, data, detmap, active::Set{Symbol} = Set{Symbol}())
    for s in _value_symbols(ex)
        s in data && continue
        (haskey(detmap, s) && !(s in active)) || return false
        push!(active, s)
        ok = _data_only(detmap[s], data, detmap, active)
        delete!(active, s)
        ok || return false
    end
    return true
end

# Declarations keep named dimensions, prior arguments and support bounds.
# Follow their definitions before predictor inlining, so shared data values
# still bind once and are consumed by name everywhere else. `observed` holds
# the data and derived-response names: their statements are observations
# whose locations lower as predictors, not declarations.
function _declaration_dependencies(sample, plate_specs, observed, detmap)
    needed = Set{Symbol}()
    for s in sample
        s.lhs in observed && continue
        for ex in (s.rhs, s.range, s.matrix)
            union!(needed, _value_symbols(ex))
        end
        s.levels === nothing || push!(needed, first(s.levels))
        s.dims === nothing || foreach(d -> union!(needed, _value_symbols(d)), s.dims)
    end
    for (_, rhs, range, _) in plate_specs, ex in (rhs, range)
        union!(needed, _value_symbols(ex))
    end
    pending = collect(needed)
    while !isempty(pending)
        nm = pop!(pending)
        haskey(detmap, nm) || continue
        for dep in _value_symbols(detmap[nm])
            dep in needed && continue
            push!(needed, dep)
            push!(pending, dep)
        end
    end
    return needed
end

# Data columns the definitions read only as whole values — the bind-time
# model-level data-input rule (`_model_level_inputs`) over the
# definitions, with each per-observation statement read (`held`: a
# response, prior, `@plate`, …) pinned observation-aligned.
# `_confirm_whole_value_data` checks these inputs against the finished plan.
function _whole_value_data(det, data::Set{Symbol}, held::Set{Symbol})
    inputs, _ = _whole_value_reads(det, data, held)
    return setdiff!(inputs, held)
end

# Names held by non-definition statements. A gathered
# value is whole even in a response (`v[g]`); its index is aligned. Plate
# cells distinguish whole calls from indexed lane reads; scans retain
# conservative statement-wide alignment. Extracted
# kernel cells remain consumers too: a schedule-chain definition moved out
# of `det` still reads its subject-level predictors. Otherwise those now
# apparently unused definitions would be classified as whole values.
function _statement_names(ast::Expr, known::Set{Symbol};
        whole_priors::Set{Symbol} = Set{Symbol}())
    out = Set{Symbol}()
    whole = Set{Symbol}()
    for st in ast.args
        st isa LineNumberNode && continue
        st isa Expr && st.head === :(=) && st.args[1] isa Symbol && continue
        # Only the indexed spelling puts the loop in argument 3. Legacy
        # `@plate result for ...` keeps its conservative whole-statement reads.
        if st isa Expr && st.head === :macrocall &&
                st.args[1] === Symbol("@plate") &&
                length(st.args) == 3 && Meta.isexpr(st.args[3], :for)
            loop = st.args[3]
            ivar = loop.args[1].args[1]
            _all_symbols!(out, loop.args[1].args[2])
            for c in loop.args[2].args
                c isa Expr && (_is_sample(c) || _is_broadcast_sample(c)) || continue
                _classify_cell_reads!(whole, out, c.args[2], known, ivar)
                _classify_cell_reads!(whole, out, c.args[3], known, ivar)
            end
        elseif st isa Expr && st.head === :macrocall
            _all_symbols!(out, st)
        elseif st isa Expr && (_is_sample(st) || _is_broadcast_sample(st)) &&
                _merge_stem(_stmt_lhs(st)) in whole_priors
            _classify_reads!(whole, out, _stmt_lhs(st), known, false)
            _classify_reads!(whole, out, last(st.args), known, true)
        else
            _classify_reads!(whole, out, st, known, false)
        end
    end
    free = copy(known)
    return union!(out, setdiff(known, free))
end

function _classify_cell_reads!(whole, aligned, ex, known, ivar, full=false)
    if ex isa Expr && ex.head === :ref && length(ex.args) == 2 && ex.args[2] === ivar
        _classify_reads!(whole, aligned, ex.args[1], known, false)
    elseif ex isa Expr && ex.head === :call
        context = full || _cell_whole_call(ex.args[1])
        for a in ex.args[2:end]
            _classify_cell_reads!(whole, aligned, a, known, ivar, context)
        end
    elseif _is_dotted_call(ex)
        for a in ex.args[2].args
            _classify_cell_reads!(whole, aligned, a, known, ivar, full)
        end
    else
        _classify_reads!(whole, aligned, ex, known, full)
    end
    return nothing
end

# The names of `among` that `ex` reads, directly or through definitions.
function _reached_names(ex, detmap, among::Set{Symbol},
        out::Set{Symbol} = Set{Symbol}(), seen::Set{Symbol} = Set{Symbol}())
    for s in _value_symbols(ex)
        s in among && push!(out, s)
        if haskey(detmap, s) && !(s in seen)
            push!(seen, s)
            _reached_names(detmap[s], detmap, among, out, seen)
        end
    end
    return out
end

# Explicit whole-value inputs must keep their model-level role in the
# finished plan. Ordinary function calls may also read observation data.
function _confirm_whole_value_data(plan::StructuralPlan, data::Set{Symbol};
        whole::Set{Symbol} = Set{Symbol}())
    isempty(whole) && return nothing
    inputs, definitions = _model_level_inputs(plan, data)
    values = union(inputs, definitions)
    whole ⊆ values || _sfail("whole-value data also read per observation: " *
        join(sort!(collect(setdiff(whole, values))), ", "))
    return nothing
end

# Emission skip set: locations never emit; absorbed definitions emit only
# while a non-skipped definition (or a response scale/nu / parameter
# argument) still names them. Fixpoint: skipping cascades through
# absorbed-only reference chains (chained intermediates vanish entirely).
function _absorbed_skip(det, canonmap, responses, paramsyms, absorbed,
        used_locs)
    skip = Set{Symbol}(used_locs)
    while true
        refs = Set{Symbol}(paramsyms)
        for r in responses
            r.scale isa Symbol && push!(refs, r.scale)
            r.nu isa Symbol && push!(refs, r.nu)
            r.threshold_effects isa Symbol && push!(refs, r.threshold_effects)
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

# Legacy panel sampling syntax has an ordinary indexed-plate replacement.
function _is_retired_panel_sample(st::Expr)
    (_is_sample(st) || _is_broadcast_sample(st)) || return false
    rhs = st.args[3]
    Meta.isexpr(rhs, :do) && !isempty(rhs.args) || return false
    call = first(rhs.args)
    return Meta.isexpr(call, :call) && !isempty(call.args) && first(call.args) === :plate
end

function _partition_statements(ast::Expr, data::Set{Symbol})
    sample = SampleStmt[]
    det = Pair{Symbol,Any}[]
    scans = ScanSpec[]
    joints = JointSampleStmt[]
    glms = GLMSampleStmt[]
    level_bindings = Dict{Symbol,Tuple{Symbol,Any}}()
    seen = Set{Symbol}()
    seelines = Dict{Symbol,Int}()
    # Definitions observed by a `.~` response (derived responses —
    # either statement order; the double-observation guard).
    derived_observed = Set{Symbol}()
    seen_doc = false
    line = 0
    cell_broadcasts = Dict{Symbol,Vector{Symbol}}()
    args, plate_ctx, plate_params = _expand_plates(ast.args, data;
        broadcasts = cell_broadcasts)
    # Defined names, collected before lowering so the checks below admit
    # forward references. The scan never throws — the main loop below owns
    # every rejection.
    detnames = Set{Symbol}()
    valueaxisnames = Set{Symbol}()
    definitions = Dict{Symbol,Any}()
    for arg in args
        arg isa Expr || continue
        st = try
            _unwrap_trivia(arg)
        catch
            continue
        end
        if st.head === :(=) && length(st.args) == 2 && st.args[1] isa Symbol
            push!(detnames, st.args[1])
            _is_levels_binding_rhs(st.args[2]) || push!(valueaxisnames, st.args[1])
            definitions[st.args[1]] = st.args[2]
        end
    end
    # Definitions a response may observe: those reading only data (a
    # literal-only definition is not built yet as a response).
    response_defs = Set{Symbol}(name for (name, rhs) in definitions
        if !isempty(_value_symbols(rhs)) && _data_only(rhs, data, definitions))
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
        # Parsed into a `ScanSpec` here; each carried array is claimed as a
        # model-level latent name.
        if arg.head === :macrocall && arg.args[1] === Symbol("@scan")
            (length(arg.args) >= 3 && arg.args[end] isa Expr) ||
                _sfail("@scan takes a `begin … end` block")
            sp = parse_scan_block(arg.args[end])
            _screen_scan(sp, data)
            for st in sp.states
                _claim!(seen, seelines, st, line)
            end
            push!(scans, sp)
            continue
        end
        arg.head === :block &&
            _sfail("nested `begin` blocks do not lower — flatten the block")
        st = _unwrap_trivia(arg)
        if Meta.isexpr(st, :call) && !isempty(st.args) &&
                first(st.args) in (:spline_basis, :hsgp_basis)
            _sfail("the implicit `$(first(st.args))(:id, ...)` declaration is retired; " *
                "use BRM-owned statistical preparation and an ordinary " *
                "`BayesianRegressionModels.rkppl_model` body with explicit priors")
        end
        if _is_retired_panel_sample(st)
            _sfail("the panel `plate(...) do` sampling form is retired; use ordinary `@plate for i in eachindex(y)` statements and explicit arrays")
        end

        if st isa Expr && st.head === :call && !isempty(st.args) &&
                first(st.args) === :r2d2
            _sfail("the implicit `r2d2(...)` declaration is retired; state " *
                "its priors and coefficient values explicitly, or use the " *
                "BRM-owned `BayesianRegressionModels.rkppl_model(:r2d2_coefs)`")
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
            if !bc && _is_glm_call(st.args[3])
                g = _parse_glm_stmt(st, line, data)
                _claim!(seen, seelines, g.response, line)
                push!(glms, g)
                continue
            end
            if bc && _is_glm_call(st.args[3])
                _sfail("GLM-object heads use whole-data `~`, not `.~` " *
                       "(`$(st.args[3].args[1])(X, alpha, beta)` — the " *
                       "object owns eta over the whole column)")
            end
            counts = _multinomial_rows_lhs(st.args[2], bc, data)
            if counts !== nothing
                for c in counts
                    _claim!(seen, seelines, c, line)
                end
                _reject_target(st.args[3], counts[1])
                push!(sample, SampleStmt(counts[1], st.args[3], true,
                    nothing, nothing, nothing, nothing, nothing, counts[2:end]))
                continue
            end
            pl = _per_level_lhs(st.args[2], st.args[3], bc)
            if pl !== nothing
                plhs, pdims = pl
                _claim!(seen, seelines, plhs, line)
                _reject_target(st.args[3], plhs)
                push!(sample, SampleStmt(plhs, st.args[3], false, nothing,
                    nothing, nothing, pdims, :per_level))
                continue
            end
            sl = _array_slices_lhs(st.args[2], st.args[3], bc, tilde, data)
            if sl !== nothing
                rlhs, rdims, slices = sl
                _claim!(seen, seelines, rlhs, line)
                _reject_target(st.args[3], rlhs)
                push!(sample, SampleStmt(rlhs, st.args[3], bc, nothing,
                    nothing, nothing, rdims, slices))
                continue
            end
            # A data-only definition observed at selected entries
            # (`y[eachindex(y)] .~ …`, which a `@plate` loop over `y` writes)
            # is a derived response, read like the same value bound as data.
            derived_range = _derived_response_range(st.args[2], bc, response_defs)
            if !derived_range
                _reject_derived_ref_lhs(st.args[2], detnames)
                arr = _external_array_lhs(st.args[2], st.args[3], data, valueaxisnames)
                arr === nothing && (arr = _array_sample_lhs(st.args[2], bc, tilde, data, valueaxisnames))
                if arr !== nothing
                    alhs, adims = arr
                    _claim!(seen, seelines, alhs, line)
                    _reject_target(st.args[3], alhs)
                    push!(sample, SampleStmt(alhs, st.args[3], bc, nothing,
                        nothing, nothing, adims))
                    continue
                end
            end
            lhs, rng, levs, mat = _sample_lhs(st.args[2], bc, tilde,
                derived_range ? union(data, (st.args[2].args[1],)) : data,
                level_bindings)
            if bc && (st.args[2] isa Symbol || derived_range)
                # A derived response observes an existing deterministic
                # definition (`ly = log.(earn)` then `ly .~ ...`) instead
                # of claiming a fresh name; either order lowers (the
                # varying forward-reference precedent). A second `.~`
                # over the same name stays a double definition.
                if lhs in seen
                    if lhs in detnames && lhs ∉ derived_observed
                        push!(derived_observed, lhs)
                    elseif lhs in derived_observed
                        _sfail("$lhs is already observed by a `.~` " *
                                "response (one response per column)")
                    else
                        _claim!(seen, seelines, lhs, line)
                    end
                else
                    _claim!(seen, seelines, lhs, line)
                    lhs in detnames && push!(derived_observed, lhs)
                end
            else
                _claim!(seen, seelines, lhs, line)
            end
            _reject_target(st.args[3], lhs)
            push!(sample, SampleStmt(lhs, st.args[3], bc, rng, levs, mat))
        elseif st.head === :(=) && length(st.args) == 2 && st.args[1] isa Symbol
            lhs = st.args[1]
            lhs === :target && _sfail("no `target` in rkppl models " *
                                      "(density comes only from `~`)")
            lhs in data && _sfail("$lhs is bound data and cannot be redefined")
            # Forward `.~`: the observation claimed first — this first
            # definition completes it (a second definition still throws).
            forward_response = lhs in seen && lhs in derived_observed &&
                !any(p -> p.first === lhs, det)
            forward_response || _claim!(seen, seelines, lhs, line)
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
    return sample, det, plate_ctx, plate_params, scans, joints, glms, cell_broadcasts
end

# ── Design-matrix extraction (slice D1) ─────────────────────────────
# `X = hcat(ones(length(x)), x, ...)` definitions leave `det` for the plan-level
# `matrices` table (the ranef-bucket/spline-basis precedent: special
# statements lower to plan tables, not kernel assignments). The generator
# emits each matrix once (`X = Float64.(hcat(...))`); predictor matmuls
# (`mu = X * b`) reference it by name. Columns are the intercept `ones(length(x))`
# plus bare data/derived-data columns. Other constructions and consumers
# keep ordinary Julia values through _route_hcat_values. Duplicate columns
# and double intercepts fail in this affine optimization's validation.

_is_hcat_def(rhs) =
    rhs isa Expr && rhs.head === :call && !isempty(rhs.args) &&
    rhs.args[1] === :hcat

# Only Base's ones/length have this recipe. A same-named module function
# keeps its own meaning. The length anchor must also be a matrix column:
# bind validates their equal lengths, so recreating the ones column is exact.
_base_matrix_call(ex, name, f) =
    ex isa Expr && ex.head === :call && length(ex.args) == 2 &&
    (ex.args[1] === name || (ex.args[1] isa GlobalRef &&
        getfield(ex.args[1].mod, ex.args[1].name) === f))

function _matrix_intercept_anchor(ex)
    _base_matrix_call(ex, :ones, Base.ones) || return nothing
    len = ex.args[2]
    _base_matrix_call(len, :length, Base.length) || return nothing
    return len.args[2] isa Symbol ? len.args[2] : nothing
end

function _matrix_columns(nm, args, column)
    isempty(args) && _sfail("matrix `$nm` calls `hcat` with no columns")
    anchor = findfirst(a -> a isa Symbol, args)
    example = anchor === nothing ? "x" : string(args[anchor])
    cols = Union{Nothing,Symbol}[]
    for a in args
        if a isa Number && !(a isa Bool) && a == 1
            _sfail("matrix `$nm`: scalar `1` is not a vector intercept " *
                "in Julia's `hcat`; use `ones(length($example))`, or keep " *
                "the intercept outside the matrix (`X = hcat($example, ...)`; " *
                "`mu = a .+ X * b`)")
        elseif a isa Symbol
            push!(cols, column(a))
        else
            c = _matrix_intercept_anchor(a)
            c !== nothing && c in args || _sfail("matrix `$nm` has a " *
                "non-column argument $(repr(a)) — use bare vector columns " *
                "or `ones(length($example))` anchored to one of them; " *
                "bind richer expressions to a name first")
            push!(cols, nothing)
        end
    end
    return cols
end

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
        cols = _matrix_columns(nm, args, c -> _matrix_column(nm, c,
            detshape, detmap, data, prior_names, plate_names, scans))
        push!(matrices, DesignMatrix(nm, cols, nm))
    end
    # Strays in the remaining definitions: inline `hcat` binds to a name;
    # matrix names lower only in predictor matmuls.
    matnames = Set{Symbol}(m.name for m in matrices)
    for (nm, rhs) in kept
        _find_hcat(rhs) && _sfail("definition `$nm` calls `hcat` outside " *
                                  "a matrix definition — bind the matrix " *
                                  "to a name first (`X = hcat(ones(length(x)), x, ...)`)")
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
               "matrices take the intercept `ones(length(x))` plus vector columns")
    end
    c in prior_names && _sfail("design matrix `$nm` over sampled " *
                               "parameter `$c` is not in slice D1 " *
                               "(data/derived columns only)")
    c in plate_names && _sfail("design matrix `$nm` over the latent " *
                               "vector `$c` is not in slice D1 (the me " *
                               "mirror stays affine)")
    any(s -> c in s.states, scans) && _sfail("design matrix `$nm` over " *
                                             "scan state `$c` is not in " *
                                             "slice D1 (data/derived " *
                                             "columns only)")
    _sfail("design matrix `$nm` over unknown name `$c` — columns are " *
           "the intercept `ones(length(x))` or bare data/derived columns")
end

# Keep the affine column-table recipe only for its supported construction
# and consumers. Every other hcat keeps Base's ordinary value semantics.
function _ordinary_hcat_values(sample, det, data, glms)
    detmap = Dict{Symbol,Any}(det)
    mats = Set{Symbol}(nm for (nm, rhs) in det if _is_hcat_def(rhs))
    glm_mats = Set{Symbol}(g.matrix for g in glms)
    setdiff!(mats, glm_mats)
    generic = Set{Symbol}()
    isempty(mats) && return generic
    params = Set{Symbol}(s.lhs for s in sample if s.lhs ∉ data)
    coefvec = Set{Symbol}(s.lhs for s in sample
        if s.broadcast && s.matrix !== nothing)
    arrays = Set{Symbol}(s.lhs for s in sample if s.dims !== nothing ||
        s.levels !== nothing || s.matrix !== nothing ||
        _is_dirichlet_call(s.rhs) || _is_ordered_call(s.rhs))
    shape(ex) = _shape_of(ex, data, detmap, Dict{Symbol,Symbol}(
        a => :array for a in arrays), Set{Symbol}())
    pure = Set{Symbol}()
    for nm in mats
        args = detmap[nm].args[2:end]
        isempty(args) && (push!(generic, nm); continue)
        all(a -> a isa Symbol && a in data ||
            (_matrix_intercept_anchor(a) !== nothing &&
             _matrix_intercept_anchor(a) in args), args) && push!(pure, nm)
        for a in args
            if a isa Symbol
                # Unknown names retain their ordinary scope diagnostic.
                a ∉ data && !haskey(detmap, a) && a ∉ params && continue
                (a in params || shape(a) !== :vector) && push!(generic, nm)
            elseif _matrix_intercept_anchor(a) === nothing ||
                    _matrix_intercept_anchor(a) ∉ args
                push!(generic, nm)
            end
        end
    end
    legacy = intersect(pure, _hcat_value_reads(det, detmap, sample, data,
        glms, arrays, Set{Symbol}()))
    used = Set{Symbol}()
    function walk(ex; whole = false)
        if ex isa Symbol
            if ex in mats
                push!(used, ex)
                whole && ex in legacy || push!(generic, ex)
            end
            return nothing
        end
        ex isa Expr || return nothing
        a = ex.args
        if ex.head === :call && length(a) == 3 && a[1] === :* &&
                a[2] isa Symbol && a[2] in mats
            push!(used, a[2])
            # Construct-owned legacy priors resolve an undeclared bare
            # coefficient vector later. Keep that candidate on the affine
            # route; ordinary models still require its stated prior.
            implicit = a[3] isa Symbol && a[3] ∉ data &&
                a[3] ∉ params && !haskey(detmap, a[3])
            if !(a[3] isa Symbol && a[3] in coefvec) &&
                    !implicit &&
                    !(a[2] in legacy && shape(a[3]) === :array)
                push!(generic, a[2])
            end
            walk(a[3])
            return nothing
        end
        fn = ex.head === :call && !isempty(a) ? a[1] : nothing
        valuecall = fn isa GlobalRef ||
            (fn isa Symbol && fn ∉ _MATRIX_OPERAND_HEADS)
        wh = whole || valuecall || ex.head in (:ref, Symbol("'"))
        for x in a
            walk(x; whole = wh)
        end
        return nothing
    end
    for (nm, rhs) in det
        nm in mats && continue
        walk(rhs)
    end
    for s in sample
        s.matrix in mats && push!(used, s.matrix)
        walk(s.rhs)
    end
    union!(generic, setdiff(mats, used))
    return generic
end

function _resolve_hcat_values(ex; keep = false, resolve = identity)
    ex isa Expr || return ex
    ex.head === :quote && return ex
    if _is_hcat_def(ex)
        head = keep ? :hcat : GlobalRef(Base, :hcat)
        value = Expr(:call, head, Any[_resolve_hcat_values(a; resolve)
            for a in ex.args[2:end]]...)
        return keep ? value : resolve(value)
    end
    return Expr(ex.head, Any[_resolve_hcat_values(a; resolve) for a in ex.args]...)
end

function _route_hcat_values(sample, det, data, glms; resolve = identity)
    generic = _ordinary_hcat_values(sample, det, data, glms)
    det = Pair{Symbol,Any}[nm => _resolve_hcat_values(rhs;
        keep = _is_hcat_def(rhs) && nm ∉ generic, resolve) for (nm, rhs) in det]
    sample = SampleStmt[SampleStmt(s.lhs, _resolve_hcat_values(s.rhs; resolve),
        s.broadcast, s.range, s.levels, s.matrix, s.dims, s.slices,
        s.count_columns) for s in sample]
    matrices = Set{Symbol}(nm for (nm, rhs) in det if nm in generic &&
        rhs isa Expr && rhs.head === :call && length(rhs.args) > 1 &&
        rhs.args[1] === GlobalRef(Base, :hcat))
    return sample, det, matrices
end

# ── `hcat` matrices read as values ───────────────────────────────────
# `X = hcat(...)` is a matrix. A program that uses it only as `X * w`,
# with `w[axes(X, 2)] .~ Fam.(args...)` a coefficient prior the matrix
# term carries (or `w` a free name), lowers it as a design matrix whose
# coefficient vector is `w`. Three reads need the matrix as a VALUE:
#
# - a definition passing `X` to a function, indexing it or taking its
#   adjoint (`var.(eachcol(X))`, `size(X, 2)`, `X[:, 1]`, `X'`);
# - `X * v` with `v` array-valued (a declared array, `z .* s`);
# - a coefficient prior on `w[axes(X, 2)]` the term cannot carry.
#
# Such an `X` lowers exactly like a bound data matrix `X`: `w[axes(X, 2)]`
# declarations are arrays, `X * w` is the data matrix-vector product, and
# data-only definitions read the matrix. Both routes use the same matrix
# (intercept `ones(length(x))` = a ones column), so the density is the same. Any other
# use (`mu = X`, `a .+ X`, `X * s` with `s` a scalar, `X` as a location or
# scale) keeps the design-matrix route and its refusal.

# The `hcat` definitions of `det` the program reads as values. GLM
# response matrices stay design matrices (their families read them).
function _hcat_value_reads(det, detmap, sample, data, glms,
        simplexes::Set{Symbol}, aligned::Set{Symbol})
    mats = Set{Symbol}(nm for (nm, rhs) in det if _is_hcat_def(rhs))
    for g in glms
        delete!(mats, g.matrix)
    end
    out = Set{Symbol}()
    isempty(mats) && return out
    coefvec = Dict{Symbol,Symbol}(s.lhs => s.matrix for s in sample
        if s.lhs ∉ data && s.broadcast && s.matrix !== nothing &&
            s.matrix in mats)
    arrays = union(simplexes, Set{Symbol}(s.lhs for s in sample
        if s.lhs ∉ data && (s.dims !== nothing ||
            (s.broadcast && (s.matrix !== nothing || s.levels !== nothing)) ||
            (!s.broadcast && _is_lkj_cholesky_call(s.rhs)))))
    memo = Dict{Symbol,Symbol}(a => :array for a in arrays)
    env = _ShapeEnv(aligned, arrays)
    shape(ex) = _shape_of(ex, data, detmap, memo, Set{Symbol}(), env)
    scalar(a::Symbol) = a ∉ data && a ∉ aligned && a ∉ arrays &&
        shape(a) === :scalar
    # `X * w` reads `X` as a value when `w` is array-valued. A vector
    # declared over a matrix (`w[axes(S, 2)]`) stays with the design-matrix
    # route, which checks its sizing against `X`; only the coefficient
    # prior can make it an array (below).
    sized_by = Set{Symbol}(s.lhs for s in sample
        if s.lhs ∉ data && s.broadcast && s.matrix !== nothing)
    valued(X, w, composed) = (composed || !(w isa Symbol && w in sized_by)) &&
        (w isa Symbol || w isa Expr) && shape(w) === :array
    # Naming a subexpression never changes its classification: a definition
    # read inside a composition (`p = X * b; mu = f(p)`, or a submodel's
    # returned product) is composed exactly as its inline spelling
    # (`mu = f(X * b)`) is, so its own right-hand side is rescanned in that
    # context until no further definition is reached.
    composed_defs = Set{Symbol}()
    for (nm, rhs) in det
        nm in mats && continue
        _matrix_value_reads!(out, rhs, mats, valued, true;
            defs = detmap, composed_defs)
    end
    for s in sample
        _matrix_value_reads!(out, s.rhs, mats, valued, false;
            defs = detmap, composed_defs)
    end
    rescanned = Set{Symbol}()
    while !issubset(composed_defs, rescanned)
        nm = first(setdiff(composed_defs, rescanned))
        push!(rescanned, nm)
        _matrix_value_reads!(out, detmap[nm], mats, valued, true;
            affine = false, defs = detmap, composed_defs)
    end
    for s in sample
        X = get(coefvec, s.lhs, nothing)
        X === nothing && continue
        _matrix_term_prior(s.rhs, scalar) || push!(out, X)
    end
    return out
end

# Arithmetic heads: `X` as one of their operands is not a value read.
const _MATRIX_OPERAND_HEADS = (:+, :-, :*, :/, :^, :\, ELEMENTWISE_OPS...)

# Value reads of the matrices `mats` in `ex`: `X * v` with `valued(X, v)`,
# and, inside a definition (`indef`), `X` as an argument of a function call
# (`eachcol(X)`, `size(X, 2)`), an indexed `X[...]`, or `X'`. A definition
# name of `defs` read in a composed (non-affine) position joins
# `composed_defs`, for the caller to rescan its right-hand side.
function _matrix_value_reads!(out::Set{Symbol}, ex, mats, valued, indef::Bool;
        affine::Bool = true, defs = nothing, composed_defs = nothing)
    if ex isa Symbol
        !affine && defs !== nothing && haskey(defs, ex) && ex ∉ mats &&
            push!(composed_defs, ex)
        return out
    end
    ex isa Expr || return out
    a = ex.args
    if ex.head === :call && length(a) == 3 && a[1] === :* &&
            a[2] isa Symbol && a[2] in mats
        valued(a[2], a[3], !affine) && push!(out, a[2])
        return _matrix_value_reads!(out, a[3], mats, valued, indef;
            affine = false, defs, composed_defs)
    end
    if indef
        isarg(x) = x isa Symbol && x in mats
        if ex.head === :call && !isempty(a) &&
                !(a[1] isa Symbol && a[1] in _MATRIX_OPERAND_HEADS)
            union!(out, Iterators.filter(isarg, a[2:end]))
        elseif (ex.head === :ref || ex.head === Symbol("'")) && isarg(a[1])
            push!(out, a[1])
        end
    end
    # Distribution constructors and their argument tuples preserve the
    # affine context. Only a mathematical composition requires a declared
    # array operand to take the ordinary matrix-value route.
    fn = (ex.head === :call || _is_dotted_call(ex)) && !isempty(a) ? a[1] : nothing
    composed = fn in _MATRIX_OPERAND_HEADS || fn in ELEMENTWISE_FNS ||
        fn in ASSIGNMENT_FNS || fn isa GlobalRef ||
        (fn isa Expr && fn.head === :.)
    child_affine = affine && !(composed && fn ∉ (:+, :.+, :-, :.-))
    for x in a
        _matrix_value_reads!(out, x, mats, valued, indef;
            affine = child_affine, defs, composed_defs)
    end
    return out
end

# Whether a matrix term carries the broadcast coefficient prior `rhs`: a
# coefficient family over literals, scalar names, or literal vectors of
# those (the `_matrix_prior_arg` grammar). A non-family head
# (`truncated.(...)`, `Exponential.(...)`) or an array-valued argument
# (`lambda .* tau`, `sd` over an array) is an array prior. A RHS that is
# not a dotted call stays with the term, whose error names the fix.
function _matrix_term_prior(rhs, scalar)
    rhs isa Expr && rhs.head === :. && length(rhs.args) == 2 &&
        rhs.args[1] isa Symbol && rhs.args[2] isa Expr &&
        rhs.args[2].head === :tuple || return true
    haskey(_COEF_FAMILIES, rhs.args[1]) || return false
    elt(x) = x isa Real || x === :Inf || (x isa Symbol && scalar(x))
    return all(a -> elt(a) ||
        (a isa Expr && a.head === :vect && all(elt, a.args)),
        rhs.args[2].args)
end

# The value matrices' plan records: `bind_data` builds each from its
# columns (the intercept `ones(length(x))` is a ones column, as in a design matrix), so
# a column is a ones column or a bound data column.
function _value_design_matrices(defs, data::Set{Symbol})
    out = DesignMatrix[]
    for (nm, rhs) in defs
        args = rhs.args[2:end]
        cols = _matrix_columns(nm, args, a -> begin
            if a ∉ data
                _sfail("matrix `$nm = $rhs` is read as a value, " *
                       "so `bind_data` builds it from its columns: each " *
                       "is a bound vector column, got " *
                       "$(repr(a)) (bind it as data)")
            end
            a
        end)
        push!(out, DesignMatrix(nm, cols, nm))
    end
    return out
end

# Pre-pass: expand top-level `@plate for i in R ... end` blocks into
# spliced top-level statements (desugar slice: observations +
# deterministic cells; per-cell sampled arrays deferred). Spliced
# statements carry the plate's line for claim messages. Also returns the
# plate context: `(lhs, line, bare-symbols)` per spliced statement for
# the post-analysis whole-vector check.
function _expand_plates(args, data::Set{Symbol};
        broadcasts::Dict{Symbol,Vector{Symbol}} = Dict{Symbol,Vector{Symbol}}())
    definitions = Dict{Symbol,Any}(arg.args[1] => arg.args[2] for arg in args
        if arg isa Expr && arg.head === :(=) && length(arg.args) == 2 &&
            arg.args[1] isa Symbol)
    plate_data = union(data, Set{Symbol}(name for (name, rhs) in definitions
        if _data_only(rhs, data, definitions)))
    expanded = Any[]
    ctx = Tuple{Symbol,Int,Set{Symbol}}[]
    params = Tuple{Symbol,Any,Union{Nothing,UnitRange{Int},Symbol,Expr},Int}[]
    # Cell locals belong to their plate, as a loop body's locals do in
    # Julia. A local name another plate already used is respelled `m#k`;
    # whole-column evaluation splices its locals at model level, so a
    # model statement reading one is refused instead of reading the cell.
    taken = union(data, _expr_names(Expr(:block, args...)))
    plate_locals = Set{Symbol}()
    spliced = Dict{Symbol,Tuple{Symbol,Int}}()
    line = 0
    for arg in args
        if arg isa LineNumberNode
            line = arg.line
            push!(expanded, arg)
            continue
        end
        if arg isa Expr && arg.head === :macrocall && !isempty(arg.args) &&
                arg.args[1] === Symbol("@plate") && length(arg.args) == 3
            # Ordinary indexed plates lower over their authored range.
            pl = line
            if length(arg.args) >= 2 && arg.args[2] isa LineNumberNode
                pl = arg.args[2].line
            end
            arg, locals = _scope_plate_locals(arg, plate_locals, taken)
            stmts, stx, prm = _desugar_plate(arg, pl, plate_data; broadcasts)
            for st in stmts
                pl > 0 && push!(expanded, LineNumberNode(pl))
                push!(expanded, st)
                Meta.isexpr(st, :(=), 2) && haskey(locals, st.args[1]) &&
                    (spliced[st.args[1]] = (locals[st.args[1]], pl))
            end
            append!(ctx, stx)
            append!(params, prm)
            continue
        end
        push!(expanded, arg)
    end
    isempty(spliced) && return expanded, ctx, params
    for arg in args
        arg isa Expr || continue
        Meta.isexpr(arg, :macrocall) && !isempty(arg.args) &&
            arg.args[1] === Symbol("@plate") && continue
        defined = Meta.isexpr(arg, :(=), 2) ? arg.args[1] : nothing
        for name in _value_symbols(arg)
            name === defined && continue
            haskey(spliced, name) || continue
            authored, pl = spliced[name]
            at = pl > 0 ? " at line $pl" : ""
            _sfail("`$authored` is a cell local of the `@plate`$at; cell " *
                   "locals stay in their cell — assign it to an indexed " *
                   "output of the plate to read its values outside")
        end
    end
    return expanded, ctx, params
end

# Bare cell locals (`m = ...` inside the loop) of one plate, respelled
# `m#k` when an earlier plate already used the name. Returns the plate
# and its locals' current spelling => authored name.
function _scope_plate_locals(st::Expr, plate_locals::Set{Symbol}, taken::Set{Symbol})
    loop = st.args[3]
    Meta.isexpr(loop, :for) && Meta.isexpr(loop.args[2], :block) ||
        return st, Dict{Symbol,Symbol}()
    names = Symbol[]
    for c in loop.args[2].args
        Meta.isexpr(c, :(=), 2) && c.args[1] isa Symbol &&
            !(c.args[1] in names) && push!(names, c.args[1])
    end
    locals = Dict{Symbol,Symbol}()
    body = loop.args[2]
    for name in names
        spelling = name
        if name in plate_locals
            k = 2
            while Symbol(name, "#", k) in taken
                k += 1
            end
            spelling = Symbol(name, "#", k)
            body = _respell_value(body, name, spelling)
        end
        push!(taken, spelling)
        push!(plate_locals, name)
        locals[spelling] = name
    end
    body === loop.args[2] && return st, locals
    return Expr(st.head, st.args[1:2]..., Expr(:for, loop.args[1], body)), locals
end

# Rename a value symbol (never a call head, keyword name or field name).
_respell_value(ex, from::Symbol, to::Symbol) = ex === from ? to : ex
function _respell_value(ex::Expr, from::Symbol, to::Symbol)
    if ex.head === :call && !isempty(ex.args)
        return Expr(:call, ex.args[1],
            (_respell_value(a, from, to) for a in ex.args[2:end])...)
    elseif ex.head === :kw && length(ex.args) == 2
        return Expr(:kw, ex.args[1], _respell_value(ex.args[2], from, to))
    elseif ex.head === :. && length(ex.args) == 2 && !(Meta.isexpr(ex.args[2], :tuple))
        return Expr(:., _respell_value(ex.args[1], from, to), ex.args[2])
    elseif ex.head === :. && length(ex.args) == 2
        return Expr(:., ex.args[1], _respell_value(ex.args[2], from, to))
    end
    return Expr(ex.head, (_respell_value(a, from, to) for a in ex.args)...)
end

function _desugar_plate(st::Expr, line::Int, data::Set{Symbol};
        broadcasts::Dict{Symbol,Vector{Symbol}} = Dict{Symbol,Vector{Symbol}}())
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
    isempty(cells) && return Expr[], Tuple{Symbol,Int,Set{Symbol}}[],
        Tuple{Symbol,Any,Union{Nothing,UnitRange{Int},Symbol,Expr},Int}[]
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
    params = Tuple{Symbol,Any,Union{Nothing,UnitRange{Int},Symbol,Expr},Int}[]
    selected = rkind[1] !== :levels && any(c -> c isa Expr &&
        (_is_sample(c) || _is_broadcast_sample(c)) &&
        Meta.isexpr(c.args[2], :ref) && c.args[2].args[1] in data, cells)
    selected &&
        return _desugar_selected_plate(cells, ivar, rkind, line, data, plate_defs;
            broadcasts)
    (rkind[1] === :levels || _plate_has_array_cells(cells)) &&
        return _desugar_array_plate(cells, ivar, rkind, line, data,
            plate_defs)
    _plate_needs_retained_cells(cells, ivar) &&
        return _desugar_selected_plate(cells, ivar, rkind, line, data, plate_defs;
            broadcasts)
    # Cell locals bound from a per-index value (they vary with the loop
    # variable, so arithmetic over them vectorizes in the desugar).
    pidx = Set{Symbol}()
    if any(c -> _cell_uses_index_value(c, ivar), cells)
        indexname = gensym(:_rkppl_index)
        indices = rkind[1] === :coloncall ? rkind[2] :
            rkind[1] === :eachindex ? Expr(:call, GlobalRef(Base, :eachindex), rkind[2]) :
            Expr(:call, GlobalRef(Base, :axes), rkind[2], rkind[3])
        indexvalues = Expr(:call, GlobalRef(Base, :collect), indices)
        push!(out, Expr(:(=), indexname,
            _plate_column_call(ivar, Pair{Symbol,Any}[], ivar, nothing; indices=indexvalues)))
        push!(plate_defs, indexname)
        push!(pidx, indexname)
        cells = Any[_cell_index_values(c, ivar, indexname) for c in cells]
    end
    for c in cells
        push!(out, _desugar_cell(c, ivar, rkind, line, data, plate_defs, ctx,
            params, pidx)...)
    end
    return out, ctx, params
end

# Partial observation loops compute their arguments inside retained RK cells.
# Slicing a fully computed vector would evaluate arithmetic in unselected cells.
function _desugar_selected_plate(cells, ivar, rkind, line, data, plate_defs;
        broadcasts::Dict{Symbol,Vector{Symbol}} = Dict{Symbol,Vector{Symbol}}())
    iterator = rkind[1] === :coloncall ? rkind[2] :
        rkind[1] === :eachindex ? Expr(:call, GlobalRef(Base, :eachindex), rkind[2]) :
        Expr(:call, GlobalRef(Base, :axes), rkind[2], rkind[3])
    indices = Expr(:call, GlobalRef(Base, :collect), iterator)
    locals = Pair{Symbol,Any}[]
    out = Expr[]
    ctx = Tuple{Symbol,Int,Set{Symbol}}[]
    params = Tuple{Symbol,Any,Union{Nothing,UnitRange{Int},Symbol,Expr},Int}[]
    pidx = Set{Symbol}()
    indexedlocals = Set{Symbol}()
    function localread(ex)
        Meta.isexpr(ex, :ref, 2) && ex.args[1] in indexedlocals &&
            ex.args[2] === ivar && return ex.args[1]
        ex isa Expr ? Expr(ex.head, map(localread, ex.args)...) : ex
    end
    # Gathers from data and declared arrays index at `i` or through an
    # integer data column. Indexing into a per-index value or a cell local
    # (`t[i][k[i]]`, `p[sel]`) is ordinary Julia inside the retained cell.
    function checkrefs(ex)
        ex isa Expr || return
        base = ex.head === :ref ? ex.args[1] : nothing
        if ex.head === :ref && _expr_has_sym(ex, ivar) && !(base isa Expr) &&
                !any(l -> first(l) === base, locals)
            for index in ex.args[2:end]
                index === ivar && continue
                if Meta.isexpr(index, :ref, 2) && index.args[2] === ivar
                    index.args[1] in data || _sfail("cell gather indexes through `$(index.args[1])`, which is not bound data")
                elseif _expr_has_sym(index, ivar)
                    _cell_lhs_error(ex, ivar)
                end
            end
        end
        foreach(checkrefs, ex.args)
    end
    for c in cells
        checkrefs(c)
        if Meta.isexpr(c, :(=), 2)
            lhs, rhs = c.args
            col = lhs isa Symbol ? lhs :
                Meta.isexpr(lhs, :ref, 2) && lhs.args[2] === ivar ? lhs.args[1] : nothing
            col isa Symbol || _sfail("selected plate assignments bind a local or an indexed scalar column")
            col in data && _sfail("cell assignment `$col = ...` redefines bound data")
            rhs = localread(rhs)
            if lhs isa Expr
                push!(out, Expr(:(=), col, _plate_column_call(ivar, locals, rhs, nothing; indices)))
                push!(indexedlocals, col)
            end
            push!(locals, col => rhs)
            continue
        end
        (_is_sample(c) || _is_broadcast_sample(c)) ||
            _sfail("cells hold `~` observations and `=` assignments only")
        lnames = Set(first.(locals))
        function arg(ex; object=false)
            Meta.isexpr(ex, :ref, 2) && ex.args[2] === ivar &&
                ex.args[1] isa Symbol && ex.args[1] ∉ lnames && return ex
            if ex isa Expr && (ex.head === :call || _is_dotted_call(ex)) &&
                    !isempty(ex.args) && (_cell_object_call(ex.args[1]) || object)
                vals = ex.head === :call ? ex.args[2:end] : ex.args[2].args
                return Expr(:call, ex.args[1], map(arg, vals)...)
            end
            if _expr_has_sym(ex, ivar) || any(n -> _expr_has_sym(ex, n), lnames)
                nm = gensym(:_rkppl_selected)
                push!(out, Expr(:(=), nm,
                    _plate_column_call(ivar, locals, ex, nothing; indices)))
                push!(plate_defs, nm)
                push!(pidx, nm)
                return Expr(:ref, nm, ivar)
            end
            return ex
        end
        obj = arg(localread(c.args[3]); object=true)
        # A dotted cell broadcasts over its own iteration's values, as in
        # Julia. Record the operands it reads per index: when the bound
        # response holds one array per index, each of these supplies that
        # index's value and every other operand is shared by all indices.
        lhs = c.args[2]
        if _is_broadcast_sample(c) && Meta.isexpr(lhs, :ref, 2) &&
                lhs.args[1] isa Symbol && lhs.args[1] in data && lhs.args[2] === ivar
            broadcasts[lhs.args[1]] = sort!(collect(_cell_index_reads(obj, ivar)))
        end
        append!(out, _desugar_cell_sample(Expr(:call, c.args[1], c.args[2], obj),
            ivar, rkind, line, data, plate_defs, ctx, params, pidx))
    end
    return out, ctx, params
end

# Whole-column evaluation (`_desugar_cell`) strips `[i]` and broadcasts each
# cell over whole columns. That equals the Julia loop only while every
# per-index value is a scalar. A definition cell that may hold an array per
# index is evaluated as retained RK plate cells instead, as plates with
# observations are: an authored broadcast, indexing or a reduction applied
# to a per-index value, or a module function or kernel called on one.
function _plate_needs_retained_cells(cells, ivar::Symbol)
    perlocals = Set{Symbol}()
    for c in cells
        Meta.isexpr(c, :(=), 2) || continue
        lhs, rhs = c.args
        _cell_reads_index(rhs, ivar, perlocals) || continue
        lhs isa Symbol && push!(perlocals, lhs)
        _retained_cell_expr(rhs, ivar, perlocals) && return true
    end
    return false
end

# A per-index value: a read at the loop index (`x[i]`, `v[g[i]]`, `t[i][k]`)
# or a cell local computed from one.
_cell_reads_index(ex, ivar, perlocals) = ex isa Symbol && ex in perlocals
function _cell_reads_index(ex::Expr, ivar, perlocals)
    ex.head === :ref && _expr_has_sym(ex, ivar) && return true
    start = ex.head === :call || (ex.head === :. && length(ex.args) == 2) ? 2 : 1
    ex.head === :kw && (start = 2)
    return any(a -> _cell_reads_index(a, ivar, perlocals), ex.args[start:end])
end

_retained_cell_expr(ex, ivar, perlocals) = false
function _retained_cell_expr(ex::Expr, ivar, perlocals)
    per(a) = _cell_reads_index(a, ivar, perlocals)
    if ex.head === :ref
        base = ex.args[1]
        (base isa Symbol ? base in perlocals : per(base)) && return true
    elseif ex.head === :. && length(ex.args) == 2 && ex.args[2] isa Expr &&
            ex.args[2].head === :tuple
        any(per, ex.args[2].args) && return true
    elseif ex.head === :call && !isempty(ex.args)
        f = ex.args[1]
        args = ex.args[2:end]
        if any(per, args)
            f isa Symbol && startswith(string(f), ".") && return true
            f isa Symbol && f in REDUCTION_FNS && return true
            f isa Symbol && (_builtin_value_head(f) || Base.isoperator(f) ||
                _cell_object_call(f)) || return true
        end
    end
    return any(a -> _retained_cell_expr(a, ivar, perlocals), ex.args)
end

# Bases of the `v[i]` reads in a cell expression, i.e. its per-index operands.
function _cell_index_reads(ex, ivar, out::Set{Symbol} = Set{Symbol}())
    ex isa Expr || return out
    if Meta.isexpr(ex, :ref, 2) && ex.args[1] isa Symbol && ex.args[2] === ivar
        push!(out, ex.args[1])
        return out
    end
    foreach(a -> _cell_index_reads(a, ivar, out), ex.args)
    return out
end

# Numeric index values become one data-only index column. Indexed reads
# retain their original syntax for the cell's index validation.
_cell_uses_index_value(ex, ivar) = ex === ivar
function _cell_uses_index_value(ex::Expr, ivar)
    ex.head === :ref && return false
    start = ex.head in (:call, :kw, :.) ? 2 : 1
    return any(a -> _cell_uses_index_value(a, ivar), ex.args[start:end])
end
_cell_index_values(ex, ivar, name) = ex === ivar ? name : ex
function _cell_index_values(ex::Expr, ivar, name)
    ex.head === :ref && return ex
    start = ex.head in (:call, :kw, :.) ? 2 : 1
    return Expr(ex.head, ex.args[1:start-1]...,
        (_cell_index_values(a, ivar, name) for a in ex.args[start:end])...)
end

# An authored row output states that the cell holds an array even when
# its value comes from an opaque Julia callable. Whole-axis and crossed
# reads also require the retained RK cell rather than scalar broadcasting.
_plate_has_array_cells(cells) = any(c -> c isa Expr && c.head === :(=) &&
    length(c.args) == 2 &&
    ((Meta.isexpr(c.args[1], :ref) && length(c.args[1].args) > 2) ||
     _has_colon_index(c.args[2]) || _has_crossed_cell_index(c.args[2])), cells)
_has_colon_index(ex) = ex isa Expr && ((ex.head === :ref &&
    any(a -> a === :(:), ex.args[2:end])) || any(_has_colon_index, ex.args))
_has_crossed_cell_index(ex) = ex isa Expr &&
    ((ex.head === :ref && count(a -> a isa Expr && a.head === :ref,
        ex.args[2:end]) > 1) || any(_has_crossed_cell_index, ex.args))

# A per-index plate with array-valued cells. Every cell means one
# iteration of the Julia loop: cell locals (`F = sd[s[i], :] .* L[s[i]]`)
# may be arrays, and each per-index column a cell defines (`r1[i] = …`,
# a scalar per index) becomes the derived column `r1` computed by one RK
# plate — the cell body, run once per index by the loop RK generates
# (the plate expression is built once array axes are known,
# `_plate_column_expr`). Observations in the same plate read those
# columns (`y[i] ~ Normal(r1[i], s)`) and lower as in a scalar plate.
function _desugar_array_plate(cells, ivar::Symbol, rkind, line, data,
        plate_defs::Set{Symbol})
    axis = rkind[1] === :levels ? rkind[2] : nothing
    indices = if axis === nothing
        iterator = rkind[1] === :coloncall ? rkind[2] :
            rkind[1] === :eachindex ? Expr(:call, GlobalRef(Base, :eachindex), rkind[2]) :
            Expr(:call, GlobalRef(Base, :axes), rkind[2], rkind[3])
        Expr(:call, GlobalRef(Base, :collect), iterator)
    else
        nothing
    end
    locals = Pair{Symbol,Any}[]
    out = Expr[]
    ctx = Tuple{Symbol,Int,Set{Symbol}}[]
    params = Tuple{Symbol,Any,Union{Nothing,UnitRange{Int},Symbol,Expr},Int}[]
    samples = Any[]
    for c in cells
        c isa Expr || _sfail("cells hold `~` observations and `=` " *
            "assignments only")
        if _is_sample(c)
            push!(samples, c)
            continue
        end
        (c.head === :(=) && length(c.args) == 2) || _sfail("cells hold `~` " *
            "observations and `=` assignments only")
        lc, rhs = c.args
        if lc isa Symbol
            lc in data && _sfail("cell assignment `$lc = ...` redefines " *
                "bound data")
            push!(locals, lc => rhs)
        elseif lc isa Expr && lc.head === :ref && length(lc.args) == 2 &&
                lc.args[1] isa Symbol && lc.args[2] === ivar
            col = lc.args[1]
            col in data && _sfail("cell assignment `$col[$ivar] = ...` " *
                "redefines bound data")
            push!(out, Expr(:(=), col,
                _plate_column_call(ivar, locals, rhs, axis; indices)))
        elseif lc isa Expr && lc.head === :ref && length(lc.args) == 3 &&
                lc.args[1] isa Symbol && lc.args[2] === ivar
            # A row per index (`b[i, 1:K] = row`): one plate column per
            # component, read back as the columns of `b` (`b[:, k]`).
            col = lc.args[1]
            col in data && _sfail("cell assignment `$(repr(lc)) = ...` " *
                "redefines bound data")
            K = _literal_range_len(lc.args[3])
            K === nothing && _sfail("cell `$(repr(c))`: a row per index " *
                "states its length, `$col[$ivar, 1:K] = ...` with a literal K")
            rowv = Symbol(:_rkppl_rowv_, col)
            rowlocals = [locals; rowv => rhs]
            parts = Symbol[]
            for k in 1:K
                pk = Symbol(:_rkppl_row_, col, :_, k)
                push!(parts, pk)
                push!(out, Expr(:(=), pk,
                    _plate_column_call(ivar, rowlocals, :($rowv[$k]), axis; indices)))
            end
            push!(out, Expr(:(=), col, Expr(:call, :_ppl_rows, parts...)))
        elseif lc isa Expr && lc.head === :ref
            _sfail("cell `$(repr(c))`: a per-index output is a value " *
                "(`$(lc.args[1])[$ivar] = ...`) or a row " *
                "(`$(lc.args[1])[$ivar, 1:K] = ...`)")
        else
            _sfail("cell assignment LHS is a local (`t = ...`) or an " *
                "`$ivar`-indexed column, got $(repr(lc))")
        end
    end
    lnames = Set{Symbol}(first.(locals))
    pidx = Set{Symbol}()
    for c in samples
        if axis !== nothing
            obj = c.args[3]
            if _expr_has_sym(obj, ivar) || any(nm -> _expr_has_sym(obj, nm), lnames)
                # Each varying scalar argument is itself one RK plate
                # column; the ordinary array prior pairs its elements
                # with the draw. Shared arguments retain their spelling.
                (obj isa Expr && obj.head === :call &&
                    obj.args[1] in _HOIST_FAMILIES) || _sfail("per-level " *
                    "varying arguments currently take an elementwise prior")
                args = Any[]
                for (i, arg) in enumerate(obj.args[2:end])
                    if _expr_has_sym(arg, ivar) || any(nm -> _expr_has_sym(arg, nm), lnames)
                        nm = Symbol(:_rkppl_level_, c.args[2].args[1], :_arg_, i)
                        push!(out, Expr(:(=), nm,
                            _plate_column_call(ivar, locals, arg, axis)))
                        push!(args, nm)
                    else
                        push!(args, arg)
                    end
                end
                c = Expr(:call, :~, c.args[2], Expr(:call, obj.args[1], args...))
            end
            append!(out, _desugar_levels_cell(c, ivar, axis, data))
            continue
        end
        any(nm -> _expr_has_sym(c.args[3], nm), lnames) && _sfail("cell " *
            "`$(repr(c))` reads a cell local of an array plate; define a " *
            "per-index column (`eta[$ivar] = ...`) and observe it")
        append!(out, _desugar_cell_sample(c, ivar, rkind, line, data,
            plate_defs, ctx, params, pidx))
    end
    return out, ctx, params
end

# Shared cell construction for observation and level axes. The fourth
# field records a level axis, so constant outputs still have one value
# per level and subsequent gathers preserve that axis.
function _plate_column_call(ivar, locals, rhs, axis; indices=nothing)
    body = Expr(:block, (Expr(:(=), k, v) for (k, v) in locals)...)
    deps = Set{Symbol}()
    for (_, v) in locals
        _cell_free_syms!(deps, v)
    end
    _cell_free_syms!(deps, rhs)
    setdiff!(deps, Set{Symbol}(first.(locals)))
    delete!(deps, ivar)
    spec = Expr(:tuple, QuoteNode(ivar), QuoteNode(body), QuoteNode(rhs))
    if axis !== nothing || indices !== nothing
        push!(spec.args, QuoteNode(axis))
        axis !== nothing && push!(deps, axis)
    end
    if indices !== nothing
        push!(spec.args, QuoteNode(indices))
        _cell_free_syms!(deps, indices)
    end
    return Expr(:call, :_ppl_plate_column, spec, sort!(collect(deps))...)
end

# Value names an expression reads (call heads, `:` and keywords skipped).
function _cell_free_syms!(out::Set{Symbol}, ex)
    if ex isa Symbol
        ex === :(:) || ex === :end || push!(out, ex)
    elseif ex isa Expr
        if ex.head === :call && !isempty(ex.args)
            foreach(a -> _cell_free_syms!(out, a), ex.args[2:end])
        elseif ex.head === :. && length(ex.args) == 2
            ex.args[2] isa Expr && foreach(a -> _cell_free_syms!(out, a),
                ex.args[2].args)
        elseif ex.head === :kw
            _cell_free_syms!(out, ex.args[2])
        else
            foreach(a -> _cell_free_syms!(out, a), ex.args)
        end
    end
    return out
end

# Plate ranges mirror the response ranges: literal `a:b` (validated via
# the desugared `y[a:b]` form), `eachindex(v)`, `axes(v, 1)`.
function _plate_range_kind(R)
    R isa Expr && R.head === :call && !isempty(R.args) || return _sfail(
        "`@plate` range must be a literal `a:b`, `eachindex(v)`, `axes(v, 1)`, or " *
        "`levels(g)` — got $(repr(R)) (values-iteration is planned)")
    if R.args[1] === :(:)
        # Folded literal endpoints give the response range and the selected
        # cells one spelling; an unbound endpoint fails at the response.
        lo, hi = length(R.args) == 3 ? _static_count.(R.args[2:3]) : (nothing, nothing)
        return (:coloncall, lo isa Integer && hi isa Integer ?
            Expr(:call, :(:), lo, hi) : R)
    end
    R.args[1] === :eachindex && length(R.args) == 2 &&
        R.args[2] isa Symbol && return (:eachindex, R.args[2])
    R.args[1] === :axes && length(R.args) == 3 && R.args[2] isa Symbol &&
        R.args[3] isa Integer && !(R.args[3] isa Bool) && R.args[3] >= 1 &&
        return (:axes, R.args[2], Int(R.args[3]))
    R.args[1] === :levels && length(R.args) == 2 && R.args[2] isa Symbol &&
        return (:levels, R.args[2])
    return _sfail("`@plate` range must be a literal `a:b`, `eachindex(v)`, " *
                  "`axes(v, 1)`, or `levels(g)` — got $(repr(R))")
end

# One cell statement → spliced top-level statement(s). A cell means what
# the same statement means in one iteration of a Julia `for` loop: every
# value in it is a scalar (`x[i]`, a gather `v[g[i]]`, a model scalar, a
# literal). The desugar is the lowering that evaluates all iterations at
# once: per-index refs strip to whole columns and every call or operator
# over a per-index operand takes its broadcast form (`a + b * x[i]` →
# `a .+ b .* x`) — Julia's own equivalence between applying a scalar
# function per element and broadcasting it. Already-dotted spellings are
# kept as written (broadcasting over scalars returns the scalar). So
# observations (`y[i] ~ OBJ`) become `y .~ OBJ.` (or `y[a:b] .~ OBJ.`
# under a literal range), and deterministic cells strip to top level
# (visible model-wide — documented looseness: desugared locals leak like
# any top-level det).
function _desugar_cell(c, ivar, rkind, line, data, plate_defs, ctx, params,
        pidx::Set{Symbol})
    c isa Expr || _sfail("cells hold `~` observations and `=` " *
                         "assignments only")
    if c.head === :macrocall && !isempty(c.args) &&
            c.args[1] === Symbol("@plate")
        _sfail("nested `@plate` blocks are not a StanBlocks form — use " *
               "factor/levels for crossed effects (`c[levels(g)]`), `@scan` " *
               "for sequential recurrence")
    end
    rkind[1] === :levels && return _desugar_levels_cell(c, ivar, rkind[2],
        data)
    if _is_sample(c) || _is_broadcast_sample(c)
        return _desugar_cell_sample(c, ivar, rkind, line, data, plate_defs,
            ctx, params, pidx)
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
        bares = _cell_bares(c.args[2], ivar, data)
        setdiff!(bares, plate_defs)
        push!(ctx, (col, line, bares))
        rhs, per = _dotify_cell(c.args[2], ivar, pidx, false)
        per && push!(pidx, col)
        return [Expr(:(=), col, rhs)]
    end
    return _sfail("cells hold `~` observations and `=` assignments only")
end

# A `@plate for k in levels(g)` cell: one iteration per level of `g`. A
# per-level declaration means what the whole-array declaration means:
# `c[k] ~ D` is `c[levels(g)] .~ D.` (one draw per level), and a row
# `b[k, :] ~ D` (or `b[k, 1:K]`) with a multivariate `D` is the row
# statement `eachrow(b[levels(g), 1:K]) .~ D` (§ array slices), and
# `L[k] ~ LKJCholesky(K, eta)` declares one correlation factor per level
# (a stacked LKJ array, `:lkj_cholesky_stack`). Varying arguments and
# definitions are emitted as RK plate columns by `_desugar_array_plate`.
function _desugar_levels_cell(c, ivar::Symbol, g::Symbol, data)
    where = "`@plate for $ivar in levels($g)` cell `$(repr(c))`"
    _is_sample(c) || _sfail("$where: expected a per-level prior")
    lhs, obj = c.args[2], c.args[3]
    (lhs isa Expr && lhs.head === :ref && length(lhs.args) in (2, 3) &&
        lhs.args[1] isa Symbol && lhs.args[2] === ivar) || _sfail("$where: " *
        "a per-level cell declares `name[$ivar]` or a row `name[$ivar, :]`")
    col = lhs.args[1]
    col in data && _sfail("$where: $col is data; per-level observations " *
        "are not supported yet")
    _expr_has_sym(obj, ivar) && _sfail("$where: per-level distribution " *
        "arguments are not supported yet; the arguments are shared across " *
        "levels")
    if length(lhs.args) == 2
        _mv_head(obj) === nothing || _sfail("$where: a multivariate " *
            "per-level draw is a row, `$col[$ivar, :] ~ $(repr(obj))`")
        # One correlation factor per level: an internal declaration the
        # statement partitioner reads as a stacked LKJ array (users write
        # the plate; `_ppl_` names are reserved).
        _is_lkj_cholesky_call(obj) && return Expr[Expr(:call, :~,
            Expr(:call, :_ppl_per_level, col, g), obj)]
        dobj, _ = _dotify_cell(obj, ivar, Set{Symbol}(), true)
        return Expr[Expr(:call, :.~, Expr(:ref, col, :(levels($g))), dobj)]
    end
    _mv_head(obj) === nothing && _sfail("$where: a row `$col[$ivar, :]` " *
        "is one draw of a multivariate distribution (`MvNormal`, " *
        "`MvNormalCholesky`, `Dirichlet`, `Ordered`)")
    ax = lhs.args[3]
    if ax === :(:)
        K = _mv_slice_len(obj)
        K === nothing && _sfail("$where: the row length is not readable " *
            "from the distribution; write it, `$col[$ivar, 1:K]`")
        ax = :(1:$K)
    end
    return Expr[Expr(:call, :.~, Expr(:call, :eachrow,
        Expr(:ref, col, :(levels($g)), ax)), obj)]
end

# The slice length of a multivariate distribution call whose arguments
# state it literally: `MvNormal(zeros(K), …)` / `MvNormalCholesky(ones(K),
# …)` / a literal mean vector, `Dirichlet(K, a)` / a literal concentration
# vector, `Ordered(D, K)`. `nothing` when it is not literal.
function _mv_slice_len(obj)
    h = _mv_head(obj)
    (h === nothing || obj.head !== :call || length(obj.args) < 2) &&
        return nothing
    a = obj.args[2]
    lit(x) = x isa Int && x >= 1 ? x : nothing
    if h === :Ordered
        return length(obj.args) == 3 ? lit(obj.args[3]) : nothing
    elseif h === :Dirichlet && length(obj.args) == 3
        return lit(a)
    end
    a isa Expr && a.head === :vect && return length(a.args)
    a isa Expr && a.head === :call && length(a.args) == 2 &&
        a.args[1] in (:zeros, :ones) && return lit(a.args[2])
    return nothing
end

# The RK plate an array-cell plate column runs: `plate(lanes...,
# Ref(shared)...) do lane..., shared...; cell; end`, one cell per index.
# Lanes are the per-index inputs: a data column read at the loop index
# (`x[i]`), or the level codes of a column that indexes an array's levels
# axis (`sd[s[i], :]`, `z[g[i], :]`, a per-level factor `L[s[i]]` —
# `_ppl_codes(s, h)`, the codes of `s` on `levels(h)`). Level cells align
# selected or different declared axes through `_ppl_level_gather` inputs.
# Every other name
# the cell reads (arrays, parameters, definitions) is passed whole with
# `Ref`. Inside the cell those reads are ordinary integer indexing
# (`sd[c, :]`, `L[:, :, c]`); RK generates the loop.
function _plate_column_expr(nm::Symbol, call::Expr,
        dims::Dict{Symbol,Vector{Any}}, data::Set{Symbol})
    spec = call.args[2]
    ivar = spec.args[1].value
    body = spec.args[2].value
    out = spec.args[3].value
    axis = length(spec.args) >= 4 ? spec.args[4].value : nothing
    indices = length(spec.args) >= 5 ? spec.args[5].value : nothing
    where = "array plate column `$nm`"
    inputs = Any[]
    lanevars = Symbol[]
    sharedsources = Dict{Symbol,Any}()
    lane(inp, var) = (var in lanevars || (push!(inputs, inp);
        push!(lanevars, var)); var)
    pos = indices !== nothing ? lane(indices, Symbol(:_ppl_pi_, nm)) :
        axis === nothing ? nothing : lane(
            Expr(:call, :_ppl_level_indices, axis), Symbol(:_ppl_pi_, axis))
    level() = lane(Expr(:call, :_ppl_level_values, axis),
        Symbol(:_ppl_pl_, axis))
    iscodeidx(a) = a isa Expr && a.head === :ref && length(a.args) == 2 &&
        a.args[1] isa Symbol && a.args[1] in data && a.args[2] === ivar
    isidx(a) = indices === nothing && axis === nothing && iscodeidx(a)
    function code(a, d, X, j)
        _is_levels_dim(d) && d.args[2] isa Symbol || _sfail("$where reads " *
            "an array axis $(repr(d)) by `$(repr(a))`; per-index reads " *
            "take a `levels(g)` axis")
        col, h = a.args[1], d.args[2]
        if d.args[1] !== :levels || _levels_subset(d) !== Colon()
            codes = Expr(:call, :_ppl_axis_codes, col, X, j)
            indices === nothing || (codes = Expr(:ref, codes, indices))
            return lane(codes,
                Symbol(:_ppl_pc_, col, :_, X, :_, j))
        end
        codes = Expr(:call, :_ppl_codes, col, h)
        indices === nothing || (codes = Expr(:ref, codes, indices))
        return lane(codes,
            Symbol(:_ppl_pc_, col, :_, h))
    end
    function rw(ex)
        indices !== nothing && ex === ivar && return pos
        axis !== nothing && ex === ivar && return level()
        ex isa Expr || return ex
        if ex.head === :ref && ex.args[1] isa Symbol
            X, idx = ex.args[1], ex.args[2:end]
            if indices !== nothing && X in data && length(idx) == 1
                linear = Symbol(:_ppl_linear_, X)
                sharedsources[linear] = Expr(:call, GlobalRef(Base, :vec), X)
                return Expr(:ref, linear, rw(only(idx)))
            end
            d = get(dims, X, nothing)
            if axis !== nothing && d !== nothing && idx[1] === ivar
                dim = length(d) == 3 && length(idx) == 1 ? d[3] : d[1]
                _is_levels_dim(dim) || return Expr(:ref, X, map(rw, idx)...)
                h = dim.args[2]
                if dim.args[1] !== :levels || _levels_subset(dim) !== Colon() || h !== axis
                    # Align values by their declared labels before entering
                    # the cell. A parameter can be a view; indexing that view
                    # at a traced lane code is not supported by Reactant.
                    ld = length(d) == 3 && length(idx) == 1 ? 3 : 1
                    value = lane(Expr(:call, :_ppl_level_gather, X, axis, ld),
                        Symbol(:_ppl_pv_, X))
                    length(d) == 1 && return value
                    ld == 3 && return Expr(:call, :reshape, value, d[1], d[2])
                    length(idx) == length(d) || _sfail("$where reads $X " *
                        "on a selected or different level axis; give one " *
                        "index per axis")
                    return Expr(:ref, value, map(rw, idx[2:end])...)
                end
                # The same scalar level axis is already aligned with
                # the plate lanes. Passing it directly also avoids an
                # unnecessary dynamic gather from a parameter view.
                h === axis && length(d) == 1 && length(idx) == 1 &&
                    return lane(X, Symbol(:_ppl_pv_, X))
                codevar = pos
                length(d) == 3 && length(idx) == 1 &&
                    return Expr(:ref, X, :(:), :(:), codevar)
                return Expr(:ref, X, codevar, map(rw, idx[2:end])...)
            end
            isidx(ex) && return lane(X, Symbol(:_ppl_pv_, X))
            if d !== nothing && !isempty(idx) && any(iscodeidx, idx)
                if length(d) == 3 && length(idx) == 1
                    return Expr(:ref, X, :(:), :(:), code(idx[1], d[3], X, 3))
                end
                base, gatherindices = X, Any[iscodeidx(a) ? a : rw(a) for a in idx]
                for j in eachindex(idx)
                    iscodeidx(idx[j]) || continue
                    gatherindices[j] = code(idx[j], d[j], X, j)
                    if _levels_subset(d[j]) !== Colon()
                        zero = length(d) == 1 ? 0.0 :
                            j == 1 ? :(zeros(1, size($X, 2))) : :(zeros(size($X, 1), 1))
                        base = Expr(:call, j == 1 ? :vcat : :hcat, zero, base)
                        gatherindices[j] = Expr(:call, :+, gatherindices[j], 1)
                    end
                end
                return Expr(:ref, base, gatherindices...)
            end
        end
        return Expr(ex.head, map(rw, ex.args)...)
    end
    stmts = Any[rw(a) for a in body.args]
    outx = rw(out)
    isempty(lanevars) && _sfail("$where reads no per-index input " *
        "(`x[$ivar]`, `z[g[$ivar], :]`): it does not vary with $ivar")
    locals = Set{Symbol}(a.args[1] for a in stmts
        if a isa Expr && a.head === :(=) && a.args[1] isa Symbol)
    free = Set{Symbol}()
    foreach(a -> _cell_free_syms!(free, a), stmts)
    _cell_free_syms!(free, outx)
    setdiff!(free, locals)
    setdiff!(free, Set(lanevars))
    ivar in free && _sfail("$where reads the loop index $ivar other than " *
        "through a data column (`x[$ivar]`) or a levels gather " *
        "(`z[g[$ivar], :]`)")
    shared = sort!(collect(free))
    append!(inputs, (Expr(:call, :Ref, get(sharedsources, nm2, nm2)) for nm2 in shared))
    lam = Expr(:->, Expr(:tuple, lanevars..., shared...),
        Expr(:block, LineNumberNode(0, :rkppl_plate), stmts..., outx))
    return Expr(:do, Expr(:call, :plate, inputs...), lam)
end

_literal_range_len(r) = (r isa Expr && r.head === :call && length(r.args) == 3 &&
    r.args[1] === :(:) && _static_count(r.args[2]) == 1 &&
    _static_count(r.args[3]) !== nothing && _static_count(r.args[3]) >= 1) ?
    _static_count(r.args[3]) : nothing

# Rows per index (`b[i, 1:K] = row` in an array plate): `b` stands for the
# matrix whose column k is the plate column `_rkppl_row_b_k`. Materialize
# that matrix as an ordinary Julia value, so aliases, indexing and whole
# consumers share one shape and the normal value-lowering machinery.
function _rewrite_plate_rows(sample, det)
    newdet = Pair{Symbol,Any}[nm =>
        (rhs isa Expr && rhs.head === :call && !isempty(rhs.args) &&
         rhs.args[1] === :_ppl_rows ?
            Expr(:call, GlobalRef(Base, :hcat), rhs.args[2:end]...) : rhs)
        for (nm, rhs) in det]
    return sample, newdet
end

_is_plate_column_call(ex) = ex isa Expr && ex.head === :call &&
    !isempty(ex.args) && ex.args[1] === :_ppl_plate_column

_plate_column_axis(ex) = _is_plate_column_call(ex) &&
    length(ex.args[2].args) == 4 ? ex.args[2].args[4].value : nothing

_expr_has_sym(ex, s::Symbol) = ex === s ||
    (ex isa Expr && any(a -> _expr_has_sym(a, s), ex.args))

# A cell `~` statement is EITHER a per-cell latent parameter declaration
# (`theta[i] ~ Normal(mu, tau)` — a non-data indexed LHS, scalar undotted
# distribution, shared-scalar args) or an observation on a sliced data column
# (`y[i] ~ Normal(mu[i], s)` — one scalar draw per index, as in a Julia loop;
# the broadcast spelling `Normal.(mu[i], s)` means the same and is kept).
# Returns the spliced top-level statement(s); a per-cell parameter records its
# spec in `params` and emits no top-level statement.
function _desugar_cell_sample(c, ivar, rkind, line, data, plate_defs, ctx,
        params, pidx::Set{Symbol})
    lhs = c.args[2]
    lhs isa Symbol && _sfail("bare per-cell sample `$lhs ~ ...` does not " *
                             "lower — index the latent (`$lhs[$ivar] ~ ...`) " *
                             "for a per-cell parameter, or write a shared " *
                             "prior outside the plate")
    (lhs isa Expr && lhs.head === :ref && length(lhs.args) >= 2 &&
        lhs.args[1] isa Symbol && lhs.args[2] === ivar &&
        all(a -> a isa Integer && !(a isa Bool) && a >= 1, lhs.args[3:end])) ||
        _cell_lhs_error(lhs, ivar)
    col = lhs.args[1]
    obj = c.args[3]
    if col ∉ data
        length(lhs.args) == 2 || _cell_lhs_error(lhs, ivar)
        # Per-cell latent PARAMETER: a scalar (undotted) distribution. Its args
        # are shared across cells (a captured scalar) or per-cell (`eta[$ivar]`,
        # a varying prior mean/scale). Preserve the authored selection: a prior
        # may read a prefix of a longer column, and an indexed singleton is
        # not a shared scalar that can broadcast to every cell.
        bares = _cell_bares(obj, ivar, data)
        setdiff!(bares, plate_defs)
        push!(ctx, (col, line, bares))
        # A data or literal iterator is evaluated as written at binding. A
        # value's iterator (`eachindex(mu)` for a definition, a sampled
        # array or a stream's location) is its extent `1:n`, bound with the
        # latent's own cell count (`_plate_extent_input`).
        indices = rkind[1] === :coloncall ? rkind[2] :
            !(rkind[2] in data) ?
                Expr(:call, :(:), 1, _plate_extent_input(col)) :
            rkind[1] === :eachindex ? Expr(:call, :eachindex, rkind[2]) :
            Expr(:call, :axes, rkind[2], rkind[3])
        indices = Expr(:call, GlobalRef(Base, :collect), indices)
        push!(params, (col, _strip_cell(_undot_cell_object(obj), ivar, indices),
            _plate_param_range(col, rkind, data), line))
        return Expr[]
    end
    bares = _cell_bares(obj, ivar, data)
    setdiff!(bares, plate_defs)
    push!(ctx, (col, line, bares))
    (obj isa Expr && (obj.head === :call || obj.head === :.) || _external_rhs(obj)) || _sfail(
        "cell objects are distribution calls " *
        "(`y[$ivar] ~ Normal(mu[$ivar], s)`), got $(repr(obj))")
    # One draw per index: the broadcast form of the object is the
    # vectorized observation (the distribution call and its wrappers take
    # their dotted form even over scalar-only arguments — `y[i] ~ Normal(0,
    # 1)` observes every row).
    obj, _ = _dotify_cell(obj, ivar, pidx, true)
    # An open RHS constructor at the root is applied once per authored
    # iteration, including qualified and lowercase caller bindings.
    if _external_rhs(obj) && Meta.isexpr(obj, :call)
        obj = Expr(:., obj.args[1], Expr(:tuple, obj.args[2:end]...))
    end
    if rkind[1] === :coloncall
        # A literal range keeps its authored indices (`indexed_observations`),
        # so the loop selects exactly its cells, as `eachindex` / `axes` do;
        # binding checks them against the response like Julia indexing.
        return Expr[Expr(:call, :.~, Expr(:ref, col, rkind[2], lhs.args[3:end]...), obj)]
    end
    index = rkind[1] === :eachindex ? Expr(:call, :eachindex, rkind[2]) :
        Expr(:call, :axes, rkind[2], rkind[3])
    return Expr[Expr(:call, :.~, Expr(:ref, col, index, lhs.args[3:end]...), obj)]
end

_undot_cell_object(ex) = ex
function _undot_cell_object(ex::Expr)
    if ex.head === :. && length(ex.args) == 2 && ex.args[1] isa Symbol &&
            _cell_object_call(ex.args[1]) && ex.args[2] isa Expr &&
            ex.args[2].head === :tuple
        return Expr(:call, ex.args[1], map(_undot_cell_object, ex.args[2].args)...)
    end
    return Expr(ex.head, map(_undot_cell_object, ex.args)...)
end

# The per-cell latent's size follows the plate range: a literal `1:N` rides as
# a UnitRange defining its own cells; `eachindex(v)` / `axes(v,
# 1)` over a data column ride as the column `v` (one cell per entry, proved
# at bind); over a definition ⇒ the definition's own observation axis.
function _plate_param_range(name::Symbol, rkind, data::Set{Symbol})
    if rkind[1] === :coloncall
        range = _lower_lhs_range(name, rkind[2])
        isempty(range) && return 1:0
        first(range) == 1 || _sfail("per-cell latent $name over " *
            "$(repr(rkind[2])) is not built yet: latent plate cells are `1:N`")
        return range
    end
    return rkind[1] === :eachindex ? Expr(:call, :eachindex, rkind[2]) :
        Expr(:call, :axes, rkind[2], rkind[3])
end

function _cell_lhs_error(lhs, ivar)
    lhs isa Expr && lhs.head === :ref || return _sfail(
        "cell responses sample `name[$ivar]` exactly — got $(repr(lhs))")
    length(lhs.args) == 2 || return _sfail(
        "one-dimensional cell refs only (`v[$ivar]`)")
    lhs.args[1] isa Symbol || return _sfail(
        "cell refs index a bare column (`v[$ivar]`)")
    return _sfail("cell reads index the loop variable (`v[$ivar]`) or " *
                  "gather through an index column (`v[g[$ivar]]`) — got " *
                  "$(repr(lhs)) (cross-index reads need `@scan`)")
end

# `v[g[i]]`: a gather through the data index column `g` (Julia indexing —
# row `i` reads entry `g[i]` of `v`). Returns `g`, or `nothing` when `ex`
# is not that shape.
function _cell_gather_index(ex::Expr, ivar)
    length(ex.args) == 2 && ex.args[1] isa Symbol || return nothing
    idx = ex.args[2]
    idx isa Expr && idx.head === :ref && length(idx.args) == 2 &&
        idx.args[1] isa Symbol && idx.args[2] === ivar || return nothing
    return idx.args[1]
end

# Value-position symbols of a cell expression, validating the loop-variable
# discipline on the way: refs are exactly `v[i]` or a gather `v[g[i]]`
# through a data index column `g`, and `i` appears only as an index.
# Function heads and kw names are positions, not refs.
function _cell_bares(ex, ivar, data::Set{Symbol})
    bares = Set{Symbol}()
    _cell_bares!(ex, ivar, bares, data)
    return bares
end

function _cell_bares!(ex::Symbol, ivar, bares, data)
    ex === ivar && _sfail("loop variable `$ivar` appears only as an " *
                          "index (`v[$ivar]`)")
    push!(bares, ex)
    return nothing
end
_cell_bares!(ex, ivar, bares, data) = nothing
function _cell_bares!(ex::Expr, ivar, bares, data)
    if ex.head === :ref
        length(ex.args) == 2 && ex.args[1] isa Symbol &&
            ex.args[2] === ivar && return nothing
        g = _cell_gather_index(ex, ivar)
        g === nothing && _cell_lhs_error(ex, ivar)
        g in data || _sfail("cell gather `$(repr(ex))` indexes through " *
                            "`$g`, which is not bound data (index columns " *
                            "are integer data columns)")
        return nothing
    end
    if ex.head === :call
        fn = ex.args[1]
        # Undotted module calls and reductions consume whole values.
        # Still walk their arguments to validate indexed reads and the
        # loop variable, while exempting bare whole arguments from the
        # per-observation column check.
        whole = _cell_whole_call(fn)
        reads = whole ? Set{Symbol}() : bares
        for a in ex.args[2:end]
            _cell_bares!(a, ivar, reads, data)
        end
        return nothing
    end
    if ex.head === :.
        start = length(ex.args) >= 1 && ex.args[1] isa Symbol ? 2 : 1
        for a in ex.args[start:end]
            _cell_bares!(a, ivar, bares, data)
        end
        return nothing
    end
    if ex.head === :kw
        for a in ex.args[2:end]
            _cell_bares!(a, ivar, bares, data)
        end
        return nothing
    end
    for a in ex.args
        _cell_bares!(a, ivar, bares, data)
    end
    return nothing
end

# Strip per-index refs to whole values (validation ran first): `v[i]` →
# `v`, a gather `v[g[i]]` → `v[g]`. A latent prior supplies its authored
# indices, retaining `v[indices]` or `v[g[indices]]` instead of reading
# every element of a potentially larger input.
_strip_cell(ex, ivar, indices=nothing) = ex
function _strip_cell(ex::Expr, ivar, indices=nothing)
    if ex.head === :ref
        g = _cell_gather_index(ex, ivar)
        selected = indices === nothing ? g :
            g === nothing ? indices : Expr(:ref, g, indices)
        return selected === nothing ? ex.args[1] : Expr(:ref, ex.args[1], selected)
    end
    return Expr(ex.head, (_strip_cell(a, ivar, indices) for a in ex.args)...)
end

# An indexed read in a per-cell latent prior (`x[i]`, `v[g[i]]`) reads a
# model value. Name an undeclared one as written, before its cell
# selection becomes a generated definition (strict declarations, `05oe96l`).
function _check_plate_prior_reads(plate_specs, names::Set{Symbol})
    for (nm, rhs, _, line) in plate_specs
        for v in sort!(collect(_ref_bases!(Set{Symbol}(), rhs)))
            v in names && continue
            at = line > 0 ? " (line $line)" : ""
            _sfail("`@plate`$at: the prior of `$nm` indexes `$v`, which " *
                   "is not data, a definition or a declared parameter — bind " *
                   "it as data or declare it")
        end
    end
    return nothing
end

_ref_bases!(out::Set{Symbol}, ex) = out
function _ref_bases!(out::Set{Symbol}, ex::Expr)
    ex.head === :ref && ex.args[1] isa Symbol && push!(out, ex.args[1])
    foreach(a -> _ref_bases!(out, a), ex.args)
    return out
end

# Wrappers and constructors an observation object broadcasts even over
# scalar-only arguments (one draw per index): distribution constructors
# (capitalized) and the response wrappers.
const _CELL_OBJECT_WRAPPERS = (:truncated, :restricted, :censored, :interval_censored,
    :weighted)
_cell_object_call(f::Symbol) =
    isuppercase(first(string(f))) || f in _CELL_OBJECT_WRAPPERS
_cell_object_call(f) = false

_cell_whole_call(f::Symbol) = f in REDUCTION_FNS || f in _WHOLE_READ_FNS ||
    (!_builtin_value_head(f) && !_cell_object_call(f))
_cell_whole_call(f::GlobalRef) = !_cell_object_call(f.name)
_cell_whole_call(f::Expr) = f.head === :. && length(f.args) == 2 &&
    f.args[2] isa QuoteNode && !_cell_object_call(f.args[2].value)
_cell_whole_call(f) = false

# Strip a validated cell expression to whole values and take the broadcast
# form of every call and operator over a per-index operand (`v[i]`, a
# gather, or a cell local in `pidx` bound from one). Returns `(expr,
# per_index)`. Already-dotted calls and operators keep their spelling;
# operators over scalar-only operands stay undotted (they are scalars in
# every iteration). `object = true` also dots distribution constructors
# and response wrappers (`_cell_object_call`) so an observation draws once
# per index.
_dotify_cell(ex, ivar, pidx, object::Bool) = (ex, false)
_dotify_cell(ex::Symbol, ivar, pidx, object::Bool) = (ex, ex in pidx)
function _dotify_cell(ex::Expr, ivar, pidx, object::Bool)
    ex.head === :ref && return (_strip_cell(ex, ivar), true)
    # Ref denotes one whole shared argument to a broadcasted constructor.
    # Broadcasting Ref itself would replace that argument by scalar wrappers.
    Meta.isexpr(ex, :call, 2) && ex.args[1] === :Ref &&
        return (ex, false)
    Meta.isexpr(ex, :call, 1) && return (ex, false)
    if ex.head === :call && !isempty(ex.args)
        f = ex.args[1]
        args = Any[]
        per = false
        for a in ex.args[2:end]
            if a isa Expr && (a.head === :parameters || a.head === :kw)
                push!(args, _strip_cell(a, ivar))
                continue
            end
            da, pa = _dotify_cell(a, ivar, pidx, object)
            push!(args, da)
            per |= pa
        end
        f isa Symbol || return (Expr(:call, f, args...), per)
        dotted = startswith(string(f), ".")
        (dotted || !(per || object && _cell_object_call(f))) &&
            return (Expr(:call, f, args...), per)
        Base.isoperator(f) &&
            return (Expr(:call, Symbol(".", f), args...), true)
        kws = filter(a -> a isa Expr && a.head === :parameters, args)
        pos = filter(a -> !(a isa Expr && a.head === :parameters), args)
        return (Expr(:., f, Expr(:tuple, kws..., pos...)), true)
    end
    if ex.head === :. && length(ex.args) == 2 && ex.args[2] isa Expr &&
            ex.args[2].head === :tuple
        args = Any[]
        per = false
        for a in ex.args[2].args
            da, pa = _dotify_cell(a, ivar, pidx, object)
            push!(args, da)
            per |= pa
        end
        return (Expr(:., ex.args[1], Expr(:tuple, args...)), per)
    end
    out = Any[]
    per = false
    for a in ex.args
        da, pa = _dotify_cell(a, ivar, pidx, object)
        push!(out, da)
        per |= pa
    end
    return (Expr(ex.head, out...), per)
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

# A `.~` range over a definition reading only data observes that derived
# response at the selected entries, as a range over the same value bound as
# data does. Levels, matrix and subset indices size coefficient priors.
function _derived_response_range(lhs, bc::Bool, response_defs::Set{Symbol})
    bc && Meta.isexpr(lhs, :ref) && length(lhs.args) >= 2 &&
        lhs.args[1] isa Symbol && lhs.args[1] in response_defs || return false
    index = lhs.args[2]
    return !(_is_levels_call(index) || _is_axes2_call(index) || Meta.isexpr(index, :ref))
end

# A ref LHS over a deterministic definition: ranges cover raw data
# columns and levels/matrix sizings size coefficient priors — neither
# observes a derived column that reads parameters (broadcast bare:
# `ly .~ ...`). Runs before `_sample_lhs` so the message names the
# position, not the fallout.
function _reject_derived_ref_lhs(lhs, detnames::Set{Symbol})
    lhs isa Expr && lhs.head === :ref && length(lhs.args) == 2 &&
        lhs.args[1] isa Symbol && lhs.args[1] in detnames || return nothing
    target = lhs.args[1]
    index = lhs.args[2]
    if _is_levels_call(index) || _is_axes2_call(index) ||
            (index isa Expr && index.head === :ref)
        _sfail("`$(repr(lhs))` sizes a coefficient prior but `$target` " *
                "is a deterministic definition — rename the definition " *
                "or the prior")
    end
    return _sfail("response range `$(repr(lhs))` covers a raw data " *
                  "column — `$target` is a derived column, broadcast bare " *
                  "(`$target .~ ...`)")
end

# Sampling-statement LHS: a bare Symbol, a one-dimensional range ref
# `y[R]` (`.~` only), a levels ref `c[levels(g)]` / `c[levels(g)][S]`
# (`.~` only), or a matrix-sized ref `b[axes(X, 2)]` (`.~` only).
# Returns `(column, range, levels, matrix)` with at most one of `range` /
# `levels` / `matrix` set (`matrix` is the sizing design matrix).
_sample_lhs(lhs::Symbol, bc, tilde, data,
        level_bindings = Dict{Symbol,Tuple{Symbol,Any}}()) =
    (lhs, nothing, nothing, nothing)

# Each row is one multivariate draw, exactly as in Julia. Only the
# structural list of category columns is stored; observations stay a loop.
function _multinomial_rows_lhs(lhs, bc, data)
    lhs isa Expr && lhs.head === :call && length(lhs.args) == 2 &&
        lhs.args[1] === :eachrow && _is_hcat_def(lhs.args[2]) || return nothing
    bc || _sfail("multinomial rows broadcast with `.~`: " *
        "`eachrow(hcat(c1, c2, ...)) .~ Multinomial.(N, Ref(s))`")
    cols = lhs.args[2].args[2:end]
    !isempty(cols) && all(c -> c isa Symbol && c in data, cols) ||
        _sfail("multinomial rows take bare count data columns: " *
            "`eachrow(hcat(c1, c2, ...))`, got $(repr(lhs))")
    return Vector{Symbol}(cols)
end

function _sample_lhs(lhs, bc, tilde, data, level_bindings)
    lhs isa Expr || _sfail("$tilde left-hand side must be a bare Symbol, " *
                           "a range ref (`y[1:N]`), a levels ref " *
                           "(`c[levels(g)]`), or a matrix-sized ref " *
                           "(`b[axes(X, 2)]`), got $(repr(lhs))")
    lhs.head === :. && _sfail("dotted left-hand side $(repr(lhs)) does " *
                              "not lower (nested targets are out of scope)")
    if lhs.head === :ref && length(lhs.args) > 2 && lhs.args[1] isa Symbol && lhs.args[1] in data
        bc || _sfail("a response slice broadcasts with `.~`")
        all(a -> a isa Integer && !(a isa Bool) && a >= 1, lhs.args[3:end]) ||
            _sfail("response trailing indices must be positive literal integers")
        range = _lower_lhs_range(lhs.args[1], lhs.args[2])
        range isa UnitRange && (range = Expr(:ref, lhs.args[1],
            Expr(:call, :(:), first(range), last(range))))
        range isa Expr || _sfail("multidimensional response slices need an index axis")
        append!(range.args, lhs.args[3:end])
        return lhs.args[1], range, nothing, nothing
    end
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
        target in data && return target, _lower_lhs_range(target, index), nothing, nothing
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

# Declared array parameters ([`ArrayParameter`](@ref)): `z[1:K] .~ ...`
# (a literal range over a non-data name) and two-axis `z[a, b] .~ ...`
# (each axis `1:K`, `levels(g)`, or `axes(M, d)`). One-axis
# `z[levels(g)]` / `z[axes(X, 2)]` have separate parse arms but the same
# ordinary array semantics. A `levels(gg)` axis over a definition `gg`
# (data computed at bind, `gg = vcat(g1, g2)`) and `axes(X, 1)` use this arm too.
# Returns `(name, dims)` or `nothing` (not an array).
function _array_sample_lhs(lhs, bc::Bool, tilde, data::Set{Symbol},
        detnames::Set{Symbol} = Set{Symbol}())
    lhs isa Expr && lhs.head === :ref || return nothing
    target = lhs.args[1]
    target isa Symbol && target ∉ data || return nothing
    idx = lhs.args[2:end]
    if length(idx) == 1
        row_axis = _is_axis_dim(idx[1]) && idx[1].args[1] === :axes &&
            idx[1].args[3] == 1 && idx[1].args[2] !== target
        _is_literal_range(idx[1]) || _is_def_levels_call(idx[1], data,
            detnames) || row_axis || _is_unique_axis(idx[1]) ||
            (idx[1] isa Symbol && idx[1] in union(data, detnames)) || return nothing
    elseif length(idx) != 2
        _sfail("array $target takes one or two axes, got $(repr(lhs))")
    end
    bc || _sfail("array `$(repr(lhs))` is declared elementwise — use " *
                 "`.~`, not $tilde (`$(repr(lhs)) .~ Normal.(0, 1)`)")
    return target, Any[_array_axis(target, a, data, detnames) for a in idx]
end

# `levels(gg)` over a definition name (not data).
_is_def_levels_call(a, data::Set{Symbol}, detnames::Set{Symbol}) =
    _is_levels_call(a) && length(a.args) == 2 && a.args[2] isa Symbol &&
    a.args[2] ∉ data && a.args[2] in detnames

# A `levels(gg)` axis over a definition needs `gg` to be data that
# `bind_data` computes: a definition calling a module function on data
# only (`gg = vcat(g1, g2)`), whose distinct values the axis enumerates.
# Runs on resolved definitions (module calls are `GlobalRef`s).
function _check_definition_levels_axes(sample, det, data::Set{Symbol})
    detmap = Dict{Symbol,Any}(det)
    for s in sample
        s.dims === nothing && continue
        for d in s.dims
            # `levels(gg)` and its positional twin `length(levels(gg)) - k`
            cnt = _levels_count(d)
            gg = _is_levels_dim(d) ? d.args[2] :
                cnt === nothing ? nothing : first(cnt)
            (gg === nothing || gg in data) && continue
            _is_bind_data_definition(gg, detmap, data, Set{Symbol}()) ||
                _sfail("array $(s.lhs) axis `levels($gg)`: $gg must be " *
                    "data — a raw column, or a definition that calls a " *
                    "function on data only (`$gg = vcat(g1, g2)`), " *
                    "computed once at bind")
        end
    end
    return nothing
end

function _is_bind_data_definition(nm::Symbol, detmap, data::Set{Symbol},
        active::Set{Symbol})
    haskey(detmap, nm) && nm ∉ active || return false
    rhs = detmap[nm]
    push!(active, nm)
    ok = all(v -> v in data ||
            _is_bind_data_definition(v, detmap, data, active),
        _expr_value_symbols(rhs))
    delete!(active, nm)
    return ok
end

# Names an expression reads as whole values: every Symbol except call
# heads, dotted function names, keyword names and the base of an index
# (`z` in `z[g]`, whose indices are still walked).
function _whole_name_reads!(out::Set{Symbol}, ex)
    if ex isa Symbol
        push!(out, ex)
    elseif ex isa Expr
        if ex.head === :ref
            ex.args[1] isa Symbol || _whole_name_reads!(out, ex.args[1])
            foreach(a -> _whole_name_reads!(out, a), ex.args[2:end])
        elseif ex.head === :call
            foreach(a -> _whole_name_reads!(out, a), ex.args[2:end])
        elseif ex.head === :kw
            _whole_name_reads!(out, ex.args[2])
        elseif ex.head === :. && length(ex.args) == 2 &&
                ex.args[2] isa Expr && ex.args[2].head === :tuple
            foreach(a -> _whole_name_reads!(out, a), ex.args[2].args)
        else
            foreach(a -> _whole_name_reads!(out, a), ex.args)
        end
    end
    return out
end

_is_literal_range(r) = r isa Expr && r.head === :call && length(r.args) == 3 &&
    r.args[1] === :(:)

# The internal per-level declaration `_ppl_per_level(L, g) ~
# LKJCholesky(K, eta)` that `@plate for k in levels(g); L[k] ~
# LKJCholesky(K, eta); end` desugars to: `(L, dims)` with dims
# `[K, K, levels(g)]` (`nothing` otherwise).
function _per_level_lhs(lhs, rhs, bc::Bool)
    (lhs isa Expr && lhs.head === :call && length(lhs.args) == 3 &&
        lhs.args[1] === :_ppl_per_level) || return nothing
    L, g = lhs.args[2], lhs.args[3]
    (L isa Symbol && g isa Symbol && !bc && _is_lkj_cholesky_call(rhs)) ||
        _sfail("internal: malformed per-level declaration $(repr(lhs))")
    K = rhs.args[2]
    return L, Any[K, K, :(levels($g))]
end

# Multivariate slice declarations ([`ArrayParameter`](@ref) slice
# families, `mv_slices.jl`): `eachrow(B[a, b]) .~ D` / `eachcol(B[a, b])
# .~ D` — Julia's `eachrow` / `eachcol` make the rows / columns of the
# two-axis array `B` the broadcast elements, and a distribution broadcasts
# as a scalar (Distributions.jl), so every slice is one draw of the
# multivariate `D` — and `b[ax] ~ D` with a multivariate normal `D` (the
# vector `b` one draw). The axes are the array axis forms (`1:K`,
# `levels(g)`, `axes(M, d)`). Returns `(name, dims, slices)` with
# `slices` one of `:rows`, `:cols`, `:vector`, or `nothing` (not a slice
# declaration).
function _array_slices_lhs(lhs, rhs, bc::Bool, tilde, data::Set{Symbol})
    if lhs isa Expr && lhs.head === :call && length(lhs.args) >= 1 &&
            lhs.args[1] in (:eachrow, :eachcol)
        it = lhs.args[1]
        length(lhs.args) == 2 && lhs.args[2] isa Expr &&
            lhs.args[2].head === :ref && length(lhs.args[2].args) == 3 &&
            lhs.args[2].args[1] isa Symbol || _sfail("slice declaration " *
                "`$(repr(lhs))` takes one two-axis array " *
                "(`$it(B[levels(g), 1:K]) .~ MvNormalCholesky(mu, F)`)")
        target = lhs.args[2].args[1]
        target in data && _sfail("`$(repr(lhs))`: $target is data — a " *
            "slice declaration declares a parameter array")
        bc || _sfail("`$(repr(lhs))` broadcasts over the slices — use " *
            "`.~`, not $tilde (`$(repr(lhs)) .~ MvNormalCholesky(mu, F)`)")
        return target, Any[_array_axis(target, a, data)
            for a in lhs.args[2].args[2:3]], it === :eachrow ? :rows : :cols
    end
    # `b[ax] ~ MvNormal(...)`: one draw of a multivariate normal.
    _mv_vector_head(rhs) || return nothing
    lhs isa Expr && lhs.head === :ref && lhs.args[1] isa Symbol &&
        lhs.args[1] ∉ data || return nothing
    target = lhs.args[1]
    head = _mv_head(rhs)
    length(lhs.args) == 2 || _sfail("`$(repr(lhs)) ~ $head(...)` draws " *
        "one vector — a two-axis array draws its slices " *
        "(`eachrow($(repr(lhs))) .~ $head(...)` or `eachcol(...)`)")
    bc && _sfail("`$(repr(lhs))` is one draw of the multivariate `$head` " *
        "— use `~`, not `.~` (`.~` would draw every element separately)")
    return target, Any[_array_axis(target, lhs.args[2], data)], :vector
end

function _array_axis(target::Symbol, a, data::Set{Symbol},
        detnames::Set{Symbol} = Set{Symbol}())
    if a isa Symbol && a in union(data, detnames)
        return Expr(:call, :_ppl_axis_values, a)
    elseif _is_unique_axis(a)
        call = a.head === :ref ? a.args[1] : a
        length(call.args) == 2 && call.args[2] isa Symbol &&
            call.args[2] in union(data, detnames) || _sfail("array $target: " *
                "`unique` enumerates a bound data column")
        d = Expr(:call, :unique, call.args[2])
        a.head === :ref && push!(d.args, QuoteNode(
            _lower_levels_subset(target, call.args[2], a.args[2])))
        return d
    end
    if _is_literal_range(a)
        lo, hi = a.args[2], a.args[3]
        count = _static_count(hi)
        count === nothing || (hi = count)
        cnt = lo === 1 ? _levels_count(hi) : nothing
        if cnt !== nothing
            # `1:length(levels(g)) - k`: a positional axis whose length is
            # the number of distinct values of `g`, less k (resolved at
            # bind). `g` is a level source exactly as for a `levels(g)`
            # axis: data, or a definition computed at bind
            # (`_check_definition_levels_axes`).
            g, k = cnt
            g in union(data, detnames) || _sfail("array $target axis " *
                "$(repr(a)): `levels($g)` needs a data grouping column " *
                "or a definition computed from data — $g is neither")
            n = Expr(:call, :length, Expr(:call, :levels, g))
            return k == 0 ? n : Expr(:call, :-, n, k)
        end
        lo === 1 && hi isa Integer && !(hi isa Bool) && hi >= 0 || _sfail(
            "array $target axis $(repr(a)) must be a literal `1:K` with " *
            "K ≥ 0, or `1:length(levels(g)) - k`")
        return Int(hi)
    end
    _is_def_levels_call(a, data, detnames) && return a
    if a isa Expr && a.head === :ref && !isempty(a.args) &&
            _is_levels_call(a.args[1])
        g, subset = _levels_subset_index(target, a, data)
        return Expr(:call, :levels, g, QuoteNode(subset))
    end
    if _is_levels_call(a)
        return Expr(:call, :levels,
            _levels_column(target, a, data, "array $target"))
    end
    if a isa Expr && a.head === :call && length(a.args) == 3 &&
            a.args[1] === :axes && a.args[2] isa Symbol && a.args[3] in (1, 2)
        return Expr(:call, :axes, a.args[2], a.args[3])
    end
    return _sfail("array $target axis $(repr(a)) must be a literal `1:K`, " *
                  "`levels(g)` (optionally selected), or `axes(M, d)` " *
                  "of a matrix `M`")
end

# Joint-response statement: `[y1, y2] ~ MvNormalCholesky([mu1, mu2], L)`
# (SB's joint form). Outcomes are bare distinct data symbols; means are
# one location expression per outcome; the factor names an explicitly
# defined scale-vector times Cholesky-factor value. Width/linkage checks belong
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
                                "an explicit factor value (`L = sd .* C` " *
                                "over declared scale and LKJCholesky priors)")
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
    (_is_levels_call(rhs) && rhs.args[1] === :levels) ||
    rhs isa Expr && rhs.head === :ref && length(rhs.args) == 2 &&
        _is_levels_call(rhs.args[1]) && rhs.args[1].args[1] === :levels

_is_unique_axis(a) = a isa Expr &&
    ((a.head === :call && !isempty(a.args) && a.args[1] === :unique) ||
     (a.head === :ref && length(a.args) == 2 && _is_unique_axis(a.args[1])))

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
        if length(s.args) == 4
            lo, step, hi = s.args[2:end]
            all(x -> x isa Integer && !(x isa Bool), (lo, step)) &&
                lo >= 1 && step != 0 || _sfail("coefficient $col: " *
                "stepped subsets need a positive start and a nonzero integer step")
            hi === :end && return (Int(lo), Int(step), :end)
            hi isa Integer && !(hi isa Bool) || _sfail("coefficient $col: " *
                "subset endpoint is a literal integer or `end`")
            return collect(Int(lo):Int(step):Int(hi))
        end
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
    # A literal `a:b` is its authored range. Each consumer states its own
    # requirement: a plate observation selects these indices, a top-level
    # response covers its whole column (the IR contract), and a per-cell
    # latent defines cells `1:N`.
    if r isa Expr && r.head === :call && length(r.args) == 3 && r.args[1] === :(:)
        lo, hi = _static_count(r.args[2]), _static_count(r.args[3])
        lo isa Integer && hi isa Integer || _sfail("response $col range " *
            "$(repr(r)) has an unbound endpoint — no count is bound in the " *
            "surface; write `eachindex($col)` or a literal range")
        return UnitRange(lo, hi)
    end
    # Self-covering forms: the column's own full index set, by construction.
    if r isa Expr && r.head === :call && !isempty(r.args) && r.args[1] === :eachindex
        length(r.args) == 2 && r.args[2] isa Symbol || _sfail(
            "response $col range takes `eachindex($col)` — " *
            "got $(repr(r))")
        return Expr(:ref, col, r)
    end
    if r isa Expr && r.head === :call && !isempty(r.args) && r.args[1] === :axes
        length(r.args) == 3 && r.args[2] isa Symbol && r.args[3] isa Integer &&
            !(r.args[3] isa Bool) && r.args[3] >= 1 || _sfail(
            "response $col range takes `axes(v, d)` with a positive literal axis — got $(repr(r))")
        return Expr(:ref, col, r)
    end
    r === :(:) && return Expr(:ref, col, r)
    (r isa Symbol || r isa Integer) &&
        _sfail("scalar cell index `$col[$(r)]` at top level does not " *
               "lower — per-cell refs live in `@plate` (slice B); " *
               "broadcast with `.~` or `eachindex`")
    return _sfail("response $col range must be `1:N`, `eachindex($col)`, " *
                  "or `axes($col, 1)` — got $(repr(r))")
end

"""One `~` / `.~` statement: scalar (`~`) or elementwise (`.~`) density.
`range` carries a literal `y[1:N]` response range or an indexed response
expression for `eachindex`, `axes` or `:` (`nothing` = bare whole-column
LHS). `levels` carries a
`(grouping column, subset)` pair for `c[levels(g)]` broadcast priors
(`nothing` otherwise). `matrix` carries the sizing design matrix for
`b[axes(X, 2)]` coefficient-vector priors (`nothing` otherwise). `dims`
carries the axes of a declared array parameter (`z[1:K] .~ ...`,
`z[levels(g), axes(Z, 2)] .~ ...` — see [`ArrayParameter`](@ref);
`nothing` otherwise). `slices` marks a multivariate slice declaration
— `:rows` for `eachrow(B[a, b]) .~ D`, `:cols` for `eachcol(B[a, b]) .~
D`, `:vector` for `b[ax] ~ D` (each slice one draw of the multivariate
`D`) — and is `nothing` otherwise. `count_columns` carries the tail of
`eachrow(hcat(c1, c2, ...))`, with `lhs` the lead count column."""
struct SampleStmt
    lhs::Symbol
    rhs::Any
    broadcast::Bool
    range::Union{Nothing,UnitRange{Int},Expr}
    levels::Any
    matrix::Union{Nothing,Symbol}
    dims::Union{Nothing,Vector{Any}}
    slices::Union{Nothing,Symbol}
    count_columns::Union{Nothing,Vector{Symbol}}
end
SampleStmt(lhs::Symbol, rhs, broadcast::Bool, range, levels, matrix, dims, slices) =
    SampleStmt(lhs, rhs, broadcast, range, levels, matrix, dims, slices, nothing)
SampleStmt(lhs::Symbol, rhs, broadcast::Bool, range, levels, matrix, dims) =
    SampleStmt(lhs, rhs, broadcast, range, levels, matrix, dims, nothing)
SampleStmt(lhs::Symbol, rhs, broadcast::Bool, range, levels, matrix) =
    SampleStmt(lhs, rhs, broadcast, range, levels, matrix, nothing)
SampleStmt(lhs::Symbol, rhs, broadcast::Bool) =
    SampleStmt(lhs, rhs, broadcast, nothing, nothing, nothing)

"""One joint `[y1, y2] ~ MvNormalCholesky([mu1, mu2], L)` statement: K
outcome columns, K mean expressions, and an explicitly defined factor value.
Plain `~`
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
    wrapper::Union{Nothing,Expr}
end

const _GLM_HEADS = (:NormalIDGLM, :BernoulliLogitGLM, :PoissonLogGLM)

function _glm_distribution(rhs)
    rhs isa Expr && rhs.head === :call && !isempty(rhs.args) || return rhs
    first(rhs.args) in (:truncated, :censored, :interval_censored) &&
        length(rhs.args) >= 2 && return rhs.args[2]
    return rhs
end
_is_glm_call(rhs) = (dist = _glm_distribution(rhs); dist isa Expr &&
    dist.head === :call && !isempty(dist.args) && first(dist.args) in _GLM_HEADS)

function _parse_glm_stmt(st::Expr, line::Int, data::Set{Symbol})
    lhs, raw = st.args[2], st.args[3]
    rhs = _glm_distribution(raw)
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
    return GLMSampleStmt(lhs, head, X, alpha, beta, sigma, line, raw === rhs ? nothing : raw)
end

_is_doc_macro(m) =
    m === Symbol("@doc") || (m isa GlobalRef && m.name === Symbol("@doc"))

function _unwrap_trivia(st::Expr)
    while st.head === :macrocall
        if st.args[1] === Symbol("@plate") && length(st.args) == 4
            _sfail("the legacy `@plate result for ...` form is retired; use ordinary indexed `@plate for ...` statements with explicit arrays")
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
# resolves in `mod` to an `RKPPLSubmodel` is rewritten into inline statements,
# so the result lowers exactly like the hand-inlined model (the submodel is
# transparent). Two kinds, read from the submodel's RETURN (a trailing
# `return x` unwraps to `x` in `_submodel_body_parts`, so both spellings lower
# identically):
#   • latent (`latent ~ sm(a,b)`, `latent` NOT data) — returns a VALUE
#     expression; every local has a private identifier and a trailing
#     `latent = <return>` binds the LHS.
#   • observation stream (`y ~ sm(a,b)`, `y` a data column) — returns a bare
#     `slot` that is the LHS of an internal `slot .~ family.(...)` response;
#     the slot maps to the data column (`y .~ family.(y_…)`), the rest is
#     namespaced under `y`, and there is NO trailing binding (the response IS
#     the binding).
#
# The namespacing rule (one rule for every statement form, so a body holds
# whatever a top-level program can):
#   • Every name the body BINDS receives one private identifier throughout the
#     body: the LHS of a `~` / `.~` / `=` statement (the base name of an
#     indexed LHS — `c[levels(g)] .~ …` binds `c`, `b[axes(X, 2)] .~ …` binds
#     `b`, a slice declaration `eachrow(B[levels(g), 1:K]) .~ …` binds `B`),
#     a `@plate` result, cell or loop variable, a `@scan` carried state,
#     step local or loop variable, and a `do`-block argument
#     (`_collect_binders!`). Index expressions keep their shape; names inside
#     them follow the same substitution. Author paths are separate metadata.
#   • Each argument is replaced by the call's argument expression (the
#     language is pure, so this has value semantics). A body may observe an
#     argument bound to a DATA column (`y .~ Normal.(mu, s)` with `y` passed
#     in); any other statement binding an argument name fails closed.
#   • Call heads, dotted function names, keyword names and macro names are
#     never renamed; every other name is left as written (a free name refers
#     to the calling program).
#   • A nested submodel call in a body expands after the enclosing body is
#     substituted, so scopes compose: `z ~ outer(...)` whose body holds
#     `w ~ inner(...)` whose body binds `b` exposes `z.w.b`. A nested call
#     resolves in the module that defined the enclosing submodel.
# Private identifiers are fresh against caller names, data, arguments and
# free body names. A static `z.b` read resolves through the scope table;
# bare `z` remains the actual returned value. Flat author spellings cannot
# collide with locals, and no wrapper enters the mathematical graph.
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
    if head isa GlobalRef
        mod, head = head.mod, head.name
    elseif head isa Expr && head.head === :. && length(head.args) == 2 &&
            head.args[2] isa QuoteNode && head.args[2].value isa Symbol
        mod = _resolve_module_path(head.args[1], mod, "submodel call")
        head = head.args[2].value
    end
    head isa Symbol || return nothing
    if !isdefined(mod, head)
        haskey(_RETIRED_CONSTRUCT_MODELS, head) && _sfail("the implicit `$head` " *
            "construct is retired; write ordinary priors and values, or use " *
            "`BayesianRegressionModels.rkppl_model(:$(_RETIRED_CONSTRUCT_MODELS[head]))`")
        head in _RETIRED_MODEL_HEADS && _sfail("the statistical model `$head` " *
            "is owned by BayesianRegressionModels; obtain its ordinary body with " *
            "`BayesianRegressionModels.rkppl_model(:$head)` and bind it in the model module")
        head === :linear_pk_log_f && _sfail("the PK-specific model `$head` " *
            "belongs to downstream RKPPLBench; import the consumer-owned submodel " *
            "into the model module")
        return nothing
    end
    val = getfield(mod, head)
    fragment = sampling_fragment(val)
    (fragment === nothing || fragment isa RKPPLSubmodel) ||
        _sfail("sampling_fragment for $head must return RKPPLSubmodel or nothing")
    return fragment
end

# Reserve the complete reachable source vocabulary before allocating any
# private name. A free name in a later/nested callee must not capture an
# identifier already allocated for an earlier call. Visit each body once;
# the expansion cycle check remains responsible for rejecting recursion.
function _reserve_submodel_names!(used::Set{Symbol}, ex, mod::Module,
        seen::Set{RKPPLSubmodel})
    ex isa Expr || return nothing
    if _is_sample(ex) || _is_broadcast_sample(ex)
        sm = _resolve_submodel(ex.args[3], mod)
        if sm !== nothing && !(sm in seen)
            push!(seen, sm)
            union!(used, _submodel_args(sm))
            for (_, default) in sm.kwdefaults
                _all_symbols!(used, default)
            end
            _all_symbols!(used, sm.body)
            _reserve_submodel_names!(used, sm.body, sm.mod, seen)
            for edit in sm.rewrites
                _all_symbols!(used, edit)
                _reserve_submodel_names!(used, edit, sm.mod, seen)
            end
        end
    end
    for arg in ex.args
        _reserve_submodel_names!(used, arg, mod, seen)
    end
    return nothing
end

# A top-level statement that is a plain `~` whose RHS resolves to a submodel
# (pure predicate; the latent/stream compatibility check happens on expansion).
function _stmt_is_submodel_call(arg, mod::Module)
    arg isa Expr || return false
    _is_block_macro(arg) && return false
    st = try
        _unwrap_trivia(arg)
    catch
        return false
    end
    st isa Expr && _is_sample(st) || return false
    _resolve_submodel(st.args[3], mod) === nothing && return false
    return st.args[2] isa Symbol
end

# `@plate` / `@scan` block statements: kept raw (never trivia-unwrapped) on
# every submodel path — their own desugar owns them.
_is_block_macro(st) = st isa Expr && st.head === :macrocall &&
    !isempty(st.args) && st.args[1] in (Symbol("@plate"), Symbol("@scan"))

mutable struct _ScopeExpansion
    scopes::Vector{SubmodelScope}
    by_binding::Dict{Symbol,SubmodelScope}
    name_paths::Dict{Symbol,Tuple{Vararg{Symbol}}}
    used::Set{Symbol}
    rewrites::Vector{Tuple{Expr,Module}}
    fixes::Dict{Symbol,ColumnData}
    bound_values::Union{Nothing,Dict{Symbol,ColumnData}}
    declared::Set{Symbol}
end
_ScopeExpansion(scopes, bindings, paths, used) =
    _ScopeExpansion(scopes, bindings, paths, used,
        Tuple{Expr,Module}[], Dict{Symbol,ColumnData}(), nothing, Set{Symbol}())

function _new_submodel_scope!(ctx::_ScopeExpansion, binding::Symbol;
        per_cell::Bool = false)
    haskey(ctx.by_binding, binding) && _sfail(
        "submodel binding `$binding` is defined more than once " *
        "(a call owns one lexical namespace)")
    path = get(ctx.name_paths, binding, (binding,))
    scope = SubmodelScope(path, binding, Dict{Symbol,Symbol}(), per_cell)
    push!(ctx.scopes, scope)
    ctx.by_binding[binding] = scope
    return scope
end

# A local's plan identifier is spelled as its authored path (`var"z.w.b"`),
# so generated programs read the names coordinates and draws already use.
# A spelling that occurs in a name in use (an author's literal `var"z.b"`, a
# name reserved from a callee body, or an earlier identifier) takes the next
# free `#k` suffix instead; the path itself stays in `name_paths`.
function _scope_private_name!(ctx::_ScopeExpansion, scope::SubmodelScope,
        name::Symbol)
    haskey(scope.locals, name) && return scope.locals[name]
    path = (scope.path..., name)
    spelling = join(string.(path), '.')
    identifier = Symbol(spelling)
    suffix = 1
    while any(nm -> occursin(string(identifier), string(nm)), ctx.used)
        suffix += 1
        identifier = Symbol(spelling, '#', suffix)
    end
    push!(ctx.used, identifier)
    scope.locals[name] = identifier
    ctx.name_paths[identifier] = path
    return identifier
end

# Property reads of a call resolve lexically before mathematical lowering.
# A nested call remains a scope; an ordinary local or the call's returned
# value remains an ordinary Julia value. No runtime value wrapper is built.
_resolve_scope_properties(x, ::_ScopeExpansion) = x
function _resolve_scope_properties(ex::Expr, ctx::_ScopeExpansion)
    if Meta.isexpr(ex, :(=), 2) || _is_sample(ex) || _is_broadcast_sample(ex)
        lhs = _stmt_lhs(ex)
        root = lhs
        dotted = false
        while root isa Expr && root.head in (:., :ref)
            dotted |= root.head === :.
            root = first(root.args)
        end
        dotted && root isa Symbol && haskey(ctx.by_binding, root) && _sfail(
            "scoped local `$(repr(lhs))` is read-only: a submodel owns " *
            "its declarations (bind a distinct caller name instead)")
    end
    if ex.head === :. && length(ex.args) == 2 &&
            ex.args[2] isa QuoteNode && ex.args[2].value isa Symbol
        parent = _resolve_scope_properties(ex.args[1], ctx)
        field = ex.args[2].value
        binding = parent isa Symbol ? parent :
            Meta.isexpr(parent, :ref) && parent.args[1] isa Symbol ?
            parent.args[1] : nothing
        scope = binding === nothing ? nothing : get(ctx.by_binding, binding, nothing)
        if scope !== nothing && haskey(scope.locals, field)
            local_name = scope.locals[field]
            if Meta.isexpr(parent, :ref) && scope.per_cell
                return Expr(:ref, local_name, parent.args[2:end]...)
            elseif parent isa Symbol
                return local_name
            end
        end
        # A property absent from the namespace is a property of the actual
        # returned value. This also covers properties of ordinary local values.
        if any(nm -> haskey(ctx.by_binding, nm) || haskey(ctx.name_paths, nm),
                _value_symbols(parent))
            return Expr(:call, GlobalRef(Base, :getproperty), parent, ex.args[2])
        end
        return Expr(:., parent, ex.args[2])
    end
    return Expr(ex.head, Any[_resolve_scope_properties(a, ctx) for a in ex.args]...)
end

function _expand_submodels(ast::Expr, data::Set{Symbol}, mod::Module;
        with_scopes::Bool = false, rewrites = Expr[], fixes = Dict(), bound_values = nothing)
    if !any(_stmt_is_submodel_call(a, mod) || _plate_has_submodel_cell(a, mod) for a in ast.args)
        isempty(rewrites) && isempty(fixes) || _sfail("merge scoped target matches no submodel declaration")
        return with_scopes ? (ast, SubmodelScope[]) : ast
    end
    # Names in use by the program (binders at every depth + data): a
    # namespaced name must be fresh against these, and every expansion adds
    # its own names (two expansions never collide).
    used = copy(data)
    _all_symbols!(used, ast)
    _reserve_submodel_names!(used, ast, mod, Set{RKPPLSubmodel}())
    for edit in rewrites
        _all_symbols!(used, edit)
        _reserve_submodel_names!(used, edit, mod, Set{RKPPLSubmodel}())
    end
    for a in ast.args
        _collect_binders!(used, a)
    end
    out = Any[]
    ctx = _ScopeExpansion(SubmodelScope[], Dict{Symbol,SubmodelScope}(),
        Dict{Symbol,Tuple{Vararg{Symbol}}}(), used)
    union!(ctx.declared, data)
    for a in ast.args
        _collect_binders!(ctx.declared, a)
    end
    ctx.rewrites = Tuple{Expr,Module}[(edit, mod) for edit in rewrites]
    ctx.fixes = Dict{Symbol,ColumnData}(fixes)
    ctx.bound_values = bound_values
    for arg in ast.args
        _expand_submodel_stmt!(out, arg, mod, data, ctx,
            RKPPLSubmodel[])
    end
    isempty(ctx.rewrites) && isempty(ctx.fixes) ||
        _sfail("merge scoped target matches no statement (or names an unsupported plate/scan cell)")
    expanded = _resolve_scope_properties(Expr(:block, out...), ctx)
    return with_scopes ? (expanded, ctx.scopes) : expanded
end

# Expand one statement into `out`, recursively: a submodel call is inlined and
# its generated statements are expanded in turn (nested calls resolve in the
# defining module of the enclosing submodel); a plate with per-cell submodel
# cells is rewritten cell by cell; anything else passes through.
function _expand_submodel_stmt!(out, arg, mod::Module, data::Set{Symbol},
        used::_ScopeExpansion,
        chain::Vector{RKPPLSubmodel})
    if _stmt_is_submodel_call(arg, mod)
        st = _unwrap_trivia(arg)
        sm, gen = _expand_one_submodel(st.args[2], st.args[3], mod, data,
            used, chain)
        inner = RKPPLSubmodel[chain; sm]
        gen = _scope_program_edits(sm, gen, st, used, data)
        for g in gen
            _expand_submodel_stmt!(out, g, sm.mod, data, used, inner)
        end
    elseif _plate_has_submodel_cell(arg, mod)
        push!(out, _expand_plate_cell_submodels(arg, mod, data, used, chain))
    else
        push!(out, arg)
    end
    return out
end

_path_property(path) = foldl((a,b) -> Expr(:., a, QuoteNode(b)), path)
_replace_statement(st, lhs, rhs) = _is_sample(st) || _is_broadcast_sample(st) ?
    Expr(:call, first(st.args), lhs, rhs) : Expr(:(=), lhs, rhs)
function _replace_scope_path(lhs, path)
    if lhs isa Expr && lhs.head === :ref
        return Expr(:ref, _path_property(path), lhs.args[2:end]...)
    elseif lhs isa Expr && lhs.head === :call && first(lhs.args) in (:eachrow, :eachcol)
        return Expr(:call, first(lhs.args), _replace_scope_path(lhs.args[2], path))
    end
    return _path_property(path)
end

# Rewrite the declaration before its child calls expand. Replacing or pinning
# a submodel use site therefore removes the entire former child program.
function _scope_program_edits(sm, gen, call, ctx, data)
    scope = ctx.by_binding[call.args[2]]
    args = _bind_submodel_arguments(call.args[3], sm)
    substitutions = merge(Dict{Symbol,Any}(zip(_submodel_args(sm), args)), scope.locals)
    variant_edits = Tuple{Expr,Module}[]
    for raw in sm.rewrites
        path = (scope.path..., _merge_path(_stmt_lhs(raw))...)
        edit = _hsubst(raw, substitutions)
        lhs = _replace_scope_path(_stmt_lhs(edit), path)
        push!(variant_edits, (_replace_statement(edit, lhs, last(edit.args)), sm.mod))
    end
    prepend!(ctx.rewrites, variant_edits)
    for (relative, value) in sm.fixed
        path = Symbol(join((scope.path..., Symbol.(split(String(relative), '.'))...), "."))
        haskey(ctx.fixes, path) || (ctx.fixes[path] = value)
    end
    direct(path) = path !== nothing && length(path) == length(scope.path) + 1 &&
        path[1:end-1] == scope.path
    program = RKPPLModel(Expr(:block, gen...), sm.mod)
    remaining = Tuple{Expr,Module}[]
    for (edit, edit_mod) in ctx.rewrites
        lhs = _stmt_lhs(edit)
        path = _merge_path(lhs)
        if !direct(path)
            push!(remaining, (edit, edit_mod))
            continue
        end
        local_name = last(path)
        haskey(scope.locals, local_name) || _sfail("merge scope $(scope.path) has no local $local_name")
        name = scope.locals[local_name]
        lhs = _replace_scope_path(lhs, (name,))
        rhs = last(edit.args)
        callee = _resolve_submodel(rhs, edit_mod)
        if callee !== nothing && first(rhs.args) isa Symbol
            rhs = Expr(:call, GlobalRef(edit_mod, first(rhs.args)), rhs.args[2:end]...)
        end
        program = Base.merge(program, _replace_statement(edit, lhs, rhs))
    end
    ctx.rewrites = remaining
    for key in collect(keys(ctx.fixes))
        path = Tuple(Symbol.(split(String(key), '.')))
        direct(path) || continue
        haskey(scope.locals, last(path)) || _sfail("merge pin $key matches no scoped statement")
        name, value = scope.locals[last(path)], pop!(ctx.fixes, key)
        program = _merge_fixes(program, NamedTuple{(name,)}((value,)); scoped_keys = false)
        if ctx.bound_values === nothing
            helper = value isa AbstractArray ? :_bound_array_value : :_bound_value
            push!(program.ast.args, Expr(:(=), name,
                Expr(:call, GlobalRef(@__MODULE__, helper), QuoteNode(value))))
        else
            ctx.bound_values[name] = value
            push!(data, name)
        end
    end
    return program.ast.args
end

# A submodel that (transitively) calls itself never terminates: fail naming
# the cycle.
function _check_submodel_cycle(sm::RKPPLSubmodel, chain)
    any(c -> c === sm, chain) || return nothing
    path = join([string(c.name) for c in chain], " → ")
    return _sfail("submodel `$(sm.name)` calls itself ($path → $(sm.name)) " *
                  "— a recursive submodel never finishes expanding")
end

# Split a submodel body into (statements, return-expression). A trailing
# explicit `return x` unwraps to `x`, so it lowers identically to the implicit
# trailing-expression form on every path (stream + latent, top-level +
# per-cell): all of them read `ret` from here. Statements are any top-level
# program statement — the downstream lowering owns their gates, exactly as
# for the hand-inlined program; `@plate` / `@scan` blocks stay raw.
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
    stmts = _desugar_destructuring(Any[
        (st isa Expr && !_is_block_macro(st)) ? _unwrap_trivia(st) : st
        for st in items[1:end-1]])
    for st in stmts
        Meta.isexpr(st, :return) && _sfail(
            "submodel `$(sm.name)`: `return` is only admitted as the " *
            "trailing expression (submodels are straight-line; no early return)")
    end
    (ret isa Expr && (_is_block_macro(ret) || _is_sample(ret) ||
        _is_broadcast_sample(ret) ||
        (ret.head === :(=) && length(ret.args) == 2))) && _sfail(
        "submodel `$(sm.name)` must end in a RETURN expression bound to the " *
        "use-site LHS (a bare value, not a `~`/`=` statement or a block)")
    return stmts, ret
end

_stmt_lhs(st::Expr) =
    (_is_sample(st) || _is_broadcast_sample(st)) ? st.args[2] : st.args[1]

# ── Binders ──────────────────────────────────────────────────────────────
# Every name a statement binds, at any depth: `~` / `.~` / `=` LHSs (the base
# name of an indexed LHS, each name of a `[a, b]` LHS), `@plate` results,
# cells and loop variables, `@scan` states, step locals and loop variables,
# and `do`-block arguments. Nested submodel calls contribute only their
# use-site LHS (their bodies bind under it when they expand). With `sampled = false` a
# `~` / `.~` LHS is skipped — the one position where a data argument is
# observed rather than bound.
function _collect_binders!(out::Set{Symbol}, st, sampled::Bool = true)
    st isa Expr || return out
    h = st.head
    if h === :macrocall
        isempty(st.args) && return out
        if st.args[1] === Symbol("@plate") && length(st.args) == 4 &&
                st.args[3] isa Symbol
            push!(out, st.args[3])
        end
        _collect_binders!(out, st.args[end], sampled)
    elseif h === :block
        for a in st.args
            _collect_binders!(out, a, sampled)
        end
    elseif h === :for && length(st.args) == 2
        _collect_iter_binders!(out, st.args[1])
        _collect_binders!(out, st.args[2], sampled)
    elseif h === :call && length(st.args) == 3 && st.args[1] in (:~, :.~)
        sampled && _collect_lhs_binders!(out, st.args[2])
        _collect_do_binders!(out, st.args[3], sampled)
    elseif h === :(=) && length(st.args) == 2
        _collect_lhs_binders!(out, st.args[1])
        _collect_do_binders!(out, st.args[2], sampled)
    end
    return out
end

_collect_lhs_binders!(out::Set{Symbol}, lhs::Symbol) = push!(out, lhs)
function _collect_lhs_binders!(out::Set{Symbol}, lhs::Expr)
    if lhs.head === :ref && !isempty(lhs.args)
        _collect_lhs_binders!(out, lhs.args[1])
    elseif lhs.head === :call && length(lhs.args) == 2 &&
            lhs.args[1] in (:eachrow, :eachcol)
        # A slice declaration (`eachrow(B[levels(g), 1:K]) .~ D`) binds `B`.
        _collect_lhs_binders!(out, lhs.args[2])
    elseif lhs.head in (:vect, :tuple)
        for a in lhs.args
            _collect_lhs_binders!(out, a)
        end
    end
    return out
end
_collect_lhs_binders!(out::Set{Symbol}, _) = out

function _collect_iter_binders!(out::Set{Symbol}, head)
    head isa Expr || return out
    if head.head === :(=) && length(head.args) == 2
        _collect_lhs_binders!(out, head.args[1])
    elseif head.head === :block
        for a in head.args
            _collect_iter_binders!(out, a)
        end
    end
    return out
end

function _collect_do_binders!(out::Set{Symbol}, rhs, sampled::Bool)
    rhs isa Expr && rhs.head === :do && length(rhs.args) == 2 || return out
    lam = rhs.args[2]
    lam isa Expr && lam.head === :-> && length(lam.args) == 2 || return out
    _collect_lhs_binders!(out, lam.args[1])
    _collect_binders!(out, lam.args[2], sampled)
    return out
end

# Every Symbol under `ex` (call heads included; QuoteNodes opaque).
function _all_symbols!(out::Set{Symbol}, ex)
    if ex isa Symbol
        push!(out, ex)
    elseif ex isa Expr
        for a in ex.args
            _all_symbols!(out, a)
        end
    end
    return out
end

# ── Hygienic substitution ────────────────────────────────────────────────
# Rename value-position Symbols per `map`.
# Call heads, dotted function names (`f.(…)`), keyword names and macro names
# are never renamed; a bare keyword shorthand (`f(; k)`, meaning `k = k`)
# expands to `k = <renamed>` when `k` is renamed. QuoteNodes remain opaque.
_hsubst(ex::Symbol, map::AbstractDict) = get(map, ex, ex)
_hsubst(ex, ::AbstractDict) = ex
function _hsubst(ex::Expr, map::AbstractDict)
    h = ex.head
    sub(a) = _hsubst(a, map)
    if h === :call && !isempty(ex.args)
        f = ex.args[1]
        if f isa Symbol && get(map, f, nothing) isa GlobalRef
            f = map[f]
        end
        args = Any[f]
        for (k, a) in enumerate(ex.args)
            k == 1 && continue
            push!(args, sub(a))
        end
        return Expr(:call, args...)
    elseif h === :. && length(ex.args) == 2 && ex.args[2] isa Expr &&
            ex.args[2].head === :tuple
        f = ex.args[1]
        f isa Symbol && get(map, f, nothing) isa GlobalRef && (f = map[f])
        return Expr(:., f, sub(ex.args[2]))
    elseif h === :kw && length(ex.args) == 2
        return Expr(:kw, ex.args[1], sub(ex.args[2]))
    elseif h === :tuple
        # `(a = v, …)`: a NamedTuple key is a name, not a value.
        return Expr(:tuple, Any[(a isa Expr && a.head === :(=) &&
            length(a.args) == 2 && a.args[1] isa Symbol) ?
            Expr(:(=), a.args[1], sub(a.args[2])) : sub(a)
            for a in ex.args]...)
    elseif h === :parameters
        return Expr(:parameters, Any[(a isa Symbol && haskey(map, a)) ?
            Expr(:kw, a, map[a]) : sub(a) for a in ex.args]...)
    elseif h === :macrocall
        return Expr(:macrocall, ex.args[1],
            Any[a isa LineNumberNode ? a : sub(a) for a in ex.args[2:end]]...)
    end
    return Expr(h, Any[sub(a) for a in ex.args]...)
end

# A call head that names an argument or a body binder would silently stay
# unrenamed (heads are never substituted): fail closed naming it.
function _check_call_heads(sm::RKPPLSubmodel, ex, names::Set{Symbol}, substitutions)
    ex isa Expr || return nothing
    f = nothing
    if ex.head === :call && !isempty(ex.args)
        f = ex.args[1]
    elseif ex.head === :. && length(ex.args) == 2 && ex.args[2] isa Expr &&
            ex.args[2].head === :tuple
        f = ex.args[1]
    end
    value = get(substitutions, f, nothing)
    callable = value isa GlobalRef && getglobal(value.mod, value.name) isa Function
    f isa Symbol && f in names && !callable && _sfail(
        "submodel `$(sm.name)`: `$f` is an argument or local name but is " *
        "called as a function — a submodel's names are values; rename it")
    for a in ex.args
        _check_call_heads(sm, a, names, substitutions)
    end
    return nothing
end

# A stream submodel returns a bare `slot` Symbol that is the LHS of exactly one
# internal `.~` response statement. Returns that statement, or `nothing` for a
# latent submodel (value return / non-response slot).
function _stream_response(sm::RKPPLSubmodel, stmts, ret)
    ret isa Symbol || return nothing
    hits = findall(st -> st isa Expr && _is_broadcast_sample(st) &&
        st.args[2] === ret, stmts)
    isempty(hits) && return nothing
    length(hits) == 1 || _sfail("stream submodel `$(sm.name)`: return `$ret` " *
        "names more than one `.~` response")
    return stmts[first(hits)]
end

# A stream used as a latent samples its private return slot. Its vector
# arguments supply the plate axis; scalars stay shared. The alias at the
# use site remains separate from the namespace containing its priors.
function _generative_stream_stmts(stmts, ret, data)
    det = Pair{Symbol,Any}[st.args[1] => st.args[2] for st in stmts
        if Meta.isexpr(st, :(=), 2) && st.args[1] isa Symbol]
    defs = Dict{Symbol,Any}(det)
    shapes = _def_shapes(det, data, defs)
    aligned(x) = _obs_axis(x, data, defs, shapes, Set{Symbol}(), _NO_SHAPE_ENV)
    function anchor(ex)
        ex isa Symbol && return aligned(ex) ? ex : nothing
        ex isa Expr || return nothing
        ex.head === :call && ex.args[1] === :Ref && return nothing
        start = ex.head in (:call, :.) ? 2 : 1
        for a in ex.args[start:end]
            found = anchor(a)
            found === nothing || return found
        end
        return nothing
    end
    function indexed(ex, ivar)
        ex isa Symbol && return aligned(ex) ? Expr(:ref, ex, ivar) : ex
        ex isa Expr || return ex
        ex.head === :call && ex.args[1] === :Ref && return ex
        start = ex.head in (:call, :.) ? 2 : 1
        return Expr(ex.head, ex.args[1:start-1]...,
            (indexed(a, ivar) for a in ex.args[start:end])...)
    end
    out = Any[]
    for st in stmts
        if _is_broadcast_sample(st) && st.args[2] === ret
            obj = st.args[3]
            source = anchor(obj)
            if source === nothing
                push!(out, Expr(:call, :~, ret, _undot_cell_object(obj)))
            else
                ivar = gensym(:_rkppl_stream_index)
                cell = Expr(:call, :~, Expr(:ref, ret, ivar),
                    _undot_cell_object(indexed(obj, ivar)))
                push!(out, Expr(:macrocall, Symbol("@plate"), LineNumberNode(1),
                    Expr(:for, Expr(:(=), ivar, Expr(:call, :eachindex, source)),
                        Expr(:block, cell))))
            end
        else
            push!(out, st)
        end
    end
    return out
end

# Bind positional arguments and declared keyword defaults.
function _bind_submodel_arguments(callexpr::Expr, sm::RKPPLSubmodel)
    posargs = Any[]
    keywords = Dict{Symbol,Any}()
    for a in callexpr.args[2:end]
        if a isa Expr && a.head === :parameters
            for kw in a.args
                kw isa Expr && kw.head === :kw && length(kw.args) == 2 ||
                    _sfail("submodel `$(sm.name)`: malformed keyword " *
                           "$(repr(a)) (expected `name = value`)")
                k = kw.args[1]
                if any(p -> first(p) === k, sm.kwdefaults)
                    haskey(keywords, k) && _sfail("submodel `$(sm.name)`: duplicate keyword `$k`")
                    keywords[k] = kw.args[2]
                    continue
                end
                _sfail("submodel `$(sm.name)`: unknown keyword `$k`; " *
                       "only declared keywords are accepted. Name model " *
                       "quantities with ordinary assignments or declarations.")
            end
        else
            push!(posargs, a)
        end
    end
    substitutions = Dict{Symbol,Any}(zip(sm.argnames, posargs))
    length(posargs) == length(sm.argnames) || _sfail(
        "submodel `$(sm.name)` expects $(length(sm.argnames)) positional arguments")
    for (k, default) in sm.kwdefaults
        value = get(keywords, k) do
            resolved = _resolve_module_calls(default, sm.mod,
                Set{Symbol}(_submodel_args(sm)),
                "submodel `$(sm.name)` keyword default `$k`")
            _hsubst(resolved, substitutions)
        end
        push!(posargs, value)
        substitutions[k] = value
    end
    return posargs
end

# The substitution for one use site: arguments → call expressions, body
# binders → `ns(name)` (except names `keep` maps itself). Fails closed when a
# binder is an argument name, except a `~` / `.~` observation of an argument
# bound to a data column.
function _submodel_substitution(sm::RKPPLSubmodel, stmts, ret, callargs,
        data::Set{Symbol}, ns, keep::AbstractDict)
    argnames = _submodel_args(sm)
    argset = Set{Symbol}(argnames)
    binders = Set{Symbol}()
    defined = Set{Symbol}()   # binders outside a `~` / `.~` LHS
    for st in stmts
        _collect_binders!(binders, st)
        _collect_binders!(defined, st, false)
    end
    submap = Dict{Symbol,Any}()
    for (a, v) in zip(argnames, callargs)
        submap[a] = v
    end
    for nm in sort!(collect(binders); by = string)
        if nm in argset
            v = submap[nm]
            observed = !(nm in defined)
            (observed && v isa Symbol && v in data) || _sfail(
                "submodel `$(sm.name)`: `$nm` is both an argument and a " *
                "local — rename the local" *
                (observed ? " (a body may observe an argument with `~`/`.~` " *
                    "only when it is bound to a data column; `$nm` is bound " *
                    "to $(repr(v)))" : ""))
            continue
        end
        submap[nm] = get(keep, nm, nothing) === nothing ? ns(nm) : keep[nm]
        v = submap[nm]
        ns.scope.locals[nm] = v isa Symbol ? v : v.args[1]::Symbol
    end
    names = union(argset, binders)
    for st in stmts
        _check_call_heads(sm, st, names, submap)
    end
    _check_call_heads(sm, ret, names, submap)
    return submap
end

# The namespace root of a use site: `ns` closes over it (`_NsRoot`).
struct _NsRoot
    root::Symbol
    indexed::Union{Nothing,Symbol}   # per-cell: the plate loop variable
    scope::SubmodelScope
    context::_ScopeExpansion
end
function (n::_NsRoot)(nm::Symbol)
    identifier = _scope_private_name!(n.context, n.scope, nm)
    return n.indexed === nothing ? identifier :
        Expr(:ref, identifier, n.indexed)
end

function _expand_one_submodel(lhs::Symbol, callexpr::Expr, mod::Module,
                              data::Set{Symbol},
                              used::_ScopeExpansion, chain)
    sm = _resolve_submodel(callexpr, mod)::RKPPLSubmodel
    _check_submodel_cycle(sm, chain)
    callargs = _bind_submodel_arguments(callexpr, sm)
    callargs = Any[_resolve_function_arg(a, mod,
        union(used.declared, Set{Symbol}(keys(used.name_paths))),
        "submodel `$(sm.name)` argument") for a in callargs]
    length(callargs) == length(_submodel_args(sm)) || _sfail(
        "submodel `$(sm.name)` expects $(length(sm.argnames)) argument(s) " *
        "$(Tuple(sm.argnames)), got $(length(callargs)) at `$lhs ~ " *
        "$(sm.name)(...)`")
    stmts, ret = _submodel_body_parts(sm)
    for st in stmts
        _all_symbols!(used.used, st)
    end
    _all_symbols!(used.used, ret)
    stream = _stream_response(sm, stmts, ret) !== nothing
    # Shape compatibility (dots follow the callee): a data column is
    # vector-shaped, so it admits only a vectorized (stream) callee; a non-data
    # LHS binds a latent value.
    if lhs in data
        stream || _sfail("`$lhs` is a data column, so `$(sm.name)` must be an " *
            "observation-stream submodel: end its body by returning a `slot` " *
            "that is the LHS of an internal `slot .~ family.(...)` response. " *
            "`$(sm.name)` returns a value (latent) — use a non-data LHS.")
        ret in _submodel_args(sm) && _sfail("stream submodel `$(sm.name)`: the " *
            "response slot `$ret` is an argument — the slot is a local name " *
            "the use-site data column replaces")
    end
    observed = stream && lhs in data
    # An observed slot binds to the data LHS. A generative slot retains
    # its private name beside every other binder in the call's namespace.
    keep = observed ? Dict{Symbol,Any}(ret => lhs) : Dict{Symbol,Any}()
    scope = _new_submodel_scope!(used, lhs)
    submap = _submodel_substitution(sm, stmts, ret,
        callargs, data, _NsRoot(lhs, nothing, scope, used), keep)
    # Body definitions call functions visible in the submodel's OWN module
    # (functions as values); resolution precedes substitution (`GlobalRef`
    # heads and function values pass `_hsubst` intact).
    bodynames = Set{Symbol}(k for k in keys(submap) if k isa Symbol)
    union!(bodynames, _submodel_args(sm))
    functions = Dict(k => v for (k, v) in submap if v isa GlobalRef &&
        getglobal(v.mod, v.name) isa Function)
    # Observations: arguments bound to data columns, and an observed slot.
    observes = Set{Symbol}(a for a in _submodel_args(sm)
        if get(submap, a, nothing) isa Symbol && submap[a] in data)
    observed && ret isa Symbol && push!(observes, ret)
    out = Any[_hsubst(_resolve_submodel_stmt(_hsubst(st, functions),
        sm, bodynames; observes), submap)
        for st in stmts]
    stream && !observed && (out = _generative_stream_stmts(out, submap[ret], data))
    # Every latent call binds its return value. An observed stream already
    # binds the response column, so it has no trailing assignment.
    observed || push!(out, Expr(:(=), lhs, _hsubst(_resolve_module_calls(
        _hsubst(ret, functions),
        sm.mod, bodynames, "submodel `$(sm.name)` return `$(repr(ret))`"),
        submap)))
    return sm, out
end

function _resolve_submodel_stmt(st, sm::RKPPLSubmodel, names::Set{Symbol};
        observes::Set{Symbol} = Set{Symbol}())
    st isa Expr || return st
    # Plate cell definitions execute in the submodel's defining module too.
    # Visit statement positions only: sampling heads and loop binders retain
    # their DSL spelling until the ordinary surface passes lower them.
    if st.head === :block
        return Expr(:block, (_resolve_submodel_stmt(a, sm, names; observes)
            for a in st.args)...)
    elseif st.head === :macrocall && !isempty(st.args) &&
            st.args[1] === Symbol("@plate")
        return Expr(:macrocall, st.args[1:end-1]...,
            _resolve_submodel_stmt(st.args[end], sm, names; observes))
    elseif st.head === :for && length(st.args) == 2
        return Expr(:for, st.args[1],
            _resolve_submodel_stmt(st.args[2], sm, names; observes))
    end
    if (_is_sample(st) || _is_broadcast_sample(st)) && _external_rhs(last(st.args)) &&
            _resolve_submodel(last(st.args), sm.mod) === nothing
        rhs = st.args[3]
        resolved = rhs isa Symbol && rhs ∉ names ?
            _resolve_call_head(rhs, sm.mod, names, "submodel sampling RHS") :
            _resolve_module_calls(rhs, sm.mod, names, "submodel sampling RHS")
        return Expr(:call, st.args[1], st.args[2],
            resolved)
    end
    if (_is_sample(st) || _is_broadcast_sample(st)) &&
            !_external_rhs(st.args[3]) &&
            any(in(observes), _sampled_lhs_names!(Set{Symbol}(), st.args[2]))
        # A module call among an observation's values resolves here, like
        # the definition naming it would (`_observation_call_definitions`
        # then names it).
        where = "submodel `$(sm.name)` sampling RHS `$(repr(st.args[3]))`"
        rhs = _map_rhs_values(st.args[3]) do value
            _bind_module_calls(value) do call
                _resolve_module_calls(call, sm.mod, names, where)
            end
        end
        return rhs == st.args[3] ? st : Expr(:call, st.args[1], st.args[2], rhs)
    end
    (st.head === :(=) && length(st.args) == 2 &&
        (st.args[1] isa Symbol || Meta.isexpr(st.args[1], :ref))) || return st
    rhs = st.args[2]
    _is_levels_binding_rhs(rhs) && return st
    return Expr(:(=), st.args[1], _resolve_module_calls(rhs, sm.mod, names,
        "submodel `$(sm.name)` definition `$(st.args[1]) = $(repr(rhs))`"))
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
# primitive already supports. A cell is scalar, so a per-cell body holds
# scalar `~` / `=` statements over bare names only (and per-cell submodel
# calls, which expand per cell in turn: `col_w_b[i]`).

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
function _expand_plate_cell_submodels(pl::Expr, mod::Module, data::Set{Symbol},
        used::_ScopeExpansion, chain)
    loop = pl.args[end]::Expr
    asg = loop.args[1]::Expr
    ivar = asg.args[1]::Symbol
    body = loop.args[2]::Expr
    cells = Any[]
    for c in body.args
        _expand_cell_stmt!(cells, c, ivar, mod, data, used, chain)
    end
    newloop = Expr(:for, asg, Expr(:block, cells...))
    return Expr(:macrocall, pl.args[1:end-1]..., newloop)
end

# Expand one plate cell into `cells`, recursively (a per-cell body may call a
# per-cell submodel in turn; it resolves in the enclosing submodel's module).
function _expand_cell_stmt!(cells, c, ivar::Symbol, mod::Module,
        data::Set{Symbol}, used::_ScopeExpansion, chain)
    call = _cell_submodel_call(c, ivar, mod)
    if call === nothing
        push!(cells, c)
    else
        sm, gen = _expand_cell_submodel(call[1], call[2], ivar, mod, data,
            used, chain)
        inner = RKPPLSubmodel[chain; sm]
        for g in gen
            _expand_cell_stmt!(cells, g, ivar, sm.mod, data, used, inner)
        end
    end
    return cells
end

_is_dotted_obj(st::Expr) =
    (_is_sample(st) || _is_broadcast_sample(st)) &&
        st.args[3] isa Expr && st.args[3].head === :.

# Inline one per-cell submodel call `col[i] ~ sm(callargs…)` into a sequence of
# `i`-indexed cell statements. The submodel's own `~`/`=` names are namespaced
# under `col` with private identifiers indexed per cell; positional args bind to
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
                               mod::Module, data::Set{Symbol},
                               used::_ScopeExpansion, chain)
    col = colref.args[1]::Symbol
    sm = _resolve_submodel(callexpr, mod)::RKPPLSubmodel
    _check_submodel_cycle(sm, chain)
    callargs = _bind_submodel_arguments(callexpr, sm)
    length(callargs) == length(_submodel_args(sm)) || _sfail(
        "submodel `$(sm.name)` expects $(length(sm.argnames)) argument(s) " *
        "$(Tuple(sm.argnames)), got $(length(callargs)) at `$col[$ivar] ~ " *
        "$(sm.name)(...)`")
    stmts, ret = _submodel_body_parts(sm)
    for st in stmts
        _all_symbols!(used.used, st)
    end
    _all_symbols!(used.used, ret)
    # A cell is scalar: a per-cell body holds scalar `~` / `=` statements over
    # bare names (no indexed priors, plates or scans).
    for st in stmts
        (st isa Expr && (_is_sample(st) || _is_broadcast_sample(st) ||
            (st.head === :(=) && length(st.args) == 2)) &&
            _stmt_lhs(st) isa Symbol) || _sfail(
            "submodel `$(sm.name)` is called per cell (`$col[$ivar] ~ " *
            "$(sm.name)(...)`), so its body holds scalar `~`/`=` statements " *
            "over bare names; `$(repr(st))` is not one (call it at top level " *
            "for plates, scans and sized priors)")
    end
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
    end
    latent_stream = col ∉ data && slot !== nothing && _is_dotted_obj(stmts[slot])
    argset = Set{Symbol}(_submodel_args(sm))
    for st in stmts
        nm = _stmt_lhs(st)
        nm in argset && _sfail("submodel `$(sm.name)`: `$nm` is both an " *
            "argument and a local statement — rename the local")
    end
    # A returned nested call keeps its own scope instead of claiming the
    # outer call's binding. The outer cell then reads that call's return.
    if slot !== nothing && _is_sample(stmts[slot]) &&
            _resolve_submodel(stmts[slot].args[3], sm.mod) !== nothing
        slot = nothing
    end
    # Build the substitution: args → call args; each internal name → its indexed
    # namespaced ref, except the direct-bound slot → `col[i]`.
    keep = slot === nothing || latent_stream ? Dict{Symbol,Any}() :
        Dict{Symbol,Any}(_stmt_lhs(stmts[slot]) => colref)
    scope = _new_submodel_scope!(used, col; per_cell = true)
    submap = _submodel_substitution(sm, stmts, ret,
        callargs, data, _NsRoot(col, ivar, scope, used), keep)
    out = Any[_hsubst(st, submap) for st in stmts]
    # Compound-return latent: bind `col[i]` to the substituted return value.
    (slot === nothing || latent_stream) &&
        push!(out, Expr(:(=), colref, _hsubst(ret, submap)))
    return sm, out
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
    # A keyword argument's name is not a value (`f(x; k = 4)`).
    ex.head === :kw && length(ex.args) == 2 &&
        return _value_symbols!(out, ex.args[2])
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
    if ex.head === :ref && !isempty(ex.args)
        # A whole axis (`P[:, 1]`) or an endpoint (`P[:, end]`) is a
        # position of the indexed value, not a name.
        _value_symbols!(out, ex.args[1])
        for i in ex.args[2:end]
            i === :(:) || _is_endpoint_position(i) || _value_symbols!(out, i)
        end
        return nothing
    end
    for a in ex.args
        ex.head === :tuple && (a = _tuple_field_value(a))
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

# A derived response observes data-only vector structure: structural
# (predictor) definitions and direct sampled-name reads fail here (the
# bind evaluator owns transitive closure — a scalar-definition chain
# into a parameter fails at bind naming the name).
function _gate_derived_response!(lhs::Symbol, ctx)
    lhs in ctx.structural && _sfail("response $lhs is a predictor " *
        "definition — responses observe data columns or data-derived " *
        "columns (`ly = log.(earn)`)")
    for v in _value_symbols(ctx.detmap[lhs])
        v in ctx.prior_names && _sfail("response $lhs reads the " *
            "sampled name $v — responses derive from bound data only")
    end
    return nothing
end

# Broadcast-LHS rejection: scalar/matrix definitions name their shape;
# anything else keeps the legacy not-data message verbatim.
function _broadcast_lhs_msg(lhs::Symbol, detshape)
    shape = get(detshape, lhs, nothing)
    shape === :scalar && return "`.~` broadcasts over a data column " *
        "or a vector-shaped derived column — $lhs is a scalar " *
        "definition (scalar parameters use `~`)"
    shape === :matrix && return "`.~` broadcasts over a data column " *
        "or a vector-shaped derived column — $lhs is a design matrix " *
        "(matrices lower only in predictor matmuls)"
    return "`.~` broadcasts over a data column — $lhs is not data " *
        "(scalar parameters use `~`)"
end

function _lower_response(lhs, rhs, range, ctx, predictors, pred_idx, coefuse;
        count_columns = nothing)
    dotted = _desugar_fused_head(lhs, rhs)
    call = _dot2call_response(lhs, dotted)
    weights, call = _peel_weighted(lhs, call, ctx)
    evidence, call = _peel_evidence(lhs, call, ctx)
    count_columns === nothing || call.args[1] === :Multinomial ||
        _sfail("response $lhs: count rows use `Multinomial.(N, Ref(s))`")
    call.args[1] === :MixtureModel && return _lower_mixture_response(lhs,
        call, range, weights, evidence, ctx, predictors, pred_idx, coefuse)
    if call.args[1] in (:CategoricalLogit, :OrderedLogistic, :Ordinal,
            :Multinomial, :Categorical)
        return _lower_leveled_response(lhs, call, range, weights, evidence,
            ctx, predictors, pred_idx, coefuse; count_columns)
    end
    family, lik_link, pred_link, loc, scale_raw, trials, nu_raw, zi_raw,
    interval_raw = _lower_response_base(lhs, call, ctx)
    # Prob-space families retain a sampled scalar or a fixed value
    # predictor. Every other family
    # routes through `_lower_location`, which admits a value location
    # here (a scalar parameter or data column under the written link).
    pname = family === BinomialProbFam ?
        _lower_prob_location(lhs, loc, ctx, predictors, pred_idx,
            family === BinomialProbFam ? "Binomial" : "ZeroInflatedBinomial") :
        family in (BetaShapeFam, GammaValueFam, WeibullValueFam) ?
        _lower_argument_predictor!(lhs, loc, ctx, predictors, pred_idx,
            coefuse, Symbol(lhs, "_value")) :
        _lower_location(lhs, loc, pred_link, ctx, predictors, pred_idx,
        coefuse; value = true)
    scale = _lower_scale_use(lhs, scale_raw, ctx, predictors, pred_idx,
        coefuse)
    nu = _lower_nu_use(lhs, nu_raw, ctx, predictors, pred_idx, coefuse)
    zi = _lower_zi_use(lhs, zi_raw, ctx, predictors, pred_idx, coefuse)
    interval = _lower_interval_use(lhs, interval_raw, ctx)
    return LikelihoodSpec(family, lik_link, lhs, pname, scale, weights,
        evidence, Symbol(lhs, "_resp"), trials, range; nu = nu, zi = zi,
        interval = interval)
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

# `y .~ MixtureModel.(vcat.(C1, ..., CK), Ref(w))` — K same-family univariate
# components + mixing weights (SB `MixtureModel` mirror). Each component
# lowers through the single-family base spelling (decomposed twin:
# predictors wrapped, params/literals bare); locations route to predictors
# (link-space) or scalar slots (sampled params / literals,
# constrained-scale); weights are a literal vector or a simplex name.
function _lower_mixture_response(lhs, call, range, weights, evidence, ctx,
        predictors, pred_idx, coefuse)
    label = Symbol(lhs, "_resp")
    range === nothing || _sfail("response $lhs: mixture responses take no " *
        "range (v1 — mixtures cover the whole column)")
    args = _plain_args(call, "`MixtureModel`")
    length(args) in (1, 2) || _sfail("response $lhs: `MixtureModel` takes " *
        "`MixtureModel.(vcat.(C1, ..., CK), Ref(w))` " *
        "(per-observation component vectors + shared weights)")
    comps = args[1]
    _is_dotted_call(comps) && comps.args[1] === :vcat ||
        _sfail("response $lhs: `MixtureModel` needs per-observation " *
            "component vectors; use `MixtureModel.(vcat.(Normal.(mu1, s), " *
            "Normal.(mu2, s)), Ref(w))`, got $(repr(comps))")
    components = comps.args[2].args
    K = length(components)
    K >= 1 || _sfail("response $lhs: `MixtureModel` needs ≥ 1 component")
    wraw = if length(args) == 1
        Expr(:vect, fill(1.0 / K, K)...)
    else
        _is_ref_call(args[2]) || _sfail("response $lhs: mixture weights are " *
            "shared as a vector — use `Ref(w)` in `MixtureModel.(..., Ref(w))`")
        args[2].args[2]
    end
    fams = LikelihoodFamily[]
    llinks = LinkFunction[]
    plinks = LinkFunction[]
    locs_raw = Any[]
    scales_raw = Any[]
    trials_raw = Any[]
    wrappeds = Bool[]
    for (k, c) in enumerate(components)
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
    canonical_family(f) = f in (BernoulliProbitFam, BernoulliCloglogFam) ?
        BernoulliLogitFam : f in (BinomialProbitFam, BinomialCloglogFam) ?
        BinomialLogitFam : f
    original_fams = copy(fams)
    fams = canonical_family.(fams)
    f = fams[1]
    all(==(f), fams) || _sfail("response $lhs: mixture components must " *
        "share one family (found $(join(unique!(string.(fams)), ", "))); " *
        "heterogeneous mixtures are not supported)")
    f in _MIXTURE_V1_FAMILIES || _sfail("response $lhs: mixture " *
        "components over $f are not admitted in v1 (admitted: Normal, " *
        "Bernoulli-logit, Poisson-log, Binomial-logit, " *
        "NegativeBinomial2-log, Gamma-log, Beta-logit)")
    canonical = _mixture_canon_link(f)
    value_components = !all(==(canonical), llinks) || !all(==(f), original_fams)
    ll = value_components ? IdentityLink : canonical
    trials = nothing
    if f === BinomialLogitFam
        t1 = trials_raw[1]
        trials = all(t -> isequal(t, t1), trials_raw) ? t1 : nothing
    end
    loc_uses = Union{Symbol,Real}[]
    for (k, loc) in enumerate(locs_raw)
        use = if value_components && wrappeds[k]
            value = _inverse_link_spelling(llinks[k], loc)
            _lower_location(lhs, value, IdentityLink, ctx, predictors,
                pred_idx, coefuse; value = true, synth = Symbol(lhs, :_mix_, k, :_value))
        else
            _lower_mixture_loc(lhs, k, loc, plinks[k], wrappeds[k], ctx,
                predictors, pred_idx, coefuse)
        end
        push!(loc_uses, use)
    end
    scale_uses = Union{Nothing,Symbol,Real,ScalePredictorRef}[]
    for (k, sc) in enumerate(scales_raw)
        lowered_sc = _mixture_component_context(lhs, k) do
            _lower_scale_use(lhs, sc, ctx, predictors, pred_idx, coefuse)
        end
        push!(scale_uses, lowered_sc)
    end
    wraw isa Symbol && wraw ∉ ctx.data && wraw ∉ ctx.dirichlet_names &&
        _sfail("response $lhs: mixture weights $wraw are undeclared; bind an input or state a Dirichlet prior")
    w = _lower_mixture_weights(lhs, wraw)
    prednames = Set{Symbol}(p.name for p in predictors)
    anchor = _mixture_anchor(lhs, loc_uses, scale_uses, w, prednames, ctx)
    return LikelihoodSpec(MixtureFam, ll, lhs, anchor, nothing, weights,
        evidence, label, trials, range; mixture_family = f,
        mixture_locs = loc_uses, mixture_scales = scale_uses,
        mixture_weights = w,
        mixture_trials = f === BinomialLogitFam && trials === nothing ?
            Union{ColumnRef,Int}[trials_raw...] : Union{ColumnRef,Int}[])
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
            return BernoulliLogitFam, IdentityLink, IdentityLink, args[1],
            nothing, nothing, false
        end
    elseif head === :Poisson
        args = _plain_args(compcall, "`Poisson`")
        if length(args) == 1 && (args[1] isa Symbol || args[1] isa Real)
            return PoissonLogFam, IdentityLink, IdentityLink, args[1], nothing,
            nothing, false
        end
    elseif head === :Binomial
        args = _plain_args(compcall, "`Binomial`")
        if length(args) == 2 && (args[2] isa Symbol || args[2] isa Real)
            return BinomialLogitFam, IdentityLink, IdentityLink, args[2],
            nothing, _lower_trials(lhs, args[1], ctx), false
        end
    elseif head === :NegativeBinomial2
        args = _plain_args(compcall, "`NegativeBinomial2`")
        if length(args) == 2 && (args[1] isa Symbol || args[1] isa Real)
            return NegativeBinomial2Fam, IdentityLink, IdentityLink, args[1], args[2],
            nothing, false
        end
    elseif head === :Gamma
        bare = _match_bare_gamma(lhs, compcall)
        bare !== nothing &&
            return GammaLogFam, IdentityLink, IdentityLink, bare[1], bare[2],
            nothing, false
    elseif head === :Beta
        bare = _match_bare_beta(lhs, compcall)
        bare !== nothing &&
            return BetaLogitFam, IdentityLink, IdentityLink, bare[1], bare[2],
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

# Mixture component locations use the same value/predictor lowering as a
# standalone response. A bare sampled parameter remains a constrained-scale
# slot; an explicit link applies equally to a parameter or a definition.
function _lower_mixture_loc(lhs, k::Int, loc, pred_link, wrapped::Bool, ctx,
        predictors, pred_idx, coefuse)
    if wrapped
        return _mixture_component_context(lhs, k) do
            _lower_location(lhs, loc, pred_link, ctx, predictors,
                pred_idx, coefuse; value=true,
                synth=Symbol(lhs, "_mix_", k, "_eta"))
        end
    end
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
        # parameter), so it never routes here. Only an undeclared
        # intercept-only def (the SB `mu ~ 1` mirror) did, and strict
        # declarations refuse that name at prior lowering: a constant
        # location is a declared scalar spelled bare.
        if loc in ctx.vecdefs && haskey(ctx.detmap, loc) ||
                _is_scalar_coef_def(loc, ctx, false)
            _derived_reads_latent(loc, ctx) &&
                !_is_design_shaped(loc, ctx) && _sfail("response $lhs: " *
                "mixture component $k location reads a latent — latent " *
                "mixture locations are a follow-up")
            return _lower_location(lhs, loc, pred_link, ctx, predictors,
                pred_idx, coefuse; value = true, synth = Symbol(lhs, "_mix_", k, "_eta"))
        end
        if haskey(ctx.detmap, loc) && pred_link === IdentityLink
            return _lower_location(lhs, loc, pred_link, ctx, predictors,
                pred_idx, coefuse; synth = Symbol(lhs, "_mix_", k, "_eta"),
                value = true)
        end
        if loc in ctx.prior_names
            wrapped && _sfail("response $lhs: mixture component $k " *
                "wraps the sampled parameter $loc in a link function — " *
                "link wrappers apply to predictors (spell sampled " *
                "parameters bare, constrained-scale)")
            return loc
        end
        if loc in ctx.data && pred_link === IdentityLink
            return _lower_location(lhs, loc, pred_link, ctx, predictors,
                pred_idx, coefuse; synth = Symbol(lhs, "_mix_", k, "_eta"),
                value = true)
        end
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
            pred_idx, coefuse; value = true, synth = Symbol(lhs, "_mix_", k, "_eta"))
    end
    loc isa Real && return loc
    loc isa Symbol && loc in ctx.prior_names && return loc
    _sfail("response $lhs: mixture component $k bare location $(repr(loc)) " *
        "must be a sampled parameter or literal; an explicit link " *
        "uses the ordinary response value lowering")
end

# Mixture weights: a literal numeric vector, a simplex parameter name,
# or a complement pair (length/sum/concentration checked at contract —
# structural, normative).
function _lower_mixture_weights(lhs, wraw)
    wraw isa Symbol && return wraw
    wraw isa Expr && wraw.head === :vect || _sfail("response $lhs: " *
        "mixture weights are a literal vector (`[0.4, 0.6]`), a " *
        "simplex parameter name, or a complement pair " *
        "(`[s, 1.0 - s]`), got $(repr(wraw))")
    # Complement pair: `[s, 1-s]` / `[1-s, s]` over one sampled
    # parameter (the collapsed Beta-weight shape).
    if length(wraw.args) == 2
        pair = _mixture_complement_pair(wraw.args)
        pair !== nothing && return pair
    end
    for (j, e) in enumerate(wraw.args)
        e isa Real && !(e isa Bool) || _sfail("response $lhs: mixture " *
            "weight $j is not a numeric literal (got $(repr(e))) — " *
            "sampled weights spell `[s, 1.0 - s]` over one parameter")
    end
    return Float64.(wraw.args)
end

# `[s, 1-s]` / `[1-s, s]` with `s` a Symbol, else `nothing`.
function _mixture_complement_pair(args)
    iscomp(e, s) = e isa Expr && e.head === :call && length(e.args) == 3 &&
        e.args[1] === :- && e.args[2] isa Real && e.args[2] == 1 &&
        e.args[3] === s
    a, b = args[1], args[2]
    a isa Symbol && iscomp(b, a) && return MixtureComplementWeights(a, true)
    b isa Symbol && iscomp(a, b) && return MixtureComplementWeights(b, false)
    return nothing
end

# The mixture anchor (the non-nullable `predictor` slot): first location
# predictor, else first scale predictor, else the weights simplex name,
# else the complement-pair parameter, else the first location/scale
# parameter name — the BRM struct order, verbatim. Fully-fixed mixtures
# fail closed before anchoring.
function _mixture_anchor(lhs, loc_uses, scale_uses, w, prednames, ctx)
    for loc in loc_uses
        loc isa Symbol && loc in prednames && return loc
    end
    for s in scale_uses
        s isa ScalePredictorRef && return s.predictor
    end
    w isa Symbol && return w
    w isa MixtureComplementWeights && return w.param
    for loc in loc_uses
        loc isa Symbol && loc in ctx.prior_names && return loc
    end
    for s in scale_uses
        s isa Symbol && s in ctx.prior_names && return s
    end
    for s in scale_uses
        s isa Symbol && return s # A data-column scale: opaque anchor, never resolved.
    end
    return lhs
end

const _LEVELED_FAMS =
    (:CategoricalLogit, :OrderedLogistic, :Ordinal, :Multinomial, :Categorical)

function _lower_leveled_response(lhs, call, range, weights, evidence, ctx,
        predictors, pred_idx, coefuse; count_columns = nothing)
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
            evidence, label, ctx; count_columns)
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
        pred_idx, coefuse; synth = Symbol(lhs, "_eta_", j), value = true)
        for (j, a) in enumerate(args)]
    return LikelihoodSpec(CategoricalLogitFam, LogitLink, lhs, pnames[1],
        nothing, weights, evidence, label, nothing, range;
        extra_predictors = pnames[2:end])
end

# Explicit cutpoints/thresholds of an ordinal response: the trailing
# argument `Ref(c)` names one declared vector shared by every observation
# (Distributions.jl `OrderedLogistic.(eta, Ref(c))` broadcast semantics).
# Either structure reads an Ordered vector or a one-axis sized vector
# with its stated element prior and support. A vector may serve several
# responses; its first use records a size source for `_lower_parameters`.
function _explicit_thresholds!(ctx, lhs::Symbol, arg, ordered::Bool,
        shown::String)
    name = arg isa Expr && arg.head === :call && length(arg.args) == 2 &&
        arg.args[1] === :Ref && arg.args[2] isa Symbol ? arg.args[2] : nothing
    if name === nothing
        arg isa Symbol && _sfail("response $lhs: the cutpoints $arg are " *
            "one vector shared by every observation — write `Ref($arg)` " *
            "(standard broadcasting would pair each observation with one " *
            "element of $arg)")
        _sfail("response $lhs: $shown takes its cutpoints as `Ref(c)` of " *
               "a declared vector, got $(repr(arg))")
    end
    if name ∉ ctx.ordered_names
        dims = get(ctx.array_dims, name, nothing)
        dims !== nothing && length(dims) == 1 || _sfail("response $lhs: " *
            "thresholds $name must be a one-axis vector declared in the " *
            "model (`$name[1:length(levels($lhs)) - 1] .~ Normal.(0, 1)`)")
    end
    # A declaration owns its vector; responses only read it. Keep the first
    # use for an inferred extent, then validate every reader's extent at bind.
    get!(ctx.threshold_uses, name, (response = lhs, ordered = ordered))
    return name
end

# Cumulative-logit ordinal reads declared cutpoints. USER 1cmodra (vectors)
# and 0d5a67r require the prior to be stated in the ordinary model body.
function _lower_ordered_logistic_response(lhs, call, range, weights,
        evidence, label, ctx, predictors, pred_idx, coefuse)
    args = _plain_args(call, "`OrderedLogistic`")
    length(args) == 2 || _sfail("response $lhs: `OrderedLogistic` " *
        "requires explicit cutpoints: declare their prior (for example " *
        "`c ~ Ordered(Normal(0, 1), length(levels($lhs)) - 1)`) and use " *
        "`$lhs .~ OrderedLogistic.(eta, Ref(c))`")
    pname = _lower_location(lhs, args[1], IdentityLink, ctx, predictors,
        pred_idx, coefuse; value = true)
    cut = _explicit_thresholds!(ctx, lhs, args[2], true, "`OrderedLogistic`")
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

# General typed ordinal:
# `y .~ Ordinal.(Cumulative(), LogitLink(), eta, Ref(c))` over declared
# thresholds (ordered iff cumulative — see `_explicit_thresholds!`).
# Optional positional arguments broadcast a
# discrimination value and a threshold-effect row with ordinary Julia
# semantics (decision 1m7stoc); broadcast keywords do not vary by row.
function _lower_ordinal_response(lhs, call, range, weights, evidence,
        label, ctx, predictors, pred_idx, coefuse)
    args = _plain_args(call, "`Ordinal`")
    4 <= length(args) <= 6 || _sfail("response $lhs: `Ordinal` requires " *
        "explicit thresholds with their stated prior: use " *
        "(structure, link, eta, Ref(c), discrimination, eachrow(effects)); " *
        "the last two arguments are optional")
    structure = _ordinal_tag(lhs, args[1], (:Cumulative, :StoppingRatio),
        "structure")
    linktag = _ordinal_tag(lhs, args[2],
        (:LogitLink, :ProbitLink, :CloglogLink), "link")
    pname = _lower_location(lhs, args[3], IdentityLink, ctx, predictors,
        pred_idx, coefuse; value = true)
    thresh = _explicit_thresholds!(ctx, lhs, args[4],
        structure === :Cumulative, "`Ordinal`")
    structure_sym = structure === :Cumulative ? :cumulative : :stopping
    disc = if length(args) < 5
        nothing
    elseif args[5] isa Real || args[5] isa Symbol && args[5] in ctx.data
        args[5]
    else
        name = _lower_location(lhs, args[5], IdentityLink, ctx, predictors,
            pred_idx, coefuse; value = true, synth = Symbol(lhs, :_disc))
        ScalePredictorRef(name, IdentityLink)
    end
    effects = length(args) == 6 ? _ordinal_effects_source(lhs, args[6], ctx) :
        nothing
    return LikelihoodSpec(OrdinalFam, _ORDINAL_LINKS[linktag], lhs, pname,
        nothing, weights, evidence, label, nothing, range;
        thresholds = thresh, ordinal_structure = structure_sym,
        discrimination = disc, threshold_effects = effects)
end

function _ordinal_effects_source(lhs, arg, ctx)
    arg isa Expr && arg.head === :call && length(arg.args) == 2 &&
        arg.args[1] === :eachrow && arg.args[2] isa Symbol ||
        _sfail("response $lhs: threshold effects broadcast as `eachrow(E)`, " *
            "where `E = X * delta` is an ordinary matrix value")
    name = arg.args[2]
    name in ctx.data || haskey(ctx.detmap, name) ||
        haskey(ctx.array_dims, name) || _sfail("response $lhs: threshold " *
            "effects $name are undeclared — define the matrix value first")
    return name
end

# Shared-simplex multinomial:
# `eachrow(hcat(c1, ..., cK)) .~ Multinomial.(N, Ref(s))` —
# the lead count column (LHS) plus the K−1 tail count columns, trials N
# (Int literal or column), and the simplex parameter `s`
# (`s ~ Dirichlet(...)` elsewhere in the model).
function _lower_multinomial_response(lhs, call, range, weights, evidence,
        label, ctx; count_columns = nothing)
    args = _plain_args(call, "`Multinomial`")
    count_columns !== nothing && length(args) == 2 && _is_ref_call(args[2]) ||
        _sfail("response $lhs: observe count rows with " *
            "`eachrow(hcat(c1, c2, ...)) .~ Multinomial.(N, Ref(s))`; " *
            "the two distribution arguments are trials and a shared " *
            "simplex vector")
    trials = _lower_trials(lhs, args[1], ctx)
    s = args[2].args[2]
    s isa Symbol || _sfail("response $lhs: multinomial probs $s must be " *
                           "a simplex parameter name " *
                           "(`s ~ Dirichlet(...)` in the model)")
    haskey(ctx.matrices, s) && _sfail("response $lhs: multinomial probs " *
                                      "$s is a design matrix — probs are a " *
                                      "simplex parameter " *
                                      "(`s ~ Dirichlet(...)` in the model)")
    return LikelihoodSpec(MultinomialFam, IdentityLink, lhs, s,
        nothing, weights, evidence, label, trials, range;
        count_columns = count_columns)
end

# Plain categorical over simplex probabilities: `y .~ Categorical(s)`.
function _lower_categorical_response(lhs, call, range, weights, evidence,
        label, ctx)
    args = _plain_args(call, "`Categorical`")
    length(args) == 1 || _sfail("response $lhs: `Categorical` takes " *
                                "`y .~ Categorical(s)` (a simplex parameter)")
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
# the factor value reads its declared scales and Cholesky factor.
function _lower_joint_response(j::JointSampleStmt, ctx, predictors, pred_idx,
        coefuse)
    tag = "[$(join(j.outcomes, ", "))]"
    rhs = get(ctx.detmap, j.factor, nothing)
    rhs isa Expr && rhs.head === :call && length(rhs.args) == 3 &&
        rhs.args[1] === :.* && all(a -> a isa Symbol &&
            a in ctx.array_decls, rhs.args[2:end]) || _sfail(
        "joint response $tag factor $(j.factor) must name an explicit " *
        "`$(j.factor) = sd .* C` over a declared scale vector " *
        "and LKJCholesky factor")
    scales, corr = rhs.args[2], rhs.args[3]
    pnames = Symbol[_lower_location(o, m, IdentityLink, ctx, predictors,
        pred_idx, coefuse; synth = Symbol(o, "_joint_", k), value = true)
        for (k, (o, m)) in enumerate(zip(j.outcomes, j.means))]
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
        haskey(coefuse, nm) && nm ∉ ctx.ordinary_parameters && _sfail(
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
        haskey(coefuse, sigma) && sigma ∉ ctx.ordinary_parameters && _sfail(
            "response $(g.response): `$(g.head)` sigma $sigma is also " *
            "a predictor coefficient — one use per name")
    end
    label = Symbol(g.response, "_resp")
    push!(ctx.matrices_used, g.matrix)
    glmuse[g.beta] = (label, g.matrix)
    ev = g.wrapper === nothing ? ResponseEvidence(:none, nothing, nothing) :
        first(_peel_evidence(g.response, g.wrapper, ctx))
    return LikelihoodSpec(fam, link, g.response, g.matrix, sigma, nothing,
        ev, label, nothing, nothing;
        glm_alpha = g.alpha, glm_beta = g.beta)
end

# A GLM coefficient vector: K per-element PopulationPriors over the
# response matrix columns, addressed by response label. An unstated
# vector fails (`_undeclared_vector` — strict declarations, decision
# 05oe96l). Stated vectors take `b[axes(X, 2)] .~ Normal.(loc,
# scale)` with scalar (shared) or length-K literal-vector
# (per-element) args — Normal-only (non-Normal betas use the
# decomposed predictor form).
function _lower_glm_beta_priors(label::Symbol, beta::Symbol, X::Symbol,
        sample, matrices::Dict{Symbol,DesignMatrix}, hyper_names::Set{Symbol})
    m = get(matrices, X, nothing)
    m === nothing && _sfail("internal: GLM prior over unknown matrix $X")
    cols = Symbol[c for c in m.columns if c !== nothing]
    K = length(cols)
    stated = nothing
    for s in sample
        s.lhs === beta || continue
        stated = s
    end
    stated === nothing && _undeclared_vector(beta, X, label)
    fam, locs, scales, _ =
        _coefficient_matrix_prior(beta, stated.rhs, label, K, hyper_names)
    fam === :normal || _sfail("response $label: GLM-object beta vectors " *
                              "are Normal-only (got `$fam`) — write the " *
                              "decomposed predictor form for other families")
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
    rhs isa Expr && rhs.head === :call && !isempty(rhs.args) &&
        rhs.args[1] === :Categorical && return rhs
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
    f === :Categorical && _sfail("response $lhs: `Categorical.(s)` " *
        "broadcasts over scalar probabilities; share the vector distribution " *
        "with `y .~ Categorical(s)`")
    return Expr(:call, f, _dot2call_spine_args(lhs, f, targs)...)
end

function _dot2call_object_error(lhs, rhs)
    if rhs isa Expr && rhs.head === :call && !isempty(rhs.args) &&
            rhs.args[1] isa Symbol && rhs.args[1] in
            (:Normal, :Bernoulli, :Poisson, :Binomial, :NegativeBinomial2,
                :Gamma, :Beta, :ZeroInflatedPoisson, :ZeroInflatedBinomial,
                :BernoulliLogit,
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
        # Bare param/literal: the mixture bare-mean slots, single-family
        # form (sorted downstream in `_lower_location`).
        (a isa Symbol || a isa Real) && return a
        return _dot2call_nested_link(lhs, a, f)
    elseif f === :Binomial && i == 2
        # Bare param/literal: see above.
        (a isa Symbol || a isa Real) && return a
        return _dot2call_nested_link(lhs, a, f)
    elseif f === :NegativeBinomial2 && i == 1
        return _dot2call_nested_link(lhs, a, f)
    elseif f === :NegativeBinomial && i == 1
        a isa Real && return a
        return _dot2call_nested_link(lhs, a, f)
    elseif f === :HurdlePoisson && i == 1
        return _dot2call_nested_link(lhs, a, f)
    elseif f === :ZeroInflatedPoisson && i == 1
        return _dot2call_nested_link(lhs, a, f)
    elseif f === :InverseGaussian && i == 1
        a isa Real && return a
        return _dot2call_nested_link(lhs, a, f)
    elseif f === :Exponential && i == 1
        return _dot2call_nested_link(lhs, a, f)
    elseif f === :BetaBinomial2 && i == 2
        return _dot2call_nested_link(lhs, a, f)
    elseif f === :Weibull && i == 2
        return _is_composed_map(a) && haskey(_AUXILIARY_LINKS, a.args[1]) ?
            _dot2call_nested_link(lhs, a, f) : a
    end
    # Gamma position 2 (`exp.(eta) ./ alpha`) passes through; the
    # response branch matches the `./` structure (link + alpha identity).
    return a
end

function _dot2call_nested_object(lhs, a)
    # A categorical distribution shares its whole simplex at every
    # wrapper level, just as it does directly under the dotted tilde.
    a isa Expr && !isempty(a.args) && a.args[1] === :Categorical &&
        return _dot2call_response(lhs, a)
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
    (a isa Symbol || a isa Real) && return a
    # Probability arguments are ordinary Julia values. Preserve their
    # broadcast tree unless this is a recognized inverse-link spelling.
    if base in (:Bernoulli, :Binomial) &&
            !(_is_dotted_call(a) && a.args[1] isa Symbol &&
              (haskey(_BERNOULLI_LINKS, a.args[1]) ||
               haskey(_LEGACY_INVERSE_LINKS, a.args[1])))
        return a
    end
    want = (base === :Bernoulli || base === :Binomial) ? "logistic/normcdf/cexpexp" :
        (base === :Beta || base === :BetaBinomial2) ? "logistic" : "exp"
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
    w isa Symbol && (w in ctx.data ||
        _is_bind_data_definition(w, ctx.detmap, ctx.data, Set{Symbol}())) ||
        _sfail("`weighted` weights must be data (a raw column or a " *
            "data-only definition), got $(repr(w))")
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
    b isa Symbol && (b in ctx.data || b in ctx.prior_names ||
        haskey(ctx.detmap, b)) && return b
    if b isa Expr
        # A bound expression reads like the definition it could be named
        # (`lo = -(t .+ 1)`): Julia-valid undotted vector math takes the
        # same canonical dotted form.
        b = _canonical_expr(b, ctx.data, ctx.detmap, ctx.detshape,
            ctx.shape_env, "response $lhs $side bound"; strict = false)
        shape = _shape_of(b, ctx.data, ctx.detmap, copy(ctx.detshape),
            Set{Symbol}(), ctx.shape_env)
        return shape === :scalar ? _composed_scalar_leaf!(lhs, b, ctx, Symbol[]) :
            _extract_column(lhs, b, ctx)
    end
    return _sfail("response $lhs bound $(repr(b)) must be a literal or a " *
                  "declared value")
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
    "`Bernoulli.(logistic.(eta))` (or `normcdf`/`cexpexp` for the inverse link, " *
    "or bare `Bernoulli.(theta)` over a sampled parameter), " *
    "`Poisson.(exp.(eta))` (or bare `Poisson.(lambda)` over a sampled " *
    "parameter), `Binomial.(n, logistic.(mu))` (or `normcdf`/`cexpexp` " *
    "for the link, or bare `Binomial.(n, theta)` over a sampled " *
    "parameter), " *
    "`NegativeBinomial2.(exp.(eta), phi)`, " *
    "`NegativeBinomial.(exp.(eta), p)`, " *
    "`Weibull.(k, exp.(eta))`, " *
    "`HurdlePoisson.(exp.(eta), p_zero)`, " *
    "`ZeroInflatedPoisson.(exp.(eta), zi)`, " *
    "`ZeroInflatedBinomial.(n, p, zi)`, " *
    "`InverseGaussian.(exp.(eta), lambda)`, " *
    "`Exponential.(exp.(eta))`, " *
    "`Gamma.(alpha, exp.(eta) ./ alpha)`, " *
    "`Beta.(logistic.(mu) .* kappa, (1 .- logistic.(mu)) .* kappa)`, " *
    "`BetaBinomial2.(n, logistic.(mu), phi)`, " *
    "`VonMises.(mu, kappa)`, `CircularVonMises.(mu, kappa, lo, hi)`, " *
    "`LogNormal.(mu, sigma)`, " *
    "`CategoricalLogit.(eta_2, ..., eta_K)`, `OrderedLogistic.(eta)`, " *
    "`Ordinal.(Cumulative(), LogitLink(), eta)`, " *
    "`Multinomial.(N, Ref(s))` over `eachrow(hcat(c1, ...))`, " *
    "or `Categorical(s)` " *
    "(or the fused heads `BernoulliLogit.(eta)`, `PoissonLog.(eta)`, " *
    "`BinomialLogit.(n, mu)`, `NegativeBinomial2Log.(eta, phi)`, " *
    "`GammaLog.(alpha, eta)`, `BetaLogit.(mu, kappa)`, which lower " *
    "identically to their decomposed spellings)"

# A location written with no link wrapper in a constrained-scale slot
# (`Bernoulli.(theta)`, `Binomial.(n, theta)`, `Poisson.(lambda)`): the
# value IS the probability or rate. The base returns it marked, so
# `_lower_location` never confuses it with the same name under a link
# (`Poisson.(exp.(a))`, where `a` is the log rate).
struct _BareSlot
    name::Symbol
end

function _lower_response_base(lhs, rhs::Expr, ctx)
    rhs.head === :call || _sfail("response $lhs: $_RESPONSE_BASE_MSG; " *
                                 "got $(repr(rhs))")
    fam = rhs.args[1]
    fam === :weighted &&
        _sfail("`weighted.(...)` goes outermost: " *
               "`y .~ weighted.(Normal.(mu, sigma), w)`")
    fam in (:Normal, :StudentT, :Bernoulli, :Poisson, :Binomial,
        :NegativeBinomial2, :NegativeBinomial, :Gamma, :Beta, :HurdlePoisson,
        :ZeroInflatedPoisson, :ZeroInflatedBinomial, :InverseGaussian, :Exponential,
        :BetaBinomial2, :VonMises, :CircularVonMises, :LogNormal, :Weibull) ||
        return _lower_response_base_error(lhs, rhs, fam)
    args = _distribution_args(fam, _plain_args(rhs, "`$fam`"))
    if fam === :Normal
        length(args) == 2 || _sfail("response $lhs: `Normal` takes " *
                                    "`Normal.(mu, sigma)`")
        return GaussianFam, IdentityLink, IdentityLink, args[1], args[2],
        nothing, nothing, nothing, nothing
    elseif fam === :StudentT
        length(args) == 3 || _sfail("response $lhs: `StudentT` takes " *
                                    "`StudentT.(nu, mu, sigma)`")
        return StudentTFam, IdentityLink, IdentityLink, args[2], args[3],
        nothing, args[1], nothing, nothing
    elseif fam === :Bernoulli
        length(args) == 1 || _sfail("response $lhs: `Bernoulli` takes " *
                                    "`Bernoulli.(logistic.(eta))` (or `normcdf`/`cexpexp` for the inverse link)")
        if args[1] isa Symbol && !haskey(ctx.detmap, args[1])
            # Bare sampled parameter (constrained-scale, no link
            # inversion): the mixture bare-mean triple. A deterministic
            # definition is a predictor, not a parameter — it falls
            # through to link lowering, which throws the link-required
            # error (bare predictors keep their link).
            return BernoulliLogitFam, LogitLink, IdentityLink,
            _BareSlot(args[1]), nothing, nothing, nothing, nothing, nothing
        end
        f, l, loc = _lower_bernoulli_link(lhs, args[1])
        return f, l, IdentityLink, loc, nothing, nothing, nothing, nothing,
        nothing
    elseif fam === :Binomial
        length(args) == 2 || _sfail("response $lhs: `Binomial` takes " *
                                    "`Binomial.(n, logistic.(mu))` (or `normcdf`/`cexpexp` for the inverse link)")
        if args[2] isa Symbol && !haskey(ctx.detmap, args[2])
            # Bare sampled parameter (constrained-scale): the mixture
            # bare-mean triple. A deterministic definition is a
            # predictor, not a parameter — it falls through to link
            # lowering, which throws the link-required error (bare
            # predictors keep their link).
            return BinomialLogitFam, LogitLink, IdentityLink,
            _BareSlot(args[2]), nothing, _lower_trials(lhs, args[1], ctx),
            nothing, nothing, nothing
        end
        f, l, loc = _lower_binomial_link(lhs, args[2])
        return f, l, IdentityLink, loc, nothing,
        _lower_trials(lhs, args[1], ctx), nothing, nothing, nothing
    elseif fam === :NegativeBinomial2
        length(args) == 2 || _sfail("response $lhs: `NegativeBinomial2` takes " *
                                    "`NegativeBinomial2.(exp.(eta), phi)`")
        link, loc = _lower_slot_link(lhs, args[1])
        return NegativeBinomial2Fam, link, link,
        loc, args[2], nothing, nothing,
        nothing, nothing
    elseif fam === :NegativeBinomial
        length(args) == 2 || _sfail("response $lhs: `NegativeBinomial` takes " *
                                    "`NegativeBinomial.(exp.(eta), p)`")
        link, loc = _lower_slot_link(lhs, args[1])
        return NegativeBinomialFam, link, link,
        loc, args[2], nothing, nothing,
        nothing, nothing
    elseif fam === :Weibull
        length(args) == 2 || _sfail("response $lhs: `Weibull` takes " *
                                    "`Weibull.(k, exp.(eta))`")
        value = args[2]
        _reject_legacy_inverse_link(lhs, value)
        if Meta.isexpr(value, :call) && !isempty(value.args) &&
                haskey(_AUXILIARY_LINKS, value.args[1])
            link, loc = _lower_slot_link(lhs, value)
            return WeibullFam, link, link, loc, args[1], nothing, nothing,
            nothing, nothing
        end
        return WeibullValueFam, IdentityLink, IdentityLink,
        value, args[1], nothing, nothing, nothing, nothing
    elseif fam === :Gamma
        length(args) == 2 || _sfail("response $lhs: `Gamma` takes shape and scale")
        a1, div = args
        if Meta.isexpr(div, :call) && length(div.args) == 3 &&
                div.args[1] === Symbol("./") && _same_aux(a1, div.args[3]) &&
                Meta.isexpr(div.args[2], :.) && div.args[2].args[1] === :exp
            link, loc, scale = _lower_gamma_args(lhs, args, ctx)
            return GammaLogFam, link, link, loc, scale, nothing, nothing,
            nothing, nothing
        end
        return GammaValueFam, IdentityLink, IdentityLink, div, a1,
        nothing, nothing, nothing, nothing
    elseif fam === :Beta
        bfam, blink, plink, loc, scale = _lower_beta_args(lhs, args, ctx)
        return bfam, blink, plink, loc, scale, nothing,
        nothing, nothing, nothing
    elseif fam === :BetaBinomial2
        length(args) == 3 || _sfail("response $lhs: `BetaBinomial2` takes " *
                                    "`BetaBinomial2.(n, logistic.(mu), phi)`")
        link, loc = _lower_slot_link(lhs, args[2])
        return BetaBinomial2Fam, link, IdentityLink,
        loc, args[3],
        _lower_trials(lhs, args[1], ctx), nothing, nothing, nothing
    elseif fam === :HurdlePoisson
        length(args) == 2 || _sfail("response $lhs: `HurdlePoisson` takes " *
                                    "`HurdlePoisson.(exp.(eta), p_zero)`")
        link, loc = _lower_slot_link(lhs, args[1])
        return HurdlePoissonFam, link, link,
        loc, args[2], nothing, nothing,
        nothing, nothing
    elseif fam === :ZeroInflatedPoisson
        length(args) == 2 || _sfail("response $lhs: `ZeroInflatedPoisson` takes " *
                                    "`ZeroInflatedPoisson.(exp.(eta), zi)`")
        link, loc = _lower_slot_link(lhs, args[1])
        return ZeroInflatedPoissonFam, link, link,
        loc, nothing, nothing, nothing,
        args[2], nothing
    elseif fam === :ZeroInflatedBinomial
        length(args) == 3 || _sfail("response $lhs: `ZeroInflatedBinomial` takes " *
                                    "`ZeroInflatedBinomial.(n, p, zi)`")
        link, loc = _lower_slot_link(lhs,
            _dot2call_nested_link(lhs, args[2], :Binomial))
        args[2] isa Symbol && loc in ctx.prior_names && (loc = _BareSlot(loc))
        return ZeroInflatedBinomialFam, link, IdentityLink, loc,
        nothing, _lower_trials(lhs, args[1], ctx), nothing, args[3], nothing
    elseif fam === :InverseGaussian
        length(args) == 2 || _sfail("response $lhs: `InverseGaussian` takes " *
                                    "`InverseGaussian.(exp.(eta), lambda)`")
        link, loc = _lower_slot_link(lhs, args[1])
        return InverseGaussianFam, link, link,
        loc, args[2], nothing, nothing,
        nothing, nothing
    elseif fam === :Exponential
        length(args) == 1 || _sfail("response $lhs: `Exponential` takes " *
                                    "`Exponential.(exp.(eta))`")
        link, loc = _lower_slot_link(lhs, args[1])
        return ExponentialLogFam, link, link,
        loc, nothing, nothing, nothing,
        nothing, nothing
    elseif fam === :VonMises
        length(args) == 2 || _sfail("response $lhs: `VonMises` takes " *
                                    "`VonMises.(mu, kappa)`")
        return VonMisesFam, IdentityLink, IdentityLink, args[1], args[2],
        nothing, nothing, nothing, nothing
    elseif fam === :CircularVonMises
        length(args) == 4 || _sfail("response $lhs: `CircularVonMises` takes " *
                                    "`CircularVonMises.(mu, kappa, lo, hi)`")
        return VonMisesFam, IdentityLink, IdentityLink, args[1], args[2],
        nothing, nothing, nothing, (args[3], args[4])
    elseif fam === :LogNormal
        length(args) == 2 || _sfail("response $lhs: `LogNormal` takes " *
                                    "`LogNormal.(mu, sigma)`")
        return LogNormalFam, IdentityLink, IdentityLink, args[1], args[2],
        nothing, nothing, nothing, nothing
    else
        length(args) == 1 || _sfail("response $lhs: `Poisson` takes " *
                                    "`Poisson.(exp.(eta))`")
        if args[1] isa Symbol && !haskey(ctx.detmap, args[1])
            # Bare sampled parameter (constrained-scale): the mixture
            # bare-mean triple. A deterministic definition is a
            # predictor, not a parameter — it falls through to link
            # lowering, which throws the link-required error (bare
            # predictors keep their link).
            return PoissonLogFam, LogLink, LogLink, _BareSlot(args[1]),
            nothing, nothing, nothing, nothing, nothing
        end
        link, loc = _lower_slot_link(lhs, args[1])
        return PoissonLogFam, link, link,
        loc, nothing, nothing, nothing,
        nothing, nothing
    end
end

function _lower_trials(lhs, t, ctx)
    t isa Bool && _sfail("response $lhs trials must be an Int data " *
                         "column or Int literal, got Bool")
    t isa Base.BitInteger && return Int(t)
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
    link, loc = _lower_slot_link(lhs,
        _dot2call_nested_link(lhs, div.args[2], :Gamma))
    a2 = div.args[3]
    _same_aux(a1, a2) || _sfail(
        "response $lhs: both `Gamma` positions must name the same alpha " *
        "(got $(repr(a1)) and $(repr(a2)))")
    return link, loc, a1
end

# Both auxiliary positions name the same use: bare names, equal literals,
# or structurally equal scale-predictor spellings (bare or one
# `exp.`/`logistic.` wrapper over the same predictor). Anything else —
# mixed wrappers, distinct predictors — is a mismatch, never a merge.
_same_aux(a, b) = a == b

const _BETA_MSG = "`Beta.(logistic.(mu) .* kappa, (1 .- logistic.(mu)) .* kappa)`"

# Preserve the mean/concentration design when the authored shapes match
# it exactly; every other Beta(alpha, beta) uses its two ordinary values.
function _lower_beta_args(lhs, args, ctx)
    length(args) == 2 || _sfail("response $lhs: Beta takes (alpha, beta)")
    # Try both mean/concentration orders without changing arbitrary shapes.
    for (a1, a2, swapped) in ((args[1], args[2], false), (args[2], args[1], true))
        if a1 isa Expr && a1.head === :call && length(a1.args) == 3 &&
                a1.args[1] === :.* && a2 isa Expr && a2.head === :call &&
                length(a2.args) == 3 && a2.args[1] === :.*
            c = a2.args[2]
            if c isa Expr && c.head === :call && length(c.args) == 3 &&
                    c.args[1] === :.- && c.args[2] == 1 &&
                    a1.args[2] == c.args[3] && _same_aux(a1.args[3], a2.args[3])
                mu = a1.args[2]
                if mu isa Symbol || mu isa Real ||
                        (_is_dotted_call(mu) && haskey(_AUXILIARY_LINKS, mu.args[1]) &&
                            length(mu.args[2].args) == 1)
                    link, loc = _lower_slot_link(lhs,
                        _dot2call_nested_link(lhs, mu, :Beta))
                    if swapped
                        if link === LogitLink
                            loc = Expr(:call, :.-, loc)
                        else
                            loc = Expr(:call, :.-, 1.0, mu)
                            link = IdentityLink
                        end
                    end
                    return BetaLogitFam, link, IdentityLink, loc, a1.args[3]
                end
            end
        end
    end
    return BetaShapeFam, IdentityLink, IdentityLink, args[1], args[2]
end

function _lower_response_base_error(lhs, rhs, fam)
    fam in (:normal, :bernoulli, :poisson, :binomial, :gamma, :beta) && _sfail(
        "response $lhs: use Distributions.jl constructors " *
        "(`Normal`, not `normal`)")
    fam === :negative_binomial2 && _sfail("response $lhs: use " *
                                          "`NegativeBinomial2` (the response " *
                                          "spelling, not the kernel endpoint)")
    fam === :negative_binomial && _sfail("response $lhs: use " *
                                        "`NegativeBinomial` (the response " *
                                        "spelling, not the kernel endpoint)")
    fam === :weibull && _sfail("response $lhs: use " *
                               "`Weibull` (the response " *
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
    fam === :zero_inflated_binomial && _sfail("response $lhs: use " *
                                 "`ZeroInflatedBinomial` (the response " *
                                 "spelling, not the kernel endpoint)")
    fam === :inverse_gaussian && _sfail("response $lhs: use " *
                                        "`InverseGaussian` (the response " *
                                        "spelling, not the kernel endpoint)")
    fam === :exponential && _sfail("response $lhs: use " *
                                   "`Exponential` (the response " *
                                   "spelling, not the kernel endpoint)")
    fam === :beta_binomial2 && _sfail("response $lhs: use " *
                                      "`BetaBinomial2` (the response " *
                                      "spelling, not the kernel endpoint)")
    fam === :von_mises && _sfail("response $lhs: use " *
                                 "`VonMises` (the response " *
                                 "spelling, not the kernel endpoint)")
    fam === :circular_von_mises && _sfail("response $lhs: use " *
                                          "`CircularVonMises` (the response " *
                                          "spelling, not the kernel endpoint)")
    fam === :lognormal && _sfail("response $lhs: use " *
                                 "`LogNormal` (the response " *
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
                  "NegativeBinomial2, NegativeBinomial, Weibull, " *
                  "HurdlePoisson, ZeroInflatedPoisson, " *
                  "ZeroInflatedBinomial, " *
                  "InverseGaussian, Exponential, BetaBinomial2, VonMises, " *
                  "CircularVonMises, LogNormal, Gamma, Beta, " *
                  "BernoulliLogit, " *
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
# recognized wrapper arrives converted to `:call` by `_dot2call_nested_link`;
# other probability expressions retain their ordinary value tree.
const _BERNOULLI_LINKS = Dict{Symbol,Tuple{LikelihoodFamily,LinkFunction}}(
    :logistic => (BernoulliLogitFam, LogitLink),
    :normcdf => (BernoulliProbitFam, ProbitLink),
    :cexpexp => (BernoulliCloglogFam, CloglogLink),
    :exp => (BernoulliLogitFam, LogLink),
)
const _BINOMIAL_LINKS = Dict{Symbol,Tuple{LikelihoodFamily,LinkFunction}}(
    :logistic => (BinomialLogitFam, LogitLink),
    :normcdf => (BinomialProbitFam, ProbitLink),
    :cexpexp => (BinomialCloglogFam, CloglogLink),
    :exp => (BinomialLogitFam, LogLink),
)

_inverse_link_spelling(link::LinkFunction, value) =
    _inverse_link_spelling(Val(link), value)
_inverse_link_spelling(::Val{IdentityLink}, value) = value
_inverse_link_spelling(::Val{LogLink}, value) = _dotted_obj(:exp, value)
_inverse_link_spelling(::Val{LogitLink}, value) = _dotted_obj(:logistic, value)
_inverse_link_spelling(::Val{ProbitLink}, value) = _dotted_obj(:normcdf, value)
_inverse_link_spelling(::Val{CloglogLink}, value) = _dotted_obj(:cexpexp, value)

function _lower_slot_link(lhs, arg)
    _reject_legacy_inverse_link(lhs, arg)
    if arg isa Expr && arg.head === :call && length(arg.args) == 2 &&
            haskey(_AUXILIARY_LINKS, arg.args[1])
        return _AUXILIARY_LINKS[arg.args[1]], arg.args[2]
    end
    (arg isa Symbol || arg isa Real) && return IdentityLink, arg
    _sfail("response $lhs: parameter takes a Julia value or a dotted " *
        "exp/logistic/normcdf/cexpexp wrapper, got $(repr(arg))")
end

function _lower_bernoulli_link(lhs, arg)
    _reject_legacy_inverse_link(lhs, arg)
    if arg isa Expr && arg.head === :call && !isempty(arg.args) &&
            haskey(_BERNOULLI_LINKS, arg.args[1])
        fam, link = _BERNOULLI_LINKS[arg.args[1]]
        return fam, link, _lower_link_arg(lhs, arg, arg.args[1])
    end
    return BernoulliLogitFam, IdentityLink, arg
end

function _lower_binomial_link(lhs, arg)
    _reject_legacy_inverse_link(lhs, arg)
    if arg isa Expr && arg.head === :call && !isempty(arg.args) &&
            haskey(_BINOMIAL_LINKS, arg.args[1])
        fam, link = _BINOMIAL_LINKS[arg.args[1]]
        return fam, link, _lower_link_arg(lhs, arg, arg.args[1])
    end
    return BinomialLogitFam, IdentityLink, arg
end

const _LEGACY_INVERSE_LINKS = Dict(
    :probit => (:normcdf, "probit names the inverse normal CDF, not the normal CDF"),
    :cloglog => (:cexpexp, "cloglog is the link log(-log(1-p)), not its inverse"),
)

function _reject_legacy_inverse_link(lhs, arg)
    arg isa Expr && arg.head === :call && !isempty(arg.args) || return nothing
    replacement = get(_LEGACY_INVERSE_LINKS, arg.args[1], nothing)
    replacement === nothing && return nothing
    inverse, reason = replacement
    _sfail("response $lhs: $reason; use `$inverse.(eta)`")
end

# Prob-space Binomial-family location: a sampled name or a fixed value
# predictor. A fully fixed likelihood contributes density without coordinates.
function _lower_prob_location(lhs, loc, ctx, predictors, pred_idx,
        what::String = "Binomial")
    loc isa Symbol && loc in ctx.prior_names && return loc
    if loc isa Real
        isfinite(loc) && 0 <= loc <= 1 || _sfail(
            "response $lhs: $what literal probability must lie in [0, 1]")
        return _value_location!(lhs, loc, IdentityLink, ctx, predictors, pred_idx)
    end
    return _sfail("response $lhs: prob-space $what probability takes a " *
                  "Beta parameter name (`theta ~ Beta(...)` in the model), " *
                  "got $(repr(loc))")
end

# Distribution arguments are values. Named affine predictors retain their
# design representation; other expressions share the ordinary assignment
# graph, including scalar calls and per-observation broadcasts.
# Link maps are ordinary functions in a value argument. The generator's
# value-math rewrite resolves logistic to its stable mathematical binding.
const _AUXILIARY_LINKS = Dict(:exp => LogLink, :logistic => LogitLink,
    :normcdf => ProbitLink, :cexpexp => CloglogLink)

function _argument_value_calls(ex)
    ex isa Expr || return ex
    args = Any[_argument_value_calls(a) for a in ex.args]
    if ex.head in (:call, :.) && !isempty(args) && args[1] === :logistic
        args[1] = GlobalRef(PPLGeneratedModels, :_ppl_logistic)
    end
    return Expr(ex.head, args...)
end

function _lower_argument_value(lhs, slot, value, ctx)
    value isa Real && return value
    value === :Inf && return Inf
    value in (:pi, :π) && value ∉ ctx.taken && return Float64(pi)
    if value isa Symbol
        value in ctx.taken || _sfail("response $lhs $slot references " *
            "undeclared name $value")
        haskey(ctx.matrices, value) && _sfail("response $lhs $slot " *
            "$value is a design matrix; use a scalar or one value per row")
        return value
    end
    value isa Expr || _sfail("response $lhs $slot must be a value, " *
        "got $(repr(value))")
    rhs = _resolve_module_calls(value, ctx.mod, ctx.taken,
        "response $lhs $slot")
    _reject_unknown_calls("response $lhs $slot", rhs; composed_maps = true)
    rhs = _argument_value_calls(rhs)
    rhs = _canonical_expr(rhs, ctx.data, ctx.detmap, ctx.detshape,
        ctx.shape_env, "response $lhs $slot")
    for name in _value_symbols(rhs)
        name in ctx.taken || name in (:pi, :Inf) || _sfail(
            "response $lhs $slot references undeclared name $name")
    end
    folded = _fold_literal(rhs)
    folded === nothing || return folded
    shape = _canon_shape(rhs, ctx)
    shape in (:scalar, :vector) || _sfail("response $lhs $slot " *
        "needs a scalar or one value per row, got $shape")
    if shape === :vector
        return _extract_column(lhs, rhs, ctx)
    end
    return get!(ctx.leaf_exprs, rhs) do
        while true
            ctx.synth[] += 1
            name = Symbol(:_rkppl_arg_, ctx.synth[])
            name in ctx.taken && continue
            push!(ctx.taken, name)
            push!(ctx.synth_assigns, AssignmentSpec(name, rhs, name))
            return name
        end
    end
end

function _lower_argument_use(lhs, slot, value, ctx, predictors, pred_idx,
        coefuse)
    value === nothing && return nothing
    if value isa Symbol &&
            _is_scale_predictor_def(value, ctx, false)
        pname = _lower_scale_predictor(lhs, value, IdentityLink, ctx,
            predictors, pred_idx, coefuse)
        return ScalePredictorRef(pname, IdentityLink)
    end
    if _is_dotted_call(value) && haskey(_AUXILIARY_LINKS, value.args[1]) &&
            length(value.args[2].args) == 1
        inner = only(value.args[2].args)
        if inner isa Symbol && _is_scale_predictor_def(inner, ctx, true)
            link = _AUXILIARY_LINKS[value.args[1]]
            pname = _lower_scale_predictor(lhs, inner, link, ctx,
                predictors, pred_idx, coefuse)
            return ScalePredictorRef(pname, link)
        end
    end
    # Bare data in nu/zi uses the same derived-value graph as an inline
    # expression, so its declared shape is present before data binding.
    if slot !== :scale && value isa Symbol && value in ctx.data
        return _extract_column(lhs, Expr(:call, :.*, 1.0, value), ctx)
    end
    return _lower_argument_value(lhs, slot, value, ctx)
end

_lower_nu_use(lhs, s, ctx, predictors, pred_idx, coefuse) =
    _lower_argument_use(lhs, :nu, s, ctx, predictors, pred_idx, coefuse)
_lower_zi_use(lhs, s, ctx, predictors, pred_idx, coefuse) =
    _lower_argument_use(lhs, :zi, s, ctx, predictors, pred_idx, coefuse)
_lower_scale_use(lhs, s, ctx, predictors, pred_idx, coefuse) =
    _lower_argument_use(lhs, :scale, s, ctx, predictors, pred_idx, coefuse)

# Principal-interval endpoints use the same value graph as other
# distribution arguments; their width and support are guarded lazily.
function _lower_interval_use(lhs, raw, ctx)
    raw === nothing && return nothing
    raw isa Tuple && length(raw) == 2 || _sfail("response $lhs: " *
        "CircularVonMises takes (mu, kappa, lo, hi)")
    return map(a -> _lower_argument_value(lhs, :interval, a, ctx), raw)
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
# admits stated-Normal coefficients through the ordinary value path,
# while a bare use
# keeps them scalar — naming a stated name must not re-bucket it.
# A retained value keeps its name for every reader, so a scale reads it.
_is_scale_predictor_def(s::Symbol, ctx, allow_stated::Bool) =
    haskey(ctx.detmap, s) && !(s in ctx.plate_names) && s ∉ ctx.value_defs &&
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
# (`a ~ Normal(0, 5)` over `eta = a .+ b .* x`).
function _is_scalar_coef_def(s::Symbol, ctx, allow_stated::Bool)
    haskey(ctx.detmap, s) || return false
    s in ctx.plate_names && return false
    _derived_reads_latent(s, ctx) && return false
    get(ctx.detshape, s, :scalar) === :scalar || return false
    rhs = ctx.detmap[s]
    _contains_scan(rhs, ctx.scan_states) && return false
    rhs isa Symbol || return false
    rhs in ctx.data && return false
    rhs in ctx.vecdefs && return false
    haskey(ctx.detmap, rhs) && return false
    rhs in ctx.prior_names && (rhs ∉ ctx.coef_priors || !allow_stated) &&
        return false
    rhs in ctx.plate_names && return false
    rhs in ctx.scan_states && return false
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

# A factor-coefficient alias has a per-observation value shape, but its
# coefficient role survives naming (including chains of bare aliases).
# Other array gathers remain value computations.
_is_factor_coefficient_alias(ex, ctx) =
    _is_factor_coefficient_alias(ex, ctx, Set{Symbol}())
function _is_factor_coefficient_alias(s::Symbol, ctx, seen::Set{Symbol})
    haskey(ctx.detmap, s) && s ∉ seen || return false
    push!(seen, s)
    return _is_factor_coefficient_alias(ctx.detmap[s], ctx, seen)
end
_is_factor_coefficient_alias(ex::Expr, ctx, seen::Set{Symbol}) =
    _is_factor_index_def(ex, ctx) && ex.args[1] in ctx.factor_decls
_is_factor_coefficient_alias(ex, ctx, seen::Set{Symbol}) = false

# Intern the raw predictor once; each auxiliary reference carries its link.
function _lower_scale_predictor(lhs, name::Symbol, link, ctx, predictors,
        pred_idx, coefuse)
    haskey(pred_idx, name) || haskey(ctx.detmap, name) ||
        return _lower_scale_predictor_error(lhs, name, ctx)
    if haskey(pred_idx, name)
        return name
    end
    if _composed_root(ctx.detmap[name], ctx)
        return _lower_composed_predictor(name, ctx.detmap[name], ctx, lhs,
            link, predictors, pred_idx, coefuse)
    end
    terms, uses = _analyze_predictor(name, ctx.detmap[name], ctx, lhs)
    _record_coefuses!(coefuse, name, uses, lhs)
    push!(predictors, PredictorSpec(name, link, terms, name))
    pred_idx[name] = length(predictors)
    return name
end

# A name the lowering synthesizes for a graph value: `base`, or `base_k`
# for the first `k` no other name of the model takes. Synthesized predictors
# are graph values under their own names, like authored ones, so they never
# shadow a user's name (user decision `1c0jiwz`).
function _predictor_name!(base::Symbol, ctx)
    nm, k = base, 0
    while nm in ctx.taken
        k += 1
        nm = Symbol(base, "_", k)
    end
    push!(ctx.taken, nm)
    return nm
end

# Distribution arguments are ordinary values. A scalar keeps its scalar
# assignment; dotted expressions retain their array computation over LPs,
# data and scalar declarations without inventing a new coefficient prior.
function _lower_argument_predictor!(lhs, ex, ctx, predictors, pred_idx,
        coefuse, base::Symbol)
    if ex isa Expr && (_canon_shape(ex, ctx) !== :scalar || _is_composed_map(ex))
        pname = _predictor_name!(base, ctx)
        _reads_latent_elementwise(ex, ctx) &&
            return _latent_value_predictor!(lhs, pname, ex, :value,
                IdentityLink, ctx, predictors, pred_idx)
        return _lower_composed_predictor(pname, ex, ctx, lhs, IdentityLink,
            predictors, pred_idx, coefuse)
    end
    return _lower_location(lhs, ex, IdentityLink, ctx, predictors, pred_idx,
        coefuse; value = true, synth = base)
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

# A location written bare in a constrained-scale slot lowers like any
# name, except that a sampled parameter there IS the probability or rate.
_lower_location(lhs, loc::_BareSlot, pred_link, ctx, predictors, pred_idx,
        coefuse; kwargs...) =
    _lower_location(lhs, loc.name, pred_link, ctx, predictors, pred_idx,
        coefuse; kwargs..., bare = true)

function _lower_location(lhs, loc, pred_link, ctx, predictors, pred_idx,
        coefuse; synth::Union{Nothing,Symbol} = nothing,
        bare::Bool = false, value::Bool = false)
    # A per-cell latent VECTOR is the whole location via a LatentTerm predictor
    # (`lp = theta`, identity design; the latent's prior lives on its
    # PlateParameter, so no coefficient use is recorded). Two spellings: a bare
    # latent (`y[i] ~ Normal.(theta[i], s)`) or a deterministic transform of one
    # (`theta[i] = mu .+ tau .* z[i]` then `y[i] ~ Normal.(theta[i], s)` — the
    # non-centered / latent-transform shape, emitted as a derived column and
    # referenced directly). A derived location with NO latent stays a design
    # predictor; a latent-reading definition WITH coefficient structure is a
    # design predictor too (`b .* theta` classifies as a ContinuousTerm over
    # the latent), while an unscaled latent summand contributes its value
    # through the same LatentTerm used for a direct location.
    if loc isa Symbol && loc in ctx.plate_names
        return _latent_predictor!(lhs, loc, pred_link, ctx, predictors, pred_idx)
    end
    if loc isa Symbol && haskey(ctx.detmap, loc) && _derived_reads_latent(loc, ctx) &&
            !_is_design_shaped(loc, ctx) && !_scalar_location_value(loc, ctx)
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
        # A pure alias of a scan state
        # (`w = u`, a latent submodel's `x = x_level` binding) is that state:
        # naming never changes legality.
        loc in ctx.scan_states && return loc
        target = _scan_alias_target(loc, ctx)
        if target !== nothing
            # The alias chain vanishes like any absorbed location.
            nm = loc
            while nm !== target
                push!(ctx.absorbed, nm)
                nm = ctx.detmap[nm]
            end
            return target
        end
        if !haskey(ctx.detmap, loc)
            # Written bare in a Bernoulli/Binomial/Poisson slot, a sampled
            # parameter is the constrained-scale probability or rate (no
            # link inversion): the mixture bare-mean slots, single-family
            # form. detmap-first preserves stated-prior aliases.
            bare && loc in ctx.prior_names && return loc
            # Anywhere else a response reads a scalar parameter or a data
            # column as a value under the written link.
            value && !bare && _is_value_location(loc, ctx) &&
                return _value_location!(lhs, loc, pred_link, ctx,
                    predictors, pred_idx; synth)
            return _lower_location_symbol_error(lhs, loc, ctx; bare,
                value)
        end
        # Retained definitions and ordinary matrix values keep their names
        # for every reader; a predictor cannot shadow the authored value.
        value && !bare &&
            (loc in ctx.value_defs || _canon_shape(loc, ctx) === :matrix) &&
            return _value_location!(lhs, loc, pred_link, ctx,
                predictors, pred_idx; synth)
        # Scalar definitions stay ordinary values, except for the already
        # admitted signed coefficient aliases (their intercept plan stays).
        value && !bare && _scalar_location_value(loc, ctx) &&
            !_scalar_intercept_location(loc, ctx) &&
            return _value_location!(lhs, loc, pred_link, ctx,
                predictors, pred_idx; synth)
        pname = loc
        haskey(pred_idx, pname) && return pname
        if _composed_root(ctx.detmap[loc], ctx) ||
                (value && _is_bare_sub_map(ctx.detmap[loc], ctx))
            # A composition reading a latent elementwise is a latent
            # transform (`w = exp.(eta) .* abs.(th)`), read as one column.
            _reads_latent_elementwise(ctx.detmap[loc], ctx) &&
                return _latent_predictor!(lhs, loc, pred_link, ctx,
                    predictors, pred_idx)
            return _lower_composed_predictor(pname, ctx.detmap[loc], ctx,
                lhs, pred_link, predictors, pred_idx, coefuse)
        end
        terms, uses = _analyze_predictor(pname, ctx.detmap[loc], ctx, lhs)
    elseif loc isa Number
        value && !bare && return _value_location!(lhs, loc, pred_link, ctx,
            predictors, pred_idx; synth)
        _sfail("response $lhs location is a literal — use an intercept-only " *
               "predictor (`eta = a`)")
    else
        value && !bare && _scalar_location_value(loc, ctx) &&
            !_scalar_intercept_location(loc, ctx) &&
            return _value_location!(lhs, loc, pred_link, ctx,
                predictors, pred_idx; synth)
        # Per-cell latents classify inline like data columns: `b .* x_true`
        # is a ContinuousTerm over the latent; an unscaled summand is a
        # LatentTerm, just as when the whole location names that latent.
        # Multi-eta responses (CategoricalLogit) index their synthetic
        # predictors; the single-eta default keeps its established name.
        pname = _predictor_name!(something(synth, Symbol(lhs, "_eta")), ctx)
        if _composed_root(loc, ctx) || (value && _is_bare_sub_map(loc, ctx))
            _reads_latent_elementwise(loc, ctx) &&
                return _latent_value_predictor!(lhs, pname, loc, :location,
                    pred_link, ctx, predictors, pred_idx)
            return _lower_composed_predictor(pname, loc, ctx, lhs,
                pred_link, predictors, pred_idx, coefuse)
        end
        terms, uses = _analyze_predictor(pname, loc, ctx, lhs)
    end
    _record_coefuses!(coefuse, pname, uses, lhs)
    push!(predictors, PredictorSpec(pname, pred_link, terms, pname))
    pred_idx[pname] = length(predictors)
    return pname
end

# Whether `ex` reads a per-cell latent elementwise: a plate latent, or a
# per-observation definition that reads one. Scalar subtrees read a latent
# only whole (`sum(th)`, `th[1]`) and stay composition scalar leaves, as do
# the leaves `_extract_composed_tree` already reads as columns (retained
# values, array-cell plate columns and declared-array gathers).
function _reads_latent_elementwise(ex, ctx)
    if ex isa Symbol
        ex in ctx.plate_names && return true
        ex in ctx.value_defs && return false
        _is_plate_column_call(get(ctx.detmap, ex, nothing)) && return false
        return haskey(ctx.detmap, ex) &&
            get(ctx.detshape, ex, :scalar) !== :scalar &&
            _derived_reads_latent(ex, ctx)
    end
    ex isa Expr || return false
    _canon_shape(ex, ctx) === :scalar && return false
    ex.head === :ref && _reads_array_value(ex, ctx; follow = false) &&
        return false
    args = ex.head === :call ? ex.args[2:end] :
        _is_dotted_call(ex) ? ex.args[2].args : ex.args
    return any(a -> _reads_latent_elementwise(a, ctx), args)
end

# An inline value reading a per-cell latent lowers as its named twin
# (`w = exp.(eta) .* th`, then `Normal.(w, s)`): one derived column holding
# the whole value, read by a LatentTerm. Composition leaves are
# sub-predictors, scalars and data columns, never a latent.
function _latent_value_predictor!(lhs, pname, ex, slot, pred_link, ctx,
        predictors, pred_idx)
    col = _lower_argument_value(lhs, slot, ex, ctx)
    return _latent_predictor!(lhs, col, pred_link, ctx, predictors, pred_idx;
        pname)
end

# Build the LatentTerm location predictor over `col` (a plate parameter or a
# derived column that reads one); the generator emits `lp = col`.
# Latent predictors stay per-response (no interning here).
function _latent_predictor!(lhs, col, pred_link, ctx, predictors, pred_idx;
        pname::Symbol = _predictor_name!(Symbol(lhs, "_loc"), ctx))
    # A nested per-cell call can return a sampled local unchanged. The
    # outer binding aliases that vector; it does not create another latent.
    source = col
    seen = Set{Symbol}()
    while source isa Symbol && haskey(ctx.detmap, source) &&
            ctx.detmap[source] isa Symbol && source ∉ seen
        push!(seen, source)
        source = ctx.detmap[source]
    end
    if source in ctx.plate_names
        union!(ctx.absorbed, seen)
        col = source
    end
    term = TermSpec(LatentTerm, [col], NamedTuple(), col, Symbol(col, "_lat"))
    push!(predictors, PredictorSpec(pname, pred_link, [term], pname))
    pred_idx[pname] = length(predictors)
    return pname
end

# A name a response may read as its location value: a data column, or a
# sampled scalar parameter of any prior (arrays, varying blocks, latents
# and scan states carry their own location arms).
_is_value_location(loc::Symbol, ctx) = loc in ctx.data ||
    (loc in ctx.prior_names && loc ∉ ctx.sized_decls &&
        loc ∉ ctx.vector_params)

# Scalar definitions/expressions, plus scalar sums the affine path refuses
# (two intercepts or a literal summand). Other vector-shaped expressions
# and declared array reads keep their established affine/composed plans.
_scalar_location_value(loc, ctx) =
    !_reads_array_value(loc, ctx) && _scalar_location_reads(loc, ctx) &&
    ((_canon_shape(loc, ctx) === :scalar &&
    !_obs_axis(loc, ctx.data, ctx.detmap, ctx.detshape, Set{Symbol}(),
        ctx.shape_env)) || _scalar_sum_location(loc, ctx))

# Require declared values: an implicit coefficient, factor reference or
# basis/trajectory constructor belongs to the established predictor path.
function _scalar_location_reads(loc, ctx,
        seen::Set{Symbol} = Set{Symbol}(); allow_data::Bool = false)
    if loc isa Symbol
        haskey(ctx.detmap, loc) || return (allow_data &&
            (loc in ctx.data || loc in ctx.plate_names)) ||
            (loc in ctx.prior_names && loc ∉ ctx.sized_decls &&
                loc ∉ ctx.vector_params)
        loc in seen && return false
        push!(seen, loc)
        value = _scalar_location_reads(ctx.detmap[loc], ctx, seen; allow_data)
        delete!(seen, loc)
        return value
    end
    loc isa Expr || return true
    loc.head === :kw && return _scalar_location_reads(loc.args[2], ctx,
        seen; allow_data)
    _is_bound_value_call(loc) && return true
    loc.head === :ref && return false
    loc.head === :call && !isempty(loc.args) &&
        loc.args[1] in _CONSTRUCT_VALUE_HEADS && return false
    args = loc.head === :call ? loc.args[2:end] :
        _is_dotted_call(loc) ? loc.args[2].args : loc.args
    # Reductions and module functions take whole column arguments. A
    # function's actual scalar/vector result is checked when evaluated.
    whole = allow_data || (loc.head === :call && !isempty(loc.args) &&
        (loc.args[1] isa GlobalRef || loc.args[1] in REDUCTION_FNS))
    return all(a -> _scalar_location_reads(a, ctx, seen; allow_data = whole), args)
end

function _scalar_sum_location(loc::Symbol, ctx,
        seen::Set{Symbol} = Set{Symbol}())
    haskey(ctx.detmap, loc) && loc ∉ seen || return false
    push!(seen, loc)
    return _scalar_sum_location(ctx.detmap[loc], ctx, seen)
end
function _scalar_sum_location(loc::Expr, ctx,
        seen::Set{Symbol} = Set{Symbol}())
    return loc.head === :call && length(loc.args) >= 3 &&
        loc.args[1] in (:+, :-, :.+, :.-) &&
        _canon_shape(loc, ctx) in (:scalar, :vector) &&
        !_obs_axis(loc, ctx.data, ctx.detmap, ctx.detshape, Set{Symbol}(),
            ctx.shape_env)
end
_scalar_sum_location(loc, ctx, seen::Set{Symbol} = Set{Symbol}()) = false

# Exactly the scalar shapes affine analysis already admits: one signed
# coefficient (stated or implicit), behind aliases or unary +/-. Keep its
# prior and coordinates when admitting the other scalar value shapes.
function _scalar_intercept_location(loc::Symbol, ctx,
        seen::Set{Symbol} = Set{Symbol}())
    haskey(ctx.detmap, loc) || return _summand_kind(loc, ctx) === :coef
    loc in seen && return false
    push!(seen, loc)
    return _scalar_intercept_location(ctx.detmap[loc], ctx, seen)
end
function _scalar_intercept_location(loc::Expr, ctx,
        seen::Set{Symbol} = Set{Symbol}())
    return loc.head === :call && length(loc.args) == 2 &&
        loc.args[1] in (:+, :-, :.+, :.-) &&
        _scalar_intercept_location(loc.args[2], ctx, seen)
end
_scalar_intercept_location(loc, ctx, seen::Set{Symbol} = Set{Symbol}()) = false

# A value location (`y .~ Normal.(mu, s)`, `Poisson.(exp.(a))`,
# `Normal.(x, s)`): as in Julia, broadcasting gives every observation the
# value, read on the written link's scale. A data column is an offset (the
# term its named twin `mu = x` lowers to); a scalar parameter is a one-leaf
# composition with no coefficient — it keeps its own name and prior — which
# the generator broadcasts over the rows.
function _value_location!(lhs, loc, pred_link, ctx, predictors,
        pred_idx; synth::Union{Nothing,Symbol} = nothing)
    # A previously refused scalar sum, including aliases of it, emits as
    # ordinary scalar definitions, rather than observation columns.
    source = loc
    seen = Set{Symbol}()
    while loc ∉ ctx.value_defs && source isa Symbol &&
            haskey(ctx.detmap, source) && source ∉ seen
        push!(seen, source)
        ctx.detshape[source] = :scalar
        source = ctx.detmap[source]
    end
    pname = _predictor_name!(something(synth, Symbol(lhs, "_eta")), ctx)
    term = if loc isa Symbol && loc in ctx.data
        TermSpec(OffsetTerm, ColumnRef[loc], NamedTuple(), loc,
            Symbol(loc, "_off"))
    else
        label = Symbol(pname, "_value")
        scalars = Symbol[]
        columns = ColumnRef[]
        tree = if loc isa Symbol
            push!(loc in ctx.value_defs ? columns : scalars, loc)
            loc
        else
            _composed_scalar_leaf!(pname, loc, ctx, scalars)
        end
        TermSpec(ComposedTerm, columns,
            (tree = tree, subs = Symbol[], scalars = scalars), label,
            label)
    end
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
                _canon_shape(f, ctx) === :vector ||
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

function _lower_location_symbol_error(lhs, loc, ctx; bare::Bool = false,
        value::Bool = false)
    bare && loc in ctx.data && _sfail("response $lhs location is the data " *
        "column $loc written bare — a bare Bernoulli/Binomial/Poisson " *
        "location is a sampled probability or rate; read data under its " *
        "link (`Poisson.(exp.($loc))`, `Bernoulli.(logistic.($loc))`)")
    !value && _is_value_location(loc, ctx) && _sfail("response $lhs " *
        "location $loc: this response's locations are predictor " *
        "definitions (`eta = a .+ b .* x`); a scalar parameter or data " *
        "column as the whole location is admitted for single-family " *
        "responses (`y .~ Normal.($loc, s)`)")
    loc in ctx.prior_names && _sfail("response $lhs location $loc is a " *
        "declared array, not one value per observation — read it by " *
        "index (`$loc[g]`) or in a definition")
    return _sfail("response $lhs location $loc is not a predictor " *
                  "definition (`$loc = ...` affine in data)")
end

function _record_coefuses!(coefuse, pname, uses, lhs)
    for (name, addr, sign) in uses
        entries = get!(coefuse, name, Tuple{Symbol,Symbol,Int}[])
        push!(entries, (pname, addr, sign))
    end
    return nothing
end

# Multiple reads are legal for ordinary parameters. Legacy coefficient
# packs require unique ownership, checked here after predictor analysis.
function _check_coefficient_uses(coefuse, ctx)
    for (name, uses) in coefuse
        length(uses) > 1 || continue
        preds = unique!(map(first, copy(uses)))
        msg = length(preds) > 1 ?
            "coefficient $name is shared across predictors " *
            "$(join(preds, ", ")) — coefficient blocks are per-predictor, " *
            "rename or duplicate it" :
            "coefficient $name is used twice in predictor $(only(preds)) " *
            "($(join(unique!(map(u -> u[2], copy(uses))), ", "))) — one " *
            "coefficient per column"
        _check_owned_coefficient(name, ctx, msg)
    end
    return nothing
end

# Other readers do not alter ordinary parameter semantics. Reject readers
# of legacy construct-owned coefficient packs that cannot represent them.
function _check_coefficient_readers(coefuse, ctx, predictors, priors,
        params, plate_parameters, assigns, derived, responses)
    isempty(coefuse) && return nothing
    function read!(who, s)
        s isa Symbol && haskey(coefuse, s) || return nothing
        return _check_owned_coefficient(s, ctx, "$s is a predictor coefficient and " *
                               "cannot also be read by $who")
    end
    for p in predictors, t in p.terms
        t.kind === ComposedTerm || continue
        foreach(v -> read!("composed predictor $(p.name)", v),
            t.options.scalars)
    end
    for pr in priors
        who = "the prior of ($(pr.predictor), $(pr.addressee))"
        read!(who, pr.location)
        read!(who, pr.scale)
    end
    for p in params, v in values(p.args)
        read!("parameter $(p.name)", v)
    end
    for p in plate_parameters, v in values(p.args)
        read!("latent $(p.name)", v)
    end
    for a in assigns, s in _value_symbols(a.expr)
        read!("assignment $(a.name)", s)
    end
    for d in derived, s in _value_symbols(d.expr)
        read!("derived column $(d.label)", s)
    end
    for r in responses
        who = "response $(r.response)"
        for v in (r.scale, r.nu, r.zi, r.mixture_weights)
            read!(who, v)
        end
        foreach(v -> read!(who, v), r.mixture_locs)
        foreach(v -> read!(who, v), r.mixture_scales)
    end
    return nothing
end

# Predictor analysis: inline deterministic structure (scalars always;
# vectors only when structural — coefficient-holding), canonicalize the
# expanded location, split the affine sum, classify each summand. Returns
# (terms, uses) with uses :: Vector{(coef name, addressee, sign)}.
# Anonymous non-affine vector substructure auto-extracts to synthetic
# derived locals, so naming a subexpression never changes legality.
"""Names a definition may not hide behind in a composition (data-only
leaves keep today's affine/synth paths; anything else routes composed)."""
function _composed_data_only(s::Symbol, ctx, seen::Set{Symbol})
    s in seen && return true
    haskey(ctx.detmap, s) || return s in ctx.data
    push!(seen, s)
    for leaf in _value_symbols(ctx.detmap[s])
        leaf === s && continue
        if leaf in ctx.data
            continue
        elseif haskey(ctx.detmap, leaf) && _composed_data_only(leaf, ctx, seen)
            continue
        else
            return false
        end
    end
    return true
end

"""A sub-predictor candidate: vector-shaped definition that is not data,
a latent/scan/varying object, or a pure data combination (those keep
today's paths — data combos merge affinely, latents keep their arms).
Under `.*` (`allow_factor`), a factor-coefficient alias (`th = c[g]`)
qualifies too — vector-shaped as a value but an affine FactorTerm under
analysis. Under `.+` it never qualifies, so `mu = a .+ th` keeps the
affine merge (its coefficients declared like any other — strict
declarations)."""
function _is_composed_sub(s::Symbol, ctx, allow_factor::Bool = false)
    haskey(ctx.detmap, s) || return false
    s in ctx.value_defs && return false
    factor_alias = _is_factor_coefficient_alias(s, ctx)
    factor_alias && !allow_factor && return false
    if get(ctx.detshape, s, :scalar) !== :vector
        # A bare alias of a varying contribution (`th = r_t`, brms
        # `theta ~ 0 + (1 | person)`) is per-observation, hence a sub.
        (allow_factor && _is_factor_index_def(ctx.detmap[s], ctx)) ||
            return false
    end
    s in ctx.data && return false
    s in ctx.plate_names && return false
    s in ctx.scan_states && return false
    _derived_reads_latent(s, ctx) && return false
    _composed_data_only(s, ctx, Set{Symbol}()) && return false
    # An interned location already has an LP node, including an offset
    # whose definition reads array values. Consumers must use that node
    # rather than reclassifying and emitting the definition a second time.
    haskey(ctx.pred_idx, s) && return true
    # A definition reading a declared array is a value (an in-graph
    # column), not an affine sub-predictor — except the bare factor alias
    # `th = c[g]` over a coefficient-capable `c[levels(g)]`, which keeps
    # its composed factor-sub meaning.
    rhs = ctx.detmap[s]
    _reads_declared_array(rhs, ctx) && !factor_alias && return false
    # Likewise a parameter offset: data and non-coefficient scalar
    # parameters combined by sums only (`w = s .+ x`, `s ~ Exponential(1)`).
    # It has no coefficient to compose, so it stays an offset local. Under
    # strict declarations the intercept beside it (`mu = a .+ w`) is a
    # declared scalar, which must not turn the sum into a composition.
    _composed_param_value(s, ctx) && return false
    return true
end

# A plain scalar parameter: sampled, not coefficient-priored, and none of
# the array / varying / latent / scan objects that carry their own arms.
_composed_scalar_param(leaf::Symbol, ctx) =
    leaf in ctx.prior_names && leaf ∉ ctx.coef_priors &&
    leaf ∉ ctx.sized_decls && leaf ∉ ctx.plate_names &&
    leaf ∉ ctx.scan_states

"""Whether a definition is a parameter offset: data (and data-only parts
of any shape) plus plain scalar parameters (`_composed_scalar_param`),
combined by sums and differences only, reading at least one parameter
(`w = s .+ x`). A product or call over a parameter (`x .* b`,
`f.(x, c)`) is not one: the composed path owns those."""
function _composed_param_value(s::Symbol, ctx)
    ok, reads = _param_offset_expr(ctx.detmap[s], ctx, Set{Symbol}([s]))
    return ok && reads
end

const _PARAM_OFFSET_OPS = (:+, :-, :.+, :.-)

# `(admissible, reads_param)` for one node of a parameter offset.
function _param_offset_expr(ex, ctx, seen::Set{Symbol})
    if ex isa Symbol
        ex in ctx.data && return (true, false)
        if haskey(ctx.detmap, ex)
            _composed_data_only(ex, ctx, Set{Symbol}()) && return (true, false)
            ex in seen && return (false, false)
            push!(seen, ex)
            return _param_offset_expr(ctx.detmap[ex], ctx, seen)
        end
        return _composed_scalar_param(ex, ctx) ? (true, true) : (false, false)
    end
    ex isa Expr || return (true, false)
    # A subexpression that reads no parameter is one data part.
    all(l -> l in ctx.data || (haskey(ctx.detmap, l) &&
        _composed_data_only(l, ctx, Set{Symbol}())), _value_symbols(ex)) &&
        return (true, false)
    ex.head === :call && !isempty(ex.args) &&
        ex.args[1] in _PARAM_OFFSET_OPS || return (false, false)
    reads = false
    for a in ex.args[2:end]
        ok, r = _param_offset_expr(a, ctx, seen)
        ok || return (false, false)
        reads |= r
    end
    return (true, reads)
end

# An array-valued DEFINITION (`b = z * (sd .* L)'`). `detshape` also
# seeds every sized declaration as `:array` (its whole-value shape),
# including coefficient-capable `c[levels(g)]`, whose role the
# declaration sets (`array_decls`, `value_arrays`); so test `detmap` too.
_is_array_def(name, ctx) = name isa Symbol && haskey(ctx.detmap, name) &&
    get(ctx.detshape, name, :scalar) === :array

# A model-level result has its own Julia shape, including an opaque call
# that collects plate outputs. Keep its name under indexing rather than
# inlining its producer into the predictor's factor-coefficient grammar.
_is_model_value_def(name, ctx) = name isa Symbol &&
    haskey(ctx.detmap, name) &&
    _model_valued(name, ctx.detmap,
        ctx.shape_env, Set{Symbol}())

# Whether `ex` reads an array that is a VALUE in every role — a value
# array (`z` declared sized, read whole, or an LKJ factor), an
# array-role declaration, or an array-valued definition — directly or
# through definitions. Unlike `_reads_array_value`, an indexed read of a
# coefficient-capable `c[levels(g)]` (`r = c[g]`, an aliased factor
# coefficient) does not count.
function _reads_value_array(ex, ctx, seen::Set{Symbol} = Set{Symbol}())
    _is_bound_array_value_call(ex) && return true
    isval(nm) = nm isa Symbol && (nm in ctx.value_arrays ||
        nm in ctx.dirichlet_names || nm in ctx.ordered_names ||
        nm in ctx.array_decls || _is_array_def(nm, ctx))
    if ex isa Symbol
        isval(ex) && return true
        haskey(ctx.detmap, ex) && ex ∉ seen || return false
        push!(seen, ex)
        return _reads_value_array(ctx.detmap[ex], ctx, seen)
    end
    ex isa Expr || return false
    ex.head === :ref && isval(ex.args[1]) && return true
    return any(a -> _reads_value_array(a, ctx, seen), ex.args)
end

# Whether a definition reads a declared array, gathered (`a0 .+ z[g]`)
# or whole (`sd .* z`). Such a definition is one named value: never an
# affine sub-predictor of a composition (`_is_composed_sub`), always one
# of its value leaves (`_extract_composed_tree`).
_reads_declared_array(ex, ctx) =
    _reads_array_value(ex, ctx) || _reads_value_array(ex, ctx)

# Whether `ex` reads a declared array value — a bare array name
# (`B * w`), or an indexed read of any array-capable declaration
# (`phi[1]`, `z[g]`, `L[2, 1]`) — directly, or (`follow`) through
# definitions.
function _reads_array_value(ex, ctx, seen::Set{Symbol} = Set{Symbol}();
        follow::Bool = true)
    _is_bound_array_value_call(ex) && return true
    if ex isa Symbol
        (ex in ctx.value_defs || ex in ctx.array_decls || ex in ctx.dirichlet_names ||
            ex in ctx.ordered_names || _is_array_def(ex, ctx)) && return true
        follow && haskey(ctx.detmap, ex) && ex ∉ seen || return false
        push!(seen, ex)
        return _reads_array_value(ctx.detmap[ex], ctx, seen)
    end
    ex isa Expr || return false
    ex.head === :ref && ex.args[1] isa Symbol &&
        (ex.args[1] in ctx.sized_decls || _is_array_def(ex.args[1], ctx)) &&
        return true
    return any(a -> _reads_array_value(a, ctx, seen; follow), ex.args)
end

"""A scalar leaf: sampled name (any prior — coefficient collisions fail
at the surface, where the sub interning is complete) or scalar-shaped
definition."""
function _is_composed_scalar(s::Symbol, ctx)
    s in ctx.prior_names && return true
    haskey(ctx.detmap, s) || return false
    return get(ctx.detshape, s, :scalar) === :scalar
end

function _composed_has_sub(node, ctx, allow_factor::Bool = false)
    node isa Symbol && return _is_composed_sub(node, ctx, allow_factor)
    node isa Expr || return false
    node.head === :call || return any(
        a -> _composed_has_sub(a, ctx, allow_factor), node.args)
    isempty(node.args) && return false
    return any(a -> _composed_has_sub(a, ctx, allow_factor), node.args[2:end])
end

# A sub-predictor whose value exists only as an LP node: one already
# interned as a predictor (a response location or scale, absorbed —
# never also a named local). A definition that is not a predictor, even
# one holding coefficient candidates, keeps the derived-column path,
# which inlines it. A name bound to a composition reads what its tree
# reads, since extraction inlines it (`prop = mu .* s2;
# sd = hypot.(s1, prop)`).
function _composed_has_lp_sub(node, ctx)
    if node isa Symbol
        haskey(ctx.pred_idx, node) && return _is_composed_sub(node, ctx, true)
        return haskey(ctx.detmap, node) && ctx.detmap[node] !== node &&
            _composed_trigger(node, ctx) &&
            _composed_has_lp_sub(ctx.detmap[node], ctx)
    end
    node isa Expr || return false
    return any(a -> _composed_has_lp_sub(a, ctx), node.args)
end

function _composed_has_scalar_leaf(node, ctx)
    node isa Symbol && return _is_composed_scalar(node, ctx)
    node isa Expr || return false
    node.head === :call || return false
    isempty(node.args) && return false
    op = node.args[1]
    op isa Symbol || return false
    # Only the additive spine counts (a scalar under `.*` is the
    # product's own factor, not an additive operand).
    op !== :.+ && op !== :.- && return false
    return any(a -> _composed_has_scalar_leaf(a, ctx), node.args[2:end])
end

function _composed_count_subs(node, ctx)
    node isa Symbol && return _is_composed_sub(node, ctx) ? 1 : 0
    node isa Expr || return 0
    # A dotted unary map over a sub (`exp.(la)`) is one sub operand.
    _is_composed_map(node) &&
        return _composed_has_sub(node, ctx, true) ? 1 : 0
    node.head === :call || return 0
    isempty(node.args) && return 0
    op = node.args[1]
    op isa Symbol || return 0
    # Recurse through the additive spine only — a sub under `.*`
    # triggers the product rule, not the operand count.
    op !== :.+ && op !== :.- && return _composed_has_sub(node, ctx) ? 1 : 0
    return sum(a -> _composed_count_subs(a, ctx), node.args[2:end])
end

"""Whether a raw predictor body routes to composed analysis: a `.*`
(or Julia-valid scalar `*`) over a sub-predictor, or a `.+`/`.−`
combining two sub-predictors (or one plus a scalar). Under `.*` a bare
factor-index definition (`th = c[g]`) counts as a sub-predictor; under
`.+` it keeps the affine merge. Affine merges (`th .+ x`), aliases, and
data-only combinations keep today's paths."""
_is_composed_map(node) = node isa Expr && node.head === :. &&
    length(node.args) == 2 && Meta.isexpr(node.args[2], :tuple)

function _composed_trigger(rhs, ctx)
    # A name bound to a composition is one (naming a subexpression never
    # changes legality): `d = be .* th; eta = d .- s1` composes like the
    # inline `eta = be .* th .- s1`.
    rhs isa Symbol && return haskey(ctx.detmap, rhs) &&
        ctx.detmap[rhs] !== rhs && _composed_trigger(ctx.detmap[rhs], ctx)
    rhs isa Expr || return false
    # An admitted elementwise map over a subtree reaching a sub-predictor
    # (`resp = logistic.((log_dose .- dl) .* exp.(dls))`) is a composition.
    _is_composed_map(rhs) && rhs.args[1] in _COMPOSED_UNARY &&
        return _composed_has_sub(rhs, ctx, true)
    # Any other elementwise map (`hypot.(s1, mu .* s2)`, `sqrt.(mu)`) or
    # dotted operator (`mu ./ s`, `mu .^ 2`) reading a sub-predictor whose
    # value exists only as an LP node composes too: the reader takes that
    # LP value, evaluated once (`_composed_has_lp_sub`). Over other
    # definitions the derived-column path already reads a named local, so
    # it stays.
    _is_composed_map(rhs) && _composed_map_fn(rhs.args[1]) &&
        return _composed_has_lp_sub(rhs, ctx)
    rhs.head === :call || return false
    isempty(rhs.args) && return false
    op = rhs.args[1]
    op isa Symbol || return false
    op in _COMPOSED_MORE_OPS && _composed_has_lp_sub(rhs, ctx) &&
        return true
    if op === :.*
        return any(a -> _composed_has_sub(a, ctx, true), rhs.args[2:end])
    elseif op === :* && length(rhs.args) == 3
        a, b = rhs.args[2], rhs.args[3]
        sa = _shape_of(a, ctx.data, ctx.detmap, Dict{Symbol,Symbol}(),
            Set{Symbol}())
        sb = _shape_of(b, ctx.data, ctx.detmap, Dict{Symbol,Symbol}(),
            Set{Symbol}())
        return (sa === :scalar && _composed_has_sub(b, ctx, true)) ||
               (sb === :scalar && _composed_has_sub(a, ctx, true))
    elseif op === :.+ || op ===:.-
        # An operand that composes on its own (a product over a sub, or
        # a name bound to a composition) makes the whole sum composed.
        any(a -> _composed_trigger(a, ctx), rhs.args[2:end]) && return true
        return _composed_count_subs(rhs, ctx) >= 2 ||
            (_composed_count_subs(rhs, ctx) >= 1 &&
                _composed_has_scalar_leaf(rhs, ctx))
    else
        return any(a -> _composed_trigger(a, ctx), rhs.args[2:end])
    end
end

# A root that is only an admitted map over ONE bare sub-predictor
# (`exp.(mu)`, `logistic.(mu)`, or a name bound to one) is a link
# spelling when the distribution slot peels that inverse. A value use
# retains the same map as a composition; inside a larger expression it
# likewise composes (`exp.(la) .* th`).
function _is_bare_sub_map(rhs, ctx)
    if rhs isa Symbol
        haskey(ctx.detmap, rhs) && ctx.detmap[rhs] !== rhs || return false
        return _is_bare_sub_map(ctx.detmap[rhs], ctx)
    end
    _is_composed_map(rhs) && rhs.args[1] in _COMPOSED_UNARY || return false
    # Multi-operand maps are not bare-sub link spellings; returning false
    # routes them to `_extract_composed_tree`, which fails closed with the
    # one-operand guidance instead of a raw `only` ArgumentError.
    length(rhs.args[2].args) == 1 || return false
    arg = only(rhs.args[2].args)
    arg isa Symbol || return false
    _composed_trigger(arg, ctx) && return false
    return _is_composed_sub(arg, ctx, true)
end

_composed_root(rhs, ctx) =
    _composed_trigger(rhs, ctx) && !_is_bare_sub_map(rhs, ctx)

"""Extract + validate a combination tree (trigger already fired).
Leaves: sub-predictors (vector defs) and scalars (sampled names,
scalar definitions) and sub-free scalar subexpressions, a literal
included (`hypot.(1.0, mu)`, `mu .^ 2`), each one scalar leaf.
Everything else fails closed with guidance."""
function _extract_composed_tree(pname, node, ctx, subs::Vector{Symbol},
        scalars::Vector{Symbol}, datas::Vector{Symbol} = Symbol[])
    where = "predictor $pname"
    if node isa Symbol
        # A retained value is read by name, also through a pure alias
        # (a submodel returning another call's value).
        target = _alias_target(node, ctx)
        if target in ctx.value_defs
            target in datas || push!(datas, target)
            return target
        end
        # A name bound to a composition inlines its tree (the definition
        # is absorbed — it never also emits as a derived column).
        if haskey(ctx.detmap, node) && _composed_trigger(node, ctx)
            push!(ctx.absorbed, node)
            push!(ctx.inlined, node)
            return _extract_composed_tree(pname, ctx.detmap[node], ctx,
                subs, scalars, datas)
        end
        if _is_composed_sub(node, ctx, true)
            node in subs || push!(subs, node)
            return node
        elseif _is_composed_scalar(node, ctx)
            node in scalars || push!(scalars, node)
            return node
        elseif node in ctx.data
            # A bound data column read elementwise in-graph (v3:
            # `(log_time .- loc) .* exp.(ls)`); it becomes a term column.
            node in datas || push!(datas, node)
            return node
        elseif _is_plate_column_call(get(ctx.detmap, node, nothing))
            # An array-cell plate column: a per-observation value, read
            # like a data column.
            node in datas || push!(datas, node)
            return node
        elseif get(ctx.detshape, node, :scalar) === :vector &&
                _reads_declared_array(ctx.detmap[node], ctx)
            # Array-derived values remain graph values, including a
            # gather (`mu_base = a0 .+ z[g]`), which its inline spelling
            # reads as a column too. Interning one as an LP would consume
            # the definition needed by other readers, such as a
            # reduction of a library contrast.
            node in datas || push!(datas, node)
            return node
        elseif get(ctx.detshape, node, :scalar) === :vector &&
                _composed_data_only(node, ctx, Set{Symbol}())
            # A data-only definition (`v = log.(x .+ 2)`) is a derived
            # column, read like the data its inline spelling reads.
            node in datas || push!(datas, node)
            return node
        elseif _is_array_def(node, ctx)
            # A model-level array definition (`r = d[:, 1]`, a column of a
            # collected row matrix) is one value leaf, as its inline
            # spelling is: naming a subexpression never changes legality.
            node in scalars || push!(scalars, node)
            return node
        elseif haskey(ctx.detmap, node)
            return _sfail("$where combines $node, which is neither an " *
                "affine sub-predictor nor a scalar (latent/scan " *
                "parts stay out of v1 compositions)")
        else
            return _sfail("$where combines $node, which names nothing — " *
                "declare it (`$node ~ Prior` or `$node = ...`)")
        end
    end
    # A sub-free scalar subexpression (a literal, `s + 1`, `sqrt(v)`) is
    # one scalar leaf, exactly as if bound to a name first.
    _is_composed_scalar_expr(node, ctx) &&
        return _composed_scalar_leaf!(pname, node, ctx, scalars)
    node isa Expr || return _sfail("$where composition node " *
        "$(repr(node)) is not admitted (v1: `. .*`/`.+`/`.−` over " *
        "sub-predictors and scalars)")
    if node.head === :ref && _reads_array_value(node, ctx; follow = false)
        column = _extract_column(pname, node, ctx)
        push!(datas, column)
        return column
    end
    if _is_composed_map(node)
        f = node.args[1]
        _composed_map_fn(f) || return _sfail("$where maps " *
            "$(repr(f)). over a composition — admitted elementwise maps: " *
            "$(join(string.(_COMPOSED_UNARY, "."), ", ")), the dotted " *
            "built-in math functions, `ifelse.`, and dotted functions " *
            "visible in the model module")
        fargs = node.args[2].args
        f isa Symbol && length(fargs) != _elementwise_arity(f) &&
            return _sfail("$where $(repr(f)). takes " *
                _operands_phrase(_elementwise_arity(f)))
        isempty(fargs) && return _sfail("$where $(repr(f)). takes at " *
            "least one operand")
        return Expr(:., f, Expr(:tuple, (_extract_composed_tree(pname, a,
            ctx, subs, scalars, datas) for a in fargs)...))
    end
    node.head === :call || return _sfail("$where composition node " *
        "$(repr(node)) is not admitted (v1: `. .*`/`.+`/`.−` over " *
        "sub-predictors and scalars)")
    isempty(node.args) && return _sfail("$where has an empty call node")
    op = node.args[1]
    # An undotted module call over data and model values (`f(x)`,
    # `f(s, x)`, `f(base)` with `base` a sub-predictor) is one model-level
    # value leaf, as its named spelling (`v = f(base)`) and a call reading
    # no column are. A sub-predictor it reads is then read by an ordinary
    # value, so lowering replans it as a retained value, as for the named
    # spelling.
    op isa GlobalRef && return _composed_scalar_leaf!(pname, node, ctx, scalars)
    op isa Symbol || return _sfail("$where has an anonymous call node")
    args = [a for a in node.args[2:end] if !(a isa LineNumberNode)]
    if op === :.* || op === :.+ || op ===:.-
        if (op === :.+ || op === :.*) && length(args) == 1
            return _extract_composed_tree(pname, only(args), ctx, subs,
                scalars, datas)
        end
        if (op === :.+ || op === :.*) && length(args) > 2
            # `.+(a, b, c)` broadcasts Julia's n-ary `+`, which folds
            # left: `(a .+ b) .+ c` elementwise (likewise `.*`). BRM
            # emits Julia's n-ary `a + b + c` this way.
            return _extract_composed_tree(pname, Expr(:call, op,
                Expr(:call, op, args[1:end-1]...), args[end]), ctx, subs,
                scalars, datas)
        end
        ok = op === :.- ? length(args) in (1, 2) : length(args) == 2
        ok || return _sfail("$where `$op` takes " *
            (op === :.- ? "one or two operands" : "at least one operand"))
        return Expr(:call, op, (_extract_composed_tree(pname, a, ctx,
            subs, scalars, datas) for a in args)...)
    elseif op in _COMPOSED_MORE_OPS
        length(args) == 2 || return _sfail("$where `$op` takes two " *
            "operands")
        return Expr(:call, op, (_extract_composed_tree(pname, a, ctx,
            subs, scalars, datas) for a in args)...)
    elseif op === :* && length(args) >= 2
        # Julia-valid scalar `*` normalizes to dotted (Base broadcasts —
        # behavior-preserving, the canonicalization doctrine); an n-ary
        # product `z * lam * tau` folds left like Julia's `*`.
        lhs = length(args) == 2 ? args[1] :
            Expr(:call, :*, args[1:end-1]...)
        return _extract_composed_tree(pname, Expr(:call, :.*, lhs,
            args[end]), ctx, subs, scalars, datas)
    elseif op === :- && length(args) == 1
        # Unary negation is ordinary Julia array arithmetic. Its
        # elementwise canonical form preserves the authored value.
        return Expr(:call, :.-, _extract_composed_tree(pname, only(args),
            ctx, subs, scalars, datas))
    elseif op === :+ || op === :- || op === :/ || op === :^
        return _sfail("$where combines vectors without dots: " *
            "$(repr(node)) — as in Julia, write the dotted form " *
            "(`. .*`/`.+`/`.−`) or bind scalar subexpressions to a name " *
            "first (`s = be + 1; eta = s .* th`)")
    else
        return _sfail("$where applies `$op` inside a composition — " *
            "compositions admit dotted operators and dotted elementwise " *
            "functions over sub-predictors, scalars and data (write `$op` " *
            "dotted, or bind the value to a name first)")
    end
end

# A composition operand with no sub-predictor and no column that is not
# itself a composition operator node (those recurse): a scalar value. A
# column read only wholly (`1 + sum(x)^2`, `sum(th) / 2`) leaves a scalar
# too, as the named definition `m = 1 + sum(x)^2` is one scalar leaf.
function _is_composed_scalar_expr(node, ctx)
    node isa Number && return true
    node isa Expr || return false
    Meta.isexpr(node, :call) && !isempty(node.args) &&
        node.args[1] in REDUCTION_FNS && return true
    _composed_has_sub(node, ctx, true) && return false
    _reads_column(node, ctx) && _canon_shape(node, ctx) !== :scalar &&
        return false
    _is_composed_map(node) && return false
    node.head === :call && !isempty(node.args) &&
        node.args[1] in (:.*, :.+, :.-, :*) && return false
    return true
end

function _composed_scalar_leaf!(pname, node, ctx, scalars)
    _reject_unknown_calls("predictor $pname", node)
    nm = get!(ctx.leaf_exprs, node) do
        k = length(ctx.leaf_exprs) + 1
        n = Symbol(:_rkppl_leaf_, k)
        while n in ctx.taken
            k += 1
            n = Symbol(:_rkppl_leaf_, k)
        end
        push!(ctx.taken, n)
        v = _fold_literal(node)
        push!(ctx.synth_assigns, AssignmentSpec(n,
            v === nothing ? node : v, n))
        n
    end
    nm in scalars || push!(scalars, nm)
    return nm
end

"""Lower a composed predictor at a response/scale location: intern the
affine subs (scale-predictor pattern, IdentityLink), then push the
composed predictor itself. Nested compositions fail in the guard."""
function _lower_composed_predictor(pname, rhs, ctx, lhs, pred_link,
        predictors, pred_idx, coefuse)
    subs = Symbol[]
    scalars = Symbol[]
    datas = Symbol[]
    tree = _extract_composed_tree(pname, rhs, ctx, subs, scalars, datas)
    for s in subs
        if haskey(pred_idx, s)
            pred = predictors[pred_idx[s]]
            all(t -> t.kind in _COMPOSED_SUB_KINDS, pred.terms) || _sfail(
                "predictor $pname: sub-predictor $s must be affine " *
                "(no nested compositions, latents, or other summands)")
            push!(ctx.absorbed, s)
            continue
        end
        srhs = ctx.detmap[s]
        terms, uses = _analyze_predictor(s, srhs, ctx, lhs;
            composed_sub = true)
        all(t -> t.kind in _COMPOSED_SUB_KINDS, terms) || _sfail(
            "predictor $pname: sub-predictor $s must be affine " *
            "(no nested compositions, latents, or other summands)")
        _record_coefuses!(coefuse, s, uses, lhs)
        push!(predictors, PredictorSpec(s, IdentityLink, terms, s))
        pred_idx[s] = length(predictors)
        # An interned sub is absorbed like a response location — its
        # definition never also emits as a derived column.
        push!(ctx.absorbed, s)
    end
    for c in scalars
        if haskey(coefuse, c)
            owners = join(unique!(map(first, copy(coefuse[c]))), ", ")
            _check_owned_coefficient(c, ctx, "predictor $pname: scalar $c is a " *
                "coefficient of predictor $owners — a scalar leaf cannot " *
                "also be a coefficient (rename one)")
        end
    end
    label = Symbol(pname, "_composed")
    term = TermSpec(ComposedTerm, ColumnRef[datas...],
        (tree = tree, subs = subs, scalars = scalars), label, label)
    push!(predictors, PredictorSpec(pname, pred_link, [term], pname))
    pred_idx[pname] = length(predictors)
    return pname
end

function _analyze_predictor(pname, rhs, ctx, lhs; composed_sub::Bool = false)
    where = "predictor $pname"
    _composed_root(rhs, ctx) && _sfail(
        "predictor $pname combines sub-predictors inside a nested " *
        "definition — compositions lower only at response/scale " *
        "locations (bind the pieces: `th = ...; eta = be .* th`, then " *
        "use `eta`)")
    expanded = _inline_structure(rhs, ctx, Set{Symbol}([pname]), where)
    _find_hcat(expanded) && _sfail(
        "predictor $pname calls `hcat` outside a matrix definition — " *
        "bind the matrix to a name first (`X = hcat(ones(length(x)), x, ...)`)")
    _reject_unknown_calls(where, expanded)
    canon = _canonical_expr(expanded, ctx.data, ctx.detmap, ctx.detshape,
        ctx.shape_env, where)
    out = Tuple{Int,Any}[]
    _collect_signed!(out, canon, 1, pname)
    terms = TermSpec[]
    uses = Tuple{Symbol,Symbol,Int}[]
    addr_owner = Dict{Symbol,Symbol}()
    for (sign, core) in out
        term, use = _classify_summand(pname, core, sign, ctx)
        if term.kind === FactorTerm && use !== nothing &&
                use[1] in ctx.ordinary_parameters && haskey(ctx.factor_axes, use[1]) &&
                any(t -> t.kind === FactorTerm && _parameter_term(t) &&
                    t.columns == term.columns && haskey(ctx.factor_axes, t.options.parameter) &&
                    ctx.factor_axes[t.options.parameter] != ctx.factor_axes[use[1]],
                    terms)
            # LevelMaps describe one design block per predictor/column.
            # A different declared subset keeps its ordinary gather value.
            term, use = _extract_summand(pname, core, sign, ctx)
        end
        if use !== nothing
            name, addr, use_sign = use
            # Matrix uses claim every element addressee (one coefficient
            # per column, across matrix and affine terms alike).
            addrs = addr in keys(ctx.matrices) ?
                _matrix_element_addressees(ctx.matrices[addr]) : (addr,)
            # Ordinary parameters can share columns and have multiple readers.
            # Legacy construct-owned coefficients retain one-owner checks.
            any(u -> u[1] === name, uses) && _check_owned_coefficient(name, ctx,
                "predictor $pname: coefficient $name is used twice")
            for a in addrs
                haskey(addr_owner, a) && addr_owner[a] !== name &&
                    !(name in ctx.ordinary_parameters ||
                        addr_owner[a] in ctx.ordinary_parameters) && _sfail(
                    "predictor $pname: column $a has two coefficients " *
                    "$(addr_owner[a]) and $name — one coefficient per column")
                addr_owner[a] = name
            end
            push!(uses, use)
            if name in ctx.ordinary_parameters && term.kind in
                    (InterceptTerm, ContinuousTerm, FactorTerm,
                     MatrixTerm)
                # Record the exact use here: an addressee lookup loses identity
                # when distinct parameters multiply the same column. The use
                # sign includes both the summand and its product factors.
                term = TermSpec(term.kind, term.columns,
                    merge(term.options, (; parameter = name, sign = use_sign)),
                    term.addressee, term.label)
            end
        end
        push!(terms, term)
    end
    # Coefficient-free values and self-priored trajectory/basis terms can
    # feed a predictor directly; an empty predictor still has no value.
    if isempty(uses) && !(!isempty(terms) &&
            all(t -> t.kind === OffsetTerm ||
                t.kind === LatentTerm ||
                t.kind === ComposedTerm ||
                t.kind === ScanSummandTerm, terms))
        _sfail("predictor $pname has no estimated coefficients or " *
               "coefficient-free value terms")
    end
    return terms, uses
end

function _inline_structure(ex, ctx, visited::Set{Symbol}, where)
    ex isa Symbol || return _inline_structure_expr(ex, ctx, visited, where)
    haskey(ctx.detmap, ex) || return ex
    ex in ctx.declaration_data && return ex
    # A computed definition stays one named value for its readers.
    (ex in ctx.shared_defs || ex in ctx.value_defs) && return ex
    # Array-valued definitions (`M = (sd .* L)'`) stay named values.
    ctx.detshape[ex] === :array && return ex
    if ex in ctx.structural || ctx.detshape[ex] ∉ (:vector, :array) ||
            _is_factor_coefficient_alias(ex, ctx)
        ex in visited && _sfail("$where: cyclic definition through $ex")
        push!(ctx.absorbed, ex)
        push!(visited, ex)
        push!(ctx.inlined, ex)
        out = _inline_structure(ctx.detmap[ex], ctx, visited, where)
        delete!(visited, ex)
        return out
    end
    return ex
end
function _inline_structure_expr(ex, ctx, visited, where)
    ex isa Expr || return ex
    # The quoted cell body reads the named dependencies carried beside it.
    # Keep both together: substituting only the dependency arguments would
    # discard definitions that the body still indexes by their original name.
    _is_plate_column_call(ex) && return ex
    if ex.head === :ref && _is_model_value_def(ex.args[1], ctx)
        return Expr(:ref, ex.args[1],
            (_inline_structure(a, ctx, visited, where) for a in ex.args[2:end])...)
    end
    return Expr(ex.head, (_inline_structure(a, ctx, visited, where)
                          for a in ex.args)...)
end

# The definition a pure alias chain (`z = q`, `q = v`) names.
function _alias_target(node::Symbol, ctx)
    seen = Set{Symbol}()
    while haskey(ctx.detmap, node) && ctx.detmap[node] isa Symbol && node ∉ seen
        push!(seen, node)
        node = ctx.detmap[node]
    end
    return node
end

# Computed definitions an inlining reader folded into itself. Each stays
# one named value (`_computed_value_def` says which qualify).
_inlined_computed_defs(ctx, canonmap, data) = Set{Symbol}(nm for nm in ctx.inlined
    if haskey(canonmap, nm) && nm ∉ ctx.shared_defs && nm ∉ ctx.value_defs &&
        _computed_value_def(nm, ctx, canonmap, data))

# A module call's model-level result, written inline or read through a
# definition that stays named for its readers: a shared definition (the
# definitions below it are no longer inlined either), or the value a
# positional read indexes, which `_inline_structure_expr` keeps named.
_reads_module_value(ex, ctx) = _contains_module_call(ex) ||
    any(s -> s in ctx.shared_defs && _def_reads_module(s, ctx),
        _value_symbols(ex)) ||
    _indexes_module_value(ex, ctx)
# `bt = f(x, a); bt[1]` indexes the call's result exactly as the inline
# `f(x, a)[1]` does. The element has the function's own shape (a tuple
# element may be a vector), so it is never a proven scalar offset.
_indexes_module_value(ex, ctx) = ex isa Expr &&
    ((ex.head === :ref && _is_model_value_def(ex.args[1], ctx) &&
        _def_reads_module(ex.args[1], ctx)) ||
     any(a -> _indexes_module_value(a, ctx), ex.args))
_def_reads_module(s, ctx) = _contains_module_call(ctx.detmap[s]) ||
    any(t -> haskey(ctx.detmap, t) && _def_reads_module(t, ctx),
        _value_symbols(ctx.detmap[s]))

# An alias or a signed alias computes nothing of its own and still reads
# through; a data-only definition folds once at binding; array and matrix
# definitions are never inlined.
function _computed_value_def(nm, ctx, canonmap, data)
    rhs = canonmap[nm]
    rhs isa Expr || return false
    _is_signed_alias(rhs) && return false
    get(ctx.detshape, nm, :scalar) in (:array, :matrix) && return false
    return !_data_only(rhs, data, canonmap)
end

_is_signed_alias(rhs) = Meta.isexpr(rhs, :call) && length(rhs.args) == 2 &&
    rhs.args[1] in (:-, :.-, :+, :.+) &&
    (rhs.args[2] isa Symbol || _is_signed_alias(rhs.args[2]))

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
    core isa Number && return _extract_summand(pname, core, sign, ctx)
    core isa Symbol && return _classify_symbol(pname, core, sign, ctx)
    core isa Expr || _sfail("predictor $pname: $(repr(core)) is not affine " *
                            "in data (terms are bare coefficients, " *
                            "`coefficient .* column`, " *
                            "`coefficients[column]`, and bare columns)")
    if _contains_scan(core, ctx.scan_states)
        return _is_scan_product(core) ? _classify_scan(pname, core, sign, ctx) :
            _extract_summand(pname, core, sign, ctx)
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
    if head === :ref && length(core.args) == 2 &&
            (core.args[1] isa Expr || core.args[1] in ctx.data ||
                haskey(ctx.detmap, core.args[1]) ||
                core.args[1] in ctx.dirichlet_names) &&
            _canon_shape(core.args[2], ctx) === :vector
        # A gather of a value (`cum[c]`) is an observation column, exactly
        # as its named form `m = cum[c]`; a levels coefficient indexed by
        # its group (`c[g]`) stays a factor below.
        return _extract_summand(pname, core, sign, ctx)
    end
    # A slice keeping every row of a per-observation value (`P[:, 1]`,
    # `(X * B)[:, 1]`) is an observation column, as its named form is.
    head === :ref && length(core.args) >= 3 && core.args[2] === :(:) &&
        _canon_shape(core, ctx) === :vector &&
        return _extract_summand(pname, core, sign, ctx)
    head === :ref && return _classify_ref(pname, core, sign, ctx)
    head === :macrocall && _sfail("predictor $pname: macros do not lower " *
        "inside predictor expressions")
    # Literal arrays and other model-level arrays keep their whole Julia
    # value; broadcasting at the response decides their observation axes.
    _canon_shape(core, ctx) in (:scalar, :array) &&
        return _extract_summand(pname, core, sign, ctx)
    if head === :call && !isempty(core.args) && core.args[1] === :.*
        return _classify_product(pname, core, sign, ctx)
    end
    # Any other per-observation summand (a computed coefficient, a
    # parameter-scaled column, `exp.(s .* x)`) is an in-graph derived
    # column: the fallback, never a refusal.
    if _canon_shape(core, ctx) === :vector
        return _extract_summand(pname, core, sign, ctx)
    end
    # A bound number keeps its constant offset value, as a literal or a
    # scalar definition (`off = 0.25`) does.
    _is_bound_value_call(core) && return _extract_summand(pname, core, sign, ctx)
    # An undotted module call returns a whole (model-level) value, even
    # over columns it reads whole.
    _contains_module_call(core) && return _extract_summand(pname, core, sign, ctx)
    _canon_shape(core, ctx) === :scalar &&
        return _extract_summand(pname, core, sign, ctx)
    return _scalar_summand_error(pname, core)
end

# A scalar-valued summand (`a .+ s * z`, `a .+ phi[1]`) has no column to
# scale: predictor terms are per-observation values.
_scalar_summand_error(pname, core) = _sfail(
    "predictor $pname: $(repr(core)) is a scalar summand, not a " *
    "per-observation term — scale a column (`($(repr(core))) .* x`) or " *
    "give the predictor one intercept (`a ~ Normal(...)`) and fold the " *
    "scalar into its prior location")

# An `X * b` matmul: the design matrix's K columns take one coefficient
# vector — declared (`b[axes(X, 2)] .~ ...`, width-checked against the
# use matrix) or free (classified here, then refused at prior lowering —
# strict declarations, `_undeclared_vector`). Coefficient-ness
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
                                   "(`$S = hcat(ones(length(x)), x, ...)`)")
        length(Smat.columns) == K || _sfail(
            "predictor $pname: coefficient vector `$r` has " *
            "$(length(Smat.columns)) elements (sized by `$S`) but matrix " *
            "`$X` has $K columns")
    else
        role = r in ctx.value_arrays ? "a declared array (an `hcat` " *
                "design matrix multiplies a coefficient vector " *
                "`b[axes($X, 2)] .~ ...`; a bound data matrix multiplies " *
                "arrays, `B * $r`)" :
            r in ctx.data ? "a data column" :
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
    factors = core.args[2:end]
    length(factors) == 2 || return _extract_summand(pname, core, sign, ctx)
    stripped = [_strip_sign(f) for f in factors]
    inner = sign * prod(first, stripped)
    states = [g for (_, g) in stripped if g isa Symbol && g in ctx.scan_states]
    others = [g for (_, g) in stripped if !(g isa Symbol && g in ctx.scan_states)]
    (inner > 0 && length(states) == 1 && length(others) == 1) ||
        return _extract_summand(pname, core, sign, ctx)
    coef = only(others)
    if coef isa Symbol && coef ∉ ctx.prior_names && coef ∉ ctx.data &&
            !haskey(ctx.detmap, coef) && coef ∉ ctx.scan_states &&
            coef ∉ ctx.plate_names
        _sfail("predictor $pname reads undeclared coefficient $coef")
    end
    # Preserve the compact direct-splice IR for a sampled scalar. Other
    # arithmetic over a trajectory is an ordinary derived vector value.
    (coef isa Symbol && coef in ctx.prior_names && coef ∉ ctx.sized_decls) ||
        return _extract_summand(pname, core, sign, ctx)
    push!(ctx.scan_coefs, coef)
    label = Symbol("scan_", pname, "_", only(states))
    return TermSpec(ScanSummandTerm, ColumnRef[],
        (scan_id = only(states), coef = coef), label, label), nothing
end

# The scan state a pure alias chain (`w = u`, `v = w`) names, or nothing.
function _scan_alias_target(nm::Symbol, ctx)
    seen = Set{Symbol}()
    while haskey(ctx.detmap, nm) && !(nm in seen)
        push!(seen, nm)
        nxt = ctx.detmap[nm]
        nxt isa Symbol || return nothing
        nm = nxt
    end
    return nm in ctx.scan_states && !isempty(seen) ? nm : nothing
end

function _classify_symbol(pname, core::Symbol, sign::Int, ctx)
    # (Design matrices never reach here as bare summands: named
    # definitions screen them at extraction, inline locations at the
    # location arm, and dotted compositions at canonicalization.)
    if core in ctx.data || core in ctx.vecdefs
        # An offset has no signed coefficient coordinate. Keep a negative
        # summand in its value expression instead of discarding the sign.
        sign < 0 && return _extract_summand(pname,
            Expr(:call, :.-, core), 1, ctx)
        return TermSpec(OffsetTerm, [core], NamedTuple(),
        core, Symbol(core, "_off")), nothing
    end
    if core in ctx.plate_names
        sign < 0 && return _extract_summand(pname,
            Expr(:call, :.-, core), 1, ctx)
        return TermSpec(LatentTerm, [core], NamedTuple(),
            core, Symbol(core, "_lat")), nothing
    end
    if core in ctx.scan_states
        # A bare scan state is a beta-free summand: the state spliced
        # unscaled (`mu = a .+ x`, the dar/`mo1` shape). Negation is a value.
        sign > 0 || return _extract_summand(pname, Expr(:call, :.-, core), 1, ctx)
        label = Symbol("scan_", pname, "_", core)
        return TermSpec(ScanSummandTerm, ColumnRef[],
            (scan_id = core, coef = nothing), label, label), nothing
    end
    haskey(ctx.detmap, core) && return _extract_summand(pname, core, sign, ctx)
    core in ctx.prior_names && core ∉ ctx.coef_priors &&
        return _extract_summand(pname, core, sign, ctx)
    return TermSpec(InterceptTerm, ColumnRef[], NamedTuple(), :Intercept,
        :intercept), (core, :Intercept, sign)
end

# Whether a summand reads a per-observation value (a data column, a
# vector definition, a per-cell latent); otherwise it is a scalar.
_reads_column(ex, ctx) = any(s -> s in ctx.data || s in ctx.vecdefs ||
    s in ctx.plate_names, _value_symbols(ex))

# A derived column reads scalars and columns: a vector-valued parameter
# (a simplex, a sized coefficient vector) read whole is a shape error.
# An indexed read (`phi[1]`, `z[g]`) is an element or a gather, not the
# whole vector; its index is still screened.
function _check_scalar_reads(pname, core, ctx)
    for v in _whole_value_symbols(core)
        v in ctx.vector_params && _sfail("predictor $pname: " *
            "$(repr(core)) reads the vector parameter $v as a scalar — " *
            "index it per observation (`$v[g]`) or per element")
    end
    return nothing
end

function _whole_value_symbols(ex)
    out = Set{Symbol}()
    _whole_value_symbols!(out, ex)
    return out
end
function _whole_value_symbols!(out::Set{Symbol}, ex)
    if ex isa Expr && ex.head === :ref && !isempty(ex.args)
        for a in ex.args[2:end]
            _whole_value_symbols!(out, a)
        end
        ex.args[1] isa Symbol || _whole_value_symbols!(out, ex.args[1])
        return nothing
    end
    if ex isa Expr
        # Recurse through the call/broadcast value positions
        # `_value_symbols` visits, one argument at a time so nested refs
        # keep their exemption.
        if ex.head === :call && !isempty(ex.args)
            foreach(a -> _whole_value_symbols!(out, a), ex.args[2:end])
            return nothing
        elseif ex.head === :. && length(ex.args) == 2 &&
               ex.args[2] isa Expr && ex.args[2].head === :tuple
            foreach(a -> _whole_value_symbols!(out, a), ex.args[2].args)
            return nothing
        end
    end
    union!(out, _value_symbols(ex))
    return nothing
end

# Anonymous value substructure becomes an offset. Model-level scalars
# stay scalar assignments; preprocessing broadcasts them over the rows.
function _extract_summand(pname, core, sign::Int, ctx)
    if core isa Symbol && core in ctx.value_arrays
        value = sign < 0 ? Expr(:call, :.-, core) : core
        leaf = _composed_scalar_leaf!(pname, value, ctx, Symbol[])
        label = Symbol(core, :_value)
        return TermSpec(ComposedTerm, ColumnRef[],
            (tree = leaf, subs = Symbol[], scalars = [leaf]), label, label), nothing
    end
    shape = _canon_shape(core, ctx)
    if core isa Number || shape in (:scalar, :array)
        e = sign < 0 ? Expr(:call, :-, core) : core
        nm = _composed_scalar_leaf!(pname, e, ctx, Symbol[])
        if shape === :array ||
                (_reads_module_value(core, ctx) && !_is_bound_value_call(core))
            return TermSpec(ComposedTerm, ColumnRef[],
                (tree = nm, subs = Symbol[], scalars = [nm]), nm, nm), nothing
        end
        return TermSpec(OffsetTerm, [nm], NamedTuple(), nm,
            Symbol(nm, "_off")), nothing
    end
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
    # A product whose factors read a declared array directly
    # (`tau .* z[g]`, `phi[1] .* x`) is a value: extracted whole as an
    # in-graph column (its scalar factors stay ordinary parameters). A
    # named column (`b .* w`, `w` a definition) keeps the coefficient
    # term.
    _reads_array_value(core, ctx; follow = false) &&
        return _extract_summand(pname, core, sign, ctx)
    inner = sign
    coefs = Symbol[]
    values = Any[]
    stripped = Any[]
    # A factor that is neither a free coefficient nor a column — a sampled
    # parameter, a computed scalar (`s * z`, `phi[1]`), a literal — makes
    # the product a computed-coefficient summand: the derived-column
    # fallback below, never a refusal.
    computed = false
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
            else
                computed = true
            end
        elseif !(g isa Number) &&
                _canon_shape(g, ctx) === :vector
            push!(values, g)
        else
            computed = true
        end
    end
    if computed || length(coefs) > 1
        _reads_column(core, ctx) || return _scalar_summand_error(pname, core)
        _check_scalar_reads(pname, core, ctx)
        return _extract_summand(pname, core, sign, ctx)
    end
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
    if core.args[1] in ctx.prior_names && core.args[1] ∉ ctx.sized_decls
        # Base owns scalar indexing too: index 1 returns the value; other
        # integer or vector indices keep their ordinary Julia failures.
        # Out of index syntax, `end` / `begin` take Julia's lowering.
        return _extract_summand(pname,
            Expr(:call, GlobalRef(Base, :getindex), core.args[1],
                ReactiveKernels._kernel_ref_indices(core.args[1],
                    core.args[2:end])...), sign, ctx)
    end
    # Reads of a declared array value (`z[g]`, `phi[1]`, `c[1]` of an
    # `Ordered` vector) or of an array-valued definition (`b[g, 1]`,
    # `b = z * (sd .* L)'`) are values, not factor coefficients: scalar
    # values by position, per-observation columns when gathered.
    (core.args[1] in ctx.value_arrays ||
        core.args[1] in ctx.dirichlet_names ||
        core.args[1] in ctx.ordered_names ||
        _is_array_def(core.args[1], ctx) ||
        _is_model_value_def(core.args[1], ctx)) &&
        return _extract_summand(pname, core, sign, ctx)
    length(core.args) == 2 || _sfail("predictor $pname: factor indexing " *
                                     "takes `coefficients[group]` exactly, " *
                                     "got $(repr(core))")
    base, idx = core.args
    # An endpoint (`s[end]`) is one literal position: a value, as in Julia.
    _is_endpoint_position(idx) &&
        return _extract_summand(pname, core, sign, ctx)
    if haskey(ctx.factor_axes, base) &&
            idx !== ctx.factor_axes[base][1]
        return _extract_summand(pname, core, sign, ctx)
    end
    # A literal element (`b[1]`, `x[1]`) is one scalar value, broadcast
    # onto the predictor as Julia does, not a per-level column.
    idx isa Integer && return _extract_summand(pname, core, sign, ctx)
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

# Strict declarations (user decision 05oe96l): a name that is not a data
# column, a definition, or a declared parameter never becomes a parameter
# with a default prior — on every entry point (`@rkppl` and direct
# `lower_rkppl` alike). It fails naming the declaration to write.
function _undeclared_coefficient(name::Symbol, pname::Symbol, alt = nothing)
    _sfail("predictor $pname: `$name` is not a data column, a definition, " *
           "or a declared parameter — declare its prior " *
           "(`$name ~ Normal(0, 1)`" *
           (alt === nothing ? "" : " or `$alt`") * ") or fix the name")
end

function _undeclared_vector(name::Symbol, X::Symbol, where::Symbol)
    _sfail("$where: coefficient vector `$name` of design matrix $X has no " *
           "prior — declare it (`$name[axes($X, 2)] .~ Normal.(0, 1)`) or " *
           "fix the name")
end

# Coefficient priors: recovered by name from `coef ~ Fam(...)` statements
# (one family per addressee — see `_COEF_FAMILIES`); an undeclared scalar
# coefficient fails (`_undeclared_coefficient`). Factor coefficients instead
# take broadcast priors (`c[levels(g)] .~ Fam.(...)`), which also size the
# block — required, never defaulted — and each one emits its LevelMap.
# Plan order follows predictors, addressees in term order.
function _lower_coefficient_priors(sample, coefuse, predictors,
        matrices::Dict{Symbol,DesignMatrix}, hyper_names::Set{Symbol}, ctx)
    stated = Dict{Symbol,Any}()
    for s in sample
        haskey(coefuse, s.lhs) && (stated[s.lhs] = s)
    end
    # Multiple reads were validated by `_check_coefficient_uses`.
    priors = PopulationPrior[]
    levelmaps = LevelMap[]
    for pred in predictors
        for t in pred.terms
            if _parameter_term(t)
                if t.kind === FactorTerm
                    name = t.options.parameter
                    s = stated[name]
                    s.levels === nothing && _sfail("parameter $name is " *
                        "read as a factor but needs a sized levels declaration")
                    gcol, subset = s.levels
                    col = only(t.columns)
                    gcol === col || _sfail("parameter $name: levels " *
                        "column $gcol differs from use column $col")
                    any(m -> m.predictor === pred.name && m.column === col,
                        levelmaps) || push!(levelmaps,
                        LevelMap(pred.name, col, [], :levels, subset))
                end
                continue
            end
            # Offsets and latent/scan terms have their own priors;
            # composed terms read their affine sub-predictors' coefficients.
            (t.kind === OffsetTerm || t.kind === LatentTerm ||
                t.kind === ScanSummandTerm ||
                t.kind === ComposedTerm) && continue
            if t.kind === MatrixTerm
                append!(priors, _lower_matrix_priors(pred, t, coefuse,
                    stated, matrices, hyper_names, ctx))
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
                    stated, levelmaps, hyper_names, ctx))
                continue
            end
            haskey(stated, name) || _undeclared_coefficient(name, pred.name)
            s = stated[name]
            s.levels !== nothing && _sfail("coefficient $name takes a " *
                                           "scalar prior, not a levels prior — it is used " *
                                           "as $(t.kind), not a factor")
            fam, loc, scale, nu =
                _coefficient_prior(name, s.rhs, pred.name, addr)
            loc, scale = _signed_prior(fam, loc, scale, sign, ctx)
            push!(priors, PopulationPrior(pred.name, addr, fam, loc, scale,
                nu))
        end
    end
    return priors, levelmaps
end

# A matrix coefficient vector: K per-element PopulationPriors over the
# use-matrix columns (`:Intercept` at intercept positions). An unstated
# vector fails (`_undeclared_vector` — strict declarations). Stated vectors take
# `b[axes(S, 2)] .~ Fam.(args...)` — one shared family with scalar
# (shared) or length-K literal-vector (per-element) args — real
# broadcast semantics.
function _lower_matrix_priors(pred, t, coefuse, stated, matrices,
        hyper_names::Set{Symbol}, ctx)
    X = t.options.matrix
    m = get(matrices, X, nothing)
    m === nothing && _sfail("internal: matrix term over unknown matrix $X")
    use = _find_use(coefuse, pred.name, X)
    use === nothing && _sfail("internal: no coefficient use for " *
                              "($(pred.name), $X)")
    name, _, sign = use
    elems = _matrix_element_addressees(m)
    K = length(elems)
    haskey(stated, name) || _undeclared_vector(name, X, pred.name)
    s = stated[name]
    # Reachable only with a matrix marker sized for this use:
    # classification accepts a use only for declared vectors (width
    # checked against the use matrix) or free names (unstated, which
    # fail above).
    s.matrix === nothing && _sfail("internal: matrix prior for $name " *
                                   "lost its sizing matrix")
    fam, locs, scales, nus =
        _coefficient_matrix_prior(name, s.rhs, pred.name, K, hyper_names)
    out = PopulationPrior[]
    for (e, l, sc, n) in zip(elems, locs, scales, nus)
        l, sc = _signed_prior(fam, l, sc, sign, ctx)
        push!(out, PopulationPrior(pred.name, e, fam, l, sc, n))
    end
    return out
end

# Dotted matrix priors peel to one shared family plus K (location,
# scale, nu) triples: each arg is a Real (shared over elements), a
# shared-hyperparameter name, or a literal K-vector (per-element) —
# Julia broadcast semantics over `Fam.(args...)`, arg positions from
# `_COEF_SHAPES`.
function _coefficient_matrix_prior(name, rhs, pname, K,
        hyper_names::Set{Symbol})
    rhs isa Expr && rhs.head === :. && length(rhs.args) == 2 &&
        rhs.args[1] isa Symbol && haskey(_COEF_FAMILIES, rhs.args[1]) &&
        rhs.args[2] isa Expr && rhs.args[2].head === :tuple || _sfail(
            "coefficient $name of predictor $pname needs a broadcast " *
            "prior (one of $_COEF_FAMILY_MSG, dotted), got $(repr(rhs))")
    head = rhs.args[1]
    fam = _COEF_FAMILIES[head]
    args = rhs.args[2].args
    if fam === :flat
        isempty(args) || _sfail("coefficient $name of predictor $pname: " *
                                "`Flat.()` takes no arguments")
        return :flat, fill(0.0, K), fill(1.0, K), fill(NaN, K)
    end
    want, lipos, spos, npos = _COEF_SHAPES[fam]
    length(args) == want || _sfail("coefficient $name of predictor $pname " *
                                   "needs `$head` with $want arguments")
    vecs = [_matrix_prior_arg(name, a, pname, K, "arg$i", hyper_names)
        for (i, a) in enumerate(args)]
    npos == 0 || all(v -> v isa Real, vecs[npos]) || _sfail(
        "coefficient $name of predictor $pname: the `$head` nu " *
        "hyperparameter must be a literal")
    nus = npos == 0 ? fill(NaN, K) : Float64.(vecs[npos])
    return fam, vecs[lipos], vecs[spos], nus
end

# One matrix-prior argument → K per-element values: a literal or a name
# (a sampled scalar or scalar assignment — hoisted expressions included)
# shared over the elements, or a K-vector `[...]` of literals and names
# (per-element). Vector-valued arguments (`lambda .* tau`) need vector
# parameter values, which this path does not read.
function _matrix_prior_arg(name, a, pname, K, role, hyper_names::Set{Symbol})
    function elt(x)
        x isa Real && return Float64(x)
        x === :Inf && return Inf
        x isa Symbol && x in hyper_names && return x
        return _sfail("coefficient $name prior $role must be a literal, a " *
            "scalar parameter or assignment name, or a $K-vector of those, " *
            "got $(repr(a))")
    end
    if a isa Expr && a.head === :vect
        length(a.args) == K || _sfail(
            "coefficient $name prior $role has $(length(a.args)) " *
            "elements for $K columns — one per column")
        return Any[elt(x) for x in a.args]
    end
    return fill(elt(a), K)
end

function _lower_factor_prior(pred, t, name, sign, stated, levelmaps,
        hyper_names::Set{Symbol}, ctx)
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
    fam, loc, scale, nu =
        _coefficient_broadcast_prior(name, s.rhs, pred.name, col, hyper_names)
    push!(levelmaps, LevelMap(pred.name, col, [], :levels, subset))
    loc, scale = _signed_prior(fam, loc, scale, sign, ctx)
    return PopulationPrior(pred.name, col, fam, loc, scale, nu)
end

# Dotted coefficient priors peel to one shared (family, location, scale,
# nu): broadcast args are literals or shared-hyperparameter names
# (per-level priors are not in slice 1), positions from `_COEF_SHAPES`.
function _coefficient_broadcast_prior(name, rhs, pname, col,
        hyper_names::Set{Symbol})
    rhs isa Expr && rhs.head === :. && length(rhs.args) == 2 &&
        rhs.args[1] isa Symbol && haskey(_COEF_FAMILIES, rhs.args[1]) &&
        rhs.args[2] isa Expr && rhs.args[2].head === :tuple || _sfail(
            "coefficient $name of predictor $pname needs a broadcast " *
            "prior (one of $_COEF_FAMILY_MSG, dotted), got $(repr(rhs))")
    head = rhs.args[1]
    fam = _COEF_FAMILIES[head]
    args = rhs.args[2].args
    if fam === :flat
        isempty(args) || _sfail("coefficient $name of predictor $pname: " *
                                "`Flat.()` takes no arguments")
        return :flat, 0.0, 1.0, NaN
    end
    want, lipos, spos, npos = _COEF_SHAPES[fam]
    length(args) == want || _sfail("coefficient $name of predictor $pname " *
                                   "needs `$head` with $want arguments")
    # Location/scale take literals or shared-hyperparameter names (the
    # centered shape: `c[levels(g)] .~ Normal.(mu_alpha, sigma_alpha)`);
    # nu stays a literal, and per-level vectors stay out of slice 1.
    # Uniform bounds stay literals (no hyperparameter bounds in
    # slice 1). The surface admits known names (sampled + scalar
    # definitions); role resolution (scalar location, positive-support
    # scale) is the contract's `_validate_priors`, which sees the
    # whole plan.
    if fam === :uniform
        all(a -> a isa Real, args) ||
            _sfail("coefficient $name of predictor $pname: `Uniform` " *
                   "bounds must be finite literals " *
                   "(`Uniform.(lo, hi)` with lo < hi), got " *
                   "($(join(map(repr, args), ", ")))")
    end
    for (i, a) in enumerate(args)
        a isa Real && continue
        if a isa Symbol
            npos != 0 && i == npos &&
                _sfail("coefficient $name of predictor $pname: the " *
                       "`$head` nu hyperparameter must be a literal, " *
                       "got $(repr(a))")
            a in hyper_names || _sfail("coefficient $name prior " *
                "parameter must be a literal or shared-hyperparameter " *
                "name (per-level priors are not in slice 1), got " *
                "$(repr(a))")
            continue
        end
        _sfail("coefficient $name prior parameter must be a literal " *
               "or shared-hyperparameter name (per-level priors are " *
               "not in slice 1), got $(repr(a))")
    end
    vals = Any[a isa Real ? Float64(a) : a for a in args]
    return fam, vals[lipos], vals[spos], npos == 0 ? NaN : vals[npos]
end

function _find_use(coefuse, pname, addr)
    for (name, uses) in coefuse
        for (p2, a2, s2) in uses
            p2 === pname && a2 === addr && return (name, a2, s2)
        end
    end
    return nothing
end

# Coefficient-prior surface heads → IR families (per-addressee,
# prior-vocab slice). `StudentT` is `(nu, location, scale)` in Stan
# order; `Flat()` takes no arguments.
const _COEF_FAMILIES = Dict{Symbol,Symbol}(
    :Normal => :normal, :StudentT => :student_t, :Cauchy => :cauchy,
    :Laplace => :laplace, :Logistic => :logistic, :Flat => :flat,
    :Uniform => :uniform,
)
const _COEF_FAMILY_MSG =
    "Normal, StudentT, Cauchy, Laplace, Logistic, Flat, Uniform"

# Coefficient-prior arg shapes → (arity, location index, scale index, nu
# index or 0). One row per family, shared by the scalar, broadcast, and
# matrix peelers — a future family adds a row, never a branch.
const _COEF_SHAPES = Dict{Symbol,NTuple{4,Int}}(
    :normal => (2, 1, 2, 0), :cauchy => (2, 1, 2, 0),
    :laplace => (2, 1, 2, 0), :logistic => (2, 1, 2, 0),
    :student_t => (3, 2, 3, 1), :uniform => (2, 1, 2, 0),
)

# Scalar coefficient prior → (family, location, scale, nu): location and
# scale are literals or names (parameters, assignments, hoisted argument
# expressions — the contract resolves their roles); `nu` and `Uniform`
# bounds are literals (`_coef_prior_expressible` routes anything else to
# an ordinary parameter). `nu` is `NaN` unless StudentT; `:flat` carries
# conventional `(0.0, 1.0, NaN)`, ignored downstream.
function _coefficient_prior(name, rhs, pname, addr)
    rhs isa Expr && rhs.head === :call && !isempty(rhs.args) &&
        rhs.args[1] isa Symbol && haskey(_COEF_FAMILIES, rhs.args[1]) ||
        _sfail("coefficient $name of predictor $pname needs a " *
               "scalar prior (one of $_COEF_FAMILY_MSG), got $(repr(rhs))")
    head = rhs.args[1]
    fam = _COEF_FAMILIES[head]
    args = _plain_args(rhs, "coefficient prior")
    if fam === :flat
        isempty(args) || _sfail("coefficient $name of predictor $pname: " *
                                "`Flat()` takes no arguments")
        return :flat, 0.0, 1.0, NaN
    end
    want, lipos, spos, npos = _COEF_SHAPES[fam]
    length(args) == want || _sfail("coefficient $name of predictor $pname " *
                                   "needs `$head` with $want arguments")
    vals = Any[_coefficient_arg(name, a, pname) for a in args]
    return fam, vals[lipos], vals[spos], npos == 0 ? NaN : vals[npos]
end

function _coefficient_arg(name, a, pname)
    a isa Real && return Float64(a)
    a === :Inf && return Inf
    _is_signed_inf(a) && return a.args[1] === :- ? -Inf : Inf
    a isa Symbol && return a
    _sfail("coefficient $name of predictor $pname: prior argument " *
           "$(repr(a)) must be a literal or a name")
end

# The signed coefficient's prior (a use `.- b .* x` stores `-b`):
# symmetric families negate the location — a literal directly, a name
# through one synthetic negating assignment — and `Uniform(lo, hi)`
# becomes `Uniform(-hi, -lo)`.
function _signed_prior(fam, loc, scale, sign::Int, ctx)
    sign == 1 && return loc, scale
    fam === :uniform && return -scale, -loc
    fam === :flat && return loc, scale
    loc isa Real && return -loc, scale
    nm = get!(ctx.negated, loc) do
        n = Symbol(:_rkppl_neg_, loc)
        k = 1
        while n in ctx.taken
            k += 1
            n = Symbol(:_rkppl_neg_, loc, :_, k)
        end
        push!(ctx.taken, n)
        push!(ctx.synth_assigns, AssignmentSpec(n, Expr(:call, :-, loc), n))
        n
    end
    return nm, scale
end

# A scalar coefficient prior the affine block can carry: a coefficient
# family over positional literal/name arguments (expression arguments are
# names by now — `_hoist_prior_args!`), literal `Uniform` bounds and
# literal StudentT `nu`. Anything else (half or truncated distributions,
# parameter bounds) is an ordinary parameter
# prior. Arity mistakes stay with the coefficient path's message.
function _coef_prior_expressible(rhs)
    args = rhs.args[2:end]
    any(a -> a isa Expr && (a.head === :parameters || a.head === :kw),
        args) && return false
    fam = _COEF_FAMILIES[rhs.args[1]]
    fam === :flat && return true
    want, _, _, npos = _COEF_SHAPES[fam]
    length(args) == want || return true
    lit(a) = a isa Real || a === :Inf || _is_signed_inf(a)
    all(a -> lit(a) || a isa Symbol, args) || return false
    # Uniform bounds stay literal (`Inf` included, so the contract's
    # finite-bounds check reports it on the coefficient path).
    fam === :uniform && return all(lit, args)
    return npos == 0 || lit(args[npos])
end

_is_signed_inf(a) = a isa Expr && a.head === :call && length(a.args) == 2 &&
    a.args[2] === :Inf && (a.args[1] === :- || a.args[1] === :+)

# Pure-literal scalar arithmetic folds to its value (`1 / 2`, `sqrt(2)`):
# `nothing` unless every leaf is a number and every call is admitted
# scalar math, with a finite real result.
const _FOLD_FNS = (:+, :-, :*, :/, :^, :sqrt, :log, :log10, :log1p, :exp,
    :expm1, :abs)
function _fold_literal(a)
    a isa Real && return Float64(a)
    a isa Expr && a.head === :call && !isempty(a.args) &&
        a.args[1] in _FOLD_FNS || return nothing
    vals = Float64[]
    for x in a.args[2:end]
        v = _fold_literal(x)
        v === nothing && return nothing
        push!(vals, v)
    end
    r = try
        getfield(Base, a.args[1])(vals...)
    catch
        return nothing
    end
    return r isa Real && isfinite(r) ? Float64(r) : nothing
end

# Preserve Julia's integer result when folding a structural count; the
# prior arithmetic folder deliberately returns floating-point values.
# Source integer literals are Base bit integers. Testing that concrete union,
# rather than `Integer`, keeps `Int(x)` concretely inferred: a package that adds
# a constructor for its own `Integer` subtype then cannot invalidate lowering.
function _static_count(ex)
    ex isa Base.BitInteger && return Int(ex)
    _fold_literal(ex) === nothing && return nothing
    value = _eval_value_expr(ex, s -> _sfail("unbound count $s"), :count)
    return value isa Base.BitInteger ? Int(value) : nothing
end

"""Prior-argument hoisting: an expression in a prior's argument position
(`b ~ Normal(0, 2 * s)`, `c[levels(g)] .~ Normal.(0, sqrt(v))`,
`b[axes(X, 2)] .~ Normal.(0, [s1, 2 * s2])`) binds to a synthetic scalar
definition named after its statement and the argument's role (`b_scale`,
`b_scale_2` for a vector's second entry; `b_arg2` where the family names no
role, `_distribution_role`), so every prior
position downstream sees a literal or a name — naming a subexpression
never changes legality. Pure-literal arithmetic folds to its value instead
(`Normal(0, 1 / 2)` keeps a literal scale). Response arguments are
predictor locations and never hoist; ordinary vector-valued arguments stay
put (the per-position peelers name them), while plate priors name their
selected columns here too. `resolve` maps an argument through the
model's module-call resolution first, so a hoisted argument calls exactly
what the same expression bound to a name would. Rewrites `sample` and
`plate_specs` in place (both fresh from `_partition_statements`) and
returns the extended definition list."""
function _hoist_prior_args!(sample::Vector{SampleStmt}, det, data::Set{Symbol},
        plate_specs; resolve = identity)
    detmap = Dict{Symbol,Any}(nm => rhs for (nm, rhs) in det)
    detnames = Set{Symbol}(keys(detmap))
    taken = union(data, detnames, Set{Symbol}(s.lhs for s in sample),
        Set{Symbol}(p[1] for p in plate_specs))
    out = Pair{Symbol,Any}[nm => rhs for (nm, rhs) in det]
    memo = Dict{Symbol,Symbol}()
    function bind(lhs::Symbol, stem::Symbol, a; scalar_only=true)
        a isa Expr && !_is_signed_inf(a) && a.head !== :vect &&
            a.head !== :parameters && a.head !== :kw || return a
        f = _fold_literal(a)
        f === nothing || return f
        r = resolve(a)
        scalar_only && _shape_of(r, data, detmap, memo, Set{Symbol}()) !== :scalar &&
            return a
        _reject_unknown_calls("the prior of $lhs (argument " *
                              "`$(repr(a))`)", r)
        a = r
        nm, k = stem, 1
        while nm in taken
            k += 1
            nm = Symbol(stem, :_, k)
        end
        push!(taken, nm)
        push!(out, nm => a)
        detmap[nm] = a
        return nm
    end
    # Positional arguments only (a `:parameters` keyword block keeps its
    # place and is never an argument position); unchanged input returns
    # itself, so untouched statements keep their exact AST.
    function hoist_args(lhs, args; scalar_only=true, role = (i, n) -> nothing)
        local res = Any[]
        nargs = count(a -> !(a isa Expr && a.head === :parameters), args)
        i = 0
        for a in args
            if a isa Expr && a.head === :parameters
                push!(res, a)
                continue
            end
            i += 1
            r = role(i, nargs)
            stem = r === nothing ? Symbol(lhs, :_arg, i) : Symbol(lhs, :_, r)
            push!(res, a isa Expr && a.head === :vect ?
                Expr(:vect, (bind(lhs, Symbol(stem, :_, j), x; scalar_only)
                    for (j, x) in enumerate(a.args))...) :
                bind(lhs, stem, a; scalar_only))
        end
        return res == args ? args : res
    end
    family_role(head) = (i, n) -> _distribution_role(head, i, n)
    function hoist_call(lhs, rhs; scalar_only=true)
        rhs isa Expr && rhs.head === :call && !isempty(rhs.args) &&
            rhs.args[1] isa Symbol || return rhs
        if rhs.args[1] === :LKJCholesky && length(rhs.args) in (3, 4)
            # Shape and orientation are structural; only eta is a prior value.
            eta = bind(lhs, Symbol(lhs, :_eta), rhs.args[3])
            eta === rhs.args[3] && return rhs
            return Expr(:call, rhs.args[1], rhs.args[2], eta, rhs.args[4:end]...)
        end
        if rhs.args[1] === :Ordered && length(rhs.args) == 3
            inner = hoist_call(lhs, rhs.args[2]; scalar_only)
            return Expr(:call, :Ordered, inner, rhs.args[3])
        end
        if rhs.args[1] in (:truncated, :restricted) && length(rhs.args) == 4
            inner = hoist_call(lhs, rhs.args[2]; scalar_only)
            bounds = hoist_args(lhs, rhs.args[3:end]; scalar_only,
                role = (i, n) -> (:lower, :upper)[i])
            inner === rhs.args[2] && bounds == rhs.args[3:end] && return rhs
            return Expr(:call, rhs.args[1], inner, bounds...)
        end
        rhs.args[1] in _HOIST_FAMILIES || return rhs
        args = rhs.args[2:end]
        new = hoist_args(lhs, args; scalar_only, role = family_role(rhs.args[1]))
        new === args && return rhs
        return Expr(:call, rhs.args[1], new...)
    end
    for (i, s) in enumerate(sample)
        s.lhs in data && continue
        rhs = s.rhs
        if s.broadcast
            # Only sized coefficient-vector priors; other `.~` are
            # responses (data or derived) whose arguments are locations.
            s.levels === nothing && s.matrix === nothing && continue
            rhs isa Expr && rhs.head === :. && length(rhs.args) == 2 &&
                rhs.args[2] isa Expr && rhs.args[2].head === :tuple &&
                rhs.args[1] in _HOIST_FAMILIES || continue
            args = rhs.args[2].args
            new = hoist_args(s.lhs, args; role = family_role(rhs.args[1]))
            new === args && continue
            rhs = Expr(:., rhs.args[1], Expr(:tuple, new...))
        else
            s.lhs in detnames && continue
            rhs = hoist_call(s.lhs, rhs)
        end
        rhs === s.rhs && continue
        sample[i] = SampleStmt(s.lhs, rhs, s.broadcast, s.range, s.levels,
            s.matrix, s.dims, s.slices, s.count_columns)
    end
    for (i, (nm, rhs, rng, line)) in enumerate(plate_specs)
        # Per-cell prior arguments also hold selected columns. Name those
        # expressions like scalar arguments before the ordinary value passes.
        new = hoist_call(nm, rhs; scalar_only=false)
        new === rhs || (plate_specs[i] = (nm, new, rng, line))
    end
    return out
end

const _PARAM_FAMILIES = Dict{Symbol,Symbol}(
    :Normal => :normal, :Cauchy => :cauchy,
    :Exponential => :exponential, :Gamma => :gamma,
    :LogNormal => :lognormal, :Beta => :beta,
    :InverseGamma => :inverse_gamma, :StudentT => :student_t, :TDist => :student_t,
    :Laplace => :laplace, :Logistic => :logistic, :Uniform => :uniform,
    :Weibull => :weibull,
)

# Families whose positional prior arguments hoist (scalar `~` and per-cell
# latent priors); `truncated(D(...), lo, hi)` hoists D's arguments only.
const _HOIST_FAMILIES = union(Set{Symbol}(keys(_PARAM_FAMILIES)),
    Set{Symbol}(keys(_COEF_FAMILIES)), Set{Symbol}((:HalfNormal, :HalfCauchy)))

function _lower_parameters(sample, coefuse, ctx, glmuse)
    params = SampledParameter[]
    syms = Set{Symbol}()
    vectors = VectorParameter[]
    arrays = ArrayParameter[]
    for s in sample
        (s.lhs in ctx.data || s.lhs in ctx.derived_responses) && continue
        if _external_rhs(s.rhs)
            shape = s.dims === nothing ? Any[] : s.dims
            args = _external_parameter(s.lhs, s.rhs, shape, ctx.mod, ctx.taken;
                observed=s.lhs in ctx.conditioned, broadcast=s.broadcast)
            p = isempty(shape) ? SampledParameter(s.lhs, :external, args, nothing, s.lhs) :
                ArrayParameter(s.lhs, :external, args, shape, nothing, s.lhs)
            push!(isempty(shape) ? params : arrays, p)
            union!(syms, _value_symbols(args.rhs))
            continue
        end
        if s.slices === :per_level
            haskey(coefuse, s.lhs) && _sfail("$(s.lhs) is a predictor " *
                "coefficient and cannot also be an LKJCholesky factor")
            p = _lower_lkj_stack(s.lhs, s.dims, s.rhs)
            push!(arrays, p)
            union!(syms, _value_symbols(p.args.arg1))
            continue
        end
        if s.slices !== nothing
            push!(arrays, _lower_array_slices(s.lhs, s.dims, s.slices, s.rhs,
                coefuse, syms))
            continue
        end
        if s.dims !== nothing
            if haskey(ctx.threshold_uses, s.lhs)
                push!(vectors, _lower_plain_thresholds(s, coefuse, ctx))
                continue
            end
            push!(arrays, _lower_array_parameter(s.lhs, s.dims, s.rhs,
                coefuse, ctx, syms))
            continue
        end
        if !s.broadcast && _is_ordered_call(s.rhs)
            push!(vectors, _lower_ordered(s.lhs, s.rhs, coefuse, ctx))
            continue
        end
        if !s.broadcast && _is_lkj_cholesky_call(s.rhs)
            haskey(coefuse, s.lhs) && _sfail(
                "$(s.lhs) is a predictor coefficient and cannot also be " *
                "an LKJCholesky factor")
            p = _lower_lkj_cholesky(s.lhs, s.rhs)
            push!(arrays, p)
            union!(syms, _value_symbols(p.args.arg1))
            continue
        end
        if _is_dirichlet_call(s.rhs)
            haskey(coefuse, s.lhs) && _sfail(
                "$(s.lhs) is a predictor coefficient and cannot also be " *
                "a Dirichlet parameter")
            p = _lower_dirichlet(s.lhs, s.rhs)
            push!(vectors, p)
            union!(syms, _value_symbols(p.args.arg1))
            continue
        end
        haskey(coefuse, s.lhs) && continue
        haskey(glmuse, s.lhs) && s.lhs ∉ ctx.ordinary_parameters && continue
        # Sized declarations are ordinary array parameters.
        if s.levels !== nothing
            gcol, sub = s.levels
            dim = sub === Colon() ? Expr(:call, :levels, gcol) :
                Expr(:call, :levels, gcol, QuoteNode(sub))
            push!(arrays, _lower_array_parameter(s.lhs,
                Any[dim], s.rhs, coefuse, ctx, syms))
            continue
        end
        if s.matrix !== nothing
            push!(arrays, _lower_array_parameter(s.lhs,
                Any[Expr(:call, :axes, s.matrix, 2)], s.rhs, coefuse, ctx,
                syms))
            continue
        end
        p = if s.lhs in ctx.conditioned && s.rhs isa Expr &&
                s.rhs.head === :call && first(s.rhs.args) === :Binomial
            args = _distribution_args(:Binomial, _plain_args(s.rhs, "`Binomial`"))
            length(args) == 2 || _sfail("observed scalar Binomial takes a trial count and probability")
            vals = Tuple(_lower_param_arg(s.lhs, a, coefuse, ctx.matrices) for a in args)
            SampledParameter(s.lhs, :binomial, (arg1=vals[1], arg2=vals[2]), nothing, s.lhs)
        else
            _lower_parameter(s.lhs, s.rhs, coefuse, ctx.matrices)
        end
        push!(params, p)
        for v in (values(p.args)..., _support_args(p.support_override)...)
            union!(syms, _value_symbols(v))
        end
    end
    return params, syms, vectors, arrays
end

_is_lkj_cholesky_call(rhs) =
    rhs isa Expr && rhs.head === :call && !isempty(rhs.args) &&
    rhs.args[1] === :LKJCholesky

# `L ~ LKJCholesky(K, eta)` (Distributions.jl; the optional third argument
# is `'L'` or `'U'`). `K` is a literal or `size(M, d)` of
# a matrix `M`; `eta` is a scalar prior value.
function _lower_lkj_cholesky(lhs, rhs)
    args = _plain_args(rhs, "`LKJCholesky`")
    (length(args) == 2 || length(args) == 3) || _sfail("parameter $lhs: " *
        "`LKJCholesky` takes (K, eta) (`$lhs ~ LKJCholesky(3, 2.0)`), got " *
        "$(length(args)) arguments")
    K, eta = args[1], args[2]
    uplo = length(args) == 3 ? args[3] : 'L'
    uplo in ('L', 'U') || _sfail("parameter $lhs: `LKJCholesky` uplo " *
        "is 'L' or 'U', got $(repr(uplo))")
    dim = if K isa Base.BitInteger
        K >= 1 || _sfail("parameter $lhs: `LKJCholesky` dimension must be " *
            "≥ 1, got $K")
        Int(K)
    elseif K isa Expr && K.head === :call && length(K.args) == 3 &&
            K.args[1] === :size && K.args[2] isa Symbol && K.args[3] in (1, 2)
        Expr(:call, :size, K.args[2], K.args[3])
    else
        _sfail("parameter $lhs: `LKJCholesky` dimension is a literal or " *
            "`size(M, d)` of a matrix `M`, got $(repr(K))")
    end
    (eta isa Symbol || (eta isa Real && !(eta isa Bool) && isfinite(eta) &&
        eta > 0)) || _sfail("parameter $lhs: LKJ shape `eta` must be a " *
        "positive scalar literal or a declared value, got $(repr(eta))")
    prior_args = (arg1 = eta isa Real ? Float64(eta) : eta,)
    uplo === 'U' && (prior_args = merge(prior_args, (; uplo)))
    return ArrayParameter(lhs, :lkj_cholesky, prior_args,
        Any[dim, dim], nothing, lhs)
end

# One LKJ correlation factor per level of `g` (a per-level `@plate` cell):
# the stacked array `:lkj_cholesky_stack` with dims `[K, K, levels(g)]`.
# K is a literal ≥ 2 (a 1×1 factor is the constant 1).
function _lower_lkj_stack(lhs, dims::Vector{Any}, rhs)
    one = _lower_lkj_cholesky(lhs, rhs)
    K = one.dims[1]
    K isa Int && K >= 2 || _sfail("per-level `$lhs[k] ~ LKJCholesky(K, " *
        "eta)`: K is a literal ≥ 2, got $(repr(K))")
    return ArrayParameter(lhs, :lkj_cholesky_stack, one.args,
        Any[K, K, dims[3]], nothing, lhs)
end

# `z[axes...] .~ Fam.(args...)`: the elementwise prior is the scalar
# statement `Fam(args...)` applied per element (Distributions.jl
# broadcast), so it lowers through the scalar parameter parser (family,
# half/truncated support) with every non-literal argument passed through
# as a value (a name or expression read per element; literal vectors per
# element). Records the names the arguments read in `syms`.
function _lower_array_parameter(lhs, dims::Vector{Any}, rhs, coefuse, ctx,
        syms::Set{Symbol})
    haskey(coefuse, lhs) && _sfail("array $lhs is used as a predictor " *
        "coefficient — read it as a value (`$lhs[g]` per observation, " *
        "`B * $lhs` through a data matrix)")
    call = _undot_distribution(lhs, rhs)
    if Meta.isexpr(rhs, :call)
        for a in call.args[2:end]
            _canon_shape(a, ctx) === :scalar || _sfail("array $lhs: " *
                "a shared scalar distribution constructor requires scalar " *
                "arguments; use `$(call.args[1]).(...)` to broadcast arguments")
        end
    end
    held = Dict{Symbol,Any}()
    call = _hold_array_args(lhs, call, held)
    p = _lower_parameter(lhs, call, coefuse, ctx.matrices)
    args = map(v -> v isa Symbol ? get(held, v, v) : v, p.args)
    for v in values(args)
        v isa Symbol || v isa Expr || continue
        for r in _value_symbols(v)
            haskey(coefuse, r) && _sfail("$r is a predictor coefficient " *
                "and cannot also be a prior argument (array $lhs)")
            push!(syms, r)
        end
    end
    return ArrayParameter(lhs, p.family, args, dims, p.support_override, lhs)
end

# Multivariate slice families (`mv_slices.jl`): the distribution head and
# its family stem. An [`ArrayParameter`](@ref) slice family is
# `<stem>_<slices>` (`:mvnormal_cholesky_rows`, `:dirichlet_cols`,
# `:mvnormal_vector`, …).
const _MV_SLICE_STEMS = (MvNormalCholesky = :mvnormal_cholesky,
    MvNormal = :mvnormal, Dirichlet = :dirichlet, Ordered = :ordered_normal)

# The head of a multivariate distribution `D(args...)` / `D.(args...)`
# (`nothing` otherwise).
function _mv_head(rhs)
    rhs isa Expr || return nothing
    h = rhs.head === :call && !isempty(rhs.args) ? rhs.args[1] :
        rhs.head === :. && length(rhs.args) == 2 ? rhs.args[1] : nothing
    h isa Symbol && haskey(_MV_SLICE_STEMS, h) || return nothing
    return h
end

# Multivariate normals also declare one vector (`b[1:K] ~ MvNormal(...)`);
# a simplex and an ordered vector keep their unsized declarations
# (`phi ~ Dirichlet(alpha)`, `c ~ Ordered(Normal(0, 1), K)`).
_mv_vector_head(rhs) = _mv_head(rhs) in (:MvNormalCholesky, :MvNormal)

_is_slice_iterator(a) = a isa Expr && a.head === :call &&
    length(a.args) == 2 && a.args[1] in (:eachrow, :eachcol)
_is_ref_call(a) = a isa Expr && a.head === :call && length(a.args) == 2 &&
    a.args[1] === :Ref

# One argument of a slice prior, as written: shared (`Ref(x)` in a dotted
# call, any value in an undotted one) or per slice (`eachrow(M)` /
# `eachcol(M)`, dotted only — standard broadcasting pairs slice `g` with
# slice `g`). Returns the stored value: the shared value itself, or the
# `eachrow(M)` / `eachcol(M)` expression.
function _slice_prior_arg(what, head, a, dotted::Bool)
    a isa Expr && a.head === :parameters &&
        _sfail("$what: `$head` takes positional arguments only (no keywords)")
    if dotted
        _is_ref_call(a) && return a.args[2]
        _is_slice_iterator(a) && return a
        a isa Real && !(a isa Bool) && return a
        per = _slice_elements_arg(a)
        per === nothing || return per
        _sfail("$what: in the dotted `$head.(…)` every argument is shared " *
            "(`Ref(x)`), iterates slices (`eachrow(M)` / `eachcol(M)`) or " *
            "lists one vector per slice (`[v1, v2, …]`), got $(repr(a)) — " *
            "a bare array would broadcast over its elements, as in Julia")
    end
    _is_slice_iterator(a) && _sfail("$what: `$(repr(a))` gives one " *
        "argument per slice, which pairs slices only in a broadcast — " *
        "write `$head.(…)` with the shared arguments in `Ref(…)`")
    _is_ref_call(a) && _sfail("$what: `$(repr(a))` marks a shared argument " *
        "of a broadcast `$head.(…)`; an undotted `$head(…)` shares every " *
        "argument already")
    return a
end

# A vector literal of vector values in a dotted slice prior,
# `D.([v1, v2, …])`: broadcasting iterates its elements, so slice `g` takes
# `v_g`, exactly as `D.(eachcol(hcat(v1, v2, …)))` pairs them. Literal
# vectors of numbers become that matrix's literal columns; other entries are
# values whose lengths Julia checks. Numbers are not vectors (`nothing`).
function _slice_elements_arg(a)
    a isa Expr && a.head === :vect && !isempty(a.args) &&
        !any(x -> x isa Real, a.args) || return nothing
    lits = all(x -> x isa Expr && x.head === :vect &&
        all(y -> y isa Real, x.args), a.args)
    if lits && length(unique!([length(x.args) for x in a.args])) == 1
        K = length(first(a.args).args)
        return Expr(:call, :eachcol, Expr(:vcat,
            (Expr(:row, (x.args[i] for x in a.args)...) for i in 1:K)...))
    end
    return Expr(:call, :eachcol, Expr(:call, GlobalRef(Base, :hcat), a.args...))
end

# A vector-valued argument (a mean, a concentration): never a scalar.
function _slice_vector_arg(what, head, role, a)
    (a isa Real || a isa Bool) && _sfail("$what: the `$head` $role is a " *
        "vector, got the scalar $(repr(a))" *
        (role === "mean" ? " (a zero mean is `zeros(K)`)" : ""))
    a isa Expr || a isa Symbol || _sfail("$what: `$head` $role " *
        "$(repr(a)) is not a value")
    return a
end

# A matrix argument shared by every slice (a factor, a covariance).
function _slice_matrix_arg(what, head, role, a)
    _is_slice_iterator(a) && _sfail("$what: the `$head` $role is one " *
        "matrix shared by every slice — write `Ref($(repr(a.args[2])))`")
    (a isa Real || a isa Bool) && _sfail("$what: the `$head` $role is a " *
        "matrix, got the scalar $(repr(a))")
    a isa Expr && a.head === :vect && _sfail("$what: the `$head` $role " *
        "is a matrix, got the vector $(repr(a))")
    a isa Expr || a isa Symbol || _sfail("$what: `$head` $role " *
        "$(repr(a)) is not a value")
    return a
end

# A scalar argument shared by every slice (an element location / scale).
function _slice_scalar_arg(what, head, role, a)
    a isa Bool && _sfail("$what: `$head` $role $(repr(a)) is not a value")
    a isa Real && return isfinite(a) ? a : _sfail("$what: `$head` $role " *
        "must be finite, got $(repr(a))")
    a isa Expr && a.head === :vect && _sfail("$what: the `$head` $role is " *
        "a scalar, got the vector $(repr(a))")
    a isa Expr || a isa Symbol || _sfail("$what: `$head` $role " *
        "$(repr(a)) is not a value")
    return a
end

function _literal_count(what, head, n)
    n isa Integer && !(n isa Bool) && n >= 1 || _sfail("$what: `$head` " *
        "takes a literal length K ≥ 1, got $(repr(n))")
    return Int(n)
end

# Family arguments (positional keys) of each multivariate head.
function _mv_slice_args(::Val{:MvNormalCholesky}, what, args)
    length(args) == 2 || _sfail("$what: `MvNormalCholesky` takes (mean " *
        "vector, covariance Cholesky factor), got $(length(args)) arguments")
    return (arg1 = _slice_vector_arg(what, :MvNormalCholesky, "mean", args[1]),
        arg2 = _slice_matrix_arg(what, :MvNormalCholesky,
            "covariance Cholesky factor", args[2]))
end
function _mv_slice_args(::Val{:MvNormal}, what, args)
    length(args) == 2 || _sfail("$what: `MvNormal` takes (mean vector, " *
        "covariance matrix), got $(length(args)) arguments (a zero mean " *
        "is `zeros(K)`)")
    return (arg1 = _slice_vector_arg(what, :MvNormal, "mean", args[1]),
        arg2 = _slice_matrix_arg(what, :MvNormal, "covariance", args[2]))
end
function _mv_slice_args(::Val{:Dirichlet}, what, args)
    if length(args) == 2
        # Symmetric `Dirichlet(K, a)` (Distributions.jl): K copies of `a`.
        K = _literal_count(what, :Dirichlet, args[1])
        a = _slice_scalar_arg(what, :Dirichlet, "concentration", args[2])
        return (arg1 = Expr(:call, :fill, a, K),)
    end
    length(args) == 1 || _sfail("$what: `Dirichlet` takes a concentration " *
        "vector `Dirichlet(alpha)` or the symmetric `Dirichlet(K, a)`, got " *
        "$(length(args)) arguments")
    return (arg1 = _slice_vector_arg(what, :Dirichlet, "concentration",
        args[1]),)
end
function _mv_slice_args(::Val{:Ordered}, what, args)
    length(args) == 2 || _sfail("$what: `Ordered` takes (element " *
        "distribution, length) — `Ordered(Normal(m, s), K)`")
    d = args[1]
    d isa Expr && d.head === :call && !isempty(d.args) &&
        d.args[1] === :Normal && length(d.args) <= 3 || _sfail("$what: " *
            "`Ordered` takes a `Normal(m, s)` element distribution, got " *
            "$(repr(d))")
    m = length(d.args) >= 2 ? d.args[2] : 0.0
    sd = length(d.args) >= 3 ? d.args[3] : 1.0
    return (arg1 = _slice_scalar_arg(what, :Normal, "location", m),
        arg2 = _slice_scalar_arg(what, :Normal, "scale", sd),
        arg3 = _literal_count(what, :Ordered, args[2]))
end

# `eachrow(B[a, b]) .~ D`, `eachcol(B[a, b]) .~ D`, `b[ax] ~ D`: every
# slice of the declared array one draw of the multivariate `D`
# (`mv_slices.jl`). The family is `<stem>_<slices>`; arguments are
# model-level values — literal vectors, `zeros(K)`, array names or
# expressions over them — shared by every slice, or per slice
# (`eachrow(M)` / `eachcol(M)` in a dotted `D.(…)`).
function _lower_array_slices(lhs, dims::Vector{Any}, slices::Symbol, rhs,
        coefuse, syms::Set{Symbol})
    haskey(coefuse, lhs) && _sfail("array $lhs is used as a predictor " *
        "coefficient — read it as a value (`$lhs[g, 1]`, `$lhs[g, :] * v`)")
    what = slices === :vector ? "array $lhs" : "slice array $lhs"
    noun = slices === :rows ? "row" : slices === :cols ? "column" : "draw"
    head = _mv_head(rhs)
    if head === nothing
        h = rhs isa Expr && rhs.head in (:call, :.) && !isempty(rhs.args) ?
            rhs.args[1] : nothing
        _sfail("$what: each $noun is a vector, drawn by a multivariate " *
            "distribution — `MvNormalCholesky(mu, F)`, `MvNormal(mu, " *
            "Sigma)`, `Dirichlet(alpha)` or `Ordered(Normal(m, s), K)` — " *
            "got $(repr(rhs))" * (h isa Symbol ? " (an elementwise prior " *
            "is `$lhs[a, b] .~ $h.(…)`)" : ""))
    end
    dotted = rhs.head === :.
    dotted && !(rhs.args[2] isa Expr && rhs.args[2].head === :tuple) &&
        _sfail("$what: malformed broadcast $(repr(rhs))")
    raw = dotted ? rhs.args[2].args : _plain_args(rhs, "`$head`")
    # `Ordered`'s arguments (an element distribution, a length) broadcast
    # as scalars, so `Ordered.(…)` is `Ordered(…)`.
    head === :Ordered && (dotted = false)
    args = Any[_slice_prior_arg(what, head, a, dotted) for a in raw]
    fargs = _mv_slice_args(Val(head), what, args)
    for v in values(fargs)
        v isa Symbol || v isa Expr || continue
        for r in _value_symbols(_is_slice_iterator(v) ? v.args[2] : v)
            haskey(coefuse, r) && _sfail("$r is a predictor coefficient " *
                "and cannot also be a prior argument (array $lhs)")
            push!(syms, r)
        end
    end
    return ArrayParameter(lhs, Symbol(_MV_SLICE_STEMS[head], :_, slices),
        fargs, dims, nothing, lhs)
end

# `Fam.(args...)` → `Fam(args...)`, recursively through the distribution
# arguments of `truncated.(...)` (`truncated.(Normal.(0, s), 0, Inf)`).
function _undot_distribution(lhs, rhs)
    Meta.isexpr(rhs, :call) && return rhs
    rhs isa Expr && rhs.head === :. && length(rhs.args) == 2 &&
        rhs.args[1] isa Symbol && rhs.args[2] isa Expr &&
        rhs.args[2].head === :tuple || _sfail("array $lhs takes an " *
            "elementwise prior `Fam.(args...)` (`$lhs[1:3] .~ " *
            "Normal.(0, 1)`), got $(repr(rhs))")
    f = rhs.args[1]
    args = Any[a isa Expr && a.head === :. && length(a.args) == 2 &&
            a.args[1] isa Symbol && haskey(_PARAM_FAMILIES, a.args[1]) ?
            _undot_distribution(lhs, a) : a for a in rhs.args[2].args]
    return Expr(:call, f, args...)
end

# Replace each non-literal distribution argument with a placeholder name
# (the scalar parser admits literals and names only), recorded in `held`
# so the lowered arguments restore the author's expression.
function _hold_array_args(lhs, call::Expr, held::Dict{Symbol,Any})
    out = Any[call.args[1]]
    for a in call.args[2:end]
        if a isa Expr && a.head === :call && !isempty(a.args) &&
                a.args[1] isa Symbol && haskey(_PARAM_FAMILIES, a.args[1])
            push!(out, _hold_array_args(lhs, a, held))
        elseif call.args[1] in (:truncated, :restricted)
            push!(out, a)  # support bounds keep the shared scalar grammar
        elseif a isa Expr && a.head !== :parameters
            ph = Symbol("#arrayarg#", length(held) + 1)
            held[ph] = a
            push!(out, ph)
        else
            push!(out, a)
        end
    end
    return Expr(call.head, out...)
end

_is_dirichlet_call(rhs) =
    rhs isa Expr && rhs.head === :call && !isempty(rhs.args) &&
    rhs.args[1] === :Dirichlet

# Concentrations are ordinary values; the simplex dimension is structural
# and is inferred at bind from their shape.
function _lower_dirichlet(lhs, rhs)
    args = _plain_args(rhs, "`Dirichlet`")
    a = _mv_slice_args(Val(:Dirichlet), "parameter $lhs", args).arg1
    alpha = if a isa Expr && a.head === :vect && all(x -> x isa Real, a.args)
        Float64.(a.args)
    elseif _is_fill_call(a) && a.args[2] isa Real
        fill(Float64(a.args[2]), a.args[3])
    else
        a
    end
    size = _is_fill_call(alpha) ? alpha.args[3] : nothing
    return VectorParameter(lhs, :simplex_dirichlet, (arg1 = alpha,), size, lhs)
end

_is_ordered_call(rhs) =
    rhs isa Expr && rhs.head === :call && !isempty(rhs.args) &&
    rhs.args[1] === :Ordered

# `length(levels(g)) - k` (k a literal Int ≥ 0; bare `length(levels(g))`
# is k = 0) → `(g, k)`, else `nothing`.
function _levels_count(ex)
    k = 0
    if ex isa Expr && ex.head === :call && length(ex.args) == 3 &&
            ex.args[1] === :- && ex.args[3] isa Integer &&
            !(ex.args[3] isa Bool) && ex.args[3] >= 0
        k = Int(ex.args[3])
        ex = ex.args[2]
    end
    ex isa Expr && ex.head === :call && length(ex.args) == 2 &&
        ex.args[1] === :length && ex.args[2] isa Expr &&
        ex.args[2].head === :call && length(ex.args[2].args) == 2 &&
        ex.args[2].args[1] === :levels && ex.args[2].args[2] isa Symbol ||
        return nothing
    return (ex.args[2].args[2], k)
end

# A vector's literal extent or data-only Julia expression. The exact linked
# default retains `nothing` in the unbound IR; binding evaluates DataAPI
# levels before the response reads its support.
function _threshold_size(lhs::Symbol, n, response::Union{Nothing,Symbol})
    n isa Base.BitInteger && n >= 0 && return Int(n)
    # Retain the historical inferred representation for the exact linked
    # default; bind resolves it from DataAPI levels, including unused levels.
    response !== nothing && _levels_count(n) == (response, 1) && return nothing
    n isa Expr && return n
    _sfail("vector $lhs length must be a nonnegative integer or a data-only expression, got $(repr(n))")
end

_resolve_extent_levels(ex) = ex isa Expr ? Expr(ex.head,
    (i == 1 && ex.head === :call && a === :levels ? GlobalRef(DataAPI,:levels) :
        _resolve_extent_levels(a) for (i,a) in enumerate(ex.args))...) : ex

# Normal's constructor defaults are explicit distribution semantics.
# Its arguments may be named values, just as in scalar priors.
function _threshold_normal_args(lhs::Symbol, d, what::String)
    d isa Expr && d.head === :call && !isempty(d.args) &&
        d.args[1] === :Normal && length(d.args) <= 3 || _sfail(
            "$what $lhs: the element prior is `Normal(m, s)`, got $(repr(d))")
    m = length(d.args) >= 2 ? d.args[2] : 0.0
    s = length(d.args) == 3 ? d.args[3] : 1.0
    all(a -> a isa Symbol || (a isa Real && !(a isa Bool) && isfinite(a)),
        (m, s)) || _sfail("$what $lhs: element prior arguments must be " *
            "finite values or declared names")
    s isa Real && s <= 0 && _sfail("$what $lhs: element prior scale " *
        "must be positive, got $s")
    return m, s
end

function _ordered_element_prior(lhs, d, coefuse, ctx)
    if d isa Expr && d.head === :call && !isempty(d.args) && d.args[1] === :Normal
        m, s = _threshold_normal_args(lhs, d, "Ordered vector")
        return :ordered_normal, (arg1 = m, arg2 = s)
    end
    p = _lower_parameter(lhs, d, coefuse, ctx.matrices)
    family = Symbol(:ordered_, p.family)
    haskey(_VECTOR_ELEMENT_FAMILIES, family) && p.support_override === nothing ||
        _sfail("Ordered vector $lhs: the element prior must have real " *
            "support (Normal, Cauchy, Laplace, Logistic or StudentT)")
    return family, p.args
end

# `c ~ Ordered(Normal(m, s), n)`: n iid `Normal(m, s)` elements restricted
# to increasing order — the `ordered_constrain` transform plus its
# Jacobian, and the elementwise prior with no `log(n!)` normalizer (Stan's
# `ordered[n] c; c ~ normal(m, s)`, Bijectors.jl's `ordered`). `c` is a
# plain `Vector{Float64}` value (`c[1]`, `c[2] - c[1]`); a cumulative
# ordinal response consumes it as its cutpoints
# (`OrderedLogistic.(eta, Ref(c))`), which also admits the data-sized
# length `length(levels(y)) - 1`. Other data-only lengths evaluate at bind,
# whether or not an ordinal response consumes the vector.
function _lower_ordered(lhs, rhs, coefuse, ctx)
    haskey(coefuse, lhs) && _sfail("$lhs is a predictor coefficient and " *
        "cannot also be an Ordered vector")
    args = _plain_args(rhs, "`Ordered`")
    length(args) == 2 || _sfail("parameter $lhs: `Ordered` takes the " *
        "element distribution and the length " *
        "(`$lhs ~ Ordered(Normal(0, 1), length(levels(y)) - 1)`), got " *
        "$(length(args)) arguments")
    family, prior_args = _ordered_element_prior(lhs, args[1], coefuse, ctx)
    n = args[2]
    use = get(ctx.threshold_uses, lhs, nothing)
    size = _threshold_size(lhs, n, use === nothing ? nothing : use.response)
    if size isa Expr
        size = _resolve_module_calls(_resolve_extent_levels(size), ctx.mod, union(ctx.data,ctx.prior_names,keys(ctx.detmap)),
            "vector $lhs length")
    end
    return VectorParameter(lhs, family, prior_args, size, lhs)
end

# Sized ordinal thresholds keep the stated real-support element family
# with an identity transform. Cumulative and stopping responses read it.
function _lower_plain_thresholds(s, coefuse, ctx)
    lhs = s.lhs
    haskey(coefuse, lhs) && _sfail("$lhs is a predictor coefficient and " *
        "cannot also be thresholds")
    rhs = _undot_distribution(lhs, s.rhs)
    family, args = _ordered_element_prior(lhs, rhs, coefuse, ctx)
    family = Symbol(replace(String(family), "ordered_" => "vector_"))
    n = only(s.dims)
    size = n isa Int ? n : _threshold_size(lhs, n,
        ctx.threshold_uses[lhs].response)
    if size isa Expr
        size = _resolve_module_calls(_resolve_extent_levels(size),ctx.mod,union(ctx.data,ctx.prior_names,keys(ctx.detmap)),
            "vector $lhs length")
    end
    return VectorParameter(lhs, family, args, size, lhs)
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
    fam === :restricted && return _lower_bounded_param(lhs, rhs, coefuse,
        matrices, :restricted)
    fam in (:HalfNormal, :HalfCauchy) &&
        return _lower_half_param(lhs, rhs, fam, coefuse, matrices)
    fam === :positive && _sfail("parameter $lhs: write `HalfNormal(s)` " *
                                "or `truncated(Normal(0, s), 0, Inf)`")
    fam in (:weighted, :censored, :interval_censored) &&
        _sfail("`$fam` applies to responses only (parameter $lhs)")
    fam in (:normal, :cauchy, :exponential, :gamma, :lognormal, :beta,
        :inverse_gamma, :student_t, :laplace, :logistic, :uniform,
        :halfnormal, :halfcauchy) &&
        _sfail("parameter $lhs: use Distributions.jl constructors " *
               "(`Normal`, not `normal`)")
    haskey(_PARAM_FAMILIES, fam) ||
        _sfail("parameter $lhs: unknown distribution `$(repr(fam))` " *
               "(admitted: Normal, Cauchy, Exponential, Gamma, LogNormal, " *
               "Beta, InverseGamma, StudentT, TDist, Laplace, Logistic, Uniform, Weibull, " *
               "HalfNormal, HalfCauchy, Flat, " *
               "Dirichlet, LKJCholesky, truncated, restricted). If `$fam` is " *
               "meant as a submodel, define it with " *
               "`@rkppl $fam(args...) = begin ... end` and " *
               "make it visible in the lowering module (`mod=`).")
    any(a -> a isa Expr && a.head === :parameters, rhs.args[2:end]) &&
        _sfail("parameter $lhs: `$fam` does not take support keywords; " *
            "use `truncated($fam(args...), lo, hi)`, `HalfNormal(s)`, " *
            "or `HalfCauchy(s)` for a normalized positive prior")
    args = _distribution_args(fam, _plain_args(rhs, "`$fam`"))
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
    haskey(matrices, a) && _sfail("parameter $lhs argument $a is a design " *
                                  "matrix — prior arguments are scalar " *
                                  "(a literal or a parameter/assignment name)")
    return a
end

function _lower_flat(lhs, rhs)
    any(a -> a isa Expr && a.head === :parameters, rhs.args[2:end]) &&
        _sfail("parameter $lhs: `Flat()` has no support keywords; use " *
            "`Exponential(s)` for a positive prior, `Uniform(lo, hi)` for " *
            "a bounded uniform, or `Flat()` for an improper real prior")
    isempty(rhs.args[2:end]) || _sfail("parameter $lhs: `Flat()` takes no arguments")
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

# Truncated base heads → (arity, location position). Halves generalize to
# every symmetric family; upper-only and finite intervals stay Normal-only
# (the gates below).
const _TRUNCATED_BASES = Dict{Symbol,Tuple{Int,Int}}(
    :Normal => (2, 1), :Cauchy => (2, 1), :StudentT => (3, 2),
    :Laplace => (2, 1), :Logistic => (2, 1),
)

# Truncation keeps the base density and its authored bounds together. The
# layout intersects them with the base support; the prior subtracts the
# probability of precisely that interval (Distributions semantics).
function _lower_truncated_param(lhs, rhs, coefuse, matrices)
    return _lower_bounded_param(lhs, rhs, coefuse, matrices, :truncated)
end

# Both forms share exactly the same support/transform lowering. Only
# `truncated` subtracts the interval mass from the authored family density.
function _lower_bounded_param(lhs, rhs, coefuse, matrices, mode)
    args = _plain_args(rhs, "`$mode`")
    length(args) == 3 || _sfail("parameter $lhs: use " *
        "`$mode(D(args...), lo, hi)`")
    obj, lower, upper = args
    obj isa Expr && obj.head === :call || _sfail("parameter $lhs: " *
        "`$mode` wraps a distribution object, got $(repr(obj))")
    base = _lower_parameter(lhs, obj, coefuse, matrices)
    mode === :truncated && base.family === :flat && _sfail("parameter $lhs: `truncated` needs a " *
        "proper distribution; bound an improper prior with `Flat` instead")
    base.support_override in (nothing, :positive) || _sfail("parameter " *
        "$lhs: nested truncations are not yet supported")
    bound(a) = a isa Symbol && a !== :Inf ?
        _lower_param_arg(lhs, a, coefuse, matrices) : _truncation_bound(lhs, a)
    lo, hi = bound(lower), bound(upper)
    if lo isa Real && hi isa Real
        lo < hi || _sfail("parameter $lhs: truncation needs lower < upper, " *
            "got ($lo, $hi)")
    end
    # Both forms intersect a half's support with the nonnegative half-line.
    base.support_override === :positive &&
        (lo = lo isa Real ? max(0.0, lo) : Expr(:call, :max, 0.0, lo))
    tag = mode === :restricted && base.support_override === :positive ?
        :restricted_half : mode
    return SampledParameter(lhs, base.family, base.args,
        (tag, lo, hi), lhs)
end

function _lower_assignment(nm, rhs, coefuse)
    rhs isa Expr || rhs isa Symbol || rhs isa Real ||
        _sfail("assignment $nm must be an expression, name or literal, " *
               "got $(repr(rhs))")
    return AssignmentSpec(nm, _argument_value_calls(rhs), nm)
end

# Vector shape rules belong to validation (contract v3); lowering checks
# coefficient discipline and types the node (vocabulary and Julia-shape
# screens ran at the definition pre-pass).
function _lower_vector_assignment(nm, rhs, coefuse)
    rhs isa Expr || rhs isa Symbol ||
        _sfail("derived column $nm must be an expression or column alias, " *
               "got $(repr(rhs))")
    return VectorAssignmentSpec(nm, _argument_value_calls(rhs), nm)
end
