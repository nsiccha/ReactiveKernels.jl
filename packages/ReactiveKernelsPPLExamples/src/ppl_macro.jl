"""
    PPLMacro

An RK-native, StanBlocks-*like* declarative PPL front-end (first cut).

`@ppl` owns a small PPL AST plus a posterior-mode parameter/observation analysis
and lowers a declarative `~` model into an ordinary `ReactiveKernels.@kernel`
have→want graph that exposes the canonical PPL workflow node set (see
`PPLWorkflow` on `kb-impl/ReactiveKernels-ppl-benchmark`, commit `59dd8bb`). It
is built entirely on ReactiveKernels' PUBLIC surface (`@kernel` / `prepare` /
`plan` / `plate`) plus the reusable distribution-kernel objects — nothing lives
in ReactiveKernels core.

Decisions: build GO `2026-09-06T14-20-02-559-1r5p4j1`; scope MINIMAL
`2026-09-06T14-20-27-206-0wqzaq4`.

# First-cut scope

- Scalar parameters take one packed-unconstrained coordinate, with support
  inferred from the sampling distribution and the matching transform +
  log-Jacobian authored in both directions — real (`normal` / `cauchy` /
  `laplace`, identity), positive (`exponential` / `gamma` / `lognormal`,
  log/exp), unit (`beta`, logit/logistic).
- **Vector parameters** — `name[size] ~ dist(…)` (real support) — take a packed
  slice of `size` coordinates (`size` a data argument or literal; the packed
  layout tracks a running offset), and their prior is the summed per-element
  authored `plate`.
- An explicit `name::constraint` override (`positive` / `unit` / `real`). A
  positive constraint on a real-support family is a **half distribution**
  (lower-truncated at 0) whose prior adds `-log(1 - cdf(0))` via the family's own
  `.cdf` — e.g. `sigma::positive ~ normal(0, 5)` (half-Normal),
  `tau::positive ~ cauchy(0, 5)` (half-Cauchy).
- `parameters` has two producers (constrain-only, and joint with `log_jacobian`
  so a params+Jacobian query shares the transform) plus named-latent inverse
  edges, so a packed, named-latent, or `parameters` HAVE boundary all route
  without recomputation or a `log(exp(x))` round trip.
- One or more **observation** streams: a `~` whose LHS is a signature (data)
  argument, lowered to an authored `plate` — the atomic referenced values are
  threaded in and the argument expressions rebuilt per cell, so a linear
  predictor is computed element-wise — summed to the buffer-free `likelihood`.
- Emits exactly the canonical nodes: `parameters`, `log_jacobian`, `prior`,
  `pointwise`, `likelihood`, `unconstrained_prior`, `constrained_logdensity`,
  `posterior`. Queried the usual way, e.g.
  `prepare(model; have = (:unconstrained, data...), want = :posterior)`.

Constrained (positive/unit) vector parameters, general bounded/interval
truncation, transforms via the `ReactiveKernels:ppl:bijectors` objects (currently
authored inline, matching the hand-written examples), and user PPL-AST
transformations are follow-up increments (todo
`2026-09-06T15-02-44-929-0mw9lhd`).

Reproduces the hand-written `beta_binomial`, `poisson_gamma`, `linear_regression`,
and `eight_schools` example densities exactly (density parity).
"""
module PPLMacro

export @ppl

# Canonical node names — mirror of `PPLWorkflow.PPL_NODES`
# (`packages/ReactiveKernelsPPLExamples/src/ppl_workflow.jl` @ `59dd8bb`, not yet
# on main). Once it lands, `import ..PPLWorkflow` and assert equality here.
const PPL_NODE_NAMES = (:parameters, :log_jacobian, :prior, :pointwise,
                        :likelihood, :unconstrained_prior,
                        :constrained_logdensity, :posterior)

# Parameter-support inferred from the sampling distribution, driving the
# unconstrained↔constrained transform + log-Jacobian:
#   :real     — identity (the packed coordinate IS the value), Jacobian 0
#   :positive — log/exp,   Jacobian = log_x
#   :unit     — logit/logistic, Jacobian = log(x) + log1p(-x)
# Explicit constraint overrides (e.g. a half-Cauchy: Cauchy prior on positive
# support) and bounded/other families are a follow-up increment.
const _SUPPORT = Dict{Symbol,Symbol}(
    :normal => :real, :cauchy => :real, :laplace => :real,
    :exponential => :positive, :gamma => :positive, :lognormal => :positive,
    :beta => :unit,
)

struct _Param
    name::Symbol
    dist::Any        # the dist-call Expr, e.g. :(normal(0.0, 5.0))
    support::Symbol  # EFFECTIVE support (transform): :real / :positive / :unit
    size::Any        # `nothing` for a scalar; a size expression for a vector
    truncate::Bool   # a constraint restricting the family's natural support (a
                     # half distribution): add -log(1 - cdf(0)) to the prior
end

# Running packed-offset arithmetic that stays a literal while every preceding
# parameter is scalar and becomes an expression once a runtime-sized vector
# parameter shifts the layout.
_offset_add(off::Int, n::Int) = off + n
_offset_add(off, n) = :( $(off) + $(n) )

struct _Obs
    data::Symbol     # the observed (data) argument name
    dist::Any        # the dist-call Expr referencing parameters / data
end

_is_line(x) = x isa LineNumberNode

# `arg` may be a bare Symbol or `name::Type`; return the bound name.
_arg_name(a::Symbol) = a
function _arg_name(a::Expr)
    a.head === :(::) || error("@ppl: unsupported argument form $(a)")
    a.args[1]::Symbol
end

_call_head(rhs) = (rhs isa Expr && rhs.head === :call) ? rhs.args[1] : nothing
_call_args(rhs) = rhs.args[2:end]

"""
    _parse(def) -> (name, dataargs, params, obs, passthrough)

Parse `@ppl name(dataargs...) = begin body end` into its PPL AST: the model
name, the declared data arguments, the ordered scalar parameters, the
observation statements, and any deterministic pass-through assignments.
"""
function _parse(def)
    (def isa Expr && def.head === :(=) && def.args[1] isa Expr &&
        def.args[1].head === :call) ||
        error("@ppl expects `name(args...) = begin … end`, got $(def)")
    call = def.args[1]
    name = call.args[1]::Symbol
    dataargs = call.args[2:end]
    datanames = Symbol[_arg_name(a) for a in dataargs]
    body = def.args[2]
    (body isa Expr && body.head === :block) ||
        error("@ppl model body must be a `begin … end` block")

    params = _Param[]
    obs = _Obs[]
    passthrough = Any[]
    seen = Set{Symbol}()
    for stmt in body.args
        _is_line(stmt) && continue
        if stmt isa Expr && stmt.head === :call && stmt.args[1] === :~
            lhs = stmt.args[2]
            rhs = stmt.args[3]
            # LHS is a bare name (scalar), `name[size]` (vector parameter), or
            # `name::constraint` (an explicit support override, e.g. a half dist).
            lname, lsize, loverride = nothing, nothing, nothing
            if lhs isa Symbol
                lname = lhs
            elseif lhs isa Expr && lhs.head === :ref && length(lhs.args) == 2 &&
                    lhs.args[1] isa Symbol
                lname, lsize = lhs.args[1], lhs.args[2]
            elseif lhs isa Expr && lhs.head === :(::) && length(lhs.args) == 2 &&
                    lhs.args[1] isa Symbol && lhs.args[2] isa Symbol
                lname, loverride = lhs.args[1], lhs.args[2]
            else
                error("@ppl: `~` left-hand side must be `name`, `name[size]`, or " *
                      "`name::constraint`, got $(lhs)")
            end
            (_call_head(rhs) isa Symbol) ||
                error("@ppl: `~` right-hand side must be a distribution call, got $(rhs)")
            if lname in datanames
                (lsize === nothing && loverride === nothing) ||
                    error("@ppl: observation `$(lname)` cannot carry a size or constraint annotation")
                push!(obs, _Obs(lname, rhs))
            else
                lname in seen && error("@ppl: parameter $(lname) declared twice")
                push!(seen, lname)
                dist = _call_head(rhs)
                haskey(_SUPPORT, dist) || error(
                    "@ppl (first cut): parameter $(lname) ~ $(dist)(…) — supported " *
                    "parameter distributions are $(sort(collect(keys(_SUPPORT)))); " *
                    "other families are a follow-up increment.")
                natural = _SUPPORT[dist]
                effective = loverride === nothing ? natural : loverride
                truncate = false
                if loverride !== nothing
                    loverride in (:real, :positive, :unit) || error(
                        "@ppl: unknown constraint `$(loverride)` on $(lname); use " *
                        "real / positive / unit")
                    if effective === natural
                        # redundant, explicit — no truncation.
                    elseif effective === :positive && natural === :real
                        truncate = true   # a half distribution (lower-truncated at 0)
                    else
                        error("@ppl (first cut): constraint `$(effective)` on a " *
                              "$(natural)-support `$(dist)` is not supported yet — only a " *
                              "positive constraint on a real-support family (a half " *
                              "distribution) or a redundant matching constraint.")
                    end
                    lsize === nothing || error(
                        "@ppl (first cut): a constrained vector parameter " *
                        "($(lname)) is a follow-up increment.")
                end
                (lsize === nothing || effective === :real) || error(
                    "@ppl (first cut): vector parameter $(lname)[$(lsize)] ~ $(dist)(…) — " *
                    "only real-support vector parameters are supported yet.")
                push!(params, _Param(lname, rhs, effective, lsize, truncate))
            end
        elseif stmt isa Expr && stmt.head === :(=)
            push!(passthrough, stmt)
        else
            error("@ppl: unsupported statement $(stmt); use `lhs ~ dist(…)`, " *
                  "`lhs = expr`, inside a model body.")
        end
    end
    isempty(params) && error("@ppl: model declares no parameters")
    isempty(obs) && error("@ppl (first cut): posterior-mode model needs at least " *
                          "one observation (`<data> ~ dist(…)`).")
    (name, dataargs, params, obs, passthrough)
end

# Collect, in first-use order, the model symbols (parameters / data / earlier
# deterministic bindings) an expression references — the atomic values that must
# be threaded into a plate cell so the body computes per element.
function _collect_refs!(refs::Vector{Symbol}, ex, modelsyms::Set{Symbol})
    if ex isa Symbol
        (ex in modelsyms && !(ex in refs)) && push!(refs, ex)
    elseif ex isa Expr
        for a in ex.args
            _collect_refs!(refs, a, modelsyms)
        end
    end
    refs
end

# Replace each model symbol in `ex` with its per-cell do-variable.
_subst(ex::Symbol, sub) = get(sub, ex, ex)
_subst(ex::Expr, sub) = Expr(ex.head, (_subst(a, sub) for a in ex.args)...)
_subst(ex, sub) = ex

"""
Lower one observation `data ~ dist(argexprs...)` into an authored `plate`
producing its pointwise vector. The observed data plus every atomic model value
the distribution arguments reference are passed as plate positional inputs
(Julia broadcast handles scalar-vs-array), and the argument EXPRESSIONS are
rebuilt per cell over the sliced do-variables — so a linear predictor such as
`alpha + beta * x` is computed element-wise, never as whole-vector arithmetic.

The do-block parameters are plain (non-gensym) names local to the block, and the
body block carries a `LineNumberNode`: `@kernel`'s plate expander types the cell
result from the parsed-source block shape, so a raw `Expr(:block, body)` (no
LNN) would leave the cell `:__return__` typed `Any` and mismatch the nested
endpoint's `Float64` output.
"""
function _obs_plate(o::_Obs, modelsyms::Set{Symbol})
    dist = _call_head(o.dist)
    distargs = _call_args(o.dist)
    refs = Symbol[]
    for a in distargs
        _collect_refs!(refs, a, modelsyms)
    end
    plate_inputs = Any[o.data; refs]
    dovars = [Symbol(:_ppl_c, i) for i in eachindex(plate_inputs)]
    obsvar = dovars[1]
    sub = Dict{Symbol,Symbol}(refs[i] => dovars[i + 1] for i in eachindex(refs))
    cell_args = [_subst(a, sub) for a in distargs]
    body = :( $(dist)($(cell_args...)).logpdf($(obsvar)) )
    plate_call = Expr(:call, :plate, plate_inputs...)
    lambda = Expr(:(->), Expr(:tuple, dovars...),
                  Expr(:block, LineNumberNode(0, Symbol("@ppl")), body))
    Expr(:do, plate_call, lambda)
end

function _lower(name, dataargs, params, obs, passthrough)
    stmts = Any[]

    # Atomic model symbols an observation cell may reference: parameters, data
    # arguments, and deterministic pass-through bindings.
    modelsyms = Set{Symbol}()
    for p in params
        push!(modelsyms, p.name)
    end
    for a in dataargs
        push!(modelsyms, _arg_name(a))
    end
    for s in passthrough
        (s isa Expr && s.head === :(=) && s.args[1] isa Symbol) &&
            push!(modelsyms, s.args[1])
    end

    # 1. Packed-unconstrained split + per-parameter constrained transform.
    #    Each parameter takes ONE packed coordinate: its UNCONSTRAINED value.
    #    `sum(view(…))` keeps the evaluator allocation-free / Reactant-friendly.
    #    Real → identity; positive → log/exp; unit → logit/logistic. BOTH
    #    transform directions are authored (forward from the unconstrained
    #    coordinate, inverse from the constrained value) so HAVE authority routes
    #    a packed, a named-latent, or a `parameters` query without recomputation
    #    or a `log(exp(x))` round trip.
    jac_terms = Any[]
    off = 0  # running packed offset BEFORE the current parameter
    for p in params
        lo = _offset_add(off, 1)
        if p.size !== nothing
            # Real-support vector parameter (identity transform): one packed
            # slice of `size` coordinates.
            hi = _offset_add(off, p.size)
            push!(stmts, :( $(p.name)::AbstractVector{Float64} =
                view(unconstrained, $(lo):$(hi)) ))
            off = hi
            continue
        end
        coord = :( sum(view(unconstrained, $(lo):$(lo))) )
        if p.support === :real
            push!(stmts, :( $(p.name)::Float64 = $(coord) ))
        elseif p.support === :positive
            u = Symbol(:_ppl_log_, p.name)
            push!(stmts, :( $(u)::Float64 = $(coord) ))
            push!(stmts, :( $(p.name)::Float64 = exp($(u)) ))
            push!(stmts, :( $(u)::Float64 = log($(p.name)) ))
            push!(jac_terms, u)
        elseif p.support === :unit
            u = Symbol(:_ppl_logit_, p.name)
            push!(stmts, :( $(u)::Float64 = $(coord) ))
            push!(stmts, :( $(p.name)::Float64 = 1 / (1 + exp(-$(u))) ))
            push!(stmts, :( $(u)::Float64 = log($(p.name)) - log1p(-$(p.name)) ))
            push!(jac_terms, :( log($(p.name)) + log1p(-$(p.name)) ))
        else
            error("@ppl: unhandled parameter support $(p.support)")
        end
        off = _offset_add(off, 1)
    end

    # 2. Deterministic pass-through assignments (transforms/covariate prep).
    append!(stmts, passthrough)

    # 3. Constrained parameters (two producers: constrain-only, and joint with
    #    the log-Jacobian so a params+Jacobian query shares the transform), the
    #    log-Jacobian, and the named-latent inverse edges.
    pnames = [p.name for p in params]
    jac_expr = isempty(jac_terms) ? :(0.0) :
               foldl((a, b) -> :( $a + $b ), jac_terms)
    push!(stmts, :( parameters = (; $(pnames...)) ))
    push!(stmts, :( log_jacobian::Float64 = $(jac_expr) ))
    push!(stmts, Expr(:(=),
        Expr(:tuple, :parameters, :( log_jacobian::Float64 )),
        Expr(:tuple, :( (; $(pnames...)) ), jac_expr)))
    _pann(p) = p.size === nothing ? :( $(p.name)::Float64 ) :
               :( $(p.name)::AbstractVector{Float64} )
    inv_lhs = Expr(:tuple, (_pann(p) for p in params)...)
    inv_rhs = Expr(:tuple, (:( parameters.$(p.name) ) for p in params)...)
    push!(stmts, Expr(:(=), inv_lhs, inv_rhs))

    # 4. Prior: sum of each parameter's log density on its constrained value.
    #    A scalar contributes `dist.logpdf(p)`; a vector contributes the summed
    #    per-element prior over an authored `plate` (same cell lowering as an
    #    observation, with the parameter itself in the sliced position). The plate
    #    is bound to its own variable first — a constructed-endpoint plate must be
    #    a whole recipe RHS, not a sub-expression of `sum(…)`.
    # A truncated (half) prior renormalizes by -log(1 - cdf(0)). The truncation
    # point 0 must reach the `.cdf` endpoint as a NAMED caller port (an endpoint's
    # explicit argument cannot be a bare literal), so bind it once.
    if any(p -> p.truncate, params)
        push!(stmts, :( _ppl_zero::Float64 = 0.0 ))
    end
    prior_terms = Any[]
    for p in params
        if p.size === nothing
            term = :( $(p.dist).logpdf($(p.name)) )
            if p.truncate
                # Half distribution (lower-truncated at 0): renormalize by
                # -log P(X > 0) = -log(1 - cdf(0)). For a symmetric-at-0 family
                # this is +log(2), matching the hand-written half-Normal /
                # half-Cauchy examples; the cdf form also covers non-symmetric
                # lower-truncation.
                term = :( $(term) - log(1 - $(p.dist).cdf(_ppl_zero)) )
            end
            push!(prior_terms, term)
        else
            pv = Symbol(:_ppl_prior_, p.name)
            push!(stmts, :( $(pv) = $(_obs_plate(_Obs(p.name, p.dist), modelsyms)) ))
            push!(prior_terms, :( sum($(pv)) ))
        end
    end
    prior_expr = length(prior_terms) == 1 ? prior_terms[1] :
                 foldl((a, b) -> :( $a + $b ), prior_terms)
    push!(stmts, :( prior::Float64 = $(prior_expr) ))

    # 5. Pointwise + fused likelihood over the observation stream(s).
    if length(obs) == 1
        push!(stmts, :( pointwise = $(_obs_plate(obs[1], modelsyms)) ))
        push!(stmts, :( likelihood::Float64 = sum(pointwise) ))
    else
        # Multiple streams: name each pointwise vector, sum each, add up.
        per = Symbol[]
        for (k, o) in enumerate(obs)
            pv = Symbol(:pointwise_, k)
            push!(per, pv)
            push!(stmts, :( $(pv) = $(_obs_plate(o, modelsyms)) ))
        end
        push!(stmts, :( pointwise = ($(per...),) ))
        like_terms = [ :( sum($(pv)) ) for pv in per ]
        like_expr = foldl((a, b) -> :( $a + $b ), like_terms)
        push!(stmts, :( likelihood::Float64 = $(like_expr) ))
    end

    # 6. The canonical workflow density nodes.
    push!(stmts, :( unconstrained_prior::Float64 = prior + log_jacobian ))
    push!(stmts, :( constrained_logdensity::Float64 = prior + likelihood ))
    push!(stmts, :( posterior::Float64 = prior + likelihood + log_jacobian ))
    push!(stmts, :( return posterior ))

    sig = Expr(:call, name, :( unconstrained::Vector{Float64} ), dataargs...)
    Expr(:(=), sig, Expr(:block, stmts...))
end

"""
    @ppl name(data...) = begin … end

Declare an sb-like PPL model and lower it to a ReactiveKernels `KernelSpec`
exposing the canonical PPL workflow nodes. See the module docstring for the
first-cut scope. The distribution objects used in the body (`normal`, …) and
`ReactiveKernels` must be in scope at the use site, exactly as the hand-authored
PPL examples require.
"""
macro ppl(def)
    (name, dataargs, params, obs, passthrough) = _parse(def)
    kernel_def = _lower(name, dataargs, params, obs, passthrough)
    esc(Expr(:macrocall, Symbol("@kernel"), __source__, kernel_def))
end

end # module PPLMacro
