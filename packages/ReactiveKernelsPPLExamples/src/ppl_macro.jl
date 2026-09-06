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

- Parameters: **scalar**, each taking one packed-unconstrained coordinate, with
  support inferred from the sampling distribution and the matching transform +
  log-Jacobian authored in both directions — real (`normal` / `cauchy` /
  `laplace`, identity), positive (`exponential` / `gamma` / `lognormal`,
  log/exp), unit (`beta`, logit/logistic).
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

Vector parameters (prior `plate`), explicit constraint overrides / half /
bounded / truncated families, transforms via the `ReactiveKernels:ppl:bijectors`
objects (currently authored inline, matching the hand-written examples), and
user PPL-AST transformations are follow-up increments (todo
`2026-09-06T15-02-44-929-0mw9lhd`).
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
    support::Symbol  # :real (first cut)
end

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
            lhs isa Symbol ||
                error("@ppl: `~` left-hand side must be a name, got $(lhs)")
            (_call_head(rhs) isa Symbol) ||
                error("@ppl: `~` right-hand side must be a distribution call, got $(rhs)")
            if lhs in datanames
                push!(obs, _Obs(lhs, rhs))
            else
                lhs in seen && error("@ppl: parameter $(lhs) declared twice")
                push!(seen, lhs)
                dist = _call_head(rhs)
                haskey(_SUPPORT, dist) || error(
                    "@ppl (first cut): parameter $(lhs) ~ $(dist)(…) — supported " *
                    "parameter distributions are $(sort(collect(keys(_SUPPORT)))); " *
                    "explicit constraint overrides and other families are a " *
                    "follow-up increment.")
                push!(params, _Param(lhs, rhs, _SUPPORT[dist]))
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
    for (i, p) in enumerate(params)
        coord = :( sum(view(unconstrained, $i:$i)) )
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
    inv_lhs = Expr(:tuple, (:( $(n)::Float64 ) for n in pnames)...)
    inv_rhs = Expr(:tuple, (:( parameters.$(n) ) for n in pnames)...)
    push!(stmts, Expr(:(=), inv_lhs, inv_rhs))

    # 4. Prior: sum of each parameter's log density on its constrained value.
    prior_terms = [ :( $(p.dist).logpdf($(p.name)) ) for p in params ]
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
