"""
    _evaluate_ppl_source(source, owner; bindings=())

Evaluate one PPL walkthrough source string in a fresh module. `owner` is bound
under its module name and `bindings` exposes the example's public data/types,
so the same source bytes can be executed by tests and displayed/executed by the
documentation build without a second model body.
"""
function _evaluate_ppl_source(source::AbstractString, owner::Module;
                              bindings = ())
    # Match the docs renderer's framing rule exactly: triple-quoted authorities
    # carry a terminal newline, while the displayed/executed panel bytes do not.
    displayed = strip(String(source), '\n')
    # Keep package loading in the source owner's dependency context. A top-level
    # anonymous module instead resolves `using` against the caller's active
    # project, forcing runners to repeat dependencies named by the source.
    sandbox_name = gensym(Symbol(nameof(owner), :Source))
    Core.eval(owner, Expr(:module, true, sandbox_name, Expr(:block)))
    sandbox = getfield(owner, sandbox_name)
    Core.eval(sandbox, :(using ReactiveKernels))
    owner_name = nameof(owner)
    Core.eval(sandbox, :(const $(owner_name) = $owner))

    for name in bindings
        name isa Symbol || throw(ArgumentError("PPL source binding must be a Symbol"))
        isdefined(owner, name) || error("$(nameof(owner)) does not define source binding $name")
        value = getfield(owner, name)
        Core.eval(sandbox, :(const $(name) = $value))
    end

    parsed = Meta.parseall(displayed; filename = "$(nameof(owner))-docs-source.jl")
    expressions = parsed.head === :toplevel ? parsed.args : Any[parsed]
    for expression in expressions
        expression isa LineNumberNode && continue
        Core.eval(sandbox, expression)
    end

    artifact = Core.eval(sandbox, :docs_example)
    artifact.model isa KernelSpec || error(
        "$(nameof(owner)) docs source did not provide its KernelSpec as `model`",
    )
    artifact.kernel isa PreparedKernel || error(
        "$(nameof(owner)) docs source did not provide a PreparedKernel",
    )
    merge(artifact, (; source = displayed, sandbox))
end
