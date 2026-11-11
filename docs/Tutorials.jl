module Tutorials

using Base64, Markdown, SHA, TOML
using ReactiveKernels
using AlgebraOfVega
using Documenter, DocumenterVitepress

const ROOT = @__DIR__
const RECEIPTS = joinpath(ROOT, "build", "receipts")

struct BuildAssets <: Documenter.Plugin end

function DocumenterVitepress.vitepress_assets(::BuildAssets)
    # The writer moves executed outputs into .documenter before this hook.
    generated = joinpath(ROOT, "build", ".documenter", "generated-assets")
    mkpath(generated)
    cp(AlgebraOfVega.vega_vendor_dir(), joinpath(generated, "aov"); force=true)
    cp(joinpath(ROOT, "build", ".documenter", "receipts"),
        joinpath(generated, "receipts"); force=true)
    return [generated]
end

struct HTMLResult
    html::String
end

# Carry UTF-8 HTML through the renderer without reinterpreting script escapes.
# The existing script directive decodes this payload before activating scripts.
Base.show(io::IO, ::MIME"text/html", result::HTMLResult) =
    print(io, "<div data-rk-html=\"", base64encode(result.html), "\"></div><script></script>")

html(content) = HTMLResult(sprint(show, MIME"text/html"(), content; context=:limit => false))
escape_html(s) = replace(string(s), '&' => "&amp;", '<' => "&lt;", '>' => "&gt;", '"' => "&quot;")
render_nodes(nodes) = join(sprint(show, MIME"text/html"(), node) for node in nodes)

function figure(spec, id, caption)
    head = render_nodes(figure_assets())
    themed = spec * config(width=520, height=300, theme=:host,
        config=(; axis=(; labelFontSize=13, titleFontSize=14)))
    plot = sprint(show, MIME"text/html"(), vdraw(themed,
        scales(Color=(; palette=["#2a78d6", "#eb6834"])); id=String(id)))
    return HTMLResult(head * "<figure class=\"rk-tutorial-figure\">" * plot *
        "<figcaption>" * escape_html(caption) * "</figcaption></figure>")
end

# AoV owns its runtime and theme behavior. The build copies these vendor assets
# through the public vega_head/vega_vendor_dir surface, never into tracked files.
# GitHub Pages filters underscore-prefixed directories during Jekyll processing.
figure_assets() = AlgebraOfVega.vega_head(; source=:vendor, base="./aov", runtime=:linked)

function generated(kernel)
    print(Base.remove_linenums!(deepcopy(code_expr(kernel))))
    return nothing
end

function export_index(mod; undocumented=false)
    root = dirname(ROOT)
    revision = strip(read(`git -C $root rev-parse HEAD`, String))
    source_root = joinpath(pkgdir(mod), "src")
    sources = [(joinpath(directory, file), readlines(joinpath(directory, file)))
        for (directory, _, files) in walkdir(source_root) for file in sort(files)
        if endswith(file, ".jl")]
    rows = ["| Name | Kind | Source |", "| --- | --- | --- |"]
    for name in sort(names(mod))
        name == nameof(mod) && continue
        Base.isexported(mod, name) || continue
        undocumented && haskey(Base.Docs.meta(mod), Base.Docs.Binding(mod, name)) && continue
        value = getfield(mod, name)
        kind = value isa AbstractString ? "Source recipe" :
            value isa Function ? "Function" : value isa Type ? "Type" :
            value isa Module ? "Module" : "Kernel or value"
        bare = lstrip(string(name), '@')
        starts = (bare * "(", bare * "{", "const " * bare * " ",
            "function " * bare * "(", "struct " * bare, "macro " * bare,
            "@kernel " * bare * "(", "@kernel " * bare * " ", "module " * bare)
        found = nothing
        for (file, lines) in sources
            line = findfirst(text -> any(prefix -> startswith(strip(text), prefix), starts), lines)
            if line !== nothing
                found = (file, line)
                break
            end
        end
        target = found === nothing ?
            "https://github.com/nsiccha/ReactiveKernels.jl/tree/$revision/$(relpath(source_root, root))" :
            "https://github.com/nsiccha/ReactiveKernels.jl/blob/$revision/$(relpath(found[1], root))#L$(found[2])"
        label = found === nothing ? "module source" : "definition"
        push!(rows, "| `$(name)` | $kind | [$label]($target) |")
    end
    return Markdown.parse(join(rows, "\n"))
end

function receipt(slug; checks)
    mkpath(RECEIPTS)
    page = joinpath(ROOT, "src", slug * ".md")
    record = Dict(
        "page" => slug,
        "source_sha256" => bytes2hex(sha256(read(page))),
        "julia_version" => string(VERSION),
        "reactivekernels_version" => string(Base.pkgversion(ReactiveKernels)),
        "checks" => Dict(string(k) => v for (k, v) in pairs(checks)),
    )
    open(joinpath(RECEIPTS, slug * ".toml"), "w") do io
        TOML.print(io, record)
    end
    return nothing
end

end
