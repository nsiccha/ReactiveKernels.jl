"""Check the source contracts shared by the local build and CI."""
function lint_docs(pages; root=@__DIR__)
    first.(pages) == ["Start", "Guides", "Examples", "Reference"] ||
        error("navigation must contain Start, Guides, Examples, Reference in order")
    paths = [last(page) for (_, group) in pages for page in group]
    length(paths) == length(unique(paths)) || error("a page belongs to multiple groups")
    actual = sort(filter(p -> endswith(p, ".md"), readdir(joinpath(root, "src"))))
    sort(paths) == actual || error("page inventory differs from docs/src")
    readme = read(joinpath(root, "..", "README.md"), String)
    landing = read(joinpath(root, "src", "index.md"), String)
    readme_example = match(r"(?ms)^\x60\x60\x60julia\n(.*?)^\x60\x60\x60", readme).captures[1]
    landing_example = match(r"(?ms)^\x60\x60\x60@example start\n(.*?)^\x60\x60\x60", landing).captures[1]
    readme_example == landing_example || error("README example must match the executed landing example")
    banned = r"\b(seamless|robust|powerful|leverage|delve|crucial|comprehensive|unlock|simply)\b|it's worth noting"i
    # Exported docstrings are rendered into the three reference pages too.
    source_roots = (joinpath(root, "..", "src"),
        joinpath(root, "..", "packages", "ReactiveKernelsPPL", "src"),
        joinpath(root, "..", "packages", "ReactiveKernelsDistributionKernels", "src"))
    for directory in source_roots, (path, _, files) in walkdir(directory), file in files
        endswith(file, ".jl") || continue
        source = read(joinpath(path, file), String)
        for doc in eachmatch(r"\"\"\"(.*?)\"\"\""s, source)
            occursin(r"(?m)^\x60\x60\x60(?:julia|julia-repl|python|javascript|js)\b", doc.captures[1]) &&
                error("$file contains a static runnable docstring fence")
        end
    end
    for (group, entries) in pages, (_, name) in entries
        body = read(joinpath(root, "src", name), String)
        occursin(r"(?m)^\x60\x60\x60(?:julia|python|javascript|js)\s*$", body) &&
            error("$name contains a static runnable code fence")
        prose = replace(body, r"(?ms)^\x60\x60\x60.*?^\x60\x60\x60\s*$" => "")
        occursin(banned, prose) && error("$name contains a banned prose term")
        words = length(split(prose))
        group == "Guides" && words >= 1500 && error("$name exceeds the guide word limit")
        if group == "Examples"
            800 <= words <= 1500 || error("$name has $words prose words; expected 800–1500")
            occursin("\x60\x60\x60@example", body) || error("$name has no executed example")
            occursin("Tutorials.figure(", body) || error("$name has no generated figure")
            occursin("Tutorials.receipt(", body) || error("$name has no execution receipt")
            occursin("<summary>Reproduce</summary>", body) || error("$name has no Reproduce footer")
        end
        for m in eachmatch(r"\]\(([^)#]+\.md)(?:#[^)]*)?\)", body)
            target = m.captures[1]
            occursin("://", target) && continue
            isfile(normpath(joinpath(root, "src", dirname(name), target))) ||
                error("$name links to missing page $target")
        end
        println("docs lint: ", group, " / ", name, " / ", words, " prose words")
    end
    return paths
end
