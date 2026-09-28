# Parse hygiene (runs FIRST in runtests.jl): every `.jl` file under
# `test/` must whole-parse with zero `:error`/`:incomplete` nodes. A file
# that fails this aborts the suite at its own `include` with a `LoadError`
# AFTER all earlier legs ran — on 2026-09-28 a missing `end` in
# `test_spline.jl` (dropped by `b83fcbcf`'s HLO-leg insertion) surfaced
# only after the minute-long XLA leg and was misattributed to
# Reactant/XLA for a full investigation cycle
# (snag `spline-xla-leg-t-d1e7f6ca`). Failing fast here names the file.
function _parse_hygiene_bad_nodes(code::String)
    n = 0
    function walk(x)
        if x isa Expr
            (x.head === :error || x.head === :incomplete) && (n += 1)
            foreach(walk, x.args)
        end
    end
    walk(Meta.parseall(code))
    return n
end

@testset "test-file parse hygiene" begin
    bad = String[]
    total = 0
    for (root, _, files) in walkdir(@__DIR__)
        for f in sort(files)
            endswith(f, ".jl") || continue
            total += 1
            p = joinpath(root, f)
            _parse_hygiene_bad_nodes(read(p, String)) > 0 && push!(bad, p)
        end
    end
    @test total > 0
    @test isempty(bad)
end
