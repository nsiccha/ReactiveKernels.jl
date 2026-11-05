# Compact cell and compact partition samplers are differentiated by ordinary
# static Reverse. Their overlay reads either a constant base entry or an active
# one; registered Enzyme 0.13.216 refuses that read (`EnzymeRuntimeActivityError`,
# "Mismatched activity"), and the Enzyme `dogfood` line repairs it (Enzyme snag
# `static-reverse-s-f9483e9c`). `_static_enzyme(f)` runs `f()` and returns its
# value, or `nothing` after recording that refusal as broken; any other error
# propagates. Dogfood validation sets RK_TEST_REQUIRE_STATIC_ENZYME=1, which
# makes the refusal fatal. Drop this gate once a registered Enzyme carries the
# repair.
function _static_enzyme(f)
    try
        f()
    catch err
        err isa Enzyme.Compiler.EnzymeRuntimeActivityError || rethrow()
        get(ENV, "RK_TEST_REQUIRE_STATIC_ENZYME", "0") == "1" && rethrow()
        @test occursin("Mismatched activity", sprint(showerror, err))
        @test_broken false
        nothing
    end
end
