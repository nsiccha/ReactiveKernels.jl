# Select the existing ragged logit integration test without running unrelated
# probability-law models. Preserve its definitions, model and every assertion.
using ReactiveKernelsPPL
const selected_logit_testsets = Ref(0)
function select_logit_testset(ex)
    ex isa Expr && ex.head === :module || return ex
    body = ex.args[end]
    filter!(body.args) do statement
        statement isa Expr && statement.head === :macrocall &&
            statement.args[1] === Symbol("@testset") || return true
        keep = statement.args[3] ==
            "logit-scale whole-event mixture with a structural-zero component"
        keep && (selected_logit_testsets[] += 1)
        keep
    end
    ex
end
Base.include(select_logit_testset, Main,
    joinpath(pkgdir(ReactiveKernelsPPL), "test", "test_whole_event_mixture.jl"))
@assert selected_logit_testsets[] == 1
