# Ordinary-Reverse lifetime diagnostic for the provisional joint-staging path.
# Compare unchanged staging with the joint prototype using the same pure graph.
# The alias arm records the known wrong gradient; ordinary and copying controls
# are assertions. Exit zero means the diagnostic completed, not that every arm
# returned a correct gradient. No production adapter patch is included.
using ReactiveKernels, Test
import Enzyme
using ReactiveKernels: stage_primal, stage_reverse, stage_residuals

alias_value(x) = sum(abs2, x; init=0.0)
alias_joint(x) = (alias_value(x), x)
copy_joint(x) = (alias_value(x), copy(x))
alias_direct(x, seed) = (2seed) .* x
alias_contract(partial, seed) = (2seed) .* partial
function alias_graph(joint, cost)
    @kernel spec(x::Vector{Float64}, seed::Float64) = begin
        @recipe (cost=4) y::Float64 = alias_value(x)
        @recipe (cost=cost) (y, partial::Vector{Float64}) = joint(x)
        @recipe (cost=4) dx::Vector{Float64} = alias_direct(x, seed)
        @recipe (cost=1) dx = alias_contract(partial, seed)
        return y, dx
    end
    spec
end
function overwrite_after(f::F, x) where {F}
    y = f(x)
    fill!(x, 0.0)
    y
end
function probe(f::F, original) where {F}
    x = copy(original)
    gradient = zeros(length(x))
    wrapped = x -> overwrite_after(f, x)
    result = Enzyme.autodiff(Enzyme.ReverseWithPrimal, Enzyme.Const(wrapped),
        Enzyme.Active, Enzyme.Duplicated(x, gradient))
    (;value=result[2], gradient, input_after=x)
end
function alias_stages(rule::R, x, seed) where {R}
    y, tape = stage_primal(rule, Val(1), x)
    y, only(stage_reverse(rule, Val(1), tape, seed))
end

println((;julia=VERSION, rk=pkgversion(ReactiveKernels), enzyme=pkgversion(Enzyme),
    library=Enzyme.Compiler.Enzyme_jll.libEnzyme_path))
for n in (0, 3, 17)
    x = [0.7sin(i) for i in 1:n]
    expected = 2 .* x
    reference = probe(alias_value, x)
    @test reference.value ≈ alias_value(x)
    @test reference.gradient ≈ expected
    for (label, joint, cost) in ((:independent, alias_joint, 10),
            (:shared_alias, alias_joint, 5), (:shared_copy, copy_joint, 5))
        rule = derivative_rule(alias_graph(joint, cost); primal=:y, covector=:seed,
            cotangents=(x=:dx,), name=:alias_probe)
        _, tape = stage_primal(rule, Val(1), x)
        @test alias_stages(rule,x,1.0)[2] ≈ expected
        if n == 3
            findings = ReactiveKernels._level_code_findings(:all, alias_stages,
                (typeof(rule),Vector{Float64},Float64))
            println((;label,stage_audit_findings=findings))
        end
        result = probe(rule, x)
        @test all(iszero,result.input_after)
        if label !== :shared_alias
            @test result.gradient ≈ expected
        end
        println((;label,n,residuals=stage_residuals(rule,Val(1)),
            residual_aliases_input=any(v -> v === x, tape),
            value_correct=isapprox(result.value,reference.value),
            gradient_correct=isapprox(result.gradient,expected),
            expected, result...))
    end
end
println("JOINT_ALIAS_PROBE_COMPLETE")
