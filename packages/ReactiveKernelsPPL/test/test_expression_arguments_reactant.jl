using Test, ReactiveKernelsPPL

# The shared checker exercises the native value, Enzyme/findiff gradient,
# Reactant value and compiled reverse pass, and returns the emitted operation
# census. Counts must stay fixed as the observation dimension changes.
# Each ordinal branch has at least two lanes at both sizes; one-lane
# scalar specialization is a finite shape case, not iteration expansion.
@testset "expression arguments: compiled values, gradients and structure" begin
    structures = Dict{String,Dict{String,Int}}()
    for n in (6,12), case in _expression_cases(n)
        @testset "$(case.label) n=$n" begin
            println("EXPRESSION_BACKEND_BEGIN ",case.label," n=",n); flush(stdout)
            ops = _pp_backend_check(case.expr,case.data,case.q)
            if n == 6
                structures[case.label] = ops
            else
                @test structures[case.label] == ops
            end
        end
    end
end

@testset "empty response argument values compile with active priors" begin
    for case in _expression_empty_cases()
        _pp_backend_check(case.expr,case.data,case.q)
    end
end

function _expression_inactive_compiled(sampler,u)
    ru = Reactant.to_rarray(u)
    compiled = Reactant.@compile sampler.kernel(ru)
    @test Float64(compiled(ru)) == -Inf
    cad = compile_ad_value_and_gradient(sampler.ad,ru)
    value,gradient = cad(ru)
    @test Float64(value) == -Inf
    @test Array(gradient) ≈ -u
end
@testset "argument domain checks compile with inactive reverse paths" begin
    for (expr,data,q) in _expression_inactive_cases()
        sampler,u = _expression_inactive_native(expr,data,q)
        Base.invokelatest(_expression_inactive_compiled,sampler,u)
    end
end


# Canonical evidence uses beta_inc; keep its existing precise backend
# limitation for arbitrary live Beta shapes as well as mean/precision shapes.
import SpecialFunctions
@testset "Beta argument evidence retains the documented compiled boundary" begin
    for kind in (:truncated,:censored,:interval_censored)
        case = _expression_beta_evidence_case(6,kind)
        bound = bind_data(lower_rkppl(case.expr,case.data;conditioned=keys(case.data)),case.data)
        built = build_kernel(bound)
        kernel = prepare_query(built,bound,:sampler)
        u = unconstrain(built.layout,case.q)
        @test Base.invokelatest(kernel,u) ≈ case.oracle(case.q)
        ru = Reactant.to_rarray(u)
        err = try
            Reactant.@code_hlo optimize=false kernel(ru)
            nothing
        catch e
            e
        end
        @test_broken err === nothing
        @test err isa MethodError && err.f === SpecialFunctions.beta_inc &&
            any(arg -> arg isa Reactant.TracedRNumber,err.args)
    end
end
