# Follow-up experiment: preserve mixed nonlinear terms and measure a
# conservative condition diagnostic with direct fallback. This diagnostic is
# NOT a validated error bound or a proposed production admission policy.
include("soss_symbolic_reduction.jl")
import ForwardDiff

struct ConditionTag end

coefficient_kernel(expr) = RK.compile(:(function(a, b)
    $(SU.Code.toexpr(expr))
end))
basis_kernel(expr) = RK.compile(:(function(y)
    $(SU.Code.toexpr(expr))
end))

function summary_residual(basis)
    graph = Graph()
    pa, pb = value!(graph, :a, Float64), value!(graph, :b, Float64)
    memo = Dict{Any,Any}(a=>pa, b=>pb)
    statistics, summands = Value[], Value[]
    for (i, (_, coefficient)) in enumerate(basis)
        stat = value!(graph, Symbol(:stat, i), Float64)
        c = emit!(graph, coefficient, memo)
        part = value!(graph, Symbol(:part, i), Float64)
        add!(graph; inputs=(c,stat), outputs=(part,), op=*)
        push!(statistics,stat); push!(summands,part)
    end
    result = value!(graph,:result,Float64)
    add!(graph; inputs=Tuple(summands), outputs=(result,), op=+)
    (;graph, have=Tuple([pa,pb,statistics...]), want=(result,), statistics)
end

# One authoritative coefficient primal: ForwardDiff supplies diagnostic
# sensitivities; Enzyme supplies the returned kernel's ordinary reverse.
# No derivative rule is authored or attached to either backend.
primal_channel(v::Real, ::Val{0}) = ForwardDiff.value(v)
primal_channel(v::ForwardDiff.Dual, ::Val{0}) = ForwardDiff.value(v)
primal_channel(v::ForwardDiff.Dual, ::Val{I}) where I = ForwardDiff.partials(v)[I]
primal_channel(v::Real, ::Val{I}) where I = 0.0

function conditioned(values, threshold)
    all(isfinite, values) || return false
    scale = sum(abs, values)
    total = sum(values)
    isfinite(scale) && isfinite(total) || return false
    scale == 0 || abs(total) > threshold * scale
end

# A Bool-only ordinary function keeps the diagnostic's arithmetic outside the
# returned differentiable value. The test exercises this through Enzyme; this
# is not an AD inactivity annotation or a custom derivative.
Base.@noinline function acceptable(coefficients, statistics, aa, bb, threshold)
    da = ForwardDiff.Dual{ConditionTag}(aa, (1.0,0.0))
    db = ForwardDiff.Dual{ConditionTag}(bb, (0.0,1.0))
    contributions = map((f,s)->f(da,db)*s, coefficients, statistics)
    conditioned(map(v->primal_channel(v,Val(0)), contributions), threshold) &&
    conditioned(map(v->primal_channel(v,Val(1)), contributions), threshold) &&
    conditioned(map(v->primal_channel(v,Val(2)), contributions), threshold)
end

struct GuardedSummary{D,R,C,S}
    direct::D
    residual::R
    coefficients::C
    statistics::S
    rows::Int
    data_ok::Bool
    threshold::Float64
end
function (k::GuardedSummary)(aa,bb)
    k.rows == 0 && return 0.0
    if k.data_ok && acceptable(k.coefficients,k.statistics,aa,bb,k.threshold)
        return k.residual(aa,bb)
    end
    k.direct(aa,bb)
end
uses_summary(k::GuardedSummary,aa,bb) = k.rows>0 && k.data_ok &&
    acceptable(k.coefficients,k.statistics,aa,bb,k.threshold)

struct DirectOnly{D}
    direct::D
    rows::Int
end
(k::DirectOnly)(aa,bb) = k.rows==0 ? 0.0 : k.direct(aa,bb)
uses_summary(::DirectOnly,aa,bb) = false

function guarded_spec(expr)
    basis = separate(expr; expand=false)
    direct = outer_graph(expr)
    basis === nothing && return (;basis,direct,residual=nothing,coefficients=(),bases=())
    (;basis,direct,residual=summary_residual(basis),
        coefficients=Tuple(coefficient_kernel(c) for (_,c) in basis),
        bases=Tuple(basis_kernel(t) for (t,_) in basis))
end

function binding_summary(f, basis, data, threshold)
    isequal(basis,1) && return (Float64(length(data)),true)
    values = try
        map(f,data)
    catch err
        # Preserve the source exception at evaluation, rather than moving it
        # to preparation. No data-dependent source rewrite is made.
        err isa DomainError || err isa OverflowError || err isa InexactError || rethrow()
        return (NaN,false)
    end
    value = sum(values)
    (value, conditioned(values,threshold))
end

function bind_guarded(spec,data; threshold=sqrt(eps(Float64)))
    direct = bind(spec.direct,data)
    spec.basis === nothing && return DirectOnly(direct,length(data))
    prepared = map((f,p)->binding_summary(f,first(p),data,threshold),
        spec.bases,Tuple(spec.basis))
    statistics = map(first,prepared)
    valid = all(last,prepared)
    r = spec.residual
    residual = prepare(r.graph; have=r.have,want=r.want,
        bound=Tuple(v=>s for (v,s) in zip(r.statistics,statistics)))
    GuardedSummary(direct,residual,spec.coefficients,statistics,length(data),valid,threshold)
end

function repeated_values(k, parameters, count)
    result=0.0
    for i in 1:count
        j=mod1(i,size(parameters,2))
        result += k(parameters[1,j],parameters[2,j])
    end
    result
end
function repeated_gradients(k, parameters, count)
    result=0.0
    for i in 1:count
        j=mod1(i,size(parameters,2))
        ga,gb=gradient(k,parameters[1,j],parameters[2,j])
        result += ga+gb
    end
    result
end

# Specialize both sides before constructing timed closures. Accessing k.direct
# inside the outer heterogeneous case loop otherwise boxes that field access.
function measure_reuse(direct::D, guarded::G, parameters, count) where {D,G}
    dv=sample_cost(()->repeated_values(direct,parameters,count);seconds=0.03)
    gv=sample_cost(()->repeated_values(guarded,parameters,count);seconds=0.03)
    dg=sample_cost(()->repeated_gradients(direct,parameters,count);seconds=0.03)
    gg=sample_cost(()->repeated_gradients(guarded,parameters,count);seconds=0.03)
    (;dv,gv,dg,gg)
end

function precise_gradient(row,data,aa,bb)
    bigdata=BigFloat.(data)
    ForwardDiff.gradient(p->row_sum(row,p[1],p[2],bigdata),BigFloat[aa,bb])
end

function run_guarded_experiment()
    println("ENV Julia=",VERSION," Enzyme=",pkgversion(Enzyme),
        " ForwardDiff=",pkgversion(ForwardDiff)," SymbolicUtils=",pkgversion(SU))
    rng=Xoshiro(20261113)
    @testset "Guarded separable summary experiment" begin
        for case in CASES
            spec=guarded_spec(case.expr)
            println("CASE ",case.name," BASIS ",spec.basis)
            @test (spec.basis!==nothing)==case.reducible
            previous_ast=nothing
            previous_type=nothing
            for n in (0,1,32,4096)
                data=0.25 .+ 2 .* rand(rng,n)
                prep=@timed bind_guarded(spec,data)
                k=prep.value
                for (aa,bb) in ((0.17,-0.21),(-0.6,0.4),(0.0,0.0))
                    ref=Reference(case.row,data)
                    truth=row_sum(case.row,BigFloat(aa),BigFloat(bb),BigFloat.(data))
                    @test isapprox(k(aa,bb),truth;rtol=3e-11,atol=3e-10)
                    @test all(isapprox.(gradient(k,aa,bb),gradient(ref,aa,bb);
                        rtol=3e-10,atol=3e-9))
                    @test all(isapprox.(gradient(k,aa,bb),
                        precise_gradient(case.row,data,aa,bb);rtol=3e-10,atol=3e-9))
                end
                if n==0
                    @test k(Inf,Inf)===0.0
                    @test gradient(k,Inf,Inf)==(0.0,0.0)
                end
                if k isa GuardedSummary
                    ast=sprint(show,RK.code_expr(k.residual))
                    if previous_ast!==nothing
                        @test ast==previous_ast
                        @test typeof(k.residual.f)===previous_type
                    end
                    previous_ast=ast; previous_type=typeof(k.residual.f)
                end
                parameters=[0.17 -0.6 0.0 0.25; -0.21 0.4 0.0 -0.1]
                selections=[uses_summary(k,parameters[1,j],parameters[2,j]) for j in 1:4]
                println("BIND case=",case.name," n=",n," first_seconds=",prep.time,
                    " bytes=",prep.bytes," retained_bytes=",Base.summarysize(k),
                    " summary_choices=",selections)
                if n in (32,4096)
                    direct_bind=sample_cost(()->bind(spec.direct,data))
                    guarded_bind=sample_cost(()->bind_guarded(spec,data))
                    for count in (1,100,10000)
                        costs=measure_reuse(k.direct,k,parameters,count)
                        println("REUSE case=",case.name," n=",n," count=",count,
                            " direct_value=",costs.dv," guarded_value=",costs.gv,
                            " direct_gradient=",costs.dg," guarded_gradient=",costs.gg,
                            " direct_bind=",direct_bind," guarded_bind=",guarded_bind)
                    end
                end
            end
            data=[0.5,1.0,2.0]
            before=bind_guarded(spec,data)
            before_value=before(0.17,-0.21)
            data .*= 1.5
            after=bind_guarded(spec,data)
            @test after(0.17,-0.21)!=before_value
            @test isapprox(after(0.17,-0.21),
                row_sum(case.row,BigFloat(0.17),BigFloat(-0.21),BigFloat.(data));
                rtol=3e-11,atol=3e-10)
            @test all(isapprox.(gradient(after,0.17,-0.21),
                precise_gradient(case.row,data,0.17,-0.21);rtol=3e-10,atol=3e-9))
            if after isa GuardedSummary
                @test sprint(show,RK.code_expr(after.residual))==
                    sprint(show,RK.code_expr(before.residual))
                @test typeof(after.residual.f)===typeof(before.residual.f)
            end
        end
        # Known errors of full expansion must reach the direct branch.
        square=guarded_spec((a*y-b)^2)
        @test square.basis===nothing
        k=bind_guarded(square,ones(32))
        for (aa,bb,expected) in ((1e8,1e8+1,32.0),(1e160,1e160,0.0))
            @test !uses_summary(k,aa,bb)
            @test k(aa,bb)==expected
            @test gradient(k,aa,bb)==gradient(k.direct,aa,bb)
        end
        # Other failures need data, primal, AND sensitivity diagnostics.
        probes=(
            ("summary_overflow",a*y,[1e308,1e308],1e-308,0.0),
            ("linear_cancellation",a*y-b,[nextfloat(1.0);ones(31)],1e16,1e16),
            ("gradient_cancellation",a*y+exp(a),[nextfloat(-exp(40.0));fill(-exp(40.0),31)],40.0,0.0),
        )
        for (name,expr,data,aa,bb) in probes
            spec=guarded_spec(expr); k=bind_guarded(spec,data)
            @test !uses_summary(k,aa,bb)
            @test k(aa,bb)==k.direct(aa,bb)
            @test gradient(k,aa,bb)==gradient(k.direct,aa,bb)
            println("FALLBACK ",name," value=",k(aa,bb),
                " gradient=",gradient(k,aa,bb),
                " unguarded_value=",k.residual(aa,bb),
                " unguarded_gradient=",gradient(k.residual,aa,bb))
        end
        # Fallback preserves direct execution; it does not certify its accuracy.
        # Record independent gradients for the cancellation probe explicitly.
        data=[nextfloat(-exp(40.0));fill(-exp(40.0),31)]
        row=(aa,bb,yy)->aa*yy+exp(aa)
        native=Reference(row,data)
        println("GRADIENT_BOUNDARY ordinary_native=",gradient(native,40.0,0.0),
            " forward_float64=",ForwardDiff.gradient(p->native(p[1],p[2]),[40.0,0.0]),
            " forward_bigfloat=",precise_gradient(row,data,40.0,0.0))
        # A source-domain exception remains at evaluation; no eager preparation
        # of a valid candidate is allowed to erase the original behavior.
        invalid=bind_guarded(guarded_spec(a*log(y)),[-1.0])
        @test !uses_summary(invalid,1.0,0.0)
        @test_throws DomainError invalid(1.0,0.0)
    end
end

abspath(PROGRAM_FILE)==(@__FILE__) && run_guarded_experiment()
