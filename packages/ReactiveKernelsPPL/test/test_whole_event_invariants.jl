module WholeEventInvariantTests
using Test, ReactiveKernels, ReactiveKernelsPPL
using Distributions: Uniform, Normal, Exponential
using LogExpFunctions: logistic, logit, log1pexp, logaddexp
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    bernoulli_product_mixture_logdensity, bernoulli_logit_product_mixture_logdensity
import DifferentiationInterface as DI
import Enzyme

function call_depths!(depths, expression, target, depth=0)
    # A plate block runs as its own program where it stood (`_PlateProgram`).
    expression isa ReactiveKernels._PlateProgram &&
        return call_depths!(depths, expression.ast, target, depth)
    expression isa Expr || return depths
    expression.head in (:quote,:inert) && return depths
    if expression.head === :call &&
            any(x -> x isa GlobalRef && isdefined(x.mod,x.name) &&
                getglobal(x.mod,x.name) === target, expression.args[1:min(2,end)])
        push!(depths, depth)
    end
    foreach(x -> call_depths!(depths,x,target,depth + (expression.head in (:for,:while))),
            expression.args)
    depths
end

# Independent Julia likelihood: a structural-zero component can explain only
# an entirely absent event. Its impossible branch contributes no derivative.
function event_likelihood(logits, weight, events, offsets, scale)
    total = 0.0
    for i in eachindex(events)
        present = log(weight)
        for j in eachindex(events[i])
            eta = logits[j] + scale * offsets[i]
            present += events[i][j] ? -log1pexp(-eta) : -log1pexp(eta)
        end
        total += any(events[i]) ? present : logaddexp(present,log1p(-weight))
    end
    total
end
function reference(raw, events, offsets, kind)
    if kind === :event_dependent
        # RKPPL packs scalar draws before the vector draw, independently of
        # the vector declaration's position in this authored model.
        scale, weight = exp(raw[1]), logistic(raw[2])
        location = logistic.(raw[3:end])
        logits = logit.(location)
        prior = -scale + raw[1]
        for i in 2:length(raw)
            prior += -log1pexp(-raw[i]) - log1pexp(raw[i])
        end
    else
        weight = logistic(raw[1])
        scale = 0.0
        prior = -log1pexp(-raw[1]) - log1pexp(raw[1])
        if kind === :probability
            location = logistic.(raw[2:end])
            logits = logit.(location)
            for i in 2:length(raw)
                prior += -log1pexp(-raw[i]) - log1pexp(raw[i])
            end
        else
            logits = raw[2:end]
            for x in logits
                prior += -(log(2pi)+x*x)/2
            end
        end
    end
    prior + event_likelihood(logits,weight,events,offsets,scale)
end

# The cache positions a non-allocating program binds to its tables' lanes.
table_slots(kernel) = [s.args[2].args[2] for s in kernel.ast.args[2].args
    if Meta.isexpr(s,:(=)) && startswith(string(s.args[1]),"##indexed_lane")]
# Fewest bytes over a few warm calls.
allocations(f) = (f(); minimum(_ -> @allocated(f()), 1:3))

function make_sampler(kind, events, offsets, raw)
    T = maximum(length,events; init=3)
    absent = kind === :probability ? zeros(T) : fill(-Inf,T)
    data = (; events,offsets,T,absent)
    source = if kind === :probability
        quote
            weight ~ Uniform(0,1)
            location[1:T] .~ Uniform.(0,1)
            @plate for i in eachindex(events)
                events[i] ~ LogDensity(bernoulli_product_mixture_logdensity,
                    hcat(location,absent),[weight,1-weight])
            end
        end
    elseif kind === :logit
        quote
            weight ~ Uniform(0,1)
            location[1:T] .~ Normal.(0,1)
            @plate for i in eachindex(events)
                events[i] ~ LogDensity(bernoulli_logit_product_mixture_logdensity,
                    hcat(location,absent),[weight,1-weight])
            end
        end
    else
        quote
            location[1:T] .~ Uniform.(0,1)
            scale ~ Exponential(1)
            weight ~ Uniform(0,1)
            @plate for i in eachindex(events)
                eta = logit.(location) .+ scale .* offsets[i]
                events[i] ~ LogDensity(bernoulli_logit_product_mixture_logdensity,
                    hcat(eta,absent),[weight,1-weight])
            end
        end
    end
    bound = bind_data(lower_rkppl(source,data;mod=@__MODULE__,conditioned=(:events,)),data)
    prepare_sampler(build_kernel(bound),bound,raw;
        backend=DI.AutoEnzyme(;mode=Enzyme.Reverse),native_forward_batch=false)
end

# Registered Enzyme (0.13.219) refuses the `hcat(location, absent)` tables here
# with EnzymeRuntimeActivityError; Enzyme dogfood differentiates them. The
# boundary is Enzyme's: benchmark/repro_enzyme_lazy_hcat_table_const_column.jl.
@testset "natural whole-event invariant expressions" begin
    events = [Bool[true,false,true],Bool[false,false,false],Bool[false,true,false],
              Bool[true,true,false],Bool[false,false,false]]
    offsets = [-0.3,0.2,0.0,0.4,-0.1]
    for kind in (:probability,:logit,:event_dependent)
        raw = kind === :event_dependent ? [0.1,-0.3,0.2,0.3,-0.2] : [-0.2,0.1,-0.3,0.2]
        sampler = make_sampler(kind,events,offsets,raw)
        ast = code_expr(sampler.kernel)
        if kind === :event_dependent
            @test call_depths!(Int[],ast,logit) == [0]
            @test isempty(call_depths!(Int[],ast,ReactiveKernels._indexed_map))
        else
            @test call_depths!(Int[],ast,ReactiveKernels._indexed_map) == [0,0]
            lanes = filter(cache -> cache isa Base.RefValue,sampler.kernel.caches)
            @test !isempty(lanes)
            @test ReactiveKernels._indexed_data_safe((lanes,sampler.kernel.constants,raw))
            # Each table keeps a lane slot typed by its array.
            tables = table_slots(sampler.kernel)
            @test length(tables) == 2
            @test all(j -> sampler.kernel.caches[j] isa
                Base.RefValue{Union{Nothing,Matrix{Float64}}},tables)
            @test isempty(ReactiveKernels._level_code_findings(:all,sampler.kernel,
                                                               (typeof(raw),)))
        end
        for values in (raw,raw .+ 0.15)
            expected_gradient = zero(values)
            expected = Enzyme.autodiff(Enzyme.ReverseWithPrimal,reference,Enzyme.Active,
                Enzyme.Duplicated(values,expected_gradient),Enzyme.Const(events),
                Enzyme.Const(offsets),Enzyme.Const(kind))[2]
            actual, gradient = sampler_value_and_gradient!(sampler,similar(values),values)
            @test sampler(values) ≈ expected rtol=1e-12
            @test actual ≈ expected rtol=1e-12
            @test gradient ≈ expected_gradient rtol=1e-11 atol=1e-12
        end
        if kind !== :event_dependent
            # The tables refill buffers the sampler keeps: with the pass, a
            # density or gradient call allocates exactly what it does without
            # it (snag posteriordb-mt-c-01c1a87b: 4 and 8 more allocations).
            kept = [sampler.kernel.caches[j][] for j in table_slots(sampler.kernel)]
            # The logit kind's `location` is a view of the unconstrained point,
            # which `_indexed_safe` does not take yet, so it builds no tables
            # (ReactiveKernels todo 10kg7nx).
            @test all(table -> table isa Matrix{Float64}, kept) broken = kind === :logit
            baseline = task_local_storage(ReactiveKernels._INDEXED_INVARIANT_DEPTH, 1) do
                make_sampler(kind,events,offsets,raw)
            end
            @test isempty(call_depths!(Int[],code_expr(baseline.kernel),
                                       ReactiveKernels._indexed_map))
            gradient = similar(raw)
            @test allocations(() -> sampler(raw)) == allocations(() -> baseline(raw))
            # Under `--check-bounds=yes` the hoisted loop's gradient keeps 64 B
            # more, whatever the table size, with fresh tables too
            # (ReactiveKernels todo 0qx2ebp).
            forced_bounds = Base.JLOptions().check_bounds == 1
            built = all(table -> table isa Matrix{Float64}, kept)
            @test allocations(() -> sampler_value_and_gradient!(sampler,gradient,raw)) ==
                allocations(() -> sampler_value_and_gradient!(baseline,gradient,raw)) broken =
                forced_bounds && built
            @test all(((j,table),) -> sampler.kernel.caches[j][] === table,
                      zip(table_slots(sampler.kernel),kept))
        end
        empty_sampler = make_sampler(kind,Vector{Bool}[],Float64[],raw)
        actual, gradient = sampler_value_and_gradient!(empty_sampler,similar(raw),raw)
        expected_gradient = zero(raw)
        expected = Enzyme.autodiff(Enzyme.ReverseWithPrimal,reference,Enzyme.Active,
            Enzyme.Duplicated(raw,expected_gradient),Enzyme.Const(Vector{Bool}[]),
            Enzyme.Const(Float64[]),Enzyme.Const(kind))[2]
        @test empty_sampler(raw) ≈ expected
        @test actual ≈ expected
        @test gradient ≈ expected_gradient
    end
end
end
