module IndexedInvariantTests
using ReactiveKernels, Test, Enzyme, DifferentiationInterface
using LogExpFunctions: log1pexp

@kernel indexed_log_choices(events, p::Matrix{Float64}) = begin
    event_terms = plate(events) do event
        components = plate(axes(p, 2)) do k
            cells = plate(eachindex(event)) do j
                value = p[j,k]
                event[j] ? log(value) : log1p(-value)
            end
            sum(cells)
        end
        sum(components)
    end
    total = sum(event_terms)
    return total
end

@kernel indexed_logit_choices(events, p::Matrix{Float64}) = begin
    event_terms = plate(events) do event
        components = plate(axes(p, 2)) do k
            cells = plate(eachindex(event)) do j
                value = p[j,k]
                -log1pexp(ifelse(event[j], -value, value))
            end
            sum(cells)
        end
        sum(components)
    end
    total = sum(event_terms)
    return total
end

function reference(p, events, logit)
    total = 0.0
    for event in events
        event_total = 0.0
        for k in axes(p, 2)
            component_total = 0.0
            for j in eachindex(event)
                x = p[j,k]
                component_total += logit ? -log1pexp(ifelse(event[j], -x, x)) :
                    (event[j] ? log(x) : log1p(-x))
            end
            event_total += component_total
        end
        total += event_total
    end
    total
end
function reference_gradient(p, events, logit)
    gradient = zero(p)
    value = Enzyme.autodiff(Enzyme.ReverseWithPrimal, reference, Active,
        Duplicated(p, gradient), Const(events), Const(logit))[2]
    value, gradient
end
nodes(x) = x isa Expr ? 1 + sum(nodes, x.args; init=0) : 0
nodes(x::ReactiveKernels._PlateProgram) = nodes(x.ast)
function table_depths!(depths, x, depth=0)
    # Retained plate programs execute their bodies at the call's loop depth.
    x isa ReactiveKernels._PlateProgram && return table_depths!(depths, x.ast, depth)
    x isa Expr || return depths
    x.head in (:quote,:inert) && return depths
    if x.head === :call && first(x.args) == GlobalRef(ReactiveKernels,:_indexed_map)
        push!(depths,depth)
    end
    foreach(a -> table_depths!(depths,a,depth + (x.head in (:for,:while))),x.args)
    depths
end
backend = AutoEnzyme(; mode=Enzyme.Reverse)

@testset "indexed invariant scalar expressions" begin
    for (spec, logit) in ((indexed_log_choices, false), (indexed_logit_choices, true))
        p = [0.2 0.3; 0.6 0.7; 0.8 0.4]
        histories = [Bool[isodd(i >> j) for j in 0:2] for i in 1:37]
        bound = prepare(spec; bound=(; events=histories))
        @test table_depths!(Int[],code_expr(bound)) == [0,0]
        @test ReactiveKernels._indexed_data_safe((bound.constants,p))
        @test isempty(ReactiveKernels._level_code_findings(:all,bound,(typeof(p),)))
        for data in (histories, [Bool[true], Bool[false, true]],
                     [Bool[], Bool[]], Vector{Bool}[], [Bool[true, false, true]])
            saved = deepcopy((p, data))
            for reader in (prepare(spec), prepare(spec; bound=(; events=data)))
                args = length(inputs(reader)) == 1 ? (p,) : (data, p)
                ad = prepare_ad(reader, backend, args...; active=:p)
                actual, gradient = ad_value_and_gradient(ad, args...)
                expected, expected_gradient = reference_gradient(p, data, logit)
                @test actual ≈ expected
                @test gradient ≈ expected_gradient
            end
            @test (p, data) == saved
        end
        longer = prepare(spec; bound=(; events=repeat(histories, 4)))
        @test nodes(code_expr(bound)) == nodes(code_expr(longer))
        # Grow positions and components as well as the observation domain.
        # Bound event extents and live matrix sizes remain runtime iteration.
        wide_p = repeat(p,4,3)
        wide_events = [repeat(event,4) for event in repeat(histories,4)]
        wide_reader = prepare(spec; bound=(; events=wide_events))
        @test nodes(code_expr(bound)) == nodes(code_expr(wide_reader))
        @test table_depths!(Int[],code_expr(wide_reader)) == [0,0]
        wide_ad = prepare_ad(wide_reader,backend,wide_p;active=:p)
        wide_value, wide_gradient = ad_value_and_gradient(wide_ad,wide_p)
        wide_expected, wide_expected_gradient = reference_gradient(wide_p,wide_events,logit)
        @test wide_value ≈ wide_expected
        @test wide_gradient ≈ wide_expected_gradient
        for changed in (p .+ 0.05, p .* 0.9)
            @test bound(changed) ≈ reference(changed, histories, logit)
        end
    end

    # Inactive singular and invalid logarithms must retain ordinary lazy
    # execution and ordinary derivatives through the guarded fallback.
    for (p, data) in ((reshape([0.0, 1.0], 2, 1), [Bool[false,true] for _ in 1:5]),
                      (reshape([-0.2, 1.2], 2, 1), [Bool[false,true] for _ in 1:5]),
                      (reshape([-0.2], 1, 1), Vector{Bool}[]))
        reader = prepare(indexed_log_choices; bound=(; events=data))
        ad = prepare_ad(reader, backend, p; active=:p)
        actual, gradient = ad_value_and_gradient(ad, p)
        expected, expected_gradient = reference_gradient(p, data, false)
        @test actual ≈ expected
        @test gradient ≈ expected_gradient
    end
    reader = prepare(indexed_log_choices)
    @test_throws BoundsError reader([Bool[true,true] for _ in 1:5], fill(0.4,1,1))
    @test_throws DomainError reader([Bool[true] for _ in 1:5], fill(-0.4,1,1))
    @test_throws TypeError reader([[1] for _ in 1:5], fill(0.4,1,1))
    cyclic = Any[]
    append!(cyclic,(cyclic,cyclic))
    @test_throws TypeError reader(cyclic,fill(0.4,2,1))
end

@kernel indexed_subterms(events, p) = begin
    event_terms = plate(events) do event
        components = plate(axes(p, 2)) do k
            cells = plate(eachindex(event)) do j
                x = p[j,k]
                event[j] > 0 ? event[j] * log(x) : event[j] + log1p(-x)
            end
            sum(cells)
        end
        sum(components)
    end
    total = sum(event_terms)
    return total
end
function subterm_reference(p, events)
    total = 0.0
    for event in events
        event_total = 0.0
        for k in axes(p, 2)
            component_total = 0.0
            for j in eachindex(event)
                x = p[j,k]
                component_total += event[j] > 0 ? event[j] * log(x) : event[j] + log1p(-x)
            end
            event_total += component_total
        end
        total += event_total
    end
    total
end
struct ObservedReads <: AbstractMatrix{Float64}
    values::Matrix{Float64}
    count::Base.RefValue{Int}
end
Base.size(x::ObservedReads) = size(x.values)
Base.getindex(x::ObservedReads, i::Int, j::Int) = (x.count[] += 1; x.values[i,j])

@testset "invariant subterms in event-dependent arithmetic" begin
    backend = AutoEnzyme(; mode=Enzyme.Reverse)
    p = [0.2 0.3; 0.6 0.7; 0.8 0.4]
    data = [[1.2, -0.1, 0.0], [0.0, 2.0], Float64[], [-1.0, 3.0, 2.0]]
    k = prepare(indexed_subterms)
    @test !isempty(table_depths!(Int[], code_expr(k)))
    for events in (data, Vector{Float64}[], [Float64[], Float64[]])
        for reader in (k, prepare(indexed_subterms; bound=(; events)))
            args = length(inputs(reader)) == 1 ? (p,) : (events, p)
            ad = prepare_ad(reader, backend, args...; active=:p)
            actual, derivative = ad_value_and_gradient(ad, args...)
            expected_derivative = zero(p)
            expected = Enzyme.autodiff(Enzyme.ReverseWithPrimal, Const(subterm_reference), Active,
                Duplicated(p, expected_derivative), Const(events))[2]
            @test actual ≈ expected
            @test derivative ≈ expected_derivative
        end
    end
    custom = ObservedReads(p, Ref(0))
    @test k(data, custom) ≈ subterm_reference(p, data)
    @test custom.count[] == size(p, 2) * sum(length, data)
    @test k(Vector{Float64}[], custom) == 0.0
    @test custom.count[] == size(p, 2) * sum(length, data)
    p32, data32 = Float32.(p), [Float32.(row) for row in data]
    ad32 = prepare_ad(k,backend,data32,p32;active=:p)
    actual32, gradient32 = ad_value_and_gradient(ad32,data32,p32)
    expected_gradient32 = zero(p32)
    expected32 = Enzyme.autodiff(Enzyme.ReverseWithPrimal,Const(subterm_reference),Active,
        Duplicated(p32,expected_gradient32),Const(data32))[2]
    @test actual32 ≈ expected32 rtol=5e-6
    @test gradient32 ≈ expected_gradient32 rtol=5e-6
end

# A source value that is also consumed directly must retain its load.
@kernel indexed_shared_values(events, p::Matrix{Float64}) = begin
    rows = plate(events) do event
        cells = plate(eachindex(event)) do j
            x = p[j,1]
            event[j] ? x + log(x) : x + log1p(-x)
        end
        sum(cells)
    end
    total = sum(rows)
    return total
end

struct MutatingEvent <: AbstractVector{Bool}
    values::Vector{Bool}
    victim::Matrix{Float64}
    reads::Base.RefValue{Int}
end
Base.size(event::MutatingEvent) = size(event.values)
function Base.getindex(event::MutatingEvent, i::Int)
    event.reads[] += 1
    event.victim[1,1] += 0.01
    event.values[i]
end

@testset "reuse retains direct consumers and observable effects" begin
    events = [Bool[true,false], Bool[false,true], Bool[true,true]]
    p = reshape([0.3,0.7],2,1)
    reader = prepare(indexed_shared_values)
    @test reader(events,p) ≈ sum(sum(p[j,1] + (e[j] ? log(p[j,1]) :
        log1p(-p[j,1])) for j in eachindex(e)) for e in events)
    actual_p, expected_p = copy(p), copy(p)
    actual_reads, expected_reads = Ref(0), Ref(0)
    custom = [MutatingEvent(e,actual_p,actual_reads) for e in events]
    reference_events = [MutatingEvent(e,expected_p,expected_reads) for e in events]
    effectful = prepare(indexed_log_choices)
    @test effectful(custom,actual_p) ≈ reference(expected_p,reference_events,false)
    @test actual_p == expected_p
    @test actual_reads[] == expected_reads[] == sum(length,events)
    @test effectful(MutatingEvent[],actual_p) == 0.0
    @test actual_reads[] == sum(length,events)
end

@testset "tuple-held observations keep storage proofs bounded" begin
    p = [0.2 0.3; 0.6 0.7; 0.8 0.4]
    short = Tuple(Bool[isodd(i >> j) for j in 0:2] for i in 1:13)
    long = (short...,short...,short...,short...)
    readers = map((short,long)) do events
        reader = prepare(indexed_log_choices;bound=(; events))
        @test table_depths!(Int[],code_expr(reader)) == [0,0]
        for bound in (false,true)
            kernel = bound ? reader : prepare(indexed_log_choices)
            args = bound ? (p,) : (events,p)
            ad = prepare_ad(kernel,backend,args...;active=:p)
            actual, gradient = ad_value_and_gradient(ad,args...)
            expected, expected_gradient = reference_gradient(p,events,false)
            @test actual ≈ expected
            @test gradient ≈ expected_gradient
        end
        reader
    end
    @test nodes(code_expr(readers[1])) == nodes(code_expr(readers[2]))
    # Inspect the reachable generated helper too: a bounded caller alone
    # would miss one emitted storage check per tuple-held observation.
    proof_sizes = map((short,long)) do events
        data = (readers[1].constants,(; events),[events],p)
        @test ReactiveKernels._indexed_data_safe(data)
        lowered = only(code_lowered(ReactiveKernels._indexed_data_safe,(typeof(data),)))
        length(lowered.code) + sum(nodes,lowered.code;init=0)
    end
    @test proof_sizes[1] == proof_sizes[2]
    custom = ObservedReads(p,Ref(0))
    @test !ReactiveKernels._indexed_data_safe((; events=short,custom))
    @test custom.count[] == 0
    @test !ReactiveKernels._indexed_data_safe(Any[short])
    @test !ReactiveKernels._indexed_passive_type(Tuple)
    # Reused native output buffers have an ordinary-array-or-nothing field.
    # The proof must cover every possible value without reading that field.
    lane = Ref{Union{Nothing,Matrix{Float64}}}(nothing)
    @test ReactiveKernels._indexed_data_safe(lane)
    lane[] = p
    @test ReactiveKernels._indexed_data_safe(lane)
    @test !ReactiveKernels._indexed_data_safe(Ref{Union{Nothing,ObservedReads}}(nothing))
end

@kernel indexed_changing_columns(columns, p::Matrix{Float64}) = begin
    rows = plate(columns) do k
        cells = plate(axes(p,1)) do j
            x = p[j,k]
            log(x)
        end
        sum(cells)
    end
    total = sum(rows)
    return total
end

@testset "enclosing coordinates do not imply indexed reuse" begin
    reader = prepare(indexed_changing_columns)
    p = [0.2 0.3; 0.6 0.7; 0.8 0.4]
    @test isempty(table_depths!(Int[], code_expr(reader)))
    @test reader([1,2],p) ≈ sum(log,p)
    @test reader(Int[],p) == 0.0
end

@kernel indexed_state(events::Vector{Vector{Bool}}, p::Matrix{Float64}) = begin
    total::Float64 = indexed_log_choices(events,p)
    return total
end

@testset "compiled reactive inputs invalidate invariant work" begin
    events = [Bool[true,false],Bool[false,true],Bool[true,true]]
    p = reshape([0.3,0.7],2,1)
    program = prepare_reactive(indexed_state)
    state = program(events,copy(p))
    source = statevalue(program,indexed_state.p)
    wanted = statevalue(program,indexed_state.total)
    @test get!(state,wanted) ≈ reference(p,events,false)
    mutate!(state,source) do values
        values[1,1] = 0.5
    end
    changed = reshape([0.5,0.7],2,1)
    @test get!(state,wanted) ≈ reference(changed,events,false)
    set!(state,statevalue(program,indexed_state.events),[Bool[],Bool[]])
    @test get!(state,wanted) == 0.0
    @test p == reshape([0.3,0.7],2,1)
end

end
