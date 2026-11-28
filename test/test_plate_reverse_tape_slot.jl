module PlateReverseTapeSlotTests
using ReactiveKernels, DifferentiationInterface, MutatingFunctions, Test
import Enzyme
const RK = ReactiveKernels

# A gathered plate, then a step that throws for a negative scale: the plate's
# augmented primal has run by then, and its reverse pass never does.
@kernel gathered_then_log(q::Vector{Float64}, s::Float64,
        idx::Vector{Int}, x::Vector{Float64}) = begin
    gathered = ReactiveKernels._readonly_gather(q, idx)
    terms = plate(gathered, x) do v, y
        sin(s * v + y)
    end
    total::Float64 = sum(terms) + log(s)
    return total
end

function reference(q, s, idx, x)
    value, gradient, scale = log(s), zeros(length(q)), 1 / s
    for i in eachindex(idx)
        j = idx[i]
        argument = s * q[j] + x[i]
        value += sin(argument)
        gradient[j] += s * cos(argument)
        scale += q[j] * cos(argument)
    end
    value, gradient, scale
end

@kernel _guarded_log_ratio(y::Int, mu::Float64, scale::Float64) = begin
    value::Float64 = y >= 0 && mu >= 0 && scale > 0 ?
        log(float(y) + scale) - log(scale) + mu * scale : -Inf
end
@kernel _sum_log_ratios(y::Vector{Int}, mu::Vector{Float64}, scale::Float64) = begin
    rows = plate(y, mu) do yi, mi
        _guarded_log_ratio(yi, mi, scale)
    end
    total::Float64 = sum(rows)
end

@testset "an arm's cotangent reaches a shared operand defined after its ready flag" begin
    scale = 1.56
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    for (y, mu) in (([0, 1, 3, 5], [1.2, 1.5, 2.1, 0.4]), (Int[], Float64[]),
                    ([-1, -2], [0.4, 1.2]))
        with_plate_reverse() do
            k = prepare_nonallocating(prepare(_sum_log_ratios; bound = (; y)), mu, scale)
            ad = prepare_ad(k, backend, mu, scale; active = (:mu, :scale))
            dest = (zero(mu), Ref(0.0))
            value, _ = ad_value_and_gradient!(ad, dest, mu, scale)
            @test value ≈ sum((yi >= 0 ? log(yi + scale) - log(scale) + mi * scale : -Inf
                               for (yi, mi) in zip(y, mu)); init = 0.0)
            @test dest[1] ≈ [yi >= 0 ? scale : 0.0 for yi in y]
            @test dest[2][] ≈ sum((yi >= 0 ? inv(yi + scale) - inv(scale) + mi : 0.0
                                    for (yi, mi) in zip(y, mu)); init = 0.0)
        end
    end
end

# The tape slots a preparation owns, read from its AD caches.
function tape_slots(prepared)
    slots = RK._PlateTapeSlot[]
    for caches in RK._ad_owned_storage(prepared)
        caches isa Tuple || continue
        append!(slots, (c for c in caches if c isa RK._PlateTapeSlot))
    end
    slots
end

function prepare_case(n, run, enabled; q = [.1, -.4, .3])
    idx = [mod1(div(i - 1, run) + 1, 3) for i in 1:n]
    x = [sin(i / 3) / 2 for i in 1:n]
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    kernel = RK.with_bound_statistics(false) do
        with_plate_reverse(enabled) do
            prepare_nonallocating(prepare(gathered_then_log; bound = (; idx, x)), q, 1.2)
        end
    end
    prepared = with_plate_reverse(enabled) do
        prepare_ad(kernel, backend, q, 1.2; active = (:q, :s))
    end
    prepared, idx, x
end

function check!(prepared, destination, point, scale, idx, x)
    pristine = copy(point)
    value, gradient, dscale = reference(point, scale, idx, x)
    v, returned = ad_value_and_gradient!(prepared, destination, point, scale)
    @test v ≈ value rtol=3e-10 atol=3e-10
    @test destination[1] ≈ gradient rtol=3e-10 atol=3e-10
    @test destination[2][] ≈ dscale rtol=3e-10 atol=3e-10
    @test returned === destination
    @test point == pristine
end

allocated(prepared, destination, point, scale) =
    @allocated ad_value_and_gradient!(prepared, destination, point, scale)

@testset "gradient destinations cannot overlap a tape slot" begin
    ext = Base.get_extension(RK, :ReactiveKernelsEnzymeExt)
    point = (zeros(3), Ref(1.0))
    gradient = (ones(3), Ref(0.0))
    slot = RK._PlateTapeSlot()
    contexts = (DifferentiationInterface.Cache((slot,)),)
    @test ext._structured_destinations(point, gradient, contexts)
    slot.tape = gradient[1]
    @test !ext._structured_destinations(point, gradient, contexts)
    slot.tape = zeros(3)
    slot.empty = gradient[1]
    @test !ext._structured_destinations(point, gradient, contexts)
    slot.empty = Float64[]
    # Larger selectors use the interval sweep rather than pairwise checks.
    points = ntuple(_ -> zeros(3), 5)
    gradients = ntuple(_ -> ones(3), 5)
    intervals = ext._ShadowIntervals()
    @test ext._interval_destinations!(intervals, points, gradients, contexts)
    slot.tape = gradients[5]
    @test !ext._interval_destinations!(intervals, points, gradients, contexts)
end

@testset "a tape slot is leased by removal" begin
    slot = RK._PlateTapeSlot()
    @test isempty(slot.tape)
    first_tape = RK._pr_take_tape!(slot, 5)
    @test first_tape == zeros(5) && isempty(slot.tape)
    first_tape .= 1:5
    RK._pr_return_tape!(slot, first_tape)
    @test slot.tape === first_tape
    # The same vector, cleared, and the slot empty while it is out.
    again = RK._pr_take_tape!(slot, 5)
    @test again === first_tape && again == zeros(5) && isempty(slot.tape)
    # A second call before the first one's reverse pass allocates its own.
    second = RK._pr_take_tape!(slot, 5)
    @test second !== again && second == zeros(5)
    RK._pr_return_tape!(slot, second)
    RK._pr_return_tape!(slot, again)
    @test slot.tape === again
    # Another length leaves the held vector alone and is adopted on return.
    longer = RK._pr_take_tape!(slot, 7)
    @test length(longer) == 7 && slot.tape === again
    RK._pr_return_tape!(slot, longer)
    @test slot.tape === longer
    @test RK._pr_take_tape!(nothing, 3) == zeros(3)
    @test RK._pr_return_tape!(nothing, zeros(3)) === nothing
    # A released slot keeps nothing: every call allocates, a return is dropped.
    released = RK._PlateTapeSlot()
    RK._pr_release_tapes!((nothing, DifferentiationInterface.Cache((zeros(2), released))))
    taken = RK._pr_take_tape!(released, 5)
    @test taken == zeros(5)
    RK._pr_return_tape!(released, taken)
    @test isempty(released.tape) && released.tape === released.empty
    @test RK._pr_take_tape!(released, 5) !== taken
end

@testset "one-pass gradients reuse the preparation's tape, $n cells" for (n, grouped_run) in
        ((128, 32), (257, 16))
    # Exercise tape ownership even on backends that cannot read run metadata.
    run = RK._PR_INDEX_RUN_REFUSAL[] === nothing ? grouped_run : 1
    q = [.1, -.4, .3]
    prepared, idx, x = prepare_case(n, run, true)
    slot = only(tape_slots(prepared))
    # A derivative audit may have evaluated the exemplar during preparation.
    # The first caller reuses any tape that evaluation returned.
    initial_tape = slot.tape
    @test isempty(initial_tape) || length(initial_tape) == length(q) + 2
    destination = (fill(71., 3), Ref(81.))
    check!(prepared, destination, q, 1.2, idx, x)
    tape = slot.tape
    isempty(initial_tape) || @test tape === initial_tape
    # The gathered parent's buffer plus the generated rule's two scalar
    # operands: the initial plate total and the scale.
    @test length(tape) == length(q) + 2
    for rep in 1:2, (point, scale) in ((q, 1.2), ([-.8, .6, 1.2], .7), (q, 1.4))
        check!(prepared, destination, point, scale, idx, x)
        @test slot.tape === tape
    end
    allocated(prepared, destination, q, 1.2)
    bytes = allocated(prepared, destination, q, 1.2)
    # Account for the structured gradient wrapper on the loaded backend.
    # The generated tape itself must add no allocations.
    ordinary, = prepare_case(n, run, false)
    allocated(ordinary, destination, q, 1.2)
    ordinary_bytes = allocated(ordinary, destination, q, 1.2)
    @info "warm one-pass gradient with an owned tape" n bytes ordinary_bytes
    @test bytes <= ordinary_bytes

    # A lease that never came back (an exception between the passes): the next
    # call allocates, is correct, and its reverse pass refills the slot.
    lost = RK._pr_take_tape!(slot, length(slot.tape))
    @test lost === tape && isempty(slot.tape)
    check!(prepared, destination, q, 1.2, idx, x)
    @test length(slot.tape) == length(tape) && slot.tape !== tape
    refilled = slot.tape
    check!(prepared, destination, [-.8, .6, 1.2], .7, idx, x)
    @test slot.tape === refilled

    # The same through a real throw after the plate.
    @test_throws DomainError ad_value_and_gradient!(prepared, destination, q, -1.2)
    after_throw = slot.tape
    @test isempty(after_throw)
    check!(prepared, destination, q, 1.2, idx, x)
    @test length(slot.tape) == length(tape)
    @info "slot after a throw behind the plate" n emptied = isempty(after_throw)
    check!(prepared, destination, q, 1.4, idx, x)

    # A second preparation owns its own slot and vector.
    other, = prepare_case(n, run, true)
    other_slot = only(tape_slots(other))
    @test other_slot !== slot
    other_destination = (fill(71., 3), Ref(81.))
    check!(other, other_destination, q, 1.2, idx, x)
    check!(prepared, destination, [-.8, .6, 1.2], .7, idx, x)
    @test other_slot.tape !== slot.tape
    check!(other, other_destination, q, 1.4, idx, x)
end

@testset "a plate inline at preparation needs no slot" begin
    q = [.1, -.4, .3]
    prepared, idx, x = prepare_case(24, 8, true)
    @test isempty(tape_slots(prepared))
    destination = (fill(71., 3), Ref(81.))
    for (point, scale) in ((q, 1.2), ([-.8, .6, 1.2], .7))
        check!(prepared, destination, point, scale, idx, x)
    end
    allocated(prepared, destination, q, 1.2)
    bytes = allocated(prepared, destination, q, 1.2)
    ordinary, = prepare_case(24, 8, false)
    @test typeof(prepared.call) === typeof(ordinary.call)
    allocated(ordinary, destination, q, 1.2)
    ordinary_bytes = allocated(ordinary, destination, q, 1.2)
    @info "warm inline gradient" bytes ordinary_bytes
    @test bytes == ordinary_bytes
end

@testset "without the generated rule there is no slot" begin
    q = [.1, -.4, .3]
    prepared, idx, x = prepare_case(128, 32, false)
    @test isempty(tape_slots(prepared))
    check!(prepared, (fill(71., 3), Ref(81.)), q, 1.2, idx, x)
end

# Every tape slot reachable from a prepared operator: the one its cache table
# owns and each shadow of it that a backend preparation keeps.
function reachable_tape_slots!(found, @nospecialize(x), seen)
    x isa Union{Module,Type,Method,Core.MethodInstance,Core.CodeInstance,Symbol,
                AbstractString,Expr,Number,Nothing,Task,IO} && return found
    isbitstype(typeof(x)) && return found
    if ismutable(x)
        x in seen && return found
        push!(seen, x)
    end
    x isa RK._PlateTapeSlot && return push!(found, x)
    if x isa Array
        eltype(x) <: Union{Number,Symbol,AbstractString} && return found
        for i in eachindex(x)
            isassigned(x, i) && reachable_tape_slots!(found, x[i], seen)
        end
        return found
    end
    for i in 1:nfields(x)
        isdefined(x, i) && reachable_tape_slots!(found, getfield(x, i), seen)
    end
    found
end
reachable_tape_slots(prepared) =
    reachable_tape_slots!(RK._PlateTapeSlot[], prepared, Base.IdSet{Any}())

function second_order_reference(q, s, idx, x, direction)
    gradient, hvp = zeros(length(q)), zeros(length(q))
    for i in eachindex(idx)
        j = idx[i]
        argument = s * q[j] + x[i]
        gradient[j] += s * cos(argument)
        hvp[j] -= s^2 * sin(argument) * direction[j]
    end
    gradient, hvp
end

# The outer pass of a Hessian-vector product differentiates the rule: it takes
# the rule's decision to reuse a slot's vector from the slot, and zeroes and
# fills its own shadow of that vector over the same length. The backend keeps
# one shadow of the slot per preparation it holds (here one for the allocating
# and one for the in-place product), each made at preparation. A shadow whose
# vector is shorter than the slot's is written past its end by the next call
# through its preparation, which no bounds check sees.
@testset "a Hessian-vector operator's tape slot and its shadows hold vectors of one length, $n cells" for (n, run) in
        ((128, 32), (257, 16))
    q, direction = [.1, -.4, .3], [.2, -.7, 1.1]
    idx = [mod1(div(i - 1, run) + 1, 3) for i in 1:n]
    x = [sin(i / 3) / 2 for i in 1:n]
    backend = SecondOrder(AutoEnzyme(; mode = Enzyme.Forward),
                          AutoEnzyme(; mode = Enzyme.Reverse))
    kernel = RK.with_bound_statistics(false) do
        with_plate_reverse() do
            prepare_nonallocating(prepare(gathered_then_log; bound = (; idx, x)), q, 1.2)
        end
    end
    prepared = with_plate_reverse() do
        prepare_ad_hvp(kernel, backend, (direction,), q, 1.2; active = :q)
    end
    owned = only(tape_slots(prepared))
    lengths() = [length(slot.tape) for slot in reachable_tape_slots(prepared)]
    @test any(slot -> slot === owned, reachable_tape_slots(prepared))
    @info "tape slots of a Hessian-vector operator" n lengths = lengths()
    @test allequal(lengths())
    # With a shorter shadow the calls below would overwrite its neighbours.
    if allequal(lengths())
        for rep in 1:2, (point, scale, tangent) in (
                (q, 1.2, direction), ([-.8, .6, 1.2], .7, [-.4, .1, .9]))
            expected_g, expected_h = second_order_reference(point, scale, idx, x, tangent)
            @test only(ad_hvp(prepared, (tangent,), point, scale)) ≈ expected_h rtol=3e-10 atol=3e-10
            @test allequal(lengths())
            g, (h,) = ad_gradient_and_hvp(prepared, (tangent,), point, scale)
            @test g ≈ expected_g rtol=3e-10 atol=3e-10
            @test h ≈ expected_h rtol=3e-10 atol=3e-10
            @test allequal(lengths())
            gradient, results = fill(71., 3), (fill(72., 3),)
            ad_gradient_and_hvp!(prepared, gradient, results, (tangent,), point, scale)
            @test gradient ≈ expected_g rtol=3e-10 atol=3e-10
            @test only(results) ≈ expected_h rtol=3e-10 atol=3e-10
            @test allequal(lengths())
        end
    end
end
end
