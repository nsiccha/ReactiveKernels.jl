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
end

@testset "one-pass gradients reuse the preparation's tape, $n cells" for (n, run) in
        ((128, 32), (257, 16))
    q = [.1, -.4, .3]
    prepared, idx, x = prepare_case(n, run, true)
    slot = only(tape_slots(prepared))
    @test isempty(slot.tape)
    destination = (fill(71., 3), Ref(81.))
    check!(prepared, destination, q, 1.2, idx, x)
    tape = slot.tape
    # The gathered parent's buffer plus the generated rule's two scalar
    # operands: the initial plate total and the scale.
    @test length(tape) == length(q) + 2
    for rep in 1:2, (point, scale) in ((q, 1.2), ([-.8, .6, 1.2], .7), (q, 1.4))
        check!(prepared, destination, point, scale, idx, x)
        @test slot.tape === tape
    end
    allocated(prepared, destination, q, 1.2)
    bytes = allocated(prepared, destination, q, 1.2)
    @info "warm one-pass gradient with an owned tape" n bytes
    @test bytes == 0

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

@testset "a call that takes two passes leaves its slot empty" begin
    q = [.1, -.4, .3]
    prepared, idx, x = prepare_case(24, 8, true)
    slot = only(tape_slots(prepared))
    destination = (fill(71., 3), Ref(81.))
    for (point, scale) in ((q, 1.2), ([-.8, .6, 1.2], .7))
        check!(prepared, destination, point, scale, idx, x)
        @test isempty(slot.tape)
    end
    allocated(prepared, destination, q, 1.2)
    bytes = allocated(prepared, destination, q, 1.2)
    @info "warm two-pass gradient with an unused slot" bytes
    @test bytes == 0
end

@testset "without the generated rule there is no slot" begin
    q = [.1, -.4, .3]
    prepared, idx, x = prepare_case(128, 32, false)
    @test isempty(tape_slots(prepared))
    check!(prepared, (fill(71., 3), Ref(81.)), q, 1.2, idx, x)
end
end
