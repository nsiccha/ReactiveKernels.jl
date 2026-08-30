module StatefulFunctionalContractsFixture

using Random
using LinearAlgebra
using ReactiveKernels

@kernel dormant(value=-1.0, count=0) = begin
    step!(flag) = begin
        flag && return true
        value = sqrt(value)
        count += 1
        return false
    end
end

@kernel drift(value=4.0, count=0) = begin
    step!(flag) = begin
        flag && return true
        value += one(value)
        count += 1
        return false
    end
end

@kernel array_drift(seed, count=0) = begin
    values = deepcopy(seed)
    step!(flag) = begin
        flag && return true
        count += 1
        return false
    end
end

# The first owned builtin scalar is deliberately floating while the loop/index
# carrier is the later integer `count`. Functional lowering must derive every
# synthetic index zero from its selected integer carrier, never from the
# unrelated predicate carrier.
@kernel float_first(seed, value=1.0, count=0) = begin
    arr = deepcopy(seed)
    step!(idx) = begin
        idx > 0 || return false
        value += 0.0
        for j in 1:idx
            arr[j] = value
            count += 1
        end
        return true
    end
end

@kernel loop_probe(count=0) = begin
    step!(lo, hi) = begin
        lo <= hi || return false
        for j in lo:hi
            count += 1
        end
        return true
    end
end

@kernel small_loop(count=0) = begin
    step!(lo::Int8, hi::Int8) = begin
        lo <= hi || return false
        for j in lo:hi
            count += j
        end
        return true
    end
end

@kernel mixed_loop(count=0, last=Int8(0)) = begin
    step!(lo::Int8, hi::Int8) = begin
        lo <= hi || return false
        for j in lo:hi
            count += 1
            last = j
        end
        return true
    end
end

@kernel outer_alias(seed, total=0.0, count=0) = begin
    a = deepcopy(seed)
    b = a
    step!(take) = begin
        take || return false
        total = b[1]
        count += 0
        a[1] = 2.0
        return true
    end
end

@kernel straight_alias(seed) = begin
    a = deepcopy(seed)
    b = a
    fit!(values) = begin
        @. a = b + values
    end
end

# These two explicit-return methods deliberately share a state whose field
# names equal the NamedTuple result layout. The functional ABI must use the
# source-derived return contract, never infer "returns state" from names or
# runtime result shape.
@kernel straight_result_contract(value=2.0, count=3) = begin
    named_result(scale) = begin
        return (; value=value * scale, count=count)
    end
    nothing_result(take) = begin
        return
    end
end

@kernel outer_alias_effect(seed, callback=nothing, count=0) = begin
    a = deepcopy(seed)
    b = a
    step!(take) = begin
        take || return false
        callback(a, b)
        count += 1
        return true
    end
end

@kernel independent_equal_scalars(left=1, right=1, callback=nothing,
        count=0) = begin
    step!(take) = begin
        take || return false
        callback(left, right)
        count += 1
        return true
    end
end

@kernel independent_arrays_machine(left, right, count=0) = begin
    a = deepcopy(left)
    b = deepcopy(right)
    step!(take) = begin
        take || return false
        copy!!(a, b)
        count += 1
        return true
    end
end

@kernel independent_arrays_straight(left, right, count=0) = begin
    a = deepcopy(left)
    b = deepcopy(right)
    fit!(values) = begin
        @. a = b + values
        count += 1
    end
end

@kernel wrapped_storage_alias(initial, count=0) = begin
    step!(take) = begin
        take || return false
        count += 1
        return true
    end
end


@kernel authored_nested_storage_write(initial, count=0) = begin
    step!(replacement) = begin
        count >= 0 || return false
        initial.nested.diagonal = replacement
        count += 1
        return true
    end
end

@kernel uniform_state(value=false, count=0) = begin
    step!(rng, take) = begin
        take || return false
        value = Random.rand(rng, Bool)
        count += 1
        return true
    end
end

@kernel bad_uniform_state(value=false, count=0) = begin
    step!(rng, take) = begin
        take || return false
        value = Random.rand(rng, Int)
        count += 1
        return true
    end
end

@kernel normal_statement(seed) = begin
    mom = deepcopy(seed)
    count = 0
    step!(rng) = begin
        Random.randn!(rng, mom)
        count += 1
        return true
    end
end

@kernel normal_indexed(seed) = begin
    mom = deepcopy(seed)
    other = deepcopy(seed)
    count = 0
    step!(rng, take) = begin
        take || return false
        mom[1] = Random.randn!(rng, other)[1]
        count += 1
        return true
    end
end

@kernel normal_replace(init, count=0) = begin
    step!(rng, take) = begin
        take || return false
        init.mom = Random.randn!(rng, init.mom)
        count += 1
        return true
    end
end

@kernel pure_contract(callback=nothing, value=0.0, count=0) = begin
    step!(input, take) = begin
        take || return false
        value = callback(input)
        count += 1
        return true
    end
end

@kernel effect_contract(callback=nothing, count=0) = begin
    step!(take) = begin
        take || return false
        callback(__self__)
        count += 1
        return true
    end
end


@kernel sequential_effect_contract(callback=nothing, count=0) = begin
    step!(take) = begin
        take || return false
        callback(__self__)
        callback(__self__)
        count += 1
        return true
    end
end

@kernel structured_effect(init, callback=nothing, count=0) = begin
    step!(take) = begin
        take || return false
        callback(init)
        count += 1
        return true
    end
end

@kernel paired_structured_effect(init, fwd, callback=nothing, count=0) = begin
    step!(take) = begin
        take || return false
        callback(init, fwd)
        count += 1
        return true
    end
end

@kernel array_effect(seed, callback=nothing, count=0) = begin
    values = deepcopy(seed)
    step!(take) = begin
        take || return false
        callback(values)
        count += 1
        return true
    end
end

struct StructuredReplacement{Mode,V}
    replacement::V
end
StructuredReplacement{Mode}(replacement::V) where {Mode,V} =
    StructuredReplacement{Mode,V}(replacement)

function (lowering::StructuredReplacement{Mode})(effect, point) where {Mode}
    replacement = Mode === :alias ?
        merge(point, (dham_dpos=lowering.replacement,)) :
        merge(point, (pos=lowering.replacement,))
    (arguments=(replacement,), result=nothing, effect_state=effect)
end

struct ArrayReplacement{V}
    replacement::V
end
(lowering::ArrayReplacement)(effect, values) = (
    arguments=(lowering.replacement,), result=nothing, effect_state=effect)

struct PairedStructuredReplacement{Mode} end
function (::PairedStructuredReplacement{Mode})(effect, init, fwd) where {Mode}
    replacement = if Mode === :broken_required_alias
        merge(fwd, (dham_dpos=copy(fwd.dham_dpos),))
    elseif Mode === :cross_canon_share
        merge(fwd, (pos=init.pos,))
    else
        error("unknown paired structured replacement mode")
    end
    (arguments=(init, replacement), result=nothing, effect_state=effect)
end

struct BrokenOuterAliasReplacement end
(::BrokenOuterAliasReplacement)(effect, a, b) = (
    arguments=(copy(a), fill(9.0, size(b))), result=nothing,
    effect_state=effect)


struct IndependentScalarReplacement end
(::IndependentScalarReplacement)(effect, left, right) = (
    arguments=(left + 1, right + 2), result=nothing,
    effect_state=effect)


struct EqualScalarReplacement end
function (::EqualScalarReplacement)(effect, left, right)
    replacement = left + right
    (arguments=(replacement, replacement), result=nothing,
     effect_state=effect)
end

struct WrongShapeEffectState end
(::WrongShapeEffectState)(effect, state) = (
    arguments=(state,), result=nothing, effect_state=[9.0])

struct PassThroughEffectState end
(::PassThroughEffectState)(effect, state) = (
    arguments=(state,), result=nothing, effect_state=effect)

struct BrokenAliasEffectState end
(::BrokenAliasEffectState)(effect, state) = (
    arguments=(state,), result=nothing,
    effect_state=(left=effect.left, right=copy(effect.right)))

struct MergedAliasEffectState end
(::MergedAliasEffectState)(effect, state) = (
    arguments=(state,), result=nothing,
    effect_state=(left=effect.left, right=effect.left))

struct AliasConfig{A}
    left::A
    right::A
end

struct AliasConfiguredEffect{C}
    config::C
end
function (lowering::AliasConfiguredEffect)(effect, state)
    config = lowering.config
    increment = config.left === config.right ? only(config.left) : -10_000
    (arguments=(state,), result=nothing,
     effect_state=effect + increment)
end


struct AliasSensitiveSequentialEffect end
function (::AliasSensitiveSequentialEffect)(effect, state)
    increment = effect.left === effect.right ? one(eltype(effect.left)) :
        convert(eltype(effect.left), 100)
    replacement = effect.left .+ increment
    (arguments=(state,), result=nothing,
     effect_state=(left=replacement, right=replacement))
end

function nested_storage_transition(backing=[1.0, 2.0])
    wrapped = (metric=Diagonal(backing), diagonal=backing)
    initial = (nested=wrapped,)
    names = (:nested,)
    groups = ((:nested,),)
    external_groups = ()
    writable_names = (:nested,)
    ensures = ()
    repairs = (nested=identity,)
    topology = ReactiveKernels._sm_topology_contract(initial)
    f = ReactiveKernels.compile(:((ensures, state) -> state))
    ReactiveKernels.CompiledStateTransition{
        names,groups,external_groups,writable_names,typeof(f),
        typeof(ensures),typeof(initial),typeof(repairs),typeof(topology)}(
            f, ensures, initial, repairs, topology)
end

end # module StatefulFunctionalContractsFixture
