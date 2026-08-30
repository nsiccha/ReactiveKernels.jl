module StatefulFunctionalContractsFixture

using Random
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

end # module StatefulFunctionalContractsFixture
