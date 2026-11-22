module ReactiveKernelsPPLCategoricalArraysExt

using CategoricalArrays: CategoricalArray, CategoricalPool, isordered, levels
import DataAPI
import ReactiveKernelsPPL: _fingerprint

# Level order, including unused levels, determines the meaning of the codes.
# In CategoricalArrays 1.x each level hashes as its unwrapped value.
function _levels_hash(x)
    values = levels(x)
    h = hash(isordered(x), hash(length(values)))
    for value in values
        h = hash(value, h)
    end
    h
end

function _fingerprint(x::Union{CategoricalArray,SubArray{<:Any,<:Any,<:CategoricalArray}},
        h::UInt, seen)
    h = hash(_levels_hash(x), hash(typeof(x), h))
    _fingerprint(DataAPI.refarray(x), h, seen)
end

# Individual CategoricalValues can also occur in ordinary arrays or records.
# Read their shared pool once per traversal, including after a caller mutation.
function _fingerprint(pool::CategoricalPool, h::UInt, seen)
    digest = get!(seen, pool) do
        _levels_hash(pool)
    end
    hash(digest, hash(typeof(pool), h))
end

end
