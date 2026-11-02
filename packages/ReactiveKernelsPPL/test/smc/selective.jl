# Focused optional integration runner; uses the same inference environment as
# runtests.jl and includes native gradients and separate-process continuation.
include("gaussian_projection.jl")
include("selective_coordinates.jl")
include("selective_existing.jl")
include("selective_edges.jl")
include("selective_growth.jl")
include("selective_factors.jl")
include("selective_replay.jl")
