# Reactant alone: its broadcast eltype probe indexes a one-dimensional
# traced view with CartesianIndex{1}. Base.reindex requires an index tuple.
using Reactant, Test
cartesian_view(x) = view(x, 2:3)[CartesianIndex(1)]
@test cartesian_view([1.0, 2.0, 3.0, 4.0]) == 2.0
x = Reactant.to_rarray([1.0, 2.0, 3.0, 4.0])
try
    Reactant.@compile cartesian_view(x)
    @test_broken true
catch error
    error isa MethodError && error.f === Base.reindex &&
        length(error.args) == 2 && error.args[2] isa CartesianIndex{1} || rethrow()
    @test_broken false
end
