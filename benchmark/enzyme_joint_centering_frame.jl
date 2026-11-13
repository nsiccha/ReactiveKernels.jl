# Backend-only reproducer for mixed active/cached matrix endpoints in a joint
# centering frame. No ReactiveKernels or WarmupHMC dependency is involved.
using Enzyme, Test

struct EndpointFrame
    values::Vector{Vector{Float64}}
end
@noinline endpoint_frame(location, log_scale) =
    EndpointFrame(collect(map(vec, (location, log_scale))))

function loss(u, data)
    frame = endpoint_frame(u[1] .+ data.offsets, data.log_scale)
    result = 0.0
    for i in eachindex(frame.values[1])
        y = (u[i + 1] - frame.values[1][i]) * exp(-frame.values[2][i])
        result += y * y / 2
    end
    result
end

println((; julia = VERSION, enzyme = pkgversion(Enzyme),
    enzyme_path = pathof(Enzyme),
    libenzyme = Enzyme.Compiler.Enzyme_jll.libEnzyme_path))
data = (; offsets = [0.1 0.3 0.5; 0.2 0.4 0.6],
    log_scale = log.([0.7 1.1 1.5; 0.9 1.3 1.7]))
u = 0.1 .* cos.(1:7)
saved = deepcopy(data)
scales = exp.(vec(data.log_scale))
mapped = (u[2:end] .- u[1] .- vec(data.offsets)) ./ scales
expected = [-sum(mapped ./ scales); mapped ./ scales]

for (label, mode) in ((:ordinary_reverse, Reverse),
        (:runtime_activity_control, set_runtime_activity(Reverse)))
    gradient = zero(u)
    try
        autodiff(mode, loss, Active, Duplicated(u, gradient), Const(data))
        @test gradient ≈ expected
        @test data == saved
        println((; label, result = :pass, gradient))
    catch err
        println((; label, result = :error, exception = typeof(err)))
        showerror(stdout, err)
        println()
        @test data == saved
    end
    flush(stdout)
end
