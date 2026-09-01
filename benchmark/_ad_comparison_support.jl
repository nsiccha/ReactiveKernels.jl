module ADComparisonSupport

using BenchmarkTools
using DifferentiationInterface
import DynamicPPL
using Pkg
using SHA
using Statistics

export RKValueGradientCall, DIValueGradientCall, TuringValueGradientCall
export measurement, build_and_first_call, record_implementation
export gradient_error, gradient_scale
export package_version, output_path
export normalized_text, normalized_read, text_sha256
export source_pin, published_source_pin, nothing_paths

const LDP = DynamicPPL.LogDensityProblems

struct RKValueGradientCall{P,G,A}
    prepared::P
    gradient::G
    arguments::A
end

(call::RKValueGradientCall)() = ad_value_and_gradient!(
    call.prepared, call.gradient, call.arguments...)

struct DIValueGradientCall{F,G,P,B,X,C}
    objective::F
    gradient::G
    preparation::P
    backend::B
    point::X
    contexts::C
end

(call::DIValueGradientCall)() = DifferentiationInterface.value_and_gradient!(
    call.objective, call.gradient, call.preparation, call.backend,
    call.point, call.contexts...)

struct TuringValueGradientCall{F,X}
    logdensity::F
    point::X
end

(call::TuringValueGradientCall)() =
    LDP.logdensity_and_gradient(call.logdensity, call.point)

function measurement(call; rounds::Int)
    benchmark = @benchmarkable $call()
    times_ns = Float64[]
    bytes = Int[]
    allocs = Int[]
    for _ in 1:rounds
        estimate = minimum(run(benchmark; samples = 200, seconds = 0.2))
        push!(times_ns, estimate.time)
        push!(bytes, estimate.memory)
        push!(allocs, estimate.allocs)
    end
    Dict(
        "times_ns" => times_ns,
        "min_ns" => minimum(times_ns),
        "median_ns" => median(times_ns),
        "bytes" => bytes,
        "median_bytes" => Int(median(bytes)),
        "allocs" => allocs,
        "median_allocs" => Int(median(allocs)),
    )
end

function build_and_first_call(builder)
    preparation = @timed builder()
    call = preparation.value
    first_execution = @timed call()
    call, first_execution.value, Dict(
        "preparation_seconds" => preparation.time,
        "preparation_bytes" => preparation.bytes,
        "first_execution_seconds" => first_execution.time,
        "first_execution_bytes" => first_execution.bytes,
    )
end

gradient_error(actual, expected) =
    maximum(abs.(actual .- expected); init = 0.0)
gradient_scale(expected) =
    maximum(abs, expected; init = eps(Float64))

function record_implementation(call, result, setup, reference_value,
                               reference_gradient; rounds, caller_owned)
    value, gradient = result
    merge(setup, measurement(call; rounds), Dict(
        "value" => Float64(value),
        "value_abs_error" => abs(Float64(value) - reference_value),
        "gradient_max_abs_error" => gradient_error(gradient, reference_gradient),
        "gradient_max_rel_error" =>
            gradient_error(gradient, reference_gradient) /
            max(gradient_scale(reference_gradient), eps(Float64)),
        "gradient_length" => length(gradient),
        "caller_owned_gradient" => caller_owned,
    ))
end

function package_version(name)
    for info in values(Pkg.dependencies())
        info.name == name && return string(info.version)
    end
    error("package $name absent from the benchmark environment")
end

function output_path(arguments = ARGS)
    for argument in arguments
        startswith(argument, "--output=") &&
            return split(argument, '='; limit = 2)[2]
    end
    nothing
end

normalized_text(text) =
    replace(String(text), "\r\n" => "\n", "\r" => "\n")
normalized_read(path) = normalized_text(read(path, String))
text_sha256(path) = bytes2hex(SHA.sha256(normalized_read(path)))

function source_pin(root, relative_path)
    absolute_path = joinpath(root, relative_path)
    Dict(
        "path" => relative_path,
        "git_blob" => readchomp(`git -C $root hash-object $absolute_path`),
        "text_sha256" => text_sha256(absolute_path),
    )
end

function published_source_pin(root, commit, relative_path)
    text = normalized_text(read(`git -C $root show $commit:$relative_path`))
    Dict(
        "commit" => commit,
        "path" => relative_path,
        "git_blob" => readchomp(`git -C $root rev-parse $commit:$relative_path`),
        "text_sha256" => bytes2hex(SHA.sha256(text)),
    ), text
end

function nothing_paths(value, path = "receipt")
    paths = String[]
    if value === nothing
        push!(paths, path)
    elseif value isa AbstractDict
        for (key, child) in value
            append!(paths, nothing_paths(child, "$path.$key"))
        end
    elseif value isa AbstractVector
        for (index, child) in pairs(value)
            append!(paths, nothing_paths(child, "$path[$index]"))
        end
    end
    paths
end

end # module ADComparisonSupport
