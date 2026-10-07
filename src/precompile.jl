# Precompile workload: cache this package's own authoring, planning, lowering
# and preparation code in the package image, so a fresh process does not
# JIT-compile it on first use. Construction compiles little per-model code
# (`_kernel_eval_definition`); without a workload a fresh process still
# compiled this package code once, and Julia serializes that compilation
# across threads. The workload authors, prepares and calls a small kernel
# with an authored plate, the shape generated programs use, and authored
# `for`/`while` loops, whose tensorized companions are expanded with
# ReactantCore's `@trace` when a kernel is defined.
@setup_workload begin
    workload_definition = :(_precompile_workload(
            x::Vector{Float64}, a::Float64, b::Float64) = begin
        cells = plate(x, a, b) do value, intercept, slope
            cell::Float64 = intercept + slope * log(value)
            cell
        end
        prefix::Vector{Float64} = let
            out = zero(cells)
            acc = 0.0
            for i in eachindex(cells)
                acc = acc + cells[i]
                out[i] = acc
            end
            out
        end
        steps::Int = let
            k = 0
            while k < length(x)
                k = k + 1
            end
            k
        end
        total::Float64 = sum(prefix) + steps
        return total
    end)
    @compile_workload begin
        workload_spec = _kernel_eval_definition(@__MODULE__, workload_definition)
        workload_kernel = Base.invokelatest(prepare, workload_spec)
        Base.invokelatest(workload_kernel, [1.0, 2.0], 0.5, 2.0)
    end
end
