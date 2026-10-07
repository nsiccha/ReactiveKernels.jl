# Precompile workload: cache this package's own authoring, planning, lowering
# and preparation code in the package image, so a fresh process does not
# JIT-compile it on first use. Construction compiles little per-model code
# (`_kernel_eval_definition`); without a workload a fresh process still
# compiled this package code once, and Julia serializes that compilation
# across threads. The workload authors, prepares and calls a small kernel
# with an authored plate, the shape generated programs use.
@setup_workload begin
    workload_definition = :(_precompile_workload(
            x::Vector{Float64}, a::Float64, b::Float64) = begin
        cells = plate(x, a, b) do value, intercept, slope
            cell::Float64 = intercept + slope * log(value)
            cell
        end
        total::Float64 = sum(cells)
        return total
    end)
    @compile_workload begin
        workload_spec = _kernel_eval_definition(@__MODULE__, workload_definition)
        workload_kernel = Base.invokelatest(prepare, workload_spec)
        Base.invokelatest(workload_kernel, [1.0, 2.0], 0.5, 2.0)
    end
end
