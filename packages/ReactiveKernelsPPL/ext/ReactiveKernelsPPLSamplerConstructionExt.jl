module ReactiveKernelsPPLSamplerConstructionExt

using ReactiveKernelsPPL: _construction_workload
using PrecompileTools: @compile_workload
import WarmupHMC
import Enzyme

# The single-trigger images each cache construction with only one sampler
# dependency present. Build this image with both method tables loaded.
@compile_workload begin
    _construction_workload()
end

end
