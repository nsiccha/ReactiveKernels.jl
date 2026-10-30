module ModelBenchmarkMatrixSpec

export RK_BENCHMARK_CONFIGURATIONS, RK_UNAVAILABLE_CONFIGURATION_COMBINATIONS

# The public execution modifiers shared by the PPL model benchmarks. This is
# deliberately not a Cartesian product: nonallocating execution currently
# returns a NonAllocatingKernel, while the public AD surface consumes
# PreparedKernel/PreparedADKernel. Until those APIs compose, inventing
# nonallocating-AD columns would overstate support. The `reactant`
# configurations describe the historical receipts measured with ReactiveKernels'
# removed Reactant path; the suite validators still map them onto those
# receipts, and the native producers record them as unsupported.
const RK_BENCHMARK_CONFIGURATIONS = (
    (id = "primal_native", differentiation = "primal",
     compiler = "native", allocation = "ordinary", data = "unbound"),
    (id = "primal_native_bound", differentiation = "primal",
     compiler = "native", allocation = "ordinary", data = "bound"),
    (id = "primal_nonallocating", differentiation = "primal",
     compiler = "native", allocation = "nonallocating", data = "unbound"),
    (id = "primal_nonallocating_bound", differentiation = "primal",
     compiler = "native", allocation = "nonallocating", data = "bound"),
    (id = "primal_reactant", differentiation = "primal",
     compiler = "reactant", allocation = "ordinary", data = "unbound"),
    (id = "primal_reactant_bound", differentiation = "primal",
     compiler = "reactant", allocation = "ordinary", data = "bound"),
    (id = "ad_native", differentiation = "value_and_gradient",
     compiler = "native", allocation = "ordinary", data = "unbound"),
    (id = "ad_native_bound", differentiation = "value_and_gradient",
     compiler = "native", allocation = "ordinary", data = "bound"),
    (id = "ad_reactant", differentiation = "value_and_gradient",
     compiler = "reactant", allocation = "ordinary", data = "unbound"),
    (id = "ad_reactant_bound", differentiation = "value_and_gradient",
     compiler = "reactant", allocation = "ordinary", data = "bound"),
)

# Modifier products that are intentionally not benchmark configurations. Data
# binding is orthogonal, so each row excludes both its bound and unbound form.
# Keeping this inventory beside the supported vocabulary makes absence an
# audited API statement rather than an easy-to-miss hole in a Cartesian table.
const RK_UNAVAILABLE_CONFIGURATION_COMBINATIONS = (
    (differentiation = "value_and_gradient", compiler = "native",
     allocation = "nonallocating",
     reason = "prepare_ad consumes PreparedKernel, not NonAllocatingKernel"),
    (differentiation = "primal", compiler = "reactant",
     allocation = "nonallocating",
     reason = "Reactant compilation consumes PreparedKernel, not NonAllocatingKernel"),
    (differentiation = "value_and_gradient", compiler = "reactant",
     allocation = "nonallocating",
     reason = "neither the AD nor Reactant public surface consumes NonAllocatingKernel"),
)

end # module ModelBenchmarkMatrixSpec
