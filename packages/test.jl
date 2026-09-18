using Pkg

const EXAMPLE_PACKAGES = (
    "ReactiveKernelsDistributionKernels",
    "ReactiveKernelsKernelExamples",
    "ReactiveKernelsBatchingExamples",
    "ReactiveKernelsPPLExamples",
    "ReactiveKernelsCompatibilityExamples",
    "ReactiveKernelsNUTSExamples",
    "ReactiveKernelsStreamingStats",
    "ReactiveKernelsHMCDiagnostics",
    "ReactiveKernelsPPL",
    "ReactiveKernelsReactantODESolvers",
)

for package in EXAMPLE_PACKAGES
    Pkg.test(package; coverage = false)
end
