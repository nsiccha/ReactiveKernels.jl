using Pkg

const EXAMPLE_PACKAGES = (
    "ReactiveKernelsDistributionKernels",
    "ReactiveKernelsKernelExamples",
    "ReactiveKernelsBatchingExamples",
    "ReactiveKernelsPPLExamples",
    "ReactiveKernelsCompatibilityExamples",
)

for package in EXAMPLE_PACKAGES
    Pkg.test(package; coverage = false)
end
