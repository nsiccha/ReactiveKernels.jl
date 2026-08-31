#!/usr/bin/env julia

using Pkg

Pkg.activate(@__DIR__)
cd(@__DIR__) do
    Pkg.develop([
        PackageSpec(path = "../.."),
        PackageSpec(path = "../../packages/ReactiveKernelsDistributionKernels"),
    ])
end
Pkg.instantiate()
