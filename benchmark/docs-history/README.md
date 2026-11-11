# Documentation research history

The documentation rewrite keeps tutorials on the site and research records here.
The historical site at revision `4d462ef4170948f2e77e3fa80cda5be5c2b6dc02`
contains the former model catalogue, backend reports and comparison pages.
Use that revision when reproducing those reports; they describe its dependencies
and workloads, not a claim about the current release.

- [Benchmark drivers](../README.md) document the maintained performance commands.
- Model sources remain in `packages/ReactiveKernelsPPLExamples/` and their owning
  packages. Removing a catalogue page does not remove its model implementation.

The current tutorials evaluate supplied parameter points and conditional queries.
They perform no statistical fit. Each build writes a receipt containing the page
source hash, package and Julia versions, and checked numerical results; figures
are generated from those results and are never tracked in git.
