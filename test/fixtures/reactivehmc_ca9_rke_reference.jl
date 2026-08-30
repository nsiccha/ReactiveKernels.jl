# Generate the RKE physical-oracle receipt from a clean ReactiveHMC checkout.
# Run with that checkout as the active project:
#
#   julia --startup-file=no --project=/path/to/ReactiveHMC.jl \
#       test/fixtures/reactivehmc_ca9_rke_reference.jl

using ReactiveHMC
import LambertW
import ReactiveObjects
using SHA

const EXPECTED_REACTIVEHMC_SHA =
    "ca9ea4ca41924bb0e1fadc01c717e1333916aba6"
const EXPECTED_ENERGIES_SHA256 =
    "1d051da9f1ca56b46e6802f66bbf36c21d58dd4442e6ad6d8118e52a93d492de"

upstream_root = Base.pkgdir(ReactiveHMC)
upstream_sha = readchomp(`git -C $upstream_root rev-parse HEAD`)
upstream_sha == EXPECTED_REACTIVEHMC_SHA || error(
    "expected ReactiveHMC $EXPECTED_REACTIVEHMC_SHA, loaded $upstream_sha",
)
isempty(readchomp(`git -C $upstream_root status --short -- src`)) ||
    error("ReactiveHMC src/ checkout is dirty")

energies_path = joinpath(upstream_root, "src", "energies.jl")
energies_sha256 = bytes2hex(sha256(read(energies_path)))
energies_sha256 == EXPECTED_ENERGIES_SHA256 || error(
    "expected energies.jl $EXPECTED_ENERGIES_SHA256, read $energies_sha256",
)

reactiveobjects_root = Base.pkgdir(ReactiveObjects)
reactiveobjects_revision = readchomp(`git -C $reactiveobjects_root rev-parse HEAD`)

println("schema = \"reactivehmc-rke-ca9-v1\"")
println()
println("[pins]")
println("reactivehmc_revision = \"$upstream_sha\"")
println("energies_sha256 = \"$energies_sha256\"")
println("reactiveobjects_revision = \"$reactiveobjects_revision\"")
println("lambertw_version = \"$(Base.pkgversion(LambertW))\"")
println("julia_version = \"$(VERSION)\"")

for (m, c) in ((1.0, 1.0), (0.8, 1.5))
    object = ReactiveHMC.rke(; m, c)
    x_sq = [0.0, 0.25, 1.0, 4.0]
    q = [0.1, 0.5, 0.9, 0.99]
    e_sq = [ReactiveHMC.e_sq(object, x) for x in x_sq]
    p_sq = [ReactiveHMC.p_sq(object, x) for x in x_sq]
    cdf_sq = [ReactiveHMC.cdf_sq(object, x) for x in x_sq]
    quantile_sq = [ReactiveHMC.quantile_sq(object, probability) for probability in q]
    roundtrip_cdf = [ReactiveHMC.cdf_sq(object, x) for x in quantile_sq]

    println()
    println("[[cases]]")
    println("m = $(repr(m))")
    println("c = $(repr(c))")
    println("c1 = $(repr(object.c1))")
    println("c2 = $(repr(object.c2))")
    println("P0_sq = $(repr(object.P0_sq))")
    println("x_sq = $(repr(x_sq))")
    println("e_sq = $(repr(e_sq))")
    println("p_sq = $(repr(p_sq))")
    println("cdf_sq = $(repr(cdf_sq))")
    println("q = $(repr(q))")
    println("quantile_sq = $(repr(quantile_sq))")
    println("roundtrip_cdf = $(repr(roundtrip_cdf))")
end
