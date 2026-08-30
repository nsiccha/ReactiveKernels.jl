using SHA

const REVISION = "4f051db7df57762a58ac851b0274fe57de342198"
const HEADER_SHA256 =
    "ab00138be5f6dee2d67108cafa11d42e99da3018b29d52d29bf4a07c545bdab5"

length(ARGS) == 2 || error(
    "usage: julia capture.jl <exact-walnutpie-checkout> <eigen-5.0.1-include-root>")
checkout = abspath(ARGS[1])
eigen = abspath(ARGS[2])
header = joinpath(checkout, "include", "walnutpie", "walnuts.hpp")
source = joinpath(@__DIR__, "upstream_macro_oracle.cpp")

readchomp(`git -C $checkout rev-parse HEAD`) == REVISION ||
    error("walnutpie checkout is not exact revision $REVISION")
success(`git -C $checkout diff --quiet`) || error("walnutpie checkout is tracked-dirty")
success(`git -C $checkout diff --cached --quiet`) ||
    error("walnutpie checkout has staged changes")
bytes2hex(sha256(read(header))) == HEADER_SHA256 ||
    error("walnuts.hpp digest does not match source lock")
isfile(joinpath(eigen, "Eigen", "Dense")) ||
    error("second argument is not the Eigen 5.0.1 include root")

mktempdir() do tmp
    exe = joinpath(tmp, "walnuts-upstream-macro-oracle")
    run(`g++ -std=c++20 -O2 -I$(joinpath(checkout, "include")) -I$eigen $source -o $exe`)
    println("# authority_repository=https://github.com/flatironinstitute/walnutpie")
    println("# authority_revision=", REVISION)
    println("# authority_path=include/walnutpie/walnuts.hpp")
    println("# authority_sha256=", HEADER_SHA256)
    println("# execution=separate_cpp_process")
    print(read(`$exe`, String))
end
