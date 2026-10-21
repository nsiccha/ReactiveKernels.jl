# Child process of test_pure_surface_boundary.jl. In a fresh process, it
# counts the compiled specializations of every method defined in the
# iterative-method compiler's source files, runs a pure-surface workload, and
# counts again, then calls into the iterative half once on purpose as a
# positive control. It prints one line per finding for the parent to check.
#
# This list is the authority for which files form the iterative-method
# compiler (docs/src/compiler.md, "Pure kernels and the iterative-method
# compiler"). `kernel_stateful.jl` is shared: the `@kernel` mode discrimination,
# method extraction and pure endpoint templates there build every pure
# method-bearing kernel object, so it is not on the list.
const ITERATIVE_FILES = (
    "binders.jl",
    "kernel_adaptation.jl",
    "kernel_codegen.jl",
    "kernel_control.jl",
    "kernel_execution.jl",
    "kernel_factory.jl",
    "kernel_lowering.jl",
    "kernel_methodir.jl",
    "kernel_structural_container.jl",
    "native_slot_compiler.jl",
    "native_slots.jl",
    "native_slots_factory.jl",
    "transpiled_program.jl",
)

using ReactiveKernels, ReactiveKernelsPPL, ReactiveKernelsDistributionKernels
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal
using DifferentiationInterface: AutoEnzyme
import Enzyme

const _OWNERS = (ReactiveKernels, ReactiveKernels.NativeSlotCompiler)
const _SRC = joinpath(pkgdir(ReactiveKernels), "src")

_iterative(m::Method) = m.module in _OWNERS &&
    dirname(String(m.file)) == _SRC && basename(String(m.file)) in ITERATIVE_FILES

function _iterative_methods()
    found = Set{Method}()
    keep(m) = (_iterative(m) && push!(found, m); nothing)
    if isdefined(Core, :methodtable)
        # Julia 1.12+: one global method table holds every method.
        Base.visit(keep, Core.methodtable)
    else
        # Julia 1.10: one table per function type, the constructor table, and
        # the table of callable non-function types. Base functions are
        # visited for the Base methods these files extend.
        tables = IdDict{Core.MethodTable,Nothing}(Core.Type.body.name.mt => nothing)
        for M in (_OWNERS..., Base), n in names(M; all = true)
            isdefined(M, n) || continue
            v = getfield(M, n)
            v isa Function && (tables[typeof(v).name.mt] = nothing)
            if v isa Type && M !== Base
                u = Base.unwrap_unionall(v)
                u isa DataType && (tables[u.name.mt] = nothing)
            end
        end
        foreach(mt -> Base.visit(keep, mt), keys(tables))
    end
    sort!(collect(found); by = m -> (String(m.file), m.line))
end

_count(m::Method) = length(collect(Base.specializations(m)))
_census(ms) = IdDict{Method,Int}(m => _count(m) for m in ms)
_where(m::Method) = string(m.name, " ", basename(String(m.file)), ":", m.line)
# Loading the package evaluates module-level constants and, on Julia 1.10, the
# nested module's `include`; nothing else from these files runs at load.
_load_time(m::Method) = m.name === :include ||
    Base.unwrap_unionall(m.sig).parameters[1] <: Type

methods_iter = _iterative_methods()
for file in ITERATIVE_FILES
    println("FILE ", file, " ", count(m -> basename(String(m.file)) == file, methods_iter))
end
before = _census(methods_iter)
for m in methods_iter
    before[m] > 0 && println(_load_time(m) ? "LOADED " : "BASELINE ", _where(m))
end

# The pure surface: a function-shaped kernel over a DistributionKernels family
# object, a pure method-bearing kernel object, prepare, prepare_ad, vectorize,
# and an @rkppl model built and queried with a native Reverse sampler.
@kernel _census_normal_total(x::Vector{Float64}, mu::Float64, s::Float64) = begin
    pointwise = plate(x) do xi
        normal(mu, s).logpdf(xi)
    end
    return sum(pointwise)
end
@kernel _census_affine(slope::Float64) = begin
    apply(x::Float64)::Float64 = slope * x
    logjac(x::Float64)::Float64 = log(abs(slope))
end
@kernel _census_square(position, shared) = begin
    result = position .* position .+ shared
    return result
end

const _BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)
x = [0.3, -1.1, 0.8, 2.0]
total = prepare(_census_normal_total)
value = total(x, 0.2, 1.3)
ad_value, ad_grad = ad_value_and_gradient(
    prepare_ad(total, _BACKEND, x, 0.2, 1.3; active = :mu), x, 0.2, 1.3)
P = [1.0 2.0; 3.0 4.0]
batched = vectorize(_census_square; batched = :position)(P, 1.0)
square = prepare(_census_square)
model = quote
    a ~ Normal(0, 1)
    b ~ Normal(0, 1)
    s ~ Exponential(1)
    mu = a .+ b .* x
    y .~ Normal.(mu, s)
end
X = collect(range(-1.0, 1.0; length = 8))
data = Dict{Symbol,Any}(:x => X, :y => 0.3 .+ 0.7 .* X .+ 0.1 .* sin.(1:8))
fit = bind_data(lower_rkppl(model, data; conditioned = (:y,), mod = @__MODULE__), data)
built = build_kernel(fit)
u = [0.1, 0.2, -0.3]
sampler = prepare_sampler(built, fit, u; backend = _BACKEND)
lp, grad = Base.invokelatest(sampler_value_and_gradient!, sampler, similar(u), u)
pointwise = Base.invokelatest(prepare_query(built, fit, :pointwise), u)
println("WORKLOAD ", isfinite(value) && ad_value ≈ value && isfinite(ad_grad) &&
    prepare(normal.logpdf)(0.0, 1.0, 0.5) ≈ -0.125 - log(2π) / 2 &&
    prepare(_census_affine.apply)(2.0, 3.0) == 6.0 &&
    batched == hcat(square(P[:, 1], 1.0), square(P[:, 2], 1.0)) &&
    isfinite(lp) && all(isfinite, grad) && sum(pointwise.y) ≈
        Base.invokelatest(prepare_query(built, fit, :likelihood), u))

after = _census(methods_iter)
for m in methods_iter
    after[m] > before[m] && println("ENTERED ", _where(m))
end

# Positive control: the detector sees a deliberate call into the iterative half.
ReactiveKernels.partial(+, 1)(2) == 3 || error("partial control failed")
control = _census(methods_iter)
println("CONTROL ", count(m -> control[m] > after[m], methods_iter))
