# Focused, real-package four-axis acceptance for the posteriordb IRT batch.
#
# Run from the repository root with the shared BridgeStan/Reactant environment:
#
#   IRT_REACTANT=0 julia --project=benchmark/all80-env \
#     packages/ReactiveKernelsPPLExamples/test/acceptance_irt_four_axis.jl
#   IRT_REACTANT=1 julia --project=benchmark/all80-env \
#     packages/ReactiveKernelsPPLExamples/test/acceptance_irt_four_axis.jl
#
# The native phase proves Reactant is absent both before package import and after
# every assertion. The Reactant phase is a separate process and lowers its
# `@compile` expressions only through acceptance_irt_reactant.jl. Each selected
# model gets six BridgeStan-valid native value/Reverse-gradient probes and one
# ordinary Reactant probe; irt_2pl additionally forwards one oracle-checked
# probability-saturation stress probe through both backends.

const DO_REACTANT = get(ENV, "IRT_REACTANT", "0") == "1"
_pkg_loaded(name) = any(id -> id.name == name, keys(Base.loaded_modules))
if !DO_REACTANT
    @assert !_pkg_loaded("Reactant") "native phase started with Reactant loaded"
end

using Random
using LinearAlgebra
import BridgeStan
import PosteriorDB
import Enzyme
using DifferentiationInterface
using ReactiveKernels
using ReactiveKernelsPPLExamples
using ReactiveKernelsPPLExamples:
    Irt2plExample, TwoplLatentRegIrtExample, Hier2plExample, GpcmLatentRegIrtExample

if !DO_REACTANT
    @assert !_pkg_loaded("Reactant") "real-package native import loaded Reactant"
end
if DO_REACTANT
    import Reactant
    @assert _pkg_loaded("Reactant") "Reactant phase failed to load Reactant"
    include(joinpath(@__DIR__, "acceptance_irt_reactant.jl"))
else
    _reactant_axes(args...) = nothing
end

# Ordinary reverse-mode AD only: no function annotation, runtime activity
# annotation, custom rule, or finite-difference substitution.
const AD_BACKEND = AutoEnzyme(mode = Enzyme.Reverse)
const PROBE_SEED = 468
const NATIVE_PROBE_COUNT = 6
const NATIVE_PROBE_SCALE = 0.3
const IRT_SATURATION_ETA = 40.0

const VALUE_TOL = 1e-6
const GRAD_TOL = 1e-3
const REACTANT_VALUE_TOL = 1e-6
const REACTANT_GRAD_TOL = 2e-3

_relative_scalar(a, b) = abs(a - b) / max(abs(b), 1.0)
_relative_vector(a, b) = maximum(abs.(a .- b) ./ max.(abs.(b), 1.0))
_matrix(x) = x isa AbstractMatrix ? x : reduce(vcat, permutedims.(x))
_float_matrix(x) = Float64.(_matrix(x))

function _stan_model(name, seed)
    posterior = PosteriorDB.posterior(PosteriorDB.database(), name)
    stan_path = PosteriorDB.path(
        PosteriorDB.implementation(PosteriorDB.model(posterior), "stan"),
    )
    data_json = PosteriorDB.load(PosteriorDB.dataset(posterior), String)
    BridgeStan.StanModel(stan_path, data_json, seed), posterior, stan_path,
    ncodeunits(data_json)
end

function _reference_finite_points(model, name, dimension, seed, count, scale)
    rng = Xoshiro(seed + sum(codeunits(name)))
    points = Vector{Vector{Float64}}()
    tries = 0
    while length(points) < count && tries < 20_000
        tries += 1
        q = scale .* randn(rng, dimension)
        value = try
            BridgeStan.log_density(model, q; propto = false, jacobian = true)
        catch
            NaN
        end
        isfinite(value) && push!(points, q)
    end
    length(points) == count ||
        error("$name selected only $(length(points))/$count finite reference probes")
    points
end

_repository_head() = readchomp(`git -C $(abspath(joinpath(pkgdir(ReactiveKernelsPPLExamples), "..", ".."))) rev-parse HEAD`)
_version_label(package) = string(something(pkgversion(package), "unknown"))

_stan_value(model, q) =
    BridgeStan.log_density(model, q; propto = false, jacobian = true)
_stan_gradient(model, q) =
    BridgeStan.log_density_gradient(model, q; propto = false, jacobian = true)[2]

function _first_use(case, q)
    # This one ordinary function is the public cold-start shape: build the eager
    # model-only template, prepare it, and execute it before any full-source call.
    graph = case.build()
    kernel = prepare(
        graph; have = case.have, want = :posterior, bound = case.bound,
    )
    kernel(q), kernel
end

const CASES = Dict{String,Any}(
    "irt_2pl" => (
        posterior = "irt_2pl-irt_2pl",
        module_name = "Irt2plExample",
        build = Irt2plExample.build_irt_2pl_graph,
        have = (:unconstrained, :y),
        bound = (; y = Irt2plExample.IRT_2PL_Y),
        package_data = (; y = Irt2plExample.IRT_2PL_Y),
        raw_data = d -> (y = Bool.(_matrix(d["y"])),),
        dimension = 144,
        scale = 0.3,
        stress = () -> begin
            y = Irt2plExample.IRT_2PL_Y
            items, persons = size(y)
            q = zeros(Float64, 2 * items + persons + 4)
            q[2:(persons + 1)] .= IRT_SATURATION_ETA
            # Oracle-first stress check: derive eta and p directly from this q
            # before either backend sees it. At eta=40, Float64 logistic output is
            # saturated beyond 1 - 1e-12 (many values round to exactly 1). This is
            # a tested finite stress point, not a universal support claim.
            theta = view(q, 2:(persons + 1))
            u_a = view(q, (persons + 3):(persons + 2 + items))
            b = view(q, (persons + 5 + items):(persons + 4 + 2 * items))
            eta = [exp(u_a[i]) * (theta[j] - b[i]) for i in 1:items, j in 1:persons]
            probability = 1.0 ./ (1.0 .+ exp.(-eta))
            @assert all(eta .>= IRT_SATURATION_ETA)
            @assert all(probability .>= 1.0 - 1.0e-12)
            @assert any(==(1.0), probability)
            q
        end,
    ),
    "two_pl" => (
        posterior = "fims_Aus_Jpn_irt-2pl_latent_reg_irt",
        module_name = "TwoplLatentRegIrtExample",
        build = TwoplLatentRegIrtExample.build_2pl_latent_reg_irt_graph,
        have = (:unconstrained, :ii, :jj, :y, :W, :I),
        bound = (; ii = TwoplLatentRegIrtExample.TWOPL_LR_II,
                 jj = TwoplLatentRegIrtExample.TWOPL_LR_JJ,
                 y = TwoplLatentRegIrtExample.TWOPL_LR_Y,
                 W = TwoplLatentRegIrtExample.TWOPL_LR_W,
                 I = TwoplLatentRegIrtExample.TWOPL_LR_I),
        package_data = (; ii = TwoplLatentRegIrtExample.TWOPL_LR_II,
                        jj = TwoplLatentRegIrtExample.TWOPL_LR_JJ,
                        y = TwoplLatentRegIrtExample.TWOPL_LR_Y,
                        W = TwoplLatentRegIrtExample.TWOPL_LR_W,
                        I = TwoplLatentRegIrtExample.TWOPL_LR_I),
        raw_data = d -> (ii = Int.(d["ii"]), jj = Int.(d["jj"]),
                         y = Bool.(d["y"]), W = _float_matrix(d["W"]),
                         I = Int(d["I"])),
        dimension = 531,
        scale = 0.3,
        stress = nothing,
    ),
    "hier_2pl" => (
        posterior = "sat-hier_2pl",
        module_name = "Hier2plExample",
        build = Hier2plExample.build_hier_2pl_graph,
        have = (:unconstrained, :ii, :jj, :y, :I, :J),
        bound = (; ii = Hier2plExample.HIER_2PL_II,
                 jj = Hier2plExample.HIER_2PL_JJ,
                 y = Hier2plExample.HIER_2PL_Y,
                 I = Hier2plExample.HIER_2PL_I,
                 J = Hier2plExample.HIER_2PL_J),
        package_data = (; ii = Hier2plExample.HIER_2PL_II,
                        jj = Hier2plExample.HIER_2PL_JJ,
                        y = Hier2plExample.HIER_2PL_Y,
                        I = Hier2plExample.HIER_2PL_I,
                        J = Hier2plExample.HIER_2PL_J),
        raw_data = d -> (ii = Int.(d["ii"]), jj = Int.(d["jj"]),
                         y = Bool.(d["y"]), I = Int(d["I"]), J = Int(d["J"])),
        dimension = 669,
        scale = 0.3,
        stress = nothing,
    ),
    "gpcm" => (
        posterior = "timssAusTwn_irt-gpcm_latent_reg_irt",
        module_name = "GpcmLatentRegIrtExample",
        build = GpcmLatentRegIrtExample.build_gpcm_latent_reg_irt_graph,
        have = (:unconstrained, :ii, :jj, :y, :W, :I),
        bound = (; ii = GpcmLatentRegIrtExample.GPCM_LR_II,
                 jj = GpcmLatentRegIrtExample.GPCM_LR_JJ,
                 y = GpcmLatentRegIrtExample.GPCM_LR_Y,
                 W = GpcmLatentRegIrtExample.GPCM_LR_W,
                 I = GpcmLatentRegIrtExample.GPCM_LR_I),
        package_data = (; ii = GpcmLatentRegIrtExample.GPCM_LR_II,
                        jj = GpcmLatentRegIrtExample.GPCM_LR_JJ,
                        y = GpcmLatentRegIrtExample.GPCM_LR_Y,
                        W = GpcmLatentRegIrtExample.GPCM_LR_W,
                        I = GpcmLatentRegIrtExample.GPCM_LR_I),
        raw_data = d -> (ii = Int.(d["ii"]), jj = Int.(d["jj"]),
                         y = Int.(d["y"]), W = _float_matrix(d["W"]),
                         I = Int(d["I"])),
        dimension = 530,
        scale = 0.3,
        stress = nothing,
    ),
)

function _parse_requested(raw, valid)
    tokens = map(strip, split(raw, ','))
    any(isempty, tokens) &&
        throw(ArgumentError("IRT_MODELS contains an empty selection token"))
    duplicates = unique(tokens[findall(i -> count(==(tokens[i]), tokens) > 1, eachindex(tokens))])
    isempty(duplicates) ||
        throw(ArgumentError("IRT_MODELS contains duplicate selections: $(join(duplicates, ", "))"))
    unknown = filter(!in(valid), tokens)
    isempty(unknown) ||
        throw(ArgumentError("unknown IRT acceptance models: $(join(unknown, ", "))"))
    tokens
end
_rejects_requested(raw) = try
    _parse_requested(raw, keys(CASES))
    false
catch error
    error isa ArgumentError || rethrow()
    true
end
@assert _parse_requested("irt_2pl, gpcm", keys(CASES)) == ["irt_2pl", "gpcm"]
@assert _rejects_requested("irt_2pl,irt_2pl")
@assert _rejects_requested("irt_2pl,,gpcm")
@assert _rejects_requested("irt_2pl,unknown")
println("IRT selection-parser controls: positive + duplicate/empty/unknown negative PASS")
const REQUESTED = _parse_requested(
    get(ENV, "IRT_MODELS", "irt_2pl,two_pl,hier_2pl,gpcm"), keys(CASES),
)
const RAN = String[]

println("IRT four-axis acceptance phase=", DO_REACTANT ? "reactant-loaded" : "native-unloaded")
println("source_head=", _repository_head())
println("julia=", VERSION, " ReactiveKernels=", _version_label(ReactiveKernels),
        " ReactiveKernelsPPLExamples=", _version_label(ReactiveKernelsPPLExamples))
println("BridgeStan=", _version_label(BridgeStan),
        " PosteriorDB=", _version_label(PosteriorDB),
        " Enzyme=", _version_label(Enzyme),
        " DifferentiationInterface=", _version_label(DifferentiationInterface),
        DO_REACTANT ? string(" Reactant=", _version_label(Reactant)) : "")
println("native_AD=ordinary AutoEnzyme(mode=Enzyme.Reverse)",
        " stan_query=propto=false,jacobian=true")
println("probe_seed=", PROBE_SEED, " native_probe_scale=", NATIVE_PROBE_SCALE,
        " native_probes_per_model=", NATIVE_PROBE_COUNT,
        " reactant_ordinary_probes_per_model=1",
        " irt_probability_saturation_stress_probes=",
        "irt_2pl" in REQUESTED ? 1 : 0,
        " eta>=", IRT_SATURATION_ETA)
println("requested models: ", join(REQUESTED, ','))
flush(stdout)

for key in REQUESTED
    case = CASES[key]
    println("\n########## $(case.posterior) ##########")
    flush(stdout)

    stan_model, posterior, stan_path, data_json_bytes =
        _stan_model(case.posterior, PROBE_SEED)
    println("  stan=", stan_path)
    println("  data_json_bytes=", data_json_bytes)
    dataset = PosteriorDB.load(PosteriorDB.dataset(posterior))
    raw = case.raw_data(dataset)
    for (name, observed) in pairs(raw)
        expected = getfield(case.package_data, name)
        @assert observed == expected "package/data mismatch for $(case.module_name).$name"
    end
    dimension = Int(BridgeStan.param_unc_num(stan_model))
    @assert dimension == case.dimension
    points = _reference_finite_points(
        stan_model, case.posterior, dimension, PROBE_SEED,
        NATIVE_PROBE_COUNT, NATIVE_PROBE_SCALE,
    )
    stress_q = case.stress === nothing ? nothing : case.stress()

    first_value, kernel = _first_use(case, points[1])
    @assert isfinite(first_value) "$(case.posterior): first-use value was not finite"
    prepared = prepare_ad(kernel, AD_BACKEND, points[1]; active = :unconstrained)

    max_value_error = 0.0
    max_gradient_error = 0.0
    for q in points
        rk_value = kernel(q)
        stan_value = _stan_value(stan_model, q)
        value_error = _relative_scalar(rk_value, stan_value)
        @assert isfinite(rk_value) "$(case.posterior): native value was not finite"
        @assert value_error < VALUE_TOL
        max_value_error = max(max_value_error, value_error)

        rk_gradient = ReactiveKernels.ad_value_and_gradient!(
            prepared, similar(q), q,
        )[2]
        stan_gradient = _stan_gradient(stan_model, q)
        gradient_error = _relative_vector(rk_gradient, stan_gradient)
        @assert all(isfinite, rk_gradient)
        @assert gradient_error < GRAD_TOL
        max_gradient_error = max(max_gradient_error, gradient_error)
    end
    println("  native value max_rel=", round(max_value_error; sigdigits = 4),
            " native-gradient max_rel=", round(max_gradient_error; sigdigits = 4))

    if stress_q !== nothing
        q = stress_q
        value = kernel(q)
        gradient = ReactiveKernels.ad_value_and_gradient!(
            prepared, similar(q), q,
        )[2]
        @assert isfinite(value) && all(isfinite, gradient)
        @assert _relative_scalar(value, _stan_value(stan_model, q)) < VALUE_TOL
        @assert _relative_vector(gradient, _stan_gradient(stan_model, q)) < GRAD_TOL
        println("  probability-saturation stress eta>=", IRT_SATURATION_ETA,
                " native ordinary-Reverse value+gradient finite and matched")
    end

    _reactant_axes(case, kernel, prepared, stan_model, points[1], stress_q)
    push!(RAN, key)
    flush(stdout)
end

@assert length(RAN) == length(REQUESTED)
@assert Set(RAN) == Set(REQUESTED)
if !DO_REACTANT
    @assert !_pkg_loaded("Reactant") "native phase loaded Reactant during acceptance"
end
println("\nIRT four-axis acceptance complete: models_ran=", length(RAN),
        " phase=", DO_REACTANT ? "reactant-loaded" : "native-unloaded")
