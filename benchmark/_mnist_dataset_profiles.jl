using DelimitedFiles
using LinearAlgebra
using SHA
using Statistics
import MLDatasets

const MNIST_DATASET_FULL_RAW = "full-raw"
const MNIST_DATASET_WREN_PCA40 = "wren-pca40"
const MNIST_WREN_OBSERVATIONS = 1000
const MNIST_WREN_COMPONENTS = 40

function _mnist_dataset_profile()
    values = String[]
    for arg in ARGS
        startswith(arg, "--dataset=") &&
            push!(values, split(arg, '='; limit = 2)[2])
    end
    length(values) <= 1 || error("--dataset may be supplied only once")
    profile = isempty(values) ? MNIST_DATASET_FULL_RAW : only(values)
    profile in (MNIST_DATASET_FULL_RAW, MNIST_DATASET_WREN_PCA40) || error(
        "unknown MNIST dataset profile $profile; expected " *
        "$MNIST_DATASET_FULL_RAW or $MNIST_DATASET_WREN_PCA40")
    profile
end

function _mnist_wren_reference_path()
    values = String[]
    for arg in ARGS
        startswith(arg, "--wren-reference=") &&
            push!(values, split(arg, '='; limit = 2)[2])
    end
    length(values) <= 1 || error("--wren-reference may be supplied only once")
    isempty(values) ? nothing : only(values)
end

_mnist_default_observations(profile) =
    profile == MNIST_DATASET_WREN_PCA40 ? MNIST_WREN_OBSERVATIONS : 60000

function _mnist_wren_reference!(metadata, path, X, y)
    isfile(path) || error("Wren reference CSV does not exist: $path")
    reference, header = readdlm(path, ',', Float64; header = true)
    expected_header = vcat("label", ["d$i" for i in 1:MNIST_WREN_COMPONENTS])
    vec(string.(header)) == expected_header ||
        error("Wren reference header must be label,d1,...,d40")
    size(reference, 2) == MNIST_WREN_COMPONENTS + 1 ||
        error("Wren reference must contain label plus 40 PCA columns")
    size(reference, 1) >= size(X, 1) || error(
        "Wren reference has $(size(reference, 1)) rows but $(size(X, 1)) were requested")

    reference_y = Int.(reference[1:size(X, 1), 1])
    reference_X = reference[1:size(X, 1), 2:end]
    metadata["wren_reference_checked"] = true
    metadata["wren_reference_csv_sha256"] = bytes2hex(sha256(read(path)))
    metadata["wren_reference_label_mismatches"] = count(reference_y .!= y)
    metadata["wren_reference_max_abs_error"] = maximum(abs.(reference_X .- X))
    metadata["wren_reference_rmse"] = sqrt(mean(abs2, reference_X .- X))
    metadata
end

function _load_mnist_dataset(profile, n; wren_reference = nothing)
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
    train = MLDatasets.MNIST(split = :train)
    total = size(train.features, 3)
    1 <= n <= total || error(
        "requested $n MNIST images but the training split contains $total")
    y = Int.(train.targets[1:n]) .+ 1

    if profile == MNIST_DATASET_FULL_RAW
        wren_reference === nothing || error(
            "--wren-reference applies only to --dataset=$MNIST_DATASET_WREN_PCA40")
        pixels = reshape(train.features[:, :, 1:n], 28 * 28, n)
        X = Matrix{Float64}(transpose(pixels))
        metadata = Dict{String,Any}(
            "dataset_profile" => profile,
            "data" => "MLDatasets MNIST train split, first N images; 784 raw Float64 pixels in [0,1]",
            "num_observations" => n,
            "num_features" => size(X, 2),
        )
        return X, y, metadata
    end

    profile == MNIST_DATASET_WREN_PCA40 || error(
        "unsupported MNIST dataset profile $profile")
    n <= MNIST_WREN_OBSERVATIONS || error(
        "$profile contains the first $MNIST_WREN_OBSERVATIONS observations")

    # Wren's benchmark data are exactly the first 1,000 standard MNIST
    # training images projected onto the top 40 unwhitened principal
    # components fitted on the complete 60,000-image training split.
    pixels = Matrix{Float64}(reshape(train.features, 28 * 28, total))
    centered = pixels .- mean(pixels; dims = 2)
    decomposition = eigen(Symmetric(
        centered * transpose(centered) / (total - 1)))
    component_indices = lastindex(decomposition.values):-1:(
        lastindex(decomposition.values) - MNIST_WREN_COMPONENTS + 1)
    component_values = decomposition.values[component_indices]
    component_vectors = decomposition.vectors[:, component_indices]
    X = Matrix(transpose(centered[:, 1:n]) * component_vectors)
    metadata = Dict{String,Any}(
        "dataset_profile" => profile,
        "data" => "Wren-compatible MNIST: first N training images projected onto the top 40 unwhitened PCA components fitted on all 60000 training images",
        "num_observations" => n,
        "num_features" => size(X, 2),
        "pca_fit_observations" => total,
        "pca_input_features" => size(pixels, 1),
        "pca_components" => MNIST_WREN_COMPONENTS,
        "pca_centered" => true,
        "pca_whitened" => false,
        "pca_explained_variance_fraction" =>
            sum(component_values) / sum(decomposition.values),
        "wren_reference_checked" => false,
    )
    wren_reference === nothing ||
        _mnist_wren_reference!(metadata, wren_reference, X, y)
    X, y, metadata
end
