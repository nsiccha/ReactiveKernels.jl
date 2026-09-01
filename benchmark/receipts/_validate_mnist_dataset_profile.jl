if !isdefined(@__MODULE__, :EXPECTED_MNIST_WREN_REFERENCE_SHA256)
    const EXPECTED_MNIST_WREN_REFERENCE_SHA256 =
        "4d7543e3fdc89d51554c49c9ba71b0f629d3acf15e1f1a8bcbdc958be7f88af0"

    function _validate_mnist_dataset_profile!(
        require,
        protocol,
        matched_protocol,
    )
        profile = get(protocol, "dataset_profile", "full-raw")
        if profile == "full-raw"
            require(Int(get(protocol, "num_observations", 0)) ==
                    Int(get(matched_protocol, "num_observations", -1)),
                    "full-raw observation count must match the native receipt")
            require(Int(get(protocol, "num_features", 0)) == 784,
                    "full-raw MNIST receipts must use 784 pixel features")
            return
        end

        require(profile == "wren-pca40",
                "dataset_profile must be full-raw or wren-pca40")
        profile == "wren-pca40" || return
        require(Int(get(protocol, "num_observations", 0)) == 1000,
                "Wren-compatible receipts must contain 1000 observations")
        require(Int(get(protocol, "num_features", 0)) == 40,
                "Wren-compatible receipts must contain 40 PCA features")
        require(Int(get(protocol, "pca_fit_observations", 0)) == 60000,
                "Wren-compatible PCA must be fitted on all 60000 training images")
        require(Int(get(protocol, "pca_input_features", 0)) == 784,
                "Wren-compatible PCA must start from 784 raw pixels")
        require(Int(get(protocol, "pca_components", 0)) == 40,
                "Wren-compatible PCA must retain 40 components")
        require(get(protocol, "pca_centered", false) == true,
                "Wren-compatible PCA must center the full training matrix")
        require(get(protocol, "pca_whitened", true) == false,
                "Wren-compatible PCA scores must remain unwhitened")
        require(isapprox(
                    Float64(get(protocol, "pca_explained_variance_fraction", 0.0)),
                    0.7861077598655901; rtol = 1e-12, atol = 1e-12),
                "Wren-compatible PCA explained-variance fraction drifted")
        require(get(protocol, "wren_reference_checked", false) == true,
                "published Wren-compatible receipt must check the copied CSV")
        require(get(protocol, "wren_reference_csv_sha256", "") ==
                EXPECTED_MNIST_WREN_REFERENCE_SHA256,
                "Wren reference CSV digest mismatch")
        require(Int(get(protocol, "wren_reference_label_mismatches", -1)) == 0,
                "Wren reference labels must match the first 1000 MNIST labels")
        require(Float64(get(protocol, "wren_reference_max_abs_error", Inf)) <= 1e-12,
                "generated PCA matrix does not match Wren to roundoff")
        require(Float64(get(protocol, "wren_reference_rmse", Inf)) <= 1e-13,
                "generated PCA matrix RMSE exceeds roundoff")
    end
end
