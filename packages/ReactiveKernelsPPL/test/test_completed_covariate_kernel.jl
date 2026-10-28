module CompletedCovariateKernelTests
using Test
# Regression: two completed covariates fused with correlated subject effects
# and a child KernelSpec. Native Reverse previously rejected the mixed
# constant/active descriptor at a non-inlined broadcast axis call.
using ReactiveKernels, ReactiveKernelsPPL, Enzyme, DifferentiationInterface
# Ordinary numerical leaves; the completed columns and predictors remain
# authored inside the PPL graph rather than hidden behind a helper.
completed_level_indices(labels, source) =
    [only(findall(==(v), labels)) for v in source]
completed_ranef_column(draws, index, margin) = draws[index, margin]
columns = (
    Jmis_age=[2], Jmis_weight=[3], Jobs_age=[1, 3], Jobs_weight=[1, 2],
    age_missing_lookup=[1, 1, 1], age_missing_mask=[0.0, 1.0, 0.0],
    age_obs=[3.2, 5.1], age_observed_component=[3.2, 0.0, 5.1],
    locations_subject_count=3, subject=[1, 2, 3],
    weight_missing_lookup=[1, 1, 1], weight_missing_mask=[0.0, 0.0, 1.0],
    weight_obs=[7.4, 8.3], weight_observed_component=[7.4, 8.3, 0.0],
    y=[1.1, 1.3, 0.9], z=[1.0, 0.8, 1.2],
)
@rkppl completed_correlated_random_coefficients(sd, L, z) = begin
    z * transpose(sd .* L)
end
ReactiveKernels.@kernel locations_cell(vc, k) = begin
    return vc
end
ReactiveKernels.@kernel locations_reader(locations_subject_count, Vc, k10) = begin
    cell_values = ReactiveKernels.plate(1:locations_subject_count) do subject
        cell_input_1 = Vc[subject]
        cell_input_2 = k10[subject]
        locations_cell(cell_input_1, cell_input_2)
    end
    result = convert(Vector{Float64}, reduce(vcat, cell_values; init=Float64[]))
    return result
end
model = @rkppl begin
    Vc_Intercept ~ Normal(0.0, 0.4)
    Vc_standardize_age ~ Normal(0.0, 0.4)
    Vc_standardize_weight ~ Normal(0.0, 0.4)
    k10_Intercept ~ Normal(0.0, 0.4)
    k10_standardize_age ~ Normal(0.0, 0.4)
    ranef_draws_p_subject_sd[1:2] .~ Exponential.(0.9)
    ranef_draws_p_subject_z[levels(subject), 1:2] .~ Normal.(0, 1)
    ranef_draws_p_subject_L ~ LKJCholesky(2, 1.0)
    ranef_draws_p_subject ~ completed_correlated_random_coefficients(ranef_draws_p_subject_sd, ranef_draws_p_subject_L, ranef_draws_p_subject_z)
    ranef_draws_p_subject_index_subject = completed_level_indices(subject, subject)
    ranef_Vc_p_subject = completed_ranef_column(ranef_draws_p_subject, ranef_draws_p_subject_index_subject, 1)
    ranef_k10_p_subject = completed_ranef_column(ranef_draws_p_subject, ranef_draws_p_subject_index_subject, 2)
    age_y_mis[axes(Jmis_age, 1)] .~ LogNormal.(1.4, 0.4)
    age = age_observed_component .+ age_missing_mask .* age_y_mis[age_missing_lookup]
    standardize_age = (age .- 4.15) ./ 1.34350288425444
    k10_ = .+(k10_Intercept .* ones(length(subject)), k10_standardize_age .* standardize_age, ranef_k10_p_subject)
    k10 = exp.(k10_)
    weight_y_mis[axes(Jmis_weight, 1)] .~ LogNormal.(2.0, 0.3)
    weight = weight_observed_component .+ weight_missing_mask .* weight_y_mis[weight_missing_lookup]
    standardize_weight = (weight .- 7.8500000000000005) ./ 0.6363961030678931
    Vc_ = .+(Vc_Intercept .* ones(length(subject)), Vc_standardize_age .* standardize_age, Vc_standardize_weight .* standardize_weight, ranef_Vc_p_subject)
    Vc = exp.(Vc_)
    locations = locations_reader(locations_subject_count, Vc, k10)
    age_obs .~ LogNormal.(1.4, 0.4)
    weight_obs .~ LogNormal.(2.0, 0.3)
    y .~ Normal.(locations, 0.8)
    z .~ Normal.(k10, 0.7)
end
bound = bind_data(lower_rkppl(model,columns;conditioned=(:age_obs, :weight_obs, :y, :z)),columns)
built = build_kernel(bound)
points = [
    [
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    ],
    [
        -0.2, -0.16666666666666666, -0.13333333333333333, -0.1,
        -0.06666666666666667, -0.03333333333333333, 0.0, 0.03333333333333333,
        0.06666666666666667, 0.1, 0.13333333333333333, 0.16666666666666666,
        0.2, 0.23333333333333334, 0.26666666666666666, 0.3,
    ],
    [
        -0.1, -0.1, -0.1, -0.1,
        -0.1, -0.1, -0.1, -0.1,
        -0.1, -0.1, -0.1, -0.1,
        -0.1, -0.1, -0.1, -0.1,
    ],
]
expected = [
    (-46.99807892597268, [
        0.4687500000000002, -1.3200101918531773, 1.9028047063179656, 0.0,
        1.2456016338839286, -0.11111111111111116, -0.11111111111111116, 0.15625000000000014,
        0.46875000000000006, -0.15624999999999997, 0.0, -0.40816326530612235,
        0.40816326530612235, 0.0, 8.75, 22.222222222222225,
    ]),
    (-43.09906754722113, [
        -9.2603346034158, -7.360139807361131, 112.29649431052462, -0.42145902438154415,
        3.1412952816065194, -1.1066458469434828, -0.27786337454247123, 0.00805457744308647,
        -0.03639153405679611, -10.577264786976315, -0.32409566760482844, -1.265217455109167,
        0.07070880595325141, -0.4741211271307146, 7.1080666078257355, 21.971069383510933,
    ]),
    (-52.07701672223896, [
        -3.660008194580721, -4.244385858497885, 55.32125891359343, 0.9173806122406127,
        2.338298895180242, 0.3823486661758908, -0.029062085473232604, 0.2818474163352538,
        0.5941714462838281, -4.479622471173918, 0.2718549366421072, -0.41222109266314133,
        0.7036057789724081, 0.17051944403904823, 9.38035320093431, 24.04345101322097,
    ]),
]
@testset "completed covariates enter a child kernel under native Reverse" begin
    @test built.layout.total == 16
    saved_columns = deepcopy(columns)
    sampler = prepare_sampler(built, bound, first(points); backend=AutoEnzyme(; mode=Enzyme.Reverse))
    for (u, (expected_value, expected_gradient)) in zip(points, expected)
        saved = copy(u); gradient = similar(u)
        value, _ = sampler_value_and_gradient!(sampler, gradient, u)
        @test isapprox(value, expected_value; atol=2e-11, rtol=2e-11)
        @test isapprox(gradient, expected_gradient; atol=2e-10, rtol=2e-10)
        @test isequal(u, saved) && isequal(columns, saved_columns)
    end
end
end
