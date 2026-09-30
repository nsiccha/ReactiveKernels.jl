# data: log_obs log_time log_dose affectable biomarker person lloq uloq
# Bordet `brm_group_specific_hsgp_mean`: per-biomarker HSGP smooths
# (`by=`) whose log length-scale / log sd each take a per-biomarker
# hyper-predictor `1 + (1 | biomarker)` (BRM default hyper priors).
begin
    hsgp_basis(:h_t, log_time; k = 10, by = biomarker,
        length_scale = 1 + (1 | biomarker), sd = 1 + (1 | biomarker))
    hsgp_basis(:h_d, log_dose; k = 10, by = biomarker,
        length_scale = 1 + (1 | biomarker), sd = 1 + (1 | biomarker))
    r_b ~ varying_effect(biomarker, [1])
    r_p ~ varying_effect(person, [1])
    log_y = a .+ b_aff .* affectable .+ hsgp(:h_t) .+ hsgp(:h_d) .+ r_b .+
        r_p
    ls = c0
    log_obs .~ censored.(Normal.(log_y, exp.(ls)), lloq, uloq)
end
