# data: subject male age_yr weight_kg indication qt_prolonging_drug_ongoing ydVc ydK10 ydK12 ydK21 ydKa ydQb ydQs ydTg ydKd ydLy0
begin
    standardize_age_yr = (age_yr .- mean(age_yr)) ./ std(age_yr)
    standardize_weight_kg = (weight_kg .- mean(weight_kg)) ./ std(weight_kg)
    a_Vc ~ Normal(2.302585092994046, 0.8)
    b_Vc_male ~ Normal(0.0, 0.1)
    b_Vc_age ~ Normal(0.0, 0.1)
    b_Vc_wt ~ Normal(0.0, 0.1)
    c_Vc[levels(indication)[2:end]] .~ Normal.(0.0, 1.0)
    a_k10 ~ Normal(-1.405170185988091, 0.8)
    b_k10_male ~ Normal(0.0, 0.1)
    b_k10_age ~ Normal(0.0, 0.1)
    b_k10_wt ~ Normal(0.0, 0.1)
    c_k10[levels(indication)[2:end]] .~ Normal.(0.0, 1.0)
    a_k12 ~ Normal(-0.34657359027997265, 2.0)
    c_k12[levels(indication)[2:end]] .~ Normal.(0.0, 1.0)
    a_k21 ~ Normal(-2.649158683274018, 2.0)
    c_k21[levels(indication)[2:end]] .~ Normal.(0.0, 1.0)
    a_ka ~ Normal(-2.0794415416798357, 0.8)
    c_ka[levels(indication)[2:end]] .~ Normal.(0.0, 1.0)
    a_qb ~ Normal(0.0, 0.8333333333333334)
    b_qb_male ~ Normal(0.0, 0.8333333333333334)
    b_qb_age ~ Normal(0.0, 0.8333333333333334)
    b_qb_ongoing ~ Normal(0.0, 0.8333333333333334)
    c_qb[levels(indication)[2:end]] .~ Normal.(0.0, 0.8333333333333334)
    a_qs ~ Normal(0.0, 0.8333333333333334)
    c_qs[levels(indication)[2:end]] .~ Normal.(0.0, 0.8333333333333334)
    a_tg ~ Normal(0.0, 1.0)
    a_tkd ~ Normal(0.0, 1.5)
    a_tb ~ Normal(3.5, 1.5)
    sigma_add ~ Exponential(0.25)
    sigma_prop ~ Exponential(0.25)
    qt_scale ~ LogNormal(0.0, 1.0)
    tgi_sigma ~ LogNormal(-2.0402208285265546, 0.5)
    tgi_c_cr ~ truncated(Normal(-2.3, 1.0), -Inf, -0.6931471805599453)
    d_p ~ varying_draws(subject, [1, 1, 1, 1, 1, 1, 1]; eta = 2.0)
    r_Vc ~ varying_slice(d_p, 1)
    r_k10 ~ varying_slice(d_p, 2)
    r_k12 ~ varying_slice(d_p, 3)
    r_k21 ~ varying_slice(d_p, 4)
    r_ka ~ varying_slice(d_p, 5)
    r_qb ~ varying_slice(d_p, 6)
    r_qs ~ varying_slice(d_p, 7)
    d_tg ~ varying_draws(subject, [1, 1]; eta = 2.0)
    r_tg ~ varying_slice(d_tg, 1)
    r_tkd ~ varying_slice(d_tg, 2)
    d_tb ~ varying_draws(subject, [1]; eta = 1.0)
    r_tb ~ varying_slice(d_tb, 1)
    log_Vc = a_Vc .+ b_Vc_male .* male .+ b_Vc_age .* standardize_age_yr .+ b_Vc_wt .* standardize_weight_kg .+ c_Vc[indication] .+ r_Vc
    log_k10 = a_k10 .+ b_k10_male .* male .+ b_k10_age .* standardize_age_yr .+ b_k10_wt .* standardize_weight_kg .+ c_k10[indication] .+ r_k10
    log_k12 = a_k12 .+ c_k12[indication] .+ r_k12
    log_k21 = a_k21 .+ c_k21[indication] .+ r_k21
    log_ka = a_ka .+ c_ka[indication] .+ r_ka
    qt_base = a_qb .+ b_qb_male .* male .+ b_qb_age .* standardize_age_yr .+ b_qb_ongoing .* qt_prolonging_drug_ongoing .+ c_qb[indication] .+ r_qb
    qt_slope = a_qs .+ c_qs[indication] .+ r_qs
    log_tgi_kg = a_tg .+ r_tg
    log_tgi_kd = a_tkd .+ r_tkd
    tgi_ly0 = a_tb .+ r_tb
    ydVc .~ Normal.(log_Vc, 1.5)
    ydK10 .~ Normal.(log_k10, 1.6)
    ydK12 .~ Normal.(log_k12, 1.7)
    ydK21 .~ Normal.(log_k21, 1.8)
    ydKa .~ Normal.(log_ka, 1.9)
    ydQb .~ Normal.(qt_base, 2.0)
    ydQs .~ Normal.(qt_slope, 2.1)
    ydTg .~ Normal.(log_tgi_kg, 2.2)
    ydKd .~ Normal.(log_tgi_kd, 2.3)
    ydLy0 .~ Normal.(tgi_ly0, 2.4)
end
