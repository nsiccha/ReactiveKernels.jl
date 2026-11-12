# data: y X Km1
# A categorical-logit response over one row of logits per observation
# (RKPPL 0.1.169). Row i of E holds observation i's K−1 non-reference
# logits, class 1 the zero reference, exactly as the positional
# `CategoricalLogit.(eta_2, …, eta_K)` reads its arguments; K is the row
# length + 1 when the model runs, so one program serves every class count.
# `eachcol(E)` pairs columns instead, an inline matrix expression
# (`eachrow(X * B)`) is one named value, and the `@plate` cell
# `y[i] ~ CategoricalLogit(E[i, :])` (or a cell local bound to that row)
# observes the same law. Weights, censoring and missing responses work as
# for the positional form; `mi()` missingness cannot select rows yet; a
# class beyond its row has log density -Inf. Tests:
# test_categorical_logit_slices.jl, test_leveled_k_invariance.jl.
begin
    B[axes(X, 2), 1:Km1] .~ Normal.(0, 1)
    E = X * B
    y .~ CategoricalLogit.(eachrow(E))
end
