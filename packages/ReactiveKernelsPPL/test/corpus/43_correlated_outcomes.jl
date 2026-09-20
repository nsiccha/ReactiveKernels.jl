# data: y1 y2 x
# Correlated outcomes: joint MvNormalCholesky response over an LKJ
# covariance factor (SB `[y1, y2] ~ MvNormalCholesky([mu1, mu2], L)` with
# `L ~ LKJCovarianceFactor(2, Exponential(1), 2)`).
begin
    a1 ~ Normal(0, 1)
    b1 ~ Normal(0, 1)
    a2 ~ Normal(0, 1)
    b2 ~ Normal(0, 1)
    mu1 = a1 .+ b1 .* x
    mu2 = a2 .+ b2 .* x
    L ~ LKJCovarianceFactor(2, Exponential(1.0), 2.0)
    [y1, y2] ~ MvNormalCholesky([mu1, mu2], L)
end
