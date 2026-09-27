# data: prop x z
begin
    mu = a .+ b .* x
    lk = c .+ d .* z
    prop .~ Beta.(logistic.(mu) .* exp.(lk), (1 .- logistic.(mu)) .* exp.(lk))
end
