# data: y x z
begin
    mu = a .+ b .* x
    sigma = c .+ d .* z
    y .~ Normal.(mu, exp.(sigma))
end
