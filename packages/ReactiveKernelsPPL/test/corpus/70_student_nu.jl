# data: y x z
begin
    mu = a .+ b .* x
    lognu = c .+ d .* z
    y .~ StudentT.(exp.(lognu), mu, 2.0)
end
