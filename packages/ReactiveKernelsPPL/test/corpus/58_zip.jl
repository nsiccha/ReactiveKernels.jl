# data: y x
begin
    a ~ Normal(0, 1)
    b ~ Normal(0, 2)
    zi ~ Beta(2, 2)
    eta = a .+ b .* x
    y .~ ZeroInflatedPoisson.(exp.(eta), zi)
end
