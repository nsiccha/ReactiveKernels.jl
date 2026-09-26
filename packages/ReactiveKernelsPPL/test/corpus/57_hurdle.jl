# data: y x
begin
    a ~ Normal(0, 1)
    b ~ Normal(0, 2)
    p_zero ~ Beta(2, 2)
    eta = a .+ b .* x
    y .~ HurdlePoisson.(exp.(eta), p_zero)
end
