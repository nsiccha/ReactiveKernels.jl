# data: y x
# A domain-fixed HSGP: the basis takes a fixed approximation domain instead
# of the data-derived one (synthetic toy program).
begin
    a ~ Normal(0, 1)
    (PHI, lambda) = hsgp_basis(x; k = 6, domain = (-5.0, 5.0))
    f ~ hsgp_effect(PHI, lambda)
    mu = a .+ f
    y .~ Normal.(mu, 1.5)
end
