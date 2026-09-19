# data: y
begin
    s ~ Dirichlet(3, 1.0)
    y .~ Categorical.(s)
end
