# data: y x
begin
    b ~ Normal(0, 1)
    eta = b .* x
    y .~ Ordinal.(StoppingRatio(), ProbitLink(), eta)
end
