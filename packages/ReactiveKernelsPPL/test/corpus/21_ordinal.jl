# data: y x
begin
    eta = b .* x
    y .~ Ordinal.(StoppingRatio(), ProbitLink(), eta)
end
