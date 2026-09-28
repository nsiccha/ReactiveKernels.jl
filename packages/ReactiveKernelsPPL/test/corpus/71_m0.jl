# data: s
begin
    p ~ Beta(1.0, 1.0)
    zi ~ Beta(1.0, 1.0)
    s .~ ZeroInflatedBinomial.(3, p, zi)
end
