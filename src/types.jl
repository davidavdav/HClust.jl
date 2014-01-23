## types.jl (c) 2014 David A. van Leeuwen 
## Hierarchical clustering, similar to R's hclust()

## Mostly following R's hclust class
type Hclust{T}
    merge::Matrix{Int}
    height::Vector{T}
    order::Vector{Int}
    labels::Vector
    method::Symbol
end
