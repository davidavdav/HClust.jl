## hclust.jl (c) 2014 David A. van Leeuwen 
## Hierarchical clustering, similar to R's hclust()

module HClust

export Hclust, hclust, cutree

include("types.jl")
include("hclust_impl.jl")
include("cutree.jl")

end
