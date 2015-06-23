## hclust.jl (c) 2014--2015 David A. van Leeuwen 
## Hierarchical clustering, similar to R's hclust()

module HClust

export Hclust, hclust, cutree

include("types.jl")
include("main.jl")
include("cutree.jl")

end
