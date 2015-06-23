## cutree.jl (c) 2014--2015 David A. van Leeuwen 
## cut a tree at height `h' or to `k' clusters
function cutree(hclust::Hclust; k::Int=1, 
                h::Real=maximum(hclust.height))
    clusters = Vector{Int}[]
    nnodes = length(hclust.labels)
    nodes = [[i::Int] for i=1:nnodes]
    N = nnodes - k
    i = 1
    while i<=N && hclust.height[i] <= h
        both = vec(hclust.merge[i,:])
        new = Int[]
            for x in both
                if x<0
                    push!(new, -x)
                    nodes[-x] = []
                else
                    append!(new, clusters[x])
                    clusters[x] = []
                end
            end
        push!(clusters, new)
        i += 1
    end
    all = vcat(clusters, nodes)
    all = all[map(length, all) .> 0]
    ## convert to a single array of cluster indices
    res = Array(Int, nnodes)
    for (i,cl) in enumerate(all)
        res[cl] = i
    end
    res
end

