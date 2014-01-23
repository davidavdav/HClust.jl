## hclust.jl (c) 2014 David A. van Leeuwen 
## Hierarchical clustering, similar to R's hclust()

require("types.jl")

## This seems to work like R's implementation, but it is extremely inefficient
function hclust{T}(d::Matrix{T}, method::Function)
    @assert size(d,1) == size(d,2)
    if !issym(d)
        d += d'                 # replaces d, which must be symmetric
    end
    mr = Int[]                  # min row
    mc = Int[]                  # min col
    h = T[]                     # height
    nc = size(d,1)              # number of clusters
    cl = -[1:nc]                # segment to cluster attribution, initially negative
    next = 1                    # next cluster label
    while next < nc 
        min = Inf
        mi = mj = 0
        cli = unique(cl)
        mask = BitVector(nc)
        for j in 1:length(cli)           # loop over for lower triangular indices, i>j
            cols = cl .== cli[j]
            for i in (j+1):length(cli)
                rows = cl.==cli[i]
                distance = method(d[rows,cols])
                if distance < min
                    min = distance
                    mi = cli[i]
                    mj = cli[j]
                    mask = cols | rows
                end
            end
        end       
        push!(mr, mi)
        push!(mc, mj)
        push!(h, min)
        cl[mask] = next
        next += 1
    end
    hcat(mr, mc), h
end

function hclust{T}(d::Matrix{T}, method::Symbol)
    if method == :complete
        h = hclust(d, maximum)
    elseif method == :single
        h = hclust(d, minimum)
    elseif method == :average
        h = hclust(d, mean)
    else
        error("Unsupported method ", method)
    end
    nc = size(d,1)
    Hclust(h..., [1:(nc-1)
], [1:nc], method)
end

function test_hclust(N::Int)
    d = rand(N,N)
    d += d'
    writedlm("hclust.txt", d)
    hclust(d, :average)
end

