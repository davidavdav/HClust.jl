## hclust.jl (c) 2014 David A. van Leeuwen 
## Hierarchical clustering, similar to R's hclust()

## Algorithms are based upon C. F. Olson, Parallel Computing 21 (1995) 1313--1325. 

require("types.jl")

## This seems to work like R's implementation, but it is extremely inefficient
## This probably scales O(n^3) or worse. We can use it to check correctness
function hclust_n3{T}(d::Matrix{T}, method::Function)
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
        ## simulate R's order
        if mi < 0 && mj < 0 && mi > mj ||
            mi > 0 && mj > 0 && mi < mj ||
            mi < 0 && mj > 0
            push!(mr, mi)
            push!(mc, mj)
        else
            push!(mr, mj)
            push!(mc, mi)
        end
        push!(h, min)
        cl[mask] = next
        next += 1
    end
    hcat(mr, mc), h
end

## Efficient single link algorithm, according to Olson, O(n^2), fig 2. 
## For each i < j compute D(i,j) (this is already given)
## For each 0 < i <= n compute Nearest Neighbor N(i)
## Repeat n-1 times
##   find i,j that minimize D(i,j)
##   merge clusters i and j
##   update D(i,j) and N(i) accordingly
function hclust_minimum{T}(d::Matrix{T})
    @assert size(d,1) == size(d,2)
    ## For each i < j compute d[i,j] (this is already given) --- we need a local copy
    if !issym(d)
        d += d'                
    else
        d = copy(d)
    end
    nc = size(d,1)
    mr = Array(Int, nc-1)       # min row
    mc = Array(Int, nc-1)       # min col
    h = Array(T, nc-1)          # height
    merges = -[1:nc]
    next = 1
    ## For each 0 < i <= n compute Nearest Neighbor N[i]
    N = zeros(Int, nc)
    for k = 1:nc
        min = Inf
        mk = 0
        for i = 1:(k-1)
            if d[i,k] < min
                min = d[i,k]
                mk = i
            end
        end
        for j = (k+1):nc
            if d[k,j] < min
                min = d[k,j]
                mk = j
            end
        end
        N[k] = mk
    end
#    printupper(d)
#    for n in N[1:nc] print(n, " ") end; println()
#    lastmin = -Inf
    ## the main loop
    while nc > 1                # O(n)
        min = d[1,N[1]]
        i = 1
        for k in 2:nc           # O(n)
            if k < N[k]
                distance = d[k,N[k]]            
            else
                distance = d[N[k],k]
            end
            if distance < min
                min = distance 
                i = k
            end
        end
 #       if (min < lastmin)
 #           error("Not finding consectutive minima ", min, " ", lastmin, " ", next)
 #       end
 #       lastmin = min
        j = N[i]
        if i > j
            i, j = j, i     # make sure i < j
        end
#        println("Merging ", i, " and ", j)
        ## update result, compatible to R's order.  It must be possible to do this simpler than this...
        if merges[i] < 0 && merges[j] < 0 && merges[i] > merges[j] ||
            merges[i] > 0 && merges[j] > 0 && merges[i] < merges[j] ||
            merges[i] < 0 && merges[j] > 0
            mr[next] = merges[i]
            mc[next] = merges[j]
        else
            mr[next] = merges[j]
            mc[next] = merges[i]
        end
        h[next] = min
        merges[i] = next
        merges[j] = merges[nc]
        ## update d, split in ranges k<i, i<k<j, j<k<=nc
        for k = 1:(i-1)         # k < i
            if d[k,i] > d[k,j]
                d[k,i] = d[k,j]
            end
        end
        for k = (i+1):(j-1)     # i < k < j
            if d[i,k] > d[k,j]
                d[i,k] = d[k,j]
            end
        end
        for k = (j+1):nc        # j < k <= nc
            if d[i,k] > d[j,k]
                d[i,k] = d[j,k]
            end
        end
        ## move the last row/col into j
        for k=1:(j-1)           # k==nc, 
            d[k,j] = d[k,nc]
        end
        for k=(j+1):(nc-1)
            d[j,k] = d[k,nc]
        end
 #       printupper(d[1:(nc-1),1:(nc-1)])
        ## update N[k], k !in (i,j)
        for k=1:nc
            if N[k] == j       # update nearest neigbors != i
                N[k] = i
            elseif N[k] == nc
                N[k] = j
            end
        end
        N[j] = N[nc]            # update N[j]
        ## update nc, next
        nc -= 1
        next += 1
        ## finally we need to update N[i], because it was nearest to j
        min = Inf
        mk = 0
        for k=1:(i-1)
            if d[k,i] < min
                min = d[k,i]
                mk = k
            end
        end
        for k = (i+1):nc
            if d[i,k] < min
                min = d[i,k]
                mk = k
            end
        end
        N[i] = mk
#        for n in N[1:nc] print(n, " ") end; println()
    end
    return hcat(mr, mc), h
end
        

## functions to comput maximum, minimum, mean for just a slice of an array
## Only consider the upper tiangle, i.e., i<j
function Base.maximum{T}(d::Matrix{T}, cl1::Vector{Int}, cl2::Vector{Int})
    max = -Inf
    mi = mj = 0
    for i in cl1 for j in cl2
        if d[i,j] > max
            max = d[i,j]
            mi = i
            mj = j
        end
    end end
    max
end

function Base.minimum{T}(d::Matrix{T}, cl1::Vector{Int}, cl2::Vector{Int})
    min = Inf
    mi = mj = 0
    for i in cl1 for j in cl2
        if d[i,j] < min
            min = d[i,j]
#            mi = i
#            mj = j
        end
    end end
    min
end
    
function Base.mean{T}(d::Matrix{T}, cl1::Vector{Int}, cl2::Vector{Int})
    s = zero(T)
    for i in cl1 for j in cl2
        s += d[i,j]
    end end
    s / (length(cl1)*length(cl2))
end

## Nearest Neighbor Chain algorithm, see Wikipedia
## It doesn't give me the same results, there must be something I miss. 
function hclust1{T}(d::Matrix{T}, method::Function)
    @assert size(d,1) == size(d,2)
    if !issym(d)
        d += d'                 # replaces d, which must be symmetric
    end
    mr = Int[]                  # min row
    mc = Int[]                  # min col
    h = T[]                     # height
    nc = size(d,1)
    cl = [1:nc]
    S = (Int,Vector{Int})[]     # stack of ID and cluster
    next = 1
    while next < nc
        if length(S) == 0
            id = shift!(cl)
            push!(S, (-id,[id]))
        end
        id, C = last(S)
        ## compute distance to single clusters
        min = Inf
        mi = 0
        for i in 1:length(cl)
            distance = method(d[cl[i],C])
            if distance < min
                mi = i
                min = distance
            end
        end
        ## and compare that to previous cluster on stack
        if mi>0
#            print("Minimum element ", cl[mi], " at ", min, " with C ", C')
        end
        if length(S) > 1 && (distance = method(d[S[end-1][2], C])) < min
            ## merge top two on stack
            id1, c1 = pop!(S)
            id2, c2 = pop!(S)
#            println("Merging ", id1, " and ", id2, " at ", distance)
            new = sort(vcat(c1, c2))
            push!(S, (next,new))
            push!(mr, id1)
            push!(mc, id2)
            push!(h, distance)
            next += 1
        else
            id = splice!(cl, mi)
            push!(S, (-id,[id]))
        end
    end
    o = sortperm(h)
    io = invperm(o)
    for i in 1:length(mr)
        if mr[i] > 0 
            mr[i] = io[mr[i]]
        end
        if mc[i] > 0
            mc[i] = io[mc[i]]
        end
        ## R's order of pairs
        if ! (mr[i] < 0 && mc[i] < 0 && mr[i] > mc[i] ||
              mr[i] > 0 && mc[i] > 0 && mr[i] < mc[i] ||
              mr[i] < 0 && mc[i] > 0)
            mr[i], mc[i] = mc[i], mr[i]
        end
    end
    hcat(mr[o], mc[o]), h[o]
end

## Another neirest neighbor algorithm, for reducible metrics
## From C. F. Olson, Parallel Computing 21 (1995) 1313--1325, fig 5
## Pick c1: 0 <= c1 <= n random
## i <- 1
## repeat n-1 times
##   repeat 
##     i++
##     c[i] = nearest neigbour c[i-1]
##   until c[i] = c[i-2] ## nearest of nearest is cluster itself
##   merge c[i] and nearest neigbor c[i]
##   if i>3 i -= 3 else i <- 1
function hclust2{T}(d::Matrix{T}, method::Function)
    @assert size(d,1) == size(d,2)
    if !issym(d)
        d += d'                 # replaces d, which must be symmetric
    end
    mr = Int[]                  # min row
    mc = Int[]                  # min col
    h = T[]                     # height
    nc = size(d,1)
    cl = map(x->[x], 1:nc)
    merges = -[1:nc]
    next = 1
    i = 1
    N = Array(Int, nc)
    N[1] = 1                      # arbitrary choice
    while nc > 1
        found=false
        min = Inf
        while !found
            i += 1
            mi = 0
            min = Inf
            Nim1 = N[i-1]
            ## c[i] = nearest neigbour c[i-1]
            for j = 1:nc if Nim1 != j
                distance = method(d, cl[Nim1], cl[j])
                if distance < min
                    min = distance
                    mi = j
                end
            end end
            N[i] = mi           # N[i+1] is nearest neigbor to N[i]
            found = i > 2 && N[i] == N[i-2]
        end
        ## merge c[i] and neirest neigbor c[i], i.e., c[i-1]
        lo, high = sort([N[i-1], N[i]])
        ## first, store the result
        push!(mr, merges[lo])
        push!(mc, merges[high])
        push!(h, min)
        merges[lo] = next
        merges[high] = merges[nc]
        next += 1
        ## then perform the actual merge
        cl[lo] = vcat(cl[lo], cl[high])
        cl[high] = cl[nc]
        nc -= 1
        if i>3
            i -= 3
        else
            i = 1
        end
        ## replace any nearest neighbor referring to nc
        for k=1:i
            if N[k] == nc+1
                N[k] = high
            end
        end
    end
    ## fix order for presenting result
    o = sortperm(h)
    io = invperm(o)
    for i in 1:length(mr)
        if mr[i] > 0 
            mr[i] = io[mr[i]]
        end
        if mc[i] > 0
            mc[i] = io[mc[i]]
        end
        ## R's order of pairs
        if ! (mr[i] < 0 && mc[i] < 0 && mr[i] > mc[i] ||
              mr[i] > 0 && mc[i] > 0 && mr[i] < mc[i] ||
              mr[i] < 0 && mc[i] > 0)
            mr[i], mc[i] = mc[i], mr[i]
        end
    end
    hcat(mr[o], mc[o]), h[o]
end

## this calls the routine that gives the correct answer, fastest
function hclust{T}(d::Matrix{T}, method::Symbol)
    if method == :single
        h = hclust_minimum(d)
    elseif method == :complete
        h = hclust2(d, maximum)
    elseif method == :average
        h = hclust2(d, mean)
    else
        error("Unsupported method ", method)
    end
    nc = size(d,1)
    ## order and label are placeholders at the moment
    Hclust(h..., [1:nc], [1:nc], method)
end

function cutree(hclust::Hclust; k::Int=Nothing, h::Real=Nothing)
    ## todo
end

## some diagnostic functions, not exported
function printupper(d::Matrix)
    n = size(d,1)
    for i = 1:(n-1)
        print(" " ^ ((i-1) * 6))
        for j = (i+1):n
            print(@sprintf("%5.2f ", d[i,j]))
        end
        println()
    end
end                
    
function test_hclust(N::Int, method=:complete)
    d = gendist(N)
    writedlm("hclust.txt", d)
    h = hclust(d, method)
    writedlm("merge.txt", h.merge)
    writedlm("height.txt", h.height)
    d
end

function gendist(N::Int)
    d = randn(N,N)
    d += d'
    d
end