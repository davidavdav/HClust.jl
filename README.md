# HClust

[![Build Status](https://travis-ci.org/davidavdav/HClust.jl.png)](https://travis-ci.org/davidavdav/HClust.jl)

Hierarchical Clustering for Julia, similar to R's `hclust()`

Status
======

The package is currently work-in-progress.  Clustering involves doing a lot of admin, and it is easy to make an error.  I've tested the results for medium sized clusters (up to 250---5000) elements, for the following methods:

| method      | validated at matrix size | time | validated |
|-------------|------------------------|------|-----------|
| `:single`   | 5000                   | 1.3  | OK
| `:complete` | 2500                   | 4.5  | OK        
| `:average`  | 2500                   | 4.5  | OK   

Usage
=====

```julia
using HClust 

d = rand(1000,1000)
d += d'  ## make sure distance matrix d is symmetric (this is optional)
h = hclust(d, :single)
```

Result
------

The output of `hclust()` is an object of type `Hclust` with the fields

 - `merge` the clusters merged in order.  Leafs are indicated by negative numbers
 - `height` the distance at which the merges take place
 - `order` a preferred grouping for drawing a dendogram.  Not implemented, always `[1:n]`. 
 - `labels` labels of the clusters.  Not implemented, now always `[1:n]`
 - `method` the name of the clustering method. 

