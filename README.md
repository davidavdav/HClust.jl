# HClust

[![Build Status](https://travis-ci.org/davidavdav/HClust.jl.png)](https://travis-ci.org/davidavdav/HClust.jl)

Hierarchical Clustering for Julia, similar to R's `hclust()`

Status
======

The package is currently work-in-progress.  Clustering involves doing a lot of admin, and it is easy to make an error.  I've tested the results for medium sized clusters (up to 250---5000) elements, for the following methods:

| method      | validated maximum size | time | validated |
|-------------|------------------------|------|-----------|
| `:single`   | 5000                   | 1.3  | OK
| `:complete` | 250                    | 31.8 | OK        
| `:average`  | 250                    | 31.7 | OK   
