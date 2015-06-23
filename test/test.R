compare <- function(method="single") {
  d <- scan("hclust.txt")
  n <- sqrt(length(d))
  d <- matrix(d, n, n)
  merge <- t(matrix(scan("merge.txt"), 2, n-1))
  height <- scan("height.txt")
  h <- hclust(as.dist(d), method)
  cat("Heights: ", sum(abs(h$height - height)), "\n")
  diffids <- merge[,1] != h$merge[,1]
  print(cbind(1:(n-1), merge, h$merge, h$height)[diffids,])
  d
}
