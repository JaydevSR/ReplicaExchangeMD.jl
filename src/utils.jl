# returns k almost equal patitions of n
@inline function equal_parts(n, k)
    ndiv = n ÷ k
    nrem = n % k
    n_parts = ntuple(i -> (i <= nrem) ? ndiv+1 : ndiv, k)  # no allocation for k upto 10
    return n_parts
end