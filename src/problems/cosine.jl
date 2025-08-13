function cosine(x, cache)
    check_x_indices(x, cache)
    return sumi(i -> cos(-x[i + 1] / 2 + x[i]^2), cache, dindices(x, 0, -1))
end

function cosine_grad!(gradient, x)
    check_gradient_indices(gradient, x)
    gradient[begin] = 0
    @inbounds for i in dindices(x, 0, -1)
        a = -sin(-x[i + 1] / 2 + x[i]^2)
        gradient[i] += 2a * x[i]
        gradient[i + 1] = -a / 2
    end
    return gradient
end

"""
NAME          COSINE 

*   Problem :
*   *********

*   Another function with nontrivial groups and
*   repetitious elements.

*   Source:
*   N. Gould, private communication.

*   SIF input: N. Gould, Jan 1996

*   classification OUR2-AN-V-0

*   number of variables

*IE N                   10             PARAMETER
*IE N                   100            PARAMETER
*IE N                   1000           PARAMETER     original value
 IE N                   10000          PARAMETER
...

Keyword arguments:
; n = 10_000, T = typeof(1.)
"""
function gen_cosine(; n = 10_000, T = typeof(1.))
    return (x0 = ones(T, n), obj = x -> cosine(x, Vector{T}(undef, n)), grad! = cosine_grad!)
end