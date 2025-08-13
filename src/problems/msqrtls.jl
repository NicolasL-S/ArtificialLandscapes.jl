function msqrtls(x_vec, A, cache)
    check_x_indices(x_vec, A)
    P = Int(sqrt(lastindex(x_vec)))
    x = reshape(x_vec, (P, P))
    cache .= (x * x .- A).^2
    return sum(cache)
end

function msqrtls_grad!(gradient_vec, x_vec, A, cache)
    check_x_indices(x_vec, A)
    check_gradient_indices(x_vec, gradient_vec)
    P = Int(sqrt(lastindex(x_vec)))
    x = reshape(x_vec, (P, P))
    gradient = reshape(gradient_vec, (P, P))
    cache .= 2(x * x .- A)
    mul!(gradient, x',cache)
    mul!(gradient, cache, x', 1, 1)
    return vec(gradient)
end

"""
NAME          MSQRTALS

*   Problem :
*   *********

*   The dense matrix square root problem by Nocedal and Liu (Case 0).

*   This is a least-squares variant of problem MSQRTA.

*   Source:  problem 201 (p. 93) in
*   A.R. Buckley,
*   "Test functions for unconstrained minimization",
*   TR 1989CS-3, Mathematics, statistics and computing centre,
*   Dalhousie University, Halifax (CDN), 1989.

*   SIF input: Ph. Toint, Dec 1989.

*   classification SUR2-AN-V-V

*   Dimension of the matrix

*IE P                   2              -PARAMETER n = 4     original value
*IE P                   7              -PARAMETER n = 49
*IE P                   10             -PARAMETER n = 100
*IE P                   23             -PARAMETER n = 529
 IE P                   32             -PARAMETER n = 1024
*IE P                   70             -PARAMETER n = 4900

Keyword arguments:
;P = 32, T = typeof(1.)
"""
function gen_msqrtals(;P = 32, T = typeof(1.))
    B = T[sin(((i - 1) * P + j)^2) for i in 1:P, j in 1:P]'
    A = B * B
    cache = similar(A)
    return (x0 = 0.2vec(B),
            obj = x_vec -> msqrtls(x_vec, A, cache), 
            grad! = (gradient_vec, x_vec) -> msqrtls_grad!(gradient_vec, x_vec, A, cache))
end

"""
NAME          MSQRTBLS

*   Problem :
*   *********

*   The dense matrix square root problem by Nocedal and Liu (Case 1)

*   This is a least-squares variant of problem MSQRTB.

*   Source:  problem 204 (p. 93) in
*   A.R. Buckley,
*   "Test functions for unconstrained minimization",
*   TR 1989CS-3, Mathematics, statistics and computing centre,
*   Dalhousie University, Halifax (CDN), 1989.

*   SIF input: Ph. Toint, Dec 1989.

*   classification SUR2-AN-V-V

*   Dimension of the matrix ( at least 3)

*IE P                   3              -PARAMETER n = 9     original value
*IE P                   7              -PARAMETER n = 49
*IE P                   10             -PARAMETER n = 100
*IE P                   23             -PARAMETER n = 529
 IE P                   32             -PARAMETER n = 1024
*IE P                   70             -PARAMETER n = 4900
...

Keyword arguments:
;P = 32, T = typeof(1.)
"""
function gen_msqrtbls(;P = 32, T = typeof(1.))
    B_init = T[sin(((i-1)*P + j)^2) for i in 1:P, j in 1:P]'
    B = copy(B_init)
    B[1,3] = 0.
    A = B * B
    cache = similar(A)
    return (x0 = vec(B - 0.8B_init), # This is an awkward way of computing 0.2B, but necessary to get the same starting point as in the .sif file.
            obj = x_vec -> msqrtls(x_vec, A, cache), 
            grad! = (gradient_vec, x_vec) -> msqrtls_grad!(gradient_vec, x_vec, A, cache))
end