function fminsurf(x_vec)
    n = lastindex(x_vec)
    Fp = sqrt(n)
    p = Int(Fp)
    x = reshape(x_vec, (p, p))
    scale = (Fp - 1)^2
    inv_scale = 1/scale
    mid = p ÷ 2
	s = (sum(x)/Fp^2)^2
    for i in 1:p - 1, j in 1:p - 1
        s += sqrt(scale / 2 * ((x[i, j] - x[i + 1, j + 1])^2 + 
             (x[i + 1, j] - x[i, j + 1])^2) + 1) * inv_scale
    end
    return s
end

function fminsurf_grad!(gradient_vec, x_vec)
    check_gradient_indices(x_vec, gradient_vec)
    n = lastindex(x_vec)
    Fp = sqrt(n)
    p = Int(Fp)
    x = reshape(x_vec, (p, p))
    gradient = reshape(gradient_vec, (p, p))
    scale = (Fp - 1)^2
    gradient .= 2sum(x)/Fp^4
    for i in 1:p - 1, j in 1:p - 1
        a = 1/(2sqrt(scale/2 * ((x[i, j] - x[i + 1, j + 1])^2 + (x[i + 1, j] - x[i, j + 1])^2) + 1))
        gradient[i, j] += a * (x[i, j] - x[i + 1, j + 1])
        gradient[i + 1, j + 1] -= a * (x[i, j] - x[i + 1, j + 1])
        gradient[i + 1, j] += a * (x[i + 1, j] - x[i, j + 1])
        gradient[i, j + 1] -= a * (x[i + 1, j] - x[i, j + 1])
    end
    return gradient_vec
end

"""
NAME          FMINSURF

*   Problem :
*   *********

*   The free boundary minimum surface problem.

*   The problem comes from the discretization of the minimum surface
*   problem on the unit square with "free boundary conditions"
*   one must find the minumum surface over the unit square 
*   (which is clearly 1.0).  Furthermore, the average distance of the surface
*   from zero is also minimized.

*   The Hessian is dense.

*   The unit square is discretized into (p-1)**2 little squares. The
*   heights of the considered surface above the corners of these little
*   squares are the problem variables,  There are p**2 of them.
*   Given these heights, the area above a little square is
*   approximated by the
*     S(i,j) = sqrt( 1 + 0.5(p-1)**2 ( a(i,j) + b(i,j) ) ) / (p-1)**2
*   where
*     a(i,j) = x(i,j) - x(i+1,j+1)
*   and
*     b(i,j) = x(i+1,j) - x(i,j+1)

*   Source: setting the boundary free in 
*   A Griewank and Ph. Toint,
*   "Partitioned variable metric updates for large structured
*   optimization problems",
*   Numerische Mathematik 39:429-448, 1982.

*   SIF input: Ph. Toint, November 1991.

*   classification OUR2-MY-V-0

*   P is the number of points in one side of the unit square

*IE P                   4              -PARAMETER n = 16     original value
*IE P                   7              -PARAMETER n = 49
*IE P                   8              -PARAMETER n = 64
*IE P                   11             -PARAMETER n = 121
*IE P                   31             -PARAMETER n = 961
*IE P                   32             -PARAMETER n = 1024
 IE P                   75             -PARAMETER n = 5625
*IE P                   100            -PARAMETER n = 10000
*IE P                   125            -PARAMETER n = 15625
...

Keyword argument:
; p = 75, T = typeof(1.)
"""
gen_fminsurf(; p = 75, T = typeof(1.)) = (x0 = T.(x0_fminsurf(p)), obj = fminsurf, grad! = fminsurf_grad!)