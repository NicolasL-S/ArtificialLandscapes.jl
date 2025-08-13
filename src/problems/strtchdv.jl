strtchdvi(x1, x2) = (x1^2 + x2^2)^(1//4) * (sin(50(x1^2+ x2^2)^(1//10)) + 1)^2

strtchdv(x) = sumi(i -> strtchdvi(x[i + 1], x[i]), zero(eltype(x)), dindices(x, 0,-1))

function strtchdv_grad!(gradient, x)
    check_gradient_indices(gradient, x)
	gradient .= 0
    for i in dindices(x, 0, -1)
        t = x[i + 1]^2 + x[i]^2
		t_01 = t^(1//10)
		a = sin(50t_01) + 1
		dsdtx2 = 2t^(1//4) * a * (0.25a + 10 * cos(50t_01) * t_01) / t
		gradient[i] += x[i] * dsdtx2
		gradient[i + 1] += x[i + 1] * dsdtx2
    end
    return gradient
end

"""
NAME          STRTCHDV

*   Problem :
*   *********

*   SCIPY global optimization benchmark example StretchedV

*   Fit: (x_i^2+x_i+1^2)^1/8 [ sin( 50(x_i^2+x_i+1^2)^1/10 ) + 1 ] + e = 0

*   Source:  Problem from the SCIPY benchmark set
*     https://github.com/scipy/scipy/tree/master/benchmarks/ ...
*             benchmarks/go_benchmark_functions

*   SIF input: Nick Gould, Jan 2020

*   classification SUR2-MN-V-0

*   Number of variables

 IE N                   10             -PARAMETER
*IE N                   100            -PARAMETER
*IE N                   1000           -PARAMETER
...

Keyword argument:
; n = 10, T = typeof(1.)
"""
function gen_strtchdv(;n = 10, T = typeof(1.))
    x0 = -ones(T, n)
    x0[1] = 1
    return (x0 = x0, obj = strtchdv, grad! = strtchdv_grad!)
end