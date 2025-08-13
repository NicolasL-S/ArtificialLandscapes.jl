function nondia(x, cache)
	check_x_indices(x, cache)
	return (x[begin] - 1)^2 + sumi(i -> 100(x[begin] - x[i - 1]^2)^2, cache, dindices(x,1,0))
end

function nondia_grad!(gradient, x)
    check_gradient_indices(gradient, x)
	suma = zero(eltype(x))
    @inbounds begin
        @simd for i in dindices(x,0,-1)
            a = 200(x[begin] - x[i]^2)
            suma += a
            gradient[i] = -a * 2x[i]
        end
        gradient[begin] += 2x[begin] - 2 + suma
		gradient[end] = 0
    end
    return gradient
end

"""
NAME          NONDIA

*   Problem :
*   --------

*   The Shanno nondiagonal extension of Rosenbrock function.

*   Source:
*   D. Shanno,
*   " On Variable Metric Methods for Sparse Hessians II: the New
*   Method",
*   MIS Tech report 27, University of Arizona (Tucson, UK), 1978.

*   See also Buckley #37 (p. 76) and Toint #15.

*   SIF input: Ph. Toint, Dec 1989.

*   classification SUR2-AN-V-0

*   Number of variables

*IE N                   10             -PARAMETER
*IE N                   20             -PARAMETER
*IE N                   30             -PARAMETER
*IE N                   50             -PARAMETER
*IE N                   90             -PARAMETER
*IE N                   100            -PARAMETER
*IE N                   500            -PARAMETER
*IE N                   1000           -PARAMETER     original value
 IE N                   5000           -PARAMETER
*IE N                   10000          -PARAMETER
...

Keyword argument:
; n = 5000, T = typeof(1.)
"""
function gen_nondia(; n = 5000, T = typeof(1.))
	return (x0 = -ones(T, n), 
			obj = x -> nondia(x, Vector{T}(undef, n)),
			grad! = nondia_grad!)
end