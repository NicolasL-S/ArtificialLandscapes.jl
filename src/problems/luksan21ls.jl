function luksan21ls(x, c)
    h = 1 / eltype(x)(length(x) + 1)
    hh = h^2
    @inbounds begin
        s = (2x[begin] + hh / 2 * (x[begin] + h + 1)^3 - x[begin + 1] + c)^2
        @simd for k in dindices(x,1,-1)
            s += (2x[k] + hh / 2 * (x[k] + k * h + 1)^3 - x[k - 1] - x[k + 1] + c)^2
        end
        s += (2x[end] + hh / 2 * (x[end] + length(x) * h + 1)^3 - x[end - 1] + c)^2
    end
    return s
end

function luksan21ls_grad!(gradient, x, c)
    check_gradient_indices(gradient, x)
	length(x) >= 2 || throw(BoundsError("x should have length at least 2"))
    T = eltype(x)
    h = 1 / T(length(x) + 1)
    hh = h^2
    @inbounds begin
        a = 2(2x[begin] + hh / 2 * (x[begin] + h + 1)^3 - x[begin + 1] + c)
        gradient[begin] = a * (2 + T(1.5) * hh * (x[begin] + h + 1)^2)
        gradient[begin + 1] = -a
        @simd for k in dindices(x, 1, -1)
            a = 2(2x[k] + hh / 2 * (x[k] + h * T(k) + 1)^3 - x[k - 1] - x[k + 1] + c)
            gradient[k-1] += -a
            gradient[k] += a * (2 + T(1.5) * hh * (x[k] + h * T(k) + 1)^2)
            gradient[k+1] = -a
        end
        a = 2(2x[end] + hh / 2 * (x[end] + length(x) * h + 1)^3 - x[end - 1] + c)
        gradient[end - 1] += -a
        gradient[end] += a * (2 + T(1.5) * hh * (x[end] + length(x) * h + 1)^2)
    end
    return gradient
end

"""
NAME          LUKSAN21LS

*   Problem :
*   *********

*   Problem 21 (modified discrete boundary value) in the paper

*     L. Luksan
*     Hybrid methods in large sparse nonlinear least squares
*     J. Optimization Theory & Applications 89(3) 575-595 (1996)

*   SIF input: Nick Gould, June 2017.

*   least-squares version

*   classification SUR2-AN-V-0
...

Keyword argument:
; n = 100, T = typeof(1.)
"""
function gen_luksan21ls(; n = 100, T = typeof(1.))
    c = 1 # Note: this c is there to be able to reuse the code for MOREBV with c = 0.
    return (x0 = T[l/(n+1) * (l/(n+1) - 1) for l in 1:n], 
            obj = x -> luksan21ls(x, c), 
            grad! = (gradient, x) -> luksan21ls_grad!(gradient, x, c))
end