function sparsqur(x, xsq)
	check_x_indices(x, xsq)
    n = length(x)
    xsq .= x.^2
    s = zero(eltype(x))
    i2 = i3 = i5 = i7 = i11 = 0
    @simd for i in eachindex(xsq)
        i2 += 2; i2 > n && (i2 -= n)
        i3 += 3; i3 > n && (i3 -= n)
        i5 += 5; i5 > n && (i5 -= n)
        i7 += 7; i7 > n && (i7 -= n)
        i11 += 11; i11 > n && (i11 -= n)
        s += i * (xsq[i] + xsq[i2] + xsq[i3] + xsq[i5] + xsq[i7] + xsq[i11])^2
    end
    return s / 8
end

function sparsqur_grad!(gradient, x, xsq)
    check_gradient_indices(gradient, x)
    eachindex(x) == eachindex(xsq) || throw(ArgumentError("xsq and x should have the same indices"))
    gradient .= 0
    n = length(x)
    xsq .= x.^2
    i2 = i3 = i5 = i7 = i11 = 0
    @inbounds for i in eachindex(xsq)
        i2 += 2; i2 > n && (i2 -= n)
        i3 += 3; i3 > n && (i3 -= n)
        i5 += 5; i5 > n && (i5 -= n)
        i7 += 7; i7 > n && (i7 -= n)
        i11 += 11; i11 > n && (i11 -= n)
        a = i * (xsq[i] + xsq[i2] + xsq[i3] + xsq[i5] + xsq[i7] + xsq[i11])
        gradient[i] += a
        gradient[i2] += a
        gradient[i3] += a
        gradient[i5] += a
        gradient[i7] += a
        gradient[i11] += a
    end
    gradient .*= x ./ 2
    return gradient
end

"""
NAME          SPARSQUR

*   Problem :
*   *********

*   A sparse quartic problem

*   SIF input: Nick Gould, November 1995

*   classification OUR2-AN-V-0

*   The number of variables 

*IE N                   10             -PARAMETER
*IE N                   50             -PARAMETER
*IE N                   100            -PARAMETER
*IE N                   1000           -PARAMETER     original value
*IE N                   5000           -PARAMETER
 IE N                   10000          -PARAMETER
 ...

Keyword argument:
; n = 10000, T = typeof(1.)
"""
function gen_sparsqur(; n = 10000, T = typeof(1.))
    x0 = ones(T, n) ./ 2
    xsq = similar(x0)
    return (x0 = x0, 
            obj = x -> sparsqur(x, xsq), 
            grad! = (gradient, x) -> sparsqur_grad!(gradient, x, xsq))
end