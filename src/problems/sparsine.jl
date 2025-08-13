function sparsine(x, sinx)
	check_x_indices(x, sinx)
    n = length(x)
    sinx .= sin.(x)
    s = 0.
    i2 = i3 = i5 = i7 = i11 = 0
    for i in eachindex(sinx)
        i2 += 2; i2 > n && (i2 -= n)
        i3 += 3; i3 > n && (i3 -= n)
        i5 += 5; i5 > n && (i5 -= n)
        i7 += 7; i7 > n && (i7 -= n)
        i11 += 11; i11 > n && (i11 -= n)
        s += i * (sinx[i] + sinx[i2] + sinx[i3] + sinx[i5] + sinx[i7] + sinx[i11])^2
    end
    return 0.5s
end

function sparsine_grad!(gradient, x, sinx)
	check_x_indices(x, sinx)
    check_gradient_indices(gradient, x)
    gradient .= 0
    n = length(x)
    sinx .= sin.(x)
    i2 = i3 = i5 = i7 = i11 = 0
    @inbounds for i in eachindex(sinx)
        i2 += 2; i2 > n && (i2 -= n)
        i3 += 3; i3 > n && (i3 -= n)
        i5 += 5; i5 > n && (i5 -= n)
        i7 += 7; i7 > n && (i7 -= n)
        i11 += 11; i11 > n && (i11 -= n)
        a = i * (sinx[i] + sinx[i2] + sinx[i3] + sinx[i5] + sinx[i7] + sinx[i11])
        gradient[i] += a
        gradient[i2] += a
        gradient[i3] += a
        gradient[i5] += a
        gradient[i7] += a
        gradient[i11] += a
    end
    gradient .*= cos.(x)
    return gradient
end

"""
NAME          SPARSINE

*   Problem :
*   *********

*   A sparse problem involving sine functions

*   SIF input: Nick Gould, November 1995

*   classification OUR2-AN-V-0

*   The number of variables 

*IE N                   10             -PARAMETER
*IE N                   50             -PARAMETER
*IE N                   100            -PARAMETER
*IE N                   1000           -PARAMETER     original value
 IE N                   5000           -PARAMETER
*IE N                   10000          -PARAMETER
...

Keyword argument:
; n = 5000, T = typeof(1.)
"""
function gen_sparsine(;n = 5000, T = typeof(1.))
    x0 = 0.5ones(T, n)
    sinx = similar(x0)
    return (x0 = x0, 
            obj = x -> sparsine(x, sinx), 
            grad! = (gradient, x) -> sparsine_grad!(gradient, x, sinx))
end