function palmer(x, Y, Xj, cache)
	check_x_indices(x, size(Xj,2))
	cache .= Y
	return sum(abs2, mul!(cache, Xj, x, -1, 1))
end

function palmer_grad!(gradient, x, Y, Xj, cache)
    check_gradient_indices(gradient, x)
	cache .= Y
	mul!(cache, Xj, x, -1, 1)
	return mul!(gradient, Xj', cache, -2, 0)
end

"""
NAME          PALMER1C

*   Problem :
*   *********

*   A linear least squares problem
*   arising from chemical kinetics.

*   model: H-N=N=N TZVP+MP2
*   fitting Y to A0 + A2 X**2 + A4 X**4 + A6 X**6 + A8 X**8 +
*                A10 X**10 + A12 X**12 + A14 X**14

*   Source:
*   M. Palmer, Edinburgh, private communication.

*   SIF input: Nick Gould, 1990.

*   classification QUR2-RN-8-0
...

Keyword argument:
; T = typeof(1.)
"""
function gen_palmer1c(; T = typeof(1.))
	X = T[-1.788963, -1.745329, -1.658063, -1.570796, -1.483530, -1.396263, -1.308997, -1.218612,
		 -1.134464, -1.047198, -0.872665, -0.698132, -0.523599, -0.349066, -0.174533, 0.0000000,
	 	 1.788963, 1.745329, 1.658063, 1.570796, 1.483530, 1.396263, 1.308997, 1.218612, 
		 1.134464, 1.047198, 0.872665, 0.698132, 0.523599, 0.349066, 0.174533, -1.8762289, 
		 -1.8325957, 1.8762289, 1.8325957]

		Y = T[ 78.596218, 65.77963, 43.96947, 27.038816, 14.6126, 6.2614, 1.538330, 0.000000, 
		1.188045, 4.6841, 16.9321, 33.6988, 52.3664, 70.1630, 83.4221, 88.3995, 
		78.596218, 65.77963, 43.96947, 27.038816, 14.6126, 6.2614, 1.538330, 0.000000, 
		1.188045,  4.6841, 16.9321, 33.6988, 52.3664, 70.1630, 83.4221, 108.18086, 
		92.733676, 108.18086, 92.733676]

		x0 = ones(8)
		cache = similar(Y)
		Xj = T[X[i]^(2j - 2) for i in eachindex(X), j in eachindex(x0)]
		return (x0 = x0,
				obj = x -> palmer(x, Y, Xj, cache),
				grad! = (gradient, x) -> palmer_grad!(gradient, x, Y, Xj, cache))
end

"""
NAME          PALMER1D

*   Problem :
*   *********

*   A linear least squares problem
*   arising from chemical kinetics.

*   model: H-N=N=N TZVP+MP2
*   fitting Y to A0 + A2 X**2 + A4 X**4 + A6 X**6 + A8 X**8 +
*                A10 X**10 + A12 X**12

*   Source:
*   M. Palmer, Edinburgh, private communication.

*   SIF input: Nick Gould, 1990.

*   classification QUR2-RN-7-0
...

Keyword argument:
; T = eltype(1.)
"""
function gen_palmer1d(; T = eltype(1.))

X = T[-1.788963, -1.745329, -1.658063, -1.570796, -1.483530, -1.396263, -1.308997, -1.218612,
	 -1.134464, -1.047198, -0.872665, -0.698132, -0.523599, -0.349066, -0.174533, 0.0000000,
	 1.788963, 1.745329, 1.658063, 1.570796, 1.483530, 1.396263, 1.308997, 1.218612, 
	 1.134464, 1.047198, 0.872665, 0.698132, 0.523599, 0.349066, 0.174533,-1.8762289,
	-1.8325957, 1.8762289, 1.8325957]

  Y = T[78.596218, 65.77963, 43.96947, 27.038816, 14.6126, 6.2614, 1.538330, 0.000000,
	   1.188045, 4.6841, 16.9321, 33.6988, 52.3664, 70.1630, 83.4221, 88.3995, 78.596218,
	65.77963, 43.96947, 27.038816, 14.6126, 6.2614, 1.538330, 0.000000, 1.188045,
	4.6841, 16.9321, 33.6988, 52.3664, 70.1630, 83.4221, 108.18086, 92.733676,
	108.18086, 92.733676]

	x0 = ones(7)
	cache = similar(Y)
	Xj = T[X[i]^(2j - 2) for i in eachindex(X), j in eachindex(x0)]
	return (x0 = x0,
			obj = x -> palmer(x, Y, Xj, cache),
			grad! = (gradient, x) -> palmer_grad!(gradient, x, Y, Xj, cache))
end

"""
NAME          PALMER2C

*   Problem :
*   *********

*   A linear least squares problem
*   arising from chemical kinetics.

*   model: H-N=C=O TZVP + MP2
*   fitting Y to A0 + A2 X**2 + A4 X**4 + A6 X**6 + A8 X**8 +
*                A10 X**10 + A12 X**12 + A14 X**14

*   Source:
*   M. Palmer, Edinburgh, private communication.

*   SIF input: Nick Gould, 1990.

*   classification QUR2-RN-8-0
...

Keyword argument:
; T = eltype(1.)
"""
function gen_palmer2c(; T = eltype(1.))

	X = T[ -1.745329,-1.570796,-1.396263, -1.221730, -1.047198, -0.937187, -0.872665, -0.698132,
	-0.523599, -0.349066, -0.174533, 0.0, 0.174533, 0.349066, 0.523599, 0.698132, 0.872665,
	0.937187, 1.047198, 1.221730, 1.396263, 1.570796, 1.745329]

	Y = T[72.676767, 40.149455, 18.8548, 6.4762, 0.8596, 0.00000, 0.2730, 3.2043, 8.1080, 13.4291,
	17.714, 19.4529, 17.7149, 13.4291, 8.1080, 3.2053, 0.2730, 0.00000, 0.8596, 6.4762,
	18.8548, 40.149455, 72.676767]

	x0 = ones(8)
	cache = similar(Y)
	Xj = T[X[i]^(2j - 2) for i in eachindex(X), j in eachindex(x0)]
	return (x0 = x0,
			obj = x -> palmer(x, Y, Xj, cache),
			grad! = (gradient, x) -> palmer_grad!(gradient, x, Y, Xj, cache))
end

"""
NAME          PALMER3C

*   Problem :
*   *********

*   A linear least squares problem
*   arising from chemical kinetics.

*   model: H-N=C=S TZVP + MP2
*   fitting Y to A0 + A2 X**2 + A4 X**4 + A6 X**6 + A8 X**8 +
*                A10 X**10 + A12 X**12 + A14 X**14

*   Source:
*   M. Palmer, Edinburgh, private comminication.

*   SIF input: Nick Gould, 1990.

*   classification QUR2-RN-8-0
...

Keyword argument:
; T = eltype(1.)
"""
function gen_palmer3c(; T = eltype(1.))

	X = T[-1.658063, -1.570796, -1.396263, -1.221730, -1.047198, -0.872665, -0.766531, -0.698132,
	-0.523599, -0.349066, -0.174533, 0.0, 0.174533, 0.349066, 0.523599, 0.698132, 0.766531,
	0.872665, 1.047198, 1.221730, 1.396263, 1.570796, 1.658063]

	Y = T[64.87939, 50.46046, 28.2034, 13.4575, 4.6547, 0.59447, 0.0000, 0.2177, 2.3029, 5.5191,
	8.5519, 9.8919, 8.5519, 5.5191, 2.3029, 0.2177, 0.0000, 0.59447, 4.6547, 13.4575, 28.2034,
	50.46046, 64.87939]

	x0 = ones(8)
	cache = similar(Y)
	Xj = T[X[i]^(2j - 2) for i in eachindex(X), j in eachindex(x0)]
	return (x0 = x0,
			obj = x -> palmer(x, Y, Xj, cache),
			grad! = (gradient, x) -> palmer_grad!(gradient, x, Y, Xj, cache))
end

"""
NAME          PALMER4C

*   Problem :
*   *********

*   A linear least squares problem
*   arising from chemical kinetics.

*    model: H-N=C=Se TZVP + MP2
*   fitting Y to A0 + A2 X**2 + A4 X**4 + A6 X**6 + A8 X**8 +
*                A10 X**10 + A12 X**12 + A14 X**14

*   Source:
*   M. Palmer, Edinburgh, private communication.

*   SIF input: Nick Gould, 1990.

*   classification QUR2-RN-8-0
...

Keyword argument:
; T = eltype(1.)
"""
function gen_palmer4c(; T = eltype(1.))
	X = T[-1.658063, -1.570796, -1.396263, -1.221730, -1.047198, -0.872665, -0.741119, -0.698132, 
	-0.523599, -0.349066, -0.174533, 0.0, 0.174533, 0.349066, 0.523599, 0.698132, 0.741119, 0.872665, 
	1.047198, 1.221730, 1.396263, 1.570796, 1.658063]

	Y = T[67.27625, 52.8537, 30.2718, 14.9888, 5.5675, 0.92603, 0.0, 0.085108, 1.867422, 5.014768, 
	8.263520, 9.8046208, 8.263520, 5.014768, 1.867422, 0.085108, 0.0, 0.92603, 5.5675, 14.9888, 
	30.2718, 52.8537, 67.27625]

	x0 = ones(8)
	cache = similar(Y)
	Xj = T[X[i]^(2j - 2) for i in eachindex(X), j in eachindex(x0)]
	return (x0 = x0,
			obj = x -> palmer(x, Y, Xj, cache),
			grad! = (gradient, x) -> palmer_grad!(gradient, x, Y, Xj, cache))
end

# PALMER5C

function gen_palmer5c(; T = eltype(1.))

	X = T[0.000000, 1.570796, 1.396263, 1.308997, 1.221730, 1.125835, 1.047198, 0.872665, 0.698132,
		0.523599, 0.349066, 0.174533]

	Y = T[83.57418, 81.007654, 18.983286, 8.051067, 2.044762, 0.000000, 1.170451, 10.479881,
	25.785001, 44.126844, 62.822177, 77.719674]

	x0 = ones(6)
	cache = similar(Y)

	b = X[2]
	a = -b
	d = 2 * b
  
	t = Array{T}(undef, 12, 15)
	for k = 1:12
	  t[k, 1] = 1
	  t[k, 2] = (2 * X[k] - a - b) / d
	  for l = 3:15
		t[k, l] = 2 * t[k, l - 1] * (2 * X[k] - a - b) / d - t[k, l - 2]
	  end
	end

	Xj = T[t[i, 2 * j - 1] for i in eachindex(X), j in eachindex(x0)]

	return (x0 = x0,
			obj = x -> palmer(x, Y, Xj, cache),
			grad! = (gradient, x) -> palmer_grad!(gradient, x, Y, Xj, cache))
end

# PALMER5D

function gen_palmer5d(; T = eltype(1.))

	X = T[0.000000, 1.570796, 1.396263, 1.308997, 1.221730, 1.125835, 1.047198, 0.872665, 0.698132, 
	0.523599, 0.349066, 0.174533]

	Y = T[83.57418, 81.007654, 18.983286, 8.051067, 2.044762, 0.000000, 1.170451, 10.479881, 25.785001, 
	44.126844, 62.822177, 77.719674]

	x0 = ones(4)
	cache = similar(Y)
	Xj = T[X[i]^(2j - 2) for i in eachindex(X), j in eachindex(x0)]
	return (x0 = x0,
			obj = x -> palmer(x, Y, Xj, cache),
			grad! = (gradient, x) -> palmer_grad!(gradient, x, Y, Xj, cache))
end

# PALMER6C

function gen_palmer6c(; T = eltype(1.))

	X = T[0.000000, 1.570796, 1.396263, 1.221730, 1.047198, 0.872665, 0.785398, 0.732789, 
	0.698132, 0.610865, 0.523599, 0.349066, 0.174533]

	Y = T[10.678659, 75.414511, 41.513459, 20.104735, 7.432436, 1.298082, 0.171300, 
	0.000000, 0.068203, 0.774499, 2.070002, 5.574556, 9.026378]

	x0 = ones(8)
	cache = similar(Y)
	Xj = T[X[i]^(2j - 2) for i in eachindex(X), j in eachindex(x0)]
	return (x0 = x0,
			obj = x -> palmer(x, Y, Xj, cache),
			grad! = (gradient, x) -> palmer_grad!(gradient, x, Y, Xj, cache))
end

# PALMER7C

function gen_palmer7c(; T = eltype(1.))
	X = T[0.000000, 0.139626, 0.261799, 0.436332, 0.565245, 0.512942, 0.610865, 0.785398, 0.959931, 
	1.134464, 1.308997, 1.483530, 1.658063]

	Y = T[4.419446, 3.564931, 2.139067, 0.404686, 0.000000, 0.035152, 0.146813, 2.718058, 9.474417, 
	26.132221, 41.451561, 72.283164, 117.630959]

	x0 = ones(8)
	cache = similar(Y)
	Xj = T[X[i]^(2j - 2) for i in eachindex(X), j in eachindex(x0)]
	return (x0 = x0,
			obj = x -> palmer(x, Y, Xj, cache),
			grad! = (gradient, x) -> palmer_grad!(gradient, x, Y, Xj, cache))
end

# PALMER8C

function gen_palmer8c(; T = eltype(1.))

	X = T[0.000000, 0.174533, 0.314159, 0.436332, 0.514504, 0.610865, 0.785398, 0.959931, 1.134464, 
	1.308997, 1.483530, 1.570796]

	Y = T[4.757534, 3.121416, 1.207606, 0.131916, 0.000000, 0.258514, 3.380161, 10.762813, 23.745996, 
	44.471864, 76.541947, 97.874528]

	x0 = ones(8)
	cache = similar(Y)
	Xj = T[X[i]^(2j - 2) for i in eachindex(X), j in eachindex(x0)]
	return (x0 = x0,
			obj = x -> palmer(x, Y, Xj, cache),
			grad! = (gradient, x) -> palmer_grad!(gradient, x, Y, Xj, cache))
end