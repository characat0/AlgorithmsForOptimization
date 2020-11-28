### A Pluto.jl notebook ###
# v0.12.11

using Markdown
using InteractiveUtils

# ╔═╡ 523aac80-19a4-11eb-3422-5fb838c9b32b
using LinearAlgebra

# ╔═╡ bab9fa0e-1ae7-11eb-273f-a151c16a7b3d
using Plots

# ╔═╡ a68add40-1aea-11eb-36a3-d196d555bccf
using Base.MathConstants

# ╔═╡ ed126440-1af4-11eb-17e5-6937883bebf1
using Calculus

# ╔═╡ fa52f540-199e-11eb-35c8-67339fabc732
md"# Second-Order Methods"

# ╔═╡ 15718df0-199f-11eb-032d-fd0d6b689a2f
md"Last chapter used the gradient, this will use the Hessian."

# ╔═╡ 702ca2c0-199f-11eb-35dd-933d4aac7827
md"## Newton's Method"

# ╔═╡ 75861cb0-199f-11eb-2b07-49eb0e71a9ab
md"For univariate optimization, we can approximate the function about a point using second-order Taylor expansion:"

# ╔═╡ bc25e040-19a1-11eb-3a1b-07b2633f6f5f
md"\$\$q(x)=f(x^{(k)})+(x-x^{(k)})f'(x^{(k)}) + \frac{(x-x^{(k)})^2}{2}f''(x^{(k)}) \$\$"

# ╔═╡ efa26b00-19a1-11eb-2a63-fb3647af723b
md"Setting its derivative to zero yields the Newton's method."

# ╔═╡ 905ca882-19a2-11eb-3b89-3f31562fd6e6
md"\$\$x^{(k+1)}=x^{(k)}-\frac{f'(x^{(k)})}{f''(x^{(k)})}\$\$"

# ╔═╡ 86014a80-19a2-11eb-0532-c9882ffed160
md"But 👀... there is a catch, if the second derivative is close to zero this method is unstable 😒."

# ╔═╡ c31b3a72-19a2-11eb-2fcf-a5526454bbda
md"Newton's method will have quadratic rate of convergence if its second derivative is not zero in the interval & its third derivative is continous on the interval & \$\$\frac{1}{2}|\frac{f'''(x^{(1)})}{f''(x^{(1)})}| < c|\frac{f'''(x^{(\*)})}{f''(x^{(\*)})}| \$\$ for some c < ∞"

# ╔═╡ 4621b570-19a3-11eb-15dd-1d9511adaff6
md"Let's see the multivariate version: \$\$x^{(k+1)}=x^{(k)}-(H^{(k)})^{-1}g^{(k)}\$\$"

# ╔═╡ 81d25162-19a3-11eb-31d8-55a435afa319
md"We can use Newton's method to supply a descent direction to line search."

# ╔═╡ 3b6a5f52-19a4-11eb-2d79-79c8b841db0d
md"Δ = \Delta"

# ╔═╡ 12f30720-19a4-11eb-2cb5-e9c5d296adfe
function newtons_method(∇f, H, x, ϵ, k_max)
	k, Δ = 1, fill(Inf, length(x))
	while norm(Δ) > ϵ && k ≤ k_max
		Δ = H(x) \ ∇f(x)
		x -= Δ
		k += 1
	end
	return x
end

# ╔═╡ 7da88860-19a4-11eb-3f5a-e9aa91151c89
md"## Secant Method"

# ╔═╡ 3aba1030-19a6-11eb-3f56-591db74acc7a
md"If we don't have access to a second derivative, we can approximate it..."

# ╔═╡ 49935532-19a6-11eb-05d6-37a6387593b8
md"\$\$f''(x^{(k)})=\frac{f'(x^{(k)})-f'(x^{(k-1)})}{x^{(k)}-^{(k-1)}}\$\$"

# ╔═╡ 7ddfb310-19a6-11eb-07dc-d3574c59b99d
md"\$\$x^{(k+1)}←x^{(k)}-\frac{x^{(k)}-^{(k-1)}}{f'(x^{(k)})-f'(x^{(k-1)})}f'(x^{(k)})\$\$"

# ╔═╡ e0673da0-19a6-11eb-0b69-7d8a0d8ee005
md"## Quasi-Newton Methods"

# ╔═╡ e8b8e8f0-19a6-11eb-2c15-078db02e2b48
md"For the multivariate case..."

# ╔═╡ f0ada732-19a6-11eb-3e23-357f481ea7b1
md"\$\$x^{(k+1)}=x^{(k)}-α^{(k)}Q^{(k)}g^{(k)}\$\$"

# ╔═╡ 01c8ecf0-19a7-11eb-3669-bfbcc7ce6439
md"Where Q is an approximation of the inverse of the Hessian and α is an step factor."

# ╔═╡ 138f3fc0-19a7-11eb-2fd3-2dca56c75a86
md"Usually Q¹ is the Identity matrix and changes according to the _Davidon-Fletcher-Powell_ (DFP) method..."

# ╔═╡ f8c55410-1ad5-11eb-3b8f-5fc50ba36694
md"\$\$Q← Q- \frac{Qγγ^{T}Q}{γ^{T}Qγ}+\frac{δδ^{T}}{δ^{T}γ}\$\$"

# ╔═╡ b30058c0-1ad6-11eb-3b00-57e0b3b04529
md"Note: All terms on the right hand are evaluated at iteration k."

# ╔═╡ 3996cdc0-1ad6-11eb-27de-593b31efbfe2
md"Where...\$\$γ^{(k+1)}≡g^{(k+1)}-g^{(k)}\$\$"

# ╔═╡ 0b8c9ece-19a7-11eb-1753-73b6f0e5aba1
md"\$\$δ^{(k+1)}≡x^{(k+1)}-x^{(k)}\$\$"

# ╔═╡ 6c0b1040-1ad6-11eb-1265-57bea959874d
md"There is an alternative method _Broyden-Fletcher-Goldfarb-Shanno_ (BFGS)"

# ╔═╡ 0aed3260-1ad7-11eb-2010-2f0b932ea0b0
md"\$\$Q←Q-(\frac{δγ^{T}Q+Qγδ^{T}}{δ⋅γ})+(1+\frac{γ^{T}Qγ}{δ⋅γ})\frac{δ^{2}}{δ⋅γ}\$\$"

# ╔═╡ 97506010-1ad7-11eb-09d8-ff72bf563f43
abstract type DescentMethod
end

# ╔═╡ b8371860-1aea-11eb-1a9f-85ca2dcfdeb7
ϕ = golden

# ╔═╡ eb2e4120-1ad7-11eb-3c4b-b528468c0fec
function golden_section_search(f::Function, a::Number, b::Number, n::Integer)::Number
	p = ϕ - 1
	d = p * b + (1 - p)*a
	yd = f(d)
	for i in 1:n-1
		c = p*a + (1-p)*b
		yc = f(c)
		if yc < yd
			b, d, yd = d, c, yc
		else
			a, b = b, c
		end
	end
	return (a+b)/2
end

# ╔═╡ 3888d6d0-1aef-11eb-3cea-a5837c7f05c4
function bracket_minimum(f::Function, x::Number; s=1e-2, k=1.5)::Tuple{Number, Number}
	a, y₁ = x, f(x)
	b, y₂ = x+s, f(x+s)
	if y₂ > y₁
		a, y₁, b, y₂ = b, y₂, a, y₁
		s = -s
	end
	n = 0
	while n < 100
		c, y₃ = b+s, f(x+s)
		if y₃ > y₂
			return a < c ? (a, c) : (c, a)
		end
		a, y₁, b, y₂ = b, y₂, c, y₃
		s *= k
		n += 1
	end
	return (a, b)
end

# ╔═╡ 10f1c520-1af2-11eb-2c30-db51b670cf0b
log(10)

# ╔═╡ 3cc95f50-1af2-11eb-1e5e-9160e542abd1


# ╔═╡ 8cc91140-1af1-11eb-315c-fd9dd8a831f6
function minimize(f::Function, x=0.0; ϵ=.001)::Number
	a, b = bracket_minimum(f, x)
	n = 10 #floor(Int, (b - a)/(ϵ*log(ϕ)))
	golden_section_search(f, a, b, n)
end

# ╔═╡ 222ea400-1aef-11eb-14ce-593303918295
function line_search(f, x̄, d)::Array{Number}
	obj(α) = f(x̄ + α*d)
	α = minimize(obj)
	return x̄+α*d
end

# ╔═╡ 3dd8b700-1af3-11eb-10cd-65dee8b7f7a8
[1, 2, 3]⋅[1, 2, 3]

# ╔═╡ 729106d0-1ad7-11eb-0181-151fa988894f
begin
	mutable struct DFP <: DescentMethod
		Q::Matrix{Float64}
	end
	DFP() = DFP(Matrix(undef, 0, 0))
	function init!(M::DFP, f, ∇f, x)
		m = length(x)
		M.Q = Matrix(1.0I, m, m)
		return M
	end
	function step!(M::DFP, f, ∇f, x)::Array{Float64, 1}
		Q, g = M.Q, ∇f(x)
		x′::Array{Float64} = line_search(f, x, -Q*g)
		g′ = ∇f(x′)
		δ = x′ - x
		γ = g′ - g
		Q[:] = Q - Q*γ*γ'*Q*(γ'*Q*γ) + δ*δ'/(δ⋅γ)
		return x′
	end
end

# ╔═╡ ba1bd25e-1af4-11eb-068b-43f3abca1b6e
f(x̄) = sum(x̄)*prod(x̄)

# ╔═╡ e4bd5d8e-1af4-11eb-3842-4167722d88a5
∇f(x̄) = Calculus.gradient(f, x̄)

# ╔═╡ f464ef10-1af4-11eb-193d-3745090c591f
M = DFP()

# ╔═╡ c47f9f62-1af5-11eb-0a05-43eceebb502c
x = [4, 5]

# ╔═╡ b7355ed0-1af5-11eb-3e68-319637063960
init!(M, f, ∇f, x)

# ╔═╡ c93f2340-1af5-11eb-2e2f-55b9643a94db
step!(M, f, ∇f, x)

# ╔═╡ 7d0b9b00-2f6f-11eb-2633-674b9bdc5853
sum(2, 2)

# ╔═╡ Cell order:
# ╟─fa52f540-199e-11eb-35c8-67339fabc732
# ╟─15718df0-199f-11eb-032d-fd0d6b689a2f
# ╟─702ca2c0-199f-11eb-35dd-933d4aac7827
# ╟─75861cb0-199f-11eb-2b07-49eb0e71a9ab
# ╟─bc25e040-19a1-11eb-3a1b-07b2633f6f5f
# ╟─efa26b00-19a1-11eb-2a63-fb3647af723b
# ╟─905ca882-19a2-11eb-3b89-3f31562fd6e6
# ╟─86014a80-19a2-11eb-0532-c9882ffed160
# ╟─c31b3a72-19a2-11eb-2fcf-a5526454bbda
# ╟─4621b570-19a3-11eb-15dd-1d9511adaff6
# ╟─81d25162-19a3-11eb-31d8-55a435afa319
# ╟─3b6a5f52-19a4-11eb-2d79-79c8b841db0d
# ╠═523aac80-19a4-11eb-3422-5fb838c9b32b
# ╠═12f30720-19a4-11eb-2cb5-e9c5d296adfe
# ╟─7da88860-19a4-11eb-3f5a-e9aa91151c89
# ╟─3aba1030-19a6-11eb-3f56-591db74acc7a
# ╟─49935532-19a6-11eb-05d6-37a6387593b8
# ╟─7ddfb310-19a6-11eb-07dc-d3574c59b99d
# ╟─e0673da0-19a6-11eb-0b69-7d8a0d8ee005
# ╟─e8b8e8f0-19a6-11eb-2c15-078db02e2b48
# ╟─f0ada732-19a6-11eb-3e23-357f481ea7b1
# ╟─01c8ecf0-19a7-11eb-3669-bfbcc7ce6439
# ╟─138f3fc0-19a7-11eb-2fd3-2dca56c75a86
# ╟─f8c55410-1ad5-11eb-3b8f-5fc50ba36694
# ╟─b30058c0-1ad6-11eb-3b00-57e0b3b04529
# ╟─3996cdc0-1ad6-11eb-27de-593b31efbfe2
# ╟─0b8c9ece-19a7-11eb-1753-73b6f0e5aba1
# ╟─6c0b1040-1ad6-11eb-1265-57bea959874d
# ╟─0aed3260-1ad7-11eb-2010-2f0b932ea0b0
# ╠═97506010-1ad7-11eb-09d8-ff72bf563f43
# ╠═bab9fa0e-1ae7-11eb-273f-a151c16a7b3d
# ╠═a68add40-1aea-11eb-36a3-d196d555bccf
# ╠═b8371860-1aea-11eb-1a9f-85ca2dcfdeb7
# ╠═eb2e4120-1ad7-11eb-3c4b-b528468c0fec
# ╠═3888d6d0-1aef-11eb-3cea-a5837c7f05c4
# ╠═10f1c520-1af2-11eb-2c30-db51b670cf0b
# ╟─3cc95f50-1af2-11eb-1e5e-9160e542abd1
# ╠═8cc91140-1af1-11eb-315c-fd9dd8a831f6
# ╠═222ea400-1aef-11eb-14ce-593303918295
# ╠═3dd8b700-1af3-11eb-10cd-65dee8b7f7a8
# ╠═729106d0-1ad7-11eb-0181-151fa988894f
# ╠═ba1bd25e-1af4-11eb-068b-43f3abca1b6e
# ╠═ed126440-1af4-11eb-17e5-6937883bebf1
# ╠═e4bd5d8e-1af4-11eb-3842-4167722d88a5
# ╠═f464ef10-1af4-11eb-193d-3745090c591f
# ╠═c47f9f62-1af5-11eb-0a05-43eceebb502c
# ╠═b7355ed0-1af5-11eb-3e68-319637063960
# ╠═c93f2340-1af5-11eb-2e2f-55b9643a94db
# ╠═7d0b9b00-2f6f-11eb-2633-674b9bdc5853
