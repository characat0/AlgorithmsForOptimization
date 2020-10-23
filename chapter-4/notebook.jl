### A Pluto.jl notebook ###
# v0.12.4

using Markdown
using InteractiveUtils

# ╔═╡ 035b4c80-12f3-11eb-3ce9-e5153281a50e
using Base.MathConstants

# ╔═╡ 754bd8f0-12f3-11eb-3530-b976559c01e3
using Plots

# ╔═╡ b2527c70-134a-11eb-3791-0bd358c258dc
using LinearAlgebra

# ╔═╡ 2dc895be-134a-11eb-0170-d38101fb1a7f
using Calculus

# ╔═╡ 8481d6c0-135d-11eb-37c3-9b4005ccc90e
using Convex, SCS

# ╔═╡ 2e88ed30-13c6-11eb-0bda-c1588b0e766d
using PlutoUI

# ╔═╡ 08a25410-12ec-11eb-36f3-a13430f80c2e
md"# Local Descent"

# ╔═╡ 3158b110-12ec-11eb-02c1-21442091ead6
md"We will focus on multivariate functions, we begin with method that choose a direction to descend with a defined step size."

# ╔═╡ 90e12c72-12ec-11eb-33df-d5c95bcb08cd
md"## Descend Direction Iteration"

# ╔═╡ 9a4de4ae-12ec-11eb-13b5-2dc735b13329
md"The most common approach is to improve a point x by taking a step that minimizes the objective function based on a local model, obtained from a Taylor approximation"

# ╔═╡ cc198b1e-12ec-11eb-3e1e-450e39b17fcf
md"This algorithm involves the following steps:"

# ╔═╡ ec332a10-12ec-11eb-1488-3f1090e39e00
md"- Check if xᵏ satisfies the termination condition."

# ╔═╡ 0655d690-12ed-11eb-3fb2-c99bc7c75be5
md"- Determine a descent direction dᵏ using local information such as gradient of Hessian. Optionally determine its lenght or asume ||dᵏ=1|| ."

# ╔═╡ 2c8f73c0-12ed-11eb-02c2-01506d115cf7
md"- Determine the step size αᵏ. Some algorithms try to optimize this step size."

# ╔═╡ 712150ce-12ed-11eb-1d47-4ba87eaa3b30
md"Compute the next design step according to: \$\$ x^{(k+1)}←x^{(k)}+α^{(k)}d^{(k)}\$\$"

# ╔═╡ bb76db50-12ed-11eb-15e1-b7493783d9e7
md"## Line search"

# ╔═╡ c7b2e030-12ed-11eb-211f-ebbae3a4411f
md"Assuming a descent direction d, we can obtain the step factor α using line search, which select the step factor that minimizes the one-dimensional function:\$\$ \min_{α} f(x+αd)\$\$"

# ╔═╡ 0a002f00-12ef-11eb-1e71-87804c2d8d0e
function bracket_minimum(f, x=0.0; s=1e-12, k=2.0)
	a, ya = x, f(x)
	b, yb = x+s, f(x+s)
	if yb > ya #If we are not going downhill
		c, yc = x-s, f(x-s)
		s = -s
	else
		c, yc = x+k*s, f(x+k*s)	
	end
	if ya > yb < yc # Definition of unimodal
		return (a, c)
	end
	return bracket_minimum(f, b; s=k*s, k=k)
end

# ╔═╡ 14897ace-12ef-11eb-2d95-d9fcc1f7fdf0
function fibonacci_search(f, a, b, n; ϵ=0.01)
	s = (1-√5)/(1+√5)
	p = 1/(φ*(1-s^(n+1))/(1-s^n))
	d = p*b + (1-p)*a
	yd = f(d)
	for i in 1:n-1
		if 1 == n-1
			c = ϵ*a + (1-ϵ)*d
		else
			c = p*a + (1-p)*b
		end
		yc = f(c)
		if yc < yd
			b, d, yd = d, c, yc
		else
			a, b = b, c
		end
		p = 1/(φ*(1-s^(n-i+1))/(1-s^(n-i)))
	end
	(a+b)/2
end

# ╔═╡ 22c726a0-12f0-11eb-0ba6-5da513dbc281
minimize(f, a, b) = fibonacci_search(f, a, b, 20)

# ╔═╡ 0269168e-12ee-11eb-3dd1-77d75ba692a9
function line_search(f, x, d)
	objective(α) = f((x + α*d)...) #where x and d are vectors of the same length
	a, b = bracket_minimum(objective) #Determine a minimum bracketing interval
	α = minimize(objective, a, b) #Using any previous algorithm
	x + α*d
end

# ╔═╡ e0c7ff80-13f9-11eb-2b71-f1e33226bf6a
begin
	f₁(x, y) = (1-x)^2 + 5(y - x^2)^2
	f₁(x̄) = f₁(x̄...)
end

# ╔═╡ 8f9f3260-12f3-11eb-18a3-571a47653d95
begin
	x_initial = [0, .1]
	x_dir = [.8, .8]
	x_opt = line_search(f₁, x_initial, x_dir)
	x = range(-1, stop=2.5, length=200)
	y = x
	color = cgrad([:red, :green, :blue])
	contour(x, y, f₁, c=color, levels=250 .^ (range(-1,stop=1,length=20)))
	scatter!(tuple(x_initial...), c=:black, label="Initial point")
	scatter!(tuple(x_opt...), c=:black, label="Optimal")
end

# ╔═╡ 496a7360-12f6-11eb-0a39-0db347ba7df9
"The best point is $(x_opt)"

# ╔═╡ 0835eac0-12f9-11eb-097e-9903d7e61752
md"α is refered as the learning rate, small values are stable but converge slowly, large values have the risk to overshoot the minimum point"

# ╔═╡ d06552b0-12f9-11eb-1f34-df23cefc2d1e
md"To try to solve this problem we can use decaying step factors (multiply the learning rate by a constant decaying factor in every iteration)."

# ╔═╡ ede4e5d0-12f9-11eb-253b-85703a7d49ed
begin
	f₂(x, y, z) = sin(x*y)+exp(y+z)-z
	f₂(x̄) = f₂(x̄...)
	initial = [1, 2, 3]
	direction = [0, -1, -1]
	obj(α) = f₂((initial + α*direction)...)
	a₂, b₂ = bracket_minimum(obj)
	α₂ = minimize(obj, a₂, b₂)
	plot(a₂:.25:b₂, obj, label="Objective function")
	scatter!((α₂, obj(α₂)), label="Optimal α")
end

# ╔═╡ 29c5c410-12fb-11eb-29c2-a1c94c7ff235
"The minimum is as α≈$α₂ with x≈$(initial+α₂*direction)"

# ╔═╡ 614fe7d0-12fb-11eb-1736-8bc134642481
md"## Approximate Line Search"

# ╔═╡ 6abaa440-12fb-11eb-0731-fb49bf5b4b91
md"The condition of *sufficient decrease* requires that the step size cause a sufficient decrease in the objective function value."

# ╔═╡ 220c8130-1348-11eb-2cc4-27aef429ae45
md"\$\$f(x^{(k+1)})≤f(x^{(k)})+βα∇_{d^{(k)}}f(x^{(k)})\$\$"

# ╔═╡ df62e7f0-134e-11eb-1d5f-193c6dddd561
md"**This is called the first Wolfe condition**"

# ╔═╡ 5a07b640-1348-11eb-2f41-155028e71b6b
md"This condition is ofter refered as the Armijo condition."

# ╔═╡ 84a4fbb0-1348-11eb-03d3-ed65b20129af
md"We can create a function that computes a step size α that satisfies this condition."

# ╔═╡ 9feb9950-1349-11eb-0d0c-d7ed01605603
function backtracking_line_search(f, ∇f, x, d, α; p=0.5, β=1e-4)
	y, g = f(x), ∇f(x)
	while f(x + α*d) > y + β*α*(g⋅d)
		α *= p
	end
	α
end

# ╔═╡ 358ca710-134a-11eb-393d-5dc3aa30d2f1
Calculus.gradient

# ╔═╡ e387f2d0-1349-11eb-3c51-11a55c4243ea
f₃ = f₂

# ╔═╡ ef3cc650-1349-11eb-2f20-1769c55e544e
∇f₃(nums...) = Calculus.gradient(f₃)(collect(nums))

# ╔═╡ 4ea70b52-134a-11eb-24bb-8ddd6484fd93
∇f₃(x̄) = ∇f₃(x̄...)

# ╔═╡ 6ac67230-134a-11eb-2990-f31f345f2a46
begin
	backtracking_line_search(f₃, ∇f₃, initial, direction, 1000)
end

# ╔═╡ 4322ef50-134b-11eb-2602-350852cc9cb0
md"There is another condition, called curvature condition, that requires the directional derivative at the next iterate to be shallower."

# ╔═╡ 5e3c30d0-134b-11eb-203d-7748430c642d
md"\$\$ ∇\_{d^{(k)}}f(x^{(k+1)})≥σ ∇\_{d^{(k)}}f(x^{(k)})\$\$"

# ╔═╡ daebb7b0-134e-11eb-1b3e-412480fe8c3e
md"**This is called the second Wolfe condition**"

# ╔═╡ 20bf4e70-134d-11eb-247f-9bff99598c13
md"\$\$|∇\_{d^{(k)}}f(x^{(k+1)})|≤-σ∇\_{d^{(k)}}f(x^{(k)})\$\$"

# ╔═╡ 1f6e8b80-134d-11eb-2400-8d6be329adcc
md"**This is called the strong curvature condition**"

# ╔═╡ 3a6e8dbe-134f-11eb-0f93-351e2ab4ddcf
md"The sufficient decrease condition (first) and the curvature condition (second) form the Wolfe conditions."

# ╔═╡ 50651cc0-134f-11eb-2b46-b1038d782cf2
md"The sufficient decrease condition with the strong curvature condition form the strong Wolfe conditions."

# ╔═╡ 649bfa60-134f-11eb-3fc6-1ba1d282880a
md"An interval guaranteed to contain step lengths satisfying the Wolfe conditions is found when one of the following holds:"

# ╔═╡ a560fb80-1350-11eb-3f52-b931d3c5a76b
md"\$\$f(x+α^{(k)}d)≥f(x)\$\$"

# ╔═╡ bc718600-1350-11eb-2e14-818244783841
md"\$\$f(x^{(k)}+α^{(k)}d^{(k)})>f(x^{(k)})+βα^{(k)}∇\_{d^{(k)}}f(x^{(k)})\$\$"

# ╔═╡ f5bda510-1350-11eb-05b3-07cf839263b3
md"\$\$∇f(x+α^{(k)}d)≥0\$\$"

# ╔═╡ 0cb3c9c0-1351-11eb-16e6-f340951ccb8a
function strong_bracketing(f, ∇f, x, d; α=1, β=1e-4, σ=.1)
	yθ, gθ, y_prev, α_prev = f(x), ∇f(x)⋅d, NaN, 0
	αlo, αhi = NaN, NaN
	#bracketing phase
	while true
		y = f(x+α*d)
		if y > yθ + β*a*gθ | (!isnan(y_prev) & y ≥ y_prev) #sufficient decrease
			αlo, αhi = α_prev, α
			brek
		end
		g = ∇f(x+α*d)⋅d
		if abs(gθ) ≤ -σ*gθ #strong curvature condition
			return α
		elseif g ≥ 0
			αlo, αhi = α, α_prev
			break
		end
		y_prev, α_prev, α = y, α, 2α
	end
	
	#zoom phase
	ylo = f(x+αlo*d)
	while true
		α = (αlo + αhi)/2
		y = f(x+α*d)
		if y > yθ + β*a*gθ | y ≥ ylo #sufficient decrease
			αhi = α
		else
			g = ∇f(x+α*d)⋅d
			if abs(g) ≤ -σ*gθ
				return α
			elseif g*(αhi - αlo) ≥ 0
				αhi = α
			end
			αlo = α
		end
	end
end

# ╔═╡ 6037c3a0-1354-11eb-2a53-e347f53030ff
md"## Trust Region Methods"

# ╔═╡ 855c9140-135b-11eb-0d0d-edd695624daf
md"These methods first choose a maximum step size and then the step direction. The radius is then expanded and contracted based on how well the model predicts function evaluations."

# ╔═╡ b9dfdda0-135b-11eb-293b-1357e85b8839
md"The logic is: start with a radius, calculate the minimum point in the contour, and move in that direction (if the decrease is greater than a threshold, otherwise expand the radius) reducing the radius by a constant factor."

# ╔═╡ c9fb7130-135c-11eb-24ec-7ff0c2f85c0f
md"η = \eta"

# ╔═╡ 2caacf10-135d-11eb-0e0c-7916ac6271b4
md"′ = \prime"

# ╔═╡ 89312e00-135d-11eb-36eb-0d6ae62b2f91
function solve_trust_region_subproblem(∇f, H, x₀, δ)
	x = Convex.Variable(length(x₀))
	p = Convex.minimize(∇f(x₀)⋅(x-x₀) + quadform(x-x₀, H(x₀))/2)
	p.constraints += norm(x-x₀) ≤ δ
	solve!(p, () -> SCS.Optimizer(verbose=0))
	(x.value, p.optval)
end

# ╔═╡ 9b5ca0ae-135c-11eb-01b0-c5e8493a801d
function trust_region_descent(f, ∇f, H, x, k_max;
	η₁=.25, η₂=.5, γ₁=.5, γ₂=2.0, δ=1.0)
	y = f(x)
	println("#it  Ratio   \t delta \t point")
	rounder(x) = round(x, digits=6)
	for k in 1:k_max
		x′, y′ = solve_trust_region_subproblem(∇f, H, x, δ)
		r = abs((y - f(x′)) / (y - y′))
		if r < η₁
			δ *= γ₁
		else
			x, y = x′, y′
			if r > η₂
				δ *= γ₂
			end
		end
		println(k, " -> ", rounder(r), "\t", δ, "  \t", rounder.(x))
	end
	x
end

# ╔═╡ f88cba72-13db-11eb-0b50-bfe67120b1c4
md"I will define symbolically all the functions, due to approximation errors it didn't work out before."

# ╔═╡ 003d8f20-135e-11eb-2ae6-db430cf84da5
begin
	#By variable
	f₄(x, y) = (1-x)^2 + 5(y - x^2)^2
	∇f₄(x, y) = [(2*(-1 + x + 10*x^3 - 10*x*y)), 10*(y-x^2)]
	H₄(x, y) = [[-20*(y - x^2) + 40*x^2 + 2 , -20*x] [-20*x , 10]]
	#Using vectors
	f₄(x̄) = f₄(x̄...)
	∇f₄(x̄) = ∇f₄(x̄...)	
	H₄(x̄) = H₄(x̄...)
end;

# ╔═╡ e66a6230-135d-11eb-1cdb-8d9506b3ffdf
begin
	x₄ = [-1, -1]
	with_terminal() do
		trust_region_descent(f₄, ∇f₄, H₄, x₄, 10)
	end
end

# ╔═╡ 17940990-13dd-11eb-349a-19702cc7fd55
function trust_descent_points(f, ∇f, H, x, k_max::Integer;
	η₁=.25, η₂=.5, γ₁=.5, γ₂=2.0, δ=1.0)::Array{Tuple}
	points = Array{Tuple, 1}(undef, k_max)
	y = f(x)
	
	for k in 1:k_max
		x′, y′ = solve_trust_region_subproblem(∇f, H, x, δ)
		r = abs((y - f(x′)) / (y - y′))
		points[k] = tuple(x..., r)
		if r < η₁
			δ *= γ₁
		else
			x, y = x′, y′
			if r > η₂
				δ *= γ₂
			end
		end
	end
	points
end

# ╔═╡ 54f35f90-13e0-11eb-1fa0-4f08e7af3c71
begin
	points = trust_descent_points(f₄, ∇f₄, H₄, x₄, 10)
	contour(x, y, f₁, c=color, levels=10 .^ (range(-5,stop=2.2,length=30)) )
	radiuses = [x[end] for x in points]
	points = [x[begin:end-1] for x in points]
	scatter!(points, c=:black, label="Approximations", markersize=3)
	plot!(points, c=:black, alpha=0.3, label="Descent")
end

# ╔═╡ 8476d170-13e5-11eb-1c42-bd1cae451004
md"## Termination Conditions"

# ╔═╡ 9eedea10-13e6-11eb-2047-2b7befc81017
md"- *Maximum iterations* (Too much or too long)"

# ╔═╡ c8ee2c2e-13e6-11eb-19e5-6f2d481f020b
md"- *Absolute improvement* (Continuation doesn't improve enough.)"

# ╔═╡ 0d7310f0-13e7-11eb-15f8-29dc03303c66
md"- *Gradient magnitude* (The rate of change is near 0)"

# ╔═╡ 309f6e1e-13e7-11eb-0cab-914cb1c7dc19
md"If there are multiple local minima we can start again from a random point."

# ╔═╡ 3d7764e0-13e7-11eb-36e7-0f7c7b5da594
md"## Excercises"

# ╔═╡ a39d72f0-13e7-11eb-03cf-db4610a63043
md"**Excercise 4.1.** Why is it important to have more than one termination condition?"

# ╔═╡ b4b73210-13e7-11eb-12cf-03a37e027d0f
md"**Answer**: Because there is no point in running the process more than it is needed, so we use different criteria for stopping (Takes too long, the improvement change doesn't worth more iterations, we are sufficiently close to the optimal)"

# ╔═╡ 1a6c3960-13e9-11eb-3b9d-3bec96635025
md"**Excercise 4.2.** The first Wolfe condition requires:"

# ╔═╡ 45e279b0-13e9-11eb-23d4-9355cdd7580f
md"\$\$f(x^{(k)}+αd^{(k)})≤f(x^{(k)})+βα∇_{d^{(k)}}f(x^{(k)})\$\$"

# ╔═╡ 470db340-13e9-11eb-02b1-254178507905
md"What is the maximum step length α that satisfies this condition, given that:"

# ╔═╡ 7dd3eac0-13e9-11eb-3d18-c5958ddf0882
md"\$\$f(x)=5+x\_1 ^2 + x\_2 ^2\\ ,x^{(k)}=[-1, -1], d=[1, 0], β=10^{-4}\$\$"

# ╔═╡ c4364210-13e9-11eb-1e5c-25dbca34084c
md"**Answer**:"

# ╔═╡ dfbc9d40-13e9-11eb-2cda-a75bb4153611
begin
	f₅(x, y) = 5 + x^2 + y^2
	f₅(x̄) = f₅(x̄...)
end

# ╔═╡ d05c7500-13e9-11eb-3a01-a5832a81d5a6
opt_α = backtracking_line_search(f₅, Calculus.gradient(f₅), [-1, -1], [1, 0], 10, p=0.99)

# ╔═╡ 558827b0-13ef-11eb-3ba6-c7e16975893d
"The maximum step length is α ≈ $(opt_α)"

# ╔═╡ Cell order:
# ╟─08a25410-12ec-11eb-36f3-a13430f80c2e
# ╟─3158b110-12ec-11eb-02c1-21442091ead6
# ╟─90e12c72-12ec-11eb-33df-d5c95bcb08cd
# ╟─9a4de4ae-12ec-11eb-13b5-2dc735b13329
# ╟─cc198b1e-12ec-11eb-3e1e-450e39b17fcf
# ╟─ec332a10-12ec-11eb-1488-3f1090e39e00
# ╟─0655d690-12ed-11eb-3fb2-c99bc7c75be5
# ╟─2c8f73c0-12ed-11eb-02c2-01506d115cf7
# ╟─712150ce-12ed-11eb-1d47-4ba87eaa3b30
# ╟─bb76db50-12ed-11eb-15e1-b7493783d9e7
# ╟─c7b2e030-12ed-11eb-211f-ebbae3a4411f
# ╠═0a002f00-12ef-11eb-1e71-87804c2d8d0e
# ╠═035b4c80-12f3-11eb-3ce9-e5153281a50e
# ╠═14897ace-12ef-11eb-2d95-d9fcc1f7fdf0
# ╠═22c726a0-12f0-11eb-0ba6-5da513dbc281
# ╠═0269168e-12ee-11eb-3dd1-77d75ba692a9
# ╠═754bd8f0-12f3-11eb-3530-b976559c01e3
# ╠═e0c7ff80-13f9-11eb-2b71-f1e33226bf6a
# ╠═8f9f3260-12f3-11eb-18a3-571a47653d95
# ╟─496a7360-12f6-11eb-0a39-0db347ba7df9
# ╟─0835eac0-12f9-11eb-097e-9903d7e61752
# ╟─d06552b0-12f9-11eb-1f34-df23cefc2d1e
# ╠═ede4e5d0-12f9-11eb-253b-85703a7d49ed
# ╟─29c5c410-12fb-11eb-29c2-a1c94c7ff235
# ╟─614fe7d0-12fb-11eb-1736-8bc134642481
# ╟─6abaa440-12fb-11eb-0731-fb49bf5b4b91
# ╟─220c8130-1348-11eb-2cc4-27aef429ae45
# ╟─df62e7f0-134e-11eb-1d5f-193c6dddd561
# ╟─5a07b640-1348-11eb-2f41-155028e71b6b
# ╟─84a4fbb0-1348-11eb-03d3-ed65b20129af
# ╠═b2527c70-134a-11eb-3791-0bd358c258dc
# ╠═9feb9950-1349-11eb-0d0c-d7ed01605603
# ╠═2dc895be-134a-11eb-0170-d38101fb1a7f
# ╠═358ca710-134a-11eb-393d-5dc3aa30d2f1
# ╠═e387f2d0-1349-11eb-3c51-11a55c4243ea
# ╠═ef3cc650-1349-11eb-2f20-1769c55e544e
# ╠═4ea70b52-134a-11eb-24bb-8ddd6484fd93
# ╠═6ac67230-134a-11eb-2990-f31f345f2a46
# ╟─4322ef50-134b-11eb-2602-350852cc9cb0
# ╟─5e3c30d0-134b-11eb-203d-7748430c642d
# ╟─daebb7b0-134e-11eb-1b3e-412480fe8c3e
# ╟─20bf4e70-134d-11eb-247f-9bff99598c13
# ╟─1f6e8b80-134d-11eb-2400-8d6be329adcc
# ╟─3a6e8dbe-134f-11eb-0f93-351e2ab4ddcf
# ╟─50651cc0-134f-11eb-2b46-b1038d782cf2
# ╟─649bfa60-134f-11eb-3fc6-1ba1d282880a
# ╟─a560fb80-1350-11eb-3f52-b931d3c5a76b
# ╟─bc718600-1350-11eb-2e14-818244783841
# ╟─f5bda510-1350-11eb-05b3-07cf839263b3
# ╠═0cb3c9c0-1351-11eb-16e6-f340951ccb8a
# ╟─6037c3a0-1354-11eb-2a53-e347f53030ff
# ╟─855c9140-135b-11eb-0d0d-edd695624daf
# ╟─b9dfdda0-135b-11eb-293b-1357e85b8839
# ╟─c9fb7130-135c-11eb-24ec-7ff0c2f85c0f
# ╟─2caacf10-135d-11eb-0e0c-7916ac6271b4
# ╠═9b5ca0ae-135c-11eb-01b0-c5e8493a801d
# ╠═8481d6c0-135d-11eb-37c3-9b4005ccc90e
# ╠═89312e00-135d-11eb-36eb-0d6ae62b2f91
# ╟─f88cba72-13db-11eb-0b50-bfe67120b1c4
# ╠═003d8f20-135e-11eb-2ae6-db430cf84da5
# ╠═2e88ed30-13c6-11eb-0bda-c1588b0e766d
# ╠═e66a6230-135d-11eb-1cdb-8d9506b3ffdf
# ╠═17940990-13dd-11eb-349a-19702cc7fd55
# ╠═54f35f90-13e0-11eb-1fa0-4f08e7af3c71
# ╟─8476d170-13e5-11eb-1c42-bd1cae451004
# ╟─9eedea10-13e6-11eb-2047-2b7befc81017
# ╟─c8ee2c2e-13e6-11eb-19e5-6f2d481f020b
# ╟─0d7310f0-13e7-11eb-15f8-29dc03303c66
# ╟─309f6e1e-13e7-11eb-0cab-914cb1c7dc19
# ╟─3d7764e0-13e7-11eb-36e7-0f7c7b5da594
# ╟─a39d72f0-13e7-11eb-03cf-db4610a63043
# ╟─b4b73210-13e7-11eb-12cf-03a37e027d0f
# ╟─1a6c3960-13e9-11eb-3b9d-3bec96635025
# ╟─45e279b0-13e9-11eb-23d4-9355cdd7580f
# ╟─470db340-13e9-11eb-02b1-254178507905
# ╟─7dd3eac0-13e9-11eb-3d18-c5958ddf0882
# ╟─c4364210-13e9-11eb-1e5c-25dbca34084c
# ╠═dfbc9d40-13e9-11eb-2cda-a75bb4153611
# ╠═d05c7500-13e9-11eb-3a01-a5832a81d5a6
# ╟─558827b0-13ef-11eb-3ba6-c7e16975893d
