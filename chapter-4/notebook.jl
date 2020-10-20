### A Pluto.jl notebook ###
# v0.12.4

using Markdown
using InteractiveUtils

# ╔═╡ 035b4c80-12f3-11eb-3ce9-e5153281a50e
using Base.MathConstants

# ╔═╡ 754bd8f0-12f3-11eb-3530-b976559c01e3
using Plots

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

# ╔═╡ 08d112f0-12f1-11eb-3ec7-bdda6f6eb3e6
f₁(x̄) = (1-x̄[1])^2 + 5(x̄[2] - x̄[1]^2)^2

# ╔═╡ e1b8f540-12f3-11eb-1aa6-ed28dd3748b8
f₁(x, y) = f₁([x, y])

# ╔═╡ 8f9f3260-12f3-11eb-18a3-571a47653d95
begin
	x_initial = [0, .1]
	x_opt = line_search(f₁, x_initial, [.8, .8])
	x = range(-1, stop=2.5, length=200)
	y = x
	color = cgrad([:red, :green, :blue])
	contour(x, y, f₁, c=color, levels=[0:.1:.5 ; 2:10:250])
	scatter!(tuple(x_initial...), c=color[f₁(x_initial)], label="Initial point")
	scatter!(tuple(x_opt...), c=color[f₁(x_opt)], label="Optimal")
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
# ╠═c7b2e030-12ed-11eb-211f-ebbae3a4411f
# ╠═0a002f00-12ef-11eb-1e71-87804c2d8d0e
# ╠═035b4c80-12f3-11eb-3ce9-e5153281a50e
# ╠═14897ace-12ef-11eb-2d95-d9fcc1f7fdf0
# ╠═22c726a0-12f0-11eb-0ba6-5da513dbc281
# ╠═0269168e-12ee-11eb-3dd1-77d75ba692a9
# ╠═754bd8f0-12f3-11eb-3530-b976559c01e3
# ╠═08d112f0-12f1-11eb-3ec7-bdda6f6eb3e6
# ╠═e1b8f540-12f3-11eb-1aa6-ed28dd3748b8
# ╠═8f9f3260-12f3-11eb-18a3-571a47653d95
# ╟─496a7360-12f6-11eb-0a39-0db347ba7df9
# ╟─0835eac0-12f9-11eb-097e-9903d7e61752
# ╟─d06552b0-12f9-11eb-1f34-df23cefc2d1e
# ╠═ede4e5d0-12f9-11eb-253b-85703a7d49ed
# ╟─29c5c410-12fb-11eb-29c2-a1c94c7ff235
# ╠═614fe7d0-12fb-11eb-1736-8bc134642481
# ╠═6abaa440-12fb-11eb-0731-fb49bf5b4b91
