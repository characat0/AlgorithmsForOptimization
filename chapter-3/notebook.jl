### A Pluto.jl notebook ###
# v0.12.4

using Markdown
using InteractiveUtils

# ╔═╡ e3d74dc0-1048-11eb-0cc5-070b61b7d0f7
using Base.MathConstants

# ╔═╡ f4a1cbf0-122b-11eb-32de-c558c4ada9d3
using Plots

# ╔═╡ 4dd8d070-123a-11eb-11be-d7d538b2b55e
using Calculus

# ╔═╡ 55eada40-1043-11eb-24f6-05ce09cfda18
md"# Bracketing"

# ╔═╡ 900a00c0-1043-11eb-233c-77a08edc03ca
md"Bracketing is the process an interval for a local minima and shrinking it."

# ╔═╡ a2527be0-1043-11eb-0063-472e67cbb285
md"## Unimodality"

# ╔═╡ b58975b0-1043-11eb-28c0-ad5c106d2bee
md"A unimodal function f is one where there is a unique x*, such that f is monotonically decreasing for x≤x* and monotonically increasing for x≥x*. So the unique glogal minimum is at x*."

# ╔═╡ 1027b3b0-1044-11eb-1869-c3f982c419f2
md"We can bracket a unimodal function finding three points a<b<c, such that f(a)>f(b)<f(c)"

# ╔═╡ 31f28510-1044-11eb-2b0d-59a2bfb98528
md"## Finding an Initial Bracket"

# ╔═╡ 3b532f5e-1044-11eb-3a72-bd015d93b64b
md"Simple bracket finder."

# ╔═╡ bee01870-1044-11eb-1cec-2bed3059fa12
function bracket_minimum(f, x=0; s=1e-12, k=2.0)
	a, ya = x, f(x)
	b, yb = a+s, f(a+s)
	if yb > ya #To guarantee search downhill
		a, b = b, a
		ya, yb = yb, ya
		s = -s
	end
	while true
		c, yc = b+s, f(b+s) #New point downhill
		if yc > yb #If the new point is higher than b
			return a < c ? (a, c) : (c, a)
			#Order according to going forward or backward (look at the sign of s)
		end
		a, ya, b, yb = b, yb, c, yc #Reasign the points
		s *= k
	end
end

# ╔═╡ 8d3c28d0-1045-11eb-35b7-a54ffd546850
md"Honoring Raja I will attempt a recursive version"

# ╔═╡ a8829f60-1046-11eb-1c49-5155ad0a99c5
function bracket_minimum_recursive(f, x=0; s=1e-12, k=2.0)
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

# ╔═╡ e40f4320-1047-11eb-1157-994f95d2da96
md"There is really no difference between these two function call"

# ╔═╡ 62593cf2-1047-11eb-068a-6984b2ded0c9
bracket_minimum(cos, -0.4; k=3.0)

# ╔═╡ 70a2ec20-1047-11eb-16bc-cde3f62d9062
a, b = bracket_minimum_recursive(cos, -0.4; k=3.0)

# ╔═╡ f89c7e20-1047-11eb-3f23-f7d352696e20
md"## Fibonacci Search"

# ╔═╡ 0603204e-1048-11eb-1afd-59dcfc353d5f
md"For n queries, the interval length that we can reduce is related to the Fibonacci sequence."

# ╔═╡ b5887930-1291-11eb-0c46-8b73fb74a79b
md"Fibonacci method is GUARANTEED to maximally shrink the bracketed interval."

# ╔═╡ 4e1f9242-128e-11eb-0d22-896e77b6993f
md"It means we can only call the function f n times."

# ╔═╡ 526b1740-1291-11eb-13be-d7905a6c1c88
md"For n queries we are GUARANTEED to shrink the length of the interval by a factor of Fn+1."

# ╔═╡ 21bb12d0-1048-11eb-2a04-2f9016623006
φ

# ╔═╡ e1fc7340-1048-11eb-1d65-8bedb36ed2f4
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
	a < b ? (a, b) : (b, a)
end

# ╔═╡ d8dcfb30-1049-11eb-12e9-8ff708bfb8d7
fibonacci_search(cos, a, b, 10)

# ╔═╡ f8825482-1049-11eb-2ed1-65649ed54300
md"## Golden Section Search"

# ╔═╡ 16324670-104a-11eb-1bf8-6d62d19c3189
md"Golden section search uses the golden ratio to approximate Fibonacci search."

# ╔═╡ 2aa25c80-104a-11eb-2e4c-e5251ba672f2
function golden_section_search(f, a, b, n)
	p = φ-1
	d = p*b + (1-p)*a
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
	a < b ? (a, b) : (b, a)
end

# ╔═╡ b64b8900-104a-11eb-0b40-a7b2972762a7
golden_section_search(cos, a, b, 10)

# ╔═╡ c123df30-104a-11eb-1821-79572ecd6264
md"## Quadratic Fit Search"

# ╔═╡ 1192175e-122c-11eb-39e8-536d1fb44362
plotly()

# ╔═╡ 12315020-11d5-11eb-00ac-4b454efefd71
md"We will approximate the minimum using a parabola"

# ╔═╡ 55336570-11d5-11eb-03f2-79acdc28c029
md"Given a<b<c and yₐ, yb, yc we will find coefficients p₁, p₂, p₃ for a quadratic function q for the points described. "

# ╔═╡ bdb87220-11d5-11eb-23b7-5b235a45821a
md"q(x)=p₁+p₂x+p₃x²"

# ╔═╡ 67d555c0-11d6-11eb-3193-4d98e253295b
md"Using its matrix form and deriving we obtain the unique minimum"

# ╔═╡ d925aa90-11d6-11eb-1dc5-f12e1507ea74
function quadratic_fit_search(f, a, b, c, n)
	ya, yb, yc = f(a), f(b), f(c)
	for i in 1:n-3
		x = 0.5*(ya*(b^2-c^2) + yb*(c^2-a^2) + yc*(a^2-b^2)) / 
		       (ya*(b-c)      + yb*(c-a)     + yc*(a-b))
		yx = f(x)
		if x > b
			if yx > yb
				c, yc = x, yx
			else
				a, ya, b, yb = b, yb, a, ya
			end
		elseif x < b
			if yx > yb
				a, ya = x, yx
			else	
			endc, yc, b, yb = b, yb, x, yx
			end
		end
	end
	return (a, b, c)
end

# ╔═╡ 7d3a8420-122c-11eb-0e67-77cea6c239c9
begin
	f₁ = sin
	points = [-3, 0, 1]
	plots = []
	for i in 1:4
		plot(f₁, -4:0.05:1.5; legend=false)
		push!(plots, scatter!(points, f₁.(points); label="Iteration $i", c=:black))
		global points = collect(quadratic_fit_search(f₁, points..., 4))
	end
	plot(plots...)
end

# ╔═╡ 6d9da430-1230-11eb-3e3c-8f144c63b0a9
md"## Shubert-Piyavskii Method"

# ╔═╡ f34bbf8e-1230-11eb-291c-b34bc200b90f
md"This is a global optimization method over a domain [a,b], meaning it is GUARANTEED to converge on the global minimum of a function irrespective of any local minima or whether the function is unimodal."

# ╔═╡ 3fc03272-1231-11eb-00da-fd904dd81289
md"It requires that the function be Lipschitz continous, meaning that it is continous and there is an upper bound on the magnitude of its derivative."

# ╔═╡ 620033d0-1231-11eb-22b9-2960c454bcb7
md"Formal definition: A function *f* is Lipschitz continous on [a,b] if there exists an *l* > 0 such that:"

# ╔═╡ 744ef080-1231-11eb-1c1d-8152b6d3c66c
md"\$\$|f(x)-f(y)|≤l|x-y|\\ for\\ all\\ x,y ∈ [a,b]\$\$"

# ╔═╡ b2c34050-1231-11eb-2573-c3f332f645e4
md"Then l is as large as the largest unsigned derivative of f in [a,b]"

# ╔═╡ da29c4c0-1231-11eb-04ee-f9afbe48f9f1
md"The drawback of this method is that we need prior knowledge of the l constant, as large values of l will yield in poor performance."

# ╔═╡ 43914230-1232-11eb-2c25-cff7d8549c71
struct Pt
	x
	y
end

# ╔═╡ 98b7a8d0-1232-11eb-37fb-6f2cb93b5f8d
function _get_sp_intersection(A, B, l)
	t = ((A.y - B.y) - l*(A.x - B.x))/2l
	Pt(A.x + t, A.y -t*l)
end

# ╔═╡ ba408f30-1232-11eb-1d2a-dbd9b15acb05
function shubert_piyavskii(f, a, b, l, ϵ; δ=0.01)
	m = (a+b)/2
	A, M, B = Pt(a, f(a)), Pt(m, f(m)), Pt(b, f(b))
	pts = [A, _get_sp_intersection(A, M, l),
		   M, _get_sp_intersection(M, B, l), B]
	Δ = Inf
	while Δ > ϵ
		i = argmin([P.y for P in pts])
		pt = pts[i]
		P = Pt(pt.x, f(pt.x))
		Δ = P.y - pt.y
		P_prev = _get_sp_intersection(pts[i-1], P, l)
		P_next = _get_sp_intersection(P, pts[i+1], l)
		deleteat!(pts, i)
		insert!(pts, i, P_next)
		insert!(pts, i, P)
		insert!(pts, i, P_prev)
	end
	intervals = []
	i = 2*(argmin([P.y for P in pts[1:2:end]])) - 1
	for j in 2:2:length(pts)
		if pts[j].y < pts[i].y
			dy = pts[i].y - pts[j].y
			x_lo = max(a, pts[j].x - dy/l)
			x_hi = min(b, pts[j].x + dy/l)
			if !isempty(intervals) && intervals[end][2] + δ ≥ x_lo
				intervals[end] = (intervals[end][1], x_hi)
			else
				push!(intervals, (x_lo, x_hi))
			end
		end
	end
	(pts[i], intervals)
end

# ╔═╡ fd3dcd10-1233-11eb-19a9-c3ce0ab0ba2e
shubert_piyavskii(f₁, -3, 1, 1, 1e-6)

# ╔═╡ 272975c0-1234-11eb-30ca-0f9b68f32fb8
md"## Bisection method"

# ╔═╡ 7f7ae820-1235-11eb-075c-8f990b4cadbc
md"The bisection method can be used to find roots of a function, or points where the function is zero. It can be used for optimization by applying them to the derivative, finding f'(x)=0."

# ╔═╡ 9f0677e0-1235-11eb-32a9-d75c02af5b5a
md"The Brent-Dekker method is an extension of the bisection method, with fast convergence properties and its popular among optimization packages."

# ╔═╡ f3239c80-1290-11eb-10f5-63ef1aa6416f
md"This method is GUARANTEED to converge within ϵ of x* within ln(|b-a|/ϵ) iterations."

# ╔═╡ a15a4100-1237-11eb-3d38-5b9c7a053aae
function bisection(f::Function, a::Number, b::Number, ϵ::Float64)
	if a > b
		a, b = b, a
	end
	
	while b-a > ϵ
		x = (a+b)/2
		if f(x) == 0
			return (x, x)
		end
		if sign(f(a)) === sign(f(b))
			a = x
		else
			b = x
		end
	end
	return (a, b)
end

# ╔═╡ 7cabd7b0-1237-11eb-09aa-eb36da4d4fbf
function bracket_sign_change(f, a, b; k=2)
	if a>b;a,b=b,a;end
	center, half_width = (b+a)/2, (b-a)/2
	while f(a)*f(b)>0
		half_width *= k
		a = center - half_width
		b = center + half_width
	end
	return (a, b)
end

# ╔═╡ 5aaf67a0-123a-11eb-3cbf-671a142ed7a3
plot(derivative(f₁))

# ╔═╡ 1de931d0-128e-11eb-16e2-c3fefe0cf1d3
md"## Excercises"

# ╔═╡ 242d80f0-128e-11eb-2220-cb19a1056a13
md"**Excercise 3.1.** Give an example of a problem when Fibonacci search is preferred over the bisection method."

# ╔═╡ 428c2972-128e-11eb-29bd-1f86dd63fa46
md"**Answer**: A problem when the function to be minimized is computational expensive and thus we need to put a limit to the times we can call it."

# ╔═╡ 91527b90-128e-11eb-17ea-6fc614f15ba0
md"**Excercise 3.2.** What is a drawback of the Shubert-Piyavskii method?"

# ╔═╡ a9243330-128e-11eb-11c3-6f086b2ba650
md"**Answer**: The fact that we have to know constant l prior to the calculation, this can be seen as a second optimization (maximization in an interval) problem. So Shubert-Piyavskii method is not suitable for any function which derivative has no easly calculated upper bond."

# ╔═╡ 07fdf800-128f-11eb-02dd-ab1979373c45
md"**Excercise 3.3.** Suppose we have f(x)=x²/2-x. Apply the bisection method to find an interval containing the minimizer of f starting with the interval [0, 1000]. Evaluate three steps of the algorithm."

# ╔═╡ 3172ea10-128f-11eb-0605-ab033ef49512
function bisection(f::Function, a::Number, b::Number, n::Integer)
	if a > b
		a, b = b, a
	end
	points = [(a, f(a)), (b, f(b))]
	while n > 0
		x = (a+b)/2
		if f(x) == 0
			return (x, x)
		end
		if sign(f(a)) === sign(f(b))
			a = x
		else
			b = x
		end
		n -= 1
		push!(points, (a, f(a)))
		push!(points, (b, f(b)))
	end
	return (a, b, points)
end

# ╔═╡ 1ba148d0-123a-11eb-2f8c-61c63dd525a5
bisection(derivative(f₁), -3, -1, 0.0001)

# ╔═╡ 54336610-128f-11eb-00be-3d68658ee4c2
f₂(x) = x^2/2 - x

# ╔═╡ 5e2f8d10-128f-11eb-00bf-859d1c94628f
res = bisection(f₂, 0, 1000, 3)

# ╔═╡ cc572aa0-128f-11eb-252a-49c4a542ecee
begin
	plot(f₂, 0:1:1100)
	scatter!(res[3])
end

# ╔═╡ 226e0bc0-1290-11eb-1901-b3d73e19299f
md"**Excercise 3.5.** Suppose we have a function f(x)=(x+2)² on the interval [0,1]. Is 2 a valid Lipschitz constant for f on that interval?"

# ╔═╡ 44b5d550-1290-11eb-03d3-0be0077c39e1
md"**Answer**: The derivative of f is 2(x+2), which is monotonically increasing on the reals, and has a maximum of 6 on the interval [0,1], so 2 is not a valid Lipschitz constant for f on the interval [0,1]"

# ╔═╡ 7c9b1160-1290-11eb-021f-f37421c86ec0
begin
	plot(derivative(x -> (x+2)^2), -.25:.05:1.2)
	scatter!((1, 6))
end

# ╔═╡ c0417d00-1290-11eb-0664-7197867ddb7f
md"**Excercise 3.6.** Suppose we have a unimodal function defined on the interval [1,32]. After three function evaluations of our choice, will we be able to narrow the optimum to an interval of at most length 10? Why or why not?"

# ╔═╡ e61c0a40-1290-11eb-31c2-89db705f93e3
md"**Answer**: If we chose the Fibonacci method, we can shrink the interval by a factor of 3, then if the length is 31 we can reduce it at most to 10.33, and as the Fibonacci method is guaranteed to maximally shrink the interval, we cannot go any further."

# ╔═╡ Cell order:
# ╟─55eada40-1043-11eb-24f6-05ce09cfda18
# ╟─900a00c0-1043-11eb-233c-77a08edc03ca
# ╟─a2527be0-1043-11eb-0063-472e67cbb285
# ╟─b58975b0-1043-11eb-28c0-ad5c106d2bee
# ╟─1027b3b0-1044-11eb-1869-c3f982c419f2
# ╟─31f28510-1044-11eb-2b0d-59a2bfb98528
# ╟─3b532f5e-1044-11eb-3a72-bd015d93b64b
# ╠═bee01870-1044-11eb-1cec-2bed3059fa12
# ╟─8d3c28d0-1045-11eb-35b7-a54ffd546850
# ╠═a8829f60-1046-11eb-1c49-5155ad0a99c5
# ╟─e40f4320-1047-11eb-1157-994f95d2da96
# ╠═62593cf2-1047-11eb-068a-6984b2ded0c9
# ╠═70a2ec20-1047-11eb-16bc-cde3f62d9062
# ╟─f89c7e20-1047-11eb-3f23-f7d352696e20
# ╟─0603204e-1048-11eb-1afd-59dcfc353d5f
# ╟─b5887930-1291-11eb-0c46-8b73fb74a79b
# ╟─4e1f9242-128e-11eb-0d22-896e77b6993f
# ╟─526b1740-1291-11eb-13be-d7905a6c1c88
# ╠═e3d74dc0-1048-11eb-0cc5-070b61b7d0f7
# ╠═21bb12d0-1048-11eb-2a04-2f9016623006
# ╠═e1fc7340-1048-11eb-1d65-8bedb36ed2f4
# ╠═d8dcfb30-1049-11eb-12e9-8ff708bfb8d7
# ╟─f8825482-1049-11eb-2ed1-65649ed54300
# ╟─16324670-104a-11eb-1bf8-6d62d19c3189
# ╠═2aa25c80-104a-11eb-2e4c-e5251ba672f2
# ╠═b64b8900-104a-11eb-0b40-a7b2972762a7
# ╟─c123df30-104a-11eb-1821-79572ecd6264
# ╠═f4a1cbf0-122b-11eb-32de-c558c4ada9d3
# ╠═1192175e-122c-11eb-39e8-536d1fb44362
# ╟─12315020-11d5-11eb-00ac-4b454efefd71
# ╟─55336570-11d5-11eb-03f2-79acdc28c029
# ╟─bdb87220-11d5-11eb-23b7-5b235a45821a
# ╟─67d555c0-11d6-11eb-3193-4d98e253295b
# ╠═d925aa90-11d6-11eb-1dc5-f12e1507ea74
# ╠═7d3a8420-122c-11eb-0e67-77cea6c239c9
# ╟─6d9da430-1230-11eb-3e3c-8f144c63b0a9
# ╟─f34bbf8e-1230-11eb-291c-b34bc200b90f
# ╟─3fc03272-1231-11eb-00da-fd904dd81289
# ╟─620033d0-1231-11eb-22b9-2960c454bcb7
# ╟─744ef080-1231-11eb-1c1d-8152b6d3c66c
# ╟─b2c34050-1231-11eb-2573-c3f332f645e4
# ╟─da29c4c0-1231-11eb-04ee-f9afbe48f9f1
# ╠═43914230-1232-11eb-2c25-cff7d8549c71
# ╠═98b7a8d0-1232-11eb-37fb-6f2cb93b5f8d
# ╠═ba408f30-1232-11eb-1d2a-dbd9b15acb05
# ╠═fd3dcd10-1233-11eb-19a9-c3ce0ab0ba2e
# ╟─272975c0-1234-11eb-30ca-0f9b68f32fb8
# ╟─7f7ae820-1235-11eb-075c-8f990b4cadbc
# ╟─9f0677e0-1235-11eb-32a9-d75c02af5b5a
# ╟─f3239c80-1290-11eb-10f5-63ef1aa6416f
# ╠═a15a4100-1237-11eb-3d38-5b9c7a053aae
# ╠═7cabd7b0-1237-11eb-09aa-eb36da4d4fbf
# ╠═4dd8d070-123a-11eb-11be-d7d538b2b55e
# ╠═5aaf67a0-123a-11eb-3cbf-671a142ed7a3
# ╠═1ba148d0-123a-11eb-2f8c-61c63dd525a5
# ╟─1de931d0-128e-11eb-16e2-c3fefe0cf1d3
# ╟─242d80f0-128e-11eb-2220-cb19a1056a13
# ╟─428c2972-128e-11eb-29bd-1f86dd63fa46
# ╟─91527b90-128e-11eb-17ea-6fc614f15ba0
# ╟─a9243330-128e-11eb-11c3-6f086b2ba650
# ╟─07fdf800-128f-11eb-02dd-ab1979373c45
# ╠═3172ea10-128f-11eb-0605-ab033ef49512
# ╠═54336610-128f-11eb-00be-3d68658ee4c2
# ╠═5e2f8d10-128f-11eb-00bf-859d1c94628f
# ╠═cc572aa0-128f-11eb-252a-49c4a542ecee
# ╟─226e0bc0-1290-11eb-1901-b3d73e19299f
# ╟─44b5d550-1290-11eb-03d3-0be0077c39e1
# ╠═7c9b1160-1290-11eb-021f-f37421c86ec0
# ╟─c0417d00-1290-11eb-0664-7197867ddb7f
# ╟─e61c0a40-1290-11eb-31c2-89db705f93e3
