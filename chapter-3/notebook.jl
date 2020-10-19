### A Pluto.jl notebook ###
# v0.12.4

using Markdown
using InteractiveUtils

# ╔═╡ e3d74dc0-1048-11eb-0cc5-070b61b7d0f7
using Base.MathConstants

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


# ╔═╡ Cell order:
# ╠═55eada40-1043-11eb-24f6-05ce09cfda18
# ╠═900a00c0-1043-11eb-233c-77a08edc03ca
# ╠═a2527be0-1043-11eb-0063-472e67cbb285
# ╠═b58975b0-1043-11eb-28c0-ad5c106d2bee
# ╠═1027b3b0-1044-11eb-1869-c3f982c419f2
# ╠═31f28510-1044-11eb-2b0d-59a2bfb98528
# ╠═3b532f5e-1044-11eb-3a72-bd015d93b64b
# ╠═bee01870-1044-11eb-1cec-2bed3059fa12
# ╠═8d3c28d0-1045-11eb-35b7-a54ffd546850
# ╠═a8829f60-1046-11eb-1c49-5155ad0a99c5
# ╠═e40f4320-1047-11eb-1157-994f95d2da96
# ╠═62593cf2-1047-11eb-068a-6984b2ded0c9
# ╠═70a2ec20-1047-11eb-16bc-cde3f62d9062
# ╠═f89c7e20-1047-11eb-3f23-f7d352696e20
# ╠═0603204e-1048-11eb-1afd-59dcfc353d5f
# ╠═e3d74dc0-1048-11eb-0cc5-070b61b7d0f7
# ╠═21bb12d0-1048-11eb-2a04-2f9016623006
# ╠═e1fc7340-1048-11eb-1d65-8bedb36ed2f4
# ╠═d8dcfb30-1049-11eb-12e9-8ff708bfb8d7
# ╠═f8825482-1049-11eb-2ed1-65649ed54300
# ╠═16324670-104a-11eb-1bf8-6d62d19c3189
# ╠═2aa25c80-104a-11eb-2e4c-e5251ba672f2
# ╠═b64b8900-104a-11eb-0b40-a7b2972762a7
# ╠═c123df30-104a-11eb-1821-79572ecd6264
