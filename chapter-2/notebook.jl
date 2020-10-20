### A Pluto.jl notebook ###
# v0.12.4

using Markdown
using InteractiveUtils

# ╔═╡ f48c5130-0e93-11eb-15a5-55f13d69cbdd
using Plots

# ╔═╡ eddcb5c0-0e96-11eb-3edf-854adf01d3f1
using Calculus

# ╔═╡ 094dc520-0ea0-11eb-335f-2bb624eae114
using DualNumbers

# ╔═╡ f7514f80-0e91-11eb-2c8b-77c92b98f3ea
md"# Derivatives and Gradients"

# ╔═╡ 09ca9e9e-0e92-11eb-056a-535e6b814fdd
md"## Derivatives"

# ╔═╡ b3c4c750-0efb-11eb-3acd-cb737f1149b8
plotly()

# ╔═╡ d53163d0-0e92-11eb-3c85-4b26c5220982
f1(x) = x^2 + x/2 - sin(x)/x

# ╔═╡ ebbf29c0-0e92-11eb-3d6d-0fad2c980734
derivative(f1)

# ╔═╡ 2256ec50-0e95-11eb-2e85-e57fe6438e9b
md"## Derivatives in Multiple Dimensions"

# ╔═╡ 2413cca0-0e97-11eb-3811-df41d41442ff
gradient = Calculus.gradient

# ╔═╡ f24f2b10-0e96-11eb-1de4-2935cc2bb539
f2(x) = x[1]*sin(x[2])+1

# ╔═╡ 15ce3c20-0e97-11eb-1826-8dc973daeaaa
gradient(f2, [2.0, 0.0])

# ╔═╡ de45a98e-0e97-11eb-36f5-410b13e18cc3
md"Directional derivative is the instantaneous rate of change of f(x) as x is moved with velocity s ∇ₛf(x)=∇f(x)ᵀs"

# ╔═╡ 983685e0-0e98-11eb-2940-455516c53510
md"Compute the directional derivative of f(x) at x=(1, 0) in the direction s=(-1, -1)"

# ╔═╡ e4a5be10-0e97-11eb-3d63-19d897342b8e
f3(x) = x[1]*x[2]

# ╔═╡ f2de2f30-0e97-11eb-15fe-4b6964bc26c9
∇f₃ = gradient(f3)

# ╔═╡ f427d0c0-0e98-11eb-3127-65364fa62190
s = [-1, -1]

# ╔═╡ 48107cb0-0e98-11eb-0d23-3bc969421cd6
∇ₛf₃ = [-1 -1]*∇f₃([1, 0])

# ╔═╡ b7b2ac52-0e98-11eb-3388-c9800aaa6e04
md"or..."

# ╔═╡ 7933dfc0-0e99-11eb-1a0d-4ffafab1f4eb
md"g(α) = f(x+αs) = (1-α)*(-α) = α^2-1"

# ╔═╡ ba194c10-0e98-11eb-0586-ab17d87ffcfe
g(α) = α^2-1

# ╔═╡ e9b8758e-0e98-11eb-2a24-bffe61d041c1
derivative(g, 0)

# ╔═╡ 79df9070-0e96-11eb-161f-974a6a555348
md"## Numerical differentiation"

# ╔═╡ a3681510-0e97-11eb-0a06-d5d8cccf3fc7
md"### Finite Difference Methods"

# ╔═╡ cbb7e812-0e97-11eb-36fa-836f3c574f12
begin
	diff_forward(f, x; h=sqrt(eps())) = (f(x+h) - f(x))/h
	diff_central(f, x; h=cbrt(eps())) = (f(x+h/2) -f(x-h/2))/h
	diff_backward(f, x; h=sqrt(eps())) = (f(x) - f(x-h))/h
end;

# ╔═╡ 7e1a5722-0e9a-11eb-199d-ebf18007bd0a
md"Note that the central difference method has O(h²) error, while forward and backward have O(h)."

# ╔═╡ 96782220-0e9a-11eb-1995-17ba8f668c40
md"### Complex Step Method"

# ╔═╡ a04f0dde-0e9a-11eb-3eb8-a391bb06ecc1
md"The complex step method bypasses any numerical cancellation issue by evaluating the function after taking a step in the imaginary direction."

# ╔═╡ 7b119910-0e9c-11eb-042b-9b5d59041f32
diff_complex(f, x; h=1e-20) = imag(f(x+h*im)) / h

# ╔═╡ 99ffd762-0e9c-11eb-08ab-e3a0c70c3eed
begin
	logrange(x1, x2, n) = (10^y for y in range(log10(x1), log10(x2), length=n))
	f₄(x) = sin(x^2)
	val = π/2
	n = 50
	ground = (x -> 2x*cos(x^2))(val)
	h₄ = logrange(1e-18, 1, n).*1
	complex_err = abs.([diff_complex(f₄,val; h=h) for h in h₄].-ground)
	forward_err = abs.([diff_forward(f₄,val; h=h) for h in h₄].-ground)
	central_err = abs.([diff_central(f₄,val; h=h) for h in h₄].-ground)
	
	plot(h₄, complex_err; xscale=:log10, yaxis=(:log10, (1e-20,10)), c=:green, label="complex")
	plot!(h₄, forward_err; xscale=:log10, yaxis=(:log10, (1e-20,10)), c=:blue, label="forward")
	plot!(h₄, central_err; xscale=:log10, yaxis=(:log10, (1e-20,10)), c=:red, label="central")
end

# ╔═╡ 003399ae-0ea0-11eb-071d-fb202e1e6c0f
md"## Automatic Differentiation"

# ╔═╡ 0b29d680-0f00-11eb-2e46-231914e2c600
md"### Forward Accumulation"

# ╔═╡ 48fd7980-0f00-11eb-13fb-95ffc1a0faef
html"Dual numbers work like this:"

# ╔═╡ 787e6ca0-0f00-11eb-1879-596402b26c01
md"ϵ² is defined to be 0."

# ╔═╡ 58861150-0f00-11eb-1fc7-8f3677ec236d
md"\$\$f(a+bϵ) = f(a)+bf'(a)ϵ\$\$"

# ╔═╡ c7f76262-0eff-11eb-00e1-e70bb1285657
f(x) = x^3

# ╔═╡ 94260b70-0f00-11eb-2f45-2d66ed42cabb
md"So passing a dual númber of the form x + ϵ will compute the derivative at point x"

# ╔═╡ 06f9ea10-0ea0-11eb-1de2-0fee662d33cc
dualpart(f(Dual(2, 1)))

# ╔═╡ 187353c0-0f00-11eb-3840-4d50eca6fe4a
md"## Excercises"

# ╔═╡ 5b6239c0-0f01-11eb-2929-61ea1efb391b
md"**Excercise 2.1.** Adopt the forward difference method to approximate the Hessian of f(x) using its gradient, ∇f(x)"

# ╔═╡ 83ecdbf0-0f3a-11eb-3342-31ed64c7890d
import Zygote

# ╔═╡ 8ace6c60-0f01-11eb-30d2-477f4750ef17
f₅(x, y) = log(x*y + max(x, 2))

# ╔═╡ 5c72b430-0f55-11eb-2033-557c120408f0
md"**Answer**:We can define the i-th row of the hessian as the derivative of the gradient of f with respect to xᵢ"

# ╔═╡ 832d412e-0f55-11eb-14b4-eff406dfc380
md"\$\$Hf_i = \frac{δ(∇f)}{δxᵢ}\$\$"

# ╔═╡ ffcb1c30-0f55-11eb-1589-110f40e2ce3e
md"Then in our code we calculate the gradient at x, and for the i-th variable calculate the gradient of f at (x₁,…, xᵢ+h,…,xₙ) and use the forward difference method to calculate the i-th row of the Hessian."

# ╔═╡ 7adc87a0-0f52-11eb-04e5-3dc816c09921
function hessian_forward(f::Function, x::Number...; h=sqrt(eps()))
	n = length(x)
	∇f = collect(Zygote.gradient(f, x...))
	Hf = Matrix{Float64}(undef, n, n)
	for i in 1:n
		x₂ = copy(collect(x))
		x₂[i] = x₂[i] + h
		Hf[i, :] = (collect(Zygote.gradient(f, x₂...)) - ∇f)./h
	end
	Hf
end

# ╔═╡ 54b84730-0f52-11eb-0243-49019d5cafca
hessian_forward(f₅, 3.0, 2.0)

# ╔═╡ 9a499612-0f56-11eb-3e41-d1338960969a
md"**Excercise 2.2.** What is the drawback of the central difference method over other finite difference methods if we already know f(x)"

# ╔═╡ ba273d70-0f56-11eb-3bff-f30c1d94fde8
md"**Answer**: We have to calculate f two times, because the central difference method does not use f evaluated at x."

# ╔═╡ 7b042520-0f58-11eb-3690-65d281b01846
md"**Excercise 2.3.** Compute the gradient of *f(x) = lnx + eˣ 1/x* for a point x close to zero. What term dominates in the expression?"

# ╔═╡ a89b70b0-0f58-11eb-0bbe-8fd26f8f02d3
f₆(x) = log(x) + exp(x) + 1/x

# ╔═╡ 28cde420-0f59-11eb-01f9-bba51a287fbc
plot(0.01:0.01:5, derivative(f₆))

# ╔═╡ f46ee8a0-0f58-11eb-016c-e1f2356a5e11
md"**Answer**: -1/x² dominates the gradient when the values of x approach 0"

# ╔═╡ 8cc81180-0f59-11eb-276c-31f1ccc9d5e1
md"**Excercise 2.4.** Suppose f(x) is a real-valued function that is also defined for complex imputs. If f(3+ih) = 2 +4ih, what is f'(3)"

# ╔═╡ b0c1348e-0f59-11eb-1942-2fce8264659a
md"**Answer**: From the complex step method, whe know that f'(x) = Im(f(x+ih))/h"

# ╔═╡ f4b71fc0-0f59-11eb-1d6a-dbe41db24e99
md"Then f'(3) = Im(2+4ih)/h = 4"

# ╔═╡ 07eb0c50-0f5a-11eb-0fbc-9778c964eeb4
md"**Excercise 2.5.** Draw the computational graph for f(x,y) = sin(x + y²). Use the computational graph with forward accumulation to compute ∂f/∂y at (x,y) = (1,1). Label the intermediate values and partial derivatives as they are propagated through the graph."

# ╔═╡ a97d4860-0f5c-11eb-3662-35d66132fd4d
md"**Answer**: Using Dual number as a means for automatic forward accumulation calculation we get that the derivative of f with respect to y at (x,y)=(1,1) is the same as the analytic result (2cos(2))."

# ╔═╡ 6d9310c0-0f5a-11eb-2173-233e40f0cd31
f₇(x,y) = sin(x+y^2)

# ╔═╡ 7a3810a0-0f5a-11eb-3bd3-59c223c142a7
dualpart(f₇(Dual(1,0), Dual(1,1)))

# ╔═╡ 4395865e-0f5d-11eb-18b1-17cccf2d30ff
md"**Excercise 2.6.** Combine the forward and backward difference methods to obtain a difference method for estimating the second-order derivative of a function f at x using three function evaluations."

# ╔═╡ 642c92b0-0f5d-11eb-34af-0fcb66558144
md"**Answer**: We can define f''(x) using the forward difference method as \$\$ \lim_{h→0} \frac{f'(x+h) - f'(x)}{h} \$\$"

# ╔═╡ 39ea4910-0f5e-11eb-18af-1b974d5ffd16
md"Using the forward backward difference we get \$\$\frac{\frac{f(x+h)-f(x)}{h} - \frac{f(x)-f(x-h)}{h}}{h}\$\$"

# ╔═╡ 7ce55980-0f5e-11eb-12de-efdfd94e073e
md"And simplyfing we get that \$\$f''(x) = \frac{f(x+h)-2f(x)+f(x-h)}{h^2}\$\$"

# ╔═╡ a03f0840-0f5e-11eb-1831-2b6083c4bd9d
second_diff_central(f, x; h=sqrt(eps())) = (f(x+h)-2f(x)+f(x-h))/h^2

# ╔═╡ d5648810-0f5e-11eb-0e61-bdab367b8599
second_diff_central(x -> x^4, 1)

# ╔═╡ Cell order:
# ╟─f7514f80-0e91-11eb-2c8b-77c92b98f3ea
# ╟─09ca9e9e-0e92-11eb-056a-535e6b814fdd
# ╠═f48c5130-0e93-11eb-15a5-55f13d69cbdd
# ╠═eddcb5c0-0e96-11eb-3edf-854adf01d3f1
# ╠═b3c4c750-0efb-11eb-3acd-cb737f1149b8
# ╠═d53163d0-0e92-11eb-3c85-4b26c5220982
# ╠═ebbf29c0-0e92-11eb-3d6d-0fad2c980734
# ╟─2256ec50-0e95-11eb-2e85-e57fe6438e9b
# ╠═2413cca0-0e97-11eb-3811-df41d41442ff
# ╠═f24f2b10-0e96-11eb-1de4-2935cc2bb539
# ╠═15ce3c20-0e97-11eb-1826-8dc973daeaaa
# ╟─de45a98e-0e97-11eb-36f5-410b13e18cc3
# ╟─983685e0-0e98-11eb-2940-455516c53510
# ╠═e4a5be10-0e97-11eb-3d63-19d897342b8e
# ╠═f2de2f30-0e97-11eb-15fe-4b6964bc26c9
# ╠═f427d0c0-0e98-11eb-3127-65364fa62190
# ╠═48107cb0-0e98-11eb-0d23-3bc969421cd6
# ╟─b7b2ac52-0e98-11eb-3388-c9800aaa6e04
# ╟─7933dfc0-0e99-11eb-1a0d-4ffafab1f4eb
# ╠═ba194c10-0e98-11eb-0586-ab17d87ffcfe
# ╠═e9b8758e-0e98-11eb-2a24-bffe61d041c1
# ╟─79df9070-0e96-11eb-161f-974a6a555348
# ╟─a3681510-0e97-11eb-0a06-d5d8cccf3fc7
# ╠═cbb7e812-0e97-11eb-36fa-836f3c574f12
# ╟─7e1a5722-0e9a-11eb-199d-ebf18007bd0a
# ╟─96782220-0e9a-11eb-1995-17ba8f668c40
# ╟─a04f0dde-0e9a-11eb-3eb8-a391bb06ecc1
# ╠═7b119910-0e9c-11eb-042b-9b5d59041f32
# ╠═99ffd762-0e9c-11eb-08ab-e3a0c70c3eed
# ╟─003399ae-0ea0-11eb-071d-fb202e1e6c0f
# ╟─0b29d680-0f00-11eb-2e46-231914e2c600
# ╠═094dc520-0ea0-11eb-335f-2bb624eae114
# ╟─48fd7980-0f00-11eb-13fb-95ffc1a0faef
# ╟─787e6ca0-0f00-11eb-1879-596402b26c01
# ╟─58861150-0f00-11eb-1fc7-8f3677ec236d
# ╠═c7f76262-0eff-11eb-00e1-e70bb1285657
# ╟─94260b70-0f00-11eb-2f45-2d66ed42cabb
# ╠═06f9ea10-0ea0-11eb-1de2-0fee662d33cc
# ╟─187353c0-0f00-11eb-3840-4d50eca6fe4a
# ╟─5b6239c0-0f01-11eb-2929-61ea1efb391b
# ╠═83ecdbf0-0f3a-11eb-3342-31ed64c7890d
# ╠═8ace6c60-0f01-11eb-30d2-477f4750ef17
# ╟─5c72b430-0f55-11eb-2033-557c120408f0
# ╟─832d412e-0f55-11eb-14b4-eff406dfc380
# ╟─ffcb1c30-0f55-11eb-1589-110f40e2ce3e
# ╠═7adc87a0-0f52-11eb-04e5-3dc816c09921
# ╠═54b84730-0f52-11eb-0243-49019d5cafca
# ╟─9a499612-0f56-11eb-3e41-d1338960969a
# ╟─ba273d70-0f56-11eb-3bff-f30c1d94fde8
# ╟─7b042520-0f58-11eb-3690-65d281b01846
# ╠═a89b70b0-0f58-11eb-0bbe-8fd26f8f02d3
# ╠═28cde420-0f59-11eb-01f9-bba51a287fbc
# ╟─f46ee8a0-0f58-11eb-016c-e1f2356a5e11
# ╟─8cc81180-0f59-11eb-276c-31f1ccc9d5e1
# ╟─b0c1348e-0f59-11eb-1942-2fce8264659a
# ╟─f4b71fc0-0f59-11eb-1d6a-dbe41db24e99
# ╟─07eb0c50-0f5a-11eb-0fbc-9778c964eeb4
# ╟─a97d4860-0f5c-11eb-3662-35d66132fd4d
# ╠═6d9310c0-0f5a-11eb-2173-233e40f0cd31
# ╠═7a3810a0-0f5a-11eb-3bd3-59c223c142a7
# ╟─4395865e-0f5d-11eb-18b1-17cccf2d30ff
# ╟─642c92b0-0f5d-11eb-34af-0fcb66558144
# ╟─39ea4910-0f5e-11eb-18af-1b974d5ffd16
# ╟─7ce55980-0f5e-11eb-12de-efdfd94e073e
# ╠═a03f0840-0f5e-11eb-1831-2b6083c4bd9d
# ╠═d5648810-0f5e-11eb-0e61-bdab367b8599
