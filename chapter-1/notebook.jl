### A Pluto.jl notebook ###
# v0.12.4

using Markdown
using InteractiveUtils

# ╔═╡ 6f7ef8c2-0e8a-11eb-2095-674a0e527503
using Plots

# ╔═╡ 64f1e390-0e62-11eb-17e1-efb3f2d998f2
md"# Introduction"

# ╔═╡ b24bd470-0e62-11eb-1629-052ecfb22b47
md"## Optimization in history"

# ╔═╡ 398daf30-0e63-11eb-1524-5f84a5e3fbdc
html"Physics -> Lower energy<br>Business -> Maximize Shareholder value<br>Biology -> Fitter organism"

# ╔═╡ 5bf4e7a0-0e63-11eb-3f3d-ddeaf5a86f17
html"Process = Algorithm"

# ╔═╡ b08e0260-0e63-11eb-3f16-97a6cbbf5297
html"Pythagoras said that the mathematics are the principles of all things"

# ╔═╡ c81654f0-0e63-11eb-3817-2d7fadabf44b
md"Aristotle used algorithms to make deductions"

# ╔═╡ d879e370-0e63-11eb-3f46-89866807d8f1
md"Algorithm comes from algoritmi, latin translation of al-Khwãrizmi (Algebra)"

# ╔═╡ 598a33c0-0e64-11eb-3b03-0df474b87d4d
md"René Descartes created coordinated system, and maybe father of derivative."

# ╔═╡ 7d004420-0e64-11eb-3612-875d350e9065
md"Linear programming, George Dantzig developed Simplex algorithm."

# ╔═╡ 4fc54c92-0e68-11eb-36dd-1d847281c994
md"## Optimization Process"

# ╔═╡ d7451ab0-0e68-11eb-3cd4-2320283e74e6
html"Design -> Evaluate -> If not good enough Change design, else Celebrate"

# ╔═╡ 1a1c9d90-0e69-11eb-2ee0-49d236ca10d7
html"It is incremental"

# ╔═╡ a14f5d20-0e69-11eb-06ad-9fa7e0d4dd98
md"## Basic Optimization Problem"

# ╔═╡ 2c1bb3a0-0e69-11eb-227c-176bf3d49f2d
md"\begin{equation} minimize\\ f(x)\end{equation}"

# ╔═╡ 29cda4f0-0e69-11eb-3c9c-df53831b702f
md"\begin{equation} subject\\ to\\ x ∈ X\end{equation}"

# ╔═╡ d0da3b50-0e69-11eb-2153-472942520fc4
html"Any optimization problem can be rewritten to this form"

# ╔═╡ 0fd48400-0e6a-11eb-0640-512e799bc4da
md"*No free lunch theorems*, there is no reason to prefer one algorithm over another, unless assumpsions about the probability of the solution in the space are made"

# ╔═╡ d10976c0-0e6b-11eb-2ff2-69bed9537e9b
md"## Constraints"

# ╔═╡ 11676e20-0e6c-11eb-3916-edbdeedf37bc
html"Constraints limit the set of possible solutions"

# ╔═╡ 1d2ff0b0-0e6c-11eb-0b47-7569fd9dc381
html"It is often best to include the constraint boundary in the feasible set, to avoid problems with the boundary producing better values"

# ╔═╡ e4d7c530-0e84-11eb-0af7-3ddba7d4dbdc
md"Prefer = over > or <"

# ╔═╡ 50aacdc0-0e6c-11eb-303e-431247210cf0
md"## Critical Points"

# ╔═╡ 5bbd94e2-0e6c-11eb-0d21-adb162aed886
html"Is hard to find a global minimum, then the best we can do is check for local minimum"

# ╔═╡ 867f0420-0e6c-11eb-1d76-531be3b6099c
md"Check that \$\$x^*\$\$ is a local minimum using the definition of limit and δ"

# ╔═╡ c96924a0-0e6c-11eb-100b-397ed6e204bf
md"There are strong local minimizers and weak local minimizers, weak have other values disctint to x* that produces the same result (Equivalent solutions)"

# ╔═╡ 1188e362-0e6d-11eb-3bc0-73d959cb2639
md"Having a zero derivative is a *necessary condition* for a local minimum, but is not a *sufficient condition*"

# ╔═╡ 868d6690-0e6d-11eb-269d-6f2adfb65ad3
md"## Conditions for Local Minima"

# ╔═╡ 93075ed0-0e6d-11eb-326a-555d59c942fa
md"### Univariate"

# ╔═╡ a6b92df0-0e6d-11eb-1367-459fb4fdcd04
html"Strong local minima if the local derivative is zero and the second derivative is positive"

# ╔═╡ c5934800-0e6d-11eb-274b-018627af8a94
md"\$\$f'(x)=0\$\$\$\$f''(x)>0\$\$"

# ╔═╡ 6ae19520-0e85-11eb-16fd-61d9c7c97fcb
html"A point CAN also be at a local minimum if:"

# ╔═╡ 700c84fe-0e85-11eb-190d-510c27002410
md"\$\$f'(x)=0\$\$ *first-order necessary condition (FONC)*"

# ╔═╡ 3cd15390-0e86-11eb-1c9f-b341030c6bb6
md"\$\$f''(x)≥0\$\$ *second-order necessary condition (SONC)*"

# ╔═╡ 755e414e-0e86-11eb-30fb-3750e2ac5ce6
md"### Multivariate"

# ╔═╡ 848446c0-0e86-11eb-15b7-fdd2a0b0c289
md"Necessary conditions for x to be at a local minimum of *f*"

# ╔═╡ a09a10b0-0e86-11eb-388f-7fc8e814db44
md"\$\$∇f(x)=0\$\$, (FONC)"

# ╔═╡ c9fc1700-0e86-11eb-3c05-2786f7440db0
md"\$\$∇^2f(x)\$\$ is positive semidefinite (SONC)"

# ╔═╡ e1305620-0e86-11eb-2b2f-975ccf60ed59
md"A matrix A is positive definite if \$\$x^TAx > 0\\ for\\ ∀ x ≠ 0\$\$"

# ╔═╡ c3196f92-0e87-11eb-3202-0b4a739eb139
md"FONC and SONC are not sufficient for optimality."

# ╔═╡ e2878c40-0e87-11eb-17b1-679f210a7a32
md"For unconstrained optimization of a twice-differentiable function, a point is guaranteed to be at a strong local minimum if the FONC is satisfied and \$\$∇^2f(x)\$\$ is positive definite. These conditions are collectively kwown as *second-order sufficient condition (SOSC)*"

# ╔═╡ 30af5470-0e88-11eb-2c4e-fd57e0780753
md"## Contour Plots"

# ╔═╡ 7a365ca0-0e8e-11eb-090c-8f9457612a7d
begin
	x = range(-0.5, stop=0.5, length=200)
	y = x
end;

# ╔═╡ 26912ce0-0e90-11eb-390b-6f45ba82afbb
md"Contour plots"

# ╔═╡ 8d3a3550-0e8a-11eb-3d1c-df304edd4b1f
begin
	f1(x, y) = x*x - y*y
	surf1 = surface(x, y, f1, camera=(30, 50), c=cgrad([:yellow, :green, :blue]))
end

# ╔═╡ 13a3e640-0e90-11eb-08b1-dd55a22f3981
cont1 = contour(x, y, f1, c=cgrad([:yellow, :green, :blue]))

# ╔═╡ 34b72770-0e90-11eb-2ec7-b32f19fa4d2e
md"Banana function"

# ╔═╡ 3c951440-0e8e-11eb-2d43-f565fdb0f155
begin
	f2(x, y) = (1-x)^2 + 5(y - x^2)^2
	contour(x, y, f2, c=cgrad([:red, :green, :blue]))
end

# ╔═╡ Cell order:
# ╟─64f1e390-0e62-11eb-17e1-efb3f2d998f2
# ╟─b24bd470-0e62-11eb-1629-052ecfb22b47
# ╟─398daf30-0e63-11eb-1524-5f84a5e3fbdc
# ╟─5bf4e7a0-0e63-11eb-3f3d-ddeaf5a86f17
# ╟─b08e0260-0e63-11eb-3f16-97a6cbbf5297
# ╟─c81654f0-0e63-11eb-3817-2d7fadabf44b
# ╟─d879e370-0e63-11eb-3f46-89866807d8f1
# ╟─598a33c0-0e64-11eb-3b03-0df474b87d4d
# ╟─7d004420-0e64-11eb-3612-875d350e9065
# ╟─4fc54c92-0e68-11eb-36dd-1d847281c994
# ╟─d7451ab0-0e68-11eb-3cd4-2320283e74e6
# ╟─1a1c9d90-0e69-11eb-2ee0-49d236ca10d7
# ╟─a14f5d20-0e69-11eb-06ad-9fa7e0d4dd98
# ╟─2c1bb3a0-0e69-11eb-227c-176bf3d49f2d
# ╟─29cda4f0-0e69-11eb-3c9c-df53831b702f
# ╟─d0da3b50-0e69-11eb-2153-472942520fc4
# ╟─0fd48400-0e6a-11eb-0640-512e799bc4da
# ╟─d10976c0-0e6b-11eb-2ff2-69bed9537e9b
# ╟─11676e20-0e6c-11eb-3916-edbdeedf37bc
# ╟─1d2ff0b0-0e6c-11eb-0b47-7569fd9dc381
# ╟─e4d7c530-0e84-11eb-0af7-3ddba7d4dbdc
# ╟─50aacdc0-0e6c-11eb-303e-431247210cf0
# ╟─5bbd94e2-0e6c-11eb-0d21-adb162aed886
# ╟─867f0420-0e6c-11eb-1d76-531be3b6099c
# ╟─c96924a0-0e6c-11eb-100b-397ed6e204bf
# ╟─1188e362-0e6d-11eb-3bc0-73d959cb2639
# ╟─868d6690-0e6d-11eb-269d-6f2adfb65ad3
# ╟─93075ed0-0e6d-11eb-326a-555d59c942fa
# ╟─a6b92df0-0e6d-11eb-1367-459fb4fdcd04
# ╟─c5934800-0e6d-11eb-274b-018627af8a94
# ╟─6ae19520-0e85-11eb-16fd-61d9c7c97fcb
# ╟─700c84fe-0e85-11eb-190d-510c27002410
# ╟─3cd15390-0e86-11eb-1c9f-b341030c6bb6
# ╟─755e414e-0e86-11eb-30fb-3750e2ac5ce6
# ╟─848446c0-0e86-11eb-15b7-fdd2a0b0c289
# ╟─a09a10b0-0e86-11eb-388f-7fc8e814db44
# ╟─c9fc1700-0e86-11eb-3c05-2786f7440db0
# ╟─e1305620-0e86-11eb-2b2f-975ccf60ed59
# ╟─c3196f92-0e87-11eb-3202-0b4a739eb139
# ╟─e2878c40-0e87-11eb-17b1-679f210a7a32
# ╟─30af5470-0e88-11eb-2c4e-fd57e0780753
# ╠═6f7ef8c2-0e8a-11eb-2095-674a0e527503
# ╟─7a365ca0-0e8e-11eb-090c-8f9457612a7d
# ╟─26912ce0-0e90-11eb-390b-6f45ba82afbb
# ╟─8d3a3550-0e8a-11eb-3d1c-df304edd4b1f
# ╟─13a3e640-0e90-11eb-08b1-dd55a22f3981
# ╟─34b72770-0e90-11eb-2ec7-b32f19fa4d2e
# ╟─3c951440-0e8e-11eb-2d43-f565fdb0f155
