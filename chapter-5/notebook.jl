### A Pluto.jl notebook ###
# v0.12.4

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 839a6840-13f2-11eb-3ee7-3b513eb25cac
using Calculus

# ╔═╡ 3a625020-13fc-11eb-3cbb-95bafc7c95f9
using Base.MathConstants

# ╔═╡ febfa170-13fc-11eb-3291-f9bdc4e9f971
using Plots

# ╔═╡ dd193610-13fe-11eb-1724-77e60a1eb306
using PlutoUI

# ╔═╡ 49266a30-1422-11eb-1d84-9d24a1b81e04
using LinearAlgebra

# ╔═╡ 8bd329d0-13f1-11eb-0e83-25a8a9620553
md"# First-Order Methods"

# ╔═╡ 68f68ae0-1510-11eb-0bb6-bb84b68d18e6
md"### Spoiler alert: Some methods are trash!"

# ╔═╡ a97881d0-150c-11eb-3391-b52524901b4c
begin
	fₙ(x, y)::Number = (1-x)^2 + 100(y - x^2)^2
	fₙ(x̄)::Number = fₙ(x̄...)
	∇fₙ(x̄) = Calculus.gradient(fₙ, x̄)
end;

# ╔═╡ c71545d0-154c-11eb-38d2-ebdde946ae88
begin
	xₙ = range(-1, 3, length=200)
	yₙ = range(-1, 2, length=200)
end;

# ╔═╡ fbe594ce-150c-11eb-3135-fb523622c0cd
@bind nₙ Slider(1:200, default=12, show_value=true)

# ╔═╡ fbc03280-150c-11eb-3400-cf19d15006ba
@bind xₙᵢ Slider(-1.5:.01:1.5, default=-0.77, show_value=true)

# ╔═╡ fb16ac10-150c-11eb-23af-816bfbad1ad9
@bind yₙᵢ Slider(-.5:.01:1.5, default=0, show_value=true)

# ╔═╡ 9cf0b980-13f1-11eb-2274-6bc775d469c8
md"## Gradient Descent"

# ╔═╡ cb1de700-13f2-11eb-2ace-afc2e372b013
md"We can choose the direction of steepest descent as the descent direction d, and this is the opposite of the gradient."

# ╔═╡ ea5c8f40-13f2-11eb-22b5-61728c0e4388
md"\$\$d^{(k)}=-\frac{g^{(k)}}{||g^{(k)}||}\$\$"

# ╔═╡ 290a8cb0-13f3-11eb-2069-4f8e02d036a8
md"It is proven (proof left to the reader as excercise) that each direction is orthogonal to the previous."

# ╔═╡ bcfb7970-13f3-11eb-3e89-d7e04bfc65ad
function fibonacci_search(f::Function, a::Number, b::Number, n::Integer; ϵ=0.01)::Number
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

# ╔═╡ 58ee91f0-13f4-11eb-1d27-31633b18d5ee
function bracket_minimum(f::Function, x=0.0; s=1e-12, k=2.0)
	try
		a, ya = x, f(x)
		global b, yb = x+s, f(x+s)
		if yb > ya #If we are not going downhill
			c, yc = x-s, f(x-s)
			s = -s
		else
			c, yc = x+k*s, f(x+k*s)	
		end
		if ya > yb < yc # Definition of unimodal
			return (a, c)
		end
		s = s*k
	catch e
		println(e)
		return (a, c)
	end
	return bracket_minimum(f, b; s=s, k=k)
end

# ╔═╡ 246f8b50-13f4-11eb-2e29-977facb5c488
minimize(f::Function, a::Number, b::Number)::Number = fibonacci_search(f, a, b, 5)

# ╔═╡ 33062200-13f4-11eb-21f1-1ffa95638977
function optimize_line_search(f, x, d)::Float64
	objective(α::Number)::Number = f(x + α*d)
	a, b = bracket_minimum(objective; k=1.05)
	α = minimize(objective, a, b)
	α
end

# ╔═╡ d330cce2-13f8-11eb-2d7b-eb0338c3d1b8
function gradient_descent_mixed(f, ∇f, x, k)::Array{Tuple, 1}
	points::Array{Tuple} = Array{Tuple}(undef, k+1)
	points[1] = tuple(x...)
	for i in 1:k
		d = -∇f(x)
		d = d/sqrt(sum(2 .^ d))
		α = optimize_line_search(f, x, d)
		x = x + α*d
		points[i+1] = tuple(x...)
	end
	points
end

# ╔═╡ dcdfefe0-13f9-11eb-337a-2d8a80e4a5e7
begin
	f₁(x, y)::Number = (1-x)^2 + 5(y - x^2)^2
	f₁(x̄)::Number = f₁(x̄...)
end

# ╔═╡ ecdf9a32-13fe-11eb-3ff9-5166ac683584
@bind n₁ Slider(1:50, default=20, show_value=true)

# ╔═╡ 6dc753e0-13ff-11eb-01ae-7d4477726252
@bind x₁ᵢ Slider(-2.5:.01:2.5, default=-0.96, show_value=true)

# ╔═╡ 9841a800-13ff-11eb-1ef2-0b366a3d3b22
@bind y₁ᵢ Slider(-2.5:.01:2.5, default=1.69, show_value=true)

# ╔═╡ 01b03910-13fa-11eb-38fe-7ff1876faefc
begin
	x₁ = range(-3, stop=3, length=200)
	y₁ = x₁
	color₁ = cgrad([:grey, :blue])
	∇f₁(x̄) = Calculus.gradient(f₁, x̄)
	points₁ = [x for x in gradient_descent_mixed(f₁, ∇f₁, [x₁ᵢ, y₁ᵢ], n₁)]
	plot₁ = contour(x₁, y₁, f₁, c=color₁, levels=700 .^ (range(-.2,stop=1,length=14)))
	scatter!((1.0, 1.0), markersize=3, markerstrokewidth=0, c=:cyan, label="Optimal point")
	scatter!((x₁ᵢ, y₁ᵢ), markersize=2, c=:black, label="Initial point")
	plot!(points₁, c=:red, label="Descent")
	plot₁
end

# ╔═╡ 8107d7b0-140c-11eb-0194-67b53124e6a2
md"The optimal point is near _$(points₁[end])_"

# ╔═╡ 965666e0-140c-11eb-3d64-6fea80cb3edc
md"The optimized value is _$(f₁(points₁[end]))_"

# ╔═╡ 8116c470-140f-11eb-1b12-230e8c180dfe
abstract type DescentMethod end

# ╔═╡ 2fdb8310-1401-11eb-1529-ed74dd276380
begin
	struct GradientDescent <: DescentMethod
		α
	end
end

# ╔═╡ a8ae43ee-140f-11eb-2adf-4f4e1d0a8ef5
init!(M::GradientDescent, f, ∇f, x) = M

# ╔═╡ 7cc17412-140f-11eb-0411-4f7deaa61fae
function step!(M::GradientDescent, f, ∇f, x)::Array{Float64, 1}
	α, g = M.α, ∇f(x)
	return x - α*g
end

# ╔═╡ bd87d4c2-1410-11eb-3a2c-07be02f2d049
f₂ = f₁

# ╔═╡ df37f230-1410-11eb-0593-d5008a92eae1
@bind n₂ Slider(1:100, default=20, show_value=true)

# ╔═╡ dea91fb0-1410-11eb-2c52-836b5e46eb28
@bind x₂ᵢ Slider(-2.5:.01:2.5, default=-0.96, show_value=true)

# ╔═╡ de622ba0-1410-11eb-2910-b1e14a97878f
@bind y₂ᵢ Slider(-2.5:.01:2.5, default=1.69, show_value=true)

# ╔═╡ fa788512-1400-11eb-286c-85633d52af21
md"Gradient descent (alone) is not miraculous 😢"

# ╔═╡ d1732fa0-13fe-11eb-39f0-670f5eaa94ec
md"## Conjugate Gradient"

# ╔═╡ 4d0090c0-141a-11eb-3bbe-d9049235aca1
md"Gradient descent is 🗑️ in narrow valleys. This method uses a quadratic approximation of the local function to find optimal points."

# ╔═╡ 52541a70-141e-11eb-2096-c75cb6bfd792
md"Conjugate Gradient Descent is GUARANTEED to optimize a n-dimensional quadratic function in n steps."

# ╔═╡ 5d834dc0-141f-11eb-25eb-63933ff6ae4c
md"This method can be applied to nonquadratic functions. Smooth functions behave quadratic close to a local minimum (oh-oh)."

# ╔═╡ 0bbda8e0-1420-11eb-25bc-c34eaf97c1b6
md"This means that we need to get close to the minimum in order to obtain benefits. 🗑️"

# ╔═╡ 2418b4c0-1420-11eb-1d36-b5c012477a99
md"This method uses information from last gradients to improve:"

# ╔═╡ 75156ee0-1420-11eb-21ba-71bbcff2a1ed
md"\$\$d^{(k+1)}=-g^{(k+1)} + β^{(k)}d^{(k)}\$\$"

# ╔═╡ 9a938440-1420-11eb-19f5-91f76c2eab12
md"**Fletcher-Reeves:**\$\$β^{(k)}=\frac{(g^{(k)})^2}{(g^{(k-1)})^2}\$\$"

# ╔═╡ 0d96b92e-1421-11eb-11f0-6dc604eeb10a
md"**Polak-Ribiere:**\$\$β^{(k)}=\frac{g^{(k)T}(g^{(k)}-g^{(k-1)})}{(g^{(k-1)})^2}\$\$"

# ╔═╡ 40b8c7e0-1421-11eb-3e76-5db9f1885668
md"👀 For guaranteed convergence on the Polak-Ribiere method we need to reset β to 0 if it goes negative."

# ╔═╡ 744aaf60-1421-11eb-0dbd-1fd4f4775c8d
begin
	mutable struct ConjugateGradientDescent <: DescentMethod
		d::Array{Number, 1}
		g::Array{Number, 1}
	end
	function ConjugateGradientDescent()
		return ConjugateGradientDescent(Number[], Number[])
	end
end

# ╔═╡ 08462370-1422-11eb-1ea0-ef8e58d85970
function init!(M::ConjugateGradientDescent, f, ∇f, x)
	M.g = ∇f(x)
	M.d = -M.g
	return M
end

# ╔═╡ 98c33ba0-1421-11eb-23ce-a76943bc3a76
function step!(M::ConjugateGradientDescent, f, ∇f, x)::Array{Float64, 1}
	d, g = M.d, M.g
	g′ = ∇f(x)
	β = (g′ ⋅ (g′ - g))/(g ⋅ g)
	β = β*(β>0)
	d′ = -g′ + β*d
	α = optimize_line_search(f, x, d′)
	x′ = x + α*d′
	M.d, M.g = d′, g′
	return x′
end

# ╔═╡ 8accdfc0-1425-11eb-18fb-29f87b39e006
f₃ = f₁

# ╔═╡ 6bcaf260-1425-11eb-1b6a-b14640cd9d64
@bind n₃ Slider(1:10, default=5, show_value=true)

# ╔═╡ 7163e650-1425-11eb-2bb9-6b6b8c3bf71a
@bind x₃ᵢ Slider(-2.5:.01:2.5, default=-0.96, show_value=true)

# ╔═╡ 7ac3cd50-1425-11eb-1a93-adaeb3dac3f4
@bind y₃ᵢ Slider(-2.5:.01:2.5, default=1.69, show_value=true)

# ╔═╡ 7d0d8450-1427-11eb-373d-fd77eca043b3
md"## Momentum"

# ╔═╡ b1426290-1427-11eb-3ff3-4d936d73df78
md"The idea is to simulate a force (that updates velocity) to avoid slow movement in a nearly flat surface (gradients with small magnitude will fail miserably)."

# ╔═╡ d250f360-1428-11eb-3067-7bf3b10eeb95
md"\$\$v^{(k+1)}=βv^{(k)}-αg^{(k)}\$\$"

# ╔═╡ bf228510-1428-11eb-35e0-ef99d695c0de
md"\$\$x^{(k+1)}=x^{(k)}+v^{(k+1)}\$\$"

# ╔═╡ ff655350-1428-11eb-2baf-079bacea6696
begin
	mutable struct Momentum <: DescentMethod
		α::Number
		β::Number
		v::Array{Number, 1}
	end
	Momentum(α, β) = Momentum(α, β, Number[])
end

# ╔═╡ b9c8eea0-1429-11eb-315c-316fc17a30a1
function init!(M::Momentum, f, ∇f, x)
	M.v = zeros(length(x))
end

# ╔═╡ 0c382480-142a-11eb-026a-77a06e8eb9d0
function step!(M::Momentum, f, ∇f, x)::Array{Float64, 1}
	α, β, v, g = M.α, M.β, M.v, ∇f(x)
	v[:] = β*v - α*g #Update velocity
	return x + v
end

# ╔═╡ 87a82de0-142a-11eb-3c40-3f50754cc0fe
begin
	f₄(x, y)::Number = (1-x)^2 + 100(y - x^2)^2
	f₄(x̄)::Number = f₄(x̄...)
end

# ╔═╡ 1b6bfab0-142c-11eb-1d7e-995795b8d26f
@bind n₄ Slider(1:40, default=21, show_value=true)

# ╔═╡ 2279db60-142c-11eb-1d5d-f16fbfdf7451
@bind x₄ᵢ Slider(-1.5:.01:1.5, default=-0.58, show_value=true)

# ╔═╡ 23190190-142c-11eb-2cc2-614ee78d3e8a
@bind y₄ᵢ Slider(0:.015:1.5, default=0.36, show_value=true)

# ╔═╡ 349bc640-142d-11eb-3baa-e7dc46b4b8a1
md"It's like the point is falling straight to the minimum 😲"

# ╔═╡ 76d33890-142d-11eb-253a-491b80114239
md"But... it might go too fast"

# ╔═╡ a38f6660-142d-11eb-21a2-8529e11f8acb
md"## Nesterov Momentum"

# ╔═╡ 9b6ad3a0-142f-11eb-3482-ed377d4bb2a8
md"If it goes too fast try predicting how fast it will go the next iteration..."

# ╔═╡ 72a17620-1477-11eb-166a-fd7dac2a01bd
md"\$\$v^{(k+1)}=βv^{(k)}-α∇f(x^{(k)}+βv^{(k)})\$\$"

# ╔═╡ 93479da0-1477-11eb-0c7b-a9446b9dd6d3
begin
	mutable struct NesterovMomentum <: DescentMethod
		α::Number
		β::Number
		v::Array{Number, 1}
	end
	function NesterovMomentum(α::Number, β::Number)
		NesterovMomentum(α, β, Number[])
	end
	function init!(M::NesterovMomentum, f, ∇f, x)
		M.v = zeros(length(x))
		return M
	end
	function step!(M::NesterovMomentum, f, ∇f, x)::Array{Float64, 1}
		α, β, v = M.α, M.β, M.v
		next_g = β*v
		v[:] = next_g - α*∇f(x + next_g)
		return x + v
	end
end

# ╔═╡ f8d6b0f0-1479-11eb-1c27-4b98bb05bd34
f₅ = f₄

# ╔═╡ fd7f9f92-1479-11eb-11de-c9b923f3fe06
@bind n₅ Slider(1:1000, default=42, show_value=true)

# ╔═╡ 0424d360-147a-11eb-2b4a-99845e4f9e5a
@bind x₅ᵢ Slider(-1.5:.01:1.5, default=-0.77, show_value=true)

# ╔═╡ 0bc30d30-147a-11eb-2480-53595b44dd79
@bind y₅ᵢ Slider(-1.5:.01:1.5, default=1.42, show_value=true)

# ╔═╡ db2fd4d0-147b-11eb-35a8-770858111cba
md"It's still going too fast..."

# ╔═╡ 0c426470-147c-11eb-1e91-f131c1a1da0b
md"## Adagrad"

# ╔═╡ ad146fb0-147c-11eb-3756-33ad44898b84
md"Adaptative subgradient method"

# ╔═╡ cbecf130-147e-11eb-0748-0d4f4af1775c
md"This method updates learning rate for each component."

# ╔═╡ e572c170-147e-11eb-0931-c5485d3e4203
md"\$\$x\_i ^{(k+1)}=x\_i ^{(k)} - \frac{α}{ϵ+\sqrt{s\_i ^{(k)}}}g\_i ^{(k)}\$\$"

# ╔═╡ 349f9d90-147f-11eb-3868-8b295d0f4e59
md"\$\$ s\_i ^{(k)} = \sum\_{j=1}^{k} (g\_i ^{(j)})^2 \$\$"

# ╔═╡ 49aaba22-148a-11eb-3782-cd3dcb9a53b1
md"ϵ is a small value (1e-8) to prevent division by zero."

# ╔═╡ 6a12c730-148a-11eb-3fa7-63cf835a531d
md"α is usually set to .01, because of all the operations it doesn't matter too much"

# ╔═╡ 7d457b40-148a-11eb-101a-15c60623e966
md"Problem: As we can see, the components of S grow over time, making the learning rate decrease (and become infinitesimally small even before convergence)."

# ╔═╡ af9daa40-148a-11eb-0b19-452c71a11b26
begin
	mutable struct Adagrad <: DescentMethod
		α::Number
		ϵ::Number
		s::Array{Float64, 1}
	end
	Adagrad(α::Number, ϵ::Number) = Adagrad(α, ϵ, Float64[])
	Adagrad(α::Number) = Adagrad(α, 1e-8)
	function init!(M::Adagrad, f, ∇f, x)
		M.s = zeros(length(x))
		return M
	end
	function step!(M::Adagrad, f, ∇f, x)::Array{Float64, 1}
		α, ϵ, s, g = M.α, M.ϵ, M.s, ∇f(x)
		s[:] += g .* g
		return x - α*g ./ (sqrt.(s) .+ ϵ)
	end
end

# ╔═╡ e6fe3080-148b-11eb-28ba-8f5a1b03e5cb
f₆ = f₄

# ╔═╡ f62399b0-148b-11eb-2078-c97839d84aaf
@bind n₆ Slider(1:100, default=22, show_value=true)

# ╔═╡ fb8c07c0-148b-11eb-3291-f985249cb0ae
@bind x₆ᵢ Slider(-1.5:.01:1.5, default=-0.77, show_value=true)

# ╔═╡ 00b019d0-148c-11eb-0c2b-bb5287624389
@bind y₆ᵢ Slider(-.5:.01:1.5, default=0, show_value=true)

# ╔═╡ cf5ede10-148c-11eb-2fe9-ddedbf608aad
md"Really slow convergence..."

# ╔═╡ c6ffcfd2-148d-11eb-3486-1fd4742afaa0
md"## RMSProp"

# ╔═╡ e83f7830-148d-11eb-0119-770094c7ad55
md"It comes to the rescue of Adagrad, solving the problem of decreasing learning rate."

# ╔═╡ 40e2a980-148e-11eb-1076-5767cba6fff6
md"⊙ = \odot (element wise multiplication)"

# ╔═╡ fc7607ae-148d-11eb-25fa-59a3d2c4c722
md"\$\$ŝ^{(k+1)}=γŝ^{(k)}+(1-γ)(g^{(k)}⊙g^{(k)})\$\$"

# ╔═╡ e5daad30-148d-11eb-224a-1bb3c3d5b20a
md"γ is tipically close to 0.9"

# ╔═╡ 6595a6b0-148e-11eb-3a6a-8dfdb2ef7933
md"\$\$RMS(g\_i)=ŝ\_i ^{(k+1)}\$\$"

# ╔═╡ 65ad4d5e-148e-11eb-1121-c139c1000455
begin
	mutable struct RMSProp <: DescentMethod
		α::Number
		γ::Number
		ϵ::Number
		ŝ::Array{Float64, 1}
	end
	RMSProp(α::Number, γ::Number, ϵ::Number) = RMSProp(α, γ, ϵ, Float64[])
	RMSProp(α::Number, γ::Number) = RMSProp(α, γ, 1e-8)
	RMSProp(α::Number) = RMSProp(α, .9)
	function init!(M::RMSProp, f, ∇f, x)
		M.ŝ = zeros(length(x))
		return M
	end
	function step!(M::RMSProp, f, ∇f, x)
		α, γ, ϵ, ŝ, g = M.α, M.γ, M.ϵ, M.ŝ, ∇f(x)
		ŝ[:] = γ*ŝ + (1-γ)*(g .* g)
		return x - α*g ./ (sqrt.(ŝ) .+ ϵ)
	end
end

# ╔═╡ f373bf20-148f-11eb-178b-ddd8ed8df869
f₇ = f₄

# ╔═╡ f71a8000-148f-11eb-08bb-911cfaa79535
@bind n₇ Slider(1:300, default=22, show_value=true)

# ╔═╡ f87c8100-148f-11eb-3938-2dd7affdd0a4
@bind x₇ᵢ Slider(-1.5:.01:1.5, default=-0.77, show_value=true)

# ╔═╡ f8981f50-148f-11eb-1b9b-3dfd2d711494
@bind y₇ᵢ Slider(-.5:.01:1.5, default=0, show_value=true)

# ╔═╡ 0f715b40-1492-11eb-0c2f-4d5ff35414b6
md"## Adadelta"

# ╔═╡ 17b8a650-1492-11eb-3fcb-9dce6ad8e093
md"Removes the learning rate parameter entirely (the same authors of RMSProp did this) 😲😲"

# ╔═╡ 0ceaa960-14df-11eb-3e04-613396d49c39
md"\$\$x\_i ^{(k+1)}=x\_i ^{(k)} - \frac{RMS(Δx\_i)}{ϵ+RMS(g\_i)}\$\$"

# ╔═╡ 35045130-14df-11eb-227e-af37a738cf78
begin
	mutable struct Adadelta <: DescentMethod
		γs::Number
		γx::Number
		ϵ::Number
		ŝ::Array{Float64, 1}
		u::Array{Float64, 1}
	end
	Adadelta(γs::Number, γx::Number, ϵ::Number) = Adadelta(γs, γx, ϵ, Float64[], Float64[])
	Adadelta(γs::Number, γx::Number) = Adadelta(γs, γx, 1e-2)
	function init!(M::Adadelta, f, ∇f, x)
		M.ŝ = zeros(length(x))
		M.u = zeros(length(x))
		return M
	end
	function step!(M::Adadelta, f, ∇f, x)::Array{Float64, 1}
		γs, γx, ϵ, ŝ, u, g = M.γs, M.γx, M.ϵ, M.ŝ, M.u, ∇f(x)
		ŝ[:] = γs*ŝ + (1-γs)*(g .* g)
		Δx = -(sqrt.(u) .+ ϵ) ./ (sqrt.(ŝ) .+ ϵ) .* g
		u[:] = γx*u + (1-γx)*(Δx .* Δx)
		return x + Δx
	end
end

# ╔═╡ 8ec902a0-14e0-11eb-0cba-f1917748a92f
f₈ = f₄

# ╔═╡ b429cc00-14e0-11eb-27ba-b76917406f92
@bind n₈ Slider(1:1000, default=102, show_value=true)

# ╔═╡ b3fc0540-14e0-11eb-1975-47351cf57db3
@bind x₈ᵢ Slider(-1.5:.01:1.5, default=-0.77, show_value=true)

# ╔═╡ b37dfba0-14e0-11eb-161e-812bbfff5a45
@bind y₈ᵢ Slider(-.5:.01:1.5, default=0, show_value=true)

# ╔═╡ 22447bd0-14e2-11eb-2385-afe72e00c514
md"## Adam"

# ╔═╡ 8d785802-14e6-11eb-2768-7d57bb0c552d
md"Adaptative moment estimation method"

# ╔═╡ 80d5c920-14e6-11eb-0b2b-aba1f51d4c40
md"Stores exponentially decaying squared gradient (RMSProp and Adadelta), but also decaying gradient like momentum."

# ╔═╡ f4919fb0-14e6-11eb-37da-7f8b001f912f
md"Initializing the gradient and squared gradient to zero introduces a bias. (Good defaults are α=.001, γᵥ=.9, γₛ=.999 and ϵ=1e-8)"

# ╔═╡ 53fc59e0-14e7-11eb-1e42-c93841a25d8b
md"The equations for Adam are:"

# ╔═╡ 751b81f0-14e7-11eb-30dd-d914bd2aae92
md"- Biased decaying momentum: \$\$ v^{(k+1)} = γ\_v v^{(k)} + (1-γ\_v)g^{(k)}\$\$"

# ╔═╡ 75af8490-14e7-11eb-266b-2f84b10c59b6
md"- Biased decaying sq. gradient: \$\$s^{(k+1)} = γ\_ss^{(k)}+(1-γ\_s)(g^{(k)}⊙g^{(k)})\$\$"

# ╔═╡ 75cb9810-14e7-11eb-0a01-15c658df3cd6
md"- Corrected decaying momentum: \$\$v̂^{(k+1)}=v^{(k+1)}/(1-γ\_v ^k)\$\$"

# ╔═╡ 75e93230-14e7-11eb-1436-155363e20db8
md"- Corrected decaying sq. gradient: \$\$ŝ^{(k+1)} = s^{(k+1)}/(1-γ\_s ^k)\$\$"

# ╔═╡ a6b66980-14e9-11eb-1f40-4b6f60bd56db
md"- Next iterate: \$\$x^{(k+1)}=x^{(k)}-αv̂^{(k+1)}/(ϵ+\sqrt{ŝ^{(k+1)}})\$\$"

# ╔═╡ c9666ac2-14e9-11eb-3f14-cbf7adf802d9
begin
	mutable struct Adam <: DescentMethod
		α::Float64
		γᵥ::Float64
		γₛ::Float64
		ϵ::Float64
		k::Integer
		v::Array{Float64, 1}
		s::Array{Float64, 1}
	end
	Adam(α::Number, γᵥ::Number, γₛ::Number, ϵ::Number) = Adam(α, γᵥ, γₛ, ϵ, 0, Float64[], Float64[])
	Adam(α::Number, γᵥ::Number, γₛ::Number) = Adam(α, γᵥ, γₛ, 1e-8)
	Adam(α::Number, γᵥ::Number) = Adam(α, γᵥ, .999)
	Adam(α::Number) = Adam(α, .9)
	
	function init!(M::Adam, f, ∇f, x)
		M.k = 0
		M.v = zeros(length(x))
		M.s = zeros(length(x))
		return M
	end
	
	function step!(M::Adam, f, ∇f, x)
		α, γᵥ, γₛ, ϵ, k = M.α, M.γᵥ, M.γₛ, M.ϵ, M.k
		s, v, g = M.s, M.v, ∇f(x)
		v[:] = γᵥ*v + (1-γᵥ)*g
		s[:] = γₛ*s + (1-γₛ)*(g .* g)
		M.k = k += 1 #Updates k at the same time
		v̂ = v/(1-γᵥ^k)
		ŝ = s/(1-γₛ^k)
		return x - (α*v̂)./(sqrt.(ŝ) .+ ϵ)
	end
end

# ╔═╡ cf904cb0-14ec-11eb-3cec-af99445c4bb3
f₉ = f₄

# ╔═╡ 3955ac20-14ee-11eb-314d-ab74464eb64b
@bind n₉ Slider(1:500, default=222, show_value=true)

# ╔═╡ 393c57c0-14ee-11eb-09fd-a51eb1d06292
@bind x₉ᵢ Slider(-1.5:.01:1.5, default=-0.77, show_value=true)

# ╔═╡ 39257460-14ee-11eb-365d-45eca91e680e
@bind y₉ᵢ Slider(-.5:.01:1.5, default=0, show_value=true)

# ╔═╡ 94087c30-14f1-11eb-2df8-27240ebdd2ae
md"## Hypergradient Descent"

# ╔═╡ 3b6a5e82-14f2-11eb-1962-d19f98382b7a
md"This methods are too sensitive to the learning rate... let's optimize it first"

# ╔═╡ 5fd33350-14f2-11eb-0548-7db92e9c8551
md"We applied gradient descent to the learning reate 🤯🤯"

# ╔═╡ 59a20190-14f3-11eb-3b9b-7d4284e3f759
begin
	mutable struct HyperGradientDescent <: DescentMethod
		α₀::Float64 #Initial learning rate
		μ::Float64 #Inception learning rate
		α::Float64 #Current learning rate
		g_prev::Array{Float64, 1}
	end
	
	HyperGradientDescent(α₀::Number, μ::Number) = HyperGradientDescent(α₀, μ, α₀, Float64[])
	
	function init!(M::HyperGradientDescent, f, ∇f, x)
		M.α = M.α₀
		M.g_prev = zeros(length(x))
		return M
	end
	
	function step!(M::HyperGradientDescent, f, ∇f, x)::Array{Float64, 1}
		α, μ, g, g_prev = M.α, M.μ, ∇f(x), M.g_prev
		α = α + μ*(g ⋅ g_prev)
		M.g_prev, M.α = g, α
		return x - α*g
	end
end

# ╔═╡ b78278a0-14fb-11eb-0c8b-dd1d29967a94
f₁₀ = f₄

# ╔═╡ bb6aa8f0-14f3-11eb-2d69-bbc0e5d83f1e
@bind n₁₀ Slider(1:100, default=22, show_value=true)

# ╔═╡ 365cf2a0-14fb-11eb-1044-779ff3953853
@bind x₁₀ᵢ Slider(-1.5:.01:1.5, default=-0.77, show_value=true)

# ╔═╡ 361e3bf0-14fb-11eb-250b-97be54cd99f7
@bind y₁₀ᵢ Slider(-.5:.01:1.5, default=0, show_value=true)

# ╔═╡ 12723860-1504-11eb-0e56-c9d2ed88cd48
md"Let's do the same to Nesterov momentum"

# ╔═╡ 583b4f30-14ff-11eb-251b-ed6cb326bb44
begin
	mutable struct HyperNesterovMomentum <: DescentMethod
		α₀::Float64 #Initial learning rate
		μ::Float64 #Inception learning rate
		β::Float64
		v::Array{Float64, 1}
		α::Float64
		g_prev::Array{Float64, 1}
	end
	
	HyperNesterovMomentum(α₀::Number, μ::Number, β::Number) = HyperNesterovMomentum(α₀, μ, β, Float64[], α₀, Float64[])
	
	function init!(M::HyperNesterovMomentum, f, ∇f, x)
		M.α = M.α₀
		M.v = zeros(length(x))
		M.g_prev = zeros(length(x))
		return M
	end
	
	function step!(M::HyperNesterovMomentum, f, ∇f, x)::Array{Float64, 1}
		α, β, v = M.α, M.β, M.v
		g, μ, g_prev = ∇f(x), M.μ, M.g_prev
		next_g = β*v
		α = α + μ*(g ⋅ (-g_prev - β*v))
		v[:] = β*v + g
		M.α, M.g_prev = α, g_prev
		return x - α*(g + β*v)
	end
end

# ╔═╡ 0d9a9980-1410-11eb-003d-8dbb0dff6ff4
function optimize!(M::DescentMethod, f, ∇f, x; k=30)::Array{Tuple}
	init!(M, f, ∇f, x)
	points::Array{Tuple} = Array{Tuple}(undef, k+1)
	for i in 1:k
		points[i] = tuple(x...)
		x = step!(M, f, ∇f, x)
	end
	points[end] = tuple(x...)
	points
end

# ╔═╡ 08a33cc0-1410-11eb-1e21-134f56f9a134
begin
	x₂ = range(-3, stop=3, length=200)
	y₂ = x₂
	color₂ = cgrad([:grey, :blue])
	∇f₂(x̄) = Calculus.gradient(f₂, x̄)
	points₂ = [x for x in optimize!(GradientDescent(.052), f₂, ∇f₂, [x₂ᵢ, y₂ᵢ], k=n₂)]
	contour(x₂, y₂, f₂, c=color₂, levels=700 .^ (range(-.2,stop=1,length=14)))
	scatter!((1.0, 1.0), markersize=3, markerstrokewidth=0, c=:cyan, label="Optimal point")
	scatter!((x₂ᵢ, y₂ᵢ), markersize=2, c=:black, label="Initial point")
	plot!(points₂, c=:red, label="Descent")
	scatter!(points₂, c=:green, markersize=1.5, markerstrokewidth=0, label="Path")
end

# ╔═╡ aec7c7a2-1411-11eb-0302-2b383519fc53
md"The optimal point is near _$(points₂[end])_ 👀 "

# ╔═╡ f2664400-1411-11eb-378e-dbf7d2ca2597
md"The optimized value is _$(f₂(points₂[end]))_ 😒🤢"

# ╔═╡ abfc14c0-1422-11eb-0746-2375b2a7f411
begin
	x₃ = range(-3, stop=3, length=200)
	y₃ = x₃
	color₃ = cgrad([:grey, :blue])
	∇f₃(x̄) = Calculus.gradient(f₃, x̄)
	points₃ = [x for x in optimize!(ConjugateGradientDescent(), f₃, ∇f₃, [x₃ᵢ, y₃ᵢ]; k=n₃)]
	contour(x₃, y₃, f₃, c=color₃, levels=700 .^ (range(-.2,stop=1,length=14)))
	scatter!((1.0, 1.0), markersize=3, markerstrokewidth=0, c=:cyan, label="Optimal point")
	scatter!((x₃ᵢ, y₃ᵢ), markersize=4, c=:black, label="Initial point")
	plot!(points₃, c=:red, label="Descent")
	scatter!(points₃, c=:green, markersize=2.5, markerstrokewidth=0, label="Path")
end

# ╔═╡ e8a44fc2-1425-11eb-1134-e7e1073a68c5
md"The optimal point is near _$(points₃[end])_ 🌟 "

# ╔═╡ 67b3aa80-1427-11eb-2483-51fb1ed66625
md"The optimized value is _$(f₃(points₃[end]))_ 🥳"

# ╔═╡ f6dc15e0-142b-11eb-30fb-718decba0cb3
begin
	x₄ = range(-1.5, stop=1.25, length=200)
	y₄ = range(-.45, stop=1.5, length=200)
	color₄ = cgrad([:yellow,:green, :blue, :red, :purple])
	∇f₄(x̄) = Calculus.gradient(f₄, x̄)
	points₄ = [x for x in optimize!(Momentum(.005, 1), f₄, ∇f₄, [x₄ᵢ, y₄ᵢ]; k=n₄)]
	contour(x₄, y₄, f₄, c=color₄, levels=2500 .^ (range(-.5,stop=1,length=25)), legend=false)
	scatter!((1.0, 1.0), markersize=3, markerstrokewidth=0, c=:cyan, label="Optimal point")
	scatter!((x₄ᵢ, y₄ᵢ), markersize=4, c=:black, label="Initial point")
	plot!(points₄, c=:red, label="Descent")
	scatter!(points₄, c=:green, markersize=2.5, markerstrokewidth=0, label="Path")
end

# ╔═╡ 09d749c0-142d-11eb-20d6-49c529dda237
md"The optimal point is near _$(points₄[end])_ 🤷‍♀️ "

# ╔═╡ 2f0e9220-142d-11eb-23f9-cd8664d7d7f7
md"The optimized value is _$(f₄(points₄[end]))_ 🤦‍♂️"

# ╔═╡ f1b79230-1479-11eb-22d6-c3540c7f6ae1
begin
	x₅ = range(-1.5, stop=1.25, length=200)
	y₅ = range(-.45, stop=1.5, length=200)
	color₅ = cgrad([:yellow,:green, :blue, :red, :purple])
	∇f₅(x̄) = Calculus.gradient(f₅, x̄)
	points₅ = [x for x in optimize!(NesterovMomentum(.0009, .95), f₅, ∇f₅, [x₅ᵢ, y₅ᵢ]; k=n₅)]
	contour(x₅, y₅, f₅, c=color₅, levels=2500 .^ (range(-.5,stop=1,length=25)), legend=false)
	scatter!((1.0, 1.0), markersize=3, markerstrokewidth=0, c=:cyan, label="Optimal point")
	scatter!((x₅ᵢ, y₅ᵢ), markersize=4, c=:black, label="Initial point")
	plot!(points₅, c=:red, label="Descent")
	scatter!(points₅, c=:green, markersize=1.5, markerstrokewidth=0, label="Path")
end

# ╔═╡ ce3736b0-147b-11eb-3acf-4917b0134727
md"The optimal point is near _$(points₅[end])_ 🤔 "

# ╔═╡ d5bd2d90-147b-11eb-3af9-7bfab1d3efe8
md"The optimized value is _$(f₅(points₅[end]))_ 🤔🤔"

# ╔═╡ 0660db82-148c-11eb-1823-9f03b67f13fa
begin
	x₆ = range(-1.5, stop=1.25, length=200)
	y₆ = range(-.45, stop=1.5, length=200)
	color₆ = cgrad([:yellow,:green, :blue, :red, :purple])
	∇f₆(x̄) = Calculus.gradient(f₆, x̄)
	points₆ = [x for x in optimize!(Adagrad(.5), f₆, ∇f₆, [x₆ᵢ, y₆ᵢ]; k=n₆)]
	contour(x₆, y₆, f₆, c=color₆, levels=2500 .^ (range(-.5,stop=1,length=25)), legend=false)
	scatter!((1.0, 1.0), markersize=3, markerstrokewidth=0, c=:cyan, label="Optimal point")
	scatter!((x₆ᵢ, y₆ᵢ), markersize=4, c=:black, label="Initial point")
	plot!(points₆, c=:red, label="Descent")
	scatter!(points₆, c=:green, markersize=1.5, markerstrokewidth=0, label="Path")
end

# ╔═╡ aa77ffa0-148c-11eb-0641-19ad5798aa8d
md"The optimal point is near _$(points₆[end])_ 🤣 "

# ╔═╡ afa20520-148c-11eb-0568-b163d9d760e5
md"The optimized value is _$(f₆(points₆[end]))_ 🐌🐌🐌"

# ╔═╡ e77893d0-148f-11eb-3caf-639d09e446f5
begin
	x₇ = range(-1.5, stop=1.5, length=200)
	y₇ = range(-.45, stop=1.5, length=200)
	color₇ = cgrad([:yellow,:green, :blue, :red, :purple])
	∇f₇(x̄) = Calculus.gradient(f₇, x̄)
	points₇ = [x for x in optimize!(RMSProp(.04, .999, 1), f₇, ∇f₇, [x₇ᵢ, y₇ᵢ]; k=n₇)]
	contour(x₇, y₇, f₇, c=color₇, levels=2500 .^ (range(-.5,stop=1,length=25)), legend=false)
	scatter!((1.0, 1.0), markersize=3, markerstrokewidth=0, c=:cyan, label="Optimal point")
	scatter!((x₇ᵢ, y₇ᵢ), markersize=4, c=:black, label="Initial point")
	plot!(points₇, c=:red, label="Descent")
	scatter!(points₇, c=:green, markersize=1, markerstrokewidth=0, label="Path")
end

# ╔═╡ 84527b80-14e0-11eb-0175-ddaa65c37cde
begin
	x₈ = range(-1.5, stop=1.5, length=200)
	y₈ = range(-.45, stop=1.5, length=200)
	color₈ = cgrad([:yellow,:green, :blue, :red, :purple])
	∇f₈(x̄) = Calculus.gradient(f₈, x̄)
	points₈ = [x for x in optimize!(Adadelta(.9, .9999, 3e-2), f₈, ∇f₈, [x₈ᵢ, y₈ᵢ]; k=n₈)]
	contour(x₈, y₈, f₈, c=color₈, levels=2500 .^ (range(-.5,stop=1,length=25)), legend=false)
	scatter!((1.0, 1.0), markersize=3, markerstrokewidth=0, c=:cyan, label="Optimal point")
	scatter!((x₈ᵢ, y₈ᵢ), markersize=4, c=:black, label="Initial point")
	plot!(points₈, c=:red, label="Descent")
	scatter!(points₈, c=:green, markersize=1, markerstrokewidth=0, label="Path")
end

# ╔═╡ ee0e6410-14ec-11eb-3aac-590c727498f8
begin
	x₉ = range(-1.5, stop=1.5, length=200)
	y₉ = range(-.45, stop=1.5, length=200)
	color₉ = cgrad([:yellow,:green, :blue, :red, :purple])
	∇f₉(x̄) = Calculus.gradient(f₉, x̄)
	points₉ = [x for x in optimize!(Adam(.3, .95, .99), f₉, ∇f₉, [x₉ᵢ, y₉ᵢ]; k=n₉)]
	contour(x₉, y₉, f₉, c=color₉, levels=2500 .^ (range(-.5,stop=1,length=25)), legend=false)
	scatter!((1.0, 1.0), markersize=3, markerstrokewidth=0, c=:cyan, label="Optimal point")
	scatter!((x₉ᵢ, y₉ᵢ), markersize=4, c=:black, label="Initial point")
	plot!(points₉, c=:red, label="Descent")
	scatter!(points₉, c=:green, markersize=1.5, markerstrokewidth=0, label="Path")
end

# ╔═╡ 3602c4b0-14fb-11eb-05ee-1fdf72a238a7
begin
	x₁₀ = range(-1.5, stop=1.5, length=200)
	y₁₀ = range(-.45, stop=1.5, length=200)
	color₁₀ = cgrad([:yellow,:green, :blue, :red, :purple])
	∇f₁₀(x̄) = Calculus.gradient(f₁₀, x̄)
	points₁₀ = [x for x in optimize!(HyperGradientDescent(.005, 1e-8), f₁₀, ∇f₁₀, [x₁₀ᵢ, y₁₀ᵢ]; k=n₁₀)]
	contour(x₁₀, y₁₀, f₁₀, c=color₁₀, levels=2500 .^ (range(-.5,stop=1,length=25)), legend=false)
	scatter!((1.0, 1.0), markersize=3, markerstrokewidth=0, c=:cyan, label="Optimal point")
	scatter!((x₁₀ᵢ, y₁₀ᵢ), markersize=4, c=:black, label="Initial point")
	plot!(points₁₀, c=:red, label="Descent")
	scatter!(points₁₀, c=:green, markersize=1.5, markerstrokewidth=0, label="Path")
end

# ╔═╡ 820ea300-1501-11eb-34aa-ef388563baa8
f₁₁ = f₄

# ╔═╡ 43917990-1501-11eb-02f1-45e5d8b91637
@bind n₁₁ Slider(1:200, default=22, show_value=true)

# ╔═╡ 4ad648c0-1501-11eb-3c64-0f6089cd7a4a
@bind x₁₁ᵢ Slider(-1.5:.01:1.5, default=-0.77, show_value=true)

# ╔═╡ 4e0f53b0-1501-11eb-0ece-15db048a27fd
@bind y₁₁ᵢ Slider(-.5:.01:1.5, default=0, show_value=true)

# ╔═╡ 4de66ef0-1501-11eb-3ffc-0d2721a515a0
begin
	x₁₁ = range(-1.5, stop=1.5, length=200)
	y₁₁ = range(-.45, stop=1.5, length=200)
	color₁₁ = cgrad([:yellow,:green, :blue, :red, :purple])
	∇f₁₁(x̄) = Calculus.gradient(f₁₁, x̄)
	points₁₁ = [x for x in optimize!(HyperNesterovMomentum(.5e-3, 3e-9, .97), f₁₁, ∇f₁₁, [x₁₁ᵢ, y₁₁ᵢ]; k=n₁₁)]
	contour(x₁₁, y₁₁, f₁₁, c=color₁₀, levels=2500 .^ (range(-.5,stop=1,length=25)), legend=false)
	scatter!((1.0, 1.0), markersize=3, markerstrokewidth=0, c=:cyan, label="Optimal point")
	scatter!((x₁₁ᵢ, y₁₁ᵢ), markersize=4, c=:black, label="Initial point")
	plot!(points₁₁, c=:red, label="Descent")
	scatter!(points₁₁, c=:green, markersize=1.5, markerstrokewidth=0, label="Path")
end

# ╔═╡ 7c1abdd0-1501-11eb-2d93-53d722c2d8dc
function plot_method!(M::DescentMethod, f, ∇f, x, k; curve=:red, path=:green)
	methodname = string(nameof(typeof(M)))
	points = [x for x in optimize!(M, f, ∇f, x; k=k)]
	plot!(points, c=curve, label="$(methodname)")
	scatter!(points, c=path, markersize=1, markerstrokewidht=0, label=nothing)
end

# ╔═╡ 4dc886b0-1501-11eb-0c20-cfbbb42f5f72
begin
	methods₁ = [
		(GradientDescent(.001), 			:red),
		(Momentum(2e-3, .5), 				:blue),
		(Adagrad(.1, 1), 					:green),
		(Adadelta(.95, .99, .02),				:cyan),
		(HyperGradientDescent(.002, 1e-8),	:yellow)	
	]
	p₁ = contour(
		xₙ, yₙ, fₙ, c=cgrad([:pink,:green, :blue]),
		levels=5000 .^ (range(0,stop=1,length=14)),
		size=(680,400)
	)
	scatter!((xₙᵢ, yₙᵢ), markersize=5, markerstrokewidth=0, c=:blue, label="Initial point")
	scatter!((1.0, 1.0), markersize=5, markerstrokewidth=0, c=:red, label="Optimal point")
	for (method, color) in methods₁
		plot_method!(method, fₙ, ∇fₙ, [xₙᵢ, yₙᵢ], nₙ, curve=color)		
	end
	p₁
end

# ╔═╡ 6dad64f0-154c-11eb-2249-fbc7007af1d7
begin
	methods₂ = [
		(ConjugateGradientDescent(), 		:red),
		(NesterovMomentum(.0002, .95), 	:purple),
		(RMSProp(.01, .6), 					:black),
		(Adam(.35, .9, .999),				:green)
	]
	p₂ = contour(
		xₙ, yₙ, fₙ, c=cgrad([:pink,:green, :blue]),
		levels=5000 .^ (range(0,stop=1,length=14)),
		size=(680,400)
	)
	scatter!((xₙᵢ, yₙᵢ), markersize=5, markerstrokewidth=0, c=:blue, label="Initial point")
	scatter!((1.0, 1.0), markersize=5, markerstrokewidth=0, c=:red, label="Optimal point")
	for (method, color) in methods₂
		plot_method!(method, fₙ, ∇fₙ, [xₙᵢ, yₙᵢ], nₙ, curve=color)		
	end
	p₂
end

# ╔═╡ 95e9f9e0-1553-11eb-25ee-5ddaf4e0e246
md"## Exercises"

# ╔═╡ 9b7ea810-1553-11eb-0b1a-f300b8082254
md"**Exercise 5.1.** Compute the gradient of \$\$x^TAx+b^Tx\$\$ when A is symmetric"

# ╔═╡ 49755460-155d-11eb-2013-7367b16844e2
md"**Answer**: By decomposing the matrix as a sum (really long) and then replacing the sum as a matrix form we get: \$\$∇f=2Ax+b\$\$"

# ╔═╡ 4fa175ae-155f-11eb-20c9-412e7896925b
md"**Exercise 5.2.** Apply gradient descent with unit step to f(x)=x⁴, compute two iterations."

# ╔═╡ 72138390-155f-11eb-0947-634e85008971
begin
	fₑ(x::Float64)::Array{Float64, 1} = [x^4]
	fₑ(x̄::Array{Float64, 1})::Array{Float64, 1} = fₑ(x̄[1])
	∇fₑ(x::Float64)::Array{Float64, 1} = [4*x^3]
	∇fₑ(x̄::Array{Float64, 1})::Array{Float64, 1} = ∇fₑ(x̄[1])
end

# ╔═╡ b6ff677e-155f-11eb-23fb-67b742123128
begin
	xₑ = range(-.01, .5, length=50)
	plot(xₑ, (x) -> fₑ(x)[1], label="Function")
	pointsₑ = [x[1] for x in optimize!(GradientDescent(1), fₑ, ∇fₑ, [.4]; k=2)]
	plot!(pointsₑ, (x) -> fₑ(x)[1], c=:red, label="Gradient descent")
	scatter!(pointsₑ, (x) -> fₑ(x)[1], c=:black, markersize=1, markerstrokewidht=0, label=nothing)
end

# ╔═╡ 2f41ecc0-1562-11eb-1411-653c8c741c5d
md"**Exercise 5.3.** Apply one step of gradient descent to f(x)=e^x+e^-x from x =10 with both a unit step and with exact line search."

# ╔═╡ 58840f00-1562-11eb-0c1a-594bdf2fd540
begin
	fₘ(x::Float64)::Array{Float64, 1} = [exp(x)+exp(-x)]
	fₘ(x̄::Array{Float64, 1})::Array{Float64, 1} = fₘ(x̄[1])
	∇fₘ(x::Float64)::Array{Float64, 1} = Calculus.derivative(fₘ, x)
	∇fₘ(x̄::Array{Float64, 1})::Array{Float64, 1} = ∇fₘ(x̄[1])
end

# ╔═╡ 7ce1ffa0-1568-11eb-26c0-9f5828fd6f13
md"**Answer**: If someone can do this without the gradient exploting to infinity really quick please help."

# ╔═╡ 9275d52e-1568-11eb-3289-0563674bb57a
md"**Exercise 5.4.** The conjugate gradient method can be used to find a search direction d when a local quadratic model of a function is available at the current point. With d as search direction, let the model be \$\$q(d) = d^T Hd+b^Td+c\$\$ for a symmetric matrix H, What is the Hessian in this case? What is the gradient of q when d=0? What can go wrong if the conjugate gradient method is applied to the quadratic model to get the search direction d?"

# ╔═╡ 8b9577b0-1569-11eb-0cc3-e3cf00f2176f
md"**Answer**: We knew that: \$\$∇q = 2Hd\$\$, then the gradient (gradient of gradient) will be: \$\$∇^2q=2H\$\$"

# ╔═╡ db7d0660-156b-11eb-2e2d-03747a944969
md"When d = 0 the gradient will also be 0. And if this happens we will be dividing by infinitesimal values when computing the next search direction. (This happened to me when using too many iterations)"

# ╔═╡ 694e9120-156c-11eb-017e-db4890cc517a
md"**Exercise 5.5.** How is Nesterov momentum an improvement over momentum?"

# ╔═╡ 786ca74e-156c-11eb-079c-639446c1339e
md"**Answer**: The problem with momentum is that it doesn't slow down, and Nesterov momentum uses the calculation of the gradient at the next step (if it overshoots the gradient will make it go back, correcting the step)."

# ╔═╡ dd160b00-156d-11eb-2aa3-6d96c0546927
md"**Exercise 5.6.** In what way is the conjugate gradient method an improvement over steepest descent?"

# ╔═╡ f095f4b0-156d-11eb-3082-754f947b8589
md"**Answer**: Gradient descent is really slow near valleys (because the next direction will always be orthogonal to the previous, causing a zig-zag movement), that is corrected when the previous gradient contributes to the next direction (making it more like a straight path to the minimum)."

# ╔═╡ a6659e10-1570-11eb-27fa-fd58fa23e5ef
md"**Exercise 5.7.** In the conjugate gradient descent, what is the normalized descent direction at the first iteration for the function \$\$f(x, y) = x^2+xy+y^2+5\$\$ when initialized at (x,y) = (1, 1)? What is the resulting point after two steps of the conjugate gradient method?"

# ╔═╡ f3f9ffde-1570-11eb-0f66-c143d09887b2
begin
	fₚ(x, y) = x^2 + x*y + y^2 + 5
	fₚ(x̄) = fₚ(x̄...)
	∇fₚ(x̄) = Calculus.gradient(fₚ, x̄)
end

# ╔═╡ e2ef7440-1576-11eb-21ad-257f12fb8cef
begin
	plotly()
	xₚ = range(-3, 3, length=40)
	yₚ = range(-3, 3, length=40)
	pₚ = contour(xₚ, yₚ, fₚ)
	plot_method!(ConjugateGradientDescent(), fₚ, ∇fₚ, [1, 1], 2)
	gr()
	pₚ
end

# ╔═╡ 9e3ea810-1577-11eb-3775-2952691b39fa
begin
	Mₚ = ConjugateGradientDescent()
	init!(Mₚ, fₚ, ∇fₚ, [1, 1])
	xₚ₀ = [1, 1]
	xₚ₁ = step!(Mₚ, fₚ, ∇fₚ, xₚ₀)
	r₁, r₂ = Mₚ.d[1], Mₚ.d[2]
	xₚ₂ = step!(Mₚ, fₚ, ∇fₚ, xₚ₁)
end;

# ╔═╡ 88e93b10-1577-11eb-23ba-17240069d759
md"**Answer**: The direction is [_$(r₁)_, _$(r₂)_], (Almost the same on each component), And the point is [_$(xₚ₂[1])_, _$(xₚ₂[2])_] after two iterations (expected because this function is polinomial of order two and the conjugate gradient should optimize it in two steps)."

# ╔═╡ 0a855b30-1579-11eb-3250-55615f863cf2
md"**Exercise 5.8.** We have a polynomial function f such that f(x) > 2 for all x in three-dimensional Euclidean space. Suppose we are using steepest descent with step lengths optimized at each step, and we want to find a local minimum of f. If our unnormalized descent direction is [1,2,3] at step k, is it possible for our unnormalized descent direction at step k+1 to be [0,0,-3]? Why or why not?"

# ╔═╡ 54c20680-1579-11eb-2ca3-a55b5a96a439
md"**Answer**: We know that in the steepest descent method the descent direction in every iteration is orthogonal to the next direction. \$\$[1,2,3]⋅[0,0,3] ≠ 0\$\$ Then the unnormalized descent direction at step k+1 cannot be [0,0,-3]. (TODO: Check the restriction for f(x) > 2)"

# ╔═╡ Cell order:
# ╟─8bd329d0-13f1-11eb-0e83-25a8a9620553
# ╟─68f68ae0-1510-11eb-0bb6-bb84b68d18e6
# ╟─a97881d0-150c-11eb-3391-b52524901b4c
# ╟─c71545d0-154c-11eb-38d2-ebdde946ae88
# ╟─fbe594ce-150c-11eb-3135-fb523622c0cd
# ╟─fbc03280-150c-11eb-3400-cf19d15006ba
# ╟─fb16ac10-150c-11eb-23af-816bfbad1ad9
# ╟─4dc886b0-1501-11eb-0c20-cfbbb42f5f72
# ╟─6dad64f0-154c-11eb-2249-fbc7007af1d7
# ╟─9cf0b980-13f1-11eb-2274-6bc775d469c8
# ╠═839a6840-13f2-11eb-3ee7-3b513eb25cac
# ╟─cb1de700-13f2-11eb-2ace-afc2e372b013
# ╟─ea5c8f40-13f2-11eb-22b5-61728c0e4388
# ╟─290a8cb0-13f3-11eb-2069-4f8e02d036a8
# ╠═3a625020-13fc-11eb-3cbb-95bafc7c95f9
# ╟─bcfb7970-13f3-11eb-3e89-d7e04bfc65ad
# ╠═58ee91f0-13f4-11eb-1d27-31633b18d5ee
# ╠═246f8b50-13f4-11eb-2e29-977facb5c488
# ╠═33062200-13f4-11eb-21f1-1ffa95638977
# ╠═d330cce2-13f8-11eb-2d7b-eb0338c3d1b8
# ╠═dcdfefe0-13f9-11eb-337a-2d8a80e4a5e7
# ╠═febfa170-13fc-11eb-3291-f9bdc4e9f971
# ╠═dd193610-13fe-11eb-1724-77e60a1eb306
# ╟─ecdf9a32-13fe-11eb-3ff9-5166ac683584
# ╟─6dc753e0-13ff-11eb-01ae-7d4477726252
# ╟─9841a800-13ff-11eb-1ef2-0b366a3d3b22
# ╠═01b03910-13fa-11eb-38fe-7ff1876faefc
# ╟─8107d7b0-140c-11eb-0194-67b53124e6a2
# ╟─965666e0-140c-11eb-3d64-6fea80cb3edc
# ╠═8116c470-140f-11eb-1b12-230e8c180dfe
# ╠═2fdb8310-1401-11eb-1529-ed74dd276380
# ╠═a8ae43ee-140f-11eb-2adf-4f4e1d0a8ef5
# ╠═7cc17412-140f-11eb-0411-4f7deaa61fae
# ╠═0d9a9980-1410-11eb-003d-8dbb0dff6ff4
# ╠═bd87d4c2-1410-11eb-3a2c-07be02f2d049
# ╟─df37f230-1410-11eb-0593-d5008a92eae1
# ╟─dea91fb0-1410-11eb-2c52-836b5e46eb28
# ╟─de622ba0-1410-11eb-2910-b1e14a97878f
# ╟─08a33cc0-1410-11eb-1e21-134f56f9a134
# ╟─aec7c7a2-1411-11eb-0302-2b383519fc53
# ╟─f2664400-1411-11eb-378e-dbf7d2ca2597
# ╟─fa788512-1400-11eb-286c-85633d52af21
# ╟─d1732fa0-13fe-11eb-39f0-670f5eaa94ec
# ╟─4d0090c0-141a-11eb-3bbe-d9049235aca1
# ╟─52541a70-141e-11eb-2096-c75cb6bfd792
# ╟─5d834dc0-141f-11eb-25eb-63933ff6ae4c
# ╟─0bbda8e0-1420-11eb-25bc-c34eaf97c1b6
# ╟─2418b4c0-1420-11eb-1d36-b5c012477a99
# ╟─75156ee0-1420-11eb-21ba-71bbcff2a1ed
# ╟─9a938440-1420-11eb-19f5-91f76c2eab12
# ╟─0d96b92e-1421-11eb-11f0-6dc604eeb10a
# ╟─40b8c7e0-1421-11eb-3e76-5db9f1885668
# ╠═744aaf60-1421-11eb-0dbd-1fd4f4775c8d
# ╠═08462370-1422-11eb-1ea0-ef8e58d85970
# ╠═49266a30-1422-11eb-1d84-9d24a1b81e04
# ╠═98c33ba0-1421-11eb-23ce-a76943bc3a76
# ╠═8accdfc0-1425-11eb-18fb-29f87b39e006
# ╟─6bcaf260-1425-11eb-1b6a-b14640cd9d64
# ╟─7163e650-1425-11eb-2bb9-6b6b8c3bf71a
# ╟─7ac3cd50-1425-11eb-1a93-adaeb3dac3f4
# ╟─abfc14c0-1422-11eb-0746-2375b2a7f411
# ╟─e8a44fc2-1425-11eb-1134-e7e1073a68c5
# ╟─67b3aa80-1427-11eb-2483-51fb1ed66625
# ╟─7d0d8450-1427-11eb-373d-fd77eca043b3
# ╟─b1426290-1427-11eb-3ff3-4d936d73df78
# ╟─d250f360-1428-11eb-3067-7bf3b10eeb95
# ╟─bf228510-1428-11eb-35e0-ef99d695c0de
# ╠═ff655350-1428-11eb-2baf-079bacea6696
# ╠═b9c8eea0-1429-11eb-315c-316fc17a30a1
# ╠═0c382480-142a-11eb-026a-77a06e8eb9d0
# ╠═87a82de0-142a-11eb-3c40-3f50754cc0fe
# ╟─1b6bfab0-142c-11eb-1d7e-995795b8d26f
# ╟─2279db60-142c-11eb-1d5d-f16fbfdf7451
# ╟─23190190-142c-11eb-2cc2-614ee78d3e8a
# ╠═f6dc15e0-142b-11eb-30fb-718decba0cb3
# ╟─09d749c0-142d-11eb-20d6-49c529dda237
# ╟─2f0e9220-142d-11eb-23f9-cd8664d7d7f7
# ╟─349bc640-142d-11eb-3baa-e7dc46b4b8a1
# ╟─76d33890-142d-11eb-253a-491b80114239
# ╟─a38f6660-142d-11eb-21a2-8529e11f8acb
# ╟─9b6ad3a0-142f-11eb-3482-ed377d4bb2a8
# ╟─72a17620-1477-11eb-166a-fd7dac2a01bd
# ╠═93479da0-1477-11eb-0c7b-a9446b9dd6d3
# ╠═f8d6b0f0-1479-11eb-1c27-4b98bb05bd34
# ╠═fd7f9f92-1479-11eb-11de-c9b923f3fe06
# ╠═0424d360-147a-11eb-2b4a-99845e4f9e5a
# ╠═0bc30d30-147a-11eb-2480-53595b44dd79
# ╠═f1b79230-1479-11eb-22d6-c3540c7f6ae1
# ╟─ce3736b0-147b-11eb-3acf-4917b0134727
# ╟─d5bd2d90-147b-11eb-3af9-7bfab1d3efe8
# ╟─db2fd4d0-147b-11eb-35a8-770858111cba
# ╟─0c426470-147c-11eb-1e91-f131c1a1da0b
# ╟─ad146fb0-147c-11eb-3756-33ad44898b84
# ╟─cbecf130-147e-11eb-0748-0d4f4af1775c
# ╟─e572c170-147e-11eb-0931-c5485d3e4203
# ╟─349f9d90-147f-11eb-3868-8b295d0f4e59
# ╟─49aaba22-148a-11eb-3782-cd3dcb9a53b1
# ╟─6a12c730-148a-11eb-3fa7-63cf835a531d
# ╟─7d457b40-148a-11eb-101a-15c60623e966
# ╠═af9daa40-148a-11eb-0b19-452c71a11b26
# ╠═e6fe3080-148b-11eb-28ba-8f5a1b03e5cb
# ╟─f62399b0-148b-11eb-2078-c97839d84aaf
# ╟─fb8c07c0-148b-11eb-3291-f985249cb0ae
# ╟─00b019d0-148c-11eb-0c2b-bb5287624389
# ╟─0660db82-148c-11eb-1823-9f03b67f13fa
# ╟─aa77ffa0-148c-11eb-0641-19ad5798aa8d
# ╟─afa20520-148c-11eb-0568-b163d9d760e5
# ╟─cf5ede10-148c-11eb-2fe9-ddedbf608aad
# ╟─c6ffcfd2-148d-11eb-3486-1fd4742afaa0
# ╟─e83f7830-148d-11eb-0119-770094c7ad55
# ╟─40e2a980-148e-11eb-1076-5767cba6fff6
# ╟─fc7607ae-148d-11eb-25fa-59a3d2c4c722
# ╟─e5daad30-148d-11eb-224a-1bb3c3d5b20a
# ╟─6595a6b0-148e-11eb-3a6a-8dfdb2ef7933
# ╠═65ad4d5e-148e-11eb-1121-c139c1000455
# ╠═f373bf20-148f-11eb-178b-ddd8ed8df869
# ╟─f71a8000-148f-11eb-08bb-911cfaa79535
# ╟─f87c8100-148f-11eb-3938-2dd7affdd0a4
# ╟─f8981f50-148f-11eb-1b9b-3dfd2d711494
# ╠═e77893d0-148f-11eb-3caf-639d09e446f5
# ╟─0f715b40-1492-11eb-0c2f-4d5ff35414b6
# ╟─17b8a650-1492-11eb-3fcb-9dce6ad8e093
# ╟─0ceaa960-14df-11eb-3e04-613396d49c39
# ╠═35045130-14df-11eb-227e-af37a738cf78
# ╠═8ec902a0-14e0-11eb-0cba-f1917748a92f
# ╠═b429cc00-14e0-11eb-27ba-b76917406f92
# ╟─b3fc0540-14e0-11eb-1975-47351cf57db3
# ╟─b37dfba0-14e0-11eb-161e-812bbfff5a45
# ╠═84527b80-14e0-11eb-0175-ddaa65c37cde
# ╟─22447bd0-14e2-11eb-2385-afe72e00c514
# ╟─8d785802-14e6-11eb-2768-7d57bb0c552d
# ╟─80d5c920-14e6-11eb-0b2b-aba1f51d4c40
# ╟─f4919fb0-14e6-11eb-37da-7f8b001f912f
# ╟─53fc59e0-14e7-11eb-1e42-c93841a25d8b
# ╟─751b81f0-14e7-11eb-30dd-d914bd2aae92
# ╟─75af8490-14e7-11eb-266b-2f84b10c59b6
# ╟─75cb9810-14e7-11eb-0a01-15c658df3cd6
# ╟─75e93230-14e7-11eb-1436-155363e20db8
# ╟─a6b66980-14e9-11eb-1f40-4b6f60bd56db
# ╠═c9666ac2-14e9-11eb-3f14-cbf7adf802d9
# ╠═cf904cb0-14ec-11eb-3cec-af99445c4bb3
# ╠═3955ac20-14ee-11eb-314d-ab74464eb64b
# ╟─393c57c0-14ee-11eb-09fd-a51eb1d06292
# ╟─39257460-14ee-11eb-365d-45eca91e680e
# ╠═ee0e6410-14ec-11eb-3aac-590c727498f8
# ╟─94087c30-14f1-11eb-2df8-27240ebdd2ae
# ╟─3b6a5e82-14f2-11eb-1962-d19f98382b7a
# ╟─5fd33350-14f2-11eb-0548-7db92e9c8551
# ╠═59a20190-14f3-11eb-3b9b-7d4284e3f759
# ╠═b78278a0-14fb-11eb-0c8b-dd1d29967a94
# ╟─bb6aa8f0-14f3-11eb-2d69-bbc0e5d83f1e
# ╟─365cf2a0-14fb-11eb-1044-779ff3953853
# ╟─361e3bf0-14fb-11eb-250b-97be54cd99f7
# ╠═3602c4b0-14fb-11eb-05ee-1fdf72a238a7
# ╟─12723860-1504-11eb-0e56-c9d2ed88cd48
# ╠═583b4f30-14ff-11eb-251b-ed6cb326bb44
# ╠═820ea300-1501-11eb-34aa-ef388563baa8
# ╟─43917990-1501-11eb-02f1-45e5d8b91637
# ╟─4ad648c0-1501-11eb-3c64-0f6089cd7a4a
# ╟─4e0f53b0-1501-11eb-0ece-15db048a27fd
# ╠═4de66ef0-1501-11eb-3ffc-0d2721a515a0
# ╠═7c1abdd0-1501-11eb-2d93-53d722c2d8dc
# ╟─95e9f9e0-1553-11eb-25ee-5ddaf4e0e246
# ╟─9b7ea810-1553-11eb-0b1a-f300b8082254
# ╟─49755460-155d-11eb-2013-7367b16844e2
# ╟─4fa175ae-155f-11eb-20c9-412e7896925b
# ╠═72138390-155f-11eb-0947-634e85008971
# ╠═b6ff677e-155f-11eb-23fb-67b742123128
# ╟─2f41ecc0-1562-11eb-1411-653c8c741c5d
# ╠═58840f00-1562-11eb-0c1a-594bdf2fd540
# ╟─7ce1ffa0-1568-11eb-26c0-9f5828fd6f13
# ╟─9275d52e-1568-11eb-3289-0563674bb57a
# ╟─8b9577b0-1569-11eb-0cc3-e3cf00f2176f
# ╟─db7d0660-156b-11eb-2e2d-03747a944969
# ╟─694e9120-156c-11eb-017e-db4890cc517a
# ╟─786ca74e-156c-11eb-079c-639446c1339e
# ╟─dd160b00-156d-11eb-2aa3-6d96c0546927
# ╟─f095f4b0-156d-11eb-3082-754f947b8589
# ╟─a6659e10-1570-11eb-27fa-fd58fa23e5ef
# ╠═f3f9ffde-1570-11eb-0f66-c143d09887b2
# ╠═e2ef7440-1576-11eb-21ad-257f12fb8cef
# ╠═9e3ea810-1577-11eb-3775-2952691b39fa
# ╟─88e93b10-1577-11eb-23ba-17240069d759
# ╟─0a855b30-1579-11eb-3250-55615f863cf2
# ╟─54c20680-1579-11eb-2ca3-a55b5a96a439
