using Plots
using LinearAlgebra
import Statistics
import DelimitedFiles

"""
    solve_LR_coef(X,y,b; LIMIT)
Newton-Rhapson Algorithm for Logistic Regression
X must have 1's appended as first column.
"""
function solve_LR_coef(X,y,b; LIMIT = 500)
    n,_ = size(X)
    # Probability function
    prob(x, β) = 1 / (1 + exp(-(dot(x,β) )))
    # Gradient of Loss Function
    ∇l(β,X,y,n) = sum( X[i,:] * (y[i] - prob(X[i,:], β)) for i = 1:n)
    # Hessian of Loss Function (not used)
    # Hl(β,X,n) = -sum( X[i,:] * transpose(X[i,:]) * prob(X[i,:], β)*(1-prob(X[i,:], β)) for i = 1:n )

    global counter = 0
    tol = 1.e-6
    # Gradient descent
    while norm(∇l(b,X,y,n)) > tol && counter < LIMIT
        W = zeros(n,n)
        P = zeros(n)
        for i = 1:n
            W[i,i] = prob(X[i,:], b)*(1-prob(X[i,:], b))
            P[i] = prob(X[i,:], b)
        end
        z = X*b + inv(W)*(y-P)
        b = inv((X'*W*X)) * X'*W*z
        global counter += 1
    end

    println("Number of iterations: $counter.")
    if counter == 500
        println("Max iterations reached.")
    end

    return b
end

# Load data
data = DelimitedFiles.readdlm("poverty.txt", '\t')
X = Matrix(Float64.(data[2:end,2:end]))
n,p = size(X)
X = hcat(ones(n), X)
y = rand(0:1, n)
b = zeros(p+1)
# Solve
b = solve_LR_coef(X,y,b)
# Report error
prob(x, β) = 1 / (1 + exp(-(dot(x,β) )))
misclass = 0
for i = 1:n
    if prob(X[i,:],b) >= 0.5 && y[i] == 0
        misclass += 1
    elseif prob(X[i,:],b) < 0.5 && y[i] == 1
        misclass += 1
    end
end
println("$misclass out of $n training observations misclassified.")


"""
    kernel_ridge()
Kernel Ridge Regression
Description.
"""
function kernel_ridge(X,y,λ,K)
    n = length(y)
    k = zeros(n,n)
    for i = 1:n
        for j = 1:n
            k[i,j] = K(X[i,:],X[j,:])
            # println("$(k[i,j])")
        end
    end
    # Solution α to min (norm(y-k*α))^2 + λ α'*k*α
    α = (k + λ*I) \ y

    # f̂(x*) = sum( K(x*,X[i,:]) * α[i] )
    return α
end

# Load data
data = DelimitedFiles.readdlm("poverty.txt", '\t')
X = Float64.(data[2:end,2:end-1])
y = Float64.(data[2:end,end])
n,p = size(X)
# # Center y and estimate β_0
# y_centered = y .- Statistics.mean(y)
# β_0 = Statistics.mean(y)
# # Create matrix of centered X columns
# X_centered = zeros(n,p)
# for j in 1:p
#     X_centered[:,j] = X[:,j] .- Statistics.mean(X[:,j])
# end

# Radial Basis Function
s = 10
K_rbf(x,z) = exp(-(norm(x-z)^2) / (2*(s^2)))
# Polynomial
c = 1; d = 2
K_poly(x,z) = (c + dot(x,z))^d

λ = 1
α_rbf = kernel_ridge(X,y,λ,K_rbf)
α_poly = kernel_ridge(X,y,λ,K_poly)

pred_kernel_ridge(x,K,X,α) = sum( K(x,X[i,:]) * α[i] for i = 1:size(X,1))

# Mean Squared Error
y_preds_rbf = zeros(n)
y_preds_poly = zeros(n)
for i = 1:n
    y_preds_rbf[i] = pred_kernel_ridge(X[i,:],K_rbf,X,α_rbf)
    y_preds_poly[i] = pred_kernel_ridge(X[i,:],K_poly,X,α_poly)
end
println("Mean square error with RBF kernel: $(Statistics.mean( (y_preds_rbf .- y).^2))")
println("Mean square error with polynomial kernel: $(Statistics.mean( (y_preds_poly .- y).^2))")


"""
    prox_grad_desc()
Proximal Gradient Descent for Least Squares regression with L1-penalty.
Description.
"""
function prox_grad_desc(X,y,β,λ; LIMIT=500)
    # Soft-thresholding function
    S(y,l) = begin
        if abs(y) <= l
            return 0
        elseif y > l
            return y - l
        else
            return y + l
        end
    end

    # Solving min 1/2*norm(Y-Xβ,2)^2 + λ*norm(β,1)
    h = 1 / max(eigvals(X'X)...)   # learning rate
    ∇L(β) = -X'*(y-X*β)
    tol = 1.e-6
    counter = 0
    while norm(∇L(β)) > tol && counter < LIMIT
        β .= S.(β - h*∇L(β), λ*h)
        counter += 1
        if counter % 10 == 0
            println("Iteration $counter, gradient norm $(norm(∇L(β)))")
        end
    end
    
    println("Number of iterations: $counter.")
    if counter == 500
        println("Max iterations reached.")
    end

    return β
end

# Load data
data = DelimitedFiles.readdlm("poverty.txt", '\t')
X = Float64.(data[2:end,2:end-1])
y = Float64.(data[2:end,end])
n,p = size(X)
# Center y and estimate β_0
y_centered = y .- Statistics.mean(y)
β_0 = Statistics.mean(y)
# Create matrix of centered X columns
X_centered = zeros(n,p)
for j in 1:p
    X_centered[:,j] = X[:,j] .- Statistics.mean(X[:,j])
end
# Initialize β
β_init = ones(size(X_centered,2))
λ = 100

β = prox_grad_desc(X_centered, y_centered, β_init, λ)

y_pred = β_0 .+ X_centered * β
MSE = Statistics.mean((y - y_pred).^2)


"""
    elastic_net()
Elastic Net.
Description.
"""
function elastic_net(X,y,β,λ,α; LIMIT=500)
    S(y,l) = begin
        if abs(y) <= l
            return 0
        elseif y > l
            return y - l
        else
            return y + l
        end
    end

    # Solving min 1/2*norm(Y-Xβ,2)^2 + λ*(α*norm(β,1) + (1-α)*norm(β,2)^2)
    h = 1 / max(eigvals(X'X)...)   # learning rate
    ∇L(β) = -X'*(y-X*β) + 2λ*(1-α)*β
    tol = 1.e-6
    counter = 0
    while norm(∇L(β)) > tol && counter < LIMIT
        β .= S.(β - h*∇L(β), λ*h*α)
        counter += 1
        if counter % 10 == 0
            println("Iteration $counter, gradient norm $(norm(∇L(β)))")
        end
    end
    
    println("Number of iterations: $counter.")
    if counter == 500
        println("Max iterations reached.")
    end

    return β
end

# Load data
data = DelimitedFiles.readdlm("poverty.txt", '\t')
X = Float64.(data[2:end,2:end-1])
y = Float64.(data[2:end,end])
n,p = size(X)
# Center y and estimate β_0
y_centered = y .- Statistics.mean(y)
β_0 = Statistics.mean(y)
# Create matrix of centered X columns
X_centered = zeros(n,p)
for j in 1:p
    X_centered[:,j] = X[:,j] .- Statistics.mean(X[:,j])
end
# Initialize β
β_init = ones(size(X_centered,2))
λ = 100
α = 0.1

β = elastic_net(X_centered, y_centered, β_init, λ, α)

y_pred = β_0 .+ X_centered * β
MSE = Statistics.mean((y - y_pred).^2)


"""
    SVM()
Support Vector Machines
Description.
"""


"""
    kernel_SVM()
Kernel Support Vector Machines
Description.
"""