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

    println("Number of iterations: $counter")
    if counter == 500
        println("Max iterations")
    end

    return b
end

data = DelimitedFiles.readdlm("poverty.txt", '\t')
X = Matrix(Float64.(data[2:end,2:end]))
n,p = size(X)
X = hcat(ones(n), X)
y = rand(0:1, n)
b = zeros(p+1)
b = solve_LR_coef(X,y,b)
# Test
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


