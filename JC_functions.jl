using Plots
using LinearAlgebra
using JuMP
#using Gurobi
import Statistics
import DelimitedFiles
import Random

"""
    gradient_descent(f, ∇f, x0; α, tol, LIMIT)
Gradient descent algorithm for a function f, gradient ∇f, initial guess x0, learning rate α, and tolerance tol. LIMIT is max iterations.
"""
function gradient_descent(f, ∇f, x0; α = 1.e-3, tol = 1.e-7, LIMIT = 5.e6)
    global counter = 0
    x = x0

    while norm(∇f(x)) > tol && counter < LIMIT
       x = x .- α .* ∇f(x)
       global counter += 1
    end

    println("Number of iterations: $counter.")
    if counter == LIMIT
        println("Max iterations reached.")
    end

    return x
end

function f(x)
    return x[1]^3 + x[2]^2
end

function ∇f(x)
    return [3. * x[1]^2; 2. * x[2]]
end

function ∇2f(x)
    return [6. * x[1]    0.;  0. 2.]
end

x = gradient_descent(f, ∇f, [1.; 2.])
println(x)

"""
    newton_method(f, ∇f, ∇2f, x0; tol, LIMIT)
Newton method for a function f, gradient ∇f, Hessian ∇2f, initial guess x0, and tolerance tol. LIMIT is max iterations.
"""
function newton_method(f, ∇f, ∇2f, x0; tol = 1.e-7, LIMIT = 500000)
    global counter = 0
    x = x0

    while norm(∇f(x)) > tol && counter < LIMIT
       x = x .- ∇2f(x) \ ∇f(x)
       global counter += 1
    end

    println("Number of iterations: $counter.")
    if counter == LIMIT
        println("Max iterations reached.")
    end

    return x
end

    println()
    println(∇2f(zeros(2)))

    x = newton_method(f, ∇f, ∇2f, [1.; 2.])
    println(x)

"""
    nonnegative_matrix_factorization(X, k; tol, LIMIT)
Performs nonnegative matrix factorization X = W * H where X is n x p, W is n x k, and H is k x p. The inner dimension k is a hyperparameter.
"""
function nonnegative_matrix_factorization(X, k; tol = 1.e-7, LIMIT = 1000)
    n, p = size(X)
    W = ones(n, k)
    H = ones(k, p)


    global counter = 0
    oldWH = X
    newWH = W * H

    while norm(newWH - oldWH) > tol && counter < LIMIT 
        oldWH = W * H

        H_num = W' * X
        H_denom = W' * W * H

        H = H .* (H_num ./ H_denom)

        W_num = X * H'
        W_denom = W * H * H'

        W = W .* (W_num ./ W_denom)

        counter +=1
        newWH = W * H
    end

    return W, H

end

println()
X = [1 2 3; 4 5 6; 7 8 9; 10 11 12]
W, H = nonnegative_matrix_factorization(X, 2)