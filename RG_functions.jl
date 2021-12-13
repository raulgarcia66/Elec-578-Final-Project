using Plots
using LinearAlgebra
using JuMP
using Gurobi
import Statistics
import DelimitedFiles
import Random

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

# Prediction function


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
        for j = i:n
            k[i,j] = K(X[i,:],X[j,:])
            k[j,i] = k[i,j]
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

# Prediction function
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

# Prediction function


y_pred = β_0 .+ X_centered * β
MSE = Statistics.mean((y - y_pred).^2)


"""
    elastic_net(X,y,β,λ,α; LIMIT)
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

# Prediction function


y_pred = β_0 .+ X_centered * β
MSE = Statistics.mean((y - y_pred).^2)


"""
    SVM(X,y,C)
Support Vector Machines
Description.
"""
function SVM(X,y,C)
    n,p = size(X)
    model = JuMP.Model(JuMP.optimizer_with_attributes(Gurobi.Optimizer, "MIPGap" => .01, "TimeLimit" => 180))
    JuMP.@variable(model, β_0)
    JuMP.@variable(model, β[1:p])
    JuMP.@variable(model, ξ[1:n] >= 0)
    JuMP.@objective(model, Min, β'*β)
    for i = 1:n
        JuMP.@constraint(model, y[i]*(X[i,:]'*β + β_0) >= 1 - ξ[i])
    end
    JuMP.@constraint(model, sum(ξ[i] for i = 1:n) <= C)

    JuMP.optimize!(model)

    return value(β_0), value.(β), value.(ξ)
end

# Load data
data = DelimitedFiles.readdlm("poverty.txt", '\t')
X = Matrix(Float64.(data[2:end,2:end]))
n,p = size(X)
# X = hcat(ones(n), X)
y = rand((-1,1), n)
C = 40
# Solve
(β_0, β, ξ) = SVM(X,y,C)

# Prediction function
SVM_classifier(x,β_0,β) = sign(x'*β + β_0)

# Report error
misclass = 0
for i = 1:n
    if SVM_classifier(X[i,:], β_0, β) != y[i]
        misclass += 1
    end
end
println("$misclass out of $n training observations misclassified.")


"""
    kernel_SVM()
Kernel Support Vector Machines
Description.
"""
function kernel_SVM(X,y,C,K)
    n = size(X,1)
    k = zeros(n,n)
    for i = 1:n
        for j = i:n
            k[i,j] = K(X[i,:],X[j,:])
            k[j,i] = k[i,j]
            # println("$(k[i,j])")
        end
    end

    model = JuMP.Model(JuMP.optimizer_with_attributes(Gurobi.Optimizer, "NonConvex" => 2, "MIPGap" => .01, "TimeLimit" => 180))
    JuMP.@variable(model, 0 <= α[1:n] <= C)
    JuMP.@objective(model,
                    Max,
                    sum(α[i] for i = 1:n) - 1/2 * sum( sum( α[i]*α[j]*y[i]*y[j]*k[i,j] for j = 1:n ) for i = 1:n)
                    )
    JuMP.@constraint(model, sum(α[i]*y[i] for i = 1:n) == 0)

    JuMP.optimize!(model)

    return value.(α)
end

# Load data
data = DelimitedFiles.readdlm("poverty.txt", '\t')
X = Matrix(Float64.(data[2:end,2:end]))
n,p = size(X)
# X = hcat(ones(n), X)
y = rand((-1,1), n)

# Radial Basis Function
s = 100
K_rbf(x,z) = exp(-(norm(x-z)^2) / (2*(s^2)))
# Polynomial
c = 0; d = 1
K_poly(x,z) = (c + dot(x,z))^d

C = 10
α_rbf = kernel_SVM(X,y,C,K_rbf)
α_poly = kernel_SVM(X,y,C,K_poly)

# find index of max α component, let that be k 
k_rbf = argmax(α_rbf)
k_poly = argmax(α_poly)

b_rbf = y[k_rbf] - sum( α_rbf[i]*y[i]*K_rbf(X[k_rbf,:],X[i,:]) for i = 1:n)
b_rbf = y[k_poly] - sum( α_poly[i]*y[i]*K_poly(X[k_poly,:],X[i,:]) for i = 1:n)

# Prediction function
kernel_SVM_classifier(x,K,X,α,b) = sign(sum( α[i]*y[i]*K(x,X[i,:]) for i = 1:size(X,1)) + b)

misclass_rbf = 0
misclass_poly = 0
for i = 1:n
    if kernel_SVM_classifier(X[i,:], K_rbf,X, α_rbf, b_rbf) != y[i]
        misclass_rbf += 1
    end
    if kernel_SVM_classifier(X[i,:], K_poly,X, α_poly, b_poly) != y[i]
        misclass_poly += 1
    end
end
println("RBF kernel: $misclass_rbf out of $n training observations misclassified.")
println("Polynomail kernel: $misclass_poly out of $n training observations misclassified.")

"""
    bagging()
Description.
"""
function bagging(X,y; B=10)

end


"""
    bootstrapper(X,y)
Description.
"""
function bootstrapper(X,y; B=10)
    n,p = size(X)
    b_samples = []
    ind_used = []
    ind_not_used = []
    for _ = 1:B
        X_b = zeros(n,p)
        y_b = zeros(n)
        ind_used_b = Set([])
        for i=1:n
            ind = rand(1:n)
            X_b[i,:] = X[ind,:]
            y_b[i] = y[ind]
            push!(ind_used_b, ind)
        end
        ind_all = Set(1:n)
        ind_not_used_b = setdiff(ind_all,ind_used_b)
        push!(b_samples, (X_b,y_b))
        push!(ind_used, ind_used_b)
        push!(ind_not_used, ind_not_used_b)
    end

    return b_samples, ind_used, ind_not_used
end

X = [ones(5)'; 2*ones(5)'; 3*ones(5)'; 4*ones(5)']
y = [1; 2; 3; 4]
b_samples, ind_used, ind_not_used = bootstrapper(X,y)
X_b, y_b = b_samples[1]
X_b
y_b
ind_used[1]
ind_not_used[1]


"""
    train_neural_network()
Description. For classification.
"""
function train_neural_network(X,y,num_hidden_layers,size_hidden_layers; activation="sigmoid",problem_type="regression",num_classes=1,num_epochs=1)
    
    if activation == "sigmoid"
        σ(z) = 1 / (1+exp(-z))
        Dσ(z) = σ(z)^2 * exp(-z)    # gradient with respect to z
    elseif activation == "ReLU"
        σ(z) = max(z,0)
        Dσ(z) = begin
            if z >= 0
                return 1
            else
                return 0
            end
        end
    end

    if problem_type == "classification"
        y_indicator = []
        for i = 1:length(y)
            push!(y_indicator, map(k -> y[i] == k, 1:num_classes))
        end

        L(y_indicator,a_L_prob) = -sum(y_indicator[k] * log(a_L_prob[k]) for k=1:num_classes)
        DL(y_indicator,a_L_prob) = [-y_indicator[k] / a_L_prob[k] for k=1:num_classes]

        # lo(x_L) = sum( exp(x_L[i]) for i = 1:length(x_L) )
        # hi(x) = exp(x)
        # Dlo = 
        # Dhi(x) = hi(x)
        # f(a_L_i, a_L) = exp(a_L_i) / sum( exp(a_L[i]) for i = 1:length(a_L) )
        # Df(a_L_i, a_l) = 
    elseif problem_type == "regression"
        L(y,a_L) = norm(y-a_L)^2
        DL(y,a_L) = 2*(a_L-y)
        # f(z) = z
    end
    
    L = num_hidden_layers + 2
    z = map(n -> zeros(size_hidden_layers), 1:num_hidden_layers)
    z = hcat(zeros(size(X,1)), z, zeros(num_classes))
    a = map(n -> zeros(size_hidden_layers), 1:num_hidden_layers)
    a = hcat(zeros(size(X,1)), z, zeros(num_classes))
    # Create weight parameters
    W = [[I(size(X,1))]]   # store identity for sake
    push!(W, ones(size_hidden_layers, size(X,1)))   # W[2] = weights from layer 1 to layer 2
    for _ = 1:(num_hidden_layers-1)
        push!(W, ones(size_hidden_layers, size_hidden_layers))
    end
    push!(W, ones(num_classes, size_hidden_layers))
    # Create biases
    b = [[zeros(size(X,1))]]
    for _ = 1:num_hidden_layers
        push!(b, zeros(size_hidden_layers))
    end
    push!(b, zeros(num_classes))

    δ = [[zeros(size(X,1))]]
    for _ = 1:num_hidden_layers
        push!(δ, zeros(size_hidden_layers))
    end
    push!(δ, zeros(num_classes))
    
    for _ = 1:num_epochs
        for i = 1:size(X,1)
            z[1] = X[i,:]   # identical transformation
            a[1] = σ.(z[1])
            for l = 2:(L)
                z[l] = W[l]*a[l-1] + b[l]
                a[l] = g.(z[l])
            end
            # z[L] = W[L]*a[L-1] + b[L]
            # for j = 1:length(a[L])
            #     a[L][i] = f(a[L][i],a[L])
            # end
            
            if problem_type == "classification"
                δ[L] = DL(y_indicator, a_L) .* Dσ(z[L])
            elseif problem_type == "regression"
                δ[L] = DL(y, a_L) .* Dσ(z[L])
            end

            for l = (L-1):-1:2
                δ[l] = W[l+1]' * δ[l+1] .* Dσ(z[l])
            end

            for l = 2:L
                W[l] = W[l] - δ[l] * a[l+1]'
                b[l] = b[l] - δ[l]
            end
        end
    end

    function f_predictor(x)
        z[1] = x   # identical transformation
        a[1] = σ.(z[1])
        for l = 2:(L)
            z[l] = W[l]*a[l-1] + b[l]
            a[l] = g.(z[l])
        end
        if problem_type == "classification"
            return argmax(a[L])
        elseif problem_type == "regression"
            return a[L][1]   # if regression, final layer has 1 output
        end
    end

    if problem_type == "classification"
        # Compute misclassication (not necessarily loss)
        for i = 1:size(X,1)
            misclass = 0
            if f_predictor(X[i,:]) != y[i]
                misclass += 1
            end
        end
        println("$misclass of $(size(X,1)) training images misclassied.")
    elseif problem_type == "regression"
        ave_loss = Statistics.mean(L(y,f_predictor))
        println("Average loss $ave_loss out of $(size(X,1)) training images.")
    end

    return f_predictor
end


#######################################################################################
# Create custom data for binary classification with: RBF, linear, quadratic

# RBF Boundary Data
Random.seed!(1)

r1 = sqrt.(rand(100))
t1 = 2pi*rand(100)
data1 = hcat(r1.*cos.(t1), r1.*sin.(t1))
scatter(data1[:,1], data1[:,2], color="blue")

r2 = sqrt.(3*rand(100) .+ 1)
t2 = 2pi*rand(100)
data2 = hcat(r2.*cos.(t2), r2.*sin.(t2))
scatter!(data2[:,1], data2[:,2], color="red")

data_rbf = vcat(hcat(data1,ones(100)), hcat(data2,-ones(100)))

X = data_rbf[:,1:2]
y = data_rbf[:,3]
n,p = size(X)

# Linear Boundary Data
X = rand(500,2)
y = zeros(500)
pos_ind = []
neg_ind = []
for i=1:size(data1,1)
    if X[i,2] > 1.5*X[i,1] - .25
        y[i] = 1
        push!(pos_ind,i)
    else
        y[i] = -1
        push!(neg_ind,i)
    end
end

class_pos = X[pos_ind,:]
class_neg = X[neg_ind,:]

scatter(class_pos[:,1], class_pos[:,2], color="blue", label="Class 1")
scatter!(class_neg[:,1], class_neg[:,2], color="red", label="Class 2")

x = LinRange(0,1,100)
f(x) = 1.5*x - .25
plot!(x, f.(x),xlims=(0,1),ylims=(0,1))

x = LinRange(1/5,4/5,10)
x_noise_class1 = f.(x) + 1/5 * (rand(10) .- 0.5)
x_noise_class2 = f.(x) + 1/5 * (rand(10) .- 0.5)
scatter!(x, x_noise_class1, color="blue")
scatter!(x, x_noise_class2, color="red")
XX = vcat(X,hcat(x,x_noise_class1), hcat(x,x_noise_class2))
yy = vcat(y,ones(10),-ones(10))

# Quadratic Boundary Data
X = rand(500,2)
y = zeros(500)
pos_ind = []
neg_ind = []
for i=1:size(data1,1)
    if X[i,2] > -(X[i,1]+.5)*(X[i,1]-1)
        y[i] = 1
        push!(pos_ind,i)
    else
        y[i] = -1
        push!(neg_ind,i)
    end
end

class_pos = X[pos_ind,:]
class_neg = X[neg_ind,:]

scatter(class_pos[:,1], class_pos[:,2], color="blue", label="Class 1")
scatter!(class_neg[:,1], class_neg[:,2], color="red", label="Class 2")

x = LinRange(0,1,100)
f(x) = -(x+0.5)*(x-1)
plot!(x, f.(x),xlims=(0,1),ylims=(0,1))

x = LinRange(0,1,10)
x_noise_class1 = f.(x) + 1/5 * (rand(10) .- 0.5)
x_noise_class2 = f.(x) + 1/5 * (rand(10) .- 0.5)
scatter!(x, x_noise_class1, color="blue")
scatter!(x, x_noise_class2, color="red")
XX = vcat(X,hcat(x,x_noise_class1), hcat(x,x_noise_class2))
yy = vcat(y,ones(10),-ones(10))

######################### TEST SYNTHETIC DATA ############################

# Radial Basis Function
s = 1/10
K_rbf(x,z) = exp(-(norm(x-z)^2) / (2*(s^2)))
# Polynomial
c = 0; d = 2
K_poly(x,z) = (c + dot(x,z))^d

C = 5
α_rbf = kernel_SVM(X,y,C,K_rbf)
α_poly = kernel_SVM(X,y,C,K_poly)

# find index of max α component, let that be k 
k_rbf = argmax(α_rbf)
k_poly = argmax(α_poly)

b_rbf = y[k_rbf] - sum( α_rbf[i]*y[i]*K_rbf(X[k_rbf,:],X[i,:]) for i = 1:n)
b_rbf = y[k_poly] - sum( α_poly[i]*y[i]*K_poly(X[k_poly,:],X[i,:]) for i = 1:n)

kernel_SVM_classifier(x,K,X,α,b) = sign(sum( α[i]*y[i]*K(x,X[i,:]) for i = 1:size(X,1)) + b)

misclass_rbf = 0
misclass_poly = 0
for i = 1:n
    if kernel_SVM_classifier(X[i,:], K_rbf,X, α_rbf, b_rbf) != y[i]
        misclass_rbf += 1
    end
    if kernel_SVM_classifier(X[i,:], K_poly,X, α_poly, b_poly) != y[i]
        misclass_poly += 1
    end
end

println("RBF kernel: $misclass_rbf out of $n training observations misclassified.")
println("Polynomail kernel: $misclass_poly out of $n training observations misclassified.")