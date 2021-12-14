
################# Run code sections to create data ###################

# Create custom data for binary classification with: RBF, linear, quadratic
# y is encoded as -1,1 for these

##################################################################################
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
for i=1:size(X,1)
    if X[i,2] > 1.5*X[i,1] - .25
        y[i] = 1
        push!(pos_ind,i)
    else
        y[i] = -1
        push!(neg_ind,i)
    end
end

#class_pos = X[pos_ind,:]
#class_neg = X[neg_ind,:]

#scatter(class_pos[:,1], class_pos[:,2], color="blue", label="Class 1")
#scatter!(class_neg[:,1], class_neg[:,2], color="red", label="Class 2")

#x = LinRange(0,1,100)
f(x) = 1.5*x - .25
#plot!(x, f.(x),xlims=(0,1),ylims=(0,1))

x = LinRange(1/5,4/5,10)
x_noise_class1 = f.(x) + 1/5 * (rand(10) .- 0.5)
x_noise_class2 = f.(x) + 1/5 * (rand(10) .- 0.5)
#scatter!(x, x_noise_class1, color="blue")
#scatter!(x, x_noise_class2, color="red")
Xx = vcat(X,hcat(x,x_noise_class1), hcat(x,x_noise_class2))
X = hcat(ones(size(Xx,1),1),Xx)   # append columns of 1's
y = vcat(y,ones(10),-ones(10))


# Quadratic Boundary Data
X = rand(500,2)
y = zeros(500)
pos_ind = []
neg_ind = []
for i=1:size(X,1)
    if X[i,2] > -(X[i,1]+.5)*(X[i,1]-1)
        y[i] = 1
        push!(pos_ind,i)
    else
        y[i] = -1
        push!(neg_ind,i)
    end
end

#class_pos = X[pos_ind,:]
#class_neg = X[neg_ind,:]

#scatter(class_pos[:,1], class_pos[:,2], color="blue", label="Class 1")
#scatter!(class_neg[:,1], class_neg[:,2], color="red", label="Class 2")

#x = LinRange(0,1,100)
f(x) = -(x+0.5)*(x-1)
#plot!(x, f.(x),xlims=(0,1),ylims=(0,1))

x = LinRange(0,1,10)
x_noise_class1 = f.(x) + 1/5 * (rand(10) .- 0.5)
x_noise_class2 = f.(x) + 1/5 * (rand(10) .- 0.5)
#scatter!(x, x_noise_class1, color="blue")
#scatter!(x, x_noise_class2, color="red")
X = vcat(X,hcat(x,x_noise_class1), hcat(x,x_noise_class2))
y = vcat(y,ones(10),-ones(10));

#######################################################################################

# Classication

# load partial training set
tr_size = 5000
train_x, train_y = MNIST.traindata(1:tr_size)
# load partial test set
te_size = 200
test_x,  test_y  = MNIST.testdata(1:te_size)

X = zeros(tr_size,784)
for i = 1:tr_size
    X[i,:] = reshape(train_x[:,:,i],1,784)
end
y = train_y[1:tr_size]

X_test = zeros(te_size,784)
for i = 1:te_size
    X_test[i,:] = reshape(test_x[:,:,i],1,784)
end
y_test = test_y;


# Regression

X_all = transpose(BostonHousing.features())
X = X_all[1:450,:]
X_test = X_all[451:end,:]
y_all = transpose(BostonHousing.targets())
y = y_all[1:450]
y_test = y_all[451:end];


# Proximal gradient descent and elastic net
X = transpose(BostonHousing.features())
y = transpose(BostonHousing.targets())
n,p = size(X)
# Center y and estimate β_0
y_centered = y .- Statistics.mean(y)
β_0 = Statistics.mean(y)
# Create matrix of centered X columns
X_centered = zeros(n,p)
for j in 1:p
    X_centered[:,j] = X[:,j] .- Statistics.mean(X[:,j])
end


# Bootstrapping example
X = [ones(5)'; 2*ones(5)'; 3*ones(5)'; 4*ones(5)']
y = [1.0; 2.0; 3.0; 4.0]


# Logistic Regression (approx. linearly separable with y encoded as 1,0)
X = rand(500,2)
y = zeros(500)
pos_ind = []
neg_ind = []
for i=1:size(X,1)
    if X[i,2] > 1.5*X[i,1] - .25
        y[i] = 1
        push!(pos_ind,i)
    else
        y[i] = 0
        push!(neg_ind,i)
    end
end

#class_pos = X[pos_ind,:]
#class_neg = X[neg_ind,:]

#scatter(class_pos[:,1], class_pos[:,2], color="blue", label="Class 1")
#scatter!(class_neg[:,1], class_neg[:,2], color="red", label="Class 2")

#x = LinRange(0,1,100)
f(x) = 1.5*x - .25
#plot!(x, f.(x),xlims=(0,1),ylims=(0,1))

x = LinRange(1/5,4/5,10)
x_noise_class1 = f.(x) + 1/5 * (rand(10) .- 0.5)
x_noise_class2 = f.(x) + 1/5 * (rand(10) .- 0.5)
#scatter!(x, x_noise_class1, color="blue")
#scatter!(x, x_noise_class2, color="red")
Xx = vcat(X,hcat(x,x_noise_class1), hcat(x,x_noise_class2))
X = hcat(ones(size(Xx,1),1),Xx)   # append columns of 1's
y = vcat(y,ones(10),zeros(10));