initialize W and b from N(0,1)
n_epoch = ؟
؟ = lr
n = number of train records
for i from 0 to n_epoch:
for each W or b:
grad[W] = 0 // grad[b] for biases
for x0,x1,y0 in train_data:
compute y
compute cost
grad[W] += dcost/dW // grad[b] and db for biases
for each W or b:
W = W – (lr * grad[w])/n // b and grad[b] for biases