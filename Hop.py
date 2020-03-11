#Quoted from https://www.bonaccorso.eu/2017/09/20/ml-algorithms-addendum-hopfield-networks/
import matplotlib.pyplot as plt
import numpy as np
from random import seed
from random import random

# Set random seed for reproducibility
np.random.seed(1000)

N = 1
nb_patterns = 3
pattern_width = 10
pattern_height = 10
max_iterations = 10

#Add 1 or -1 randomly to array
def RandomArray() :
    rd = random()
    if rd < 0.5 :
        return -1
    else :
        return 1
    
# Initialize the patterns
X = np.zeros((nb_patterns, pattern_width * pattern_height))
X[0] = [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        -1,-1,-1,-1, 1, 1,-1,-1,-1,-1,
        -1,-1,-1,-1, 1, 1,-1,-1,-1,-1,
        -1,-1,-1,-1, 1, 1,-1,-1,-1,-1,
        -1,-1,-1,-1, 1, 1,-1,-1,-1,-1,
        -1,-1,-1,-1, 1, 1,-1,-1,-1,-1,
        -1,-1,-1,-1, 1, 1,-1,-1,-1,-1,
        -1,-1,-1,-1, 1, 1,-1,-1,-1,-1,
        -1,-1,-1,-1, 1, 1,-1,-1,-1,-1,
        -1,-1,-1,-1, 1, 1,-1,-1,-1,-1]

X[1] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1,-1,-1,-1,-1,-1,-1,-1,-1, 1,
        1,-1,-1,-1,-1,-1,-1,-1,-1, 1,
        1,-1,-1,-1,-1,-1,-1,-1,-1, 1,
        1,-1,-1,-1,-1,-1,-1,-1,-1, 1,
        1,-1,-1,-1,-1,-1,-1,-1,-1, 1,
        1,-1,-1,-1,-1,-1,-1,-1,-1, 1,
        1,-1,-1,-1,-1,-1,-1,-1,-1, 1,
        1,-1,-1,-1,-1,-1,-1,-1,-1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

X[2] = [1,-1,-1,-1,-1,-1,-1,-1,-1, 1,
        1,-1,-1,-1,-1,-1,-1,-1,-1, 1,
        1,-1,-1,-1,-1,-1,-1,-1,-1, 1,
        1,-1,-1,-1,-1,-1,-1,-1,-1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1,-1,-1,-1,-1,-1,-1,-1,-1, 1,
        1,-1,-1,-1,-1,-1,-1,-1,-1, 1,
        1,-1,-1,-1,-1,-1,-1,-1,-1, 1,
        1,-1,-1,-1,-1,-1,-1,-1,-1, 1]

# Show the patterns
fig, ax = plt.subplots(1, nb_patterns, figsize=(10, 5))

for i in range(nb_patterns):
    ax[i].matshow(X[i].reshape((pattern_height, pattern_width)), cmap='gray')
    ax[i].set_xticks([])
    ax[i].set_yticks([])
    
plt.show()

# Train the network
W = np.zeros((pattern_width * pattern_height, pattern_width * pattern_height))

for i in range(pattern_width * pattern_height):
    for j in range(pattern_width * pattern_height):
        if i == j or W[i, j] != 0.0:
            continue
            
        w = 0.0
        
        for n in range(nb_patterns):
            w += X[n, i] * X[n, j]
            
        W[i, j] = w / X.shape[0]
        W[j, i] = W[i, j]
        
# Create a corrupted test pattern
x_test = np.array([])
for i in range(pattern_width * pattern_height) :
    arr = RandomArray()
    x_test = np.append(x_test, arr)

# Recover the original patterns
A = x_test.copy()

for _ in range(max_iterations):
    for i in range(pattern_width * pattern_height):
        A[i] = 1.0 if np.dot(W[i], A) > 0 else -1.0
        
for i in range(nb_patterns) :
    cnt = 0 
    for j in range(pattern_width * pattern_height):
        if X[i][j] == (-1) * A[j] :
            cnt += 1
    if cnt == pattern_width * pattern_height :
        for j in range(pattern_width * pattern_height):
            A[j] *= -1
            
# Show corrupted and recovered patterns
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].matshow(x_test.reshape(pattern_height, pattern_width), cmap='gray')
ax[0].set_title('Corrupted pattern')
ax[0].set_xticks([])
ax[0].set_yticks([])

ax[1].matshow(A.reshape(pattern_height, pattern_width), cmap='gray')
ax[1].set_title('Recovered pattern')
ax[1].set_xticks([])
ax[1].set_yticks([])

plt.show()