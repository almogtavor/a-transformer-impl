import numpy as np

def dropout(x, dropout_rate):
    # Create a mask with the same shape as x
    mask = (np.random.rand(*x.shape) > dropout_rate).astype(np.float32)
    # Scale the outputs to keep expected value the same
    return (x * mask) / (1.0 - dropout_rate)

x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print("Before one dropout:\n", x)
print("After one dropout:\n", dropout(x, 0.5))

num_runs = 50000
sum_output = np.zeros_like(x)

for _ in range(num_runs):
    out = dropout(x, 0.5)
    sum_output += out

average_output = sum_output / num_runs

print("Original input:\n", x)
# By forcing the network to not rely too heavily on any single neuron, dropout encourages it to learn more robust, distributed representations.
# That way, the model generalizes better to new data.
# on average, across many runs, the output remains statistically the same as the input.
print("Average output over", num_runs, "runs:\n", average_output)
