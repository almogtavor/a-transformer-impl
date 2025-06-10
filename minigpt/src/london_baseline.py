# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.
import utils

DEV_PATH = "../birth_dev.tsv"

# Count how many lines/examples are in the dev set:
with open(DEV_PATH, encoding="utf-8") as f:
    num_examples = sum(1 for _ in f)
# Build a list of "London" predictions - one per example:
predictions = ["London"] * num_examples
total, correct = utils.evaluate_places(DEV_PATH, predictions) # compute accuracy
print(f"Always‐London baseline: {correct}/{total} = {correct/total*100:.2f}%")
