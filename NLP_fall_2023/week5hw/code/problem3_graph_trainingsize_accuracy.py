import matplotlib.pyplot as plt

# Define the data points
training_size = [1, 2, 4, 8]
accuracy = [207/270, 251/270, 263/270, 264/270]

# Create the scatter plot with specific x-axis values
plt.scatter(training_size, accuracy, marker='o')

# Customize the plot
plt.title('Model Accuracy vs. Training Size')
plt.xlabel('Training Size')
plt.ylabel('Accuracy')
plt.grid(True)

# Show the plot
plt.show()
