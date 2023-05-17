import tensorflow as tf

learning_rate = 0.01
num_iterations = 10

optimizer = tf.keras.optimizers.Adam(learning_rate)

x = tf.Variable(tf.ones([1, 1]))
y = tf.Variable(tf.ones([1, 1]))

for step in range(num_iterations):
    with tf.GradientTape() as tape:
        area = x * y
        perimeter = 2 * x + 2 * y
        # calculate the loss that we want to minimize
        difference_sq = tf.math.square(perimeter - 100)
        #calculate the gradient
        gradients = tape.gradient(difference_sq, [x, y])
        # update x,y
        optimizer.apply_gradients(zip(gradients, [x, y]))
        
        print("Iteration", step)
        print("x:", x.numpy())
        print("y:", y.numpy())
        print("Area:", area.numpy())
        print("Perimeter:", perimeter.numpy())
        print("Squared error:", tf.norm(tf.math.sqrt(difference_sq)).numpy())
        print()


maximum_area = x * y
print("With 100 ft of wire fencing, the maximum area of the garden is", maximum_area.numpy()[0][0], "square feet.")
