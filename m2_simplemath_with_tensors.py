import tensorflow as tf

x = tf.constant([100, 200, 300], name='x')
y = tf.constant([1, 2, 3], name='y')

# sums up all elements within that element
sum_x = tf.reduce_sum(x, name="sum_x")

# multiplies all elements in the list
prod_y = tf.reduce_prod(y, name='prod_y')

# two with same scalar can be used in other div ops
final_div = tf.div(sum_x, prod_y, name="final_div")

# calculates average of the elements in a tensor (1d tensor on fly)
final_mean = tf.reduce_mean([sum_x, prod_y], name='final_mean')

sess = tf.Session()

print('X: ', sess.run(x))
print('Y: ', sess.run(y))

print('sum(x): ', sess.run(sum_x))
print('prod(y): ', sess.run(prod_y))
print('sum(x) / prod(y): ', sess.run(final_div))
print('mean(sum(x) / prod(y)): ', sess.run(final_mean))

writer = tf.summary.FileWriter('./m2_example4', sess.graph)