# session 的使用
import tensorflow as tf

a = tf.constant([1.0, 2.0], name="a", dtype=tf.float32)
b = tf.constant([2, 3], name="b", dtype=tf.float32)
result = a + b
print(result)
# ----------------***************------------------
# 使用普通的会话模式 生成 关闭
sess = tf.Session()
sess.run(result)
print(sess.run(result))
sess.close()
# ----------------***************------------------
# 使用上下文管理器使用会话 不需要close
with tf.Session() as sess:
    a = tf.constant([1.0, 2.0], name="a", dtype=tf.float32)
    b = tf.constant([2, 3], name="b", dtype=tf.float32)
    result = a + b
    print(sess.run(result))

# ----------------***************------------------
# tf.matmul(x,w) 实现矩阵乘法
with tf.Session() as sess:
    x = tf.constant([1, 2], shape=[1, 2])
    w = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])
    a = tf.matmul(x, w)
    print(sess.run(a))

# ----------------***************------------------
# tensorflow 变量
weights = tf.Variable(tf.random_normal([2, 3], stddev=2))
biases = tf.Variable(tf.zeros([3]))
# 定义计算图所有的计算
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

x = tf.constant([[0.7, 0.9]])  # 1*2

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
# 通过会话计算结果
sess = tf.Session()
# 分别初始化
# sess.run(w1.initializer)
# sess.run(w2.initializer)
# 一次性初始化
init_op = tf.global_variables_initializer()
sess.run(init_op)
print(sess.run(y))
sess.close()

# ----------------***************------------------
# 3.4.4 多个样例的前向传播过程
# 定义计算图
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

x = tf.placeholder(tf.float32, shape=[3, 2], name="input")
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
# 会话计算结果
sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)
print(sess.run(y, feed_dict={x: [[0.7, 0.9],
                                 [0.1, 0.4],
                                 [0.5, 0.8]]}))

# ----------------***************------------------
# 3.4.4 简单的反向传播算法
from numpy.random import RandomState

# 定义训练数据batch的大小
batch_size = 8
# 定义神经网络的参数
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
#
x = tf.placeholder(tf.float32, shape=[None, 2], name="x-input")
y = tf.placeholder(tf.float32, shape=[None, 1], name="y-input")

# 定义前向传播过程
a = tf.matmul(x, w1)
y_ = tf.matmul(a, w2)

# 定义损失函数和反向传播优化算法
y_ = tf.sigmoid(y_)
cross_entropy = -tf.reduce_mean(
    y * tf.log(tf.clip_by_value(y_, 1e-10, 1.0)) + (1 - y) * tf.log(tf.clip_by_value(1 - y_, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# 通过随机数生成模拟数据
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]
print(Y)

# 创建会话运行tensorflow程序
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(w1))
    print(sess.run(w2))
    STEPS = 5000
    for i in range(STEPS):
        start = (i * batch_size) % dataset_size  # ????
        end = min(start + batch_size, dataset_size)

        sess.run(train_step, feed_dict={x: X[start:end], y: Y[start:end]})
        # print({x:X[start:end],y:Y[start:end]})
        if i % 1000 == 0:
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y: Y})
            print("after {} training step(s), cross entropy on all data is {}".format(i, total_cross_entropy))

    print(sess.run(w1))
    print(sess.run(w2))
