import tensorflow as tf
import numpy as np

class FNN():
    def __init__(self, d_input, d_hidden_layers, d_output, learning_rate):
        super().__init__()
        self.d_input = d_input
        self.d_hidden_layers = d_hidden_layers
        self.d_output = d_output
        self.d_layers = d_hidden_layers + [d_output]
        self._layers = []
        self.learning_rate = learning_rate

        self.observations = tf.placeholder(shape=(None, d_input), dtype=tf.float32)
        outputs = self.observations
        for i in range(len(self.d_layers)):
            if i < (len(self.d_layers) - 1):
                outputs = tf.layers.dense(outputs,
                                          units=self.d_layers[i],
                                          activation=tf.tanh,
                                          kernel_initializer=tf.initializers.random_normal(stddev=1))
            else:
                outputs = tf.layers.dense(outputs, units=self.d_layers[i], activation=None,
                                          kernel_initializer=tf.initializers.random_normal(stddev=1))
            self._layers.append(outputs)

        # policy
        self.policy = tf.nn.softmax(outputs)
        self.log_policy = tf.nn.log_softmax(outputs)

        # log probabilities for taken actions
        self.actions = tf.placeholder(shape=(None,), dtype=tf.int32)
        self.actions_enc = tf.one_hot(self.actions, self.d_output)
        self.log_proba = tf.reduce_sum(self.actions_enc * self.log_policy, axis=1)

        # loss & train operation
        self.returns = tf.placeholder(shape=(None,), dtype=tf.float32)
        self.loss = - tf.reduce_mean(self.log_proba * self.returns, axis=0)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def init_session(self):
        config_tf = tf.ConfigProto(device_count={'GPU': 0})
        self.session = tf.Session(config=config_tf)
        self.session.run(tf.global_variables_initializer())

    def action(self, obs):
        policy_ = self.session.run(self.policy, {self.observations: obs.reshape(1,-1)})[0]
        action = np.argmax(np.random.multinomial(1, policy_))
        return action

    def close_session(self):
        self.session.close()

    def update(self, feed_observations, feed_actions, feed_returns):
        feed = {self.observations: feed_observations, self.actions: feed_actions, self.returns: feed_returns}
        loss, _ = self.session.run([self.loss, self.train_op], feed)
        return loss

