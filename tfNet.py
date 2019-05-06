import tensorflow as tf


class QNet(tf.keras.Model):
    def __init__(self, num_last_observations, state_shape, action_size):
        self.num_last_observations = num_last_observations
        self.state_shape = state_shape
        self.action_size = action_size
        super().__init__(name='')
        self.conv1 = tf.keras.layers.Conv2D(
            32,
            kernel_size=(3, 3),
            strides=(1, 1),
            # shape: (NUM_LAST_FRAMES, H, W)
            input_shape=(self.num_last_observations,) + self.state_shape)
        self.conv2 = tf.keras.layers.Conv2D(
            32,
            kernel_size=(3, 3),
            strides=(1, 1))
        self.pool = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(512)
        self.dense2 = tf.keras.layers.Dense(256)
        self.dense3 = tf.keras.layers.Dense(self.action_size)

    def call(self, input_tensor):
        x = self.conv1(input_tensor)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = tf.nn.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = tf.nn.relu(self.dense1(x))
        x = tf.nn.relu(self.dense2(x))
        x = self.dense3(x)
        return x


class PolicyNet(tf.keras.Model):
    def __init__(self, num_last_observations, state_shape, action_size):
        self.num_last_observations = num_last_observations
        self.state_shape = state_shape
        self.action_size = action_size
        super().__init__(name='')
        self.conv1 = tf.keras.layers.Conv2D(
            32,
            kernel_size=(3, 3),
            strides=(1, 1),
            # shape: (NUM_LAST_FRAMES, H, W)
            input_shape=(self.num_last_observations,) + self.state_shape)
        self.conv2 = tf.keras.layers.Conv2D(
            32,
            kernel_size=(3, 3),
            strides=(1, 1))
        self.pool = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(512)
        self.dense2 = tf.keras.layers.Dense(256)
        self.dense3 = tf.keras.layers.Dense(self.action_size)

    def call(self, input_tensor):
        x = self.conv1(input_tensor)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = tf.nn.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = tf.nn.relu(self.dense1(x))
        x = tf.nn.tanh(self.dense2(x))
        x = tf.nn.softmax(self.dense3(x))
        return x


def policy_gradient(q_model, policy_model, states):
    with tf.GradientTape() as tape:
        tape.watch(policy_model.trainable_variables)
        actions = policy_model(states)
        qs = q_model(states)
        loss_value = -1*tf.reduce_mean(actions*qs)

    return tape.gradient(loss_value, policy_model.trainable_variables)
