from keras import Sequential
from keras.layers import Dense, Conv2D, Activation, Flatten
from keras.optimizers import RMSprop
from convDQN import ConvDQNAgent
import snake_logger


qLogger=snake_logger.QLogger()


class DeeperConvDQNAgent(ConvDQNAgent):
    def _build_model(self):
        model = Sequential()

        # Convolutions.
        model.add(Conv2D(
            32,
            kernel_size=(3, 3),
            strides=(1, 1),
            data_format='channels_first',
            # shape: (NUM_LAST_FRAMES, H, W)
            input_shape=(self.num_last_observations,) + self.state_shape
        ))
        model.add(Activation('relu'))

        model.add(Conv2D(
            64,
            kernel_size=(3, 3),
            strides=(1, 1),
            data_format='channels_first'
        ))
        model.add(Activation('relu'))

        model.add(Conv2D(
            64,
            kernel_size=(3, 3),
            strides=(1, 1),
            data_format='channels_first'
        ))
        model.add(Activation('relu'))

        # Dense layers.
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dense(self.action_size))

        model.summary() # print model summary
        model.compile(RMSprop(), 'MSE')
        return model
