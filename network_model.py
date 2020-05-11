import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras.layers import Conv2D, Flatten, Dense, Lambda, Input, Multiply, Layer
from keras.layers.merge import _Merge
from keras.initializers import VarianceScaling
from keras.optimizers import Adam
from keras.models import Model, load_model
from keras.backend import mean, dot
import numpy as np

class DQN:
    def __init__(self, num_actions, frame_height=84, frame_width=84, frame_stack_length=4, hidden=1024, batch_size=32,
                 path="output", path2="models1"):
        self.num_actions = num_actions
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.frame_stack_length = frame_stack_length
        self.hidden = hidden
        self.batch_size = batch_size
        self.path = path
        self.path2 = path2

        self.INPUT_SHAPE = (self.frame_height, self.frame_width, self.frame_stack_length)
        self.action_space = (num_actions,)
        self.__build_model()
        self.__all_actions_activation = np.ones((1, self.num_actions))
        self.__all_actions_activation_batch = np.ones((self.batch_size , self.num_actions))

    def __build_model(self):
        self.input = Input(self.INPUT_SHAPE)
        self.input_normalized = Lambda(lambda x: x / 255.0)(self.input)
        self.actions_input = Input(self.action_space)

        self.conv1 = Conv2D(filters=32, kernel_size=[8, 8], strides=4,
                            kernel_initializer=VarianceScaling(scale=2),
                            padding='valid', activation='relu', use_bias=False)(self.input_normalized)
        self.conv2 = Conv2D(filters=64, kernel_size=[4, 4], strides=2,
                            kernel_initializer=VarianceScaling(scale=2),
                            padding='valid', activation='relu', use_bias=False)(self.conv1)
        self.conv3 = Conv2D(filters=64, kernel_size=[3, 3], strides=1,
                            kernel_initializer=VarianceScaling(scale=2),
                            padding='valid', activation='relu', use_bias=False)(self.conv2)
        self.conv4 = Conv2D(filters=self.hidden, kernel_size=[7, 7], strides=1,
                            kernel_initializer=VarianceScaling(scale=2),
                            padding='valid', activation='relu', use_bias=False)(self.conv3)

        #self.value_stream_unflattened = Lambda(lambda x: x[:, :, :, 0 : self.hidden // 2])(self.conv4)
        #self.advantage_stream_unflattened = Lambda(lambda x: x[:, :, :, self.hidden // 2 : self.hidden])(self.conv4)
        self.value_stream_unflattened = FirstHalf(output_dim=(None, 1, 1, self.hidden // 2))(self.conv4)
        self.advantage_stream_unflattened = SecondHalf(output_dim=(None, 1, 1, self.hidden // 2))(self.conv4)

        self.value_stream = Flatten()(self.value_stream_unflattened)
        self.advantage_stream = Flatten()(self.advantage_stream_unflattened)

        self.advantage = Dense(units=self.num_actions, kernel_initializer=VarianceScaling(scale=2))(self.advantage_stream)
        self.value = Dense(units=1, kernel_initializer=VarianceScaling(scale=2))(self.value_stream)

        self.q_values = QLayer()([self.value, self.advantage])

        self.filtered_output = Multiply()([self.q_values, self.actions_input])

        self.__model = Model(input=[self.input, self.actions_input], output=self.filtered_output)
        self.__model.compile(optimizer=Adam(lr=0.00001), loss="mean_squared_error")

        print(self.__model.summary())

    def get_weights(self):
        return self.__model.get_weights()

    def set_weights(self, weights):
        self.__model.set_weights(weights)

    def get_best_action(self, state):
        state = np.reshape(state, (1,) + state.shape) #expand dims(1 because only one state is input)
        return np.argmax(self.__model.predict([state, self.__all_actions_activation]))

    def get_best_actions_batch(self, states):
        return np.argmax(self.__model.predict([states, self.__all_actions_activation_batch]), axis=1)

    def predict_batch(self, states, one_hot_encoded_actions):
        return self.__model.predict([states, one_hot_encoded_actions])

    def fit_batch(self, states, ohe_best_actions_current_states, target_q_values):
        self.__model.fit([states, ohe_best_actions_current_states], target_q_values, batch_size=self.batch_size, nb_epoch=1, verbose=0)

    def save_model(self, iter):
        self.__model.save(self.path2 + "/" + str(iter) + ".h5")

    def load_model(self, iter):
        self.__model = load_model(self.path + "/" + str(iter) + ".h5",
                                  custom_objects={'FirstHalf': FirstHalf,
                                                  'SecondHalf': SecondHalf,
                                                  'QLayer': QLayer})

class QLayer(_Merge):
    '''Q Layer that merges an advantage and value layer'''
    def _merge_function(self, inputs):
        '''Assume that the inputs come in as [value, advantage]'''
        output = inputs[0] + (inputs[1] - mean(inputs[1], axis=1, keepdims=True))
        return output


class FirstHalf(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(FirstHalf, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(FirstHalf, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return x[:,:,:, 0: self.output_dim[3]]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], input_shape[3] // 2)

    def get_config(self):
        config = {'output_dim': self.output_dim}
        base_config = super(FirstHalf, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SecondHalf(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(SecondHalf, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(SecondHalf, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return x[:,:,:, self.output_dim[3]: self.output_dim[3] * 2]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], input_shape[3] // 2)

    def get_config(self):
        config = {'output_dim': self.output_dim}
        base_config = super(SecondHalf, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))