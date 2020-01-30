import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten , Conv2D , MaxPooling2D
from keras.optimizers import Adam , Linear



from keras.callbacks import TensorBoard

#...

# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)


class dqnagent:
    def  create_model(self):
        model = Sequential()
        model.add(Conv2D(256,(3,3), input_shape = env.observation_space_values,activation = 'relu'))
        model.add(MaxPooling2D((2,2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(256,(3,3), activation = 'relu'))
        model.add(MaxPooling2D((2,2)))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Dense(env.action_space_values , activation = 'linear'))
        model.compile(loss = 'mse' , Adam(lr = 0.001) , metrics = ['acc'] )
        return model

    def __init__(self):
        self.model = self.create_model()

        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = dwque(maxlen = replay_memory_size)

        self.tensorboard = ModifiedTensorBoard(log_dir = "logs/{}-{}".format(model_name , int(time.time())))

        self.target_update_counter  = 0
    
    # (obs_space , action ,reward , new_obs_space , done)
    def update_replay_memory(self,transition):
        self.replay_memory.append(transition)
    
    def get_qus(self,state):
        return self.model.predict(np.array(state).reshape(-1, *state_shape)/255)[0]
        

