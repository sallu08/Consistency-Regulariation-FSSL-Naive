import numpy as np
import os
import tensorflow as tf
import flwr as fl
from tensorflow import keras
import tensorflow.keras.layers
from keras.models import load_model
import keras.backend as K

num_classes = 10
input_shape = (28, 28, 1)
    

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"




def mse(act, pred):

    loss = K.square(pred - act)  # (batch_size, 2)                
    # summing both loss values along batch dimension 
    loss = K.sum(loss, axis=1)
   
    return loss



# Define Flower client
class My_Client(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        

    def get_properties(self, config):
        """Get properties of client."""
        raise Exception("Not implemented")

    def get_parameters(self):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]

        # Train the model using hyperparameters from config
        
        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size,
            epochs,
        
            validation_split=0.1,
        )

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        results = {
            "loss": history.history["loss"][0],
       
            "accuracy": history.history["acc"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_acc"][0],
        }
        return parameters_prime, num_examples_train, results



    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, 48, steps=steps)
        num_examples_test = len(self.x_test)
        
        return loss, num_examples_test, {"accuracy": accuracy}


def maina() -> None:

  
    # Load a subset of data to simulate the local data partition
    (x_train, y_train), (x_test, y_test),modeli = load_partition()

    # data_augmentation = tf.keras.Sequential(
    #   [
      
        
    #   ],name='aug'
    # )

      # Load and compile Keras model

    model=tf.keras.models.Sequential([
        tf.keras.layers.RandomRotation(0.3,input_shape=input_shape),
        tf.keras.layers.RandomZoom(0.1),
        modeli,
        
        ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.99)
                  ,loss=mse,metrics=['acc'])





    # Start Flower client
    client = My_Client(model, x_train, y_train, x_test, y_test)

    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=client,
        #root_certificates=Path(".cache/certificates/ca.crt").read_bytes(),
    )

    


def load_partition():
    """Load 1/10th of the training and test data to simulate a partition."""
    

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train=x_train[42000:44000]
    y_train=y_train[42000:44000]
    
    x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_train=x_train / 255.0
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
    x_test=x_test/255.0
    
    
    
    
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    
    modeli = load_model('my_modeli.h5')
    modeli.load_weights('my_modeli_weights.h5')
    modeli.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.99),
                  loss='categorical_crossentropy', metrics=['acc'])
    
    y_train1 = modeli.predict(x_train)
    # Convert predictions classes to one hot vectors 
    y_train1 = np.argmax(y_train1,axis = 1) 
    y_train1 = keras.utils.to_categorical(y_train, num_classes)
   


    
    return (
        x_train,
        y_train1,
    ), (
        x_test,
        y_test,
    ),modeli

if __name__ == "__main__":
    maina()

    