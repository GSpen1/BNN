from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import keras_tuner
import pandas as pd
import numpy as np
import math 
import matplotlib.pyplot as plt 
# Check Keras version and import accordingly
version_fn = getattr(tf.keras, "version", None)
if version_fn and version_fn().startswith("3."):
    import tf_keras as keras
else:
    keras = tf.keras

from keras.models import Model, Sequential
from keras.layers import Dropout, Input, Dense
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras import layers, regularizers
import pickle 


def load_pkl(file_name):
    """
    Loads .pkl file from path: file_name
    """
    with open(file_name, "rb") as handle:
        data = pickle.load(handle)
    print(f"Data loaded: {file_name}")
    return data



#scaler code 
def mm_norm(arr, normP):
    """
    Min max normalisation
    """
    arrmax = normP[0]
    arrmin = normP[1]
    return (arr - arrmin) / (arrmax - arrmin)


def mm_rev(norm, normP):
    """
    Reverse min max normalisation
    """
    arrmax = normP[0]
    arrmin = normP[1]
    return norm * (arrmax - arrmin) + arrmin

#C:\Users\gs420\OneDrive - Imperial College London\PhD\Code\Hybrid training\Hybrid training\UQ\dataset_all_enhanced_Integrals_per_half_cycle_no_final_time.pkl
dataset = load_pkl("data/dataset_hybrid_SPCE_one_step_behind.pkl")


with open("activation_function.txt", "r") as f:
    activation_function = f.read()
inputs_label = [
    "Cfeed1",
    "Cfeed2",
    "Cfeed3",
    "Cfeed4",
    "Cmod1",
    "Cmod2",
    "QI1",
    "QI2",
    "Qmax",
    "Cin1",
    "Cin2",
    "Cin3",
    "Cin4",
    "SPCEout1",
    "SPCEout2",
    "SPCEout3",
    "SPCEout4",
]
outputs_label = ["Cout2", "Cout3", "Cout4"]






# Taking mechanistic Data
X = dataset[inputs_label]
y = dataset[outputs_label]

X = X.dropna()
y = y.dropna() 

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



Xtrain = X_train.to_numpy(dtype="float64")
Ytrain = y_train.to_numpy(dtype="float64")
Xtest = X_test.to_numpy(dtype="float64")
Ytest = y_test.to_numpy(dtype="float64")


# Min max normalisation - Using Nanmax/min to ensure no NaN in max or min 
Xmax = np.nanmax(Xtrain,axis=0)
Xmin = np.nanmin(X_train, axis=0)
Ymax = np.nanmax(Ytrain, axis=0)
Ymin = np.nanmin(Ytrain,axis=0)
normP = ((Xmax, Xmin), (Ymax, Ymin))

train_set = (mm_norm(Xtrain, normP[0]), mm_norm(Ytrain, normP[1]))
test_set = (mm_norm(Xtest, normP[0]), mm_norm(Ytest, normP[1]))
Xtest_df = pd.DataFrame(test_set[0], columns=inputs_label)
Ytest_df = pd.DataFrame(test_set[1], columns=outputs_label)
Xtrain_df = pd.DataFrame(train_set[0], columns=inputs_label)
Ytrain_df = pd.DataFrame(train_set[1], columns=outputs_label)

X_test = Xtest_df.to_numpy() 
y_test = Ytest_df.to_numpy() 
X_train = Xtrain_df.to_numpy() 
y_test = Ytrain_df.to_numpy() 

# Save the DataFrames to CSV
Xtest_df.to_csv("X_Test_Set.csv", index=False)
Ytest_df.to_csv("Y_Test_Set.csv", index=False)

n_features = len(inputs_label)
n_output = len(outputs_label) 




def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_model = keras.Sequential(
        [
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(n), scale_diag=tf.ones(n)
                )
            )
        ]
    )
    return prior_model


def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_model = keras.Sequential(
        [
            tfp.layers.VariableLayer(
                tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
            ),
            tfp.layers.MultivariateNormalTriL(n),
        ]
    )
    return posterior_model


def create_model_inputs():
    inputs = {}
    for feature_name in inputs_label:
        inputs[feature_name] = layers.Input(
            name=feature_name, shape=(1,), dtype=tf.float32
        )
    return inputs

def create_bnn(inputs3, n_features, hidden_units, n_outputs, input_labels): 

    inputs = create_model_inputs() 



    input_size = len(inputs)

    features = keras.layers.concatenate(list(inputs.values()))
    features = layers.BatchNormalization()(features)
    print("Features before DenseVariational:", features)

    for units in hidden_units: 
        features = tfp.layers.DenseVariational(
            units = units, 
            make_prior_fn = prior, 
            make_posterior_fn = posterior,
            kl_weight = 1/input_size,  
            activation = 'sigmoid'

         )(features) 
        

    distribution_params = layers.Dense(units = 2)(features)
    outputs = tfp.layers.IndependentNormal(n_outputs)(distribution_params)

    model = keras.Model(inputs = inputs , outputs = outputs)

    return model 


# def negative_loglikelihood(targets, estimated_dist):  
#     return -estimated_dist.log_prob(targets)

def NLL(y_true, y_pred, n_output): 
    mu = y_pred[:, :n_output]
    log_sigma = y_pred[:, n_output:]

    #sigma = tf.nn.softplus(log_sigma)  # Ensures positivity and avoids instability
    sigma = tf.math.exp(log_sigma)

    dist = tfp.distributions.Normal(loc=mu, scale=sigma)
    
    return -tf.reduce_mean(dist.log_prob(y_true))  # Negative Log Likelihood



       
#network params: 
num_epochs = 32
learning_rate = 0.0001
hidden_units = [128,64,32] #not in use




def create_bnn_model(input_shape, n_output):
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(input_shape,)))  
    model.add(tfp.layers.DenseFlipout(128, activation='relu'))
    model.add(tfp.layers.DenseFlipout(128, activation='relu'))
    model.add(tfp.layers.DenseFlipout(n_output*2, activation='linear'))  
    return model

model = create_bnn_model(n_features, n_output)

model.compile(
    optimizer = keras.optimizers.RMSprop(learning_rate),
    loss = lambda y_true, y_pred: NLL(y_true, y_pred, n_output)
)

model.fit(X_train, y_train, epochs = num_epochs, validation_data = test_set)

import datetime 
now = datetime.datetime.now() 
formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
model.save(rf'C:\Users\gs420\OneDrive - Imperial College London\PhD\Code\Hybrid training\Hybrid training\UQ\Models\BNN_{formatted_datetime}.keras')


y_pred = model.predict(X_test)
#y_test = y_test.to_numpy()  superceeded 

Cout_2_pred = y_pred[:,1]
Cout2 = y_test[:,1]

plt.scatter(Cout_2_pred,Cout2)
plt.show() 









        
   





 