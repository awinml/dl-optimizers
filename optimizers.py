import pandas as pd
import numpy as np
import tensorflow as tf
from keras.optimizers import SGD, Adagrad, RMSprop, Adadelta, Adam
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock
from sklearn import linear_model, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

tf.random.set_seed(0)

import streamlit as st

data = datasets.load_breast_cancer()
df = pd.DataFrame(data["data"], columns=data["feature_names"])
df["target"] = data["target"]

X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, random_state=0
)


def plot_loss(history):
    fig, ax = plt.subplots()
    ax.plot(history.history["loss"], label="loss")
    ax.plot(history.history["val_loss"], label="val_loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Error")
    ax.set_title("Train Loss vs Validation Loss")
    ax.legend()
    ax.grid(True)
    return fig


def create_model():
    model = Sequential()
    model.add(
        Dense(32, kernel_initializer="normal", input_dim=30, activation="leaky_relu")
    )
    model.add(Dense(16, kernel_initializer="uniform", activation="leaky_relu"))
    model.add(Dropout(rate=0.3))
    model.add(Dense(16, kernel_initializer="uniform", activation="sigmoid"))
    model.add(Dropout(rate=0.4))
    model.add(Dense(1, activation="sigmoid"))
    return model


def fit_model(model, optmizer, X_train, X_test, y_test, batch_size=32):
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    callback = EarlyStopping(monitor="loss", patience=10)
    history = model.fit(
        X_train,
        y_train,
        # Setting Batch Size to number of samples for Vanilla GD
        batch_size=X_train.shape[0],
        validation_data=(X_test, y_test),
        epochs=150,
        callbacks=[callback],
        verbose=0,
    )
    return history


gd_types = [
    "Gradient Descent",
    "Stochastic Gradient Descent",
    "Mini-Batch Gradient Descent",
    "Adagrad",
    "RMSProp",
    "Adam",
]

with st.sidebar:
    choice = st.selectbox("Optimizer:", options=gd_types)

    if (
        choice == "Gradient Descent"
        or choice == "Stochastic Gradient Descent"
        or choice == "Mini-Batch Gradient Descent"
    ):
        lr = st.slider(
            "Learning Rate:", min_value=0.01, max_value=1.00, value=0.01, step=0.01
        )
        if choice == "Mini-Batch Gradient Descent":
            batch_size = st.slider(
                "Batch Size:", min_value=1, max_value=100, value=50, step=10
            )
        else:
            batch_size = st.select_slider(
                "Batch Size:", [1, 2, 4, 8, 16, 32, 64], disabled=True
            )
        momentum = st.slider(
            "Momentum Factor:", min_value=0.01, max_value=1.00, value=0.01, step=0.01
        )
        nag = st.checkbox("Nesterov Accelerated Momentum")

    elif choice == "Adagrad":
        lr = st.slider(
            "Learning Rate:", min_value=0.01, max_value=1.00, value=0.1, step=0.01
        )
        batch_size = st.slider(
            "Batch Size:", min_value=1, max_value=100, value=50, step=10
        )

    elif choice == "RMSProp":
        lr = st.slider(
            "Learning Rate:", min_value=0.01, max_value=1.00, value=0.01, step=0.01
        )
        batch_size = st.slider(
            "Batch Size:", min_value=1, max_value=100, value=50, step=10
        )
        rho = st.slider(
            "Exponential Decay Rate:", min_value=0.1, max_value=1.0, value=0.9, step=0.1
        )

    elif choice == "Adam":
        lr = st.slider(
            "Learning Rate:", min_value=0.01, max_value=1.00, value=0.01, step=0.01
        )
        batch_size = st.slider(
            "Batch Size:", min_value=1, max_value=100, value=50, step=10
        )
        beta1 = st.slider(
            "Exponential Decay Rate for Moments:",
            min_value=0.1,
            max_value=1.0,
            value=0.9,
            step=0.1,
        )
        beta2 = st.slider(
            "Exponential Decay Rate for Variance:",
            min_value=0.01,
            max_value=1.00,
            value=0.99,
            step=0.01,
        )

st.title("Optimizers in Deep Learning")

st.write(
    "A Neural Network has been trained on the Breast Cancer Dataset. We monitor the convergence of different optimizers during the training."
)

st.subheader(choice)

if choice == "Gradient Descent":
    model = create_model()
    optimizer = SGD(learning_rate=lr, momentum=momentum, nesterov=nag)
    history = fit_model(
        model, optimizer, X_train, X_test, y_test, batch_size=X_train.shape[0]
    )
    st.pyplot(plot_loss(history))

elif choice == "Stochastic Gradient Descent":
    model = create_model()
    optimizer = SGD(learning_rate=lr, momentum=momentum, nesterov=nag)
    history = fit_model(model, optimizer, X_train, X_test, y_test, batch_size=1)
    st.pyplot(plot_loss(history))

elif choice == "Mini-Batch Gradient Descent":
    model = create_model()
    optimizer = SGD(learning_rate=lr, momentum=momentum, nesterov=nag)
    history = fit_model(
        model, optimizer, X_train, X_test, y_test, batch_size=batch_size
    )
    st.pyplot(plot_loss(history))

elif choice == "Adagrad":
    model = create_model()
    optimizer = Adagrad(learning_rate=lr)
    history = fit_model(
        model, optimizer, X_train, X_test, y_test, batch_size=batch_size
    )
    st.pyplot(plot_loss(history))

elif choice == "RMSProp":
    model = create_model()
    optimizer = RMSprop(learning_rate=lr, rho=rho)
    history = fit_model(
        model, optimizer, X_train, X_test, y_test, batch_size=batch_size
    )
    st.pyplot(plot_loss(history))

elif choice == "Adam":
    model = create_model()
    optimizer = Adam(learning_rate=lr, beta_1=beta1, beta_2=beta2)
    history = fit_model(
        model, optimizer, X_train, X_test, y_test, batch_size=batch_size
    )
    st.pyplot(plot_loss(history))


st.write("The dataset can be viewed below:")

st.dataframe(data=df, width=1000, height=200)
