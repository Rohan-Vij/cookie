import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import streamlit as st
import plotly.graph_objects as go

# Create the Streamlit web app
st.title("Cookie Quality Predictor")

# Explanation text
st.markdown("This web app predicts the quality of a cookie based on its attributes. The survey data represents cookies "
            "with attributes such as Chocolatey, Salty, Soft, Sweet, and Bitter. The output value 1 indicates a good "
            "cookie, while 0 indicates a bad cookie. You can adjust the number of hidden layers in the neural network "
            "using the slider on the left.")

# Define the survey data with a missing value
survey = np.array([
    [0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 1]
])

# Split the data into features and labels
X = survey[:, 0:5]
y = survey[:, 5]

# Display training data
st.header("Training Data")

st.markdown("The training data consists of two samples of cookies, each represented by their attributes such as "
            "Chocolatey, Salty, Soft, Sweet, and Bitter. The last column 'Output (Good cookie?)' represents the "
            "label, where 1 indicates a good cookie and 0 indicates a bad cookie.")

st.markdown("As humans, we have our own neural networks inside of us with billions of neurons and trilions of connections - our brain! From quickly taking a glance at the data, we can observe that this data is pretty simple. While it has 5 features (or attributes of the cookies), the only one that matters to whether or not the cookie is good is whether it is chocolatey. If the cookie is chocolatey, it is good - and vice versa.")

st.markdown("This data is simple for two reasons - it is easy for humans to understand, so we can logically verify the results of the model. It also keeps the number of datapoints down to just two, as that is all that is needed for the model to understand such a simple data pattern.")

training_df = pd.DataFrame(survey, columns=[
                           'Chocolatey', 'Salty', 'Soft', 'Sweet', 'Bitter', 'Output (Good cookie?)'])


tab1, tab2 = st.tabs(["Human Data", "Binary Data"])

with tab1:
    st.subheader("Human Data")
    st.markdown(
        "The human-readable data represents the training data with 'Yes' and 'No' instead of binary values.")
    st.write(training_df.replace({1: "Yes", 0: "No"}))

with tab2:
    st.subheader("Binary Data")
    st.markdown(
        "The binary data represents the training data with binary values (0 and 1).")
    st.write(training_df)


# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Function to make predictions


def predict_cookie_quality(features):
    features = scaler.transform([features])
    prediction = mlp.predict(features)[0]
    return "Good cookie!" if prediction == 1 else "Bad cookie!"


# Sidebar for adjusting the number of hidden layers
st.sidebar.header("Model Configuration")
num_hidden_layers = st.sidebar.slider(
    "Number of Hidden Layers", min_value=1, max_value=5, value=1)

# Train the model with the selected number of hidden layers
hidden_layer_sizes = tuple([5] * num_hidden_layers)  # 5 nodes per hidden layer
mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                    activation='relu',
                    max_iter=1000,
                    random_state=1
                    )
mlp.fit(X, y)


# Display neural network diagram
st.header("Neural Network")

st.markdown("This model is an MLP classifier from scikit-learn's Python library. MLP stands for multi-layer perceptron. Multilayer simply means it has more than an input and output layer - it has what are called \"hidden layers\" inbetween, where the data is transformed and processed. Perceptron is a fancy word for neuron, which is a node in the neural network.")

st.markdown("The \"input layer\" on the left side is where the binary (0 and 1) values of the cookie are passed. Each input node then connects to the 5 subsequent nodes in layer 1.")

st.markdown("Each connection between nodes has a different weight, or importance. This weight can be positive (green) or negative (red). This plotly graph was made so that the higher the absolute value of the weight of a connection, the thicker the line and more intense the color.")

st.markdown("You can change the number of hidden layers by moving around the slider on the right. As the pattern the model has to recongize is very simple, the accuracy of the model will not change - it will be able to evalute a cookie correctly with 1 hidden layer and with 5.")

st.markdown("These hidden layers are often called a \"black box\" because we don't know what is going on inside of them. We can only see and understand the input and output of the model, but not the intermediate steps. However, we can see that the chocolate input node has the most connections with the most weight out of all of the other input nodes, meaning that we can infer the model knows it is the most important.")

fig = go.Figure()

# Draw connecting lines between nodes (under nodes)


def draw_line(start_x, start_y, end_x, end_y, weight=None):
    if weight is not None:
        # Set color based on coefficient value
        color_scale = 255
        color_value = int((weight + 1) / 2 * color_scale)
        if color_value > color_scale:
            color_value = color_scale
        elif color_value < 0:
            color_value = 0

        if weight >= 0:
            # Green for positive coefficients
            line_color = f'rgb(0, {color_value}, 0)'
        else:
            # Red for negative coefficients
            line_color = f'rgb({color_value + 150}, 0, 0)'
        line_width = np.abs(weight) * 5
    else:
        line_color = 'white'
        line_width = 2
    fig.add_shape(type="line", x0=start_x, y0=start_y, x1=end_x, y1=end_y,
                  line=dict(color=line_color, width=line_width), layer='below')  # Set layer to 'below' to render lines under nodes


# Connect input layer to the first hidden layer
for i in range(5):
    for j in range(5):
        draw_line(0, 10 - i * 2, 1, 10 - j * 2, mlp.coefs_[0][i][j])

# Connect hidden layers
for layer in range(num_hidden_layers - 1):
    for i in range(5):
        for j in range(5):
            draw_line(layer + 1, 10 - i * 2, layer + 2, 10 -
                      j * 2, mlp.coefs_[layer + 1][i][j])

# Connect last hidden layer to the output layer
for j in range(5):
    draw_line(num_hidden_layers, 10 - j * 2, num_hidden_layers +
              1, 6, mlp.coefs_[num_hidden_layers][j][0])

# Input layer
input_x = [0] * 5
input_y = [10, 8, 6, 4, 2]
input_features = ['Chocolatey', 'Salty', 'Soft', 'Sweet', 'Bitter']
for i, (x, y, feature) in enumerate(zip(input_x, input_y, input_features)):
    fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers+text', name='Input Layer',
                             marker=dict(size=60, color='yellow',
                                         line=dict(width=2, color='white')),
                             text=feature,
                             textposition="middle center", textfont=dict(color='black')))

# Hidden layers
for layer in range(num_hidden_layers):
    hidden_x = [layer + 1] * 5
    hidden_y = [10, 8, 6, 4, 2]
    for i, (x, y) in enumerate(zip(hidden_x, hidden_y)):
        node_label = f'Node {i + 1}'
        fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers+text', name=f'Hidden Layer {layer + 1}',
                                 marker=dict(size=40, color='white',
                                             line=dict(width=2, color='black')),
                                 text=node_label,
                                 textposition="middle center", textfont=dict(color='black')))

# Output layer
output_x = [num_hidden_layers + 1]
output_y = [6]
fig.add_trace(go.Scatter(x=output_x, y=output_y, mode='markers+text', name='Output',
                         marker=dict(size=60, color='black',
                                     line=dict(width=2, color='white')),
                         text='Output', textposition="middle center", textfont=dict(color='white')))

# Add text labels above each layer
for layer in range(num_hidden_layers + 1):
    if layer == 0:
        text = "Input Layer"
    else:
        text = f"Layer {layer}"
    fig.add_annotation(
        x=layer, y=11,  # Position of the label
        xref='x', yref='y',
        text=text,  # Text label for the layer
        showarrow=False,  # No arrow to the text
        font=dict(size=14, color='white'),
    )

fig.update_layout(showlegend=False, xaxis=dict(showline=False, showticklabels=False, zeroline=False),
                  yaxis=dict(showline=False, showticklabels=False,
                             zeroline=False),
                  height=600, width=600)
st.plotly_chart(fig)

# Explanation text for custom test set
st.markdown("You can also enter your own custom test sample below to see the prediction for that sample. Adjust the "
            "features of the sample using the checkboxes and click the 'Predict' button.")

# Allow the user to enter their own data for custom test
st.header("Custom Test Sample")

st.markdown("Enter 'Yes' or 'No' for each feature to represent the presence or absence of that attribute in the "
            "cookie sample. Then, click the 'Predict' button to see if it's a good or bad cookie.")

features = []
for i, feature in enumerate(input_features):
    feature_value = st.checkbox(feature, key=f"feature_{i}")
    features.append(int(feature_value))

# Predict and display result for custom test sample
if st.button("Predict"):
    st.subheader("Prediction Result")
    prediction = predict_cookie_quality(features)
    st.write(f"Prediction: {prediction}")
