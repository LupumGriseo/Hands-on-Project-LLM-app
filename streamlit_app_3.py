import streamlit as st
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Function to train and cache the model
@st.cache
def train_model():
    clf = RandomForestClassifier()
    clf.fit(X, y)
    return clf

# Function to classify Iris species
def classify_iris(model, sepal_length, sepal_width, petal_length, petal_width):
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    return iris.target_names[prediction][0]

def main():
    st.title("Iris Species Classifier")

    # Initializing session state
    if 'sepal_length' not in st.session_state:
        st.session_state['sepal_length'] = 5.4
    if 'sepal_width' not in st.session_state:
        st.session_state['sepal_width'] = 3.4
    if 'petal_length' not in st.session_state:
        st.session_state['petal_length'] = 1.6
    if 'petal_width' not in st.session_state:
        st.session_state['petal_width'] = 0.4

    # Sidebar for input features
    with st.sidebar:
        st.subheader("Input Features")
        st.session_state.sepal_length = st.slider("Sepal Length", min_value=4.0, max_value=8.0, value=st.session_state.sepal_length)
        st.session_state.sepal_width = st.slider("Sepal Width", min_value=2.0, max_value=4.5, value=st.session_state.sepal_width)
        st.session_state.petal_length = st.slider("Petal Length", min_value=1.0, max_value=7.0, value=st.session_state.petal_length)
        st.session_state.petal_width = st.slider("Petal Width", min_value=0.1, max_value=2.5, value=st.session_state.petal_width)

    # Main section
    st.subheader("Predicted Species")
    if st.button("Classify"):
        model = train_model()  # Using the cached model
        species = classify_iris(model, st.session_state.sepal_length, st.session_state.sepal_width, st.session_state.petal_length, st.session_state.petal_width)
        st.success(f"The Iris is predicted to be a {species}")

if __name__ == "__main__":
    main()