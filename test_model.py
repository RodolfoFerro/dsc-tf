import tensorflow as tf
import numpy as np

from source.model_utils import load_model


model = load_model('models/checkpoint')

examples = [
    "The movie was great!",
    "This is not good movie. For me, it is bad.",
    "The movie was terrible...",
    "Cool movie."
]

results = model.predict(examples)
print(np.round(results))