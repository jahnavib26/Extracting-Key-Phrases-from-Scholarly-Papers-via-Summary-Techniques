
import time
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.utils.vis_utils import plot_model

# Define all file paths and parameters in dictionaries for easier management
test_sets = {
    'semeval_220_first_paragraphs': {
        'size': 211,
        'folder': '220',
    },
    'semeval_400_first_3_paragraphs': {
        'size': 732,
        'folder': '400',
    },
    'nus_summarization': {
        'size': 211,
        'folder': 'summarization_experiment',
    },
    'acm_summarization': {
        'size': 2304,
        'folder': 'summarization_experiment',
    }
}

# Function to configure file paths based on test set
def configure_paths(test_set):
    if test_set in test_sets:
        details = test_sets[test_set]
        base_path = f'data\preprocessed_data\first_paragraphs_fulltext\{details["folder"]}\'
        return {
            'x_test': f'{base_path}x_SEMEVAL_FIRST_PARAGRAPHS_FULLTEXT_TEST_data_preprocessed.hdf',
            'y_test': f'{base_path}y_SEMEVAL_FIRST_PARAGRAPHS_FULLTEXT_TEST_data_preprocessed',
            'x': f'{base_path}x_SEMEVAL_FIRST_PARAGRAPHS_FULLTEXT_preprocessed_TEXT',
            'y': f'{base_path}y_SEMEVAL_FIRST_PARAGRAPHS_FULLTEXT_preprocessed_TEXT',
            'size': details['size']
        }
    else:
        print("Invalid test set specified.")
        sys.exit(1)

# Example usage of the function
paths = configure_paths('semeval_220_first_paragraphs')

# Load and operate on the model
model = load_model('bi_lstm_crf_model.h5')
plot_model(model, to_file='model_architecture.png', show_shapes=True)

start_time = time.time()
# Placeholder for the model training or testing code
# Here we might perform some operations such as model evaluation or prediction
results = model.evaluate(x_test, y_test)
print("Model evaluation results:", results)

# Plotting operations
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.savefig('model_accuracy_over_epochs.png')

print(f"Execution time: {time.time() - start_time:.2f} seconds")
