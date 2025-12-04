import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.manifold import TSNE
import pandas as pd

INPUT_SIZE = (299, 299)

def labelled_unlabbeled_split_fpaths(x_train_path, c_train, n_labelled=100, n_unlabelled=None):
    '''
    Perform labelled/unlabelled split, whilst maintaining x_train represented as filepaths
    '''

    if n_unlabelled is None:
        # Labelled are first n_labelled points
        x_train_l, c_train_l = x_train_path[:n_labelled], c_train[:n_labelled]

        # Non-labelled are all the data-points
        x_train_u, c_train_u = x_train_path[n_labelled:], c_train[n_labelled:]
    else:
        # Otherwise, select randomly
        from random import randrange
        id_ub = len(x_train_path) - (n_labelled + n_unlabelled)
        id = randrange(id_ub)

        x_train_l, c_train_l = x_train_path[id : id+n_labelled], c_train[id : id+n_labelled]

        x_train_u, c_train_u = x_train_path[id+n_labelled : id+n_labelled+n_unlabelled], \
                               c_train[id+n_labelled : id+n_labelled+n_unlabelled]

    print('x_train_l length:', len(x_train_l))
    print('c_train_l shape:', c_train_l.shape)
    print('x_train_u length:', len(x_train_u))
    print('c_train_u shape:', c_train_u.shape)

    return x_train_l, c_train_l, x_train_u, c_train_u

def compute_tsne_embedding(x_data_paths, model, layer_ids, layer_names, batch_size=256):
    '''
    Compute tSNE latent space embeddings for specified layers of the DNN model
    '''

    h_l_list_agg = compute_activation_per_layer(x_data_paths, layer_ids, model,
                                                batch_size,
                                                aggregation_function=aggregate_activations)
    h_l_embedding_list = []

    for i, h_l in enumerate(h_l_list_agg):
        h_embedded = TSNE(n_components=2, n_jobs=4).fit_transform(h_l)
        h_l_embedding_list.append(h_embedded)
        print(layer_names[i])
    return h_l_embedding_list



def visualise_hidden_space(x_data_paths, c_data, c_names, layer_names, layer_ids, model, batch_size=256):

    # Compute tSNE embeddings
    h_l_embedding_list = compute_tsne_embedding(x_data_paths, model, layer_ids, layer_names, batch_size)

    # Create figure of size |n_concepts| * |n_layers|
    n_concepts = len(c_names)
    n_rows = n_concepts
    n_cols = len(h_l_embedding_list)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(3 * n_cols, 4 * n_rows))

    # Plot the embeddings of every layer, highlighting concept values
    for i, h_2 in enumerate(h_l_embedding_list):
        for j in range(1, n_concepts):
            ax = axes[j-1, i]
            ax.scatter(h_2[:, 0], h_2[:, 1], c=c_data[:, j], marker = '.')
            ax.set_title(layer_names[i], fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])

            if i == 0:
                ax.set_ylabel(c_names[j], fontsize=20)

    return fig


def aggregate_activations(activations):
    if len(activations.shape) == 4:
        score_val = np.mean(activations, axis=(1, 2))
    elif len(activations.shape) == 3:
        score_val = np.mean(activations, axis=(1))
    elif len(activations.shape) == 2:
        score_val = activations
    else:
        raise ValueError("Unexpected data dimensionality")

    return score_val

def preprocessing_input_torch(x):
    #x: (1,channels,width,height)
    #print(x.shape)
    x = np.float32(x) / 255.0
    mean = [0.485, 0.456, 0.406]
    std =  [0.229, 0.224, 0.225]
    for chanel in range(x.shape[2]):
        x[:,:,chanel] = (x[:,:,chanel] - mean[chanel])/ std[chanel]
    return x



def load_img(img_path):

    x_data = Image.open(img_path).convert('RGB')
    img = np.array(x_data.resize(INPUT_SIZE, Image.BILINEAR))
    #img = np.float32(img) / 255.0
    
    #img_array=np.reshape(img,(1,299,299,3))
    x_data = preprocessing_input_torch(img)
    #print('img loaded',x_data.shape)
    return x_data



def load_batch(img_paths):
    '''
    Load a batch of images using load_img()
    :param img_paths_and_train_flag: list of pairs of (img_path, train_flag)
    '''

    x_data = []

    for img_path in img_paths:
        x_data.append(load_img('../../../isic-data/edraAtlas/allimages/'+img_path+'.jpg'))

    x_data = np.array(x_data)

    return x_data

def flatten_activations(x_data):
    '''
    Flatten all axes except the first one
    '''

    if len(x_data.shape) > 2:
        n_samples = x_data.shape[0]
        shape = x_data.shape[1:]
        flattened = np.reshape(x_data, (n_samples, np.prod(shape)))
    else:
        flattened = x_data

    return flattened

def compute_activations_from_paths(model, x_data_paths, batch_size):

    batch_size = batch_size
    n_samples = len(x_data_paths)
    n_batches = math.ceil(n_samples / batch_size)
    hidden_features = []

    for i in range(n_batches):
        start = batch_size * i
        end = min(n_samples, batch_size * (i + 1))
        paths = x_data_paths[start:end]
        x_data = load_batch(paths)
        #print(x_data.shape)
        batch_hidden_features = model.predict(x_data)
        hidden_features.append(batch_hidden_features)

        print("Processing batch ", str(i), " of ", str(n_batches))

    hidden_features = np.concatenate(hidden_features)

    return hidden_features


def compute_activation_per_layer(x_data_paths, layer_ids, model, batch_size=128,
                                 aggregation_function=flatten_activations):
    '''
    Compute activations of x_data for 'layer_ids' layers
    For every layer, aggregate values using 'aggregation_function'

    Returns a list of size |layer_ids|, in which element L[i] is the activations
    computed from the model layer model.layers[layer_ids[i]]
    '''

    hidden_features_list = []

    for layer_id in layer_ids:
        print('layer_id',layer_id)
        # Compute and aggregate hidden activtions
        #hidden_features = model.get_layer_activations(self, x_data, layer_id)
        
        output_layer = model.layers[layer_id]
        reduced_model = tf.keras.Model(inputs=model.inputs, outputs=[output_layer.output])
        hidden_features = compute_activations_from_paths(reduced_model, x_data_paths, batch_size)

        flattened = aggregation_function(hidden_features)

        hidden_features_list.append(flattened)

    return hidden_features_list

def plot_summary(concept_model):

    # For decision trees, also save their plots
    if concept_model.clf_type == "DT":

        from sklearn.tree import plot_tree
        import matplotlib.pyplot as plt

        dt = concept_model.clf
        fig, ax = plt.subplots(figsize=(120, 120))  # whatever size you want

        plot_tree(dt,
                  ax=ax,
                  feature_names=concept_model.concept_names,
                  filled=True,
                  rounded=True,
                  proportion=True,
                  precision=2,
                  class_names=concept_model.class_names,
                  impurity=False)

        plt.show()
        return(fig)

    elif concept_model.clf_type == 'LR':
        coeffs = concept_model.get_sorted_coef()
        print("LR Coefficients: ", coeffs)
        return None