
#pip install -U tf2onnx

import tf2onnx
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import numpy as np

iv4 = 'keras_h5/iv4_t3i2018_nf.h5'

rn50 = 'keras_h5/rn50_t3i2018_nf.h5'

onnx_iv4= 'onnx/iv4_i18nf_5runs_4_keras.onnx'
onnx_rn = 'onnx/rn50_i18nf_5runs_4_keras.onnx'

def create_onnx(path_model, path_onnx, name_model):
    
    if name_model=='rn50':
        spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
    elif name_model =='iv4':
        spec = (tf.TensorSpec((None, 299, 299, 3), tf.float32, name="input"),)
    else:
        print('Model not specified')
        spec= None

    model = tf.keras.models.load_model(path_model)
    
    _, _ = tf2onnx.convert.from_keras(model,
                    input_signature= spec, opset=None, custom_ops=None,
                    custom_op_handlers=None, custom_rewriter=None,
                    inputs_as_nchw=None, extra_opset=None,
                    shape_override=None, target=None, large_model=False, output_path=path_onnx)

def create_graph(path_model,path_graph,graph_name):
    model = tf.keras.models.load_model(path_model)
    # Convert Keras model to ConcreteFunction
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(
        tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()
    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-" * 60)
    print("Frozen model layers: ")
    for layer in layers:
        print(layer)
    print("-" * 60)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)
    # Save frozen graph to disk
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                    logdir=path_graph,
                    name=f"{graph_name}.pb",
                    as_text=False)
    # Save its text representation
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                    logdir=path_graph,
                    name=f"{graph_name}.pbtxt",
                    as_text=True)


#create_onnx(iv4,onnx_iv4,'iv4')

create_graph(rn50,'graphs/','rn50_i18t3nf')