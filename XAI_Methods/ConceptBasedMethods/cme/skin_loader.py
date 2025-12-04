import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from sklearn.metrics import roc_auc_score
from numpy import genfromtxt


from utils import flatten_activations, compute_activations_from_paths, compute_activation_per_layer, load_img, load_batch,preprocessing_input_torch
INPUT_SIZE = (299, 299)


def load_data(path_npy):
    
    c_train = genfromtxt(path_npy+'X_train_concepts.csv', delimiter=',')
    #c_val = genfromtxt(path_npy+'X_val_concepts.csv', delimiter=',')
    y_train_p = genfromtxt(path_npy+'y_train.csv', delimiter=',')
    #y_val = genfromtxt(path_npy+'y_val.csv', delimiter=',')
    
    X_attr_train= c_train #np.concatenate([c_train,c_val])
    y_train= y_train_p #np.concatenate([y_train_p,y_val])
    
    #X_attr_train = genfromtxt(path_npy+'X_train_concepts.csv', delimiter=',')
    #y_train = genfromtxt(path_npy+'y_train.csv', delimiter=',')
    
    X_attr_test = genfromtxt(path_npy+'X_test_concepts.csv', delimiter=',')
    y_test = genfromtxt(path_npy+'y_test.csv', delimiter=',')
    attr_names = genfromtxt(path_npy+'concept_names.csv', delimiter=',', dtype='unicode')
    img_names_train = genfromtxt(path_npy+'img_names_train.csv', delimiter=',',dtype='unicode')
    #img_names_val = genfromtxt(path_npy+'img_names_val.csv', delimiter=',',dtype='unicode')
    #img_names_train = np.concatenate([img_names_train,img_names_val])
    img_names_test = genfromtxt(path_npy+'img_names_test.csv', delimiter=',', dtype='unicode')
    
    return img_names_train, X_attr_train, y_train, img_names_test, X_attr_test, y_test, attr_names




def predict_image(img_array):
    img_array = preprocessing_input_torch(img_array)
    prediction = softmax(model.predict(img_array))
    cls = prediction.argmax()
    print(prediction, cls )
    return cls
    
label_to_class = {
    'melanoma': 1,
    'non-melanoma':    0
}
class_to_label = {v: k for k, v in label_to_class.items()}

def _plot(model, img_path, cls_true):
    """plot original image, heatmap from cam and superimpose image"""
    
    img_ori = Image.open(img_path).convert('RGB')
    img = np.array(img_ori.resize((299,299), Image.BILINEAR))
    #img = np.float32(img) / 255.0
    
    img_array=np.reshape(img,(1,299,299,3))
    #img_array_transp = np.transpose(img_array,[0,3,1,2])
 
    cls_pred = predict_image(img_array)
    
    fig, axs = plt.subplots(ncols=2, figsize=(10, 2))

    axs[0].imshow(img_ori)
    axs[0].set_title('original image')
    axs[0].axis('off')

    plt.suptitle('True label: ' + class_to_label[cls_true] + ' / Predicted label : ' + class_to_label[cls_pred])
    plt.tight_layout()
    plt.show()
    
def softmax(x, axis_n = 1):
    f = np.exp(x)/np.sum(np.exp(x), axis = axis_n, keepdims = True)
    return f

def get_data_from_df(args, filename="edra_test.csv",split= 'test'):
    df = pd.read_csv(args['path_npy']+filename,sep=';')
    if split == 'full':
        sub_df = df.copy()
    else:
        sub_df = df.loc[df['split'] == split]
    y = sub_df['label'].values
    sub_df['label']=sub_df.label.astype(str)
    # if your image names have no extension
    sub_df['image'] = sub_df['image'].astype(str) + '.jpg' 

    datagen = ImageDataGenerator( preprocessing_function=preprocessing_input_torch)
    dset = datagen.flow_from_dataframe(sub_df,
                                                  directory=args['path_imgs'],
                                                  x_col="image",
                                                  y_col="label",
                                                  target_size=INPUT_SIZE,
                                                  batch_size=32,
                                                  class_mode='binary',
                                                  shuffle= False
                                                  )
    return dset,y


def get_data_from_df_prev(args, filename="edra_test.csv",mode= 'test'):
    df = pd.read_csv(args['path_npy']+filename)
    if mode =='alceu':
        df = pd.read_csv(args['path_npy']+filename,sep=';')
    y = df['label'].values
    df['label']=df.label.astype(str)
    # if your image names have no extension
    df['image'] = df['image'].astype(str) + '.jpg' 

    datagen = ImageDataGenerator( preprocessing_function=preprocessing_input_torch)
    dset = datagen.flow_from_dataframe(df,
                                                  directory=args['path_imgs'],
                                                  x_col="image",
                                                  y_col="label",
                                                  target_size=INPUT_SIZE,
                                                  batch_size=32,
                                                  class_mode='binary',
                                                  shuffle= False
                                                  )
    return dset,y

def model_predict(model,args,mode='test'):
    set_d= None
    if mode == 'test':
        set_d,y= get_data_from_df(args,filename="derm7pt_full.csv",split = 'test')
    if mode == 'train':
        set_d,y= get_data_from_df(args,filename="derm7pt_full.csv", split = 'train')
    if mode == 'full':
        set_d,y= get_data_from_df(args,filename="derm7pt_full.csv", split = 'full')
    if mode == 'alceu':
        set_d,y_true= get_data_from_df(args,filename="derm7pt_dermato.csv",mode = 'alceu')
    y_pred_log = softmax(model.predict(set_d))
    y_pred = y_pred_log[:,1]#.argmax(axis=1)
    y_pred_cls = y_pred_log.argmax(axis=1)
    auc = roc_auc_score(y, y_pred)
    return auc,y_pred_cls,y
    
    
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


def get_layer_activations(model, x_data, layer_id):

        if layer_id >= 0:
            out_layer = -(self.n_layers - layer_id)
        else:
            out_layer = layer_id

        with torch.no_grad():

            self.model.eval()

            x_data_t = torch.from_numpy(x_data)

            # Assumes you need to set a property variable
            if self.use_gpu:
                input_var = torch.autograd.Variable(x_data_t).cuda()
            else:
                input_var = torch.autograd.Variable(x_data_t).to("cpu")

            self.model.out_layer = out_layer

            if out_layer == -1:
                activations = self.model(input_var)[0]
            else:
                activations = self.model(input_var)

            if self.use_gpu:
                activations = activations.cpu().numpy()
            else:
                activations = activations.numpy()

        return activations
    
