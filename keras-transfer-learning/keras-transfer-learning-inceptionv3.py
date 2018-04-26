
# coding: utf-8

# In[1]:


import numpy as np

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model

from keras.optimizers import SGD, rmsprop

from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import tensorflow as tf
tf.test.gpu_device_name()


# In[3]:

# In this program, We'll demonstrate both transfer learning and fine-tuning. We can use either or both if we like.
# 
# 1. **Transfer learning:** freeze all but the penultimate layer and re-train the last Dense layer
# 2. **Fine-tuning:** un-freeze the lower convolutional layers and retrain more layers
# 
# Doing both, in that order, will ensure a more stable and consistent training. This is because the large gradient updates triggered by randomly initialized weights could wreck the learned weights in the convolutional base if not frozen. Once the last layer has stabilized (transfer learning), then we move onto retraining more layers (fine-tuning).

# In[4]:


im_width, im_height = 299, 299


# In[5]:


epochs = 50
batch_size = 32


# In[6]:


nb_train_samples = 20000
nb_valid_samples = 5000


# In[7]:


train_data_dir = 'data/cats-vs-dogs/train/'
valid_data_dir = 'data/cats-vs-dogs/valid/'


# In[8]:


# data prep
train_datagen = ImageDataGenerator(
    preprocessing_function = preprocess_input, 
    rotation_range = 30,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)
valid_datagen = ImageDataGenerator(
    preprocessing_function = preprocess_input, 
    rotation_range = 30,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)


# In[9]:


train_generator = train_datagen.flow_from_directory(
    directory = train_data_dir, 
    target_size = (im_width, im_height),
    batch_size = batch_size,
    class_mode='categorical',
    shuffle=True
)
valid_generator = valid_datagen.flow_from_directory(
    directory=valid_data_dir, 
    target_size = (im_width, im_height),
    batch_size = batch_size,
    class_mode='categorical',
    shuffle=True
)


# In[10]:


# setup base model from InceptionV3
base_model = InceptionV3(weights='imagenet', include_top=False) # exclude the top FC layer


# In[11]:


print(base_model.summary())


# In[12]:


for i, layer in enumerate(base_model.layers):
    print('Layer {0}: Name: {1}'.format(i, layer.name))


# In[13]:


# Add new last FC layer
top_model = base_model.output
top_model = GlobalAveragePooling2D()(top_model)
top_model = Dense(units=1024, activation='relu')(top_model) # new FC Layer, random init
predictions = Dense(units=2, activation='softmax')(top_model) # new softmax layer


# In[14]:


final_model = Model(inputs=base_model.input, outputs=predictions)


# In[15]:


# transfer learning
# Freeze all layers of the base model
for layer in base_model.layers:
    layer.trainable = False


# In[16]:


final_model.compile(optimizer=rmsprop(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])


# In[17]:


history_trf_learn = final_model.fit_generator(
    generator=train_generator,
    epochs=epochs,
    steps_per_epoch=nb_train_samples // batch_size,
    validation_data=valid_generator,
    validation_steps=nb_valid_samples // batch_size,
).history


# **Plotting Train vs Validation Loss:**

# In[18]:


# In[19]:


# setup fine-tuning
# freeze some of the bottom layers and retrain the remaining top layers
for layer in final_model.layers[:172]:
    layer.trainable=False
for layer in final_model.layers[172:]:
    layer.trainable=True


# When fine-tuning, it's important to lower the learning rate relative to the rate that was used when training from scratch (lr=0.0001), otherwise, the optimization could destabilize and the loss diverge.

# In[20]:


final_model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])


# In[21]:


history_fine_tune = final_model.fit_generator(
    generator=train_generator,
    epochs=epochs,
    steps_per_epoch=nb_train_samples // batch_size,
    validation_data=valid_generator,
    validation_steps=nb_valid_samples // batch_size,
).history


# In[23]:


final_model.save('model_weights/keras-transfer-learning-inceptionv3.h5')


# In[24]:


from keras.preprocessing import image
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input

from PIL import Image


# In[25]:


target_size = (229, 229) #fixed size for InceptionV3 architecture


# In[26]:


def predict(model, img, target_size):
    """Run model prediction on image
    Args:
    model: keras model
    img: PIL format image
    target_size: (w,h) tuple
    Returns:
    list of predicted labels and their probabilities
    """
    if img.size != target_size:
        img = img.resize(target_size)

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return preds[0]


# In[27]:


# In[28]:


model = load_model('model_weights/keras-transfer-learning-inceptionv3.h5')


# In[29]:


from glob import glob
img_cats = glob('data/cats-vs-dogs/train/cats/*.jpg')
img_dogs = glob('data/cats-vs-dogs/train/dogs/*.jpg')


# In[30]:


np.random.permutation(img_dogs[:5] + img_cats[:5])


# In[31]:


n_test = 8
in_imgs = np.random.permutation(img_dogs[:n_test//2] + img_cats[:n_test//2])
preds = [predict(model, Image.open(img), target_size) for img in in_imgs]



# In[ ]:




