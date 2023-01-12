# %%
# 1. Import necessary packages
from tensorflow_examples.models.pix2pix import pix2pix
from IPython.display import clear_output
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import cv2


# %% Data Preparation
root_path = r"F:\Python Test 2\STEP_AI01_Arif_Rabbani_assesment_4\data-science-bowl-2018"
train_path = r"F:\Python Test 2\STEP_AI01_Arif_Rabbani_assesment_4\data-science-bowl-2018\data-science-bowl-2018-2\train"
test_path = r"F:\Python Test 2\STEP_AI01_Arif_Rabbani_assesment_4\data-science-bowl-2018\data-science-bowl-2018-2\test"


# %%
# 2.2 Prepare empty list
image = []
masks = []

# 2.3 Load images using opencv
image_dir = os.path.join(train_path,'inputs')
for image_file in os.listdir(image_dir):
    img = cv2.imread(os.path.join(image_dir,image_file))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(128,128))
    image.append(img)


# 2.4 Load masks using opencv

masks_dir = os.path.join(train_path,'masks')
for masks_file in os.listdir(masks_dir):
    mask = cv2.imread(os.path.join(masks_dir,masks_file))
    mask = cv2.cvtColor(mask,cv2.COLOR_BGR2RGB)
    mask = cv2.resize(mask,(128,128))
    masks.append(mask)


# %%
# 2.5 Convert list of np array into a np array

images_np = np.array(image)
masks_np = np.array(masks)

# %%
# 3 Data Preprocessing
# 3.1 expand masks dimension
#masks_np_exp = np.expand_dims(masks_np,axis=-1)
# Check the mask output
print(np.unique(masks_np_exp[0]))

# %%
# 3.2 Convert mask values from [0,255] into [0,1]
converted_masks = np.round(masks_np/255.0).astype(np.int64)
# Check the mask output
print(np.unique(converted_masks[0]))

# %%
# 3.3 Normalize the images
converted_images = images_np/255.0

# %%
# 4. Perform train-test split
from sklearn.model_selection import train_test_split
SEED = 42
X_train,X_test,y_train,y_test = train_test_split(converted_images,converted_masks,test_size=0.2,random_state=SEED)

# %%
# 5. Convert the numpy arrays into tensor slices
X_train_tensor = tf.data.Dataset.from_tensor_slices(X_train)
X_test_tensor = tf.data.Dataset.from_tensor_slices(X_test)
y_train_tensor = tf.data.Dataset.from_tensor_slices(y_train)
y_test_tensor = tf.data.Dataset.from_tensor_slices(y_test)

# %%
#6 Combine the images and masks using the zip method
train_dataset = tf.data.Dataset.zip((X_train_tensor,y_train_tensor))
test_dataset = tf.data.Dataset.zip((y_train_tensor,y_test_tensor))

# %%
# 7. Define mini pipeline for data augmentation through a single layer subclassing
TRAIN_LENGTH = len(train_dataset)
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 16
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

class Augment(tf.keras.layers.Layer):
  def __init__(self, seed=42):
    super().__init__()
    # both use the same seed, so they'll make the same random changes.
    self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
    self.augment_labels = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)

  def call(self, inputs, labels):
    inputs = self.augment_inputs(inputs)
    labels = self.augment_labels(labels)
    return inputs, labels


# %%
# 8. Build the input pipeline, applying the augmentation after batching the inputs
train_batches = (
    train_dataset
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .repeat()
    .map(Augment())
    .prefetch(buffer_size=tf.data.AUTOTUNE))

test_batches = test_dataset.batch(BATCH_SIZE)
# %%
# 9. Visualize an image example and its corresponding mask from the dataset
def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()

# %%
for images, masks in train_batches.take(4):
  sample_image, sample_mask = images[0], masks[0]
  display([sample_image, sample_mask])

# %%
#10. Model development
#10.1. Use a pretrained model as the feature extractor
base_model = keras.applications.MobileNetV2(input_shape=[128,128,3],include_top=False)
base_model.summary()
#10.2. Use these activation layers as the outputs from the feature extractor (some of these outputs will be used to perform concatenation at the upsampling path)
layer_names = [
    'block_1_expand_relu',      #64x64
    'block_3_expand_relu',      #32x32
    'block_6_expand_relu',      #16x16
    'block_13_expand_relu',     #8x8
    'block_16_project'          #4x4
    ]

base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

#10.3. Instantiate the feature extractor
down_stack = keras.Model(inputs=base_model.input,outputs=base_model_outputs)
down_stack.trainable = False

#10.4. Define the upsampling path
up_stack = [
    pix2pix.upsample(512,3),    #4x4 --> 8x8
    pix2pix.upsample(256,3),    #8x8 --> 16x16
    pix2pix.upsample(128,3),    #16x16 --> 32x32
    pix2pix.upsample(64,3)      #32x32 --> 64x64
]

#10.5. Use functional API to construct the entire U-net

def unet(output_channels:int):
    inputs = keras.layers.Input(shape=[128,128,3])
    #Downsample through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    #Build the upsampling path and establish the concatenation
    for up, skip in zip(up_stack,skips):
        x = up(x)
        concat = keras.layers.Concatenate()
        x = concat([x,skip])
    
    #Use a transpose convolution layer to perform the last upsampling, this will become the output layer
    last = keras.layers.Conv2DTranspose(filters=output_channels,kernel_size=3,  strides=2,padding='same') #64x64 --> 128x128
    outputs = last(x)

    model = keras.Model(inputs=inputs,outputs=outputs)

    return model

# %%
#10.6. Use the function to create the model
OUTPUT_CHANNELS = 2
model = unet(OUTPUT_CHANNELS)
model.summary()
keras.utils.plot_model(model)

# %%
#11. Compile the model
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',loss=loss,metrics=['accuracy'])

# %%
#12. Create functions to show predictions
def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask,axis=-1)
    pred_mask = pred_mask[...,tf.newaxis]
    return pred_mask[0]
def show_predictions(dataset=None,num=1):
    if dataset:
        for image,mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0],mask[0],create_mask(pred_mask)])
    else:
        display([sample_image,sample_mask,create_mask(model.predict(sample_image[tf.newaxis,...]))])

show_predictions()

# %%
#13. Create a callback function to make use of the show_predictions function
class DisplayCallback(keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs=None):
        clear_output(wait=True)
        show_predictions()
        print('\nSample prediction after epoch {}\n'.format(epoch+1))

# %%
#14. Model training
EPOCHS = 5
VAL_SUBSPLITS = 5
VALIDATION_STEPS = len(train_dataset) // BATCH_SIZE // VAL_SUBSPLITS
history = model.fit(train_batches,validation_data=test_batches,validation_steps=VALIDATION_STEPS,epochs=EPOCHS,steps_per_epoch=STEPS_PER_EPOCH,callbacks=[DisplayCallback()])

# %%
#15. Model deployment
show_predictions(test_batches,3)

