import tensorflow as tf
import datetime
import pandas as pd
import time
# import wandb
# from wandb.keras import WandbCallback
from collections import Counter
from tqdm import tqdm
import os

from model import Model


def color_conversion(x):
    # Outputs a tensor of the same shape as the images tensor, containing the YUV value of the pixels.
    # The output is only well defined if the value in images are in [0,1].<----- So you have to rescale before
    return tf.image.rgb_to_yuv(x) / 255.


# %%
NUM_EPOCHS = 1000
MAX_PATIENCE = 20
BATCH_SIZE = 32
PATCH_SIZE = 96
STRIDE = 96
LR = 1e-7


# wandb.init(project="coulotte", entity="steviedale")

# wandb.config = {
#     'batch_size': BATCH_SIZE,
#     'patch_size': PATCH_SIZE,
#     'stride': STRIDE,
#     'learning_rate': LR,
#     'description': "running training on phase 2 dataset + GBG + macbook_2021"
# }

df = pd.read_csv('../dataset/dataframes/10k.csv')
live_df = df[df['label'] == 'live']
screen_df = df[df['label'] == 'screen']
live_df = live_df.sample(frac=1)
screen_df = screen_df.sample(frac=1)
live_split = int(len(live_df) * 0.8)
screen_split = int(len(screen_df) * 0.8)
live_train_df = live_df[:live_split]
screen_train_df = screen_df[:screen_split]
live_valid_df = live_df[live_split:]
screen_valid_df = screen_df[screen_split:]
train_df = pd.concat([live_train_df, screen_train_df])
valid_df = pd.concat([live_valid_df, screen_valid_df])

print(f"train: {len(train_df['original_path'].unique())}")
print(f"train_live: {len(live_train_df['original_path'].unique())}")
print(f"train_screen: {len(screen_train_df['original_path'].unique())}")
print(f"valid: {len(valid_df['original_path'].unique())}")
print(f"valid_live: {len(live_valid_df['original_path'].unique())}")
print(f"valid_screen: {len(screen_valid_df['original_path'].unique())}")

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(data_format=tf.keras.backend.image_data_format(), 
                                                                preprocessing_function=color_conversion)
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    target_size=(96, 96),
    x_col='path',
    y_col='label',
    classes=('live', 'screen'),
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode='categorical')

valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(data_format=tf.keras.backend.image_data_format(), 
                                                                preprocessing_function=color_conversion)
valid_generator = valid_datagen.flow_from_dataframe(
    dataframe=valid_df,
    target_size=(96, 96),
    x_col='path',
    y_col='label',
    classes=('live', 'screen'),
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode='categorical')

for df in train_df, valid_df:
    print(len(df))
    for path in tqdm(df['path']):
        assert(os.path.exists(path))

### Calculate Class Weights
counter = Counter(train_generator.classes)
max_val = float(max(counter.values()))
class_weights_train = {class_id : max_val/num_images for class_id, num_images in counter.items()}
print(class_weights_train)

counter = Counter(valid_generator.classes)
max_val = float(max(counter.values()))
class_weights_valid = {class_id : max_val/num_images for class_id, num_images in counter.items()}
print(class_weights_valid)

### Load Model
model = Model.create_model(input_shape=(96, 96, 3), num_classes=2, activation="selu")
model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=LR), metrics=["accuracy", "AUC"])
model.summary()

### Define Callbacks
saved_weights_dir = 'saved_weights'
if not os.path.exists(saved_weights_dir):
	os.mkdir(saved_weights_dir)

best_model_save_path = os.path.join(saved_weights_dir, 'epoch_{epoch:02d}_val_loss_{val_loss:.2f}.hdf5')
checkpoint = tf.keras.callbacks.ModelCheckpoint(best_model_save_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

log_dir = os.path.join(saved_weights_dir, "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=MAX_PATIENCE, verbose=1)

# callbacks_list = [checkpoint, tensorboard_callback, earlystopping, WandbCallback()]
callbacks_list = [checkpoint, tensorboard_callback, earlystopping]

### Train Model
start_time = time.time()
print(f'Start time: {round(start_time)}')

train_generator.on_epoch_end()
valid_generator.on_epoch_end()
model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=NUM_EPOCHS,
        validation_data=valid_generator,
        validation_steps=len(valid_generator),
        class_weight=class_weights_train,
        callbacks=callbacks_list)

end_time = time.time()
print(f"Duration: {round(end_time-start_time)}")

### Save Model
print("[INFO] dumping architecture and weights to file...")
final_model_save_path = os.path.join(saved_weights_dir, 'final_model.hdf5')
model.save(final_model_save_path)


