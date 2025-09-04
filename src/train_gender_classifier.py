import numpy as np
import pandas as pd
from utils.separate_date_articulator_that_is_new import return_emotions_mood_weather_mixer_combinations
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from utils.datasets import DataManager
from models.cnn import mini_XCEPTION, model_allofasudden_that_uses_tensorflow
from utils.data_augmentation import ImageGenerator
from utils.datasets import split_imdb_data

# parameters
batch_size = 32
num_epochs = 1000
validation_split = .2
do_random_crop = False
patience = 100
num_classes = 2
dataset_name = 'imdb'
input_shape = (64, 64, 1)
if input_shape[2] == 1:
    grayscale = True
images_path = '../datasets/imdb_crop/'
log_file_path = '../trained_models/gender_models/gender_training.log'
trained_models_path = '../trained_models/gender_models/gender_mini_XCEPTION'

# data loader
data_loader = DataManager(dataset_name)
ground_truth_data = data_loader.get_data()
train_keys, val_keys = split_imdb_data(ground_truth_data, validation_split)
print('Number of training samples:', len(train_keys))
print('Number of validation samples:', len(val_keys))
image_generator = ImageGenerator(ground_truth_data, batch_size,
                                 input_shape[:2],
                                 train_keys, val_keys, None,
                                 path_prefix=images_path,
                                 vertical_flip_probability=0,
                                 grayscale=grayscale,
                                 do_random_crop=do_random_crop)

# onigiri - as of 2025
df_weather_mood = DataManager("onigiri")
all_possible_combinations_input, y_true = return_emotions_mood_weather_mixer_combinations(df_weather_mood, batch_size,num_epochs,patience)
all_possible_combinations_input = all_possible_combinations_input.to_numpy()
y_true = y_true.to_numpy()

mood_model = model_allofasudden_that_uses_tensorflow(
    sequence_length=all_possible_combinations_input.shape[1],
    face_front_pixel=all_possible_combinations_input.shape[2],
    face_back_pixel=all_possible_combinations_input.shape[3],
    in_channels=all_possible_combinations_input.shape[4],
    out_features=y_true.shape[1] if y_true.ndim > 1 else 1,
    number_of_conv3d_layers=3,
    num_weather_types=10,
    conv3d_channels=32,
    fc_features=128,
    spatial_kernel_size=3,
    temporal_kernel_size=3
)

mood_model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["mae"]
)

mood_model.summary()

# ---- 3) Callbacks (match the style from your example) ----
# fill these in (same variable names you used before)
patience = 10
log_file_path = "mood_train_log.csv"
trained_models_path = "checkpoints/cnn3d_gsp"  # no extension; we'll format epochs/metrics into the filename

early_stop = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=max(1, patience // 2), verbose=1)
csv_logger = CSVLogger(log_file_path, append=False)

# For TF 2.x, use metric names you actually log; here we use val_mae since it's in metrics.
# If you prefer val_loss, change the format accordingly.
model_names = trained_models_path + ".onigiri_df2j3i_dif982183nfdsfuh982h312jkhkdsahbadyfgasdfr234.hdf5"
model_checkpoint = ModelCheckpoint(
    filepath=model_names,
    monitor="val_loss",
    verbose=1,
    save_best_only=True,
    save_weights_only=False
)

callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

# ---- 4A) Fit with arrays / tf.data (recommended) ----
history = mood_model.fit(
    all_possible_combinations_input, y_true,
    epochs=num_epochs,
    batch_size=batch_size,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)



# model parameters/compilation
model = mini_XCEPTION(input_shape, num_classes)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# model callbacks
early_stop = EarlyStopping('val_loss', patience=patience)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                              patience=int(patience/2), verbose=1)
csv_logger = CSVLogger(log_file_path, append=False)
model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
model_checkpoint = ModelCheckpoint(model_names,
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=False)
callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

# training model
model.fit_generator(image_generator.flow(mode='train'),
                    steps_per_epoch=int(len(train_keys) / batch_size),
                    epochs=num_epochs, verbose=1,
                    callbacks=callbacks,
                    validation_data=image_generator.flow('val'),
                    validation_steps=int(len(val_keys) / batch_size))
