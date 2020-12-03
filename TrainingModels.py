import Models
from keras.regularizers import l2
from keras.optimizers import RMSprop, SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
num_features = 64
num_classes = 6
height, width = 48, 48
epochs = 100
batch_size = 64

train_data_dir = './fer2013/train'
validation_data_dir = './fer2013/validation'

# Using data augmentation
train_datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=0,
    width_shift_range=0.0,
    height_shift_range=0.0,
    shear_range=0,
    zoom_range=0.0,
    horizontal_flip=True,
    vertical_flip=False)

validation_datagen = ImageDataGenerator(rescale=1. / 255)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    color_mode='grayscale',
    target_size=(height, width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    color_mode='grayscale',
    target_size=(height, width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

checkpoint = ModelCheckpoint("./emotion.h5",
                             monitor="val_loss",
                             mode="min",
                             save_best_only=True,
                             verbose=1)

earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=20,
                          verbose=0,
                          restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.1,
    patience=10, verbose=0, mode='auto',
    min_delta=0.0001, cooldown=0, min_lr=0)

# we put our call backs into a callback list
callbacks = [earlystop, checkpoint, reduce_lr]
if __name__ == '__main__':
    Models.get_model1().compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    Models.get_model2().compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    Models.get_model3().compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    Models.get_model4().compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    nb_train_samples = 28273
    nb_validation_samples = 3534
    epochs = 50

    history1 = Models.get_model1().fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)

    history2 = Models.model2.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)

    history3 = Models.model3.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)

    history4 = Models.model4.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)