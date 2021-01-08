import pickle

from keras.regularizers import l2
from keras.optimizers import RMSprop, SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

train_data_dir = './fer2013/train'
validation_data_dir = './fer2013/validation'
BS = 128
Resize_pixelsize = 197
batch_size = 64
nb_train_samples = 28273
nb_validation_samples = 3534
epochs = 50

# Using data augmentation
def get_datagen(dataset,  modelType = 'grayscale', aug=False):
    if aug:
        datagen = ImageDataGenerator(
                            rescale=1./255,
                            featurewise_center=False,
                            featurewise_std_normalization=False,
                            rotation_range=10,
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            zoom_range=0.1,
                            horizontal_flip=True)
    else:
        datagen = ImageDataGenerator(rescale=1./255)

    if modelType == 'TransferLearning':
        return datagen.flow_from_directory(
                dataset,
                target_size=(197, 197),
                color_mode='rgb',
                shuffle = True,
                class_mode='categorical',
                batch_size=BS)
    else:
        return datagen.flow_from_directory(
                dataset,
                target_size=(48, 48),
                color_mode='grayscale',
                shuffle = True,
                class_mode='categorical',
                batch_size=BS)

train_generator  = get_datagen(train_data_dir,'TransferLearning', True)
validation_generator = get_datagen(validation_data_dir, 'TransferLearning')

checkpoint = ModelCheckpoint("./ResNet50.h5",monitor='val_accuracy',
                            save_best_only=True, mode='max',verbose=1)

reduce_lr = ReduceLROnPlateau(monitor='val_loss' , factor=0.1, patience=2, min_lr=0.00001,model='auto')


earlystop = EarlyStopping(monitor='val_accuracy',
                          min_delta=0.0001,
                          patience=15,
                          verbose=0,
                          restore_best_weights=True)


callbacks = [checkpoint, earlystop, reduce_lr]

if __name__ == '__main__':
    # compiling models
    import Models
    model1 = Models.FERC2()
    model1.compile(optimizer=Adam(lr=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
    model1.summary()
    model2 = Models.FERC()
    model2.compile(optimizer=Adam(lr=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
    model2.summary()
    model3 = Models.ResNet50()
    model3.compile(optimizer=Adam(lr=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
    model3.summary()
    model4 = Models.Proposed_model()
    model4.compile(optimizer=Adam(lr=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
    model4.summary()



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

    history_FERC2 = model2.fit(
                x=train_generator,
                steps_per_epoch=24176 // BS,
                epochs=50,
                validation_data=validation_generator,
                validation_steps=3006 // BS,
                callbacks=callbacks
            )
    with open('./FERC2_History', 'wb') as file_pi:
                pickle.dump(history_FERC2.history, file_pi)

    history4 = Models.model4.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)

