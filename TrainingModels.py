'''
Things done in this file:
1. Import models from Models
2. Apply data augmentation
3. Compile the models
4. Train the models
'''

import pickle
import Models
from keras.optimizers import Adam
from livelossplot import PlotLossesKeras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

train_data_dir = './fer2013/train'
validation_data_dir = './fer2013/validation'
BS = 128
Resize_pixelsize = 197


# Using data augmentationa
def get_datagen(dataset, modelType='grayscale', aug=False):
    if aug:
        datagen = ImageDataGenerator(
            rescale=1. / 255,
            featurewise_center=False,
            featurewise_std_normalization=False,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True)
    else:
        datagen = ImageDataGenerator(rescale=1. / 255)

    if modelType == 'TransferLearning':
        return datagen.flow_from_directory(
            dataset,
            target_size=(197, 197),
            color_mode='rgb',
            shuffle=True,
            class_mode='categorical',
            batch_size=BS)
    else:
        return datagen.flow_from_directory(
            dataset,
            target_size=(48, 48),
            color_mode='grayscale',
            shuffle=True,
            class_mode='categorical',
            batch_size=BS)


if __name__ == '__main__':
    # compiling models
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

    # training models
    train_generator = None
    validation_generator = None
    for i in range(4):
        path = "./Model" + i + ".h5"
        checkpoint = ModelCheckpoint(path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=0.00001, model='auto')
        callbacks = [PlotLossesKeras(), checkpoint, reduce_lr]
        if i == 2:
            train_generator = get_datagen(train_data_dir, 'TransferLearning', True)
            validation_generator = get_datagen(validation_data_dir, 'TransferLearning')
            history_resnet50 = model3.fit_generator(
                generator=train_generator,
                validation_data=validation_generator,
                steps_per_epoch=24176 // BS,
                validation_steps=3006 // BS,
                shuffle=True,
                epochs=50,
                callbacks=callbacks,
                use_multiprocessing=True,
            )
            with open('./ResNt50_History', 'wb') as file_pi:
                pickle.dump(history_resnet50.history, file_pi)
        else:
            train_generator = get_datagen(train_data_dir, aug=True)
            validation_generator = get_datagen(validation_data_dir)
            history_FERC2 = model1.fit(
                x=train_generator,
                steps_per_epoch=24176 // BS,
                epochs=50,
                validation_data=validation_generator,
                validation_steps=3006 // BS,
                callbacks=callbacks
            )
            with open('./FERC_History', 'wb') as file_pi:
                pickle.dump(history_FERC2.history, file_pi)

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

            history_Proposed_Model = model4.fit(
                x=train_generator,
                steps_per_epoch=24176 // BS,
                epochs=50,
                validation_data=validation_generator,
                validation_steps=3006 // BS,
                callbacks=callbacks
            )
            # saving history
            with open('./Proposed_Model_history', 'wb') as file_pi:
                pickle.dump(history_Proposed_Model.history, file_pi)
