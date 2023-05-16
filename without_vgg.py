#importing the necessary library

from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPool2D , Flatten , Dropout , BatchNormalization
from keras_preprocessing.image import ImageDataGenerator




#using image data generator for importing images from drive
train_data =ImageDataGenerator(rescale=1./255,
                               shear_range = 0.2,
                               zoom_range=0.2,
                               horizontal_flip=True,
                               featurewise_center=False,
                               samplewise_center=False,
                               featurewise_std_normalization=False,
                               samplewise_std_normalization=False,
                               zca_whitening=False,
                               rotation_range=30,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               vertical_flip=False)
test_data=ImageDataGenerator(rescale=1./255)

training_set=train_data.flow_from_directory('D:/Files/Project/archive/chest_xray/test',
                                            target_size=(150,150),
                                            batch_size=10,
                                            class_mode='categorical')
test_set=test_data.flow_from_directory('D:/Files/Project/archive/chest_xray/train',
                                      target_size = (150, 150),
                                      batch_size=10,
                                      class_mode='categorical')

r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=2000,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)

#checking accuracy
evaluation = model.evaluate(test_set)
print(f"Test Accuracy: {evaluation[1] * 100:.2f}%")
evaluation = model.evaluate(training_set)
print(f"Train Accuracy: {evaluation[1] * 100:.2f}%")

model.save('D:/aalok_project/flask-app-20220727T063329Z-001/flask-app/flask-app/model112_vgg19.h5')

