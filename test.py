from keras_preprocessing import image
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
import numpy as np
model=load_model('D:\\aalok_project\\flask-app-20220727T063329Z-001\\flask-app\\flask-app\\models\\model113_vgg19.h5') #Loading our model
img=image.load_img('D:\\aalok_project\\flask-app-20220727T063329Z-001\\flask-app\\flask-app\\uploads\\NORMAL2-IM-1438-0001.jpeg',target_size=(224,224))
imagee=image.img_to_array(img) #Converting the X-Ray into pixels
imagee=np.expand_dims(imagee, axis=0)
img_data=preprocess_input(imagee)
prediction=model.predict(img_data)
print(prediction[0][0])
if prediction[0][0]>prediction[0][1]: 
    
    #Printing the prediction of model.
    print('Person is safe.')
else:
    print('Person is affected with Pneumonia.')
#print(f'Predictions: {prediction}')