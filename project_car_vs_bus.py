#!/usr/bin/env python
# coding: utf-8

# In[1]:


import librosa as lb
import librosa.display
import numpy as np
from sklearn.model_selection import train_test_split
from scipy import signal
import IPython.display as ipd
import matplotlib.pyplot as plt


# In[2]:


#loading data
path='C:/Users/Python/Desktop/AudioProject/dataset_vehicles/train/'

car_files = lb.util.find_files(path +'car/', ext=['wav']) 
car_files = np.asarray(car_files)
for car_sound in car_files: 
    car_sound, sr = librosa.load(car_sound, sr=20050)  

    
bus_files = lb.util.find_files(path +'bus/', ext=['wav']) 
bus_files = np.asarray(bus_files)
for bus_sound in bus_files: 
    bus_sound, sr = librosa.load(bus_sound, sr=20050)   


# In[3]:


#checking length to see all files loaded and so that we can use values next
print(len(car_files))
print(len(bus_files))
print(len(car_files)+ len(bus_files))

print(len(car_sound))
print(len(bus_sound))


# In[4]:


#correct size segments function
def division_to_correct_size(s, num_s = 259, common = 52):
    #num_s = both len(car_files) and len(bus_files)
    #common = 1/5 * both len(car_files) and len(bus_files)
  
  sound_data=[]
  for data_bal in range(0, len(s), common):
    begin = data_bal
    end   = data_bal + num_s
    segments = s[begin:end]
    
    if(len(segments)==259):
      sound_data.append(segments)
    
  return sound_data


# In[5]:


#use previous function to files
car_sound = division_to_correct_size(car_sound)
bus_sound = division_to_correct_size(bus_sound)


# In[6]:


#Checking what are the shapes and that files loaded correctly
print(car_files[:5])
print('----------------------')
print(car_files.shape)
print('----------------------')
#print(data)
print('----------------------')
print(car_sound)
#print(car_sound.shape)
print('----------------------')

print('----------------------')
print(bus_files[:5])
print('----------------------')
print(bus_files.shape)
print('----------------------')
print(bus_sound)


# In[7]:


#checking that division_to_correct_size-function worked correctly
print(len(car_files))
print(len(bus_files))
print(len(car_sound))
print(len(bus_sound))


# In[8]:


#visualizing car_sound-file
plt.figure(figsize = (15, 5))
plt.plot(np.linspace(0, 5, num=len(car_sound), endpoint=False), car_sound[0:])
plt.title('Car')
plt.xlabel('Time / s')
plt.ylabel('Amplitude')


# In[ ]:





# In[9]:


#visualizing bus_sound-file
plt.figure(figsize = (15, 5))
plt.plot(np.linspace(0, 5, num=len(bus_sound), endpoint=False), bus_sound[0:])
plt.title('Bus')
plt.xlabel('Time / s')
plt.ylabel('Amplitude')


# In[10]:


#Concatenating the separate the vehicle classes and labels
all_audio = np.concatenate([car_sound,bus_sound])
#zeros and ones - label separation
labels_car = np.zeros(len(car_sound))
labels_bus = np.ones(len(bus_sound))
labels = np.concatenate([labels_car,labels_bus])

print(all_audio.shape)
print(labels.shape)


# In[11]:


# creating train_test_split
x_train, x_val, y_train, y_val = train_test_split(np.array(all_audio),np.array(labels), stratify=labels,test_size = 0.2,
                                                   random_state=42,shuffle=True)


# In[12]:


#checking that train_test_split worked.
print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)


# In[13]:


#reshaping
x_train_features  = x_train.reshape(len(x_train),-1,1)
x_val_features = x_val.reshape(len(x_val),-1,1)
print("reshaped array",x_train_features.shape)
print("reshaped array",x_val_features.shape)


# In[14]:


#Model for CNN
from keras.layers import Conv1D, Dropout, MaxPooling1D, GlobalMaxPool1D, Dense, Input 
from keras.models import *
from keras.callbacks import *
from keras import backend as back
def conv_net(x_train):
  back.clear_session()
  inputs = Input(shape=(x_train.shape[1],x_train.shape[2]))
  x = Conv1D(8, 13, padding='same', activation='relu')(inputs)
  x = Dropout(0.27)(x)
  x = MaxPooling1D(2)(x)
  x = Conv1D(16, 11, padding='same', activation='relu')(x)
  x = Dropout(0.27)(x)
  x = MaxPooling1D(2)(x)
  x = GlobalMaxPool1D()(x)
  x = Dense(16, activation='relu')(x)
  outputs = Dense(1,activation='sigmoid')(x) #softmax -> 0.65
  model = Model(inputs, outputs)
  model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
  mc = ModelCheckpoint('best_val_acc_model.hdf5', monitor='val_acc',
                                 verbose=1, save_best_only=True, mode='max')
  return model, mc


# 
# 

# In[15]:


model, mc = conv_net(x_train_features)


# In[16]:


#summary for CNN
model.summary()


# In[17]:


#fit the model
history=model.fit(x_train_features, y_train ,epochs=85, callbacks=[mc], 
batch_size=64, validation_data=(x_val_features,y_val))


# In[18]:


#load the newly acquired model weights
model.load_weights('best_val_acc_model.hdf5')


# In[19]:


#plot model accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.xlabel('epoch ')
plt.ylabel('accuracy')
plt.legend(['train','validation'],loc = 'upper left')
plt.show()


# In[20]:


#plot model loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train','validation'],loc = 'upper left')
plt.show()


# In[21]:


# evaluate model
_, acc = model.evaluate(x_val_features,y_val)
print("Validation Accuracy:",acc)
print("Validation Loss:", _)


# In[22]:


# test if prediction works
i=2                       #change this index for to test other sounds
play_audio = x_val[i].reshape(-1,1)
ipd.Audio(play_audio, rate=sr)


# In[23]:


feat = x_val_features[i]
probability = model.predict(feat.reshape(1,-1,1))
if (probability[0][0] < 0.5 ):
 prediction ='car sound'
else:
 prediction ='bus sound'
print("Prediction:", prediction)
print("Probability:", probability[0][0])
print("if probability is between 0 and 0.4999... -> car (lower is better)")
print("if probability is between 0.5 and 1.0 -> bus (higher is better)")

plt.plot(probability[0][0])
p1= [probability[0][0]]
p2 = [1-(probability[0][0])]
plt.title('Car or Bus')
plt.xlabel('bus')
plt.ylabel('car')
plt.legend([prediction],loc = 'upper left')
plt.scatter(p1, p2)
plt.show()


# In[24]:


# creating log-spectrogram
def create_log_spectrogram(all_audio, sr, ep_s=0.00000000001):
   nperseg  = 25
   noverlap = 20
   f, t, spec = signal.spectrogram(all_audio,fs=sr,
                           nperseg=nperseg,noverlap=noverlap,detrend=False)
   return f, t, np.log(spec.T.astype(np.float32) + ep_s)


# In[25]:


#feature extraction
def feature_extraction(x_train):
 feature_data=[]
 for feat in x_train:
   _,_, spectrogram = create_log_spectrogram(feat, sr)  #needs two _,_

   standard_deviation = np.std(spectrogram, axis=0)
   mean_average = np.mean(spectrogram, axis=0)
   spectrogram = (spectrogram - mean_average) / standard_deviation
   feature_data.append(spectrogram)
 return np.array(feature_data)


# In[26]:


x_train_features  = feature_extraction(x_train)
x_val_features = feature_extraction(x_val)


# In[27]:


model, mc = conv_net(x_train_features)
model.summary()


# In[34]:


#fit model
history=model.fit(x_train_features, y_train, epochs=85, callbacks=[mc], batch_size=64, validation_data=(x_val_features,y_val))


# In[29]:


#load new weights and evaluate model
model.load_weights('best_val_acc_model.hdf5')

_,acc = model.evaluate(x_val_features,y_val)
print("Accuracy:",acc)
print("Validation Loss:", _)


# In[30]:


#plot model accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.xlabel('epoch ')
plt.ylabel('accuracy')
plt.legend(['train','validation'],loc = 'upper left')
plt.show()


# In[31]:


#plot model loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train','validation'],loc = 'upper left')
plt.show()


# In[32]:


#test model
i=2                #change this index for to test other sounds
play_audio = x_val[i]
ipd.Audio(play_audio, rate=sr)


# In[33]:


feat = x_val_features[i]
probability = model.predict(feat.reshape(1,-1,feat.shape[1]))
if (probability[0][0] < 0.5 ):  
    prediction ='car sound'
else:
    prediction ='bus sound'
print("Prediction:", prediction)
print("Probability:", probability[0][0])
print("if probability is between 0 and 0.4999... -> car (lower is better)")
print("if probability is between 0.5 and 1.0 -> bus (higher is better)")

plt.plot(probability[0][0])
p1= [probability[0][0]]
p2 = [1-(probability[0][0])]
plt.title('Car or Bus')
plt.xlabel('bus')
plt.ylabel('car')
plt.legend([prediction],loc = 'upper left')
plt.scatter(p1, p2)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




