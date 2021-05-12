
import tflite_runtime.interpreter as tflite
import platform
import numpy as np
import pandas as pd
import time
import pickle
import sys
from glob import glob
from sklearn.metrics import f1_score, accuracy_score

DATASET_PATH=sys.argv[1]

accuracies = []
micro=[]
macro=[]
weighted=[]
f1=np.ndarray((5,10))
times = []
length=5

for model_nr in np.arange(5):  
  MODEL_PATH=DATASET_PATH+'/models/model_{}.h5'.format(model_nr)
  DATA_PATH=DATASET_PATH+'/data/model_{}/'.format(model_nr)
  TFLITE_PATH=DATASET_PATH+'/models/model_{}.tflite'.format(model_nr)
  QUANTIZED_TFLITE_PATH=DATASET_PATH+'/models/quant_model_{}.tflite'.format(model_nr)
  TPU_QUANTIZED_TFLITE_PATH=DATASET_PATH+'/models/quant_model_{}_edgetpu.tflite'.format(model_nr)

  
    
  X_test=np.load(DATA_PATH+'X_test.npy')
  y_test=np.load(DATA_PATH+'y_test.npy')
  X_train=np.load(DATA_PATH+'X_train.npy')
  y_train=np.load(DATA_PATH+'y_train.npy')
  
  acc = 0
  y_pred=np.zeros(len(X_test))

  interpreter = interpreter = tflite.Interpreter(model_path=(TPU_QUANTIZED_TFLITE_PATH),
                                                 experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
  interpreter.allocate_tensors()
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  input_shape = input_details[0]['shape']
  for i in range(len(X_test)):
      print(i)
#       set_input(interpreter, x_test[i])
      input_data = X_test[i].reshape(input_shape).astype(np.int8)
      #input_data = input_data.astype(np.float32)
      interpreter.set_tensor(input_details[0]['index'], input_data)
      start = time.perf_counter()
      interpreter.invoke()
      inference_time = time.perf_counter() - start
      times.append(inference_time * 1000)
      output_data = interpreter.get_tensor(output_details[0]['index'])
      y_pred[i]=np.argmax(output_data)
      if np.argmax(output_data) == np.argmax(y_test[i]):
          acc += 1

  Y_test = np.argmax(y_test, axis=1) # Convert one-hot to index
  micro.append(f1_score(Y_test, y_pred, average='micro'))
  macro.append(f1_score(Y_test, y_pred, average='macro'))
  weighted.append(f1_score(Y_test, y_pred, average='weighted'))
  f1[model_nr]=(f1_score(Y_test, y_pred, average=None))
  acc = (acc/len(X_test)) * 100
  accuracies.append(acc)
  Y_test = np.argmax(y_test, axis=1)

print("Accuracy mean, std")
print(np.mean(accuracies))
print(np.std(accuracies))
print("time mean, std (ms)")
print(np.mean(times))
print(np.std(times))
print("Micro f1 mean, std")
print(np.mean(micro))
print(np.std(micro))
print("Macro f1 mean, std")
print(np.mean(macro))
print(np.std(macro))
print("Weighted f1 mean, std")
print(np.mean(weighted))
print(np.std(weighted))
print("f1 multiclass mean, std")
print(f1.mean(axis=0))
print(f1.std(axis=0))