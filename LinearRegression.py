import os
import cv2
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import time
st=time.time()

# Define paths to the folders containing ground truth and predicted images
orignal = 'D:/Krutika/Cascade_Unet/Results/orignal'
pred_folder = 'D:/Krutika/Cascade_Unet/Results/new/sec_unet'

# Load the ground truth and predicted images
X = []
y = []
for filename in os.listdir(orignal):
    gt_path = os.path.join(orignal, filename)
    if os.path.isfile(gt_path):
        gt_img = cv2.imread(gt_path)
        if gt_img is not None:
            y.append(gt_img / 255.0)

for fl in os.listdir(pred_folder):
    pred_path = os.path.join(pred_folder, fl)
    if os.path.isfile(pred_path):
        pred_img = cv2.imread(pred_path)
        if pred_img is not None:
            X.append(pred_img / 255.0)

# Convert X and y to numpy arrays
X = np.array(X)
y = np.array(y)
print("X shape:", X.shape)
print("y shape:", y.shape)
X = X.reshape(-1, 256*256, 3)
y = y.reshape(-1, 256*256, 3)
print(X[1])
print(y[1])
predicted_img = np.zeros_like(X)

for i in range(1,2):
    shadowed = X[:, :, i]
    shadow_removed = y[:, :, i]
    a=time.time()
    lr = LinearRegression()
    b=time.time()
    print("regression time",b-a)
    a=time.time()
    lr.fit(shadowed, shadow_removed)
    b=time.time()
    print("model fit time",b-a)
    a=time.time()
    predicted = lr.predict(shadowed).reshape(-1, 256, 256)
    b=time.time()
    print("model predict time",b-a)
    predicted_img[:, :, i] = predicted

predicted_img = predicted_img.reshape(-1, 256, 256, 3)
#cv2.imwrite("D:/Krutika/Cascade_Unet/Results/LR.png", predicted_img * 255.0)
 
for i in range(len(predicted_img)):
 print("world")
 cv2.imwrite("D:/Krutika/Cascade_Unet/Results/"+f'LR_{i}.png', predicted_img[i] * 255.0)
et = time.time() 
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')
#  diff = X[i].flatten() - predicted_img[i].flatten()
#  difff = diff*255
#         #print(difff)
#  diffff= np.sqrt(difff*difff)
#         #print(diffff)
#  su=np.sum(diffff)
#             # print(len(difff))
#  su = su/len(difff)
#  print(su)
# Split the color channels
# shadowed_b, shadowed_g, shadowed_r = cv2.split(X)
# shadow_removed_b, shadow_removed_g, shadow_removed_r = cv2.split(y)

# # Perform linear regression on each color channel separately
# lr = LinearRegression()
# lr.fit(shadowed_b.reshape(-1, 1), shadow_removed_b.reshape(-1, 1))
# predicted_b = lr.predict(shadowed_b.reshape(-1, 1)).reshape(shadowed_b.shape)

# lr.fit(shadowed_g.reshape(-1, 1), shadow_removed_g.reshape(-1, 1))
# predicted_g = lr.predict(shadowed_g.reshape(-1, 1)).reshape(shadowed_g.shape)

# lr.fit(shadowed_r.reshape(-1, 1), shadow_removed_r.reshape(-1, 1))
# predicted_r = lr.predict(shadowed_r.reshape(-1, 1)).reshape(shadowed_r.shape)

# # Merge the predicted color channels into an image
# predicted_img = np.zeros_like(X)
# predicted_img[:, :, 0] = predicted_b
# predicted_img[:, :, 1] = predicted_g
# predicted_img[:, :, 2] = predicted_r

# cv2.imwrite("D:/Krutika/Cascade_Unet/Results/Regression_image/predicted_img.png", predicted_img*255)

# # Train a linear regression model
# model = LinearRegression()
# model.fit(X_train, y_train)

# Evaluate the model on the testing set
#y_pred = model.predict(X_test)
#rmse = mean_squared_error(y_test, y_pred, squared=False)
#print("RMSE:", rmse)
