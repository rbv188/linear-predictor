'''
This project is to investigate linear predictors to time series data.
The two datasets provided are sunspots-1.dat and speech-1.dat .
The sunspots dataset measures the count of number of sunspots observed in a given year.
The speech dataset is sampled speech waveform data
'''


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

'''
Function to calculate rxl
input: order of filter(p),training data (data)
output: array from rx0 to rxp
'''
def rx_l(p,data):
    rx = []
    T = len(data)
    #iterate from rx0 to rxp 
    for i in range(0,p+1):
        temp_sum = 0
        #rx[l] = (1/t){sum i=0 to len(data)-1+l} (rx[l]*rx[i+l])
        for j in range(0,T-i):
            temp_sum += ((data[j])*(data[j+i]))
        temp_sum = float(temp_sum)/T
        rx.append(temp_sum)
    return np.array(rx)

#order of filters: 2 to 39
order = range(2,40)

#Dictioanry containning corelation vectors and matrices for different
#order filters.
#Key:order of filter
#Value:vector and matrix for order
corelation_vectors_sunspots = {}
corelation_matrices_sunspots = {}
corelation_vectors_speech = {}
corelation_matrices_speech = {}

#Seperate data into training and test data. First Half of the data
#is used for training and the other half for test
sunspots = np.loadtxt('sunspots-1.dat')
sunspots_train = sunspots[:(len(sunspots)/2)]
sunspots_test = sunspots[(len(sunspots)/2):]
speech = np.loadtxt('speech-1.dat')
speech_train = speech[:len(speech)/2]
speech_test = speech[(len(speech)/2):]


#Populate corelation_matrices and corelation_vector dictionary for
#different order filters
for o in order: #iterating through different orders
    rx_l_result_sunspots = rx_l(o,sunspots_train)
    rx_l_result_speech = rx_l(o,speech_train)
    corelation_vector_sunspots = rx_l_result_sunspots[1:]
    corelation_vector_speech = rx_l_result_speech[1:]  #corelation_vecotr for order o
    temp_vector_sunspots = rx_l_result_sunspots[:-1]
    temp_vector_speech = rx_l_result_speech[:-1]
    
    corelation_matrix_sunspots = np.zeros((o,o)) #empty corealation matrix
    corelation_matrix_speech = np.zeros((o,o)) #empty corealation matrix
    
    for i in range(0,o):
        for j in range(0,o):
            corelation_matrix_sunspots[i][j] = temp_vector_sunspots[abs(i-j)] #filling corealtion 
            corelation_matrix_speech[i][j] = temp_vector_speech[abs(i-j)]     #matrix using rx
    corelation_vectors_sunspots[o] = corelation_vector_sunspots
    corelation_vectors_speech[o] = corelation_vector_speech
    corelation_matrices_sunspots[o] = corelation_matrix_sunspots
    corelation_matrices_speech[o] = corelation_matrix_speech

#list to hold mean square error 
temp_list_sunspots = []
temp_list_speech = []

#MSE vs order for speech training set (Training set performance measure)
for o in order:
    squared_error_speech = 0
    count = 0
    for i in range(0,len(speech)/2-o):
        Xn = np.flipud(np.array(speech[i:i+o]))
        coeffecients = np.dot((np.linalg.inv(corelation_matrices_speech[o])),corelation_vectors_speech[o])
        predict = np.dot(np.transpose(coeffecients),Xn)
        difference_square = np.square(predict - speech[i+o])
        squared_error_speech += difference_square
        count += 1
    temp_list_speech.append((squared_error_speech/count))

#MSE vs order for sunspots training set (Training set performance measure)
for o in order:
    squared_error_sunspots = 0
    count = 0
    for i in range(0,len(sunspots)/2-o):
        Xn = np.flipud(np.array(sunspots[i:i+o]))
        coeffecients = np.dot((np.linalg.inv(corelation_matrices_sunspots[o])),corelation_vectors_sunspots[o])
        predict = np.dot(np.transpose(coeffecients),Xn)
        difference_square = np.square(predict - sunspots[i+o])
        squared_error_sunspots += difference_square
        count += 1
    temp_list_sunspots.append((squared_error_sunspots/count))

#list to hold mean square error 
temp_list_sunspots_test = []
temp_list_speech_test = []

#MSE vs order for speech test set (Test set performance measure)
for o in order:
    squared_error_speech = 0
    count = 0
    for i in range((len(speech)/2-o),len(speech)-o):
        Xn = np.flipud(np.array(speech[i:i+o]))
        coeffecients = np.dot((np.linalg.inv(corelation_matrices_speech[o])),corelation_vectors_speech[o])
        predict = np.dot(np.transpose(coeffecients),Xn)
        difference_square = np.square(predict - speech[i+o])
        squared_error_speech += difference_square
        count += 1
    temp_list_speech_test.append((squared_error_speech/count))

#MSE vs order for sunspots test set (Test set performance measure)
for o in order:
    squared_error_sunspots = 0
    count = 0
    for i in range((len(sunspots)/2-o),len(sunspots)-o):
        Xn = np.flipud(np.array(sunspots[i:i+o]))
        coeffecients = np.dot((np.linalg.inv(corelation_matrices_sunspots[o])),corelation_vectors_sunspots[o])
        predict = np.dot(np.transpose(coeffecients),Xn)
        difference_square = np.square(predict - sunspots[i+o])
        squared_error_sunspots += difference_square
        count += 1
    temp_list_sunspots_test.append((squared_error_sunspots/count))

fig, (ax1, ax2,ax3, ax4) = plt.subplots(nrows=4, ncols=1)
fig.text(0.5, 0.04, 'prediction order', ha='center', fontsize=20)
fig.text(0.04, 0.5, 'mean squared error', va='center', rotation='vertical', fontsize=20)
ax1.plot(order,temp_list_sunspots)
ax1.set_title("sunspots Training set perforamnce")
ax2.plot(order,temp_list_speech)
ax2.set_title("speech Training set perforamnce")
ax3.plot(order,temp_list_sunspots_test)
ax3.set_title("sunspots test set perforamnce")
ax4.plot(order,temp_list_speech_test)
ax4.set_title("speech test set perforamnce")
plt.show()


#plot of prediction and true values for sunspots for order 7 (least mean square error)
plt_test_sunspots = [0]*7
for i in range(0,len(sunspots)-7):
    Xn = np.flipud(np.array(sunspots[i:i+7]))
    coeffecients = np.dot((np.linalg.inv(corelation_matrices_sunspots[7])),corelation_vectors_sunspots[7])
    predict = np.dot(np.transpose(coeffecients),Xn)
    plt_test_sunspots.append(predict)

#plot of prediction and true values for speech for order 15 (least mean square error)
plt_test_speech = [0]*15
for i in range(0,len(speech)-15):
    Xn = np.flipud(np.array(speech[i:i+15]))
    coeffecients = np.dot((np.linalg.inv(corelation_matrices_speech[15])),corelation_vectors_speech[15])
    predict = np.dot(np.transpose(coeffecients),Xn)
    plt_test_speech.append(predict)

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
fig.text(0.5, 0.04, 'Time', ha='center', fontsize=20)
fig.text(0.04, 0.5, 'Magnitude', va='center', rotation='vertical', fontsize=20)
ax1.plot(sunspots,color='r')
ax1.plot(plt_test_sunspots,color='b',marker='.')
ax1.set_title("sunspots prediction vs real data for order 7")
ax1.legend(["real data","predicted data"])
ax2.plot(speech,color='r')
ax2.plot(plt_test_speech,color='b',marker='.')
ax2.set_title("speech prediction vs real data for order 15")
ax2.legend(["real data","predicted data"])
plt.show()
