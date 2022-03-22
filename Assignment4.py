import pandas as pd

data = pd.read_csv('play_tennis.csv')

def prior(data, Y):
    column = sorted(list(data[Y].unique()))
    prior = {}
    for i in column:
        prior[i] = len(data[data[Y]==i])/len(data)
    return prior

def likelyhood(data, X, X_label, Y, Y_label):
    data = data[data[Y] == Y_label]
    length = len(data)
    datax = data[data[X] == X_label]
    lengthx = len(datax)
    return lengthx/length

def bayes(data, X, X_label, Y, Y_label):
    Y_dict = prior(data, Y)
    X_dict = prior(data, X)
    p_of_y = Y_dict[Y_label]
    p_of_x = X_dict[X_label]
    return (likelyhood(data, X, X_label, Y, Y_label) * p_of_y)/p_of_x

def predict(data, X, X_label, Y, Y_label):
    dic = {}
    choices = sorted(data[Y].unique())
    if Y_label == choices[0]:
        opposite = choices[1]
    if Y_label == choices[1]:
        opposite = choices[0]
    bayest = bayes(data, X, X_label, Y, Y_label)
    dic[Y_label] = bayest
    dic[opposite] = 1 - bayest
    print(f'P({Y_label}|{X_label}):')
    if dic[Y_label] > dic[opposite]:
        answer = Y_label + ' is a higher probability in terms of playing.'
    else:
        answer = opposite + ' is a higher probability in terms of playing.'
    return (dic, answer)

def bayes4(data, X, X_label, Y, Y_label, X2, X_label2, X3, X_label3, X4, X_label4):
    Y_dict = prior(data, Y)
    X_dict = prior(data, X)
    X_dict2 = prior(data, X2)
    X_dict3 = prior(data, X3)
    X_dict4 = prior(data, X4)
    p_of_y = Y_dict[Y_label]
    p_of_x = X_dict[X_label]
    p_of_x2 = X_dict2[X_label2]
    p_of_x3 = X_dict3[X_label3]
    p_of_x4 = X_dict4[X_label4]
    return (likelyhood(data, X, X_label, Y, Y_label) * likelyhood(data, X2, X_label2, Y, Y_label) * likelyhood(data, X3, X_label3, Y, Y_label) * likelyhood(data, X4, X_label4, Y, Y_label) * p_of_y)/(p_of_x * p_of_x2 * p_of_x3 * p_of_x4)

def predict4(data, X, X_label, Y, Y_label, X2, X_label2, X3, X_label3, X4, X_label4):
    dic = {}
    choices = sorted(data[Y].unique())
    if Y_label == choices[0]:
        opposite = choices[1]
    if Y_label == choices[1]:
        opposite = choices[0]
    bayest = bayes4(data, X, X_label, Y, Y_label, X2, X_label2, X3, X_label3, X4, X_label4)
    dic[Y_label] = bayest
    dic[opposite] = 1-bayest
    print(f'P({Y_label}|{X_label, X_label2, X_label3, X_label4 }):')
    if dic[Y_label] > dic[opposite]:
        answer = Y_label + ' is a higher probability in terms of playing.'
    else:
        answer = opposite + ' is a higher probability in terms of playing.'
    return dic, answer

#Tests
print(prior(data, 'Outlook'))
print(likelyhood(data, 'Outlook', 'Sunny', 'Play Tennis', 'Yes'))
print(bayes(data, 'Outlook', 'Sunny', 'Play Tennis', 'Yes'))
print(predict(data, 'Outlook', 'Sunny', 'Play Tennis', 'Yes'))
print('______________________________________________________________________________')
print(bayes4(data, 'Outlook', 'Sunny', 'Play Tennis', 'Yes','Temperature', 'Mild', 'Humidity', 'Normal','Wind', 'Weak'))
print(predict4(data, 'Outlook', 'Sunny', 'Play Tennis', 'Yes','Temperature', 'Mild', 'Humidity', 'Normal','Wind', 'Weak'))
