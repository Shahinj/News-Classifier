import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
import sklearn.tree as tree
import graphviz


import torch
from torch.autograd import Variable
import urllib
import hashlib
import timeit
import random

random.seed(0)

import os
os.chdir('/home/shahin/Desktop/csc411')

## PART 1
def get_word_counts():
    real_dict = {}
    fake_dict = {}
    
    #get all counts
    for line in open("clean_real.txt"):
        seen = {}
        for word in line.split():
            if not word in seen.keys():
                seen[word] = True
                if word in real_dict.keys():
                    real_dict[word] += 1
                else:
                    real_dict[word] = 1
    
    for line in open("clean_fake.txt"):
        seen = {}
        for word in line.split():
            if not word in seen.keys():
                seen[word] = True
                if word in fake_dict.keys():
                    fake_dict[word] += 1
                else:
                    fake_dict[word] = 1
    
    return real_dict, fake_dict
    
     

## PART 2
def get_test_train_valid():
    np.random.seed(0)
    all_real = np.array([])
    all_fake = np.array([])
    
    for line in open("clean_real.txt"):
        all_real = np.append(all_real, line)

    for line in open("clean_fake.txt"):
        all_fake = np.append(all_fake, line)
        
    #shuffle sets
    np.random.shuffle(all_real)
    np.random.shuffle(all_fake)
    
    #split sets
    real_test_index = int(np.floor(0.15*np.size(all_real)))
    real_train_index = int(np.floor(0.85*np.size(all_real)))
    real_test, real_train, real_valid = np.split(all_real,[real_test_index, real_train_index])
    
    fake_test_index = int(np.floor(0.15*np.size(all_fake)))
    fake_train_index = int(np.floor(0.85*np.size(all_fake)))
    fake_test, fake_train, fake_valid = np.split(all_fake, [fake_test_index, fake_train_index])
    
    return real_test, real_train, real_valid, fake_test, fake_train, fake_valid
 
def get_test_train_valid_more():
    np.random.seed(0)
    all_real = np.array([])
    all_fake = np.array([])

    
    for line in open("clean_fake_new.txt"):
        all_fake = np.append(all_fake, line)
    
    cnt = 0
    for line in open("clean_real_new.txt"):
        if(cnt == len(all_fake)):
            break
        cnt += 1
        all_real = np.append(all_real, line)
        
    #shuffle sets
    np.random.shuffle(all_real)
    np.random.shuffle(all_fake)
    
    #split sets
    real_test_index = int(np.floor(0.15*np.size(all_real)))
    real_train_index = int(np.floor(0.85*np.size(all_real)))
    real_test, real_train, real_valid = np.split(all_real,[real_test_index, real_train_index])
    
    fake_test_index = int(np.floor(0.15*np.size(all_fake)))
    fake_train_index = int(np.floor(0.85*np.size(all_fake)))
    fake_test, fake_train, fake_valid = np.split(all_fake, [fake_test_index, fake_train_index])
    
    return real_test, real_train, real_valid, fake_test, fake_train, fake_valid

    
def build_tables(real_train, fake_train):
    x_real = {}         #p(x|real)
    x_fake = {}         #p(x|fake)
    label = {}
    label["real"] = real_train.size
    label["fake"] = fake_train.size
    
    for line in real_train:
        seen = {}
        for word in line.split():
            if not word in seen.keys():
                seen[word] = True
                if word in x_real.keys():
                    x_real[word] += 1.0
                else:
                    x_real[word] = 1.0
    
    for line in fake_train:
        seen = {}
        for word in line.split():
            if not word in seen.keys():
                seen[word] = True
                if word in x_fake.keys():
                    x_fake[word] += 1.0
                else:
                    x_fake[word] = 1.0
    #x_real = {word:count/real_count for (word, count) in x_real.items()}
    #x_fake = {word:count/fake_count for (word, count) in x_fake.items()}
    
    return label, x_real, x_fake

def naive_bayes(label, x_real, x_fake, headline, m, p):
    real_log_sum = 0.0
    fake_log_sum = 0.0
    
    for word in headline.split():
        #real
        if word in x_real.keys():
            prob = (x_real[word] + m*p)/(label["real"] + m)
        else:
            prob = m*p/(label["real"] + m)
        real_log_sum += np.log(prob)
        #fake
        if word in x_fake.keys():
            prob = (x_fake[word] + m*p)/(label["fake"] + m)
        else:
            prob = m*p/(label["fake"] + m)
        fake_log_sum += np.log(prob)
    
    real_prob = np.exp(real_log_sum)*label["real"]/(label["real"]+label["fake"])
    fake_prob = np.exp(fake_log_sum)*label["fake"]/(label["real"]+label["fake"])
    
    if real_prob > fake_prob:
        return "real"
    else:
        return "fake"
  
def bayes_test(label, x_real, x_fake, real_set, fake_set, m, p):
    correct = 0.0
    total = 0.0
    
    for headline in real_set:
        if naive_bayes(label, x_real, x_fake, headline, m, p) == "real":
            correct += 1.0
        total += 1.0
    
    for headline in fake_set:
        if naive_bayes(label, x_real, x_fake, headline, m, p) == "fake":
            correct+= 1.0
        total += 1.0
    return correct/total
    
    

##PART 4
    
def get_vocab_og():
    
    vocab = []
    map = {}
    
    for line in open("clean_fake.txt"):
        words = str.split(line)
        for word in words:
            if word not in vocab:
                vocab.append(word)
    
    for line in open("clean_real.txt"):
        words = str.split(line)
        for word in words:
            if word not in vocab:
                vocab.append(word)
        
    for word in vocab:
        map[word] = vocab.index(word)

    return map

def get_logistic_sets(real_test, real_train, real_valid, fake_test, fake_train, fake_valid,vocab = get_vocab_og):

    vocab = vocab()
    x = np.zeros((len(vocab),real_train.shape[0] + fake_train.shape[0]))
    y = np.zeros((1,real_train.shape[0] + fake_train.shape[0]))
    
    case = 0
    for news in real_train:
        words = str.split(news)
        for word in words:
            if word not in vocab:
                continue
            row = vocab.get(word)
            x[row,case] = 1
        y[0,case] = 1
        case += 1
        
    for news in fake_train:
        words = str.split(news)
        for word in words:
            if word not in vocab:
                continue
            row = vocab.get(word)
            x[row,case] = 1
        y[0,case] = 0
        case += 1
        
    train_set = x
    train_set_label = y
    
    x = np.zeros((len(vocab),real_test.shape[0] + fake_test.shape[0]))
    y = np.zeros((1,real_test.shape[0] + fake_test.shape[0]))
    
    case = 0
    for news in real_test:
        words = str.split(news)
        for word in words:
            if word not in vocab:
                continue
            row = vocab.get(word)
            x[row,case] = 1
        y[0,case] = 1
        case += 1
        
    for news in fake_test:
        words = str.split(news)
        for word in words:
            if word not in vocab:
                continue
            row = vocab.get(word)
            x[row,case] = 1
        y[0,case] = 0
        case += 1
        
    test_set = x
    test_set_label = y
    
    
    x = np.zeros((len(vocab),real_valid.shape[0] + fake_valid.shape[0]))
    y = np.zeros((1,real_valid.shape[0] + fake_valid.shape[0]))
    
    case = 0
    for news in real_valid:
        words = str.split(news)
        for word in words:
            if word not in vocab:
                continue
            row = vocab.get(word)
            x[row,case] = 1
        y[0,case] = 1
        case += 1
        
    for news in fake_valid:
        words = str.split(news)
        for word in words:
            if word not in vocab:
                continue
            row = vocab.get(word)
            x[row,case] = 1
        y[0,case] = 0
        case += 1
        
    valid_set = x
    valid_set_label = y
        
    return train_set,train_set_label, test_set, test_set_label, valid_set, valid_set_label

def get_vocab():
    
    vocab = []
    map = {}
    
    cnt_fake = 0
    for line in open("clean_fake_new.txt"):
        cnt_fake += 1
        words = str.split(line)
        for word in words:
            if word not in vocab:
                vocab.append(word)
    
    cnt = 0
    for line in open("clean_real_new.txt"):
        if (cnt == cnt_fake):
            break
        cnt += 1
        words = str.split(line)
        for word in words:
            if word not in vocab:
                vocab.append(word)
        
    for word in vocab:
        map[word] = vocab.index(word)

    return map
    

def f(x, y, theta):
    '''
    cost function
    input: training set and their label(x,y = 0 or 1) and thetas
    output: cost function
    '''
    # return sum( (y - dot(theta.T,x)) ** 2)      #J
    return sum( y *  np.log(logistic(np.matmul(theta.T[:,1:],x[:,:]) + theta.T[:,0])) + (1-y) * np.log(1-logistic(np.matmul(theta.T[:,1:],x[:,:]) + theta.T[:,0])))
    
    
def df(x,y,theta,reg= 0):
    '''
    gradient function
    input: training set and their label(x,y = 0 or 1) and thetas
    output: derivative of cost function
    '''
    # return -2*sum( (y - dot(theta.T,x)) * x, 1).T  + 2*reg*theta.reshape((theta.shape[0],))    #J, axis=1 indicates that row is constant, columns add
    sigma = logistic(np.matmul(theta.T,x))
    return -1 * np.matmul(x,(y-sigma).T) + 2*reg*theta
    
    
def grad_descent(f, df, x, y, init_t, alpha,max_iter,EPS):
    
    prev_t = init_t-10*EPS
    t = init_t.copy()
    iter  = 0
    sub = np.zeros(init_t.shape)
    while norm(t - prev_t) >  EPS and iter < max_iter:
        prev_t = t.copy()
        sub = alpha*df(x, y, t).reshape(sub.shape)
        t -= sub
        iter += 1
    return t        #t is the fitted thetas
    
def stochastic_grad_descent(f, df, x, y, init_t, alpha,max_iter,EPS,regularization):
    np.random.seed(0)
    prev_t = init_t-10*EPS
    t = init_t.copy()
    iter  = 0
    sub = np.zeros(init_t.shape)
    while norm(t - prev_t) >  EPS and iter < max_iter:
        columns = np.random.choice(x.shape[1],size = 100,replace = False)
        stoch_x = x[:,columns]  
        stoch_y = y[:,columns] 
        prev_t = t.copy()
        sub = alpha*df(stoch_x, stoch_y, t,regularization).reshape(sub.shape)
        t -= sub
        iter += 1
    return t        #t is the fitted thetas



def train_logistic(train_set,train_set_label,learning_rate = 0.0001,max_iterations = 3000,error = 1e-5, regularization = 2):
    #add bias term to train set
    train_set_w_bias = np.zeros((train_set.shape[0]+1,train_set.shape[1]))
    train_set_w_bias[0:] = 1
    train_set_w_bias[1:] = train_set
    
    #initialize theta and parameters
    np.random.seed(0)
    init_theta = np.random.rand(train_set_w_bias.shape[0],1)
    theta = stochastic_grad_descent(f,df, train_set_w_bias, train_set_label,init_theta,learning_rate,max_iterations,error,regularization)
    
    return theta
    
def logistic(output):
    return 1.0/(1+exp(-1*output))

def plot_learning_curve(train_set,train_set_label, test_set, test_set_label, valid_set, valid_set_label,learning_rate = 0.001,max_iterations = 10000,error = 1e-5,regulatization = 1):
    '''
    input: (optional) learning curve of gradient descent with momentum
    output: the learning curve of gradient descent, using all the images
    '''
    np.random.seed(0)
    
    train_set_w_bias = np.zeros((train_set.shape[0]+1,train_set.shape[1]))
    train_set_w_bias[0:] = 1
    train_set_w_bias[1:] = train_set
    init_t= np.random.rand(train_set_w_bias.shape[0],1)
    
    np.random.seed(0)
    prev_t = init_t-10*error
    t = init_t.copy()
    iter  = 0
    sub = np.zeros(init_t.shape)
    performance = np.zeros((max_iterations,4))
    prev_perf = None
    
    x = train_set_w_bias
    y = train_set_label
    
    
    while (norm(t - prev_t) >  error) and iter < max_iterations:
        print (iter)
        columns = np.random.choice(x.shape[1],size = 100,replace = False)
        stoch_x = x[:,columns]  
        stoch_y = y[:,columns] 
        prev_t = t.copy()
        sub = learning_rate*df(stoch_x, stoch_y, t,regulatization).reshape(sub.shape)
        t -= sub
        performance[iter:iter+1,0] = iter
        performance[iter:iter+1,1] = test_performance(train_set,train_set_label,t)
        performance[iter:iter+1,2] = test_performance(valid_set,valid_set_label,t)
        performance[iter:iter+1,3] = test_performance(test_set,test_set_label,t)
        iter += 1
        print (iter)
        
            
    #need to plot
    plt.figure()
    plt.plot(performance[:,0],performance[:,1])   #train set plot
    plt.plot(performance[:,0],performance[:,2])   #valid set plot
    plt.plot(performance[:,0],performance[:,3])   #test set plot
    plt.title('Learning Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Performance')
    plt.legend(['Performance of training set', 'Performance of validation set','Performance of test set'])
    plt.show()

    return performance


def test_performance(x, label, w):
    '''
    input: a set and its labels, the weight matrix and bias matrix
    output: percentage correct classified
    '''
    h = logistic(np.matmul(w.T[:,1:],x[:,:]) + w.T[:,0])
    diff = np.round(np.abs(label - h))
    return (diff.shape[1]-np.sum(diff))/diff.shape[1]*100
    
def get_optimum_param():
    '''
    input: none
    output: an array with the performances of different learning rates and max iterations
    '''
    real_test, real_train, real_valid, fake_test, fake_train, fake_valid = get_test_train_valid()
    
    train_set,train_set_label, test_set, test_set_label, valid_set, valid_set_label = get_logistic_sets(real_test, real_train, real_valid, fake_test, fake_train, fake_valid)
    
    alphas = [0.1,0.01, 0.001, 0.0001, 0.00001]
    iterations = [100, 1000, 10000, 50000]
    errors = [1e-3,1e-4,1e-5,1e-6]
    regs = [1,2,3,4,5,10]
    # alphas = [0.01, 0.001]
    # iterations = [100, 1000]
    # errors = [1e-3,10]
    # regs = [0,1]
    performance = np.zeros((len(alphas)*len(iterations)*len(errors)*len(regs),7))
    i = 0
    for iter in iterations:
        for reg in regs:
            for l_r in alphas:
                for error in errors:
                    start_time = time.time()
                    w = train_logistic(train_set,train_set_label,l_r,iter,error,reg)
                    performance[i,4] = float(time.time() - start_time)
                    performance[i,0] = l_r
                    performance[i,1] = iter
                    performance[i,2] = error
                    performance[i,3] = reg                
                    performance[i,5] = test_performance(train_set, train_set_label, w)
                    performance[i,6] = test_performance(valid_set, valid_set_label, w)
                    i += 1
                    print ('done: alpha=' + str(l_r) + 'iterations=' +str(iter) + 'error=' + str(error) + 'lambda=' + str(reg) + 'valid=' + str(test_performance(valid_set, valid_set_label, w)) + 'train=' + str(test_performance(train_set, train_set_label, w)))
                    # print(performance)
    return performance
    
def test_tree(actual, predicted_labels):
    prediction = predicted_labels.reshape(1,predicted_labels.shape[0])
    diff = np.abs(actual - prediction)
    return (diff.shape[1]-np.sum(diff))/diff.shape[1]*100
    
    
def tree_classifier_curve():
    real_test, real_train, real_valid, fake_test, fake_train, fake_valid = get_test_train_valid()
    
    train_set,train_set_label, test_set, test_set_label, valid_set, valid_set_label = get_logistic_sets(real_test, real_train, real_valid, fake_test, fake_train, fake_valid)
    
    depth_list = np.arange(0,1000,5)
    performance = np.zeros((len(depth_list),3))
    i = 0
    for depth in depth_list:
        if depth == 0:
            depth = 1
        clf = tree.DecisionTreeClassifier(max_depth = depth) 
        clf.fit(train_set.T,train_set_label.T)
        performance[i,0] = depth
        performance[i,1] = test_tree(train_set_label,clf.predict(train_set.T))
        performance[i,2] = test_tree(valid_set_label,clf.predict(valid_set.T))
        i += 1
        
    
    plt.figure()
    plt.plot(performance[:,0],performance[:,1])   #train set plot
    plt.plot(performance[:,0],performance[:,2])   #valid set plot
    plt.title('Performance vs Depth of Decision Tree')
    plt.xlabel('Depth')
    plt.ylabel('Performance')
    plt.legend(['Performance of training set', 'Performance of validation set'])
    plt.show()
    
    return performance
    
def visualize_tree(classifier):
 
    vocab = get_vocab()
    
    features = sorted(vocab,key=vocab.get)
    
    graph = graphviz.Source(tree.export_graphviz(clf, out_file=None, feature_names = features, class_names=['real','fake'], filled=True, rounded=True))
    graph.render('Fake_News_decision_tree')
    
def tree_classifier(depth = 85):
    real_test, real_train, real_valid, fake_test, fake_train, fake_valid = get_test_train_valid()
    
    train_set,train_set_label, test_set, test_set_label, valid_set, valid_set_label = get_logistic_sets(real_test, real_train, real_valid, fake_test, fake_train, fake_valid)
    
    clf = tree.DecisionTreeClassifier(max_depth = depth) 
    clf.fit(train_set.T,train_set_label.T)
    return clf
    
def H(split_l, split_r):
    total = float(split_l + split_r)
    return -1*split_l/total*np.log2(split_l/total) - split_r/total*np.log2(split_r/total)
    
    
    
def bonus_forest(depth = 500):
    #performance of 81%
    from sklearn.ensemble import RandomForestClassifier as f_clf
    real_test, real_train, real_valid, fake_test, fake_train, fake_valid = get_test_train_valid()
    
    train_set,train_set_label, test_set, test_set_label, valid_set, valid_set_label = get_logistic_sets(real_test, real_train, real_valid, fake_test, fake_train, fake_valid)
    
    clf = f_clf(40,max_depth = depth) 
    clf.fit(train_set.T,train_set_label.T)
    return clf
    
def bonus_nn_param():
    from sklearn.neural_network import MLPClassifier as nn
    real_test, real_train, real_valid, fake_test, fake_train, fake_valid = get_test_train_valid()
    
    train_set,train_set_label, test_set, test_set_label, valid_set, valid_set_label = get_logistic_sets(real_test, real_train, real_valid, fake_test, fake_train, fake_valid)
    
    iter = np.arange(1,1000,50)
    hidden_list = np.arange(1,25,3)
    performance = np.zeros((hidden_list.shape[0] * iter.shape[0] ,3))
    network = nn()
    network.activation = 'relu'     #relu/logistic/tanh
    network.learning_rate_init = 0.0001 #float
    network.learning_rate = 'constant'  #adaptive/constant
    i = 0
    for iteration in iter:
        for hidden in hidden_list:
            network.hidden_layer_sizes = tuple([hidden]*4)
            network.max_iter = iteration
            network.fit(train_set.T,train_set_label.T)
            performance[i,0] = hidden
            performance[i,1] = iteration                
            performance[i,2] = network.score(valid_set.T,valid_set_label.T)
            print ('done: #neuron=' + str(hidden) + 'iterations=' +str(iteration) + 'valid=' + str(performance[i,2]))
            i += 1
    return performance
    
def bonus_nn_torch():
    
    np.random.seed(0)
    torch.random.manual_seed(0)

    #vocabulary from project 3 data
    real_test, real_train, real_valid, fake_test, fake_train, fake_valid = get_test_train_valid()    
    train_set,train_set_label, test_set, test_set_label, valid_set, valid_set_label = get_logistic_sets(real_test, real_train, real_valid, fake_test, fake_train, fake_valid,get_vocab_og)
    real_test_more, real_train_more, real_valid_more, fake_test_more, fake_train_more, fake_valid_more = get_test_train_valid_more()
    train_set_more,train_set_label_more, test_set_more, test_set_label_more, valid_set_more, valid_set_label_more = get_logistic_sets(real_test_more, real_train_more, real_valid_more, fake_test_more, fake_train_more, fake_valid_more,get_vocab_og)
    
    #vocabulary from kaggle data
    # real_test, real_train, real_valid, fake_test, fake_train, fake_valid = get_test_train_valid()    
    # train_set,train_set_label, test_set, test_set_label, valid_set, valid_set_label = get_logistic_sets(real_test, real_train, real_valid, fake_test, fake_train, fake_valid,get_vocab)
    # real_test_more, real_train_more, real_valid_more, fake_test_more, fake_train_more, fake_valid_more = get_test_train_valid_more()
    # train_set_more,train_set_label_more, test_set_more, test_set_label_more, valid_set_more, valid_set_label_more = get_logistic_sets(real_test_more, real_train_more, real_valid_more, fake_test_more, fake_train_more, fake_valid_more,get_vocab)

    print('done loading sets')
    
    dtype_float = torch.FloatTensor
    dtype_long = torch.LongTensor
    dim_x = train_set.shape[0]
    dim_h1 = 512
    dim_h2 = 256
    dim_o = 1
    max_iter = 100
    learning_rate = 1e-4;
    
    #train on small data
    # x = Variable(torch.from_numpy(train_set.T), requires_grad = False).type(dtype_float)
    # y = Variable(torch.from_numpy(train_set_label.T), requires_grad = False).type(dtype_float)
    
    #train on kaggle data
    x = Variable(torch.from_numpy(train_set_more.T), requires_grad = False).type(dtype_float)
    y = Variable(torch.from_numpy(train_set_label_more.T), requires_grad = False).type(dtype_float)
    

    model = torch.nn.Sequential( torch.nn.Linear(dim_x,dim_h1), torch.nn.Tanh(), torch.nn.Linear(dim_h1,dim_h2), torch.nn.Tanh(), torch.nn.Linear(dim_h2,dim_o), torch.nn.Sigmoid())
    loss_fn = torch.nn.BCELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    
    
    performance = np.zeros((max_iter,3))
    
    for t in range(0,max_iter):
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        
        model.zero_grad()
        loss.backward()
        optimizer.step()
        
        x_perf = Variable(torch.from_numpy(valid_set_more.T), requires_grad = False).type(dtype_float)
        y_pred_test = model(x_perf).data.numpy()
        score_more = np.mean(np.round(y_pred_test[:,0]) == valid_set_label_more.reshape(valid_set_label_more.shape[1],))

        x_perf = Variable(torch.from_numpy(valid_set.T), requires_grad = False).type(dtype_float)
        y_pred_test = model(x_perf).data.numpy()
        score_less = np.mean(np.round(y_pred_test[:,0]) == valid_set_label.reshape(valid_set_label.shape[1],))
                
        performance[t,0] = t
        performance[t,1] = score_more
        performance[t,2] = score_less
        print('iter=' + str(t) + ' more:' + str(score_more) + ' less:' + str(score_less))
    
    #plot the learning curve
    plt.figure()
    plt.plot(performance[:,0],performance[:,1])   #train set plot
    plt.plot(performance[:,0],performance[:,2])   #valid set plot
    plt.title('Performance vs epoch')
    plt.xlabel('epoch')
    plt.ylabel('Performance')
    plt.legend(['Performance on Kaggle data', 'Performance on project 3 data'])
    plt.show()
        
    return model, performance


def bonus_nn_search():
    
    np.random.seed(0)
    torch.random.manual_seed(0)
    
    real_test, real_train, real_valid, fake_test, fake_train, fake_valid = get_test_train_valid()
    
    train_set,train_set_label, test_set, test_set_label, valid_set, valid_set_label = get_logistic_sets(real_test, real_train, real_valid, fake_test, fake_train, fake_valid,get_vocab_og)
    
    print('done loading sets')

    iterations = [100,250,500,750,1000]
    alphas = [1e-2,1e-3,1e-4]
    hidden = [1,2,3,4,5]
    neurons = [512,256,128,64,32]

    dtype_float = torch.FloatTensor
    dtype_long = torch.LongTensor
    dim_x = train_set.shape[0]
    
    x = Variable(torch.from_numpy(train_set.T), requires_grad = False).type(dtype_float)
    y = Variable(torch.from_numpy(train_set_label.T), requires_grad = False).type(dtype_float)
    performance = np.zeros((len(iterations)*len(hidden)*len(alphas),4))
    i = 0
    
    for max_iter in iterations:
        for learning_rate in alphas:
            for layer in hidden:
                torch.random.manual_seed(0)
                model = torch.nn.Sequential()
                loss_fn = torch.nn.BCELoss()
                model.add_module('input',torch.nn.Linear(dim_x,neurons[0])) 
                for layer_cnt in range (1,layer+1):
                    model.add_module('activation'+str(layer_cnt), torch.nn.Tanh())
                    if (layer_cnt+1) in range(1,layer+1):
                        output_dim = neurons[layer_cnt]
                    else:
                        output_dim = 1
                    model.add_module('fc'+str(layer_cnt),torch.nn.Linear(neurons[layer_cnt-1],output_dim))
                model.add_module('output', torch.nn.Sigmoid())
                optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
                for t in range(0,max_iter):
                    y_pred = model(x)
                    loss = loss_fn(y_pred, y)
                    
                    model.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                x_perf = Variable(torch.from_numpy(valid_set.T), requires_grad = False).type(dtype_float)
                y_pred_test = model(x_perf).data.numpy()
                score = np.mean(np.round(y_pred_test[:,0]) == valid_set_label.reshape(valid_set_label.shape[1],))
                        
                performance[i,0] = max_iter
                performance[i,1] = learning_rate
                performance[i,2] = layer
                performance[i,3] = score
                                
                print('max_iter:' + str(max_iter) + ' layers:' + str(layer) + ' learning_rate:' + str(learning_rate) + ' performance:' + str(score))
                i += 1
    
    
    return performance


def bonus_nn_derivatives():

    #parameters are obtained from search
    np.random.seed(0)
    torch.random.manual_seed(0)
    
    real_test, real_train, real_valid, fake_test, fake_train, fake_valid = get_test_train_valid()
    
    train_set,train_set_label, test_set, test_set_label, valid_set, valid_set_label = get_logistic_sets(real_test, real_train, real_valid, fake_test, fake_train, fake_valid,get_vocab_og)
    
    print('done loading sets')
    
    dtype_float = torch.FloatTensor
    dtype_long = torch.LongTensor
    dim_x = train_set.shape[0]
    dim_h1 = 512
    dim_h2 = 256
    dim_o = 1
    max_iter = 100
    learning_rate = 1e-4;
    
    x = Variable(torch.from_numpy(train_set.T), requires_grad = False).type(dtype_float)
    y = Variable(torch.from_numpy(train_set_label.T), requires_grad = False).type(dtype_float)
        
    
    model = torch.nn.Sequential( torch.nn.Linear(dim_x,dim_h1), torch.nn.Tanh(), torch.nn.Linear(dim_h1,dim_h2), torch.nn.Tanh(), torch.nn.Linear(dim_h2,dim_o), torch.nn.Sigmoid())
    loss_fn = torch.nn.BCELoss()
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    
    
    for t in range(0,max_iter):
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        
        model.zero_grad()
        loss.backward()
        optimizer.step()
        
    print('done training')
    derivatives = np.zeros((train_set.shape[0],1))
    for i in range(0,train_set.shape[0]):
        headline = np.zeros((train_set.shape[0],1))
        headline[i,0] = 1
        x = Variable(torch.from_numpy(headline.T), requires_grad = True).type(dtype_float)
        derivatives[i,0] = torch.autograd.grad(model(x),x)[0][0,i].data[0]

    
    return derivatives
    

if __name__ == "__main__":
    
    #PART 4
    real_test, real_train, real_valid, fake_test, fake_train, fake_valid = get_test_train_valid()
    
    train_set,train_set_label, test_set, test_set_label, valid_set, valid_set_label = get_logistic_sets(real_test, real_train, real_valid, fake_test, fake_train, fake_valid)
    
    # theta = train_logistic(train_set,train_set_label,0.001,10000,1e-5,1)
    # 
    # # 
    # print 'Test performance'
    # print(test_performance(test_set,test_set_label,theta))
    # 
    # print 'Valid performance'
    # print(test_performance(valid_set,valid_set_label,theta))
    # 
    # print 'Train performance'
    # print(test_performance(train_set,train_set_label,theta))
        
    
    # plot_learning_curve(train_set,train_set_label, test_set, test_set_label, valid_set, valid_set_label)
    
    # result = get_optimum_param()
    # 
    # import pandas as pd
    # 
    # res = pd.DataFrame(result)
    # res.to_csv('optimum_result_part_4.csv')
    # print res
    
    #result = tree_classifier()
    # part_8_a(result)
    # part_8_b(result)
     
    # result = bonus_forest()
    # result = bonus_nn_param()
    network,result = bonus_nn_torch()
    # plt.figure()
    # plt.plot(result[:1000,0],result[:1000,1])   #train set plot
    # plt.plot(result[:1000,0],result[:1000,2])   #valid set plot
    # plt.title('Performance vs epoch')
    # plt.xlabel('epoch')
    # plt.ylabel('Performance')
    # plt.legend(['Performance on more data', 'Performance on small data'])
    # plt.show()
    
    # search = bonus_nn_search()
    # derivative = bonus_nn_derivatives()
    # vocabulary = get_vocab_og()
    # derivative_words = {}
    # for i in range (0,derivative.shape[0]):
    #     val = derivative[i][0]
    #     word =  [word for word in vocabulary if vocabulary[word] == i][0]
    #     derivative_words[word] = val
    # influence = sorted(derivative_words, key=derivative_words.get)
    