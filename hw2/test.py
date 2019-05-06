from __future__ import print_function, division

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

from imblearn.over_sampling import SMOTE,ADASYN,RandomOverSampler
#import arff
from scipy.io import arff
import numpy as np
# read csv data
import pandas as pd
#from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
#from keras.models import Sequential
#from keras.layers.core import Dense,Activation
from keras.optimizers import SGD
from keras.utils import np_utils
class CGAN():
    def __init__(self):
        self.rows=1
        self.cols=9
        self.channel=1
        self.shape=(self.rows,self.cols)
        self.num_classes=2
        self.latent_dim=100

        optimizer=Adam(0.0002,0.5)

        self.discriminator=self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        self.generator=self.build_generator()

        noise=Input(shape=(self.latent_dim,))

        label=Input(shape=(1,))
        data=self.generator([noise,label])

        self.discriminator.trainable=False

        valid=self.discriminator([data,label])

        self.combined=Model([noise,label],valid)
        self.combined.compile(loss=['binary_crossentropy'],
                              optimizer=optimizer)

    def build_generator(self):
        model=Sequential()

        model.add(Dense(100,input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(200))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(300))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.shape),activation='tanh'))
        model.add(Reshape(self.shape))

        model.summary()

        noise=Input(shape=(self.latent_dim,))
        #print(noise)
        label=Input(shape=(1,),dtype='int32')
        label_embedding=Flatten()(Embedding(self.num_classes,self.latent_dim)(label))
        model_input=multiply([noise,label_embedding])
        data=model(model_input)

        return Model([noise,label],data)

    def build_discriminator(self):
        model=Sequential()

        model.add(Dense(200,input_dim=np.prod(self.shape)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(200))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(200))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(1,activation='sigmoid'))
        model.summary()

        data=Input(shape=self.shape)
        #print(data.shape)
        label=Input(shape=(1,),dtype='int32')
        #print(label)

        label_embedding=Flatten()(Embedding(self.num_classes,np.prod(self.shape))(label))
        flat_data=Flatten()(data)

        model_input=multiply([flat_data,label_embedding])

        validity=model(model_input)

        return Model([data,label],validity)
    def train(self,train_x,train_y,epochs,batch_size=25):

        train_x=(train_x.astype(np.float32)-5)/5
        train_x=np.expand_dims(train_x,axis=3)
        print(train_x.shape)
        train_y=train_y.reshape(-1,1)
        print(train_y.shape)

        valid=np.ones((batch_size, 1))
        fake=np.zeros((batch_size,1))

        for epoch in range(epochs):
            #d_loss_real=
            idx=np.random.randint(0,train_x.shape[0],batch_size)
            data,label=train_x[idx],train_y[idx]
            print(data.shape)
            data=data.swapaxes(2,1)
            #print(data)
            #print(label.shape)
            #print(idx)

            noise=np.random.normal(0,1,(batch_size,100))
            #print(noise)

            gen_data=self.generator.predict([noise,label])
            #print(gen_data)
            #gen_data=np.expand_dims(gen_data,axis=3)

            #d_loss_real = self.discriminator.fit([data,label],valid)
            #print(d_loss_real)
            #d_loss_fake = self.discriminator.fit([gen_data,label],fake)
            #print(d_loss_fake)
            d_loss_real=self.discriminator.train_on_batch([data,label],valid)
            d_loss_fake=self.discriminator.train_on_batch([gen_data,label],fake)
            d_loss=0.5*np.add(d_loss_real,d_loss_fake)

            sampled_label = np.random.randint(0, 2, batch_size).reshape(-1, 1)

            g_loss=self.combined.train_on_batch([noise,sampled_label],valid)
            #self.combined.fit([noise,sampled_label],valid)


            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

    def sample_data(self):
        r=1
        noise = np.random.normal(0, 1, (r, 100))#.reshape(-1, 1)
        sampled_label = np.arange(1, 2,r).reshape(-1, 1)
        #sampled_label=1

        gen_data=self.generator.predict([noise,sampled_label])
        gen_data=(gen_data*5+5).astype(int)
        #gen_data=

        return  gen_data[0]




def loadBNNdata():
    csvData = pd.read_csv("d:\HW2_data\BNNdata_20080701.csv",sep=',').values
    X=csvData[:,:-1]
    Y=csvData[:,-1]
    print(X.shape[1])
    print(Y)
    #train_rows=np.array([0:679],[729:1408])
    #test_rows=np.array([679:729],[1408:])
    train_x=[]
    train_y=[]
    test_x=[]
    test_y=[]
    #train_x=X[train_rows]
    #train_y=Y[train_rows]

    #test_x=X[test_rows]
    #test_y=Y[test_rows]
    testleng=50
    i=k=j=0
    while i<len(X):
        if Y[i]==1:
            if k<testleng:
                test_x.append(X[i])
                test_y.append(Y[i])
                k=k+1
                i=i+1
            else:
                train_x.append(X[i])
                train_y.append(Y[i])
                j=j+1
                i=i+1
        else:
            if k<2*testleng:
                test_x.append(X[i])
                test_y.append(Y[i])
                k=k+1
                i=i+1
            else:
                train_x.append(X[i])
                train_y.append(Y[i])
                j=j+1
                i=i+1
    #print(np.array(test_x).shape[0])
    #print(np.array(train_x).shape[0])
    #print(test_y)
    return train_x,test_x,train_y,test_y
def loadarffdata(filename):

    with open("d:\HW2_data\heart-statlog.numeric.arff") as f:
        dataDictionary=arff.load(f)
        f.close()
    #print(dataDictionary)
    #arffData=np.array(dataDictionary)
    arffData = np.array(dataDictionary['data'])

    X=arffData[:,:-1]
    Y=arffData[:,-1]
    #print(X)
    #print(Y)
    return X,Y

def load_data(filename):
    '''
    假设这是鸢尾花数据,csv数据格式为：
    0,5.1,3.5,1.4,0.2
    0,5.5,3.6,1.3,0.5
    1,2.5,3.4,1.0,0.5
    1,2.8,3.2,1.1,0.2
    每一行数据第一个数字(0,1...)是标签,也即数据的类别。
     '''

    data = np.genfromtxt(filename, delimiter=',')
    x = data[:, 0:data.shape[1]-1]  # 数据特征
    y = data[:, data.shape[1]-1].astype(int)  # 标签
    scaler = StandardScaler()
    x_std = scaler.fit_transform(x)  # 标准化
    # 将数据划分为训练集和测试集，test_size=.3表示30%的测试集
    x_train, x_test, y_train, y_test = train_test_split(x_std, y, test_size=.1)
    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.1)
    return x_train, x_test, y_train, y_test

def svm_c(x_train, x_test, y_train, y_test):
    '''
    # rbf核函数，设置数据权重
    svc = SVC(kernel='linear', class_weight='balanced',)
    c_range = np.logspace(-5, 15, 11, base=2)
    gamma_range = np.logspace(-9, 3, 13, base=2)
    # 网格搜索交叉验证的参数范围，cv=3,3折交叉
    param_grid = [{'kernel': ['linear'], 'C': c_range, 'gamma': gamma_range}]
    grid = GridSearchCV(svc, param_grid, cv=3, n_jobs=-1)
    # 训练模型
    clf = grid.fit(x_train, y_train)
    # 计算测试集精度
    score = grid.score(x_test, y_test)
    '''
    #print('score:%s' % score)
    #clf= SVC(kernel='linear')
    #clf=clf.fit(x_train,y_train)
    #predict
    #y_pred=clf.predict(x_test)
    #accuracy=metrics.accuracy_score(y_test,y_pred)
    clf1=SVC(C=0.1, kernel='linear', decision_function_shape='ovr')
    clf1.fit(np.array(x_train), np.array(y_train).ravel())
    score=clf1.score(np.array(x_test),np.array(y_test))

    #print("accruacy:",accuracy)
    return score*100

def to_categorical(y,num_classes=None):
    y=np.array(y,dtype='int')
    input_shape=y.shape
    if input_shape and input_shape[-1]==1 and len(input_shape)>1:
        input_shape=tuple(input_shape[:-1])
    y=y.ravel()
    if not num_classes:
        num_classes=np.max(y)+1
    n=y.shape[0]
    categorical=np.zeros((n,num_classes))
    categorical[np.arange(n),y]=1
    output_shape=input_shape+(num_classes,)
    categorical=np.reshape(categorical,output_shape)
    return categorical

def S_nn(n,m):
    model=Sequential()
    model.add(Dense(100,input_dim=n))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(50))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(100))
    model.add(LeakyReLU(alpha=0.2))
    #model.add(Activation('sigmoid'))
    model.add(Dense(m))
    model.add(Activation('softmax'))
    return model
def train_nn(x_train, x_test, y_train, y_test):
    x_train=np.array(x_train)
    x_test=np.array(x_test)
    y_train=np.array(y_train)
    y_test=np.array(y_test)
    featurenum=np.array(x_train).shape[1]
    classnum=2
    y_train=to_categorical(y_train)
    y_test=to_categorical(y_test)
    model=S_nn(featurenum,classnum)
    sgd=SGD(lr=0.01)
    model.compile(optimizer=sgd,loss='mse',metrics=['accuracy'])
    #model.summary()
    model_info=model.fit(x_train,y_train,epochs=10,verbose=2)
    y_pred=model.predict(x_test)
    #print(y_pred)
    y_pred=np.argmax(y_pred,axis=1)
    #print(y_pred)
    y_test=np.argmax(y_test,axis=1)
    #print(y_test)
    num_correct=np.sum(y_pred==y_test)
    #print(num_correct)
    accuracy=float(num_correct)/y_pred.shape[0]
    return (accuracy*100)

'''
train_x, test_x, train_y,test_y=loadBNNdata()
#x_train, x_test, y_train,y_test=load_data('d:\HW2_data\BNNdata_20080701.csv')
#print('train',x_train.shape[0])
#print('test',x_test.shape[0])
#print(y_test.sum())
svm_c(train_x, test_x, train_y,test_y)
print('nn accuracy:',train_nn(train_x, test_x, train_y,test_y))
'''


def loadarffdata(filename):

    #with open(filename) as f:
        #dataDictionary=arff.load(f)
        #f.close()
    #print(dataDictionary)
    #arffData = np.array(dataDictionary['data'])
    #X=arffData[:,:-1]
    #Y=arffData[:,-1]

    data=arff.loadarff(filename)
    #print(data)
    arffData=pd.DataFrame(data[0])
    #print(arffData)

    arffData=arffData.replace({b'recurrence-events':1,b'no-recurrence-events':0})
    #print(arffData)
    arffData_all=list(arffData.shape)[0]
    arffData_class=list(arffData['Class'].value_counts())
    #print(arffData_all)
    #print(arffData_class)
    arffData=np.array(arffData,dtype=int)
    #print(arffData)
    X=arffData[:,:-1]
    Y=arffData[:,-1]
    #print(X)
    #print(Y)

    train_x=[]
    train_y=[]
    test_x=[]
    test_y=[]
    k=j=0
    for i in  range(arffData_all):
        if arffData[i][-1]==1:
            if k<50:
                test_x.append(arffData[i][:-1])
                test_y.append(arffData[i][-1])
                k=k+1
            else:
                train_x.append(arffData[i][:-1])
                train_y.append(arffData[i][-1])
        else:
            if j<50:
                test_x.append(arffData[i][:-1])
                test_y.append(arffData[i][-1])
                j=j+1
            else:
                train_x.append(arffData[i][:-1])
                train_y.append(arffData[i][-1])

    print(np.array(test_x).shape)
    print(np.array(train_x).shape)
    #print(np.sum(train_y))
    #print(test_y)
    return X,Y,train_x,test_x,train_y,test_y





'''
    features_mean=list(arffData.columns[0:9])
    plt.figure(figsize=(10, 10))
    sns.heatmap(arffData[features_mean].corr(), annot=True, square=True, cmap='coolwarm')
    plt.show()

    color_dic={'M':'red','B':'blue'}
    colors=arffData['Class'].map(lambda x:color_dic.get(x))
    sm=pd.plotting.scatter_matrix(arffData[features_mean],alpha=0.4,figsize=((15,15)))
    plt.show()

    bins = 12
    plt.figure(figsize=(15, 15))
    for i, feature in enumerate(features_mean):
        rows = int(len(features_mean) / 2)

        plt.subplot(rows+1, 2, i + 1)

        sns.distplot(arffData[arffData['Class'] == b'recurrence-events'][feature], bins=bins, color='red', label='M');
        sns.distplot(arffData[arffData['Class'] == b'no-recurrence-events'][feature], bins=bins, color='blue', label='B');

        plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15, 15))
    for i, feature in enumerate(features_mean):
        rows = int(len(features_mean) / 2)

        plt.subplot(rows+1, 2, i + 1)

        sns.boxplot(x='Class', y=feature, data=arffData, palette="Set1")

    plt.tight_layout()
    plt.show()
'''





def loadallarffdata():
    file1=r'D:\Fabian\DeepLearning\hw2\HW2_data\breast-cancer.numeric.arff'
    return loadarffdata(file1)
def smote_balance(x,y):

    x_resampled,y_resampled=SMOTE().fit_resample(x,y)
    return x_resampled,y_resampled

def adasyn_balance(x,y):
    x_resampled,y_resampled=ADASYN().fit_resample(x,y)
    return x_resampled,y_resampled

def random_balance(x,y):
    ros=RandomOverSampler(random_state=0)
    x_resampled,y_resampled=ros.fit_resample(x,y)
    return x_resampled,y_resampled

X,Y,train_x,test_x,train_y,test_y=loadallarffdata()

accuracysvm1=svm_c(train_x, test_x, train_y,test_y)
accuracynn1=train_nn(train_x, test_x, train_y,test_y)


cgan1=CGAN()
cgan1.train(X,Y,1000)
train_x1=train_x.copy()
train_y1=train_y.copy()
for i in range(126):
    gen_data = cgan1.sample_data()
    train_x1.append(gen_data[0])
    train_y1.append(1)
print(np.array(train_x).shape)
accuracysvm2=svm_c(train_x1, test_x, train_y1,test_y)
accuracynn2=train_nn(train_x1, test_x, train_y1,test_y)

