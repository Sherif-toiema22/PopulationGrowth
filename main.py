from tkinter import Tk, Label, Entry, Button

import pandas as pd
import numpy as np
df=pd.read_csv('world_population.csv')
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor
from matplotlib import pyplot
from seaborn import lineplot, displot, distplot, scatterplot, boxplot
import seaborn as sbn
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
#
# # ------------preprocessing---------///
#
#
#
# # -----------negative num in Rank--------//
# # cleaned_Rank=[]
# #
# # num_of_contries=235
# # for m in df.loc[:,'Rank']:
# #     if(m<0):
# #         cleaned_Rank.append(36)
# #         # in this code i hardly try to handle problem of detect the rank directly :)
# #         # for n in sorted_data:
# #         #     if(c!=n):
# #         #         cleaned_Rank.append(n)
# #         #     else:
# #         #         c+=1
# #
# #
# #     else:
# #         cleaned_Rank.append(m)
# #
#
#
#  # ----------non value in capital--------//
# # counter=0
# # location=0
# # c=0
# # cleaned_capital=[]
# # for m in df.loc[:,'Capital']:
# #     counter+=1
# #     if(str(m)=='nan'):
# #         print('ohhh  noooo')
# #         location=counter
# #         for m in df.loc[:, 'Country/Territory']:
# #             c += 1
# #             if (c == location):
# #                 cleaned_capital.append("country name is " + m)
# #                 print()
# #                 break
# #     else:
# #         cleaned_capital.append(m)
#
#
# # for l in cleaned_capital:
# #     print(l)
#  # ----------negative  in population in 2022--------//
# # absolute_value=0
# # c=0
# # high=0
# # counter=0
# # cleaned_2022_Population=[]
# # for m in df.loc[:,'2022 Population']:
# #     c+=1
# #     rate = 0
# #     if(m<=0):
# #         high=c
# #         for m in df.loc[:, 'Growth Rate']:
# #             counter += 1
# #             if (high == counter):
# #                 rate = m
# #                 break
# #         cn = 0
# #         for m in df.loc[:, '2020 Population']:
# #             cn += 1
# #             if (high == cn):
# #                 absolute_value = m * rate
# #                 cleaned_2022_Population.append(absolute_value)
# #
# #     else:
# #         cleaned_2022_Population.append(m)
#
#
#
#
# # for o in cleaned_2022_Population:
# #     print(o)
#
# # -----------------(cleaned Continent)----------------//
# # cleaned_Continent=[]
# # default_Continent = list(df.loc[:, 'Continent'].mode())
# # print(default_Continent)
# # for m in df.loc[:,'Continent']:
# #     if(str(m)=='nan'):
# #         print("oh shit")
# #         cleaned_Continent.append(default_Continent)
# #     else:
# #         cleaned_Continent.append(m)
#
# #         cleaned data         //
#
# # cleaned_df=pd.DataFrame({'Rank':cleaned_Rank,'CCA3':df.loc[:,'CCA3'],'Country/Territory':df.loc[:,'Country/Territory'],
# #                           'Capital':cleaned_capital,'Continenet':cleaned_Continent,'2022 Population':cleaned_2022_Population,
# #                           '2020 population':df.loc[:,'2020 Population'],'2015 Population':df.loc[:,'2015 Population'],
# #                           '2010 Population':df.loc[:,'2010 Population'],'2000 Population':df.loc[:,'2000 Population'],
# #                           '1990 Population':df.loc[:,'1990 Population'],'1980 Population':df.loc[:,'1980 Population'],
# #                           '1970 Population':df.loc[:,'1970 Population'],
# #                           'Growth Rate':df.loc[:,'Growth Rate'],
# #                           'World Population Percentage':df.loc[:,'World Population Percentage']})
# #
# # cleaned_df.to_csv('cleaned-data.csv')
# #
#
#
# # # # --------------------Visulisation-------------//
dataset=pd.read_csv('cleaned-data.csv')
# # lineplot(data=dataset)
# # # distplot(a=dataset['1970 Population'],bins=False)
# # # scatterplot(data=dataset['1970 Population'])
# # # boxplot(data=dataset['1970 Population'])
# # sbn.distplot(dataset['Rank'],hist=False,bins=5)
# # #
# # pyplot.show()
#
# #------------------------Algorithms------------------#



Population=dataset.values
X=Population[:, 7:-1]
y=Population[:, 6]
# print(X)
# print(dataset)
# print(X.shape)
# print(y)
#
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.33,random_state=44,shuffle=True)

# # --------------------Linear Regression Algorithm-------------//

def Linear_Regression_Algorithm():

    LinearRegressionModel = LinearRegression(fit_intercept= True,normalize=True,copy_X=True)
    LinearRegressionModel.fit(X_train,y_train)

    #Accuracy
    # print(LinearRegressionModel.score(X_train,y_train))
    # print(LinearRegressionModel.score(X_test,y_test))

    #Prediction
    # y_pred=LinearRegressionModel.predict(X_test)
    # print(y_pred)


    # # mean_absolute_error()
    # print(mean_absolute_error(y_test,y_pred))
    #
    # # mean_squared_error()
    # print(mean_squared_error(y_test,y_pred))
    #
    # # median_absolute_error()
    # print(median_absolute_error(y_test,y_pred))

    y_predd=int(LinearRegressionModel.predict([[population_2020entry,population_2015entry,population_2010entry,
                                        population_2000entry,population_1990entry,population_1980entry,population_1970entry,rateentry
    ]]))
    print(int(y_predd))
    res=y_predd


    return [y_predd]




# --------------------Logistic Regression Algorithm-------------//
def Logistic_Regression_Algorithm(Algorithmentry):
    Algonum = 2
    res = 334

    LogisticRegressionModel=LogisticRegression(penalty='12',solver='sav',C=1.0,random_state=33)
    LogisticRegressionModel.fit(X_train,y_train)

    #Accuracy
    # print(LogisticRegressionModel.score(X_train,y_train))
    # print(LogisticRegressionModel.score(X_test,y_test))

    #Prediction
    # y_pred=LogisticRegressionModel.predict(X_test)


    # mean_absolute_error()
    # print(mean_absolute_error(y_test,y_pred))

    # mean_squared_error()
    # print(mean_squared_error(y_test,y_pred))

    # median_absolute_error()
    # print(median_absolute_error(y_test,y_pred))
    y_predlogisticmodel=LogisticRegressionModel.predict([[int(population_2020entry),int(population_2015entry),int(population_2010entry),
                                                int(population_2000entry),int(population_1990entry),int(population_1980entry),int(population_1970entry),int(rateentry)
           ]])
    res=y_predlogisticmodel
    return [res]


# --------------------Neural Network Algorithm-------------//
def Neural_Network_Algorithm(Algorithmentry):
    Algonum = 3
    res = 334
    if Algonum == Algorithmentry:

        MLPRegressorModel=MLPRegressor(activation='tanh',solver='lbfgs',learning_rate='constant')
        MLPRegressorModel.fit(X_train,y_train)


        #Accuracy
        print(MLPRegressorModel.score(X_train,y_train))
        print(MLPRegressorModel.score(X_test,y_test))

        #Prediction
        y_pred=MLPRegressorModel.predict(X_test)


        # mean_absolute_error()
        print(mean_absolute_error(y_test,y_pred))

        # mean_squared_error()
        print(mean_squared_error(y_test,y_pred))

        # median_absolute_error()
        print(median_absolute_error(y_test,y_pred))

        y_predMLPmodel=MLPRegressorModel.predict([[population_2020entry,population_2015entry,population_2010entry,population_2000entry,population_1990entry,population_1980entry,population_1970entry,rate
        ]])
        res=y_predMLPmodel
    return [res]

# # # --------------------SVM Algorithm-------------//
def SVM_Algorithm(Algorithmentry):
    Algonum = 4
    res = 334
    if Algonum == Algorithmentry:
        SVRModel = SVR(C=1.0 ,epsilon=.1,kernel='rbf')
        SVRModel.fit(X_train,y_train)


        #Accuracy
        print(SVRModel.score(X_train,y_train))
        print(SVRModel.score(X_test,y_test))

        #Prediction
        y_pred=SVRModel.predict(X_test)
        print(y_pred)

        # mean_absolute_error()
        print(mean_absolute_error(y_test,y_pred))

        # mean_squared_error()
        print(mean_squared_error(y_test,y_pred))

        # median_absolute_error()
        print(median_absolute_error(y_test,y_pred))


        y_predSVMmodel=SVRModel.predict([[population_2020entry,population_2015entry,population_2010entry,
                                          population_2000entry,population_1990entry,population_1980entry,population_1970entry,rate]])


        print(int(y_pred))
        y_predict=SVRModel.predict(X_test)
        print(y_predict[:10])
        res = y_predSVMmodel
    return [res]



# # --------------------KMeans Algorithm-------------//
def KMeans_Algorithm(Algorithmentry):
    Algonum = 5
    res = 334
    if Algonum == Algorithmentry:
        KMeansModel=KMeans(n_clusters=5,init='k-means++',random_state=33,algorithm='auto')
        KMeansModel.fit(X_train,y_train)


        #Accuracy
        print(KMeansModel.score(X_train,y_train))
        print(KMeansModel.score(X_test,y_test))

        #Prediction
        y_pred=KMeansModel.predict(X_test)
        # print(y_pred)

        # mean_absolute_error()
        print(mean_absolute_error(y_test,y_pred))

        # mean_squared_error()
        print(mean_squared_error(y_test,y_pred))

        # median_absolute_error()
        print(median_absolute_error(y_test,y_pred))

        y_predKMeansmodel=KMeansModel.predict([[population_2020entry,population_2015entry,population_2010entry,
                                          population_2000entry,population_1990entry,population_1980entry,population_1970entry,rate]])

        res = y_predKMeansmodel
    return [res]
        #
# # --------------------Naive Bayes Algorithm-------------//
def Naive_Bayes_Algorithm(Algorithmentry):
    Algonum = 6
    res = 334
    if Algonum == Algorithmentry:
        GaussianNBModel=GaussianNB()
        GaussianNBModel.fit(X_train,y_train)


        #Accuracy
        print(GaussianNBModel.score(X_train,y_train))
        print(GaussianNBModel.score(X_test,y_test))

        #Prediction
        y_pred=GaussianNBModel.predict(X_test)


        # mean_absolute_error()
        print(mean_absolute_error(y_test,y_pred))

        # mean_squared_error()
        print(mean_squared_error(y_test,y_pred))

        # median_absolute_error()
        print(median_absolute_error(y_test,y_pred))

        y_predNaiveBayesKmodel=GaussianNBModel.predict([[population_2020entry,population_2015entry,population_2010entry,
                                          population_2000entry,population_1990entry,population_1980entry,population_1970entry,rate]])


        res = y_predNaiveBayesKmodel
    return [res]

# # # --------------------Decision tree Algorithm-------------//
def Decision_tree_Algorithm(Algorithmentry):
    Algonum = 7
    res = 334
    if Algonum == Algorithmentry:
        regressorModel=DecisionTreeRegressor(random_state=0)
        regressorModel.fit(X_train,y_train)

        #Accuracy
        print(regressorModel.score(X_train,y_train))
        print(regressorModel.score(X_test,y_test))

        #Prediction
        y_pred=regressorModel.predict(X_test)


        # mean_absolute_error()
        print(mean_absolute_error(y_test,y_pred))

        # mean_squared_error()
        print(mean_squared_error(y_test,y_pred))

        # median_absolute_error()
        print(median_absolute_error(y_test,y_pred))

        y_predDecisiontreeKmodel=regressorModel.predict([[population_2020entry,population_2015entry,population_2010entry,
                                          population_2000entry,population_1990entry,population_1980entry,population_1970entry,rate]])

        res = y_predDecisiontreeKmodel
    return [res]

# ?????????????????????????????????
top=Tk()
top.title("Population Growth")
top.minsize(500,600)

# enter population_2020
population_2020=Label(text="population_2020")
population_2020.pack()
population_2020entry=Entry()
population_2020entry.pack()

# enter population_2015
population_2015=Label(text="population_2015")
population_2015.pack()
population_2015entry=Entry()
population_2015entry.pack()

# enter population_2010
population_2010=Label(text="population_2010")
population_2010.pack()
population_2010entry=Entry()
population_2010entry.pack()

# enter population_2000
population_2000=Label(text="population_2000")
population_2000.pack()
population_2000entry=Entry()
population_2000entry.pack()

# enter population_1990
population_1990=Label(text="population_1990")
population_1990.pack()
population_1990entry=Entry()
population_1990entry.pack()

# enter population_1980
population_1980=Label(text="population_1980")
population_1980.pack()
population_1980entry=Entry()
population_1980entry.pack()

# enter population_1970
population_1970=Label(text="population_1970")
population_1970.pack()
population_1970entry=Entry()
population_1970entry.pack()

# enter Rate
rate=Label(text="Rate")
rate.pack()
rateentry=Entry()
rateentry.pack()

# enter Algorithm
Algorithm=Label(text="Algorithm Name")
Algorithm.pack()
Algorithmentry=Entry()
Algorithmentry.pack()

# Algorithminput = Algorithmentry


def add1():
    ressss=Linear_Regression_Algorithm(Algorithmentry.get())
    result= Label(text="result is "+str(ressss))
    result.pack()

def add2():
    ressss=Logistic_Regression_Algorithm(Algorithmentry.get())
    result= Label(text="result is "+str(ressss))
    result.pack()

def add3():
    ressss=Neural_Network_Algorithm(Algorithmentry.get())
    result= Label(text="result is "+str(ressss))
    result.pack()

def add4():
    ressss=str(SVM_Algorithm(Algorithmentry.get()))
    result= Label(text="result is "+str(ressss))
    result.pack()

def add5():
    ressss=KMeans_Algorithm(Algorithmentry.get())
    result= Label(text="result is "+str(ressss))
    result.pack()

def add6():
    ressss=Naive_Bayes_Algorithm(Algorithmentry.get())
    result= Label(text="result is "+str(ressss))
    result.pack()

def add7():
    ressss=Decision_tree_Algorithm(Algorithmentry.get())
    result= Label(text="result is "+str(ressss))
    result.pack()

# but=Button(text="Add", command=add)
# but.pack()
#




but1=Button(top,text="Result1", command=add1)
but2=Button(top,text="Result2", command=add2)
but3=Button(top,text="Result3", command=add3)
but4=Button(top,text="Result4", command=add4)
but5=Button(top,text="Result5", command=add5)
but6=Button(top,text="Result6", command=add6)
but7=Button(top,text="Result7", command=add7)

but1.pack()
but2.pack()
but3.pack()
but4.pack()
but5.pack()
but6.pack()
but7.pack()
# but1.pack()



top.mainloop()