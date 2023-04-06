from sklearn import preprocessing
import torch
import numpy as np
import pandas as pd

def data_generating_energy(data_path, num_random, noise_level):
    df = pd.read_csv(data_path + "energy.csv", sep=",")
    df = df.drop(['Unnamed: 10','Unnamed: 11','Y1'], axis = 1)
    df = df.dropna(axis=0)
    np.random.seed(129)
    msk_1 = np.random.rand(len(df)) < 0.8
    traindf = df[msk_1]
    testdf = df[~msk_1]
    # training set
    trainx = traindf.drop('Y2', axis=1).values
    trainy = traindf['Y2'].values
    # test set
    testx = testdf.drop('Y2', axis=1).values
    testy = testdf['Y2'].values

    scalerx = preprocessing.StandardScaler().fit(trainx)
    trainx = scalerx.transform(trainx)
    testx = scalerx.transform(testx)

    scalery = preprocessing.StandardScaler().fit(trainy.reshape(-1,1))
    trainy = scalery.transform(trainy.reshape(-1,1)).reshape(1, -1)
    testy = scalery.transform(testy.reshape(-1,1)).reshape(1, -1)
    
    # transfer data into tensor
    x = torch.tensor(trainx, dtype = torch.float)
    y = torch.tensor(trainy[0], dtype = torch.float).view(-1,1)
    x_test = torch.tensor(testx, dtype = torch.float)
    y_test = torch.tensor(testy[0], dtype = torch.float).view(-1,1)
    
    if num_random != 0:
        rand_features = torch.randn(x.shape[0], num_random) * 2
        x = torch.cat((x, rand_features), 1)
        rand_features = torch.randn(x_test.shape[0], num_random) *2
        x_test = torch.cat((x_test, rand_features), 1)
    
    if noise_level != 0:
        noise = torch.randn_like(y) * np.sqrt(noise_level); noise_test = torch.randn_like(y_test) * np.sqrt(noise_level)
        y = y * np.sqrt(1-noise_level) + noise; y_test = y_test * np.sqrt(1-noise_level) + noise_test;     
        
    num_useful = 8.; num_feature = x.shape[1]

    return x, y, x_test, y_test, num_useful, num_feature

def data_generating_concrete(data_path, num_random, noise_level):
    df = pd.read_csv(data_path + "Concrete_Data.csv", sep=",")
    # df = df.drop(['Unnamed: 10','Unnamed: 11','Y1'], axis = 1)
    df = df.dropna(axis=0)
    np.random.seed(129)
    msk_1 = np.random.rand(len(df)) < 0.8
    traindf = df[msk_1]
    testdf = df[~msk_1]
    # training set
    trainx = traindf.drop('Concrete compressive strength(MPa, megapascals) ', axis=1).values
    trainy = traindf['Concrete compressive strength(MPa, megapascals) '].values
    # test set
    testx = testdf.drop('Concrete compressive strength(MPa, megapascals) ', axis=1).values
    testy = testdf['Concrete compressive strength(MPa, megapascals) '].values

    # le csv est rempli comme de la merde
    for i in range(len(trainx)):
      for j in range(len(trainx[0])):
        if type(trainx[i][j]) == str:
          trainx[i][j] = trainx[i][j].replace(",", ".")
          if trainx[i][j][-1].isspace():
              trainx[i][j] = trainx[i][j][:-1]

      if type(trainy[i]) == str:
          trainy[i] = trainy[i].replace(",", ".")
          if trainy[i][-1].isspace():
              trainy[i] = trainy[i][:-1]

    for i in range(len(testx)):
      for j in range(len(testx[0])):
        if type(testx[i][j]) == str:
          testx[i][j] = testx[i][j].replace(",", ".")
          if testx[i][j][-1].isspace():
              testx[i][j] = testx[i][j][:-1]

      if type(testy[i]) == str:
          testy[i] = testy[i].replace(",", ".")
          if testy[i][-1].isspace():
              testy[i] = testy[i][:-1]

    scalerx = preprocessing.StandardScaler().fit(trainx)
    trainx = scalerx.transform(trainx)
    testx = scalerx.transform(testx)

    scalery = preprocessing.StandardScaler().fit(trainy.reshape(-1,1))
    trainy = scalery.transform(trainy.reshape(-1,1)).reshape(1, -1)
    testy = scalery.transform(testy.reshape(-1,1)).reshape(1, -1)
    
    # transfer data into tensor
    x = torch.tensor(trainx, dtype = torch.float)
    y = torch.tensor(trainy[0], dtype = torch.float).view(-1,1)
    x_test = torch.tensor(testx, dtype = torch.float)
    y_test = torch.tensor(testy[0], dtype = torch.float).view(-1,1)
    
    if num_random != 0:
        rand_features = torch.randn(x.shape[0], num_random) * 2
        x = torch.cat((x, rand_features), 1)
        rand_features = torch.randn(x_test.shape[0], num_random) *2
        x_test = torch.cat((x_test, rand_features), 1)
    
    if noise_level != 0:
        noise = torch.randn_like(y) * np.sqrt(noise_level); noise_test = torch.randn_like(y_test) * np.sqrt(noise_level)
        y = y * np.sqrt(1-noise_level) + noise; y_test = y_test * np.sqrt(1-noise_level) + noise_test;     
        
    num_useful = 9.; num_feature = x.shape[1]

    return x, y, x_test, y_test, num_useful, num_feature

def data_generating_yacht(data_path, num_random, noise_level):
    df = pd.read_fwf(data_path + "yacht_hydrodynamics.data", sep=",")
    # df = df.drop(['Unnamed: 10','Unnamed: 11','Y1'], axis = 1)
    df.set_axis(["x1", "x2", "x3","x4", "x5", "x6","x7"],
                    axis=1,inplace=True)
    df = df.dropna(axis=0)
    np.random.seed(129)
    msk_1 = np.random.rand(len(df)) < 0.8
    traindf = df[msk_1]
    testdf = df[~msk_1]
    print(df, df.columns)
    # training set
    trainx = traindf.drop('x7', axis=1).values
    trainy = traindf['x7'].values
    # test set
    testx = testdf.drop('x7', axis=1).values
    testy = testdf['x7'].values

    scalerx = preprocessing.StandardScaler().fit(trainx)
    trainx = scalerx.transform(trainx)
    testx = scalerx.transform(testx)

    scalery = preprocessing.StandardScaler().fit(trainy.reshape(-1,1))
    trainy = scalery.transform(trainy.reshape(-1,1)).reshape(1, -1)
    testy = scalery.transform(testy.reshape(-1,1)).reshape(1, -1)
    
    # transfer data into tensor
    x = torch.tensor(trainx, dtype = torch.float)
    y = torch.tensor(trainy[0], dtype = torch.float).view(-1,1)
    x_test = torch.tensor(testx, dtype = torch.float)
    y_test = torch.tensor(testy[0], dtype = torch.float).view(-1,1)
    
    if num_random != 0:
        rand_features = torch.randn(x.shape[0], num_random) * 2
        x = torch.cat((x, rand_features), 1)
        rand_features = torch.randn(x_test.shape[0], num_random) *2
        x_test = torch.cat((x_test, rand_features), 1)
    
    if noise_level != 0:
        noise = torch.randn_like(y) * np.sqrt(noise_level); noise_test = torch.randn_like(y_test) * np.sqrt(noise_level)
        y = y * np.sqrt(1-noise_level) + noise; y_test = y_test * np.sqrt(1-noise_level) + noise_test;     
        
    num_useful = 6.; num_feature = x.shape[1]

    return x, y, x_test, y_test, num_useful, num_feature
