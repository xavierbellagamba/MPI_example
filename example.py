import numpy as np
from mpi4py import MPI
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import os
import sys
from datetime import datetime

def generateIndKFold(K, n):
    '''
    Returns the data indices of the K folders
    '''
    return np.random.randint(0, K, n)


def createTrainValDataset(X, y, ind_K, k):
    ind_K_train = []
    ind_K_val = []
    for i in range(len(ind_K)):
        if ind_K[i] == k:
            ind_K_val.append(i)
        else:
            ind_K_train.append(i)

    X_val = X[ind_K_val, :]
    X_train = X[ind_K_train, :]
    
    y_val = y[ind_K_val]
    y_train = y[ind_K_train]

    return X_train, y_train, X_val, y_val

def main():
    #Initialize MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    rootRank = 0

    
    #Load data and prepare CV folder
    '''
    All cores will load the data, then the manager will send the indices 
    to the workers to enable the creation of the training and test sets
    '''
    pwd = os.getcwd()
    csv_path = os.path.join(pwd, 'data.csv')
    data = np.asarray(pd.read_csv(csv_path))
    X = data[:, 3:-2]
    y = data[:, -1]
    y = y.astype(str)

    K = 5
    ind_K = np.zeros(y.shape).astype(int)

    #Parameters to test
    learning_rate = [0.05, 0.01, 0.005, 0.001]
    N_estimator = [20, 50, 100, 200, 500, 750]


    #Manager
    if rank == rootRank:
        startTime = datetime.now()
        #Prepare CV
        '''
        Indices are yet unavailableto the workers
        '''
        ind_K = generateIndKFold(K, y.shape[0])

    #Broadcasting indices
    comm.barrier()#Very important: without it, the workers who did nothing would have gone through without receiving the info
    comm.Bcast([ind_K, MPI.INT], rootRank)
    '''
    Now indices are available to every core, but only rootRank has performed the computation
    '''

    #Manager
    if rank == rootRank:
        #Create job list 
        job = []
        for i_lr in range(len(learning_rate)):
            for i_ne in range(len(N_estimator)):
                for k in range(K):
                    job.append([i_lr, i_ne, k])

        #Number of workers
        n_worker = size-1

        #Send initial jobs
        n_job = len(job)
        id_next_job = 0
        worker_lib = np.zeros((n_worker))
        for i in range(1, size):
            #Send index of job
            '''
            Tag ensure the receiving core has the right information
            dest indicates the core the info must be sent to
            '''
            comm.send(job[i-1], dest=i, tag=0)
            id_next_job = id_next_job + 1 #Required to keep track of the right next job

        #Initialize receiver matrices and variables
        job_compl = 0 #Number of completed jobs
        CV_err = [[[[] for l in range(K)] for j in range(len(N_estimator))] for i in range(len(learning_rate))]

        print('Start the ' + str(K) + '-Fold crossvalidation...')
        sys.stdout.flush()
    
        CV_err_mean = [[[] for k in range(len(N_estimator))] for i in range(len(learning_rate))]
        CV_err_std = [[[] for k in range(len(N_estimator))] for i in range(len(learning_rate))]
    
        #Do the rest
        while(True):
            #Receive the rank of the worker having completed job
            ind_free = -1
            ind_free = comm.recv(source=MPI.ANY_SOURCE, tag=1)
            job_compl = job_compl + 1
        
            #Receive job characteristics
            ind_job = []
            ind_job = comm.recv(source=ind_free, tag=2)
        
            #Receive CV_error_mean of the completed job
            CV_i = -1.0
            CV_i = comm.recv(source=ind_free, tag=3)
        
            #Store results
            CV_err[ind_job[0]][ind_job[1]][ind_job[2]] = CV_i
        
            print('\t Core \t' + str(ind_free) + ':\t' + str(job_compl) + '/' + str(n_job) )
            sys.stdout.flush()
        
            #Check if all jobs are completed
            if id_next_job >= n_job:
                comm.send([-1, -1, -1, -1], dest=ind_free, tag=0)
                worker_lib[ind_free-1] = 1
            
            #If not send next one
            else:
                comm.send(job[id_next_job], dest=ind_free, tag=0)
                id_next_job = id_next_job + 1
            
            #If all workers are free, save results and close program
            if np.sum(worker_lib) == n_worker:
                print('Compute the CV metrics and plot results...')
                sys.stdout.flush()
                break 

        #Once all the job are done, compute the CV metrics (mean and std)
        lr_best = -1.
        N_best = -1
        CV_min = 1000000.0
        CV_min_std = -1.0
        for i_lr in range(len(learning_rate)):
            for i_est in range(len(N_estimator)):
                CV_err_mean[i_lr][i_est] = np.mean(CV_err[i_lr][i_est])
                CV_err_std[i_lr][i_est] = np.std(CV_err[i_lr][i_est])
                if CV_min > CV_err_mean[i_lr][i_est]:
                    CV_min = CV_err_mean[i_lr][i_est]
                    CV_min_std = CV_err_std[i_lr][i_est]
                    lr_best = learning_rate[i_lr]
                    N_best = N_estimator[i_est]

        print('Best learning rate: ', lr_best)
        print('Best number of estimators: ', N_best)
        print('Mean CV error of best model (accuracy): ', CV_min)
        print('Standard deviation of the best model CV error (accuracy): ', CV_min_std)
        print('Runtime: ', datetime.now() - startTime)

    #Worker
    else:
        #Receive initial index of image
        job_i = [-1, -1, -1]
        job_i = comm.recv(source=rootRank, tag=0)
    
        while not job_i[0] == -1:        
            #Perform job
            #Create training folders
            X_train, y_train, X_val, y_val = createTrainValDataset(X, y, ind_K, job_i[2])

            #Create and fit the model
            BT = GradientBoostingClassifier(learning_rate=learning_rate[job_i[0]], n_estimators=N_estimator[job_i[1]], 
                min_samples_split=4, min_samples_leaf=2, subsample=0.5, max_depth=4, max_features=3, 
                verbose=0, validation_fraction=0.1, n_iter_no_change=5)
            BT = BT.fit(X_train, y_train)
        
            #Test on validation set
            y_val_hat = BT.predict(X_val)
        
            #Estimate the error
            accuracy_count = 0
            for k in range(len(y_val)):
                if y_val[k] == y_val_hat[k]:
                    accuracy_count = accuracy_count + 1
            accuracy = 100.*float(accuracy_count)/float(len(y_val))
            CV_i = 1.0-accuracy/100.
        
            #Send results
            comm.send(rank, dest=rootRank, tag=1)
            comm.send(job_i, dest=rootRank, tag=2)
            comm.send(CV_i, dest=rootRank, tag=3)
        
            #Receive next job ID
            job_i = comm.recv(source=rootRank, tag=0)

  
if __name__== "__main__":
    main()
    


