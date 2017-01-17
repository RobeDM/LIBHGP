/*
 ============================================================================
 Author      : Roberto Diaz Morales
 ============================================================================
 
 Copyright (c) 2017 Roberto Díaz Morales
 Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files
 (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge,
 publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
 subject to the following conditions:
 The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
 FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 
 ============================================================================
 */

/**
 * @file IOFunctions.h
 * @author Roberto Diaz Morales
 * @date 17 Jan 2017
 * @brief Input and Output structures and procedures.
 *
 * Input and Output structures and procedures.
 */


#ifndef IOSTRUCTURES_
#define IOSTRUCTURES_

#include <stdio.h>

/**
 * @brief Training parameters.
 *
 * This struct stores the training parameters.
 */

typedef struct properties{
    double *kernelHyperParam; /**< Hyperparameters of the kernel function. */    
    int kernelType; /**< The kernel function (linear=0, rbf=1). */
    double *noiseParam; /**< Power noise. */    
    int Threads; /**< Number of threads to parallelize the operations. */
    double Eta; /**< Convergence criteria. */
}properties;


/**
 * @brief Testing parameters.
 *
 * This struct stores the testing parameters.
 */

typedef struct predictProperties{
    int Labels; /**< If the dataset to test is labeled. */
    int Threads; /**< Number of threads to make the predictions on the dataset. */
}predictProperties;


/**
 * @brief It represents a trained model.
 *
 * This structures saves all the variables of a trained model needed to classify future data.
 */

typedef struct model{
    int kernelType; /**< The kernel function (linear=0, rbf=1). */
    double *kernelHyperParam; /**< Hyperparameters of the kernel function. */
    
    int nData; /**< To tell if the datasets are sparse or not. */

    double *weights; /**< The weight associated to every support vector. */   
    double bias; /**< The bias term of the classification function. */    
    
    struct gp_sample **x; /**< The support vectors. */    
    double *quadratic_value; /**< Array that contains the norm L2 of every sample. */    
    int nElem; /**< Number of features distinct than zero in the dataset. */       
    int maxdim; /**< Number of dimensions of the dataset. */
    struct gp_sample* features; /**< Array of features.*/  
}model;


/**
 * @brief A single feature of a data.
 *
 * This structure represents a single feature of a data. It is composed of a features index and its value.
 */

typedef struct gp_sample{
    int index; /**< The feature index. */   
    double value; /**< The feature value. */   
}gp_sample;


/**
 * @brief A dataset.
 *
 * This structure represents a dataset, a collection of samples and its associated target.
 */

typedef struct gp_dataset{
    int l; /**< If the dataset is labeled or not. */   
    int sparse; /**< If the dataset is sparse or not. */   
    int maxdim; /**< The number of features of the dataset. */   
    double *y; /**< The label of every sample. */   
    struct gp_sample **x; /**< Pointer to the first feature of every sample. */   
    double *quadratic_value; /**< The L2 norm of every sample. It is used to compute kernel functions faster.*/
    struct gp_sample* features; /**< Array of features.*/  
}gp_dataset;

/**
 * @brief Free dataset memory
 *
 * Free memory allocated by a dataset.
 * @param data The dataset
 */

void freeDataset (gp_dataset data);

/**
 * @brief Free model memory
 *
 * Free memory allocated by a model.
 * @param data The model
 */

void freeModel (model modelo);

/**
 * @brief It reads a file that contains a labeled dataset in libsvm format.
 *
 * It reads a file that contains a labeled dataset in libsvm format, the format is the following one:
 * +0.3 1:5 7:2 15:6
 * +1.1 1:5 7:2 15:6 23:1
 * -1.6 2:4 3:2 10:6 11:4
 * ...
 *
 * @param filename A string with the name of the file that contains the dataset.
 * @return The struct with the dataset information.
 */

gp_dataset readTrainFile(char filename[]);


/**
 * @brief It reads a file that contains an unlabeled dataset in libsvm format.
 *
 * It reads a file that contains an unlabeled dataset in libsvm format. The format si the following one:
 * 1:5 7:2 15:6
 * 1:5 7:2 15:6 23:1
 * 2:4 3:2 10:6 11:4
 * ...
 *
 * @param filename A string with the name of the file that contains the dataset.
 * @return The struct with the dataset information.
 */

gp_dataset readUnlabeledFile(char filename[]);


/**
 * @brief It stores a trained model into a file.
 *
 * It stores the struct of a trained model into a file.
 * @param mod The struct with the model to store.
 * @param Output The name of the file.
 */

void storeModel(model * mod, FILE *Output);

/**
 * @brief It loads a trained model from a file.
 *
 * It loads a trained model
 * @param mod The pointer with the struct to load results.
 * @param Input The name of the file.
 */

void readModel(model * mod, FILE *Input);


/**
 * @brief It writes the content of a double array into a file.
 *
 * It writes the content of a double array into a file. It is used to save the predictions of a model on a dataset.
 * @param fileoutput The name of the file.
 * @param predictions The array with the information to save.
 * @param size The length of the array.
 */

void writeOutput (char fileoutput[], double *predictions, int size);

#endif

