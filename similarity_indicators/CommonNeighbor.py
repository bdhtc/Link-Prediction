#coding=UTF-8
'''
Created on 2016年11月15日

@author: ZWT
'''
import numpy as np
import time

def Cn(MatrixAdjacency_Train):
    similarity_StartTime = time.perf_counter()
    Matrix_similarity = np.dot(MatrixAdjacency_Train,MatrixAdjacency_Train)
    
    similarity_EndTime = time.perf_counter()
    print("    SimilarityTime: %f s" % (similarity_EndTime - similarity_StartTime))
    return Matrix_similarity

