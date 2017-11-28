
import sys
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import mean_squared_error

def recognize():
    
    # do recognize and evaluation
    thres = 0.4     # thres, tune to prec=recall, if smaller, make prec smaller
    n_labels = len( cfg.labels )
    
    gt_roll = []
    pred_roll = []
    result_roll = []

    # For each class
    y_true_binary_c = []
    y_true_file_c = []
    y_true_binary_m = []
    y_true_file_m = []
    y_true_binary_f = []
    y_true_file_f = []
    y_true_binary_v = []
    y_true_file_v = []
    y_true_binary_p = []
    y_true_file_p = []
    y_true_binary_b = []
    y_true_file_b = []
    y_true_binary_o = []
    y_true_file_o = []
    pred_roll_c=[]
    gt_roll_c=[]
    pred_roll_m=[]
    gt_roll_m=[]
    pred_roll_f=[]
    gt_roll_f=[]
    pred_roll_v=[]
    gt_roll_v=[]
    pred_roll_p=[]
    gt_roll_p=[]
    pred_roll_b=[]
    gt_roll_b=[]
    pred_roll_o=[]
    gt_roll_o=[]

    # Loading the data 
    # For RMS
    # #list_audios = open('RMS_Results/list_audios.npy','r')
    # list_audios = np.load('RMS_Results/list_audios.npy')
    # y_pred = np.load('RMS_Results/y_pred.npy')
    # y_real = np.load('RMS_Results/y_real.npy')
    # y_sig = np.load('RMS_Results/y_sig.npy')

    # For grad
    #list_audios = open('RMS_Results/list_audios.npy','r')
    list_audios = np.load('GRAD_Results/list_audios.npy')
    y_pred = np.load('GRAD_Results/y_pred.npy')
    y_real = np.load('GRAD_Results/y_real.npy')
    y_sig = np.load('GRAD_Results/y_sig.npy')

    print list_audios.shape
    print y_pred.shape
    print y_real.shape
    print y_sig.shape

    # making the string on the correct way
    for i in xrange(0,list_audios.shape[0]):
        chain = str(list_audios[i])
        chain = chain.replace('your_dir/', '')
        chain = chain.replace('.16kHz.wav', '')
        list_audios[i] = chain

    # Getting the data for every category
    y_true_file_c = list_audios
    y_true_binary_c = y_real[:,0]  * 1.0
    pred_roll_c = y_pred[:,0] * 1.0
    gt_roll_c = y_real[:,0] * 1.0

    y_true_file_m= list_audios
    y_true_binary_m= y_real[:,1] * 1.0
    pred_roll_m= y_pred[:,1] * 1.0
    gt_roll_m= y_real[:,1] * 1.0

    y_true_file_f= list_audios
    y_true_binary_f= y_real[:,2] * 1.0
    pred_roll_f= y_pred[:,2] * 1.0
    gt_roll_f= y_real[:,2] * 1.0

    y_true_file_v= list_audios
    y_true_binary_v= y_real[:,3] * 1.0
    pred_roll_v= y_pred[:,3] * 1.0
    gt_roll_v= y_real[:,3] * 1.0

    y_true_file_p= list_audios
    y_true_binary_p= y_real[:,4] * 1.0
    pred_roll_p= y_pred[:,4] * 1.0
    gt_roll_p= y_real[:,4] * 1.0

    y_true_file_b= list_audios
    y_true_binary_b= y_real[:,5] * 1.0
    pred_roll_b= y_pred[:,5] * 1.0
    gt_roll_b= y_real[:,5] * 1.0

    y_true_file_o= list_audios
    y_true_binary_o= y_real[:,6] * 1.0
    pred_roll_o= y_pred[:,6] * 1.0
    gt_roll_o= y_real[:,6] * 1.0

    pred_roll = y_pred * 1.0
    gt_roll = y_real * 1.0


    prec, recall, fvalue, support = precision_recall_fscore_support( pred_roll, gt_roll , average='micro')
    prec_c, recall_c, fvalue_c, support_c = precision_recall_fscore_support( pred_roll_c, gt_roll_c , average='micro')
    prec_m, recall_m, fvalue_m, support_m = precision_recall_fscore_support( pred_roll_m, gt_roll_m , average='micro')
    prec_f, recall_f, fvalue_f, support_f = precision_recall_fscore_support( pred_roll_f, gt_roll_f , average='micro')
    prec_v, recall_v, fvalue_v, support_v = precision_recall_fscore_support( pred_roll_v, gt_roll_v , average='micro')
    prec_p, recall_p, fvalue_p, support_p = precision_recall_fscore_support( pred_roll_p, gt_roll_p , average='micro')
    prec_b, recall_b, fvalue_b, support_b = precision_recall_fscore_support( pred_roll_b, gt_roll_b , average='micro')
    prec_o, recall_o, fvalue_o, support_o = precision_recall_fscore_support( pred_roll_o, gt_roll_o , average='micro')

    EER_c= mean_squared_error( pred_roll_c, gt_roll_c )
    EER_m= mean_squared_error( pred_roll_m, gt_roll_m )
    EER_f= mean_squared_error( pred_roll_f, gt_roll_f )
    EER_v= mean_squared_error( pred_roll_v, gt_roll_v )
    EER_p= mean_squared_error( pred_roll_p, gt_roll_p )
    EER_b= mean_squared_error( pred_roll_b, gt_roll_b )
    EER_o= mean_squared_error( pred_roll_o, gt_roll_o )

    print 'EER'
    print 'EER_c ' + str(EER_c)
    print 'EER_m ' + str(EER_m)
    print 'EER_f ' + str(EER_f)
    print 'EER_v ' + str(EER_v)
    print 'EER_p ' + str(EER_p)
    print 'EER_b ' + str(EER_b)
    print 'EER_o ' + str(EER_o)

    print 'Precision'
    print 'Precision_c ' + str(prec_c)
    print 'Precision_m ' + str(prec_m)
    print 'Precision_f ' + str(prec_f)
    print 'Precision_v ' + str(prec_v)
    print 'Precision_p ' + str(prec_p)
    print 'Precision_b ' + str(prec_b)
    print 'Precision_o ' + str(prec_o)

    print 'Recall'
    print 'Recall_c ' + str(recall_c)
    print 'Recall_m ' + str(recall_m)
    print 'Recall_f ' + str(recall_f)
    print 'Recall_v ' + str(recall_v)
    print 'Recall_p ' + str(recall_p)
    print 'Recall_b ' + str(recall_b)
    print 'Recall_o ' + str(recall_o)

    print 'Fvalue'
    print 'Fvalue_c ' + str(fvalue_c)
    print 'Fvalue_m ' + str(fvalue_m)
    print 'Fvalue_f ' + str(fvalue_f)
    print 'Fvalue_v ' + str(fvalue_v)
    print 'Fvalue_p ' + str(fvalue_p)
    print 'Fvalue_b ' + str(fvalue_b)
    print 'Fvalue_o ' + str(fvalue_o)


    EER=(EER_c+EER_m+EER_v+EER_p+EER_f+EER_b+EER_o)/7.0
    prec2=(prec_c+prec_m+prec_f+prec_v+prec_p+prec_b+prec_o)/7.0
    recall2=(recall_c+recall_m+recall_f+recall_v+recall_p+recall_b+recall_o)/7.0
    fvalue2=(fvalue_c+fvalue_m+fvalue_f+fvalue_v+fvalue_p+fvalue_b+fvalue_o)/7.0

    print 'Prec: '+str(prec) + ' Recall: ' + str(recall) + ' fvalue: ' +str(fvalue)
    print 'Prec_2: '+str(prec2) + ' Recall_2: ' + str(recall2) + ' fvalue_2: ' +str(fvalue2)
    print 'EER: ' + str(EER)


if __name__ == '__main__':
    recognize()
