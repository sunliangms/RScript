# In this script, we demonstrate how to use the functions provided in classification_measures.R
# to compute the performance metrics in both binary-class and multi-class classification
# Also many examples are borrowed from scikit.learn directly, so our consistent results
# also demostrate the correctness of our codes.

source('classification_measures.R')


#============== Binary-Class Classification =================

# Compute AUC
y_true = c(0, 0, 1, 1)
y_pred = c(0.1, 0.4, 0.35, 0.8)
auc1 = AUC(y_true, y_pred)
# auc1 should be 0.75

# Compute accuracy, precision, recall, and F1
y_true = c(0, 0, 1, 1, 1)
y_pred = c(0, 0, 0, 1, 1)

# Compute confusion matrix first
Confb1 = confusion_binary(y_true, y_pred)
Mb1 = Confb1$confusionMatrix
# Mb1 should be [2, 1; 0, 2]

# Compute accuracy
accb1 = accuracyY(y_true, y_pred)
accb2 = accuracyConf(Mb1)
# both accb1 and accb2 should be 0.8

# Compute precision, recall and F1 for binary-class classification
pb1 = precision_binary(Mb1)
# the precision pb1 should be 1
rb1 = recall_binary(Mb1)
# the recall rb1 should be 0.6666667
F1b1 = F1_binary(Mb1)
# the F1 F1b1 should be 0.8




#============== Multi-Class Classification ==============
# Compute the confusion matrix
y_true = c(2, 0, 2, 2, 0, 1)
y_pred = c(0, 0, 2, 2, 0, 2)
Conf1 = confusion_multi(y_true, y_pred)
# output should be 
# array([[2, 0, 0],
#        [0, 0, 1],
#        [1, 0, 2]])

# Compute the accuracy
y_pred = c(0, 2, 1, 3)
y_true = c(0, 1, 2, 3)
Conf2 = confusion_multi(y_true, y_pred)
acc1c = accuracyConf(Conf2$confusionMatrix)
acc1y = accuracyY(y_true, y_pred)
# acc1 and acc1b should b 0.5


# Test micro and macro-averaged F1
y_true = c(0, 1, 2, 0, 1, 2)
y_pred = c(0, 2, 1, 0, 0, 1)
Conf1 = confusion_multi(y_true, y_pred)
M1 = Conf1$confusionMatrix
macroF1_1 = macroF1(M1)
# macroF1_1 should be 0.26
microF1_1 = microF1(M1)
# microF1_1 should be 0.33...
# compute F1 list for each individual label
F1_list = F1_multi(M1)
# F1_list should be c(0.8,  0.0 ,  0.0)
# compute precision for each individual label
precision_list = precision_multi(M1)
# precision_list should be c(0.6666667 0.0000000 0.0000000)
# compute recall for each individual label
recall_list = recall_multi(M1)
# recall_list should be c(1 0 0)

# Test MAUC for multi-class classification
g_vec = c(1,2,3,2,2,1,1)
P_matrix = cbind(c(0.9,0.4,0.5,0.4,0.6,0.2,0.3), c(0.4,0.6,0.2,0.5,0.8,0.9,0.3),c(0.3,0.2,0.7,0.3,0.5,0.6,0.6))
mauc = MAUC(g_vec, P_matrix)
# mauc is 0.7222222
