# PSO-Classification-Algorithm
%% PSO Neural Network Classification
% So, there was no proper evolutionary classification Matlab code in the web,
% Which decided to make one with PSO.
% This code gets data input for classification which contains data and 
% labels and stores it into 'netdata'. data consists of 300 samples for 6 
% classes which includes 40 features. You can extract your features and
% label it as it is a supervised model. These features are extracted SURF
% features out of small objects images. Now, System is combination of PSO
% and typical shallow neural network. Neural network itself makes the
% initial structure or body of the system but PSO has duty of weighting the
% neurons in training, which true power of evolutionary algorithms present
% here. Finally, result compared with SVM, KNN, and TREE classification 
% algorithms as confusion matrix and final recognition accuracy.
% There are three important parameters of 'NH' (number of hidden layers),
% 'SwarmSize' and 'MaxIteration' which effect the performance of thee
% system significantly. So, in order to get desired result, you should play
% with these parameters based on your data. The only drawback here is that
% labeling is done manually, which you can fix it yourself easily, but it
% is after main stage of training. This code could be expanded to be
% trained with other evolutionary algorithms such as GA or DE. If you find
% any problem, please contact me as below:
 
% Email: mosavi.a.i.buali@gmail.com
% Author: Seyed Muhammad Hossein Mousavi
 
% Also this code is part of the following project, so please cite below 
% after using the code: 
 
% Mousavi, Seyed Muhammad Hossein, et al. "A PSO fuzzy-expert system: As an
% assistant for specifying the acceptance by NOET measures, at PH. D 
% level." 2017 Artificial Intelligence and Signal Processing Conference
% (AISP). IEEE, 2017.
 
% Thank you for citing the paper and enjoy the code (hope it help you (Be happy :)
![PSO Classification Algorithm](https://user-images.githubusercontent.com/11339420/147828020-78672c9d-8ac2-4175-b4e5-277af543483e.JPG)

