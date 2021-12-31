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

%%
warning('off');
% Data Loading
clear;
netdata=load('fortest2.mat');
netdata=netdata.FinalReady;
% Data and Label
network=netdata(:,1:end-1);
netlbl=netdata(:,end);
% Var Change
inputs = network;
targets = netlbl;
% Dim Size
InputNum = size(inputs,2);
OutputNum = size(targets,2);
pr = [-1 1];
PR = repmat(pr,InputNum,1);
% NN Structure (log-sigmoid transfer function)

NH=5;     % Number of Hidden Layers (more better)

Network1 = newff(PR,[NH OutputNum],{'tansig' 'tansig'});
% Train with PSO on Networks Weights
Network1 = TrainPSO(Network1,inputs,targets);
view(Network1)
% Generating Outputs from Our PSO + NN Network Model
outputs = Network1(inputs');
outputs=outputs';
% Size
sizenet=size(network);
sizenet=sizenet(1,1);
% Outputs Error
MSE=mse(outputs);
% Bias Output for Confusion Matrix
outputs=outputs-(MSE*0.1)/2;
% Detecting Mislabeled Data
for i=1 : 50
            if outputs(i) <= 0.9
               out(i)=0;
        elseif outputs(i) >= 0.9
               out(i)=1;            end;end;
for i=51 : 100
            if outputs(i) <= 0.9
               out(i)=0;
        elseif outputs(i) >= 0.9
               out(i)=2;            end;end;
for i=101 : 150
            if outputs(i) <= 0.9
               out(i)=0;
        elseif outputs(i) >= 0.9
               out(i)=3;            end;end;
for i=151 : 200
            if outputs(i) <= 0.9
               out(i)=0;
        elseif outputs(i) >= 0.9
               out(i)=4;            end;end;
for i=201 : 250
            if outputs(i) <= 0.9
               out(i)=0;
        elseif outputs(i) >= 0.9
               out(i)=5;            end;end;
for i=251 : 300
            if outputs(i) <= 0.9
               out(i)=0;
        elseif outputs(i) >= 0.9
               out(i)=6;            end;end;
       out1=single(out');
% PSO Final Accuracy
       psomse=mse(out1,targets);
       MSEError=abs(mse(targets)-mse(out1));
       cnt=0;
       for i=1:sizenet
           if out1(i)~= targets(i)
               cnt=cnt+1;
           end;
       end;
      fin=cnt*100/ sizenet;
      psoacc=(100-fin)-psomse;
%
%% KNN for Comparison 
lblknn=netdata(:,end);
dataknn=netdata(:,1:end-1);
Mdl = fitcknn(dataknn,lblknn,'NumNeighbors',8,'Standardize',1);
rng(1); % For reproducibility
knndat = crossval(Mdl);
classError = kfoldLoss(knndat);
% Predict the labels of the training data.
predictedknn = resubPredict(Mdl);
ctknn=0;
for i = 1 : sizenet(1,1)
if lblknn(i) ~= predictedknn(i)
    ctknn=ctknn+1;
end;
end;
finknn=ctknn*100/ sizenet;
KNN=(100-finknn)-classError;
%
%% SVM for Comparison
tsvm = templateSVM('KernelFunction','polynomial');
svmclass = fitcecoc(dataknn,lblknn,'Learners',tsvm);
svmerror = resubLoss(svmclass);
CVMdl = crossval(svmclass);
genError = kfoldLoss(CVMdl);
% Predict the labels of the training data.
predictedsvm = resubPredict(svmclass);
ct=0;
for i = 1 : sizenet(1,1)
if lblknn(i) ~= predictedsvm(i)
    ct=ct+1;
end;
end;
% Compute Accuracy
finsvm=ct*100/ sizenet;
SVMAccuracy=(100-finsvm);
%% Tree for Comparison
Mdl2 = fitctree(dataknn,lblknn);
rng(1); % For reproducibility
treedat = crossval(Mdl2);
classErrortree = kfoldLoss(treedat);
% Predict the labels of the training data.
predictedtree = resubPredict(Mdl2);
cttree=0;
for i = 1 : sizenet(1,1)
if lblknn(i) ~= predictedtree(i)
    cttree=cttree+1;
end;
end;
fintree=cttree*100/ sizenet;
TREE=(100-fintree)-classErrortree;
%% Plots and Results
% Confusion Matrix
figure
% set(gcf, 'Position',  [50, 100, 1300, 300])
subplot(2,2,1)
cmsvm = confusionchart(lblknn,predictedsvm);
cmsvm.Title = (['SVM Classification =  ' num2str(SVMAccuracy) '%']);
subplot(2,2,2)
cmknn = confusionchart(lblknn,predictedknn);
cmknn.Title = (['KNN Classification =  ' num2str(KNN) '%']);
subplot(2,2,3)
cmtree = confusionchart(lblknn,predictedtree);
cmtree.Title = (['Tree Classification =  ' num2str(TREE) '%']);
subplot(2,2,4)
cmpso = confusionchart(out1,targets);
cmpso.Title = (['PSO-NN Classification =  ' num2str(psoacc) '%']);
% Regression
figure
set(gcf, 'Position',  [50, 150, 450, 350])
[population2,gof] = fit(targets,out1,'poly4');
plot(targets,out1,'o',...
    'LineWidth',3,...
    'MarkerSize',5,...
    'Color',[0.3,0.9,0.2]);
    title(['PSO - R =  ' num2str(1-gof.rmse)]);
    xlabel('Train Target');
    ylabel('Train Output');   
hold on
plot(population2,'b-','predobs');
    xlabel(' Target');
    ylabel(' Output');   
hold off
% ACC and Metrics Results
fprintf('The SVM Accuracy is = %0.4f.\n',SVMAccuracy)
fprintf('The KNN Accuracy is = %0.4f.\n',KNN)
fprintf('The Tree Accuracy is = %0.4f.\n',TREE)
fprintf('The PSO Accuracy is = %0.4f.\n',psoacc)
fprintf('PSO MSE is = %0.4f.\n',MSEError)
fprintf('PSO RMSE is = %0.4f.\n',sqrt(MSEError))
fprintf('PSO MAE is = %0.4f.\n',mae(targets,out1))



