clc; clearvars; clear all;
load("vehicles_.mat");  

% Dividir train (80 %) / test (20 %)

N       = numel(vehicles);
rng(42);
idx     = randperm(N);
nTest   = round(0.20 * N);
idxTest = idx(1:nTest);
idxTrain= idx(nTest+1:end);

vTrain  = vehicles(idxTrain);
vTest   = vehicles(idxTest);

% Etiquetas
hTrain  = [vTrain.high];
hTest   = [vTest.high];

% Discretizar en 3 clases 

edges      = linspace(-1,1,4);
hMin       = min(hTrain);
hMax       = max(hTrain);
hTN        = map(hTrain,hMin,hMax,0.1,1);
yLog       = log10(hTN);
yMin       = min(yLog);   
yMax       = max(yLog);
yNormTrain = map(yLog,yMin,yMax,-1,1);
clasesTr  = discretize(yNormTrain,edges,1:3)';

hTN2       = map(hTest,hMin,hMax,0.1,1);
yLog2      = log10(hTN2);
yNormTest  = map(yLog2,yMin,yMax,-1,1);
clasesTe  = discretize(yNormTest,edges,1:3)';

YTrain     = categorical(clasesTr);
YTest      = categorical(clasesTe);


% Interpolar todas las series a la misma longitud

Ntr = numel(vTrain);
lp  = zeros(Ntr,1);
for i = 1:Ntr
    p_trains = vTrain(i).power;
    lp(i)    = length(p_trains);
end

T         = round(mean(lp)/2);            % longitud fija
PmatTrain = zeros(numel(vTrain), T);

for i = 1:Ntr
    p_trains = vTrain(i).power;
    PmatTrain(i,:) = inter2mean(p_trains,T);
end

Nts = numel(vTest);
PmatTest = zeros(numel(vTest), T);

for i = 1:Nts
    p_test = vTest(i).power;
    PmatTest(i,:) = inter2mean(p_test,T);
end

% Bloque MODWT + Normalización global

wname   = 'db1';
nLevels = 4; 
nFeat = nLevels + 1;

WmatTrain = zeros(Ntr, nFeat, T);

for i = 1:Ntr
    WmatTrain(i,:,:) = modwt(PmatTrain(i,:), wname, nLevels,"TimeAlign",true);
end

wMin = min(WmatTrain,[],[1 3]);
wMax = max(WmatTrain,[],[1 3]);

W3dTrain = 2*(WmatTrain - wMin)./ (wMax - wMin)-1;

XTrainSeq = cell(Ntr,1);
for i = 1:Ntr
    XTrainSeq{i} = squeeze(W3dTrain(i,:,:));
end

WmatTest = zeros(Nts, nFeat, T);

for i = 1:Nts
    WmatTest(i,:,:) = modwt(PmatTest(i,:), wname, nLevels,"TimeAlign",true);
end

W3dTest = 2*(WmatTest - wMin)./ (wMax - wMin)-1;

XTestSeq  = cell(Nts,1);
for i = 1:Nts
    XTestSeq{i} = squeeze(W3dTest(i,:,:));
end

%% 
% 

%% Definición de la CNN

% -- Pesos de clases --
tbl = countcats(YTrain);           
classWeights = sum(tbl)./tbl;      

classNames = cellstr(categories(YTrain));

layers = [
    sequenceInputLayer(nFeat ,'Normalization','none','Name','in','MinLength',T)

    convolution1dLayer(5,32,'Padding','same','Name','conv1')
    batchNormalizationLayer('Name','batch_norm1')
    reluLayer('Name','relu1')
    dropoutLayer(0.3,'Name','droppout1')
    maxPooling1dLayer(2,'Stride',2,'Name','max_pooling1')

    convolution1dLayer(5,128,'Padding','same','Name','conv_2')
    batchNormalizationLayer('Name','batch_norm2')
    reluLayer('Name','relu2')
    dropoutLayer(0.3,'Name','droppout2')

    globalAveragePooling1dLayer('Name','global_avg_pooling')

    fullyConnectedLayer(128,'Name','fully_conected1')
    dropoutLayer(0.3,'Name','droppout3')
    reluLayer('Name','relu3')

    fullyConnectedLayer(3,'Name','fully_conected2')
    softmaxLayer('Name','softmax')
    classificationLayer( ...
        'Name','out', ...
        'Classes',      classNames, ...
        'ClassWeights', classWeights )
];

% 1) Crea el grafo de capas
lgraph = layerGraph(layers);

figure;
plot(lgraph)
title('Arquitectura de la red (layerGraph)')
%%
% Entrenamiento

opts = trainingOptions('adam', ...
    'InitialLearnRate',1e-3, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.5, ...
    'LearnRateDropPeriod',5, ...
    'MaxEpochs',30, ...
    'MiniBatchSize',16, ...
    'Shuffle','every-epoch', ...
    'Verbose',false, ...      
    'Plots','training-progress'); 


%% K‑Fold
K = 5;
cv = cvpartition(YTrain,'KFold',K);

accFolds = zeros(K,1);
precFolds = zeros(K,1);
recFolds = zeros(K,1);
f1Folds   = zeros(K,1);

for k = 1:K
    trIdx  = training(cv,k);
    valIdx = test(cv,k);
    
    Xtr = XTrainSeq(trIdx);   
    Ytr = YTrain(trIdx);

    Xvl = XTrainSeq(valIdx);  
    Yvl = YTrain(valIdx);
    
    opts.ValidationData      = {Xvl, Yvl};
    opts.ValidationFrequency = floor(sum(trIdx)/opts.MiniBatchSize);
    
    net = trainNetwork(Xtr, Ytr, layers, opts);
    Ypred = classify(net, Xvl);
    
    C = confusionmat(Yvl, Ypred);
    TP = sum(diag(C));
    FP = sum(C,'all') - TP - sum(sum(C,2) - diag(C));
    FN = sum(sum(C,2) - diag(C));

    % Para multiclasificación, se calcula promedio macro
    precPerClass = diag(C) ./ sum(C,1)';
    recPerClass  = diag(C) ./ sum(C,2);
    f1PerClass   = 2 * (precPerClass .* recPerClass) ./ (precPerClass + recPerClass);
    
    accFolds(k)  = mean(Ypred == Yvl);
    precFolds(k) = mean(precPerClass, 'omitnan');
    recFolds(k)  = mean(recPerClass, 'omitnan');
    f1Folds(k)   = mean(f1PerClass, 'omitnan');
    
    fprintf('Fold %d — Acc: %.2f%%, Prec: %.3f, Rec: %.3f, F1: %.3f\n',...
        k, accFolds(k)*100, precFolds(k), recFolds(k), f1Folds(k));
end

fprintf('CV mean Acc: %.2f%% ± %.2f%%\n', mean(accFolds)*100, std(accFolds)*100);
fprintf('CV mean Prec: %.3f ± %.3f\n', mean(precFolds), std(precFolds));
fprintf('CV mean Rec:  %.3f ± %.3f\n', mean(recFolds),  std(recFolds));
fprintf('CV mean F1:   %.3f ± %.3f\n', mean(f1Folds),    std(f1Folds));


netFinal = trainNetwork(XTrainSeq, YTrain, layers, opts);

% Evaluación

YpredTest = classify(netFinal, XTestSeq);


Ctest = confusionmat(YTest, YpredTest);
precTestPerClass = diag(Ctest) ./ sum(Ctest,1)';
recTestPerClass  = diag(Ctest) ./ sum(Ctest,2);
f1TestPerClass   = 2 * (precTestPerClass .* recTestPerClass) ./ (precTestPerClass + recTestPerClass);

accTest  = mean(YpredTest == YTest);
precTest = mean(precTestPerClass, 'omitnan');
recTest  = mean(recTestPerClass, 'omitnan');
f1Test   = mean(f1TestPerClass, 'omitnan');

fprintf('Test — Acc: %.2f%%, Prec: %.3f, Rec: %.3f, F1: %.3f\n',...
    accTest*100, precTest, recTest, f1Test);

figure;
confusionchart(YTest, YpredTest);
title('Matriz de Confusión — Test Set');

%%
% Selecciona 3 índices aleatorios (sin reemplazo)
sel = randperm(Nts, 3);

fprintf(' Índice original | Altura real | Clase Real | Clase Predicha')
for k = 1:numel(sel)
    i_test      = sel(k);             % índice en vTest
    idx_orig    = idxTest(i_test);    % índice en vehicles
    altura_real = hTest(i_test);       % altura real en metros
    clase_real  = YTest(i_test);       % clase real (categorical)
    prediccion  = YpredTest(i_test);   % clase predicha (categorical)
    
    fprintf(' %14d  | %11.2f |  %9s | %9s\n', ...
        idx_orig, altura_real, string(clase_real), string(prediccion));
end
%%

function x_map = map(x,in_min,in_max,out_min,out_max)
    x_map = (x - in_min) .* ((out_max - out_min)/(in_max - in_min)) + out_min;
    x_map = max( min(x_map, out_max), out_min );
end

function Y = inter2mean(x,N_mean)
    N = length(x);

    x_old = linspace(1, N, N);
    x_new = linspace(1, N, N_mean);
        
    % Interpolación lineal;
    Y = interp1(x_old, x, x_new, 'spline');
end