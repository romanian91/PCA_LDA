% Task2.m
clc;clear all;close all;

% load sample set
load('faceDat.mat');

[classnum, samplenum] = size(faceDat);
[m, n] = size(faceDat(1, 1).Image);

trainset = [];  %training set
testset = [];   %testing set
testindex = []; %class label of test samples

% vectorize
for i = 1:classnum
    for j = 1:(samplenum/2)
        I = faceDat(i, j).Image;
        trainset(:,(i-1)*(samplenum/2)+j) = double(reshape(I', 1, m*n)');
    end
    for k = (j+1):samplenum
        I = faceDat(i, k).Image;
        testset(:,(i-1)*(samplenum/2)+(k-j)) = double(reshape(I', 1, m*n)');
        testindex(1,(i-1)*(samplenum/2)+(k-j)) = i;
    end
end
% normalization
trainset = trainset/255;
testset = testset/255;

% mean of one class
for i = 1:classnum
    classmean(:,i) = mean(trainset(:,(i-1)*(samplenum/2)+1:i*(samplenum/2)), 2);
end
% mean of training set
samplemean = mean(trainset, 2);

% Within-class scatter matrix: Sw
Sw = 0;
for i = 1:classnum
    for j = 1:(samplenum/2)
        diff = trainset(:,(i-1)*(samplenum/2)+j)-classmean(:,i);
        Sw = Sw + diff*diff';
    end
end
% Between-class scatter matrix: Sb
Sb = 0;
for i = 1:classnum
    diff = classmean(:,i) - samplemean;
    Sb = Sb + (samplenum/2)*diff*diff';
end

% WHY Sw is singular for the training set:
% because the feature num >> sample num.

% eigenvectors and eigenvalues
[eigvector, eigvalue] = eig(pinv(Sw)*Sb);
% sort eigenvalues
deigvalue = diag(eigvalue);
[T index] = dsort(deigvalue);
for i=1:size(eigvector,2)
    peigvector(:,i) = eigvector(:,index(i));
    peigvalue(i) = deigvalue(index(i));
end

%% TASK1.1: Display The First 20 Eigenfaces
for i = 1:20
    eigface = (reshape(peigvector(:,i), n, m)');
    f = figure;imshow(mat2gray(eigface));
%     saveas(f, ['eigenface', num2str(i), '.tif']);
end
% close all;

%% TASK1.2: Testing For Identification Percentage
[dim, testnum] = size(testset);
IP_DIM = [];   %Identification Percentage and K-value
for kval = 5:5:2500
    % get needed number of eigenvectors
    peigvec = peigvector(:,1:kval);

    % projection
    trainPCAproj = peigvec' * trainset;
    testPCAproj = peigvec' * testset;

    % calculate accuracy
    count = 0;  %number of testing right samples
    for i = 1:testnum
        testproj = testPCAproj(:,i);
        difftotal = []; %distance for all classes
        for j = 1:classnum
            diff = 0;   %distance for one class
            for k = (j-1)*(samplenum/2)+1:j*(samplenum/2)
                trainproj = trainPCAproj(:,k);
                diff = diff + norm(testproj - trainproj);    %Euclidean distance
            end
            difftotal(1,j) = diff/(samplenum/2);
        end
        
        % find class with minimum distance
        [mindist, minindex] = min(difftotal);
        
        % count testing right samples
        if (minindex == testindex(i))
            count = count + 1;
        end
    end
    % Identification Percentage
    IP = double(count)/testnum;
    
    % store results
    IP_DIM = [IP_DIM;kval IP];
end

% plot and save figure
x = IP_DIM(:,1);
y = IP_DIM(:,2);
f = figure;plot(x,y);
saveas(f, 'IP_DIM.tif');
% save results
save('IP_DIM.mat', 'IP_DIM');

%% TASK1.3: Provide A Table
kvalues = [5;10;20;40;60;100;150;200;400;1000;2000];
TABLE = [];

% eigenvalue sum
evalsum = sum(peigvalue);
for i = 1:size(kvalues, 1)
    kval = kvalues(i);
    
    totalvar = sum(peigvalue(1:kval))/evalsum;
    
    ip = IP_DIM(find(IP_DIM(:,1)==kval),2);
    
    % store results
    TABLE = [TABLE;kval totalvar ip];
end
save('TABLE.mat', 'TABLE');