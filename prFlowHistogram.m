clc;
close all;
clear;

trafficTypes = {'25_75', '30_85', '32_90'};
architectureTypes = {'A1_pod100', 'A2_pod100', 'A3_pod100'};

rootdir = pwd;
curlist = {};
trafficMatrices = {};
flowAverage = zeros(3, 1);

%% connection histograms
for i = 1:3
    for j = 1:3
        tmp = {rootdir, trafficTypes(i), architectureTypes(j)};
        curlist = [curlist, joinPath(tmp)];
%         disp(curlist{end})
        connectionDistribution(curlist{end}, j);
    end
    trafficMatrixPath = joinPath({rootdir, trafficTypes(i), {'trafficMatrix'}});
    flowAverage(i) = trafficAverage(trafficMatrixPath);
end

%% A3
% for i = 1:1
%     % import cnklist
%     tmp = {rootdir, trafficTypes(i), architectureTypes(3)};
%     curlist = [curlist, joinPath(tmp)];
%     %         disp(curlist{end})
%     
%     % average traffic per connection
%     connectionDistribution(curlist{end}, 3);
%     trafficMatrixPath = joinPath({rootdir, trafficTypes(i), {'trafficMatrix'}});
%     flowAverage(i) = trafficAverage(trafficMatrixPath);
% end

%%
% load betaString.mat
% trafficHistogram = zeros(5, 22);
% curdir = dir(curlist{1});
% % process for each traffic matrix
% for i = 1:length(curdir)
%     if curdir(i).isdir && ~strcmp(curdir(i).name, '.') && ~strcmp(curdir(i).name, '..')
%         folderName = strcat(curdir(i).folder, '\', curdir(i).name);
%         filenames = dir(folderName);
%         
%         foldername = filenames(1).folder;
%         for j = 1:length(filenames)
%             filename = filenames(j).name;
%             tmp = strsplit(filename, '_');
%             if strcmp(tmp{1}, 'cnklist')
%                 filenameRoot = strjoin(tmp(1:end-1), '_');
%                 break
%             end
%         end
%         
%         filenames = {};
%         for j = 1:length(betaString)
%             tmp = strcat(foldername, '\', filenameRoot, '_', betaString(j), '.csv');
%             filenames = [filenames, strcat(foldername, '\', filenameRoot, '_', betaString(j), '.csv')];
%         end
%         
%         counter = 1;
%         for j = 1:length(filenames)
%             filename = filenames(j);
%             disp(filename)
%             [~,~,~,~,~,~,tfk_slot] = importfileConnectionAllocation(filename{1}, 2, inf);
%             edges = [49, 99, 199, 399, 999, 1999];
%             [N, ~, ~] = histcounts(tfk_slot, edges);
%             trafficHistogram(:, counter) = trafficHistogram(:, counter)+N'/sum(N);
%             counter = counter+1;
%         end
%         
%     end
% end
% trafficHistogram = trafficHistogram/20;
% tmp = strsplit(curdir(1).folder, '\');
% filename = strcat('tfkhist', '_', tmp{7}, '_', tmp{8}, '.mat');
% save(filename, 'trafficHistogram')

%%
% load betaVector.mat
% betaString = {};
% for i = 1:length(betaVector)
%     betaString = [betaString, num2str(betaVector(i))];

%%
% connectionDistribution(curlist{1})
