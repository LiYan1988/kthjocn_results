clc;
close all;
clear;
load betaVector.mat
flowAverage = [237.5855, 238.0523];

files = dir;
for i = 1:length(files)
    tmp = strsplit(files(i).name, '_');
    if ~files(i).isdir && strcmp(tmp{1}, 'tfkhist')
        load(files(i).name)
        tmp = strsplit(files(i).name, {'_', '.'});
        figureStrings = tmp(2:5);
        tmp = strsplit(files(i).name, '_');
        if strcmp(tmp{2}, '25')
            beta = betaVector/flowAverage(1);
        elseif strcmp(tmp{2}, '30')
            beta = betaVector/flowAverage(2);
        end
        plotArea(beta, trafficHistogram, figureStrings);
%         plotArea(betaVector, trafficHistogram, figureStrings);
    end
end

