clc;
close all;
clear;

trafficTypes = {'25_75', '30_85', '32_90'};
architectureTypes = {'A1_pod100', 'A2_pod100', 'A3_pod100'};

rootdir = pwd;
curlist = {};
trafficMatrices = {};
flowAverage = zeros(3, 1);
for i = 1:2
    for j = 1:3
        tmp = {rootdir, trafficTypes(i), architectureTypes(j)};
        curlist = [curlist, joinPath(tmp)];
        connectionDistribution(curlist{end});
    end
    trafficMatrixPath = joinPath({rootdir, trafficTypes(i), {'trafficMatrix'}});
    flowAverage(i) = trafficAverage(trafficMatrixPath);
end

% 25-75, Arch 1
% plotPareto(curlist{1}, flowAverage(1), 0.9)

% 25-75, Arch 2
% plotPareto(curlist{2}, flowAverage(1), 0.9)

% 25-75, Arch 3
% plotPareto(curlist{3}, flowAverage(1), 0.9)

% 30-85, Arch 1
% plotPareto(curlist{4}, flowAverage(2), 0.9)

% 30-85, Arch 2
% plotPareto(curlist{5}, flowAverage(2), 0.9)

% 30-85, Arch 3
% plotPareto(curlist{6}, flowAverage(2), 0.88)

% 32-90, Arch 1
% plotPareto(curlist{7}, flowAverage(3), 0.9)

% 32-90, Arch 2
% plotPareto(curlist{8}, flowAverage(3), 0.85)

% 32-90, Arch 3
% plotPareto(curlist{9}, flowAverage(3), 0.88)

%%
% Arch 2 in all traffic types are good enough, maybe with a little
% adjustment in the curves. They show good trade-off and close to optimal
% solutions.

% Arch 1 in 25-75 and 30-85 are okay, they show good trade-off and good
% optimalities. 
% Problems:
% 1. Why the connection increases a little when beta increase? This is a
% very small increase, around 0.2%, should be neglected.
% 2. Why in 32-90, the curve is flat? Maybe redo simulations.

% Arch 3
% Good trade-off in 25-75 and 30-85, bad trade-off in 32-90.
% Maybe because too many elephants in 32-90, redo?
% Bad optimalities in all types, longer running time is required.

% To-do:
% 1. Don't include 32-90, its arch 1 and 3 are strange
% 2. Redo arch 3, longer running time, make heuristic better
% 3. analyze composition of data rates
% 4. Simulate for uniform data rates