function [beta, connection_ub, throughput_ub, obj_ub, connection_he, ...
    throughput_he, obj_he] = collectData(mainFolder)

curdir = dir(mainFolder);
trafficMatrixNames = {};
results = {};
for i = 1:length(curdir)
    if curdir(i).isdir && ~strcmp(curdir(i).name, '.') && ~strcmp(curdir(i).name, '..')
        trafficMatrixNames = [trafficMatrixNames, curdir(i).name];
        folderName = strcat(curdir(i).folder, '\', curdir(i).name);
        filenames = dir(folderName);
        for j = 1:length(filenames)
            filename = filenames(j).name;
            tmp = strsplit(filename, '_');
            if strcmp(tmp{1}, 'result')
%                 disp(filename)
                filepath = strcat(filenames(j).folder, '\', filename);
                results = [results, importfileResultsAsMatrix(filepath)];
            end
        end
    end
end

betaNumber = size(results{1}, 1);
matrixNumber = length(results);

beta = zeros(betaNumber, matrixNumber);
connection_ub = zeros(betaNumber, matrixNumber);
throughput_ub = zeros(betaNumber, matrixNumber);
obj_ub = zeros(betaNumber, matrixNumber);
% connection_lb = zeros(betaNumber, matrixNumber);
% throughput_lb = zeros(betaNumber, matrixNumber);
% obj_lb = zeros(betaNumber, matrixNumber);
connection_he = zeros(betaNumber, matrixNumber);
throughput_he = zeros(betaNumber, matrixNumber);
obj_he = zeros(betaNumber, matrixNumber);

for i = 1:matrixNumber
    beta(:, i) = results{i}(:, 1);
    connection_ub(:, i) = results{i}(:, 2);
    throughput_ub(:, i) = results{i}(:, 3);
    obj_ub(:, i) = results{i}(:, 4);
    connection_he(:, i) = results{i}(:, 8);
    throughput_he(:, i) = results{i}(:, 9);
    obj_he(:, i) = results{i}(:, 10);
%     figure;
%     plot(connection_ub(:, i), throughput_ub(:, i))
%     hold on;
%     plot(connection_he(:, i), throughput_he(:, i))
end