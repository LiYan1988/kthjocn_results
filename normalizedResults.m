function [beta, connection_ub_ave, throughput_ub_ave, connection_he_ave,...
    throughput_he_ave] = normalizedResults(mainFolder)

[beta, connection_ub, throughput_ub, obj_ub, connection_he, ...
    throughput_he, obj_he] = collectData(mainFolder);

connection_ub_ave = mean(connection_ub, 2);
throughput_ub_ave = mean(throughput_ub, 2);
connection_he_ave = mean(connection_he, 2);
throughput_he_ave = mean(throughput_he, 2);
beta = beta(:, 1);

connection_he_ave = smooth(beta, connection_he_ave, 5);
throughput_he_ave = smooth(beta, throughput_he_ave, 5);
