addpath(genpath('Data_Generation'));
% task_list = [1, 550; 551, 1100; 1101, 1650; 1651, 2200; 2201, 2751;];
task_list = [551, 1100; 1500, 1800; 2201, 2751;];
main_Gen(task_list, 21000, 10, 10);



function H = main_Gen(task_list, total, n_bs, n_user)
num_path = 1;
n_task = size(task_list, 1);
for task = 1:n_task
    start_row = task_list(task, 1);
    end_row = task_list(task, 2);
    [DeepMIMO_dataset,~] = DeepMIMO_Dataset_Generator_RA(num_path, n_bs, start_row, end_row);
    [H, Y]= Gen_label(DeepMIMO_dataset, n_user, total);
    save(sprintf('Data_Generation/DeepMIMO Dataset/DeepMIMO_dataset_task%d.mat', task), 'H', 'Y', '-v7.3');
end
end



function index = compute_dist(user_loc)
BS_loc = [
    240, 397, 6;
    280, 397, 6;
    240, 497, 6;
    280, 497, 6;
    240, 597, 6;
    280, 597, 6;
    240, 647, 6;
    280, 647, 6;
    240, 747, 6;
    280, 747, 6;];

dist_all = sqrt(sum((BS_loc - user_loc).^2, 2));
[~, index] = min(dist_all);
end


function [H, Y] = Gen_label(DeepMIMO_dataset, n_user, total)
n_bs = size(DeepMIMO_dataset, 2);
total_user = size(DeepMIMO_dataset{1}.user, 2);
index = randi([1, total_user], n_user, total);

H  = zeros(total, n_bs, n_user);
Y  = zeros(total, n_user);

for i = 1:total
    active_BS = zeros(n_user, 1);
    for k = 1:n_user
        user = index(k, i);
        active_BS(k) = compute_dist(DeepMIMO_dataset{1}.user{user}.loc);
    end
    for k = 1:n_user
        user = index(k, i);
        for bs = 1:n_bs
%           active_bs = active_BS(bs);
            active_bs = bs;
            H(i, bs, k) = abs(DeepMIMO_dataset{active_bs}.user{user}.channel);
        end
    end
    H_input = permute(H(i, :, :),[2 3 1]);
    Y(i, :) = WMMSE_sum_rate(H_input, n_user);
end
end


