addpath(genpath('Data_Generation'));
task_list = [551, 1100; 1101, 1650; 1651, 2200;];
% task_list = [551, 1100; 826, 1375; 1101, 1650; 1376, 1925; 1651, 2200;];
main_Gen(task_list, 21000, 10, 10);

function H = main_Gen(task_list, total, n_bs, n_user)
num_path = 1;
n_task = size(task_list, 1);
for task = 1:n_task
    start_row = task_list(task, 1);
    end_row = task_list(task, 2);
    [DeepMIMO_dataset,~] = DeepMIMO_Dataset_Generator_RA(num_path, n_bs, start_row, end_row);
    H = Gen_channel(DeepMIMO_dataset, n_user, total);
    save(sprintf('Data_Generation/DeepMIMO Dataset/DeepMIMO_dataset_task%d.mat', task), 'H', '-v7.3');
end
end


function H = Gen_channel(DeepMIMO_dataset, n_user, total)
n_bs = size(DeepMIMO_dataset, 2);
total_user = size(DeepMIMO_dataset{1}.user, 2);
index = randi([1, total_user], n_user, total);

H  = zeros(total, n_bs, n_user);
for i = 1:total
    for k = 1:n_user
        user = index(k, i);
        for bs = 1:n_bs
            H(i, bs, k) = abs(DeepMIMO_dataset{bs}.user{user}.channel);
        end
    end
end
end
