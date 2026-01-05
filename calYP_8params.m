function Y = calYP_8params(q, dq, ddq)
    persistent Y_func theta_syms

    if isempty(Y_func)
        base_dir = fileparts(mfilename('fullpath'));

        % 1) 加载8参数回归矩阵（由 tau123_global.m 生成）
        mat_file_candidates = {
            fullfile(base_dir, '..', '..', '最小参数集分离_new', 'Regressor_Y_Final.mat')
            'D:\Desktop\动力学参数辨识完整方案\最小参数集分离_new\Regressor_Y_Final.mat'
        };

        mat_file = '';
        for i = 1:numel(mat_file_candidates)
            if exist(mat_file_candidates{i}, 'file')
                mat_file = mat_file_candidates{i};
                break;
            end
        end
        if isempty(mat_file)
            error('找不到 Regressor_Y_Final.mat，请先运行 tau123_global.m 生成该文件。');
        end

        data = load(mat_file);
        if ~isfield(data, 'Y')
            error('Regressor_Y_Final.mat中缺少变量Y');
        end
        if isfield(data, 'Theta')
            theta_syms = data.Theta;
        else
            theta_syms = [];
        end

        Y_sym = data.Y;

        % 2) 读取几何参数 l2/l3/g
        l2_val = 0.4;
        l3_val = 0.08;
        g_val = 9.81;

        config_file_candidates = {
            fullfile(base_dir, '..', '..', '激励轨迹辨识', 'cad_parameters_config.json')
            'D:\Desktop\动力学参数辨识完整方案\激励轨迹辨识\cad_parameters_config.json'
        };

        config_file = '';
        for i = 1:numel(config_file_candidates)
            if exist(config_file_candidates{i}, 'file')
                config_file = config_file_candidates{i};
                break;
            end
        end

        if ~isempty(config_file)
            try
                json_text = fileread(config_file);
                config = jsondecode(json_text);
                link_params = config.link_parameters;
                l2_val = link_params.l2.value;
                l3_val = link_params.l3.value;
                g_val  = link_params.g.value;
            catch
                % 保持默认值
            end
        end

        % 3) 代入几何参数并生成数值函数
        syms q1 q2 q3 dq1 dq2 dq3 ddq1 ddq2 ddq3 real
        syms l2 l3 g real

        Y_num = subs(Y_sym, [l2, l3, g], [l2_val, l3_val, g_val]);

        Y_func = matlabFunction(Y_num, 'Vars', {q1, q2, q3, dq1, dq2, dq3, ddq1, ddq2, ddq3});

        % 4) 可选：检查列数
        if size(Y_sym, 2) ~= 8
            error('回归矩阵Y列数为%d，期望为8。', size(Y_sym, 2));
        end
    end

    Y = Y_func(q(1), q(2), q(3), dq(1), dq(2), dq(3), ddq(1), ddq(2), ddq(3));
end
