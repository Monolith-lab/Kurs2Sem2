clear; clc; % Очистка рабочего пространства и командного окна для чистого запуска

tic; % Запуск таймера

% --- Имя файла для сохранения результатов ---
filename_excel = 'РЗ1_Тета.xlsx';

% --- Константы ---
FoL = 0.2083;
BiL = 1.1579;
FoR = 0.2793;
BiR = 1; 

% --- Переменные L и R (массивы безразмерных координат) ---
L_values = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
R_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1];

% --- Общие параметры ---
n_desired_roots = 1000; % Уменьшено для ускорения отладки, можно увеличить до 1000 или более
epsilon = 1e-9; % Малая величина для отступов от границ интервалов и сравнений
fzero_options = optimset('TolX', 1e-9, 'Display', 'off'); % Опции для fzero, 'iter' для отладки

% =========================================================================
% Функция для поиска положительных корней характеристических уравнений
% =========================================================================
function roots_found = find_positive_roots(equation_func, num_roots_to_find, interval_generator_func, fzero_opts, small_epsilon, geometry_type)
    roots_found = zeros(num_roots_to_find, 1);
    num_found = 0;
    k_interval_idx = 0; % Индекс для генератора интервалов
    max_interval_generation_attempts = num_roots_to_find * 15; % Ограничение на попытки найти интервалы
    current_attempts = 0;

    % fprintf('ОТЛАДКА (%s): Запуск find_positive_roots. Требуется: %d\n', geometry_type, num_roots_to_find);

    while num_found < num_roots_to_find && current_attempts < max_interval_generation_attempts
        interval = interval_generator_func(k_interval_idx);
        current_attempts = current_attempts + 1;
        k_interval_idx = k_interval_idx + 1;

        if isempty(interval) || interval(1) >= interval(2) || interval(1) < -small_epsilon
             % fprintf('ОТЛАДКА (%s): Неверный или пустой интервал для k_idx=%d. Пропуск.\n', geometry_type, k_interval_idx-1);
            continue;
        end

        check_a = interval(1) + small_epsilon*10; % Небольшой отступ от границ для проверки знака
        check_b = interval(2) - small_epsilon*10;
        if check_a >= check_b % Если отступы "съели" интервал
            check_a = interval(1);
            check_b = interval(2);
        end

        val_a = equation_func(check_a);
        val_b = equation_func(check_b);

        % fprintf('ОТЛАДКА (%s): k_idx=%d, Интервал=[%.4e, %.4e], f(a)=%.4e, f(b)=%.4e\n', ...
        %         geometry_type, k_interval_idx-1, interval(1), interval(2), val_a, val_b);

        if isnan(val_a) || isnan(val_b) || isinf(val_a) || isinf(val_b)
             % fprintf('ОТЛАДКА (%s): NaN/Inf на границах интервала. Пропуск.\n', geometry_type);
             continue;
        end

        if val_a * val_b <= 0 % Условие наличия корня (функция меняет знак)
            try
                % fprintf('ОТЛАДКА (%s): Попытка fzero в [%.4e, %.4e]\n', geometry_type, interval(1), interval(2));
                root_candidate = fzero(equation_func, interval, fzero_opts);

                if ~isempty(root_candidate) && isreal(root_candidate) && root_candidate > small_epsilon/2 % Корень действительный и положительный
                    is_unique = true;
                    if num_found > 0
                        if any(abs(roots_found(1:num_found) - root_candidate) < 1e-7) % Проверка на дублирование
                            is_unique = false;
                            % fprintf('ОТЛАДКА (%s): Корень %.4e является дубликатом.\n', geometry_type, root_candidate);
                        end
                    end

                    if is_unique
                        num_found = num_found + 1;
                        roots_found(num_found) = root_candidate;
                        % fprintf('ОТЛАДКА (%s): Найден корень #%d: %.4e\n', geometry_type, num_found, root_candidate);
                    end
                end
            catch ME
                % fprintf('ОТЛАДКА (%s): Ошибка fzero в [%.4e, %.4e]: %s\n', geometry_type, interval(1), interval(2), ME.message);
            end
        else
            % fprintf('ОТЛАДКА (%s): Функция не меняет знак в интервале [%.4e, %.4e]. f(a)=%.3e, f(b)=%.3e\n', ...
            %    geometry_type, interval(1), interval(2), val_a, val_b);
        end

        if num_found >= num_roots_to_find
            break; % Нашли достаточное количество корней
        end
    end

    if num_found < num_roots_to_find
        warning('find_positive_roots (%s): Найдено %d из %d требуемых положительных корней. Возможны пропуски или необходимо увеличить max_interval_generation_attempts.', geometry_type, num_found, num_roots_to_find);
    end
    roots_found = roots_found(1:num_found); % Возвращаем только найденные корни
end

% =========================================================================
% Генераторы интервалов и функции обработки для ПЛАСТИНЫ
% =========================================================================
function interval = slab_interval_generator(k_idx, BiL_val, ep)
    % k_idx = 0, 1, 2, ...
    if BiL_val >= -ep
        lower = k_idx * pi + ep;
        upper = (k_idx + 0.5) * pi - ep;
        if k_idx == 0, lower = ep; end
    else
        lower = (k_idx + 0.5) * pi + ep;
        upper = (k_idx + 1.0) * pi - ep;
    end
    if upper <= lower, interval = []; else, interval = [lower, upper]; end
end

function [slab_roots, slab_theta_sums, slab_theta_components, L_vals_out] = process_slab_case(BiL_val, FoL_val, L_arr, num_roots, fzero_opts_loc, ep)
    fprintf('--- Обработка геометрии "Пластина" ---\n');
    fprintf('BiL = %f, FoL = %f\n', BiL_val, FoL_val);

    if abs(BiL_val) < ep
        fprintf('BiL близок к 0: Корни для пластины x_m = m*pi (положительные).\n');
        slab_roots = ((1:num_roots)' * pi);
    else
        eq_slab = @(x) x .* tan(x) - BiL_val;
        interval_gen_slab_local = @(k_idx) slab_interval_generator(k_idx, BiL_val, ep);
        fprintf('Поиск %d положительных корней для пластины...\n', num_roots);
        slab_roots = find_positive_roots(eq_slab, num_roots, interval_gen_slab_local, fzero_opts_loc, ep, 'Пластина');
    end
    actual_num_slab_roots = length(slab_roots);
    fprintf('Найдено %d корней для пластины.\n', actual_num_slab_roots);

    slab_theta_sums = zeros(length(L_arr), 1);
    slab_theta_components = nan(length(L_arr), num_roots); % Хранение каждого слагаемого Тета
    L_vals_out = L_arr(:);

    for i_L = 1:length(L_arr)
        L_current = L_arr(i_L);
        current_sum_L_theta = 0;

        % Для случая Bi=0 и некоторых решений, может быть начальный член ряда = 1 или другая константа
        % Если ваше решение для BiL=0 начинается с 1 (например, для безразмерной температуры 1-Theta)
        if abs(BiL_val) < ep
            current_sum_L_theta = current_sum_L_theta + 1.0;
            % Этот член "1.0" не является частью ряда по корням, поэтому не добавляем его в slab_theta_components
            % Если он должен быть частью суммы Тета, то его нужно учесть отдельно или в первом компоненте
            % Для простоты, сейчас он просто добавляется к итоговой сумме.
        end

        for r_idx = 1:actual_num_slab_roots
            r = slab_roots(r_idx);
            term_coeff_numerator = 2*sin(r);
            term_coeff_denominator = r+sin(r)*cos(r);

            term_coeff = 0;
            if abs(term_coeff_denominator) > 1e-10
                term_coeff = term_coeff_numerator / term_coeff_denominator;
            elseif abs(term_coeff_numerator) < 1e-10 && abs(r) > ep % 0/0 for r != 0
                term_coeff = 0;
            elseif abs(r) < ep % r is close to 0
                 term_coeff = 0; % Or limit if applicable
            else
                 warning('Пластина: Потенциальное Inf/NaN для корня r=%f, BiL=%f. Коэффициент установлен в 0.', r, BiL_val);
            end

            theta_component_val = term_coeff * cos(r*L_current) * exp(-r^2*FoL_val);
            current_sum_L_theta = current_sum_L_theta + theta_component_val;
            slab_theta_components(i_L, r_idx) = theta_component_val;
        end
        slab_theta_sums(i_L) = current_sum_L_theta;
        fprintf('L = %f, Тета_Пластина = %f\n', L_current, current_sum_L_theta);
    end
    % Если найдено меньше корней, чем num_roots, обрежем slab_theta_components
    if actual_num_slab_roots < num_roots
        slab_theta_components = slab_theta_components(:, 1:actual_num_slab_roots);
    end
end

% =========================================================================
% Генераторы интервалов и функции обработки для ЦИЛИНДРА
% =========================================================================
function zeros_J0 = get_J0_zeros(num_zeros_needed, fzero_opts_loc, ep)
    persistent cached_J0_zeros num_cached_J0_zeros;
    if isempty(cached_J0_zeros) || num_cached_J0_zeros < num_zeros_needed
        % fprintf('ОТЛАДКА (Нули J0): Вычисление/обновление кэша нулей J0(x) до %d нулей...\n', num_zeros_needed);
        new_zeros_J0 = zeros(num_zeros_needed, 1);
        current_num_in_cache = 0;
        if ~isempty(cached_J0_zeros) && num_cached_J0_zeros > 0
            new_zeros_J0(1:num_cached_J0_zeros) = cached_J0_zeros(1:num_cached_J0_zeros);
            current_num_in_cache = num_cached_J0_zeros;
        end

        last_found_zero = 0;
        if current_num_in_cache > 0
            last_found_zero = new_zeros_J0(current_num_in_cache);
        end

        for i_zero = (current_num_in_cache + 1) : num_zeros_needed
            lower_search_bnd = 0; upper_search_bnd = 0;
            if i_zero == 1
                lower_search_bnd = 2.0; upper_search_bnd = 3.0; % Первый ноль J0 ~2.4048
            else
                lower_search_bnd = last_found_zero + pi/2;
                upper_search_bnd = last_found_zero + pi + pi/2;
            end
            % fprintf('ОТЛАДКА (Нули J0): Поиск нуля J0 #%d в [%.4f, %.4f]\n', i_zero, lower_search_bnd, upper_search_bnd);
            try
                val_at_lower = besselj(0, lower_search_bnd);
                val_at_upper = besselj(0, upper_search_bnd);
                if val_at_lower * val_at_upper > 0 && abs(val_at_lower)>ep && abs(val_at_upper)>ep
                    warning('get_J0_zeros: Знаки одинаковые для нуля J0 #%d в [%.2f, %.2f]. f(a)=%.2e, f(b)=%.2e. Попытка расширить интервал.', ...
                            i_zero, lower_search_bnd, upper_search_bnd, val_at_lower, val_at_upper);
                    if i_zero == 1, lower_search_bnd = 1.0; upper_search_bnd = 4.0;
                    else, lower_search_bnd = last_found_zero + 0.5; upper_search_bnd = last_found_zero + 2*pi; end
                    % fprintf('ОТЛАДКА (Нули J0): Повторный поиск нуля J0 #%d в новом интервале [%.4f, %.4f]\n', i_zero, lower_search_bnd, upper_search_bnd);
                end
                found_j0_zero = fzero(@(x) besselj(0,x), [lower_search_bnd, upper_search_bnd], fzero_opts_loc);
                if ~isnan(found_j0_zero) && found_j0_zero > last_found_zero + ep/2
                    new_zeros_J0(i_zero) = found_j0_zero;
                    last_found_zero = found_j0_zero;
                    % fprintf('ОТЛАДКА (Нули J0): Найден ноль J0 #%d: %.6f\n', i_zero, last_found_zero);
                else
                    prev_zero_val_str = 'N/A'; if i_zero > 1, prev_zero_val_str = sprintf('%.4f', new_zeros_J0(i_zero-1)); end
                    error('Не удалось найти ноль J0 #%d или он не строго возрастает (найдено %.4f, последний был %s).', i_zero, found_j0_zero, prev_zero_val_str);
                end
            catch ME_j0
                 fprintf('Детали ошибки для нуля J0 #%d: Интервал: [%.4f, %.4f], f(low)=%.4e, f(high)=%.4e\n', i_zero, lower_search_bnd, upper_search_bnd, besselj(0,lower_search_bnd), besselj(0,upper_search_bnd));
                 error('Ошибка при поиске нуля J0 #%d: %s.', i_zero, ME_j0.message);
            end
        end
        cached_J0_zeros = new_zeros_J0; num_cached_J0_zeros = num_zeros_needed;
        % fprintf('ОТЛАДКА (Нули J0): Кэш для нулей J0 обновлен. Содержимое: %s\n', mat2str(cached_J0_zeros', 4));
    end
    zeros_J0 = cached_J0_zeros(1:num_zeros_needed);
end

function interval = cylinder_interval_generator(k_idx_root, BiR_val, zeros_J0_arr, ep)
    lower_bnd_interval = 0;
    if k_idx_root > 0
        if k_idx_root > length(zeros_J0_arr), interval = []; return; end
        lower_bnd_interval = zeros_J0_arr(k_idx_root);
    end
    if (k_idx_root + 1) > length(zeros_J0_arr), interval = []; return; end
    upper_bnd_interval = zeros_J0_arr(k_idx_root + 1);

    final_lower = lower_bnd_interval + ep;
    if k_idx_root == 0, final_lower = ep; end % Первый интервал от epsilon
    final_upper = upper_bnd_interval - ep;

    if final_upper <= final_lower, interval = []; else, interval = [final_lower, final_upper]; end
end

function [cyl_roots, cyl_theta_sums, cyl_theta_components, R_vals_out] = process_cylinder_case(BiR_val, FoR_val, R_arr, num_roots_to_find, fzero_opts_loc, ep)
    fprintf('--- Обработка геометрии "Цилиндр" ---\n');
    fprintf('BiR = %f, FoR = %f\n', BiR_val, FoR_val);
    cyl_roots_found = [];

    if abs(BiR_val) < ep
        fprintf('BiR близок к 0: Корни для цилиндра - положительные нули J1(x).\n');
        cyl_roots_found = zeros(num_roots_to_find,1);
        last_J1_zero = 0;
        for i_rt = 1:num_roots_to_find
            lower_search_J1 = last_J1_zero + ep*10;
            if i_rt == 1 && last_J1_zero < ep, lower_search_J1 = 3.0; end % Первый ноль J1 ~3.8317
            upper_search_J1 = lower_search_J1 + pi + 1.0;
            try
                current_J1_zero = fzero(@(x) besselj(1,x), [lower_search_J1, upper_search_J1], fzero_opts_loc);
                 if ~isnan(current_J1_zero) && current_J1_zero > last_J1_zero + ep
                    cyl_roots_found(i_rt) = current_J1_zero; last_J1_zero = current_J1_zero;
                else, error('Не удалось найти ноль J1 #%d для BiR=0 или он не возрастает.', i_rt); end
            catch ME_j1_zero, error('Ошибка при поиске нуля J1 #%d для BiR=0: %s. Интервал [%.2f, %.2f]', i_rt, ME_j1_zero.message, lower_search_J1, upper_search_J1); end
        end
    else
        eq_cyl = @(x) x .* besselj(1,x) - BiR_val .* besselj(0,x);
        num_j0_zeros_needed = num_roots_to_find + 1; % +1 для определения последнего интервала
        zeros_J0_for_intervals = get_J0_zeros(num_j0_zeros_needed, fzero_opts_loc, ep);
        interval_gen_cyl_local = @(k_idx) cylinder_interval_generator(k_idx, BiR_val, zeros_J0_for_intervals, ep);
        fprintf('Поиск %d положительных корней для цилиндра (BiR != 0)...\n', num_roots_to_find);
        cyl_roots_found = find_positive_roots(eq_cyl, num_roots_to_find, interval_gen_cyl_local, fzero_opts_loc, ep, 'Цилиндр');
    end
    actual_num_cyl_roots = length(cyl_roots_found);
    fprintf('Найдено %d корней для цилиндра.\n', actual_num_cyl_roots);
    cyl_roots = cyl_roots_found;

    cyl_theta_sums = zeros(length(R_arr), 1);
    cyl_theta_components = nan(length(R_arr), num_roots_to_find); % Хранение каждого слагаемого Тета
    R_vals_out = R_arr(:);

    for i_R = 1:length(R_arr)
        R_current = R_arr(i_R);
        current_sum_R_theta = 0;

        % Аналогично пластине, если для BiR=0 есть начальный член
        if abs(BiR_val) < ep
            current_sum_R_theta = current_sum_R_theta + 1.0;
        end

        for r_idx = 1:actual_num_cyl_roots
            r_val = cyl_roots(r_idx);
            J0_r = besselj(0,r_val);
            J1_r = besselj(1,r_val);

            term_coeff_numerator = 2*J1_r;
            term_coeff_denominator = r_val * (J0_r^2 + J1_r^2);

            term_coeff = 0;
            if abs(r_val) < ep % r_val is close to 0
                term_coeff = 0;
            elseif abs(term_coeff_denominator) > 1e-10
                term_coeff = term_coeff_numerator / term_coeff_denominator;
            elseif abs(term_coeff_numerator) < 1e-10 % 0/0
                term_coeff = 0;
            else
                warning('Цилиндр: Потенциальное Inf/NaN для корня r=%f, BiR=%f. Коэффициент установлен в 0.', r_val, BiR_val);
            end

            theta_component_val = term_coeff * besselj(0,r_val*R_current) * exp(-r_val^2*FoR_val);
            current_sum_R_theta = current_sum_R_theta + theta_component_val;
            cyl_theta_components(i_R, r_idx) = theta_component_val;
        end
        cyl_theta_sums(i_R) = current_sum_R_theta;
        fprintf('R = %f, Тета_Цилиндр = %f\n', R_current, current_sum_R_theta);
    end
    if actual_num_cyl_roots < num_roots_to_find
        cyl_theta_components = cyl_theta_components(:, 1:actual_num_cyl_roots);
    end
end

% =========================================================================
% Основная часть: вызов обработчиков и запись в Excel
% =========================================================================
all_excel_data_to_write = {}; % Структура для хранения данных для записи

% --- Обработка для пластины ---
if ~isempty(L_values)
    [slab_roots_res, slab_theta_sums_res, slab_theta_comps_res, slab_L_vals_out] = ...
        process_slab_case(BiL, FoL, L_values, n_desired_roots, fzero_options, epsilon);

    num_L = length(slab_L_vals_out);
    actual_slab_roots_count = length(slab_roots_res);

    % Таблица с итоговыми Тета и корнями
    roots_slab_matrix_repeated = nan(num_L, actual_slab_roots_count);
    if actual_slab_roots_count > 0
        for i_row=1:num_L
            roots_slab_matrix_repeated(i_row, :) = slab_roots_res(1:actual_slab_roots_count)';
        end
    end
    
    T_slab_main_data = array2table([slab_L_vals_out, slab_theta_sums_res], ...
        'VariableNames', {'Значение_L', 'Тета_Пластина'});
    
    root_slab_var_names = arrayfun(@(x) sprintf('Корень_Пластина_%d', x), 1:actual_slab_roots_count, 'UniformOutput', false);
    T_slab_roots_cols = array2table(roots_slab_matrix_repeated, 'VariableNames', root_slab_var_names);
    
    all_excel_data_to_write{end+1} = struct('SheetName', 'ДанныеПластины', 'Table', [T_slab_main_data, T_slab_roots_cols]);

    % Таблица с компонентами Тета для пластины
    theta_comp_slab_var_names = arrayfun(@(x) sprintf('Слагаемое_Тета_Корень_%d', x), 1:actual_slab_roots_count, 'UniformOutput', false);
    T_slab_theta_components = array2table([slab_L_vals_out, slab_theta_comps_res(:, 1:actual_slab_roots_count)], ...
        'VariableNames', [{'Значение_L'}, theta_comp_slab_var_names]);
    all_excel_data_to_write{end+1} = struct('SheetName', 'КомпонентыТетаПластины', 'Table', T_slab_theta_components);

else
    disp('Массив L_values пуст, обработка для пластины пропущена.');
end

% --- Обработка для цилиндра ---
if ~isempty(R_values)
    [cyl_roots_res, cyl_theta_sums_res, cyl_theta_comps_res, cyl_R_vals_out] = ...
        process_cylinder_case(BiR, FoR, R_values, n_desired_roots, fzero_options, epsilon);

    num_R = length(cyl_R_vals_out);
    actual_cyl_roots_count = length(cyl_roots_res);

    % Таблица с итоговыми Тета и корнями
    roots_cyl_matrix_repeated = nan(num_R, actual_cyl_roots_count);
    if actual_cyl_roots_count > 0
        for i_row=1:num_R
            roots_cyl_matrix_repeated(i_row, :) = cyl_roots_res(1:actual_cyl_roots_count)';
        end
    end

    T_cyl_main_data = array2table([cyl_R_vals_out, cyl_theta_sums_res], ...
        'VariableNames', {'Значение_R', 'Тета_Цилиндр'});
        
    root_cyl_var_names = arrayfun(@(x) sprintf('Корень_Цилиндр_%d', x), 1:actual_cyl_roots_count, 'UniformOutput', false);
    T_cyl_roots_cols = array2table(roots_cyl_matrix_repeated, 'VariableNames', root_cyl_var_names);
    
    all_excel_data_to_write{end+1} = struct('SheetName', 'ДанныеЦилиндра', 'Table', [T_cyl_main_data, T_cyl_roots_cols]);

    % Таблица с компонентами Тета для цилиндра
    theta_comp_cyl_var_names = arrayfun(@(x) sprintf('Слагаемое_Тета_Корень_%d', x), 1:actual_cyl_roots_count, 'UniformOutput', false);
    T_cyl_theta_components = array2table([cyl_R_vals_out, cyl_theta_comps_res(:, 1:actual_cyl_roots_count)], ...
        'VariableNames', [{'Значение_R'}, theta_comp_cyl_var_names]);
    all_excel_data_to_write{end+1} = struct('SheetName', 'КомпонентыТетаЦилиндра', 'Table', T_cyl_theta_components);
else
    disp('Массив R_values пуст, обработка для цилиндра пропущена.');
end

% --- Запись всех таблиц в Excel ---
if ~isempty(all_excel_data_to_write)
    if isfile(filename_excel)
        delete(filename_excel);
        fprintf('Удален существующий файл: %s\n', filename_excel);
    end

    for i_data = 1:length(all_excel_data_to_write)
        data_item = all_excel_data_to_write{i_data};
        try
            writetable(data_item.Table, filename_excel, 'Sheet', data_item.SheetName);
            fprintf('Данные для листа "%s" записаны в Excel.\n', data_item.SheetName);
        catch ME_excel
            warning('Ошибка записи данных для листа "%s" в Excel: %s', data_item.SheetName, ME_excel.message);
            disp('Убедитесь, что файл не открыт и у вас есть права на запись.');
        end
    end
else
    disp('Нет данных для записи в Excel.');
end

toc; % Остановка таймера и вывод времени выполнения
disp('Расчеты и запись в файл завершены.');

% Очистка persistent переменных (если запускать из редактора много раз без `clear` в начале, это может быть полезно)