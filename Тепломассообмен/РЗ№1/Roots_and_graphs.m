clear; clc; % Очистка рабочего пространства и командного окна для чистого запуска
close all; % Закрыть все открытые графики

tic; % Запуск таймера

% =========================================================================
% --- Входные параметры, задаваемые пользователем ---
% =========================================================================

% 1. Имена файлов для сохранения результатов
filename_excel = 'Результаты_Температуры.xlsx';

% 2. Константы для поиска корней и точности расчетов (обычно не меняются)
n_desired_roots = 100; % Количество искомых положительных корней для каждого случая
epsilon = 1e-9; % Малая величина для отступов от границ интервалов и сравнений
fzero_options = optimset('TolX', 1e-9, 'Display', 'off'); % Опции для fzero (подавляем вывод итераций)

% 3. Температуры
t0 = 100; % Начальная равномерная температура тела (в Цельсиях)
tf = 20;  % Температура окружающей среды (в Цельсиях)

% 4. Параметры для расчета чисел Фурье
a = 1.2e-7; % Коэффициент температуропроводности (м^2/с)
r0 = 0.05;  % Характерный размер для цилиндра (радиус, м)
l0 = 0.02;  % Характерный размер для пластины (половина толщины, м)

% 5. Диапазон времени для расчета (в минутах)
t_minutes_start = 10; % Начальное время
t_minutes_end = 120;  % Конечное время
t_minutes_step = 10;  % Шаг по времени
t_minutes_array = t_minutes_start:t_minutes_step:t_minutes_end;
t_seconds_array = t_minutes_array * 60; % Перевод времени в секунды для формулы Фурье

% 6. Числа Био (безразмерные, задаются напрямую)
BiL = 0.47; % Число Био для пластины
BiR = 0.4;  % Число Био для цилиндра

% 7. Конфигурации точек (R, L) для которых будут строиться графики изменения температуры со временем.
% Это CELL-массив, где каждый элемент - это CELL-массив из двух элементов {R_values, L_values}.
% R_values и L_values могут быть как скалярными числами (например, 0 или 0.5), так и массивами чисел (например, [0, 0.5, 1]).
% Если оба R_values и L_values являются массивами, то будут построены графики для всех комбинаций (R,L) (декартово произведение).
R_L_plot_configurations = {
    {0, 0},             % Центр тела
    {1, 1},             % Поверхность тела
    {0.5, 0.5},         % Середина тела
    {0, [0.5 1]},       % Ось цилиндра (R=0) и 2 точки по толщине пластины (L=0.5, L=1)
    {[0.5 1], 0}        % Середина и поверхность цилиндра (R=0.5, R=1) и центр пластины (L=0)
};

% 8. Разрешение ВНУТРЕННЕЙ расчетной сетки для построения пространственных профилей
% и общих 3D графиков (рекомендуется достаточное количество точек для гладкости).
% Всегда должна быть как минимум одна точка.
R_grid_points = linspace(0, 1, 21); % Например, 21 точка от 0 до 1
L_grid_points = linspace(0, 1, 21); % Например, 21 точка от 0 до 1

% =========================================================================
% --- Вспомогательные функции (могут быть вынесены в отдельные файлы .m) ---
% =========================================================================

% Функция для поиска положительных корней характеристических уравнений
function roots_found = find_positive_roots(equation_func, num_roots_to_find, interval_generator_func, fzero_opts, small_epsilon, geometry_type)
    roots_found = zeros(num_roots_to_find, 1);
    num_found = 0;
    k_interval_idx = 0;
    max_interval_generation_attempts = num_roots_to_find * 20; 
    current_attempts = 0;

    while num_found < num_roots_to_find && current_attempts < max_interval_generation_attempts
        interval = interval_generator_func(k_interval_idx);
        current_attempts = current_attempts + 1;
        k_interval_idx = k_interval_idx + 1;

        if isempty(interval) || interval(1) >= interval(2) || interval(1) < -small_epsilon
            continue; 
        end

        check_a = interval(1) + small_epsilon*10; 
        check_b = interval(2) - small_epsilon*10;
        if check_a >= check_b 
            check_a = interval(1);
            check_b = interval(2);
            if check_a >= check_b, continue; end 
        end

        val_a = equation_func(check_a);
        val_b = equation_func(check_b);

        if isnan(val_a) || isnan(val_b) || isinf(val_a) || isinf(val_b)
             continue; 
        end

        if val_a * val_b <= 0 
            try
                root_candidate = fzero(equation_func, interval, fzero_opts);

                if ~isempty(root_candidate) && isreal(root_candidate) && root_candidate > small_epsilon/2 
                    is_unique = true;
                    if num_found > 0
                        if any(abs(roots_found(1:num_found) - root_candidate) < 1e-7) 
                            is_unique = false;
                        end
                    end

                    if is_unique
                        num_found = num_found + 1;
                        roots_found(num_found) = root_candidate;
                    end
                end
            catch ME
                    % Логирование ошибки fzero, если нужно отлаживать
            end
        end

        if num_found >= num_roots_to_find
            break; 
        end
    end

    if num_found < num_roots_to_find
        warning('find_positive_roots (%s): Найдено %d из %d требуемых положительных корней. Возможны пропуски или необходимо увеличить max_interval_generation_attempts.', geometry_type, num_found, num_roots_to_find);
    end
    roots_found = roots_found(1:num_found); 
end

% Генератор интервалов для ПЛАСТИНЫ
function interval = slab_interval_generator(k_idx, BiL_val, ep)
    lower = k_idx * pi + ep;
    upper = (k_idx + 0.5) * pi - ep;
    
    if upper <= lower, interval = []; else, interval = [lower, upper]; end
end

% Вспомогательная функция для получения нулей функции Бесселя первого рода J0(x)
function zeros_J0 = get_J0_zeros(num_zeros_needed, fzero_opts_loc, ep)
    persistent cached_J0_zeros num_cached_J0_zeros;
    if isempty(cached_J0_zeros) || num_cached_J0_zeros < num_zeros_needed
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
                lower_search_bnd = 2.0; upper_search_bnd = 3.0; 
            else
                lower_search_bnd = last_found_zero + pi/2; 
                upper_search_bnd = last_found_zero + pi + pi/2; 
            end
            
            val_lower = besselj(0, lower_search_bnd);
            val_upper = besselj(0, upper_search_bnd);

            if val_lower * val_upper > 0 && abs(val_lower) > ep && abs(val_upper) > ep 
                warning('get_J0_zeros: Initial interval for J0 #%d [%.4f, %.4f] does not straddle root. f(a)=%.2e, f(b)=%.2e. Attempting wider search.', ...
                        i_zero, lower_search_bnd, upper_search_bnd, val_lower, val_upper);
                lower_search_bnd = max(ep, last_found_zero + 0.1); 
                upper_search_bnd = last_found_zero + 2*pi + 0.5; 
                val_lower = besselj(0, lower_search_bnd); 
                val_upper = besselj(0, upper_search_bnd);
                if val_lower * val_upper > 0 && abs(val_lower) > ep && abs(val_upper) > ep
                    error('get_J0_zeros: Still failed to find an interval straddling J0 #%d after widening. Check logic or increase range.', i_zero);
                end
            end

            try
                found_j0_zero = fzero(@(x) besselj(0,x), [lower_search_bnd, upper_search_bnd], fzero_opts_loc);
                if ~isnan(found_j0_zero) && found_j0_zero > last_found_zero + ep/2
                    new_zeros_J0(i_zero) = found_j0_zero;
                    last_found_zero = found_j0_zero;
                else
                    error('Could not find J0 zero #%d or it is not strictly increasing (found %.4f, last was %.4f).', i_zero, found_j0_zero, last_found_zero);
                end
            catch ME_j0
                fprintf('Детали ошибки для нуля J0 #%d: Интервал: [%.4f, %.4f], f(low)=%.4e, f(high)=%.4e\n', i_zero, lower_search_bnd, upper_search_bnd, besselj(0,lower_search_bnd), besselj(0,upper_search_bnd));
                error('Ошибка при поиске нуля J0 #%d: %s.', i_zero, ME_j0.message);
            end
        end
        cached_J0_zeros = new_zeros_J0; num_cached_J0_zeros = num_zeros_needed;
    end
    zeros_J0 = cached_J0_zeros(1:num_zeros_needed);
end

% Генератор интервалов для ЦИЛИНДРА (возвращает анонимную функцию)
function cylinder_interval_generator_func = cylinder_interval_generator(zeros_J0_arr, ep)
    cylinder_interval_generator_func = @(k_idx) get_interval(k_idx, zeros_J0_arr, ep);

    function interval = get_interval(k_idx_root, zeros_J0_arr_loc, ep_loc)
        if k_idx_root == 0 
            lower_bnd_interval = ep_loc;
            upper_bnd_interval = zeros_J0_arr_loc(1) - ep_loc;
        else 
            if (k_idx_root+1) > length(zeros_J0_arr_loc)
                interval = []; return; 
            end
            lower_bnd_interval = zeros_J0_arr_loc(k_idx_root) + ep_loc;
            upper_bnd_interval = zeros_J0_arr_loc(k_idx_root+1) - ep_loc;
        end

        if upper_bnd_interval <= lower_bnd_interval
            interval = [];
        else
            interval = [lower_bnd_interval, upper_bnd_interval];
        end
    end
end

% Функция расчета безразмерной температуры Тета для ПЛАСТИНЫ
function theta_sums = calculate_theta_slab(BiL_val, FoL_val, L_arr, slab_roots, ep)
    theta_sums = zeros(length(L_arr), 1);
    
    actual_num_slab_roots = length(slab_roots);
    for i_L = 1:length(L_arr)
        L_current = L_arr(i_L);
        current_sum_L_theta = 0;

        for r_idx = 1:actual_num_slab_roots
            r = slab_roots(r_idx);
            term_coeff_numerator = 2*sin(r);
            term_coeff_denominator = r+sin(r)*cos(r);

            term_coeff = 0;
            if abs(term_coeff_denominator) > ep 
                term_coeff = term_coeff_numerator / term_coeff_denominator;
            elseif abs(term_coeff_numerator) < ep && abs(r) > ep 
                term_coeff = 0; 
            else
                warning('Пластина: Потенциальное Inf/NaN для корня r=%f, BiL=%f. Коэффициент установлен в 0.', r, BiL_val);
                term_coeff = 0;
            end
            theta_component_val = term_coeff * cos(r*L_current) * exp(-r^2*FoL_val);
            current_sum_L_theta = current_sum_L_theta + theta_component_val;
        end
        theta_sums(i_L) = current_sum_L_theta;
    end
    theta_sums(theta_sums < 0) = 0;
    theta_sums(theta_sums > 1) = 1;
end

% Функция расчета безразмерной температуры Тета для ЦИЛИНДРА
function theta_sums = calculate_theta_cylinder(BiR_val, FoR_val, R_arr, cyl_roots, ep)
    theta_sums = zeros(length(R_arr), 1);

    actual_num_cyl_roots = length(cyl_roots);
    for i_R = 1:length(R_arr)
        R_current = R_arr(i_R);
        current_sum_R_theta = 0;

        for r_idx = 1:actual_num_cyl_roots
            r_val = cyl_roots(r_idx);
            J0_r = besselj(0,r_val);
            J1_r = besselj(1,r_val);

            term_coeff_numerator = 2*J1_r; 
            term_coeff_denominator = r_val * (J0_r^2 + J1_r^2);

            term_coeff = 0;
            if abs(r_val) < ep 
                 term_coeff = 0;
            elseif abs(term_coeff_denominator) > ep 
                term_coeff = term_coeff_numerator / term_coeff_denominator;
            elseif abs(term_coeff_numerator) < ep 
                term_coeff = 0;
            else
                warning('Цилиндр: Потенциальное Inf/NaN для корня r=%f, BiR=%f. Коэффициент установлен в 0.', r_val, BiR_val);
                term_coeff = 0;
            end
            theta_component_val = term_coeff * besselj(0,r_val*R_current) * exp(-r_val^2*FoR_val);
            current_sum_R_theta = current_sum_R_theta + theta_component_val;
        end
        theta_sums(i_R) = current_sum_R_theta;
    end
    theta_sums(theta_sums < 0) = 0;
    theta_sums(theta_sums > 1) = 1;
end

% =========================================================================
% --- Основная часть расчета ---
% =========================================================================

% --- 1. Поиск корней характеристических уравнений (зависит только от Bi) ---
fprintf('--- 1. Поиск корней для пластины (BiL = %f) ---\n', BiL);
if abs(BiL) < epsilon 
    slab_roots_res = (1:n_desired_roots)' * pi; 
    fprintf('BiL близок к 0. Использование аналитических корней (m*pi) для пластины.\n');
else
    eq_slab = @(x) x .* tan(x) - BiL;
    slab_roots_res = find_positive_roots(eq_slab, n_desired_roots, @(k_idx) slab_interval_generator(k_idx, BiL, epsilon), fzero_options, epsilon, 'Пластина');
end
fprintf('Найдено %d корней для пластины.\n', length(slab_roots_res));

fprintf('--- 2. Поиск корней для цилиндра (BiR = %f) ---\n', BiR);
if abs(BiR) < epsilon 
    cyl_roots_res = zeros(n_desired_roots,1);
    last_J1_zero = 0;
    for i_rt = 1:n_desired_roots
        lower_search_J1 = last_J1_zero + epsilon*10; 
        if i_rt == 1 
            lower_search_J1 = 3.0; 
            upper_search_J1 = 4.5;
        else 
            lower_search_J1 = last_J1_zero + pi/2;
            upper_search_J1 = last_J1_zero + pi + pi/2;
        end
        
        val_lower_j1 = besselj(1, lower_search_J1);
        val_upper_j1 = besselj(1, upper_search_J1);
        if val_lower_j1 * val_upper_j1 > 0 && abs(val_lower_j1) > ep && abs(val_upper_j1) > ep
            warning('BiR=0: Initial interval for J1 #%d [%.4f, %.4f] does not straddle root. f(a)=%.2e, f(b)=%.2e. Attempting wider search.', ...
                    i_rt, lower_search_J1, upper_search_J1, val_lower_j1, val_upper_j1);
            lower_search_J1 = max(ep, last_J1_zero + 0.1);
            upper_search_J1 = last_J1_zero + 2*pi + 0.5;
            val_lower_j1 = besselj(1, lower_search_J1);
            val_upper_j1 = besselj(1, upper_search_J1);
            if val_lower_j1 * val_upper_j1 > 0 && abs(val_lower_j1) > ep && abs(val_upper_j1) > ep
                error('BiR=0: Still failed to find an interval straddling J1 #%d after widening.', i_rt);
            end
        end

        try
            current_J1_zero = fzero(@(x) besselj(1,x), [lower_search_J1, upper_search_J1], fzero_options);
             if ~isnan(current_J1_zero) && current_J1_zero > last_J1_zero + epsilon
                cyl_roots_res(i_rt) = current_J1_zero; last_J1_zero = current_J1_zero;
            else, error('Не удалось найти ноль J1 #%d для BiR=0 или он не возрастает.', i_rt); end
        catch ME_j1_zero
            fprintf('Детали ошибки для нуля J1 #%d (BiR=0): Интервал: [%.4f, %.4f], f(low)=%.4e, f(high)=%.4e\n', i_rt, lower_search_J1, upper_search_J1, besselj(1,lower_search_J1), besselj(1,upper_search_J1));
            error('Ошибка при поиске нуля J1 #%d для BiR=0: %s.', i_rt, ME_j1_zero.message);
        end
    end
    fprintf('BiR близок к 0. Использование аналитических корней (нулей J1(x)) для цилиндра.\n');
else
    eq_cyl = @(x) x .* besselj(1,x) - BiR .* besselj(0,x);
    num_j0_zeros_needed = n_desired_roots + 1; 
    zeros_J0_for_intervals = get_J0_zeros(num_j0_zeros_needed, fzero_options, epsilon);
    cyl_interval_gen_func = cylinder_interval_generator(zeros_J0_for_intervals, epsilon);
    cyl_roots_res = find_positive_roots(eq_cyl, n_desired_roots, cyl_interval_gen_func, fzero_options, epsilon, 'Цилиндр');
end
fprintf('Найдено %d корней для цилиндра.\n', length(cyl_roots_res));

% --- 3. Расчет температур и сбор данных для вывода ---
num_R_grid = length(R_grid_points);
num_L_grid = length(L_grid_points);
num_T_steps = length(t_minutes_array);

% Матрица для хранения температур: Температура(индекс_R_координаты, индекс_L_координаты, индекс_времени)
Temperature_3D_matrix = zeros(num_R_grid, num_L_grid, num_T_steps);

% Буферы для данных Excel (предварительное выделение памяти)
T_excel_data = cell(num_R_grid * num_L_grid * num_T_steps, 4); 
excel_data_rows_counter = 0;

SlabTheta_excel_data = cell(num_L_grid * num_T_steps, 4);
slab_theta_data_counter = 0;

CylinderTheta_excel_data = cell(num_R_grid * num_T_steps, 4);
cyl_theta_data_counter = 0;

fprintf('--- 3. Расчет температур и сбор данных для вывода ---\n');
for t_idx = 1:num_T_steps
    current_t_minutes = t_minutes_array(t_idx);
    current_t_seconds = t_seconds_array(t_idx);

    current_FoR = a * current_t_seconds / (r0^2);
    current_FoL = a * current_t_seconds / (l0^2);

    fprintf('Время: %.1f мин (FoL=%.4e, FoR=%.4e)\n', current_t_minutes, current_FoR, current_FoL);

    theta_slab_at_time_t = calculate_theta_slab(BiL, current_FoL, L_grid_points, slab_roots_res, epsilon);
    theta_cyl_at_time_t = calculate_theta_cylinder(BiR, current_FoR, R_grid_points, cyl_roots_res, epsilon);
    
    teta_itog_2D = theta_cyl_at_time_t * theta_slab_at_time_t'; 

    Temperature_2D = teta_itog_2D * (t0 - tf) + tf;
    
    Temperature_3D_matrix(:,:,t_idx) = Temperature_2D;

    % Сбор данных для Excel
    % Температура
    [L_mesh_for_excel, R_mesh_for_excel] = meshgrid(L_grid_points, R_grid_points); 
    current_t_column_temp = repmat(current_t_minutes, num_R_grid * num_L_grid, 1);
    current_data_block_temp = [R_mesh_for_excel(:), L_mesh_for_excel(:), current_t_column_temp, Temperature_2D(:)];
    
    start_row = excel_data_rows_counter + 1;
    end_row = excel_data_rows_counter + num_R_grid * num_L_grid;
    T_excel_data(start_row:end_row, :) = num2cell(current_data_block_temp);
    excel_data_rows_counter = end_row;

    % Тета Пластины
    current_t_column_L = repmat(current_t_minutes, num_L_grid, 1);
    current_FoL_column = repmat(current_FoL, num_L_grid, 1);
    current_data_block_L = [L_grid_points(:), current_t_column_L, current_FoL_column, theta_slab_at_time_t(:)];

    start_row = slab_theta_data_counter + 1;
    end_row = slab_theta_data_counter + num_L_grid;
    SlabTheta_excel_data(start_row:end_row, :) = num2cell(current_data_block_L);
    slab_theta_data_counter = end_row;

    % Тета Цилиндра
    current_t_column_R = repmat(current_t_minutes, num_R_grid, 1);
    current_FoR_column = repmat(current_FoR, num_R_grid, 1);
    current_data_block_R = [R_grid_points(:), current_t_column_R, current_FoR_column, theta_cyl_at_time_t(:)];

    start_row = cyl_theta_data_counter + 1;
    end_row = cyl_theta_data_counter + num_R_grid;
    CylinderTheta_excel_data(start_row:end_row, :) = num2cell(current_data_block_R);
    cyl_theta_data_counter = end_row;
end

% Обрезка буферов до фактического размера
if excel_data_rows_counter < size(T_excel_data, 1), T_excel_data = T_excel_data(1:excel_data_rows_counter, :); end
if slab_theta_data_counter < size(SlabTheta_excel_data, 1), SlabTheta_excel_data = SlabTheta_excel_data(1:slab_theta_data_counter, :); end
if cyl_theta_data_counter < size(CylinderTheta_excel_data, 1), CylinderTheta_excel_data = CylinderTheta_excel_data(1:cyl_theta_data_counter, :); end

% Преобразование в таблицы для удобства записи
T_data_table = cell2table(T_excel_data, 'VariableNames', {'R_координата', 'L_координата', 'Время_минуты', 'Температура_Цельсий'});
SlabTheta_table = cell2table(SlabTheta_excel_data, 'VariableNames', {'L_координата', 'Время_минуты', 'FoL', 'Тета_Пластина'});
CylinderTheta_table = cell2table(CylinderTheta_excel_data, 'VariableNames', {'R_координата', 'Время_минуты', 'FoR', 'Тета_Цилиндр'});
SlabRoots_table = table((1:length(slab_roots_res))', slab_roots_res, 'VariableNames', {'Индекс_Корня', 'Значение_Корня'});
CylinderRoots_table = table((1:length(cyl_roots_res))', cyl_roots_res, 'VariableNames', {'Индекс_Корня', 'Значение_Корня'});


% --- 4. Подготовка данных для GUI графиков ---
fprintf('--- 4. Подготовка данных для GUI графиков ---\n');

% Определяем, какие координаты меняются в расчетной сетке (для общих графиков)
is_R_grid_changing = num_R_grid > 1;
is_L_grid_changing = num_L_grid > 1;

% Индекс последнего момента времени
last_t_idx = num_T_steps;
last_t_minutes = t_minutes_array(last_t_idx);

% Создаем массивы для хранения имен графиков и функций-обработчиков
plot_names = {};
plot_functions = {}; % Каждая функция будет принимать ax_handle

% =========================================================================
% Группа 1: Графики температур по координате при Fo по последнему времени (профили)
% =========================================================================
fprintf('   -> Подготовка профилей температуры по координатам при t=%.1f мин (по всей сетке)\n', last_t_minutes);

if is_R_grid_changing && ~is_L_grid_changing % Только R меняется на сетке
    plot_names{end+1} = sprintf('Профиль T(R) при L=%.2f, t=%.1f мин', L_grid_points(1), last_t_minutes);
    plot_functions{end+1} = @(ax) plot_profile_R_fixed_L(ax, R_grid_points, squeeze(Temperature_3D_matrix(:, 1, last_t_idx)), L_grid_points(1), last_t_minutes);

elseif is_L_grid_changing && ~is_R_grid_changing % Только L меняется на сетке
    plot_names{end+1} = sprintf('Профиль T(L) при R=%.2f, t=%.1f мин', R_grid_points(1), last_t_minutes);
    plot_functions{end+1} = @(ax) plot_profile_L_fixed_R(ax, L_grid_points, squeeze(Temperature_3D_matrix(1, :, last_t_idx)), R_grid_points(1), last_t_minutes);

elseif is_R_grid_changing && is_L_grid_changing % R и L меняются на сетке - 3D график и срезы
    % 3D surface plot for T(R,L) at last time
    plot_names{end+1} = sprintf('3D профиль T(R,L) при t=%.1f мин', last_t_minutes);
    plot_functions{end+1} = @(ax) plot_surf_R_L(ax, L_grid_points, R_grid_points, Temperature_3D_matrix(:,:,last_t_idx), last_t_minutes);

    % Slices for T(R) at different L values
    plot_L_indices = unique([1, round(num_L_grid/2), num_L_grid]);
    if length(L_grid_points) < 3, plot_L_indices = 1:num_L_grid; end 

    for i = 1:length(plot_L_indices)
        l_idx = plot_L_indices(i);
        plot_names{end+1} = sprintf('Профиль T(R) при L=%.2f (t=%.1f мин)', L_grid_points(l_idx), last_t_minutes);
        plot_functions{end+1} = @(ax) plot_profile_R_fixed_L(ax, R_grid_points, squeeze(Temperature_3D_matrix(:, l_idx, last_t_idx)), L_grid_points(l_idx), last_t_minutes);
    end

    % Slices for T(L) at different R values
    plot_R_indices = unique([1, round(num_R_grid/2), num_R_grid]);
    if length(R_grid_points) < 3, plot_R_indices = 1:num_R_grid; end

    for i = 1:length(plot_R_indices)
        r_idx = plot_R_indices(i);
        plot_names{end+1} = sprintf('Профиль T(L) при R=%.2f (t=%.1f мин)', R_grid_points(r_idx), last_t_minutes);
        plot_functions{end+1} = @(ax) plot_profile_L_fixed_R(ax, L_grid_points, squeeze(Temperature_3D_matrix(r_idx, :, last_t_idx)), R_grid_points(r_idx), last_t_minutes);
    end
    
else % R и L оба фиксированы (единственная точка на сетке)
    plot_names{end+1} = sprintf('Температура в точке (R=%.2f, L=%.2f) при t=%.1f мин', R_grid_points(1), L_grid_points(1), last_t_minutes);
    plot_functions{end+1} = @(ax) plot_single_point_temp(ax, Temperature_3D_matrix(1,1,last_t_idx), R_grid_points(1), L_grid_points(1), last_t_minutes);
end

% =========================================================================
% Группа 2: Графики изменения температуры со временем для каждой пары (R, L)
% =========================================================================
fprintf('   -> Подготовка графиков изменения температуры со временем для заданных конфигураций (R,L) точек.\n');

% Одна функция для всех временных графиков, чтобы их можно было сравнить
plot_names{end+1} = 'Изменение температуры со временем в заданных точках';
plot_functions{end+1} = @(ax) plot_temp_over_time_configs(ax, t_minutes_array, R_L_plot_configurations, R_grid_points, L_grid_points, Temperature_3D_matrix);

% =========================================================================
% Группа 3: 3D графики температур по координате и времени
% =========================================================================
fprintf('   -> Подготовка 3D графиков температуры по координате и времени (по всей сетке).\n');

if is_R_grid_changing && ~is_L_grid_changing % Только R меняется
    plot_names{end+1} = sprintf('3D T(R,t) при L=%.2f', L_grid_points(1));
    plot_functions{end+1} = @(ax) plot_surf_R_t(ax, t_minutes_array, R_grid_points, squeeze(Temperature_3D_matrix(:,1,:)), L_grid_points(1));

elseif is_L_grid_changing && ~is_R_grid_changing % Только L меняется
    plot_names{end+1} = sprintf('3D T(L,t) при R=%.2f', R_grid_points(1));
    plot_functions{end+1} = @(ax) plot_surf_L_t(ax, t_minutes_array, L_grid_points, squeeze(Temperature_3D_matrix(1,:,:)), R_grid_points(1));

elseif is_R_grid_changing && is_L_grid_changing % R и L меняются (визуализация 4D)
    % 3D T(R, t) для нескольких L
    plot_L_indices_for_3D_time = unique([1, round(num_L_grid/2), num_L_grid]);
    if length(L_grid_points) < 3, plot_L_indices_for_3D_time = 1:num_L_grid; end

    for i = 1:length(plot_L_indices_for_3D_time)
        l_idx = plot_L_indices_for_3D_time(i);
        plot_names{end+1} = sprintf('3D T(R,t) при L=%.2f', L_grid_points(l_idx));
        plot_functions{end+1} = @(ax) plot_surf_R_t(ax, t_minutes_array, R_grid_points, squeeze(Temperature_3D_matrix(:, l_idx, :)), L_grid_points(l_idx));
    end

    % 3D T(L, t) для нескольких R
    plot_R_indices_for_3D_time = unique([1, round(num_R_grid/2), num_R_grid]);
    if length(R_grid_points) < 3, plot_R_indices_for_3D_time = 1:num_R_grid; end

    for i = 1:length(plot_R_indices_for_3D_time)
        r_idx = plot_R_indices_for_3D_time(i);
        plot_names{end+1} = sprintf('3D T(L,t) при R=%.2f', R_grid_points(r_idx));
        plot_functions{end+1} = @(ax) plot_surf_L_t(ax, t_minutes_array, L_grid_points, squeeze(Temperature_3D_matrix(r_idx, :, :)), R_grid_points(r_idx));
    end
    
else % R и L оба фиксированы - температура меняется только со временем
    plot_names{end+1} = sprintf('Температура в точке (R=%.2f, L=%.2f) со временем', R_grid_points(1), L_grid_points(1));
    plot_functions{end+1} = @(ax) plot_temp_over_time_single(ax, t_minutes_array, Temperature_3D_matrix(1,1,:), R_grid_points(1), L_grid_points(1));
end


% =========================================================================
% --- Создание GUI ---
% =========================================================================
fprintf('--- 5. Создание GUI для просмотра графиков ---\n');

% Создаем главное окно фигуры
fig = figure('Name', 'Интерактивный просмотр температурных графиков', ...
             'NumberTitle', 'off', 'Units', 'normalized', ...
             'Position', [0.1 0.1 0.8 0.8]); % Позиция и размер окна

% Создаем панель для списка графиков (левая часть)
panel_list = uipanel(fig, 'Title', 'Выберите график', ...
                     'Units', 'normalized', ...
                     'Position', [0.01 0.01 0.25 0.98]); % 25% ширины слева

% Создаем Listbox для выбора графиков
listbox = uicontrol(panel_list, 'Style', 'listbox', ...
                    'String', plot_names, ...
                    'Units', 'normalized', ...
                    'Position', [0.05 0.05 0.9 0.9]);

% Создаем панель для отображения графика (правая часть)
panel_plot = uipanel(fig, 'Title', 'Выбранный график', ...
                     'Units', 'normalized', ...
                     'Position', [0.28 0.01 0.71 0.98]); % Остальные 71% ширины справа

% Создаем оси для рисования графиков внутри правой панели
ax_main = axes(panel_plot, 'Units', 'normalized', 'Position', [0.1 0.1 0.8 0.8]);

% Устанавливаем коллбэк для listbox. Передаем ax_main как дополнительный аргумент.
set(listbox, 'Callback', @(src, event) listbox_callback(src, event, plot_functions, ax_main));

% Отображаем первый график при запуске
if ~isempty(plot_functions)
    listbox.Value = 1; % Выбираем первый элемент в списке
    % Вызываем коллбэк для первого графика, передавая ax_main
    listbox_callback(listbox, [], plot_functions, ax_main); 
end

% =========================================================================
% --- Callback функция для Listbox ---
% =========================================================================
% ax_handle теперь явно передается как аргумент
function listbox_callback(src, ~, plot_functions_ref, ax_handle)
    cla(ax_handle); % Очищаем текущие оси
    reset(ax_handle); % Сбрасываем все свойства осей, включая вид и масштабирование
    hold(ax_handle, 'off'); % Сбрасываем режим удержания графика
    
    selected_idx = src.Value; % Получаем индекс выбранного элемента
    
    if selected_idx > 0 && selected_idx <= length(plot_functions_ref)
        % Вызываем соответствующую функцию-обработчик для рисования
        plot_functions_ref{selected_idx}(ax_handle);
    end
end


% =========================================================================
% --- Вспомогательные функции для построения отдельных графиков (для GUI) ---
% =========================================================================

% Профиль T(R) при фиксированном L (2D график)
function plot_profile_R_fixed_L(ax, R_coords_data, T_data, L_val, t_val)
    plot(ax, R_coords_data, T_data, '-o');
    xlabel(ax, 'Безразмерная координата R');
    ylabel(ax, 'Температура, °C');
    title(ax, sprintf('Профиль T(R) при L=%.2f, t=%.1f мин', L_val, t_val));
    grid(ax, 'on');
    set(ax, 'XLim', [min(R_coords_data) max(R_coords_data)]);
    view(ax, 2); % Устанавливаем 2D вид
end

% Профиль T(L) при фиксированном R (2D график)
function plot_profile_L_fixed_R(ax, L_coords_data, T_data, R_val, t_val)
    plot(ax, L_coords_data, T_data, '-o');
    xlabel(ax, 'Безразмерная координата L');
    ylabel(ax, 'Температура, °C');
    title(ax, sprintf('Профиль T(L) при R=%.2f, t=%.1f мин', R_val, t_val));
    grid(ax, 'on');
    set(ax, 'XLim', [min(L_coords_data) max(L_coords_data)]);
    view(ax, 2); % Устанавливаем 2D вид
end

% 3D Профиль T(R,L)
function plot_surf_R_L(ax, L_coords_data, R_coords_data, T_data_2D, t_val)
    [L_mesh, R_mesh] = meshgrid(L_coords_data, R_coords_data); 
    surf(ax, L_mesh, R_mesh, T_data_2D);
    xlabel(ax, 'Безразмерная координата L');
    ylabel(ax, 'Безразмерная координата R');
    zlabel(ax, 'Температура, °C');
    title(ax, sprintf('Температура T(R,L) при t=%.1f мин', t_val));
    colorbar(ax);
    view(ax, 3); % Устанавливаем 3D вид
end

% Температура в одной точке (числовой вывод)
function plot_single_point_temp(ax, T_val, R_val, L_val, t_val)
    text(ax, 0.5, 0.5, sprintf('T = %.2f °C\n(R=%.2f, L=%.2f, t=%.1f мин)', T_val, R_val, L_val, t_val), ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 14);
    axis(ax, 'off');
    title(ax, 'Температура в единственной заданной точке');
    view(ax, 2); % Устанавливаем 2D вид
end

% Изменение температуры со временем для нескольких заданных пар (R,L) (2D график)
function plot_temp_over_time_configs(ax, t_array, R_L_plot_configs_data, R_grid, L_grid, Temp_3D_matrix_data)
    hold(ax, 'on');
    num_configs = length(R_L_plot_configs_data);
    plot_colors_configs_local = lines(max(1, num_configs)); 
    legend_entries_local = {}; 

    for config_idx = 1:num_configs
        current_R_set = R_L_plot_configs_data{config_idx}{1};
        current_L_set = R_L_plot_configs_data{config_idx}{2};

        if isscalar(current_R_set), current_R_set = [current_R_set]; end
        if isscalar(current_L_set), current_L_set = [current_L_set]; end

        for r_val_sel = current_R_set
            for l_val_sel = current_L_set
                [~, r_idx_closest] = min(abs(R_grid - r_val_sel)); 
                [~, l_idx_closest] = min(abs(L_grid - l_val_sel)); 
                
                temp_over_time = squeeze(Temp_3D_matrix_data(r_idx_closest, l_idx_closest, :));
                
                plot(ax, t_array, temp_over_time, '-o', 'Color', plot_colors_configs_local(mod(config_idx-1, size(plot_colors_configs_local,1))+1,:));
                
                legend_entries_local{end+1} = sprintf('R=%.2f, L=%.2f', R_grid(r_idx_closest), L_grid(l_idx_closest));
            end
        end
    end
    xlabel(ax, 'Время, мин');
    ylabel(ax, 'Температура, °C');
    title(ax, 'Изменение температуры со временем в различных точках');
    legend(ax, unique(legend_entries_local), 'Location', 'bestoutside');
    grid(ax, 'on');
    hold(ax, 'off');
    view(ax, 2); % Устанавливаем 2D вид
end

% 3D график T(R,t) для фиксированного L
function plot_surf_R_t(ax, t_array_data, R_coords_data, T_data_2D, L_val)
    surf(ax, t_array_data, R_coords_data, T_data_2D);
    xlabel(ax, 'Время, мин');
    ylabel(ax, 'Безразмерная координата R');
    zlabel(ax, 'Температура, °C');
    title(ax, sprintf('Температура T(R,t) при L=%.2f', L_val));
    colorbar(ax); view(ax, 3); % Устанавливаем 3D вид
end

% 3D график T(L,t) для фиксированного R
function plot_surf_L_t(ax, t_array_data, L_coords_data, T_data_2D, R_val)
    surf(ax, t_array_data, L_coords_data, T_data_2D); 
    xlabel(ax, 'Время, мин');
    ylabel(ax, 'Безразмерная координата L');
    zlabel(ax, 'Температура, °C');
    title(ax, sprintf('Температура T(L,t) при R=%.2f', R_val));
    colorbar(ax); view(ax, 3); % Устанавливаем 3D вид
end

% Температура в одной точке со временем (если сетка состоит из одной точки) (2D график)
function plot_temp_over_time_single(ax, t_array_data, T_data_1D, R_val, L_val)
    plot(ax, t_array_data, squeeze(T_data_1D), '-o'); 
    xlabel(ax, 'Время, мин');
    ylabel(ax, 'Температура, °C');
    title(ax, sprintf('Изменение температуры для R=%.2f, L=%.2f', R_val, L_val));
    grid(ax, 'on');
    view(ax, 2); % Устанавливаем 2D вид
end

% =========================================================================
% --- 6. Запись результатов в Excel ---
% =========================================================================
fprintf('--- 6. Запись результатов в Excel ---\n');

% --- Запись в Excel ---
if isfile(filename_excel)
    delete(filename_excel); 
    fprintf('Удален существующий файл Excel: %s\n', filename_excel);
end

try
    writetable(T_data_table, filename_excel, 'Sheet', 'РезультатыТемпературы');
    writetable(SlabTheta_table, filename_excel, 'Sheet', 'ТетаПластины');
    writetable(CylinderTheta_table, filename_excel, 'Sheet', 'ТетаЦилиндра');
    writetable(SlabRoots_table, filename_excel, 'Sheet', 'КорниПластины');
    writetable(CylinderRoots_table, filename_excel, 'Sheet', 'КорниЦилиндра');
    fprintf('Данные успешно записаны в Excel.\n');
catch ME_excel
    warning('Ошибка записи данных в Excel: %s', ME_excel.message);
    disp('Убедитесь, что файл не открыт и у вас есть права на запись в данную директорию.');
end 

toc; % Остановка таймера и вывод времени выполнения
disp('Расчеты, запись в файл и GUI завершены.');

% Очистка persistent переменных
clear find_positive_roots slab_interval_generator get_J0_zeros cylinder_interval_generator;