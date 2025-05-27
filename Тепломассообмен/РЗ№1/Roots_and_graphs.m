clear; clc; % Очистка рабочего пространства и командного окна для чистого запуска
close all; % Закрыть все открытые графики
clear functions; % Очистка persistent переменных во всех функциях, чтобы обеспечить чистый запуск

tic; % Запуск таймера

% =========================================================================
% --- Входные параметры, задаваемые пользователем ---
% =========================================================================

% 1. Имя файла для сохранения результатов
filename_excel = 'Результаты_Температуры.xlsx';

% 2. Константы для поиска корней и точности расчетов
% Рекомендуется не изменять эти значения, если не понимаете их влияние.
n_desired_roots = 100; % Количество искомых положительных корней для каждого случая
epsilon = 1e-9; % Малая величина для отступов от границ интервалов и сравнений, и для сравнений с нулем
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
% Для типичных задач теплопередачи Bi >= 0.
BiL = 0.47; % Число Био для пластины
BiR = 0.4;  % Число Био для цилиндра

% 7. Конфигурации точек (R, L) для которых будут строиться графики изменения температуры со временем.
% Это CELL-массив, где каждый элемент - это CELL-массив из двух элементов {R_values, L_values}.
% R_values и L_values могут быть как скалярными числами (например, 0 или 0.5), так и массивами чисел (например, [0, 0.5, 1]).
% Будут построены графики для всех комбинаций (R,L) в пределах каждой конфигурации.
R_L_plot_configurations = {
    {0, 0},             % Центр тела (R=0, L=0)
    {1, 1},             % Поверхность тела (R=1, L=1)
    {0.5, 0.5},         % Середина тела по обоим направлениям (R=0.5, L=0.5)
    {0, [0.5 1]},       % Ось цилиндра (R=0) и 2 точки по толщине пластины (L=0.5, L=1)
    {[0.5 1], 0}        % Середина и поверхность цилиндра (R=0.5, R=1) и центр пластины (L=0)
};

% 8. Разрешение ВНУТРЕННЕЙ расчетной сетки для построения простр
% 
% анственных профилей
% и общих 3D графиков (рекомендуется достаточное количество точек для гладкости).
% Всегда должна быть как минимум одна точка.
R_grid_points = linspace(0, 1, 21); % Например, 21 точка от 0 до 1
L_grid_points = linspace(0, 1, 21); % Например, 21 точка от 0 до 1

% 9. Количество точек для интерполяции сплайном Акимы на 2D графиках
% Увеличьте это значение для более гладких кривых, но это может замедлить отрисовку.
n_interp_points_akima = 200;

% =========================================================================
% --- ВАЛИДАЦИЯ ВХОДНЫХ ПАРАМЕТРОВ ---
% =========================================================================
fprintf('--- Валидация входных параметров ---\n');
if ~isscalar(t0) || ~isnumeric(t0) || ~isscalar(tf) || ~isnumeric(tf)
    error('Начальная и конечная температуры (t0, tf) должны быть скалярными числами.');
end
if abs(t0 - tf) < epsilon 
    warning('Начальная температура тела (t0) практически равна температуре окружающей среды (tf). Температура тела останется постоянной и равной tf. Безразмерная температура (Тета) не будет иметь физического смысла в своей обычной интерпретации, так как (t0-tf) близко к нулю.');
end

if ~isscalar(a) || ~isnumeric(a) || a <= 0
    error('Коэффициент температуропроводности (a) должен быть положительным числом.');
end
if ~isscalar(r0) || ~isnumeric(r0) || r0 <= 0
    error('Характерный размер для цилиндра (r0) должен быть положительным числом.');
end
if ~isscalar(l0) || ~isnumeric(l0) || l0 <= 0
    error('Характерный размер для пластины (l0) должен быть положительным числом.');
end

if ~isnumeric(t_minutes_array) || isempty(t_minutes_array) || any(t_minutes_array < 0)
    error('Диапазон времени (t_minutes_array) должен содержать положительные числа.');
end
if t_minutes_step <= 0
    error('Шаг по времени (t_minutes_step) должен быть положительным.');
end

if ~isscalar(BiL) || ~isnumeric(BiL) || BiL < 0
    error('Число Био для пластины (BiL) должно быть неотрицательным числом.');
end
if ~isscalar(BiR) || ~isnumeric(BiR) || BiR < 0
    error('Число Био для цилиндра (BiR) должно быть неотрицательным числом.');
end

if ~isnumeric(R_grid_points) || isempty(R_grid_points) || any(R_grid_points < 0) || any(R_grid_points > 1)
    error('Координаты сетки R (R_grid_points) должны быть числовым массивом в диапазоне [0, 1].');
end
if ~isnumeric(L_grid_points) || isempty(L_grid_points) || any(L_grid_points < 0) || any(L_grid_points > 1)
    error('Координаты сетки L (L_grid_points) должны быть числовым массивом в диапазоне [0, 1].');
end

if ~isscalar(n_desired_roots) || ~isnumeric(n_desired_roots) || n_desired_roots <= 0 || mod(n_desired_roots,1) ~= 0
    error('Количество искомых корней (n_desired_roots) должно быть положительным целым числом.');
end
if ~isscalar(epsilon) || ~isnumeric(epsilon) || epsilon <= 0
    error('Значение epsilon должно быть малым положительным числом.');
end
if ~isscalar(n_interp_points_akima) || ~isnumeric(n_interp_points_akima) || n_interp_points_akima < 2 || mod(n_interp_points_akima,1) ~= 0
    error('Количество точек для интерполяции (n_interp_points_akima) должно быть целым числом >= 2.');
end


fprintf('Валидация завершена. Параметры корректны.\n');

% =========================================================================
% --- Основная часть расчета ---
% =========================================================================

% --- 1. Поиск корней характеристических уравнений ---
slab_roots_res = [];
cyl_roots_res = [];

fprintf('--- 1. Поиск корней для пластины (BiL = %f) ---\n', BiL);
if abs(BiL) < epsilon 
    slab_roots_res = (1:n_desired_roots)' * pi;
    fprintf('BiL близок к 0. Использование аналитических корней (m*pi) для пластины.\n');
else
    eq_slab = @(x) x .* tan(x) - BiL;
    slab_roots_res = find_positive_roots(eq_slab, n_desired_roots, @(k_idx) slab_interval_generator(k_idx, epsilon), fzero_options, epsilon, 'Пластина');
end
fprintf('Найдено %d корней для пластины.\n', length(slab_roots_res));

fprintf('--- 2. Поиск корней для цилиндра (BiR = %f) ---\n', BiR);
if abs(BiR) < epsilon 
    cyl_roots_res = zeros(n_desired_roots,1);
    last_J1_zero = 0;
    for i_rt = 1:n_desired_roots
        lower_search_J1 = 0; 
        upper_search_J1 = 0; 
        if i_rt == 1
            lower_search_J1 = 3.0; 
            upper_search_J1 = 4.5; 
        else
            lower_search_J1 = last_J1_zero + pi/2; 
            upper_search_J1 = last_J1_zero + pi + pi/2; 
        end
        lower_search_J1 = max(lower_search_J1, epsilon); 

        val_lower_j1 = besselj(1, lower_search_J1);
        val_upper_j1 = besselj(1, upper_search_J1);

        search_attempts_j1 = 0; max_search_attempts_j1 = 5;
        while val_lower_j1 * val_upper_j1 > 0 && abs(val_lower_j1) > epsilon && abs(val_upper_j1) > epsilon && search_attempts_j1 < max_search_attempts_j1
            search_attempts_j1 = search_attempts_j1 + 1;
            lower_search_J1_new = max(epsilon, last_J1_zero + 0.1 * search_attempts_j1);
            upper_search_J1_new = last_J1_zero + (pi + 0.5) * (1 + search_attempts_j1 * 0.5);
            if search_attempts_j1 == 1
                warning('BiR=0: Начальный интервал для J1 #%d [%.4f, %.4f] не содержит корень (f(a)=%.2e, f(b)=%.2e). Попытка %d/%d: новый интервал [%.4f, %.4f].', ...
                    i_rt, lower_search_J1, upper_search_J1, val_lower_j1, val_upper_j1, search_attempts_j1, max_search_attempts_j1, lower_search_J1_new, upper_search_J1_new);
            end
            lower_search_J1 = lower_search_J1_new;
            upper_search_J1 = upper_search_J1_new;
            val_lower_j1 = besselj(1, lower_search_J1);
            val_upper_j1 = besselj(1, upper_search_J1);
        end
        if val_lower_j1 * val_upper_j1 > 0 && abs(val_lower_j1) > epsilon && abs(val_upper_j1) > epsilon
             error('BiR=0: Не удалось найти интервал, содержащий корень J1 #%d, после %d попыток. Последний интервал: [%.4f, %.4f], f(low)=%.2e, f(high)=%.2e', ...
                i_rt, max_search_attempts_j1, lower_search_J1, upper_search_J1, val_lower_j1, val_upper_j1);
        end

        try
            current_J1_zero = fzero(@(x) besselj(1,x), [lower_search_J1, upper_search_J1], fzero_options);
            if ~isnan(current_J1_zero) && current_J1_zero > last_J1_zero + epsilon 
                cyl_roots_res(i_rt) = current_J1_zero; 
                last_J1_zero = current_J1_zero;
            else
                 error('Не удалось найти ноль J1 #%d для BiR=0 или он не возрастает (найдено %.4f, последний был %.4f).', i_rt, current_J1_zero, last_J1_zero); 
            end
        catch ME_j1_zero
            fprintf('Детали ошибки для нуля J1 #%d (BiR=0): Интервал: [%.4f, %.4f], f(low)=%.4e, f(high)=%.4e\n', i_rt, lower_search_J1, upper_search_J1, besselj(1,lower_search_J1), besselj(1,upper_search_J1));
            error('Ошибка при поиске нуля J1 #%d для BiR=0: %s.', i_rt, ME_j1_zero.message);
        end
    end
    fprintf('BiR близок к 0. Использование аналитических корней (нулей J1(x)) для цилиндра.\n');
else
    eq_cyl = @(x) x .* besselj(1,x) - BiR .* besselj(0,x);
    num_j0_zeros_needed = n_desired_roots + 5; 
    zeros_J0_for_intervals = get_J0_zeros(num_j0_zeros_needed, fzero_options, epsilon);
    cyl_interval_gen_func = @(k_idx) get_cylinder_interval(k_idx, zeros_J0_for_intervals, epsilon);
    cyl_roots_res = find_positive_roots(eq_cyl, n_desired_roots, cyl_interval_gen_func, fzero_options, epsilon, 'Цилиндр');
end
fprintf('Найдено %d корней для цилиндра.\n', length(cyl_roots_res));

% --- 3. Расчет температур и сбор данных для вывода ---
num_R_grid = length(R_grid_points);
num_L_grid = length(L_grid_points);
num_T_steps = length(t_minutes_array);

Temperature_3D_matrix = zeros(num_R_grid, num_L_grid, num_T_steps);

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

    fprintf('  Время: %.1f мин (FoL=%.4e, FoR=%.4e)\n', current_t_minutes, current_FoL, current_FoR);

    theta_slab_at_time_t = calculate_theta_slab(BiL, current_FoL, L_grid_points, slab_roots_res, epsilon);
    theta_cyl_at_time_t = calculate_theta_cylinder(BiR, current_FoR, R_grid_points, cyl_roots_res, epsilon);

    teta_itog_2D = theta_cyl_at_time_t * theta_slab_at_time_t'; 

    if abs(t0 - tf) < epsilon 
        Temperature_2D = ones(size(teta_itog_2D)) * tf; 
    else
        Temperature_2D = teta_itog_2D * (t0 - tf) + tf;
    end
    
    Temperature_3D_matrix(:,:,t_idx) = Temperature_2D;

    [L_mesh_for_excel, R_mesh_for_excel] = meshgrid(L_grid_points, R_grid_points);
    current_t_column_temp = repmat(current_t_minutes, num_R_grid * num_L_grid, 1);
    current_data_block_temp = [R_mesh_for_excel(:), L_mesh_for_excel(:), current_t_column_temp, Temperature_2D(:)];

    start_row = excel_data_rows_counter + 1;
    end_row = excel_data_rows_counter + num_R_grid * num_L_grid;
    T_excel_data(start_row:end_row, :) = num2cell(current_data_block_temp);
    excel_data_rows_counter = end_row;

    current_t_column_L = repmat(current_t_minutes, num_L_grid, 1);
    current_FoL_column = repmat(current_FoL, num_L_grid, 1);
    current_data_block_L = [L_grid_points(:), current_t_column_L, current_FoL_column, theta_slab_at_time_t(:)];

    start_row = slab_theta_data_counter + 1;
    end_row = slab_theta_data_counter + num_L_grid;
    SlabTheta_excel_data(start_row:end_row, :) = num2cell(current_data_block_L);
    slab_theta_data_counter = end_row;

    current_t_column_R = repmat(current_t_minutes, num_R_grid, 1);
    current_FoR_column = repmat(current_FoR, num_R_grid, 1);
    current_data_block_R = [R_grid_points(:), current_t_column_R, current_FoR_column, theta_cyl_at_time_t(:)];

    start_row = cyl_theta_data_counter + 1;
    end_row = cyl_theta_data_counter + num_R_grid;
    CylinderTheta_excel_data(start_row:end_row, :) = num2cell(current_data_block_R);
    cyl_theta_data_counter = end_row;
end

T_excel_data = T_excel_data(1:excel_data_rows_counter, :);
SlabTheta_excel_data = SlabTheta_excel_data(1:slab_theta_data_counter, :);
CylinderTheta_excel_data = CylinderTheta_excel_data(1:cyl_theta_data_counter, :);

T_data_table = cell2table(T_excel_data, 'VariableNames', {'R_координата', 'L_координата', 'Время_минуты', 'Температура_Цельсий'});
SlabTheta_table = cell2table(SlabTheta_excel_data, 'VariableNames', {'L_координата', 'Время_минуты', 'FoL', 'Тета_Пластина'});
CylinderTheta_table = cell2table(CylinderTheta_excel_data, 'VariableNames', {'R_координата', 'Время_минуты', 'FoR', 'Тета_Цилиндр'});
SlabRoots_table = table((1:length(slab_roots_res))', slab_roots_res, 'VariableNames', {'Индекс_Корня', 'Значение_Корня'});
CylinderRoots_table = table((1:length(cyl_roots_res))', cyl_roots_res, 'VariableNames', {'Индекс_Корня', 'Значение_Корня'});

% --- 4. Подготовка данных для GUI графиков ---
fprintf('--- 4. Подготовка данных для GUI графиков ---\n');

is_R_grid_changing = num_R_grid > 1;
is_L_grid_changing = num_L_grid > 1;

last_t_idx = num_T_steps;
last_t_minutes = t_minutes_array(last_t_idx);

plot_names = {};
plot_functions = {}; 

% =========================================================================
% Группа 1: Графики температур по координате при Fo по последнему времени (профили)
% =========================================================================
fprintf('   -> Подготовка профилей температуры по координатам при t=%.1f мин (по всей сетке)\n', last_t_minutes);

if is_R_grid_changing && ~is_L_grid_changing 
    plot_names{end+1} = sprintf('Профиль T(R) при L=%.2f, t=%.1f мин', L_grid_points(1), last_t_minutes);
    plot_functions{end+1} = @(ax) plot_profile_R_fixed_L(ax, R_grid_points, squeeze(Temperature_3D_matrix(:, 1, last_t_idx)), L_grid_points(1), last_t_minutes, n_interp_points_akima);

elseif is_L_grid_changing && ~is_R_grid_changing 
    plot_names{end+1} = sprintf('Профиль T(L) при R=%.2f, t=%.1f мин', R_grid_points(1), last_t_minutes);
    plot_functions{end+1} = @(ax) plot_profile_L_fixed_R(ax, L_grid_points, squeeze(Temperature_3D_matrix(1, :, last_t_idx)), R_grid_points(1), last_t_minutes, n_interp_points_akima);

elseif is_R_grid_changing && is_L_grid_changing 
    plot_names{end+1} = sprintf('3D профиль T(R,L) при t=%.1f мин', last_t_minutes);
    plot_functions{end+1} = @(ax) plot_surf_R_L(ax, L_grid_points, R_grid_points, Temperature_3D_matrix(:,:,last_t_idx), last_t_minutes);

    plot_L_indices = unique([1, round(num_L_grid/2), num_L_grid]);
    if length(L_grid_points) < 3, plot_L_indices = 1:num_L_grid; end

    for i = 1:length(plot_L_indices)
        l_idx = plot_L_indices(i);
        plot_names{end+1} = sprintf('Профиль T(R) при L=%.2f (t=%.1f мин)', L_grid_points(l_idx), last_t_minutes);
        plot_functions{end+1} = @(ax) plot_profile_R_fixed_L(ax, R_grid_points, squeeze(Temperature_3D_matrix(:, l_idx, last_t_idx)), L_grid_points(l_idx), last_t_minutes, n_interp_points_akima);
    end

    plot_R_indices = unique([1, round(num_R_grid/2), num_R_grid]);
    if length(R_grid_points) < 3, plot_R_indices = 1:num_R_grid; end

    for i = 1:length(plot_R_indices)
        r_idx = plot_R_indices(i);
        plot_names{end+1} = sprintf('Профиль T(L) при R=%.2f (t=%.1f мин)', R_grid_points(r_idx), last_t_minutes);
        plot_functions{end+1} = @(ax) plot_profile_L_fixed_R(ax, L_grid_points, squeeze(Temperature_3D_matrix(r_idx, :, last_t_idx)), R_grid_points(r_idx), last_t_minutes, n_interp_points_akima);
    end

else 
    plot_names{end+1} = sprintf('Температура в точке (R=%.2f, L=%.2f) при t=%.1f мин', R_grid_points(1), L_grid_points(1), last_t_minutes);
    plot_functions{end+1} = @(ax) plot_single_point_temp(ax, Temperature_3D_matrix(1,1,last_t_idx), R_grid_points(1), L_grid_points(1), last_t_minutes);
end

% =========================================================================
% Группа 2: Графики изменения температуры со временем для каждой пары (R, L)
% =========================================================================
fprintf('   -> Подготовка графиков изменения температуры со временем для заданных конфигураций (R,L) точек.\n');

plot_names{end+1} = 'Изменение температуры со временем в заданных точках';
plot_functions{end+1} = @(ax) plot_temp_over_time_configs(ax, t_minutes_array, R_L_plot_configurations, R_grid_points, L_grid_points, Temperature_3D_matrix, n_interp_points_akima);

% =========================================================================
% Группа 3: 3D графики температур по координате и времени
% =========================================================================
fprintf('   -> Подготовка 3D графиков температуры по координате и времени (по всей сетке).\n');

if is_R_grid_changing && ~is_L_grid_changing 
    plot_names{end+1} = sprintf('3D T(R,t) при L=%.2f', L_grid_points(1));
    plot_functions{end+1} = @(ax) plot_surf_R_t(ax, t_minutes_array, R_grid_points, squeeze(Temperature_3D_matrix(:,1,:)), L_grid_points(1));

elseif is_L_grid_changing && ~is_R_grid_changing 
    plot_names{end+1} = sprintf('3D T(L,t) при R=%.2f', R_grid_points(1));
    plot_functions{end+1} = @(ax) plot_surf_L_t(ax, t_minutes_array, L_grid_points, squeeze(Temperature_3D_matrix(1,:,:)), R_grid_points(1));

elseif is_R_grid_changing && is_L_grid_changing 
    plot_L_indices_for_3D_time = unique([1, round(num_L_grid/2), num_L_grid]);
    if length(L_grid_points) < 3, plot_L_indices_for_3D_time = 1:num_L_grid; end

    for i = 1:length(plot_L_indices_for_3D_time)
        l_idx = plot_L_indices_for_3D_time(i);
        plot_names{end+1} = sprintf('3D T(R,t) при L=%.2f', L_grid_points(l_idx));
        plot_functions{end+1} = @(ax) plot_surf_R_t(ax, t_minutes_array, R_grid_points, squeeze(Temperature_3D_matrix(:, l_idx, :)), L_grid_points(l_idx));
    end

    plot_R_indices_for_3D_time = unique([1, round(num_R_grid/2), num_R_grid]);
    if length(R_grid_points) < 3, plot_R_indices_for_3D_time = 1:num_R_grid; end

    for i = 1:length(plot_R_indices_for_3D_time)
        r_idx = plot_R_indices_for_3D_time(i);
        plot_names{end+1} = sprintf('3D T(L,t) при R=%.2f', R_grid_points(r_idx));
        plot_functions{end+1} = @(ax) plot_surf_L_t(ax, t_minutes_array, L_grid_points, squeeze(Temperature_3D_matrix(r_idx, :, :)), R_grid_points(r_idx));
    end

else 
    plot_names{end+1} = sprintf('Температура в точке (R=%.2f, L=%.2f) со временем', R_grid_points(1), L_grid_points(1));
    plot_functions{end+1} = @(ax) plot_temp_over_time_single(ax, t_minutes_array, Temperature_3D_matrix(1,1,:), R_grid_points(1), L_grid_points(1), n_interp_points_akima);
end


% =========================================================================
% --- Создание GUI ---
% =========================================================================
fprintf('--- 5. Создание GUI для просмотра графиков ---\n');

fig = figure('Name', 'Интерактивный просмотр температурных графиков', ...
             'NumberTitle', 'off', 'Units', 'normalized', ...
             'Position', [0.1 0.1 0.8 0.8]); 

panel_list = uipanel(fig, 'Title', 'Выберите график', ...
                     'Units', 'normalized', ...
                     'Position', [0.01 0.01 0.25 0.98]); 

listbox = uicontrol(panel_list, 'Style', 'listbox', ...
                    'String', plot_names, ...
                    'Units', 'normalized', ...
                    'Position', [0.05 0.15 0.9 0.8]); 

save_button = uicontrol(panel_list, 'Style', 'pushbutton', ...
                        'String', 'Сохранить график', ...
                        'Units', 'normalized', ...
                        'Position', [0.05 0.05 0.9 0.07], ... 
                        'Callback', @(src, event) save_button_callback(listbox, ax_main, plot_names));


panel_plot = uipanel(fig, 'Title', 'Выбранный график', ...
                     'Units', 'normalized', ...
                     'Position', [0.28 0.01 0.71 0.98]); 

ax_main = axes(panel_plot, 'Units', 'normalized', 'Position', [0.1 0.1 0.8 0.8]);

set(listbox, 'Callback', @(src, event) listbox_callback(src, event, plot_functions, ax_main));

if ~isempty(plot_functions)
    listbox.Value = 1; 
    listbox_callback(listbox, [], plot_functions, ax_main);
else
    cla(ax_main, 'reset');
    text(ax_main, 0.5, 0.5, 'Нет доступных графиков для отображения.', ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 12);
    axis(ax_main, 'off');
end


% =========================================================================
% --- 6. Запись результатов в Excel ---
% =========================================================================
fprintf('--- 6. Запись результатов в Excel ---\n');

try
    if isfile(filename_excel) 
        delete(filename_excel);
        fprintf('Удален существующий файл Excel: %s\n', filename_excel);
    end

    writetable(T_data_table, filename_excel, 'Sheet', 'РезультатыТемпературы');
    writetable(SlabTheta_table, filename_excel, 'Sheet', 'ТетаПластины');
    writetable(CylinderTheta_table, filename_excel, 'Sheet', 'ТетаЦилиндра');
    writetable(SlabRoots_table, filename_excel, 'Sheet', 'КорниПластины');
    writetable(CylinderRoots_table, filename_excel, 'Sheet', 'КорниЦилиндра');

    input_params_data = {
        't0', t0, 'Начальная равномерная температура тела, °C';
        'tf', tf, 'Температура окружающей среды, °C';
        'a', a, 'Коэффициент температуропроводности, м^2/с';
        'r0', r0, 'Характерный размер для цилиндра (радиус), м';
        'l0', l0, 'Характерный размер для пластины (половина толщины), м';
        't_minutes_start', t_minutes_start, 'Начальное время расчета, мин';
        't_minutes_end', t_minutes_end, 'Конечное время расчета, мин';
        't_minutes_step', t_minutes_step, 'Шаг по времени, мин';
        'BiL', BiL, 'Число Био для пластины';
        'BiR', BiR, 'Число Био для цилиндра';
        'n_desired_roots', n_desired_roots, 'Количество искомых корней';
        'epsilon', epsilon, 'Малая величина для точности расчетов';
        'n_interp_points_akima', n_interp_points_akima, 'Точек для сплайна Акимы'
    };
    input_params_table = cell2table(input_params_data, 'VariableNames', {'Параметр', 'Значение', 'Описание'});
    writetable(input_params_table, filename_excel, 'Sheet', 'Входные_Параметры');

    fprintf('Данные успешно записаны в Excel.\n');
catch ME_excel
    warning('MyScript:ExcelWriteError', 'Ошибка записи данных в Excel: %s', ME_excel.message);
    disp('Убедитесь, что файл не открыт и у вас есть права на запись в данную директорию.');
end

toc; 
disp('Расчеты, запись в файл и GUI завершены.');


% =========================================================================
% --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (ДОЛЖНЫ ИДТИ В КОНЦЕ ФАЙЛА СКРИПТА) ---
% =========================================================================

% --- Callback функция для кнопки "Сохранить график" ---
function save_button_callback(listbox_handle, ax_handle, plot_names_ref)
    selected_idx = listbox_handle.Value;
    if selected_idx > 0 && selected_idx <= length(plot_names_ref)
        current_plot_name = plot_names_ref{selected_idx};
        save_current_plot(ax_handle, current_plot_name);
    else
        msgbox('Сначала выберите график для сохранения.', 'Предупреждение', 'warn');
    end
end

% --- Функция для сохранения текущего графика ---
function save_current_plot(ax, plot_name_str)
    output_folder = 'Plots';
    if ~exist(output_folder, 'dir') 
        mkdir(output_folder);
    end

    filename_clean = matlab.lang.makeValidName(plot_name_str); 
    if length(filename_clean) > 100 
        filename_clean = [filename_clean(1:90) '_' matlab.lang.makeValidName(datahash(plot_name_str))]; 
    end
    full_filename = fullfile(output_folder, [filename_clean, '.png']);

    try
        exportgraphics(ax, full_filename, 'Resolution', 300); 
        fprintf('График "%s" сохранен как "%s".\n', plot_name_str, full_filename);
    catch ME
        warning('MyScript:PlotSaveError', 'Ошибка при сохранении графика "%s": %s', plot_name_str, ME.message);
    end
end

% --- Функция для применения общих настроек к осям графика ---
function apply_common_plot_settings(ax)
    set(ax, 'FontSize', 10, 'LineWidth', 0.8);
    grid(ax, 'on'); 
    ax.Box = 'on'; 

    if isprop(ax, 'GridAlpha')
        ax.GridAlpha = 0.15;
    end
    if isprop(ax, 'MinorGridAlpha')
        ax.MinorGridAlpha = 0.05;
    end

    if isprop(ax, 'XAxis') && ~isempty(ax.XAxis)
        set(ax.XAxis, 'FontSize', 10, 'LineWidth', 0.8);
        if isprop(ax, 'XMinorGrid') 
           ax.XMinorGrid = 'on';
        end
    end
    if isprop(ax, 'YAxis') && ~isempty(ax.YAxis)
        set(ax.YAxis, 'FontSize', 10, 'LineWidth', 0.8);
        if isprop(ax, 'YMinorGrid')
            ax.YMinorGrid = 'on';
        end
    end
    if isprop(ax, 'ZAxis') && ~isempty(ax.ZAxis)
        set(ax.ZAxis, 'FontSize', 10, 'LineWidth', 0.8);
        isVisibleZ = false;
        if isprop(ax.ZAxis, 'Visible') && ~isempty(ax.ZAxis.Visible) 
            isVisibleZ = strcmp(ax.ZAxis.Visible, 'on');
        end
        hasZLabel = false;
        if isprop(ax.ZLabel, 'String') && ~isempty(ax.ZLabel.String) 
            hasZLabel = true; 
        end

        if isVisibleZ || hasZLabel
            if isprop(ax, 'ZMinorGrid')
                ax.ZMinorGrid = 'on';
            end
        else
             if isprop(ax, 'ZMinorGrid') 
                ax.ZMinorGrid = 'off';
            end
        end
    end
    
    if isprop(ax, 'Title') && isprop(ax.Title, 'FontSize')
        ax.Title.FontSize = 12;
    end
    if isprop(ax, 'XLabel') && isprop(ax.XLabel, 'FontSize')
        ax.XLabel.FontSize = 11;
    end
    if isprop(ax, 'YLabel') && isprop(ax.YLabel, 'FontSize')
        ax.YLabel.FontSize = 11;
    end
    if isprop(ax, 'ZLabel') && isprop(ax.ZLabel, 'FontSize')
        ax.ZLabel.FontSize = 11;
    end
end

% Функция для поиска положительных корней характеристических уравнений
function roots_found = find_positive_roots(equation_func, num_roots_to_find, interval_generator_func, fzero_opts, small_epsilon, geometry_type)
    roots_found = zeros(num_roots_to_find, 1);
    num_found = 0;
    k_interval_idx = 0; 
    max_interval_generation_attempts = num_roots_to_find * 30; 
    current_attempts = 0;
    last_root_val = -Inf; 

    while num_found < num_roots_to_find && current_attempts < max_interval_generation_attempts
        interval = interval_generator_func(k_interval_idx);
        current_attempts = current_attempts + 1;
        k_interval_idx = k_interval_idx + 1;

        if isempty(interval) || interval(1) >= interval(2) || interval(1) < -small_epsilon 
            continue;
        end
        
        if (interval(2) - interval(1)) < small_epsilon || interval(2) < last_root_val + small_epsilon 
             continue;
        end
        check_a = max(interval(1), last_root_val + small_epsilon); 
        check_b = interval(2);

        if check_a >= check_b 
            continue;
        end

        val_a = equation_func(check_a);
        val_b = equation_func(check_b);

        if isnan(val_a) || isnan(val_b) || isinf(val_a) || isinf(val_b)
             continue;
        end

        if val_a * val_b <= 0 
            try
                root_candidate = fzero(equation_func, [check_a, check_b], fzero_opts);

                if ~isempty(root_candidate) && isreal(root_candidate) && root_candidate > last_root_val + small_epsilon/2 
                    is_too_close_to_existing = false;
                    if num_found > 0
                        if any(abs(roots_found(1:num_found) - root_candidate) < small_epsilon*10) 
                            is_too_close_to_existing = true;
                        end
                    end

                    if ~is_too_close_to_existing
                        num_found = num_found + 1;
                        roots_found(num_found) = root_candidate;
                        last_root_val = root_candidate; 
                    end
                end
            catch ME
                 % fprintf('ОТЛАДКА: Ошибка fzero в [%.4e, %.4e] для %s: %s\n', check_a, check_b, geometry_type, ME.message);
            end
        end

        if num_found >= num_roots_to_find
            break; 
        end
    end

    if num_found < num_roots_to_find
        warning('find_positive_roots (%s): Найдено %d из %d требуемых положительных корней. Рассмотрите возможность увеличения n_desired_roots или проверки генератора интервалов.', geometry_type, num_found, num_roots_to_find);
    end
    roots_found = sort(roots_found(1:num_found)); 
end

% Генератор интервалов для ПЛАСТИНЫ (для Bi > 0)
function interval = slab_interval_generator(k_idx, ep)
    lower = k_idx * pi + ep;
    upper = (k_idx + 0.5) * pi - ep;

    if k_idx == 0 && lower < ep 
        lower = ep; 
    end

    if upper <= lower
        interval = [];
    else
        interval = [lower, upper];
    end
end

% Вспомогательная функция для получения нулей функции Бесселя первого рода J0(x)
function zeros_J0 = get_J0_zeros(num_zeros_needed, fzero_opts_loc, ep)
    persistent cached_J0_zeros num_cached_J0_zeros_val; 
    if isempty(cached_J0_zeros) || num_cached_J0_zeros_val < num_zeros_needed
        new_zeros_J0 = zeros(num_zeros_needed, 1);
        current_num_in_cache = 0;
        if ~isempty(cached_J0_zeros) && num_cached_J0_zeros_val > 0
            len_to_copy = min(num_cached_J0_zeros_val, num_zeros_needed);
            new_zeros_J0(1:len_to_copy) = cached_J0_zeros(1:len_to_copy);
            current_num_in_cache = len_to_copy;
        end

        last_found_zero = 0;
        if current_num_in_cache > 0
            last_found_zero = new_zeros_J0(current_num_in_cache);
        end

        for i_zero = (current_num_in_cache + 1) : num_zeros_needed
            lower_search_bnd = 0; upper_search_bnd = 0;
            if i_zero == 1
                lower_search_bnd = 2.0; upper_search_bnd = 3.0; 
                if last_found_zero > lower_search_bnd 
                   lower_search_bnd = last_found_zero + pi/2;
                   upper_search_bnd = last_found_zero + pi + pi/2;
                end
            else
                lower_search_bnd = last_found_zero + pi/2;
                upper_search_bnd = last_found_zero + pi + pi/2;
            end
            
            lower_search_bnd = max(lower_search_bnd, ep); 
            if upper_search_bnd <= lower_search_bnd
                upper_search_bnd = lower_search_bnd + pi; 
            end

            val_lower = besselj(0, lower_search_bnd);
            val_upper = besselj(0, upper_search_bnd);

            search_attempts = 0; max_search_attempts = 5;
            while val_lower * val_upper > 0 && abs(val_lower) > ep && abs(val_upper) > ep && search_attempts < max_search_attempts
                search_attempts = search_attempts + 1;
                lower_search_bnd_new = max(ep, last_found_zero + 0.1 * search_attempts); 
                upper_search_bnd_new = last_found_zero + (pi + 0.5) * (1 + search_attempts * 0.5) ; 

                if search_attempts == 1 
                    warning('get_J0_zeros: Начальный интервал для J0 #%d [%.4f, %.4f] не содержит корень (f(a)=%.2e, f(b)=%.2e). Попытка %d/%d: новый интервал [%.4f, %.4f].', ...
                        i_zero, lower_search_bnd, upper_search_bnd, val_lower, val_upper, search_attempts, max_search_attempts, lower_search_bnd_new, upper_search_bnd_new);
                end
                lower_search_bnd = lower_search_bnd_new;
                upper_search_bnd = upper_search_bnd_new;
                val_lower = besselj(0, lower_search_bnd);
                val_upper = besselj(0, upper_search_bnd);
            end
            if val_lower * val_upper > 0 && abs(val_lower) > ep && abs(val_upper) > ep
                 error('get_J0_zeros: Не удалось найти интервал, содержащий J0 #%d, после %d попыток расширения. Последний интервал: [%.4f, %.4f], f(low)=%.2e, f(high)=%.2e', ...
                    i_zero, max_search_attempts, lower_search_bnd, upper_search_bnd, val_lower, val_upper);
            end

            try
                found_j0_zero = fzero(@(x) besselj(0,x), [lower_search_bnd, upper_search_bnd], fzero_opts_loc);
                if ~isnan(found_j0_zero) && found_j0_zero > last_found_zero + ep/2 
                    new_zeros_J0(i_zero) = found_j0_zero;
                    last_found_zero = found_j0_zero;
                else
                    error('Не удалось найти ноль J0 #%d или он не строго возрастает (найдено %.4f, последний был %.4f).', i_zero, found_j0_zero, last_found_zero);
                end
            catch ME_j0
                fprintf('Детали ошибки для нуля J0 #%d: Интервал: [%.4f, %.4f], f(low)=%.4e, f(high)=%.4e\n', i_zero, lower_search_bnd, upper_search_bnd, besselj(0,lower_search_bnd), besselj(0,upper_search_bnd));
                error('Ошибка при поиске нуля J0 #%d: %s.', i_zero, ME_j0.message);
            end
        end
        cached_J0_zeros = new_zeros_J0; num_cached_J0_zeros_val = num_zeros_needed;
    end
    zeros_J0 = cached_J0_zeros(1:min(num_zeros_needed, num_cached_J0_zeros_val)); 
end

% Генератор интервалов для ЦИЛИНДРА (корни уравнения x*J1(x) - Bi*J0(x) = 0)
function interval = get_cylinder_interval(k_idx_root, zeros_J0_arr_loc, ep_loc)
    if k_idx_root == 0 
        lower_bnd_interval = ep_loc; 
        if isempty(zeros_J0_arr_loc) || length(zeros_J0_arr_loc) < 1
            error('get_cylinder_interval: zeros_J0_arr_loc пуст или не содержит достаточного количества нулей для первого интервала.');
        end
        upper_bnd_interval = zeros_J0_arr_loc(1) - ep_loc;
    else 
        if (k_idx_root+1) > length(zeros_J0_arr_loc) || k_idx_root > length(zeros_J0_arr_loc)
            warning('get_cylinder_interval: Недостаточно нулей J0 для генерации интервала #%d. Доступно %d нулей.', k_idx_root, length(zeros_J0_arr_loc));
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

% Функция расчета безразмерной температуры Тета для ПЛАСТИНЫ
function theta_sums = calculate_theta_slab(BiL_val, FoL_val, L_arr, slab_roots, ep)
    theta_sums = zeros(length(L_arr), 1); 

    if abs(BiL_val) < ep
        theta_sums(:) = 1.0;
        return; 
    end
    
    if isempty(slab_roots)
        warning('calculate_theta_slab: Массив корней для пластины пуст. Theta будет NaN.');
        theta_sums(:) = NaN; 
        return;
    end

    for i_L = 1:length(L_arr)
        L_current = L_arr(i_L);
        current_sum_L_theta = 0;

        for r_idx = 1:length(slab_roots)
            r = slab_roots(r_idx);
            if isnan(r) || r < ep/2 
                continue; 
            end

            term_coeff_numerator = 2*sin(r);
            term_coeff_denominator = r+sin(r)*cos(r);
            term_coeff = 0;

            if abs(term_coeff_denominator) > ep 
                term_coeff = term_coeff_numerator / term_coeff_denominator;
            elseif abs(term_coeff_numerator) < ep 
                term_coeff = 0; 
            else 
                warning('Пластина: Потенциальное Inf/NaN для коэффициента корня r=%f (числитель=%.2e, знаменатель=%.2e), BiL=%f. Коэффициент установлен в 0.', r, term_coeff_numerator, term_coeff_denominator, BiL_val);
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

    if abs(BiR_val) < ep
        theta_sums(:) = 1.0;
        return; 
    end

    if isempty(cyl_roots)
        warning('calculate_theta_cylinder: Массив корней для цилиндра пуст. Theta будет NaN.');
        theta_sums(:) = NaN; 
        return;
    end

    for i_R = 1:length(R_arr)
        R_current = R_arr(i_R);
        current_sum_R_theta = 0;

        for r_idx = 1:length(cyl_roots)
            r_val = cyl_roots(r_idx);
            if isnan(r_val) || r_val < ep/2 
                continue;
            end

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
                warning('Цилиндр: Потенциальное Inf/NaN для коэффициента корня r=%f (числитель=%.2e, знаменатель=%.2e), BiR=%f. Коэффициент установлен в 0.', r_val, term_coeff_numerator, term_coeff_denominator, BiR_val);
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
% --- Callback функция для Listbox (для GUI) ---
% =========================================================================
function listbox_callback(src, ~, plot_functions_ref, ax_handle)
    cla(ax_handle, 'reset'); 
    hold(ax_handle, 'off'); 
    if verLessThan('matlab', '9.1') % R2016b
        % Для старых версий может не быть disableDefaultInteractivity
    else
        disableDefaultInteractivity(ax_handle); 
    end

    selected_idx = src.Value; 

    if selected_idx > 0 && selected_idx <= length(plot_functions_ref)
        plot_functions_ref{selected_idx}(ax_handle);
        apply_common_plot_settings(ax_handle); 
        if verLessThan('matlab', '9.1') 
            % Для старых версий может не быть enableDefaultInteractivity
        else
            enableDefaultInteractivity(ax_handle); 
        end
        drawnow; 
    end
end


% =========================================================================
% --- Вспомогательные функции для построения отдельных графиков (для GUI) ---
% =========================================================================

% Профиль T(R) при фиксированном L (2D график) с интерполяцией Акимы
function plot_profile_R_fixed_L(ax, R_coords_data, T_data, L_val, t_val, n_interp)
    T_data = T_data(:); % Убедимся, что это вектор-столбец
    R_coords_data = R_coords_data(:); % Убедимся, что это вектор-столбец
    
    plot(ax, R_coords_data, T_data, 'o', 'MarkerFaceColor', 'b', 'DisplayName', 'Расчетные точки');
    hold(ax, 'on');
    
    if length(R_coords_data) >= 2 % Интерполяция возможна только для 2+ точек
        R_interp = linspace(min(R_coords_data), max(R_coords_data), n_interp);
        T_interp = interp1(R_coords_data, T_data, R_interp, 'makima');
        plot(ax, R_interp, T_interp, '-', 'Color', [0 0 0.7], 'LineWidth', 1.5, 'DisplayName', 'Сплайн Акимы');
    end
    
    xlabel(ax, 'Безразмерная координата R');
    ylabel(ax, 'Температура, °C');
    title(ax, sprintf('Профиль T(R) при L=%.2f, t=%.1f мин', L_val, t_val));
    set(ax, 'XLimSpec', 'tight'); 
    legend(ax, 'show', 'Location', 'best');
    hold(ax, 'off');
    view(ax, 2); 
end

% Профиль T(L) при фиксированном R (2D график) с интерполяцией Акимы
function plot_profile_L_fixed_R(ax, L_coords_data, T_data, R_val, t_val, n_interp)
    T_data = T_data(:);
    L_coords_data = L_coords_data(:);

    plot(ax, L_coords_data, T_data, 'o', 'MarkerFaceColor', 'r', 'DisplayName', 'Расчетные точки');
    hold(ax, 'on');

    if length(L_coords_data) >= 2
        L_interp = linspace(min(L_coords_data), max(L_coords_data), n_interp);
        T_interp = interp1(L_coords_data, T_data, L_interp, 'makima');
        plot(ax, L_interp, T_interp, '-', 'Color', [0.7 0 0], 'LineWidth', 1.5, 'DisplayName', 'Сплайн Акимы');
    end
        
    xlabel(ax, 'Безразмерная координата L');
    ylabel(ax, 'Температура, °C');
    title(ax, sprintf('Профиль T(L) при R=%.2f, t=%.1f мин', R_val, t_val));
    set(ax, 'XLimSpec', 'tight');
    legend(ax, 'show', 'Location', 'best');
    hold(ax, 'off');
    view(ax, 2); 
end

% 3D Профиль T(R,L)
function plot_surf_R_L(ax, L_coords_data, R_coords_data, T_data_2D, t_val)
    [L_mesh, R_mesh] = meshgrid(L_coords_data, R_coords_data);
    surf(ax, L_mesh, R_mesh, T_data_2D, 'EdgeColor','none', 'FaceAlpha',0.8); 
    xlabel(ax, 'Безразмерная координата L');
    ylabel(ax, 'Безразмерная координата R');
    zlabel(ax, 'Температура, °C');
    title(ax, sprintf('Температура T(R,L) при t=%.1f мин', t_val));
    colorbar(ax);
    view(ax, -30, 30); 
    axis tight;
end

% Температура в одной точке (числовой вывод)
function plot_single_point_temp(ax, T_val, R_val, L_val, t_val)
    cla(ax); 
    text(ax, 0.5, 0.5, sprintf('T = %.2f °C\n(R=%.2f, L=%.2f, t=%.1f мин)', T_val, R_val, L_val, t_val), ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 14);
    axis(ax, 'off'); 
    title(ax, 'Температура в единственной заданной точке');
    view(ax, 2); 
end

% Изменение температуры со временем для нескольких заданных пар (R,L) (2D график) с интерполяцией Акимы
function plot_temp_over_time_configs(ax, t_array, R_L_plot_configs_data, R_grid, L_grid, Temp_3D_matrix_data, n_interp)
    hold(ax, 'on');
    num_unique_points = 0;
    for k=1:length(R_L_plot_configs_data)
        num_unique_points = num_unique_points + numel(R_L_plot_configs_data{k}{1}) * numel(R_L_plot_configs_data{k}{2});
    end
    
    plot_colors_configs_local = lines(max(1, num_unique_points)); 
    legend_entries_local = {};
    line_counter = 0;

    for config_idx = 1:length(R_L_plot_configs_data)
        current_R_set = R_L_plot_configs_data{config_idx}{1};
        current_L_set = R_L_plot_configs_data{config_idx}{2};

        if isscalar(current_R_set), current_R_set = [current_R_set]; end 
        if isscalar(current_L_set), current_L_set = [current_L_set]; end

        for r_val_sel = current_R_set
            for l_val_sel = current_L_set
                line_counter = line_counter + 1;
                [~, r_idx_closest] = min(abs(R_grid - r_val_sel));
                [~, l_idx_closest] = min(abs(L_grid - l_val_sel));

                temp_over_time = squeeze(Temp_3D_matrix_data(r_idx_closest, l_idx_closest, :));
                
                plot(ax, t_array, temp_over_time, 'o', 'Color', plot_colors_configs_local(line_counter,:), 'MarkerFaceColor', plot_colors_configs_local(line_counter,:));
                
                if length(t_array) >= 2
                    t_interp = linspace(min(t_array), max(t_array), n_interp);
                    temp_interp = interp1(t_array, temp_over_time, t_interp, 'makima');
                    plot(ax, t_interp, temp_interp, '-', 'Color', plot_colors_configs_local(line_counter,:), 'LineWidth', 1.5);
                end
                
                legend_entries_local{end+1} = sprintf('R=%.2f, L=%.2f', R_grid(r_idx_closest), L_grid(l_idx_closest));
            end
        end
    end
    xlabel(ax, 'Время, мин');
    ylabel(ax, 'Температура, °C');
    title(ax, 'Изменение температуры со временем в различных точках');
    if ~isempty(legend_entries_local)
        % Удаляем дубликаты в легенде, если графики перекрываются (только линии, а не точки)
        [~, ia] = unique(legend_entries_local, 'stable');
        legend(ax, legend_entries_local(ia), 'Location', 'bestoutside'); 
    end
    hold(ax, 'off');
    view(ax, 2); 
    axis tight;
end

% 3D график T(R,t) для фиксированного L
function plot_surf_R_t(ax, t_array_data, R_coords_data, T_data_2D, L_val) 
    [T_mesh, R_mesh] = meshgrid(t_array_data, R_coords_data);
    surf(ax, T_mesh, R_mesh, T_data_2D, 'EdgeColor','none', 'FaceAlpha',0.8);
    xlabel(ax, 'Время, мин');
    ylabel(ax, 'Безразмерная координата R');
    zlabel(ax, 'Температура, °C');
    title(ax, sprintf('Температура T(R,t) при L=%.2f', L_val));
    colorbar(ax); 
    view(ax, -30, 30); 
    axis tight;
end

% 3D график T(L,t) для фиксированного R
function plot_surf_L_t(ax, t_array_data, L_coords_data, T_data_2D, R_val) 
    [T_mesh, L_mesh] = meshgrid(t_array_data, L_coords_data);
    surf(ax, T_mesh, L_mesh, T_data_2D, 'EdgeColor','none', 'FaceAlpha',0.8);
    xlabel(ax, 'Время, мин');
    ylabel(ax, 'Безразмерная координата L');
    zlabel(ax, 'Температура, °C');
    title(ax, sprintf('Температура T(L,t) при R=%.2f', R_val));
    colorbar(ax); 
    view(ax, -30, 30);
    axis tight;
end

% Температура в одной точке со временем (2D график) с интерполяцией Акимы
function plot_temp_over_time_single(ax, t_array_data, T_data_1D, R_val, L_val, n_interp)
    T_data_1D_sq = squeeze(T_data_1D);
    plot(ax, t_array_data, T_data_1D_sq, 'o', 'MarkerFaceColor','b', 'DisplayName', 'Расчетные точки');
    hold(ax, 'on');

    if length(t_array_data) >= 2
        t_interp = linspace(min(t_array_data), max(t_array_data), n_interp);
        T_interp = interp1(t_array_data, T_data_1D_sq, t_interp, 'makima');
        plot(ax, t_interp, T_interp, '-', 'Color', [0 0 0.7], 'LineWidth', 1.5, 'DisplayName', 'Сплайн Акимы');
    end
    
    xlabel(ax, 'Время, мин');
    ylabel(ax, 'Температура, °C');
    title(ax, sprintf('Изменение температуры для R=%.2f, L=%.2f', R_val, L_val));
    legend(ax, 'show', 'Location', 'best');
    hold(ax, 'off');
    view(ax, 2); 
    axis tight;
end