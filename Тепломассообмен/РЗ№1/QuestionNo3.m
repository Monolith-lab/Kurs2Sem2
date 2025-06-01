%% Очистка и подготовка рабочего пространства
% Эти команды нужны, чтобы начать работу с "чистого листа" каждый раз, когда вы запускаете скрипт.
% Это помогает избежать ошибок из-за данных, оставшихся от предыдущих запусков.
clear;           % Удаляет все переменные из рабочей области MATLAB.
clc;             % Очищает командное окно MATLAB (то, где вы видите сообщения).
close all;       % Закрывает все открытые графические окна (фигуры).
clear functions; % Очищает кэшированные функции, что полезно, если вы меняете вспомогательные функции.

tic; % Запускаем таймер. Это позволяет узнать, сколько времени занимает выполнение всего скрипта.

%% =========================================================================
%% --- Входные параметры, задаваемые пользователем (по условию задачи) ---
% В этом разделе мы определяем все исходные данные, которые даны в условии задачи.
% Это важно, чтобы модель точно соответствовала заданию.
% =========================================================================

% 1. Имя файла для сохранения результатов (по условию задачи сохраняется)
% Здесь мы указываем, как будет называться файл Excel, куда будут записаны все расчеты.
filename_excel = 'Результаты_Температуры_Задачи3.xlsx';

% 2. Константы для поиска "корней" (собственных значений) и точности расчетов
% Эти числа определяют, насколько точно математические расчеты будут искать решения
% для тепловых процессов. Обычно их не нужно менять.
n_desired_roots = 25; % Количество "корней" (собственных значений), которые мы будем искать.
                      % Чем больше корней, тем точнее расчеты, но тем дольше. 25 обычно достаточно.
epsilon = 1e-9;       % Очень маленькое число, используемое для сравнения с нулем или для
                      % создания небольших отступов в математических интервалах.
fzero_options = optimset('TolX', 1e-9, 'Display', 'off'); % Настройки для функции 'fzero'.
                                                           % 'fzero' — это "поисковик", который находит,
                                                           % где математическая функция становится равной нулю.
                                                           % 'TolX' - это точность поиска, 'Display', 'off'
                                                           % означает, что fzero не будет показывать свои внутренние
                                                           % шаги в командном окне.

% 3. Температуры (по условию задачи)
t0 = 760; % Начальная температура цилиндра, °C. Это температура, с которой он начал охлаждаться.
tf = 22;  % Температура окружающей среды (воды в бассейне), °C. Это температура, к которой стремится цилиндр.

% 4. Геометрические размеры цилиндра (по условию задачи)
d_mm = 380;       % Диаметр цилиндра в миллиметрах (мм).
L_total_mm = 440; % Общая длина цилиндра в миллиметрах (мм).

% Переводим миллиметры в метры, так как в физических формулах используются метры.
r0 = d_mm / 2 / 1000;      % Радиус цилиндра (r0). Делим диаметр на 2, затем на 1000 (для перевода в метры).
l0 = L_total_mm / 2 / 1000; % Половина длины цилиндра (l0). Для расчетов обычно используют половину толщины/длины.

% Вычисляем объем цилиндра. Это нужно будет для расчета количества отданного тепла.
V_cylinder_total = pi * r0^2 * (2 * l0); % Объем = pi * (радиус^2) * (полная длина).

% 5. Свойства материала цилиндра (Сталь 15Cr,10Ni) и среды (по условию задачи)
rho = 7865;      % Плотность материала, [кг/м^3]. Сколько весит кубический метр стали.
cp = 460;        % Удельная теплоёмкость материала, [Дж/(кг·К)]. Сколько энергии нужно, чтобы нагреть 1 кг стали на 1 градус.
lambda = 19;     % Теплопроводность материала, [Вт/(м·К)]. Насколько хорошо сталь проводит тепло.
alpha_conv = 100; % Коэффициент теплоотдачи, [Вт/(м^2·К)]. Насколько хорошо тепло переходит от поверхности стали к воде.

% Расчет коэффициента температуропроводности 'a'
% Это свойство материала, которое показывает, насколько быстро температура распространяется внутри тела.
a = lambda / (rho * cp); % Формула: a = теплопроводность / (плотность * удельная теплоёмкость). Единицы: [м^2/с].

% Расчет чисел Био (Bi)
% Число Био - это безразмерный критерий, который показывает соотношение между
% теплоотдачей с поверхности тела и теплопроводностью внутри тела.
% Bi < 0.1 обычно означает, что температура внутри тела равномерна.
% BiL - для охлаждения "плоской пластины" (в нашем случае, торцы цилиндра).
BiL = alpha_conv * l0 / lambda; % Формула: Bi = (коэффициент теплоотдачи * характерный размер) / теплопроводность.
% BiR - для охлаждения "бесконечного цилиндра" (в нашем случае, боковая поверхность цилиндра).
BiR = alpha_conv * r0 / lambda; % Формула та же, но с радиусом как характерным размером.

% 6. Время расчета (по условию задачи, t1 = 32 мин)
t1_minutes = 32; % Момент времени τ₁ из Задания 1 и Задания 3.

% Создаем массивы времени для различных расчетов.
% 't_minutes_for_calc' используется для точного расчета температурного поля в момент t1.
t_minutes_for_calc = unique([0, 1:1:t1_minutes]); % Массив времени от 0 до t1 с шагом 1 минута.
                                                 % 'unique' и 'sort' гарантируют, что 0 и t1_minutes включены.
t_seconds_for_calc = t_minutes_for_calc * 60;    % Переводим минуты в секунды, так как 'a' (температуропроводность)
                                                 % выражена в м^2/с.

% 't_minutes_for_plots_and_regular_regime' используется для построения графиков
% изменения температуры со временем, а также для анализа "регулярного режима" (Задание 2).
% Диапазон времени здесь больше, чтобы можно было увидеть, как температура стабилизируется.
t_minutes_for_plots_and_regular_regime = unique([t_minutes_for_calc, 0:2:120]); % От 0 до 120 минут с шагом 2 минуты.
t_seconds_for_plots_and_regular_regime = t_minutes_for_plots_and_regular_regime * 60; % Переводим в секунды.

% 7. Разрешение ВНУТРЕННЕЙ расчетной сетки для построения пространственных профилей
% Эти параметры определяют, сколько точек будет использоваться для построения профилей
% температуры внутри цилиндра по радиусу (R) и длине (L).
% R_grid_points - безразмерные координаты от центра (0) до поверхности (1) по радиусу.
R_grid_points = linspace(0, 1, 21); % 21 равномерно расположенная точка от 0 до 1.
% L_grid_points - безразмерные координаты от центра (0) до торца (1) по длине.
L_grid_points = linspace(0, 1, 21); % 21 равномерно расположенная точка от 0 до 1.

% 8. Количество точек для интерполяции сплайном Акимы на 2D графиках
% Это число определяет, насколько гладкими будут линии на графиках. Больше точек = глаже линия.
n_interp_points_akima = 200;

%% =========================================================================
%% --- ВАЛИДАЦИЯ ВХОДНЫХ ПАРАМЕТРОВ ---
% Это важный блок для проверки, что все введенные данные имеют смысл.
% Например, температура должна быть числом, а размеры - положительными.
% Если найдена проблема, скрипт остановится и выдаст понятное сообщение об ошибке.
% =========================================================================
fprintf('--- Валидация входных параметров ---\n'); % Выводим сообщение в командное окно.

% Проверяем, являются ли температуры скалярными (одиночными) числами.
if ~isscalar(t0) || ~isnumeric(t0) || ~isscalar(tf) || ~isnumeric(tf)
    error('Начальная и конечная температуры (t0, tf) должны быть скалярными числами.');
end
% Если начальная температура тела очень близка к температуре среды, предупреждаем.
% В этом случае температура тела практически не будет меняться.
if abs(t0 - tf) < epsilon 
    warning('Начальная температура тела (t0) практически равна температуре окружающей среды (tf). Температура тела останется постоянной и равной tf. Безразмерная температура (Тета) не будет иметь физического смысла в своей обычной интерпретации, так как (t0-tf) близко к нулю.');
end

% Проверяем коэффициент температуропроводности и характерные размеры.
% Они должны быть положительными числами.
if ~isscalar(a) || ~isnumeric(a) || a <= 0
    error('Коэффициент температуропроводности (a) должен быть положительным числом.');
end
if ~isscalar(r0) || ~isnumeric(r0) || r0 <= 0
    error('Характерный размер для цилиндра (r0) должен быть положительным числом.');
end
if ~isscalar(l0) || ~isnumeric(l0) || l0 <= 0
    error('Характерный размер для пластины (l0) должен быть положительным числом.');
end

% Проверяем массив времени. Он должен содержать неотрицательные числа.
if ~isnumeric(t_minutes_for_calc) || isempty(t_minutes_for_calc) || any(t_minutes_for_calc < 0)
    error('Массив времени для расчета (t_minutes_for_calc) должен содержать неотрицательные числа.');
end

% Проверяем числа Био. Они должны быть неотрицательными.
if ~isscalar(BiL) || ~isnumeric(BiL) || BiL < 0
    error('Число Био для пластины (BiL) должно быть неотрицательным числом.');
end
if ~isscalar(BiR) || ~isnumeric(BiR) || BiR < 0
    error('Число Био для цилиндра (BiR) должно быть неотрицательным числом.');
end

% Проверяем координаты сетки. Они должны быть массивом чисел в диапазоне от 0 до 1.
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

fprintf('Валидация завершена. Параметры корректны.\n'); % Сообщение об успешной валидации.

%% =========================================================================
%% --- Основная часть расчета ---
% Здесь происходят основные математические вычисления для решения задачи.
% =========================================================================

% --- 1. Поиск "корней" (собственных значений) характеристических уравнений ---
% Для каждого типа геометрии (пластина, цилиндр) существуют свои математические
% уравнения, описывающие тепловой процесс. Решение этих уравнений - это
% набор бесконечных "корней" (собственных значений), которые определяют, как
% температура меняется со временем и по координатам. Мы ищем первые 'n_desired_roots' из них.

slab_roots_res = []; % Переменная для хранения корней для пластины.
cyl_roots_res = [];  % Переменная для хранения корней для цилиндра.

fprintf('--- 1. Поиск корней для пластины (BiL = %f) ---\n', BiL);
% Если число Био (BiL) очень близко к нулю, это означает особый случай,
% и корни можно найти по простой аналитической формуле (m*pi).
if abs(BiL) < epsilon 
    slab_roots_res = (1:n_desired_roots)' * pi; % Корни будут 1*pi, 2*pi, 3*pi и т.д.
    fprintf('BiL близок к 0. Использование аналитических корней (m*pi) для пластины.\n');
else % В общем случае, когда BiL не ноль, корни ищутся численно.
    eq_slab = @(x) x .* tan(x) - BiL; % Это характеристическое уравнение для пластины.
    slab_roots_res = find_positive_roots(eq_slab, n_desired_roots, @(k_idx) slab_interval_generator(k_idx, epsilon), fzero_options, epsilon, 'Пластина');
    % 'find_positive_roots' - это наша вспомогательная функция, которая ищет корни.
    % Она использует 'fzero' для поиска, где `x * tan(x) - BiL` становится нулем.
end
fprintf('Найдено %d корней для пластины.\n', length(slab_roots_res));

fprintf('--- 2. Поиск корней для цилиндра (BiR = %f) ---\n', BiR);
% Аналогично для цилиндра. Если число Био (BiR) очень близко к нулю,
% корни соответствуют нулям функции Бесселя первого рода J1(x).
if abs(BiR) < epsilon 
    cyl_roots_res = zeros(n_desired_roots,1);
    last_J1_zero = 0;
    % Ищем нули функции Бесселя J1(x).
    for i_rt = 1:n_desired_roots
        % Определяем интервалы, где предположительно находится следующий ноль J1(x).
        % Это помогает 'fzero' быстрее и надежнее найти корень.
        lower_search_J1 = 0; 
        upper_search_J1 = 0; 
        if i_rt == 1
            lower_search_J1 = 3.0; 
            upper_search_J1 = 4.5; % Приблизительный интервал для первого нуля J1(x).
        else
            lower_search_J1 = last_J1_zero + pi/2; % Следующие интервалы определяются относительно предыдущего нуля.
            upper_search_J1 = last_J1_zero + pi + pi/2; 
        end
        lower_search_J1 = max(lower_search_J1, epsilon); % Убеждаемся, что нижняя граница больше нуля.

        val_lower_j1 = besselj(1, lower_search_J1); % Вычисляем J1 в нижней границе.
        val_upper_j1 = besselj(1, upper_search_J1); % Вычисляем J1 в верхней границе.

        search_attempts_j1 = 0; max_search_attempts_j1 = 5;
        % Если функция J1 не меняет знак в интервале (нет корня), пытаемся расширить интервал.
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

        try % Используем 'try-catch' для обработки возможных ошибок при поиске корня.
            current_J1_zero = fzero(@(x) besselj(1,x), [lower_search_J1, upper_search_J1], fzero_options);
            if ~isnan(current_J1_zero) && current_J1_zero > last_J1_zero + epsilon 
                cyl_roots_res(i_rt) = current_J1_zero; 
                last_J1_zero = current_J1_zero;       % Обновляем значение последнего найденного корня.
            else
                 error('Не удалось найти ноль J1 #%d для BiR=0 или он не возрастает (найдено %.4f, последний был %.4f).', i_rt, current_J1_zero, last_J1_zero); 
            end
        catch ME_j1_zero
            fprintf('Детали ошибки для нуля J1 #%d (BiR=0): Интервал: [%.4f, %.4f], f(low)=%.4e, f(high)=%.4e\n', i_rt, lower_search_J1, upper_search_J1, besselj(1,lower_search_J1), besselj(1,upper_search_J1));
            error('Ошибка при поиске нуля J1 #%d: %s.', i_rt, ME_j1_zero.message);
        end
    end
    fprintf('BiR близок к 0. Использование аналитических корней (нулей J1(x)) для цилиндра.\n');
else % В общем случае, когда BiR не ноль.
    eq_cyl = @(x) x .* besselj(1,x) - BiR .* besselj(0,x); % Характеристическое уравнение для цилиндра.
                                                           % Здесь используются функции Бесселя J0(x) и J1(x).
    num_j0_zeros_needed = n_desired_roots + 5; 
    zeros_J0_for_intervals = get_J0_zeros(num_j0_zeros_needed, fzero_options, epsilon); 
    cyl_interval_gen_func = @(k_idx) get_cylinder_interval(k_idx, zeros_J0_for_intervals, epsilon);
    cyl_roots_res = find_positive_roots(eq_cyl, n_desired_roots, cyl_interval_gen_func, fzero_options, epsilon, 'Цилиндр');
end
fprintf('Найдено %d корней для цилиндра.\n', length(cyl_roots_res));

% --- 3. Расчет температурного поля и сбор данных ---
% Здесь мы вычисляем температуру в каждой точке внутри цилиндра (по координатам R и L)
% для каждого заданного момента времени.

% Определяем размеры массивов для хранения результатов.
num_R_grid = length(R_grid_points);     % Количество точек по радиусу.
num_L_grid = length(L_grid_points);     % Количество точек по длине.
num_T_steps_calc = length(t_minutes_for_calc); % Количество моментов времени для расчетов до t1.
num_T_steps_plots = length(t_minutes_for_plots_and_regular_regime); % Количество моментов времени для графиков.

% Создаем 3D-матрицы для хранения температур.
% Temperature_3D_matrix_calc: для расчета температурного поля до t1 (Задание 1).
Temperature_3D_matrix_calc = zeros(num_R_grid, num_L_grid, num_T_steps_calc);
% Temperature_3D_matrix_plots: для более широкого диапазона времени (для графиков и Задания 2).
Temperature_3D_matrix_plots = zeros(num_R_grid, num_L_grid, num_T_steps_plots);

% Переменные для сбора данных, которые будут записаны в Excel.
T_excel_data = cell(num_R_grid * num_L_grid * num_T_steps_plots, 4); % Для общего температурного поля.
excel_data_rows_counter = 0; % Счетчик строк.

SlabTheta_excel_data = cell(num_L_grid * num_T_steps_plots, 4); % Для безразмерной температуры пластины.
slab_theta_data_counter = 0;

CylinderTheta_excel_data = cell(num_R_grid * num_T_steps_plots, 4); % Для безразмерной температуры цилиндра.
cyl_theta_data_counter = 0;

fprintf('--- 3. Расчет температур и сбор данных для вывода ---\n');

% Расчет для "короткого" диапазона времени (до t1 = 32 мин).
% Этот массив температур будет использоваться для Задания 1.
for t_idx = 1:num_T_steps_calc
    current_t_minutes = t_minutes_for_calc(t_idx);   % Текущее время в минутах.
    current_t_seconds = t_seconds_for_calc(t_idx);   % Текущее время в секундах.

    % !!! НАЧАЛО ИСПРАВЛЕНИЯ для t=0 !!!
    if current_t_seconds == 0
        % В момент времени t=0 вся заготовка имеет начальную температуру t0
        Temperature_2D = ones(num_R_grid, num_L_grid) * t0;
    else
        % Для t > 0 используем аналитическое решение
        % Вычисляем числа Фурье (Fo) для текущего времени.
        current_FoR = a * current_t_seconds / (r0^2); % Для цилиндра (с радиусом r0).
        current_FoL = a * current_t_seconds / (l0^2); % Для пластины (с половиной длины l0).

        % Вычисляем безразмерные температуры Тета (Theta).
        theta_slab_at_time_t = calculate_theta_slab(BiL, current_FoL, L_grid_points, slab_roots_res, epsilon);
        theta_cyl_at_time_t = calculate_theta_cylinder(BiR, current_FoR, R_grid_points, cyl_roots_res, epsilon);

        % Поскольку цилиндр охлаждается со всех сторон, его безразмерная температура
        % является произведением безразмерных температур "бесконечной пластины" (для торцов)
        % и "бесконечного цилиндра" (для боковой поверхности). Это принцип суперпозиции.
        teta_itog_2D = theta_cyl_at_time_t * theta_slab_at_time_t'; 

        % Переводим безразмерную температуру обратно в градусы Цельсия.
        if abs(t0 - tf) < epsilon % Если начальная температура = температуре среды, температура не меняется.
            Temperature_2D = ones(size(teta_itog_2D)) * tf; 
        else % Общая формула для перевода безразмерной температуры в абсолютную.
            Temperature_2D = teta_itog_2D * (t0 - tf) + tf;
        end
    end % !!! КОНЕЦ ИСПРАВЛЕНИЯ для t=0 !!!
    
    Temperature_3D_matrix_calc(:,:,t_idx) = Temperature_2D; % Сохраняем температуру в матрицу.

    % Выводим информацию о времени, если оно совпадает с t1_minutes.
    if current_t_minutes == t1_minutes
         fprintf('  Время: %.1f мин (FoL=%.4e, FoR=%.4e) - (для Задания 1)\n', current_t_minutes, current_FoL, current_FoR);
    end
end

% Расчет для "длинного" диапазона времени (до 120 мин).
% Этот массив температур будет использоваться для графиков T(t) и Задания 2.
for t_idx = 1:num_T_steps_plots
    current_t_minutes = t_minutes_for_plots_and_regular_regime(t_idx);
    current_t_seconds = t_seconds_for_plots_and_regular_regime(t_idx);

    % !!! НАЧАЛО ИСПРАВЛЕНИЯ для t=0 !!!
    if current_t_seconds == 0
        % В момент времени t=0 вся заготовка имеет начальную температуру t0
        Temperature_2D = ones(num_R_grid, num_L_grid) * t0;
        % Для сбора данных в Excel и корректного Fo/Theta для t=0
        theta_slab_at_time_t = ones(size(L_grid_points)) * 1.0; % Безразмерная температура = 1.0 при t=0
        theta_cyl_at_time_t = ones(size(R_grid_points)) * 1.0; % Безразмерная температура = 1.0 при t=0
        current_FoL = 0; % Число Фурье равно 0 при t=0
        current_FoR = 0; % Число Фурье равно 0 при t=0
    else
        % Для t > 0 используем аналитическое решение
        current_FoR = a * current_t_seconds / (r0^2);
        current_FoL = a * current_t_seconds / (l0^2);
        
        theta_slab_at_time_t = calculate_theta_slab(BiL, current_FoL, L_grid_points, slab_roots_res, epsilon);
        theta_cyl_at_time_t = calculate_theta_cylinder(BiR, current_FoR, R_grid_points, cyl_roots_res, epsilon);

        teta_itog_2D = theta_cyl_at_time_t * theta_slab_at_time_t'; 

        if abs(t0 - tf) < epsilon 
            Temperature_2D = ones(size(teta_itog_2D)) * tf; 
        else
            Temperature_2D = teta_itog_2D * (t0 - tf) + tf;
        end
    end % !!! КОНЕЦ ИСПРАВЛЕНИЯ для t=0 !!!
    
    Temperature_3D_matrix_plots(:,:,t_idx) = Temperature_2D; % Сохраняем температуру в матрицу для графиков.

    % Выводим информацию о времени с определенной периодичностью.
    if mod(current_t_minutes, 20) == 0 && current_t_minutes > 0
         fprintf('  Время: %.1f мин (FoL=%.4e, FoR=%.4e) - (для графиков/регулярного режима)\n', current_t_minutes, current_FoL, current_FoR);
    end
    
    % Собираем данные в удобном формате для последующей записи в Excel.
    [L_mesh_for_excel, R_mesh_for_excel] = meshgrid(L_grid_points, R_grid_points);
    current_t_column_temp = repmat(current_t_minutes, num_R_grid * num_L_grid, 1);
    current_data_block_temp = [R_mesh_for_excel(:), L_mesh_for_excel(:), current_t_column_temp, Temperature_2D(:)];

    % Добавляем данные в общий массив для Excel.
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

% Обрезаем массивы данных до фактического количества заполненных строк.
T_excel_data = T_excel_data(1:excel_data_rows_counter, :);
SlabTheta_excel_data = SlabTheta_excel_data(1:slab_theta_data_counter, :);
CylinderTheta_excel_data = CylinderTheta_excel_data(1:cyl_theta_data_counter, :);

% Преобразуем данные в формат таблицы MATLAB. Это удобно для работы и записи в Excel.
T_data_table = cell2table(T_excel_data, 'VariableNames', {'R_координата', 'L_координата', 'Время_минуты', 'Температура_Цельсий'});
SlabTheta_table = cell2table(SlabTheta_excel_data, 'VariableNames', {'L_координата', 'Время_минуты', 'FoL', 'Тета_Пластина'});
CylinderTheta_table = cell2table(CylinderTheta_excel_data, 'VariableNames', {'R_координата', 'Время_минуты', 'FoR', 'Тета_Цилиндр'});
SlabRoots_table = table((1:length(slab_roots_res))', slab_roots_res, 'VariableNames', {'Индекс_Корня', 'Значение_Корня'});
CylinderRoots_table = table((1:length(cyl_roots_res))', cyl_roots_res, 'VariableNames', {'Индекс_Корня', 'Значение_Корня'});


%% =========================================================================
%% --- 4. Расчеты для вопросов 1, 2, 3 (по условию задачи) ---
% Здесь мы выполняем специфические расчеты, которые требуются в каждом пункте задания.
% =========================================================================

fprintf('--- 4. Расчеты для ответов на вопросы задачи ---\n');

% Задание 1: Рассчитать температурное поле в цилиндре как функцию радиуса r и линейной координаты x
% в момент времени t1 = 32 мин. Результаты свести в таблицу, построить графики.

% Находим индекс (позицию) времени t1 = 32 мин в массиве, который использовался для расчетов.
[~, t1_idx_calc] = min(abs(t_minutes_for_calc - t1_minutes));
% Извлекаем 2D-матрицу температур для этого момента времени.
T_at_t1 = Temperature_3D_matrix_calc(:,:,t1_idx_calc);

% Подготовка данных для таблицы, которая будет отображаться в GUI (Графическом Интерфейсе Пользователя).
% Объединяем координаты R, L и соответствующие им температуры в одну таблицу.
[L_mesh_Q1, R_mesh_Q1] = meshgrid(L_grid_points, R_grid_points);
Q1_table_data = [R_mesh_Q1(:), L_mesh_Q1(:), T_at_t1(:)];
Q1_table = array2table(Q1_table_data, 'VariableNames', {'R_координата', 'L_координата', 'Температура_Цельсий'});

% Задание 2: Имитируя эксперимент, рассчитать температуру в центре цилиндра как функцию времени,
% затем для стадии регулярного режима охлаждения вычислить "экспериментальный" темп охлаждения
% цилиндра и температуропроводность материала заготовки.

% Выбираем температуру в центре цилиндра (R=0, L=0) из матрицы температур для графиков.
% 'squeeze' убирает лишние размерности массива, делая его одномерным вектором.
[~, t_center_idx_R] = min(abs(R_grid_points - 0)); % Находим индекс для R=0 (центр по радиусу).
[~, t_center_idx_L] = min(abs(L_grid_points - 0)); % Находим индекс для L=0 (центр по длине).
T_center_over_time = squeeze(Temperature_3D_matrix_plots(t_center_idx_R, t_center_idx_L, :));

% Расчет для стадии регулярного режима:
% Регулярный режим - это стадия охлаждения, когда распределение температуры
% внутри тела становится "постоянным" по форме, и вся температура тела падает
% экспоненциально со временем. На этом этапе скорость охлаждения подчиняется
% простому закону.

% Чтобы найти темп охлаждения, мы используем график зависимости ln((T-Tf)/(T0-Tf)) от времени.
% Этот график должен быть прямой линией в регулярном режиме, а ее наклон (с минусом)
% даст нам "экспериментальный" темп охлаждения (m_exp).
% Мы берем последние точки (обычно 5-10) для линейной регрессии, чтобы убедиться,
% что мы находимся в регулярном режиме.
min_points_for_regression = 5; % Минимальное количество точек для выполнения регрессии.
if length(t_seconds_for_plots_and_regular_regime) >= min_points_for_regression
    num_points_to_fit = min(10, length(t_seconds_for_plots_and_regular_regime) - 1); % Выбираем до 10 последних точек.
    if num_points_to_fit >= 2 % Для линейной регрессии нужно минимум 2 точки.
        % Выбираем данные для регрессии.
        t_regime_fit = t_seconds_for_plots_and_regular_regime(end - num_points_to_fit + 1 : end);
        T_center_regime_fit = T_center_over_time(end - num_points_to_fit + 1 : end);

        % Вычисляем безразмерную температуру Theta для этих точек.
        theta_center_regime_fit = (T_center_regime_fit - tf) / (t0 - tf);
        
        % Отфильтровываем значения Theta, которые очень малы или отрицательны,
        % чтобы избежать ошибок при взятии логарифма (ln(<=0)).
        valid_indices = theta_center_regime_fit > epsilon;
        if sum(valid_indices) < 2
            warning('Недостаточно допустимых точек для расчета регулярного режима (после фильтрации theta <= epsilon).');
            m_exp = NaN;       % Устанавливаем NaN (Not a Number), если расчет невозможен.
            a_exp_calc = NaN;
        else
            t_regime_fit = t_regime_fit(valid_indices);
            theta_center_regime_fit = theta_center_regime_fit(valid_indices);
            ln_theta_center_regime_fit = log(theta_center_regime_fit); % Берем натуральный логарифм.
            
            % Выполняем линейную регрессию: y = P(1)*x + P(2).
            % P(1) - это наклон прямой, P(2) - это свободный член.
            P = polyfit(t_regime_fit, ln_theta_center_regime_fit, 1);
            m_exp = -P(1); % "Экспериментальный" темп охлаждения 'm' равен отрицательному наклону. [1/с]

            % Для расчета "экспериментальной" температуропроводности (a_exp_calc)
            % используется связь между темпом охлаждения и первыми корнями
            % характеристических уравнений.
            mu1 = cyl_roots_res(1); % Первый корень для цилиндра.
            nu1 = slab_roots_res(1); % Первый корень для пластины.
            if mu1 < epsilon || nu1 < epsilon
                 warning('Первые корни близки к нулю. Расчет "экспериментальной" температуропроводности может быть некорректен.');
                 a_exp_calc = NaN;
            else
                % Формула для температуропроводности из стадии регулярного режима:
                % a = m / ( (mu1/R0)^2 + (nu1/L0)^2 )
                 a_exp_calc = m_exp / ( (mu1/r0)^2 + (nu1/l0)^2 ); % "Экспериментальная" температуропроводность. [м^2/с]
            end
        end
    else
        m_exp = NaN; a_exp_calc = NaN;
        warning('Недостаточно точек для линейной регрессии в регулярном режиме. (Нужно минимум 2)');
    end
else
    m_exp = NaN; a_exp_calc = NaN;
    warning('Недостаточно точек для линейной регрессии в регулярном режиме.');
end

% Задание 3: Вычислить количество теплоты, отданной цилиндром за время охлаждения
% от его начала до момента t1.

% Время t1 в секундах.
current_t_seconds_at_t1 = t1_minutes * 60;
% Числа Фурье для этого момента времени.
FoL_at_t1 = a * current_t_seconds_at_t1 / (l0^2);
FoR_at_t1 = a * current_t_seconds_at_t1 / (r0^2);

% Расчет безразмерной СРЕДНЕЙ температуры для всего объема тела.
% Это делается отдельно для "пластины" (охлаждение торцов) и "цилиндра" (охлаждение боковой поверхности).
theta_avg_slab_at_t1 = calculate_theta_avg_slab(BiL, FoL_at_t1, slab_roots_res, epsilon);
theta_avg_cyl_at_t1 = calculate_theta_avg_cylinder(BiR, FoR_at_t1, cyl_roots_res, epsilon);

% Общая безразмерная средняя температура всего цилиндра является
% произведением средних безразмерных температур пластины и цилиндра.
theta_avg_total_at_t1 = theta_avg_slab_at_t1 * theta_avg_cyl_at_t1;

% Переводим общую безразмерную среднюю температуру в абсолютную температуру в °C.
T_avg_at_t1 = theta_avg_total_at_t1 * (t0 - tf) + tf;

% Количество отданной теплоты (Q) вычисляется по формуле:
% Q = V * ρ * cp * (T_initial_avg - T_final_avg).
% Где T_initial_avg = t0, а T_final_avg = T_avg_at_t1.
% Или эквивалентно: Q = V * ρ * cp * (t0 - tf) * (1 - Theta_avg_total).
Q_total_lost = V_cylinder_total * rho * cp * (t0 - T_avg_at_t1); % Результат в Джоулях [Дж].

%% =========================================================================
%% --- 5. Подготовка данных для GUI графиков ---
% Здесь мы определяем, какие графики будут доступны в нашем графическом интерфейсе
% и как они будут строиться.
% =========================================================================
fprintf('--- 5. Подготовка данных для GUI графиков ---\n');

% Определяем t1_idx для графиков, которые будут использовать массив plots.
[~, t1_idx_plots] = min(abs(t_minutes_for_plots_and_regular_regime - t1_minutes));

plot_names = {};      % Массив для хранения названий графиков, которые будут отображаться в списке GUI.
plot_functions = {}; % Массив для хранения функций, которые будут строить эти графики.

% =========================================================================
% Группа 1: Графики температур по координате при t1 = 32 мин (профили из Задания 1)
% Это "мгновенные снимки" распределения температуры внутри цилиндра в определенный момент времени.
% =========================================================================
fprintf('   -> Подготовка профилей температуры по координатам при t=%.1f мин (по всей сетке)\n', t1_minutes);

% t(x, 0, τ₁) - Профиль T(L) при R=0 (температура вдоль оси цилиндра)
[~, r_idx_at_0] = min(abs(R_grid_points - 0)); % Находим индекс для R=0.
plot_names{end+1} = sprintf('1.1. Профиль T(L) при R=0 (ось), t=%.1f мин', t1_minutes);
plot_functions{end+1} = @(ax) plot_profile_L_fixed_R(ax, L_grid_points, squeeze(T_at_t1(r_idx_at_0, :)), R_grid_points(r_idx_at_0), t1_minutes, n_interp_points_akima);

% t(x, r₀, τ₁) - Профиль T(L) при R=1 (температура вдоль поверхности цилиндра)
[~, r_idx_at_1] = min(abs(R_grid_points - 1)); % Находим индекс для R=1 (поверхность по радиусу).
plot_names{end+1} = sprintf('1.2. Профиль T(L) при R=1 (поверхность), t=%.1f мин', t1_minutes);
plot_functions{end+1} = @(ax) plot_profile_L_fixed_R(ax, L_grid_points, squeeze(T_at_t1(r_idx_at_1, :)), R_grid_points(r_idx_at_1), t1_minutes, n_interp_points_akima);

% t(0, r, τ₁) - Профиль T(R) при L=0 (температура по радиусу в центре цилиндра по длине)
[~, l_idx_at_0] = min(abs(L_grid_points - 0)); % Находим индекс для L=0 (центр по длине).
plot_names{end+1} = sprintf('1.3. Профиль T(R) при L=0 (центр по длине), t=%.1f мин', t1_minutes);
plot_functions{end+1} = @(ax) plot_profile_R_fixed_L(ax, R_grid_points, squeeze(T_at_t1(:, l_idx_at_0)), L_grid_points(l_idx_at_0), t1_minutes, n_interp_points_akima);

% t(L/2, r, τ₁) - Профиль T(R) при L=1 (температура по радиусу на торце цилиндра)
[~, l_idx_at_1] = min(abs(L_grid_points - 1)); % Находим индекс для L=1 (конец по длине/торец).
plot_names{end+1} = sprintf('1.4. Профиль T(R) при L=1 (торец), t=%.1f мин', t1_minutes);
plot_functions{end+1} = @(ax) plot_profile_R_fixed_L(ax, R_grid_points, squeeze(T_at_t1(:, l_idx_at_1)), L_grid_points(l_idx_at_1), t1_minutes, n_interp_points_akima);

% Дополнительный 3D профиль, показывающий температуру по всему объему в момент t1.
if length(R_grid_points) > 1 && length(L_grid_points) > 1
    plot_names{end+1} = sprintf('1.5. 3D профиль T(R,L) при t=%.1f мин', t1_minutes);
    plot_functions{end+1} = @(ax) plot_surf_R_L(ax, L_grid_points, R_grid_points, T_at_t1, t1_minutes);
end

% =========================================================================
% Группа 2: Графики изменения температуры со временем (для Задания 2)
% =========================================================================
fprintf('   -> Подготовка графиков изменения температуры со временем.\n');

% График изменения температуры в центре цилиндра со временем.
% Это то, что требуется для Задания 2.
plot_names{end+1} = sprintf('2.1. T(t) в центре (R=0, L=0)');
plot_functions{end+1} = @(ax) plot_temp_over_time_single(ax, t_minutes_for_plots_and_regular_regime, T_center_over_time, R_grid_points(t_center_idx_R), L_grid_points(t_center_idx_L), n_interp_points_akima);

% График ln((T-Tf)/(T0-Tf)) от времени для анализа регулярного режима.
% На этом графике в регулярном режиме должна быть прямая линия, наклон которой
% дает темп охлаждения.
plot_names{end+1} = sprintf('2.2. ln(Theta) от времени в центре (для регулярного режима)');
plot_functions{end+1} = @(ax) plot_ln_theta_center_over_time(ax, t_seconds_for_plots_and_regular_regime, T_center_over_time, t0, tf, epsilon);


% =========================================================================
% Группа 3: Общие 3D графики температур по координате и времени
% Эти графики показывают, как меняется температурный профиль во времени
% вдоль определенного направления.
% =========================================================================
fprintf('   -> Подготовка 3D графиков температуры по координате и времени (по всей сетке).\n');

% Если меняется только радиус, строим 3D график T(R,t) при фиксированной длине.
if length(R_grid_points) > 1 && length(L_grid_points) <= 1 
    plot_names{end+1} = sprintf('3.1. 3D T(R,t) при L=%.2f', L_grid_points(1));
    plot_functions{end+1} = @(ax) plot_surf_R_t(ax, t_minutes_for_plots_and_regular_regime, R_grid_points, squeeze(Temperature_3D_matrix_plots(:,1,:)), L_grid_points(1));

% Если меняется только длина, строим 3D график T(L,t) при фиксированном радиусе.
elseif length(L_grid_points) > 1 && length(R_grid_points) <= 1 
    plot_names{end+1} = sprintf('3.2. 3D T(L,t) при R=%.2f', R_grid_points(1));
    plot_functions{end+1} = @(ax) plot_surf_L_t(ax, t_minutes_for_plots_and_regular_regime, L_grid_points, squeeze(Temperature_3D_matrix_plots(1,:,:)), R_grid_points(1));

% Если меняются обе координаты, можем построить срезы 3D-поля по одной из координат.
elseif length(R_grid_points) > 1 && length(L_grid_points) > 1 
    % Срезы по длине (фиксируя L)
    plot_L_indices_for_3D_time = unique([1, round(num_L_grid/2), num_L_grid]); % Выбираем начало, середину и конец.
    if length(L_grid_points) < 3, plot_L_indices_for_3D_time = 1:num_L_grid; end % Если точек мало, берем все.

    for i = 1:length(plot_L_indices_for_3D_time)
        l_idx = plot_L_indices_for_3D_time(i);
        plot_names{end+1} = sprintf('3.3. 3D T(R,t) при L=%.2f', L_grid_points(l_idx));
        plot_functions{end+1} = @(ax) plot_surf_R_t(ax, t_minutes_for_plots_and_regular_regime, R_grid_points, squeeze(Temperature_3D_matrix_plots(:, l_idx, :)), L_grid_points(l_idx));
    end

    % Срезы по радиусу (фиксируя R)
    plot_R_indices_for_3D_time = unique([1, round(num_R_grid/2), num_R_grid]);
    if length(R_grid_points) < 3, plot_R_indices_for_3D_time = 1:num_R_grid; end

    for i = 1:length(plot_R_indices_for_3D_time)
        r_idx = plot_R_indices_for_3D_time(i);
        plot_names{end+1} = sprintf('3.4. 3D T(L,t) при R=%.2f', R_grid_points(r_idx));
        plot_functions{end+1} = @(ax) plot_surf_L_t(ax, t_minutes_for_plots_and_regular_regime, L_grid_points, squeeze(Temperature_3D_matrix_plots(r_idx, :, :)), R_grid_points(r_idx));
    end
end


%% =========================================================================
%% --- 6. Создание GUI (Графического Интерфейса Пользователя) ---
% Здесь мы создаем окно, кнопки, списки и поля, чтобы вы могли удобно
% просматривать результаты и графики.
% =========================================================================
fprintf('--- 6. Создание GUI для просмотра результатов ---\n');

% Создаем основное окно (фигуру) GUI.
fig = figure('Name', 'Решение Задачи 3: Теплопередача в Цилиндре', ...
             'NumberTitle', 'off', ...          % Убираем номер окна из заголовка.
             'Units', 'normalized', ...         % Размеры и позиции в долях от размера экрана.
             'WindowStyle', 'normal', ...       % Важно: стандартный стиль окна, позволяющий перемещать.
             'Resize', 'on');                   % Важно: разрешаем изменение размера окна, включая максимизацию.

% Делаем окно полноэкранным
% Проверяем версию MATLAB для использования WindowState
if verLessThan('matlab', '8.4') % WindowState доступен с R2014b
    % Для старых версий используем normalized Position [0 0 1 1]
    set(fig, 'Position', [0 0 1 1]);
else
    % Для новых версий используем WindowState 'maximized'
    set(fig, 'WindowState', 'maximized');
end

% --- ПАНЕЛЬ 1: "Выбранный график" ---
% Эту панель создаем первой, чтобы 'ax_main' (область для рисования графиков)
% была доступна, когда другие элементы GUI будут на нее ссылаться.
panel_plot = uipanel(fig, 'Title', 'Выбранный график', ...
                     'Units', 'normalized', ...
                     'Position', [0.32 0.01 0.67 0.98]); 

% Создаем оси для рисования графиков внутри 'panel_plot'.
ax_main = axes(panel_plot, 'Units', 'normalized', 'Position', [0.1 0.1 0.8 0.8]);

% --- ПАНЕЛЬ 2: "Выберите график" ---
% Эта панель теперь размещается над панелью с численными результатами.
% Занимает верхнюю часть левой стороны окна.
panel_list = uipanel(fig, 'Title', 'Выберите график', ...
                     'Units', 'normalized', ...
                     'Position', [0.01 0.61 0.30 0.38]); 

% Создаем список (listbox) для выбора графиков.
listbox = uicontrol(panel_list, 'Style', 'listbox', ... 
                    'String', plot_names, ... 
                    'Units', 'normalized', ...
                    'Position', [0.05 0.15 0.9 0.8], ...
                    'Callback', @(src, event) listbox_callback(src, event, plot_functions, ax_main));

% Создаем кнопку "Сохранить график".
save_button = uicontrol(panel_list, 'Style', 'pushbutton', ... 
                        'String', 'Сохранить график', ... 
                        'Units', 'normalized', ...
                        'Position', [0.05 0.05 0.9 0.07], ... 
                        'Callback', @(src, event) save_button_callback(listbox, ax_main, plot_names));

% --- ПАНЕЛЬ 3: "Численные результаты (Ответы на вопросы)" ---
panel_results = uipanel(fig, 'Title', 'Численные результаты (Ответы на вопросы)', ...
                        'Units', 'normalized', ...
                        'Position', [0.01 0.01 0.30 0.59]);

% --- Распределение элементов внутри panel_results (УЛУЧШЕННОЕ) ---
% Общие параметры макета внутри панели (координаты нормализованы к панели, 0-1)
margin_left_panel = 0.05;
content_width_panel = 0.90; % Ширина для текста и таблицы
margin_top_panel = 0.02;    % Отступ сверху панели

% Новые, более тонкие настройки отступов
gap_after_title_q1 = 0.01;        % Отступ после заголовка Задания 1
gap_after_title_q2 = 0.01;        % Отступ после заголовка Задания 2 (ИЗМЕНЕНО ДЛЯ "ПРИЖАТИЯ")
gap_after_title_q3 = 0.01;        % Отступ после заголовка Задания 3
gap_between_texts_in_q2 = 0.005;  % Отступ между строчками в Задании 2 (уменьшен)
spacing_q1_q2 = 0.015;            % Отступ между блоками Задание 1 и Задание 2
spacing_q2_q3 = 0.005;            % Отступ между блоками Задание 2 и Задание 3 (уменьшен)
margin_bottom_panel = 0.02;       % Отступ снизу панели (для выравнивания)

% Фиксированные высоты заголовков и текстовых строк (примерные)
h_title_q1 = 0.04;
h_title_q2 = 0.04; % Увеличена высота для "Задание 2"
h_title_q3 = 0.04; % Увеличена высота для "Задание 3"
h_text_q2_line = 0.035; 
h_text_q3_line = 0.035; 

% Расчет высот для адаптивных элементов
% Общая доступная высота для всех элементов внутри panel_results
total_content_height_available = 1.0 - margin_top_panel - margin_bottom_panel;

% Суммируем все "фиксированные" высоты (заголовки и все отступы, кроме таблиц/адаптивных текстов)
fixed_elements_height = h_title_q1 + gap_after_title_q1 + ...
                        h_title_q2 + gap_after_title_q2 + (2 * h_text_q2_line) + gap_between_texts_in_q2 + ... % 2 строки текста в Q2
                        h_title_q3 + gap_after_title_q3 + h_text_q3_line + ... % 1 строка текста в Q3
                        spacing_q1_q2 + spacing_q2_q3;

% Рассчитываем, сколько места осталось для таблицы Задания 1.
h_table_q1_calculated = total_content_height_available - fixed_elements_height;

% Задаем минимальную высоту для таблицы, чтобы избежать ошибок и обеспечить видимость
min_h_table_q1 = 0.10; % Например, 10% от высоты панели
if h_table_q1_calculated < min_h_table_q1
    h_table_q1_calculated = min_h_table_q1;
end


% Размещение элементов сверху вниз
current_y = 1.0 - margin_top_panel;

% --- Задание 1 ---
% Заголовок
current_y = current_y - h_title_q1;
uicontrol(panel_results, 'Style', 'text', 'String', 'Задание 1: Температура в момент t=32 мин (фрагмент)', ...
    'Units', 'normalized', 'Position', [margin_left_panel, current_y, content_width_panel, h_title_q1], ...
    'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 9);
current_y = current_y - gap_after_title_q1; % Отступ после заголовка Q1
% Таблица
current_y = current_y - h_table_q1_calculated;
uitable_Q1 = uitable(panel_results, 'Data', Q1_table{1:5:end,:}, 'ColumnName', Q1_table.Properties.VariableNames, ...
    'Units', 'normalized', 'Position', [margin_left_panel, current_y, content_width_panel, h_table_q1_calculated], ...
    'RowStriping', 'on', 'FontSize', 9, 'ColumnWidth', 'auto');
current_y = current_y - spacing_q1_q2; % Отступ между Q1 и Q2

% --- Задание 2 ---
% Заголовок
current_y = current_y - h_title_q2;
uicontrol(panel_results, 'Style', 'text', 'String', 'Задание 2:', ...
    'Units', 'normalized', 'Position', [margin_left_panel, current_y, content_width_panel, h_title_q2], ...
    'HorizontalAlignment', 'left', 'FontWeight', 'bold', 'FontSize', 9);
current_y = current_y - gap_after_title_q2; % Отступ после заголовка Q2 (ИЗМЕНЕНО)
% Темп охлаждения
current_y = current_y - h_text_q2_line;
uicontrol(panel_results, 'Style', 'text', 'String', sprintf(' "Экспериментальный" темп охлаждения m = %s [1/с]', format_scientific_notation(m_exp, 4)), ...
    'Units', 'normalized', 'Position', [margin_left_panel, current_y, content_width_panel, h_text_q2_line], ...
    'HorizontalAlignment', 'left', 'FontSize', 9);
current_y = current_y - gap_between_texts_in_q2; % Уменьшенный отступ между строчками Q2
% Температуропроводность
current_y = current_y - h_text_q2_line;
uicontrol(panel_results, 'Style', 'text', 'String', sprintf(' "Экспериментальная" температуропроводность a = %s [м^2/с]', format_scientific_notation(a_exp_calc, 4)), ...
    'Units', 'normalized', 'Position', [margin_left_panel, current_y, content_width_panel, h_text_q2_line], ...
    'HorizontalAlignment', 'left', 'FontSize', 9);
current_y = current_y - spacing_q2_q3; % Уменьшенный отступ между Q2 и Q3

% --- Задание 3 ---
% Заголовок
current_y = current_y - h_title_q3;
uicontrol(panel_results, 'Style', 'text', 'String', 'Задание 3:', ...
    'Units', 'normalized', 'Position', [margin_left_panel, current_y, content_width_panel, h_title_q3], ...
    'HorizontalAlignment', 'left', 'FontWeight', 'bold', 'FontSize', 9);
current_y = current_y - gap_after_title_q3; % Отступ после заголовка Q3
% Количество теплоты
current_y = current_y - h_text_q3_line;
uicontrol(panel_results, 'Style', 'text', 'String', sprintf(' Количество отданной теплоты Q = %.2f [МДж]', Q_total_lost / 1e6), ...
    'Units', 'normalized', 'Position', [margin_left_panel, current_y, content_width_panel, h_text_q3_line], ...
    'HorizontalAlignment', 'left', 'FontSize', 9);
% current_y в этой точке будет примерно равен margin_bottom_panel.

% Инициализация GUI: Выбираем первый график в списке при запуске.
if ~isempty(plot_functions)
    listbox.Value = 1; % Выбираем первый элемент в списке.
    listbox_callback(listbox, [], plot_functions, ax_main); % Имитируем "нажатие" на первый элемент.
else
    % Если графиков нет, выводим сообщение об этом.
    cla(ax_main, 'reset');
    text(ax_main, 0.5, 0.5, 'Нет доступных графиков для отображения.', ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 12);
    axis(ax_main, 'off');
end


%% =========================================================================
%% --- 7. Запись результатов в Excel (сохраняем, как в исходном коде) ---
% Все расчетные данные и результаты сохраняются в один файл Excel,
% на разных листах, чтобы было удобно просматривать и анализировать.
% =========================================================================
fprintf('--- 7. Запись результатов в Excel ---\n');

try % Используем 'try-catch' для обработки возможных ошибок при записи в Excel.
    % Если файл с таким именем уже существует, удаляем его, чтобы избежать ошибок.
    if isfile(filename_excel) 
        delete(filename_excel);
        fprintf('Удален существующий файл Excel: %s\n', filename_excel);
    end

    % Записываем каждую таблицу данных на отдельный лист в Excel-файле.
    writetable(T_data_table, filename_excel, 'Sheet', 'ТемпературноеПоле');
    writetable(Q1_table, filename_excel, 'Sheet', 'Температура_при_t32мин');
    writetable(SlabTheta_table, filename_excel, 'Sheet', 'ТетаПластины');
    writetable(CylinderTheta_table, filename_excel, 'Sheet', 'ТетаЦилиндра');
    writetable(SlabRoots_table, filename_excel, 'Sheet', 'КорниПластины');
    writetable(CylinderRoots_table, filename_excel, 'Sheet', 'КорниЦилиндра');

    % Записываем все входные параметры задачи на отдельный лист.
    input_params_data = {
        't0', t0, 'Начальная равномерная температура тела, °C';
        'tf', tf, 'Температура окружающей среды, °C';
        'a', a, 'Коэффициент температуропроводности, м^2/с';
        'r0', r0, 'Характерный размер для цилиндра (радиус), м';
        'l0', l0, 'Характерный размер для пластины (половина толщины), м';
        'rho', rho, 'Плотность, кг/м^3';
        'cp', cp, 'Удельная теплоёмкость, Дж/(кг К)';
        'lambda', lambda, 'Теплопроводность, Вт/(м К)';
        'alpha_conv', alpha_conv, 'Коэффициент теплоотдачи, Вт/(м^2 К)';
        't1_minutes', t1_minutes, 'Момент времени t1, мин';
        'BiL', BiL, 'Число Био для пластины';
        'BiR', BiR, 'Число Био для цилиндра';
        'n_desired_roots', n_desired_roots, 'Количество искомых корней';
        'epsilon', epsilon, 'Малая величина для точности расчетов';
        'n_interp_points_akima', n_interp_points_akima, 'Точек для сплайна Акимы'
    };
    input_params_table = cell2table(input_params_data, 'VariableNames', {'Параметр', 'Значение', 'Описание'});
    writetable(input_params_table, filename_excel, 'Sheet', 'Входные_Параметры');

    % Добавляем результаты Задания 2 и Задания 3 на отдельный лист.
    results_q_data = {
        'm_exp', m_exp, '1/с', 'Экспериментальный темп охлаждения';
        'a_exp_calc', a_exp_calc, 'м^2/с', 'Экспериментальная температуропроводность';
        'Q_total_lost', Q_total_lost, 'Дж', 'Количество отданной теплоты до t1'
    };
    results_q_table = cell2table(results_q_data, 'VariableNames', {'Параметр', 'Значение', 'Единицы_измерения', 'Описание'});
    writetable(results_q_table, filename_excel, 'Sheet', 'Результаты_Q2_Q3', 'WriteMode', 'append');

    fprintf('Данные успешно записаны в Excel.\n');
catch ME_excel % Если произошла ошибка при записи в Excel, выводим сообщение об этом.
    warning('MyScript:ExcelWriteError', 'Ошибка записи данных в Excel: %s', ME_excel.message);
    disp('Убедитесь, что файл не открыт и у вас есть права на запись в данную директорию.');
end

toc; % Останавливаем таймер и выводим общее время выполнения скрипта.
disp('Расчеты, запись в файл и GUI завершены.');

%% =========================================================================
%% --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (ДОЛЖНЫ ИДТИ В КОНЦЕ ФАЙЛА СКРИПТА) ---
% Эти функции используются основным скриптом для выполнения конкретных задач,
% таких как поиск корней, расчет температур или построение графиков.
% Они объявляются в конце файла, чтобы основной скрипт мог их использовать.
% =========================================================================

% --- Вспомогательная функция для форматирования числа в виде "мантисса * 10^степень" ---
% Эта функция нужна для красивого вывода научных чисел в GUI.
function s = format_scientific_notation(num, precision)
    if isnan(num) || isinf(num)
        s = 'NaN'; % Если число не определено, выводим "NaN".
        return;
    end
    
    % Если число очень маленькое или ноль, чтобы избежать логарифма нуля,
    % выводим его как простой ноль.
    if abs(num) < eps % 'eps' - очень маленькое число, машинная точность.
        s = sprintf('0.0000'); 
        return;
    end

    exponent = floor(log10(abs(num))); % Определяем степень 10 (например, для 12345 это 4).
    mantissa = num / (10^exponent);    % Вычисляем мантиссу (число перед "* 10^", например, 1.2345).

    % Формируем строку с числом в желаемом виде.
    % Например, для 12345.67 с точностью 2: "1.23 * 10^4".
    % sprintf('%.*f * 10^%d', precision, mantissa, exponent)
    % %.*f - форматирует число с 'precision' знаками после запятой.
    % * 10^ - это просто текстовая строка.
    % %d - форматирует степень как целое число.
    s = sprintf('%.*f * 10^%d', precision, mantissa, exponent); % Используем стандартный символ * для умножения
end


% --- Callback функция для кнопки "Сохранить график" ---
% Эта функция выполняется, когда пользователь нажимает кнопку "Сохранить график".
function save_button_callback(listbox_handle, ax_handle, plot_names_ref)
    selected_idx = listbox_handle.Value; % Получаем индекс (номер) выбранного в списке графика.
    if selected_idx > 0 && selected_idx <= length(plot_names_ref) % Проверяем, что выбран допустимый график.
        current_plot_name = plot_names_ref{selected_idx}; % Получаем название выбранного графика.
        save_current_plot(ax_handle, current_plot_name);   % Вызываем функцию для сохранения графика.
    else
        msgbox('Сначала выберите график для сохранения.', 'Предупреждение', 'warn'); % Сообщение, если график не выбран.
    end
end

% --- Функция для сохранения текущего графика ---
% Эта функция берет текущий график из осей GUI и сохраняет его как PNG-файл.
function save_current_plot(ax, plot_name_str)
    output_folder = 'Plots'; % Папка, куда будут сохраняться графики.
    if ~exist(output_folder, 'dir') % Проверяем, существует ли папка.
        mkdir(output_folder);      % Если нет, создаем ее.
    end

    % Делаем имя файла "чистым" и совместимым с файловой системой.
    % `matlab.lang.makeValidName` очищает строку, удаляя недопустимые символы.
    filename_clean = matlab.lang.makeValidName(plot_name_str); 
    
    % Дополнительная постобработка для удаления лишних подчеркиваний.
    % 1. Заменяем все последовательности из двух и более подчеркиваний на одно.
    filename_clean = regexprep(filename_clean, '__+', '_');
    % 2. Удаляем все подчеркивания в конце строки.
    filename_clean = regexprep(filename_clean, '_+$', '');
    
    full_filename = fullfile(output_folder, [filename_clean, '.png']); % Полный путь к файлу.

    try % Пытаемся сохранить график.
        exportgraphics(ax, full_filename, 'Resolution', 300); % Сохраняем график с разрешением 300 dpi.
        fprintf('График "%s" сохранен как "%s".\n', plot_name_str, full_filename); % Сообщение об успехе.
    catch ME % Если произошла ошибка при сохранении, выводим предупреждение.
        warning('MyScript:PlotSaveError', 'Ошибка при сохранении графика "%s": %s', plot_name_str, ME.message);
    end
end

% --- Функция для применения общих настроек к осям графика ---
% Эта функция применяет стандартные настройки (шрифты, сетка, рамка) ко всем графикам,
% чтобы они выглядели единообразно и читабельно.
function apply_common_plot_settings(ax)
    set(ax, 'FontSize', 10, 'LineWidth', 0.8); % Размер шрифта и толщина линий осей.
    grid(ax, 'on'); % Включаем основную сетку.
    ax.Box = 'on';  % Включаем рамку вокруг графика.

    % Настройки прозрачности сетки.
    if isprop(ax, 'GridAlpha')
        ax.GridAlpha = 0.15;
    end
    if isprop(ax, 'MinorGridAlpha')
        ax.MinorGridAlpha = 0.05;
    end

    % Настройки для осей X, Y, Z (шрифты, толщина линий, малые деления сетки).
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
    
    % Настройки размеров шрифтов для заголовков и подписей осей.
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

% --- Функция для поиска положительных "корней" (собственных значений) характеристических уравнений ---
% Это универсальная функция, которая ищет, где заданная математическая функция
% пересекает ноль (т.е., находит корни). Она используется как для пластины, так и для цилиндра.
function roots_found = find_positive_roots(equation_func, num_roots_to_find, interval_generator_func, fzero_opts, small_epsilon, geometry_type)
    roots_found = zeros(num_roots_to_find, 1); % Массив для хранения найденных корней.
    num_found = 0;   % Счетчик найденных корней.
    k_interval_idx = 0; % Индекс для генерации интервалов.
    max_interval_generation_attempts = num_roots_to_find * 30; % Максимальное количество попыток найти интервал.
    current_attempts = 0; % Текущее количество попыток.
    last_root_val = -Inf; % Значение последнего найденного корня, чтобы корни шли по возрастанию.

    % Цикл продолжается, пока не найдено достаточно корней или не исчерпаны попытки.
    while num_found < num_roots_to_find && current_attempts < max_interval_generation_attempts
        interval = interval_generator_func(k_interval_idx); % Генерируем "интервал" - диапазон, где предположительно находится корень.
        current_attempts = current_attempts + 1;
        k_interval_idx = k_interval_idx + 1;

        % Пропускаем некорректные или уже пройденные интервалы.
        if isempty(interval) || interval(1) >= interval(2) || interval(1) < -small_epsilon 
            continue;
        end
        
        if (interval(2) - interval(1)) < small_epsilon || interval(2) < last_root_val + small_epsilon 
             continue;
        end
        check_a = max(interval(1), last_root_val + small_epsilon); % Убеждаемся, что интервал начинается после последнего корня.
        check_b = interval(2);

        if check_a >= check_b 
            continue;
        end

        val_a = equation_func(check_a); % Вычисляем значение функции на нижней границе интервала.
        val_b = equation_func(check_b); % Вычисляем значение функции на верхней границе интервала.

        if isnan(val_a) || isnan(val_b) || isinf(val_a) || isinf(val_b) % Пропускаем, если значения некорректны.
             continue;
        end

        % Если значения функции на границах интервала имеют разные знаки (val_a * val_b <= 0),
        % это означает, что корень находится внутри этого интервала.
        if val_a * val_b <= 0 
            try % Пытаемся найти корень с помощью 'fzero'.
                root_candidate = fzero(equation_func, [check_a, check_b], fzero_opts);

                % Если корень найден, он реальный и больше предыдущего, добавляем его в список.
                if ~isempty(root_candidate) && isreal(root_candidate) && root_candidate > last_root_val + small_epsilon/2 
                    is_too_close_to_existing = false;
                    % Проверяем, не слишком ли близок найденный корень к уже существующим (чтобы избежать дубликатов).
                    if num_found > 0
                        if any(abs(roots_found(1:num_found) - root_candidate) < small_epsilon*10) 
                            is_too_close_to_existing = true;
                        end
                    end

                    if ~is_too_close_to_existing
                        num_found = num_found + 1;
                        roots_found(num_found) = root_candidate;
                        last_root_val = root_candidate; % Обновляем значение последнего корня.
                    end
                end
            catch ME
                 % Отладочная информация об ошибках fzero (закомментирована, чтобы не засорять вывод).
            end
        end

        if num_found >= num_roots_to_find
            break; % Если нашли достаточно корней, выходим из цикла.
        end
    end

    % Если не удалось найти все требуемые корни, выводим предупреждение.
    if num_found < num_roots_to_find
        warning('find_positive_roots (%s): Найдено %d из %d требуемых положительных корней. Рассмотрите возможность увеличения n_desired_roots или проверки генератора интервалов.', geometry_type, num_found, num_roots_to_find);
    end
    roots_found = sort(roots_found(1:num_found)); % Сортируем найденные корни по возрастанию.
end

% --- Генератор интервалов для ПЛАСТИНЫ (для Bi > 0) ---
% Эта функция предлагает интервалы, в которых, как ожидается, находятся корни
% характеристического уравнения для пластины.
function interval = slab_interval_generator(k_idx, ep)
    lower = k_idx * pi + ep;     % Нижняя граница интервала.
    upper = (k_idx + 0.5) * pi - ep; % Верхняя граница интервала.
    % Корни для пластины обычно располагаются между k*pi и (k+0.5)*pi.

    if k_idx == 0 && lower < ep % Для первого интервала убеждаемся, что нижняя граница не слишком мала.
        lower = ep; 
    end

    if upper <= lower % Если интервал некорректен, возвращаем пустой.
        interval = [];
    else
        interval = [lower, upper]; % Возвращаем корректный интервал.
    end
end

% --- Вспомогательная функция для получения нулей функции Бесселя первого рода J0(x) ---
% Нули функции Бесселя J0(x) используются для определения интервалов поиска корней
% для цилиндра, когда BiR > 0.
function zeros_J0 = get_J0_zeros(num_zeros_needed, fzero_opts_loc, ep)
    % 'persistent' означает, что эти переменные сохраняют свое значение между вызовами функции.
    % Это нужно, чтобы не пересчитывать нули J0 каждый раз, если они уже были найдены.
    persistent cached_J0_zeros num_cached_J0_zeros_val; 
    
    % Если кэш пуст или нужно больше нулей, чем есть в кэше, начинаем поиск.
    if isempty(cached_J0_zeros) || num_cached_J0_zeros_val < num_zeros_needed
        new_zeros_J0 = zeros(num_zeros_needed, 1);
        current_num_in_cache = 0;
        % Копируем уже найденные нули из кэша.
        if ~isempty(cached_J0_zeros) && num_cached_J0_zeros_val > 0
            len_to_copy = min(num_cached_J0_zeros_val, num_zeros_needed);
            new_zeros_J0(1:len_to_copy) = cached_J0_zeros(1:len_to_copy);
            current_num_in_cache = len_to_copy;
        end

        last_found_zero = 0; % Значение последнего найденного нуля J0.
        if current_num_in_cache > 0
            last_found_zero = new_zeros_J0(current_num_in_cache);
        end

        % Цикл для поиска каждого последующего нуля J0.
        for i_zero = (current_num_in_cache + 1) : num_zeros_needed
            lower_search_bnd = 0; upper_search_bnd = 0;
            % Определяем начальные интервалы для поиска нулей J0.
            if i_zero == 1
                lower_search_bnd = 2.0; upper_search_bnd = 3.0; % Приблизительный интервал для первого нуля J0.
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

            val_lower = besselj(0, lower_search_bnd); % Значение J0 в нижней границе.
            val_upper = besselj(0, upper_search_bnd); % Значение J0 в верхней границе.

            search_attempts = 0; max_search_attempts = 5;
            % Если функция J0 не меняет знак в интервале, пытаемся расширить интервал.
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
                 error('get_J0_zeros: Не удалось найти интервал, содержащий J0 #%d, после %d попыток. Последний интервал: [%.4f, %.4f], f(low)=%.2e, f(high)=%.2e', ...
                    i_zero, max_search_attempts, lower_search_bnd, upper_search_bnd, val_lower, val_upper);
            end

            try % Ищем ноль J0 с помощью 'fzero'.
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
        cached_J0_zeros = new_zeros_J0; num_cached_J0_zeros_val = num_zeros_needed; % Сохраняем в кэш.
    end
    zeros_J0 = cached_J0_zeros(1:min(num_zeros_needed, num_cached_J0_zeros_val)); % Возвращаем нули J0.
end

% --- Генератор интервалов для ЦИЛИНДРА (корни уравнения x*J1(x) - Bi*J0(x) = 0) ---
% Эта функция предлагает интервалы, в которых, как ожидается, находятся корни
% характеристического уравнения для цилиндра, используя нули функции J0(x).
function interval = get_cylinder_interval(k_idx_root, zeros_J0_arr_loc, ep_loc)
    if k_idx_root == 0 % Для первого корня (вне интервалов между нулями J0).
        lower_bnd_interval = ep_loc; 
        if isempty(zeros_J0_arr_loc) || length(zeros_J0_arr_loc) < 1
            error('get_cylinder_interval: zeros_J0_arr_loc пуст или не содержит достаточного количества нулей для первого интервала.');
        end
        upper_bnd_interval = zeros_J0_arr_loc(1) - ep_loc; % Первый корень для цилиндра находится до первого нуля J0.
    else % Для остальных корней, которые находятся между последовательными нулями J0.
        if (k_idx_root+1) > length(zeros_J0_arr_loc) || k_idx_root > length(zeros_J0_arr_loc)
            warning('get_cylinder_interval: Недостаточно нулей J0 для генерации интервала #%d. Доступно %d нулей.', k_idx_root, length(zeros_J0_arr_loc));
            interval = []; return; 
        end
        lower_bnd_interval = zeros_J0_arr_loc(k_idx_root) + ep_loc;
        upper_bnd_interval = zeros_J0_arr_loc(k_idx_root+1) - ep_loc;
    end

    if upper_bnd_interval <= lower_bnd_interval % Если интервал некорректен, возвращаем пустой.
        interval = []; 
    else
        interval = [lower_bnd_interval, upper_bnd_interval]; % Возвращаем корректный интервал.
    end
end

% --- Функция расчета безразмерной температуры Тета для ПЛАСТИНЫ (для конкретной точки) ---
% Эта функция вычисляет безразмерную температуру Theta(x,t) в любой точке 'x'
% и в любой момент времени 't' для бесконечной пластины.
% Используется ряд Фурье с найденными корнями характеристического уравнения.
function theta_sums = calculate_theta_slab(BiL_val, FoL_val, L_arr, slab_roots, ep)
    theta_sums = zeros(length(L_arr), 1); % Массив для результатов.

    if abs(BiL_val) < ep % Особый случай, когда Bi близко к 0. Температура равномерна и не меняется.
        theta_sums(:) = 1.0; % Theta = 1, что означает T = T0 (температура не меняется).
        return; 
    end
    
    if isempty(slab_roots) % Если корни не найдены, выводим предупреждение.
        warning('calculate_theta_slab: Массив корней для пластины пуст. Theta будет NaN.');
        theta_sums(:) = NaN; 
        return;
    end

    for i_L = 1:length(L_arr) % Проходим по всем точкам по координате L.
        L_current = L_arr(i_L); % Текущая безразмерная координата.
        current_sum_L_theta = 0; % Сумма для текущей точки.

        for r_idx = 1:length(slab_roots) % Проходим по всем найденным корням.
            r = slab_roots(r_idx); % Текущий корень.
            if isnan(r) || r < ep/2 % Пропускаем некорректные корни.
                continue; 
            end

            % Вычисляем коэффициент ряда (An) для пластины.
            % Это "вес" каждого члена в сумме.
            term_coeff_numerator = 2*BiL_val*sin(r); % Числитель коэффициента.
            term_coeff_denominator = r*(r^2 + BiL_val^2 + BiL_val); % Знаменатель.
            term_coeff = 0;

            if abs(term_coeff_denominator) > ep % Если знаменатель не ноль, вычисляем коэффициент.
                term_coeff = term_coeff_numerator / term_coeff_denominator;
            elseif abs(term_coeff_numerator) < ep % Если и числитель, и знаменатель малы, коэффициент 0.
                term_coeff = 0; 
            else % Если знаменатель ноль, но числитель нет (делим на ноль), выводим предупреждение.
                warning('Пластина: Потенциальное Inf/NaN для коэффициента корня r=%f (числитель=%.2e, знаменатель=%.2e), BiL=%f. Коэффициент установлен в 0.', r, term_coeff_numerator, term_coeff_denominator, BiL_val);
                term_coeff = 0;
            end
            
            % Вычисляем текущий член ряда: Коэффициент * косинус (положение) * экспонента (время).
            theta_component_val = term_coeff * cos(r*L_current) * exp(-r^2*FoL_val);
            current_sum_L_theta = current_sum_L_theta + theta_component_val; % Добавляем к общей сумме.
        end
        theta_sums(i_L) = current_sum_L_theta; % Сохраняем результат для текущей точки.
    end
    % Ограничиваем значение Theta между 0 и 1, так как температура не может быть
    % ниже температуры среды (0) или выше начальной температуры (1).
    theta_sums(theta_sums < 0) = 0; 
    theta_sums(theta_sums > 1) = 1; 
end

% --- Функция расчета безразмерной температуры Тета для ЦИЛИНДРА (для конкретной точки) ---
% Аналогично для бесконечного цилиндра, но используются функции Бесселя.
function theta_sums = calculate_theta_cylinder(BiR_val, FoR_val, R_arr, cyl_roots, ep)
    theta_sums = zeros(length(R_arr), 1); 

    if abs(BiR_val) < ep % Особый случай, когда Bi близко к 0.
        theta_sums(:) = 1.0; 
        return; 
    end

    if isempty(cyl_roots) % Если корни не найдены.
        warning('calculate_theta_cylinder: Массив корней для цилиндра пуст. Theta будет NaN.');
        theta_sums(:) = NaN; 
        return;
    end

    for i_R = 1:length(R_arr) % Проходим по всем точкам по координате R.
        R_current = R_arr(i_R); % Текущая безразмерная координата.
        current_sum_R_theta = 0; % Сумма для текущей точки.

        for r_idx = 1:length(cyl_roots) % Проходим по всем найденным корням.
            r_val = cyl_roots(r_idx); % Текущий корень.
            if isnan(r_val) || r_val < ep/2 
                continue;
            end

            J0_r = besselj(0,r_val); % Вычисляем функцию Бесселя J0(x) в корне r_val.
            J1_r = besselj(1,r_val); % Вычисляем функцию Бесселя J1(x) в корне r_val.

            % Вычисляем коэффициент ряда (Cn) для цилиндра.
            term_coeff_numerator = 2*BiR_val*J1_r; 
            term_coeff_denominator = r_val * (J0_r^2 + J1_r^2); % Знаменатель коэффициента.
            term_coeff = 0;

            if abs(r_val) < ep % Если корень близок к нулю.
                 term_coeff = 0; 
            elseif abs(term_coeff_denominator) > ep % Если знаменатель не ноль.
                term_coeff = term_coeff_numerator / term_coeff_denominator;
            elseif abs(term_coeff_numerator) < ep % Если и числитель, и знаменатель малы.
                term_coeff = 0; 
            else % Если знаменатель ноль, но числитель нет.
                warning('Цилиндр: Потенциальное Inf/NaN для коэффициента корня r=%f (числитель=%.2e, знаменатель=%.2e), BiR=%f. Коэффициент установлен в 0.', r_val, term_coeff_numerator, term_coeff_denominator, BiR_val);
                term_coeff = 0;
            end
            
            % Вычисляем текущий член ряда: Коэффициент * J0(положение) * экспонента (время).
            theta_component_val = term_coeff * besselj(0,r_val*R_current) * exp(-r_val^2*FoR_val);
            current_sum_R_theta = current_sum_R_theta + theta_component_val; % Добавляем к общей сумме.
        end
        theta_sums(i_R) = current_sum_R_theta; % Сохраняем результат для текущей точки.
    end
    theta_sums(theta_sums < 0) = 0;
    theta_sums(theta_sums > 1) = 1;
end

% --- Функция расчета безразмерной СРЕДНЕЙ температуры Тета для ПЛАСТИНЫ (для Задания 3) ---
% Эта функция вычисляет среднюю температуру по всему объему "бесконечной пластины".
% Это нужно для расчета общего количества отданного тепла.
function theta_avg = calculate_theta_avg_slab(BiL_val, FoL_val, slab_roots, ep)
    theta_avg = 0;
    if abs(BiL_val) < ep % Особый случай, когда Bi близко к 0.
        theta_avg = 1.0; 
        return; 
    end
    if isempty(slab_roots) % Если корни не найдены.
        warning('calculate_theta_avg_slab: Массив корней для пластины пуст. Theta_avg будет NaN.');
        theta_avg = NaN; 
        return;
    end

    for r_idx = 1:length(slab_roots) % Проходим по всем найденным корням.
        r = slab_roots(r_idx); % Текущий корень.
        if isnan(r) || r < ep/2 
            continue; 
        end
        
        % Коэффициент ряда для средней безразмерной температуры пластины.
        % Этот коэффициент отличается от коэффициента для температуры в точке.
        An_avg_coeff = (2 * BiL_val) / (r^2 + BiL_val^2 + BiL_val);
        
        % Член ряда для средней температуры.
        if abs(r) > ep
            term_avg = An_avg_coeff * (sin(r)/r) * exp(-r^2 * FoL_val);
        else 
            term_avg = 0; 
        end
        theta_avg = theta_avg + term_avg; % Добавляем к общей сумме.
    end
    theta_avg = max(0, min(1, theta_avg)); % Ограничиваем значение Theta_avg между 0 и 1.
end


% --- Функция расчета безразмерной СРЕДНЕЙ температуры Тета для ЦИЛИНДРА (для Задания 3) ---
% Аналогично для средней температуры по объему "бесконечного цилиндра".
function theta_avg = calculate_theta_avg_cylinder(BiR_val, FoR_val, cyl_roots, ep)
    theta_avg = 0;
    if abs(BiR_val) < ep % Особый случай, когда Bi близко к 0.
        theta_avg = 1.0; 
        return; 
    end
    if isempty(cyl_roots) % Если корни не найдены.
        warning('calculate_theta_avg_cylinder: Массив корней для цилиндра пуст. Theta_avg будет NaN.');
        theta_avg = NaN; 
        return;
    end

    for r_idx = 1:length(cyl_roots) % Проходим по всем найденным корням.
        r_val = cyl_roots(r_idx); % Текущий корень.
        if isnan(r_val) || r_val < ep/2 
            continue;
        end

        % Коэффициент ряда для средней безразмерной температуры цилиндра.
        % Эта формула из стандартных источников (например, Cengel, Incropera)
        % для объемно-средней температуры цилиндра.
        term_avg_coeff = (2 * BiR_val^2 * besselj(1, r_val)^2) / (r_val^2 * (BiR_val^2 + r_val^2));
        
        % Член ряда для средней температуры.
        term_avg = term_avg_coeff * exp(-r_val^2 * FoR_val);
        theta_avg = theta_avg + term_avg; % Добавляем к общей сумме.
    end
    theta_avg = max(0, min(1, theta_avg)); % Ограничиваем значение Theta_avg между 0 и 1.
end


% =========================================================================
% --- Callback функция для Listbox (для GUI) ---
% Эта функция вызывается каждый раз, когда пользователь выбирает другой
% график в списке. Она очищает текущие оси и рисует новый выбранный график.
% =========================================================================
function listbox_callback(src, ~, plot_functions_ref, ax_handle)
    cla(ax_handle, 'reset'); % Очищаем текущие оси графика.
    hold(ax_handle, 'off');  % Выключаем режим "удержания" графика (чтобы новый график заменил старый).
    % Отключаем интерактивность по умолчанию (панорамирование, масштабирование),
    % чтобы график можно было нарисовать полностью перед тем, как пользователь начнет с ним взаимодействовать.
    if verLessThan('matlab', '9.1') % Проверяем версию MATLAB (функция доступна с R2016b).
        % Для старых версий может не быть disableDefaultInteractivity
    else
        disableDefaultInteractivity(ax_handle); 
    end

    selected_idx = src.Value; % Получаем индекс выбранного графика.

    if selected_idx > 0 && selected_idx <= length(plot_functions_ref) % Проверяем, что индекс корректен.
        plot_functions_ref{selected_idx}(ax_handle); % Вызываем функцию, которая рисует выбранный график.
        apply_common_plot_settings(ax_handle);       % Применяем стандартные настройки к новому графику.
        % Включаем интерактивность обратно после отрисовки.
        if verLessThan('matlab', '9.1') 
            % Для старых версий может не быть enableDefaultInteractivity
        else
            enableDefaultInteractivity(ax_handle); 
        end
        drawnow; % Обновляем графическое окно.
    end
end


% =========================================================================
% --- Вспомогательные функции для построения отдельных графиков (для GUI) ---
% Эти функции содержат код для рисования различных типов графиков.
% =========================================================================

% --- Профиль T(R) при фиксированном L (2D график) ---
% Показывает, как температура меняется по радиусу, если мы находимся на определенной глубине L.
function plot_profile_R_fixed_L(ax, R_coords_data, T_data, L_val, t_val, n_interp)
    T_data = T_data(:); % Преобразуем данные в вектор-столбец.
    R_coords_data = R_coords_data(:); 
    
    % Рисуем расчетные точки (кружочки).
    plot(ax, R_coords_data, T_data, 'o', 'MarkerFaceColor', 'b', 'DisplayName', 'Расчетные точки');
    hold(ax, 'on'); % Удерживаем график, чтобы можно было добавить еще линии.
    
    % Если точек достаточно, рисуем интерполированную кривую для гладкости.
    if length(R_coords_data) >= 2 
        R_interp = linspace(min(R_coords_data), max(R_coords_data), n_interp); % Создаем больше точек для интерполяции.
        T_interp = interp1(R_coords_data, T_data, R_interp, 'makima'); % Выполняем интерполяцию.
        plot(ax, R_interp, T_interp, '-', 'Color', [0 0 0.7], 'LineWidth', 1.5, 'DisplayName', 'Интерполированная кривая'); % Рисуем интерполированную линию.
    end
    
    xlabel(ax, 'Безразмерная координата R'); % Подпись оси X.
    ylabel(ax, 'Температура, °C');           % Подпись оси Y.
    title(ax, sprintf('Профиль T(R) при L=%.2f, t=%.1f мин', L_val, t_val)); % Заголовок графика.
    set(ax, 'XLimSpec', 'tight'); % Автоматически подстраиваем границы оси X.
    legend(ax, 'show', 'Location', 'northeast'); % ИЗМЕНЕНО: Легенда в правом верхнем углу
    hold(ax, 'off'); % Отключаем режим удержания.
    view(ax, 2); % 2D вид (сверху).
end

% --- Профиль T(L) при фиксированном R (2D график) ---
% Показывает, как температура меняется по длине, если мы находимся на определенном радиусе R.
function plot_profile_L_fixed_R(ax, L_coords_data, T_data, R_val, t_val, n_interp)
    T_data = T_data(:);
    L_coords_data = L_coords_data(:);

    plot(ax, L_coords_data, T_data, 'o', 'MarkerFaceColor', 'r', 'DisplayName', 'Расчетные точки');
    hold(ax, 'on');

    if length(L_coords_data) >= 2
        L_interp = linspace(min(L_coords_data), max(L_coords_data), n_interp);
        T_interp = interp1(L_coords_data, T_data, L_interp, 'makima');
        plot(ax, L_interp, T_interp, '-', 'Color', [0.7 0 0], 'LineWidth', 1.5, 'DisplayName', 'Интерполированная кривая');
    end
        
    xlabel(ax, 'Безразмерная координата L');
    ylabel(ax, 'Температура, °C');
    title(ax, sprintf('Профиль T(L) при R=%.2f, t=%.1f мин', R_val, t_val));
    set(ax, 'XLimSpec', 'tight');
    legend(ax, 'show', 'Location', 'northeast'); % ИЗМЕНЕНО: Легенда в правом верхнем углу
    hold(ax, 'off');
    view(ax, 2); 
end

% --- 3D Профиль T(R,L) ---
% Показывает распределение температуры по радиусу и длине в виде 3D-поверхности.
function plot_surf_R_L(ax, L_coords_data, R_coords_data, T_data_2D, t_val)
    [L_mesh, R_mesh] = meshgrid(L_coords_data, R_coords_data); % Создаем сетку из координат L и R.
    surf(ax, L_mesh, R_mesh, T_data_2D, 'EdgeColor','none', 'FaceAlpha',0.8); % Рисуем 3D-поверхность.
                                                                                % 'EdgeColor','none' - без линий сетки на поверхности.
                                                                                % 'FaceAlpha',0.8 - небольшая прозрачность.
    xlabel(ax, 'Безразмерная координата L');
    ylabel(ax, 'Безразмерная координата R');
    zlabel(ax, 'Температура, °C');
    title(ax, sprintf('Температура T(R,L) при t=%.1f мин', t_val));
    colorbar(ax); % Добавляем цветовую шкалу, показывающую значения температуры.
    view(ax, -30, 30); % Устанавливаем угол обзора 3D-графика.
    axis tight; % Автоматически подстраиваем границы осей.
end

% --- Температура в одной точке со временем (2D график) ---
% Показывает, как температура меняется со временем в одной конкретной точке (R,L).
function plot_temp_over_time_single(ax, t_array_data, T_data_1D, R_val, L_val, n_interp)
    T_data_1D_sq = squeeze(T_data_1D); % Убираем лишние размерности.
    plot(ax, t_array_data, T_data_1D_sq, 'o', 'MarkerFaceColor','b', 'DisplayName', 'Расчетные точки');
    hold(ax, 'on');

    if length(t_array_data) >= 2
        t_interp = linspace(min(t_array_data), max(t_array_data), n_interp);
        T_interp = interp1(t_array_data, T_data_1D_sq, t_interp, 'makima');
        plot(ax, t_interp, T_interp, '-', 'Color', [0 0 0.7], 'LineWidth', 1.5, 'DisplayName', 'Интерполированная кривая');
    end
    
    xlabel(ax, 'Время, мин');
    ylabel(ax, 'Температура, °C');
    title(ax, sprintf('Изменение температуры для R=%.2f, L=%.2f', R_val, L_val));
    legend(ax, 'show', 'Location', 'northeast'); % ИЗМЕНЕНО: Легенда в правом верхнем углу
    hold(ax, 'off');
    view(ax, 2); 
    axis tight;
end

% --- График ln(Theta) от времени в центре для регулярного режима ---
% Этот график используется для определения темпа охлаждения 'm' в Задании 2.
% В регулярном режиме точки на этом графике должны выстраиваться в прямую линию.
function plot_ln_theta_center_over_time(ax, t_seconds_data, T_center_data, t0_val, tf_val, ep)
    % Проверяем t0 и tf, чтобы избежать деления на ноль, если t0 == tf
    if abs(t0_val - tf_val) < ep
        ln_theta_center = zeros(size(T_center_data)); % Если t0 = tf, Theta = 1, ln(Theta) = 0
    else
        theta_center = (T_center_data - tf_val) / (t0_val - tf_val); % Вычисляем безразмерную температуру.
        theta_center(theta_center <= ep) = ep; % Избегаем логарифма нуля или отрицательного числа.
        ln_theta_center = log(theta_center); % Берем натуральный логарифм.
    end

    plot(ax, t_seconds_data/60, ln_theta_center, 'o', 'MarkerFaceColor','g', 'DisplayName', 'ln(Theta) точки');
    hold(ax, 'on');

    % Подготовка данных для линейной регрессии.
    min_points_for_regression = 5;
    if length(t_seconds_data) >= min_points_for_regression
        num_points_to_fit = min(10, length(t_seconds_data) - 1); 
        if num_points_to_fit >= 2
            t_regime_fit = t_seconds_data(end - num_points_to_fit + 1 : end);
            ln_theta_center_regime_fit = ln_theta_center(end - num_points_to_fit + 1 : end);
            
            % Отфильтровываем недействительные точки (Inf или NaN).
            valid_indices_fit = ~isinf(ln_theta_center_regime_fit) & ~isnan(ln_theta_center_regime_fit);
            if sum(valid_indices_fit) >= 2
                t_regime_fit_valid = t_regime_fit(valid_indices_fit);
                ln_theta_center_regime_fit_valid = ln_theta_center_regime_fit(valid_indices_fit);
                P = polyfit(t_regime_fit_valid, ln_theta_center_regime_fit_valid, 1); % Выполняем линейную регрессию.
                
                % Рисуем линию регрессии.
                % Значение 'm' теперь отображается только в легенде и в корректном формате.
                plot(ax, t_seconds_data/60, polyval(P, t_seconds_data), 'r-', 'LineWidth', 1.5, ...
                    'DisplayName', sprintf('Линейная регрессия (m = %s [1/с])', format_scientific_notation(-P(1), 4))); 
            else
                 warning('Недостаточно допустимых точек для построения линии регрессии на графике ln(Theta).');
            end
        end
    end

    xlabel(ax, 'Время, мин');
    ylabel(ax, 'ln((T-T_ж)/(T_0-T_ж))'); % Подпись оси Y в соответствии с формулой.
    title(ax, 'ln(Безразмерной температуры) в центре цилиндра от времени');
    legend(ax, 'show', 'Location', 'northeast'); % ИЗМЕНЕНО: Легенда в правом верхнем углу
    hold(ax, 'off');
    view(ax, 2); 
    axis tight;
end


% --- 3D график T(R,t) для фиксированного L ---
% Показывает, как температурный профиль по радиусу меняется со временем,
% для определенной глубины по длине цилиндра.
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

% --- 3D график T(L,t) для фиксированного R ---
% Показывает, как температурный профиль по длине меняется со временем,
% для определенного радиуса.
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