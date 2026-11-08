#import "title_page_lab.typ": title_page
#import "listings.typ": setup-listings, listing, tbl-caption

#show: setup-listings
#set page(
  paper: "a4",
  margin: (top: 2cm, bottom: 2cm, left: 3cm, right: 1.5cm),
  numbering: none,
  number-align: center,
)
#set text(font: "Times New Roman", size: 14pt, lang: "ru")
#set par(
  leading: 0.65em,
  spacing: 0.65em,
  justify: true,
  first-line-indent: (amount: 1.25cm, all: true),
)
#set heading(numbering: none)
#show heading: set text(size: 14pt, weight: "bold")
#show heading: it => block(above: 1.5em, below: 1.5em, sticky: true)[
  #set par(first-line-indent: 0pt)
  #it.body
]
#show outline.entry.where(level: 1): strong

#let result-table(num, title, columns, ..cells) = block(
  width: 100%, breakable: false, above: 1em, below: 1em,
)[
  #set par(first-line-indent: 0pt, justify: false)
  #tbl-caption(num, title)
  #block(width: 100%, above: 0pt, below: 0pt)[
    #align(center, table(
      columns: columns,
      fill: (_, row) => if row == 0 { luma(220) } else { white },
      stroke: 0.5pt,
      inset: 6pt,
      align: (column, _) => if column == 0 { left + horizon } else { center + horizon },
      ..cells.pos(),
    ))
  ]
]

#let chart(path, number, title, width: 88%) = block(
  width: 100%, breakable: false, above: 0.8em, below: 0.8em,
)[
  #set par(first-line-indent: 0pt, justify: false)
  #block(width: 100%, above: 0pt, below: 0pt)[
    #align(center, image(path, width: width))
  ]
  #block(width: 100%, above: 0.3em, below: 0pt)[
    #align(center, strong[Рисунок #number — #title])
  ]
]

#title_page(
  4, "Информационные технологии анализа и обработки данных", 7,
  проверил: (должность: [доктор технических наук, профессор], фио: [В. Г. Суфиянов]),
)
#pagebreak()
#[
  #set par(first-line-indent: 0pt, justify: false)
  #block(width: 100%, below: 0.65em)[
    #align(center, strong[СОДЕРЖАНИЕ])
  ]
  #outline(title: none, indent: auto, depth: 2)
]
#pagebreak()
#set page(numbering: "1")
#counter(page).update(3)

= Введение

Цель работы — исследовать аномальные наблюдения, провести очистку и оценить восстановление пропущенных значений средствами R. Выполнены одномерная, двумерная и многомерная диагностика выбросов, анализ категориальных сочетаний и преобразование Бокса–Кокса по индивидуальному заданию.

Вариант 7 задаёт набор Stack Overflow Developer Survey 2011–2022. Копия CSV за 2022 год из #link("https://www.kaggle.com/datasets/yasirabdaali/stack-overflow-developer-survey-20112022")[архива Kaggle] содержит посторонние двоичные данные после анкеты с идентификатором около 68 810. Для расчётов взят #link("https://survey.stackoverflow.co/2022/")[оригинальный выпуск] с 73~268 анкетами.

= Задание 1. Подготовка данных

Выбраны 9~133 профессиональных разработчика из США с полной занятостью. Ограничение одной страной уменьшает неоднородность рынков труда. Годовая компенсация измеряется в долларах США, признаки опыта — в годах.

Значения стажа «менее 1 года» и «более 50 лет» заменены на 0,5 и 51 соответственно. Для годовой компенсации допустимы значения от 1 доллара, для стажа программирования — от 0 до 51 года, для опыта работы — от 0 до 100 лет. Нечисловые значения и записи вне диапазона переводятся в пропуски. Возраст с ответом «предпочитаю не сообщать» учитывается как недоступный.

#listing("1.1", "Преобразование и проверка числовых значений")[
  ```r
  parse_number <- function(values) {
    values[values %in% "Less than 1 year"] <- "0.5"
    values[values %in% "More than 50 years"] <- "51"
    suppressWarnings(as.numeric(values))
  }
  invalid_number <- function(values, lower, upper = Inf) {
    number <- parse_number(values)
    !is.na(values) &
      (!is.finite(number) | number < lower | number > upper)
  }
  ```
]

#result-table(
  "1.1", "Пропуски и нарушения числовых ограничений", (2.8fr, 1fr, 1.4fr),
  table.header([*Признак*], [*Пропуски*], [*Нарушения*]),
  [Годовая компенсация], [1~556], [0],
  [Общий стаж программирования], [28], [0],
  [Профессиональный стаж программирования], [304], [0],
  [Опыт работы], [2~829], [0],
)

= Задание 2. Диагностика числовых выбросов

Для одномерной проверки использованы логарифмы 7~577 положительных компенсаций. Правило Тьюки отметило 572 наблюдения. Критерий Граббса из `outliers` выделил нижнее крайнее значение, соответствующее 36 долларам в год. Получено $G approx "8,871"$ и $p < "2,2" dot 10^(-16)$. Интерпретация критерия зависит от приближения распределения логарифмов нормальным.

#listing("2.1", "Критерий Граббса и правило Тьюки")[
  ```r
  income <- prepared |>
    filter(!is.na(ConvertedCompYearly)) |>
    transmute(
      ResponseId, income = ConvertedCompYearly,
      log_income = log(income)
    )
  grubbs_result <- outliers::grubbs.test(income$log_income)
  tukey_values <- boxplot.stats(income$log_income)$out
  income <- mutate(
    income, box_flag = log_income %in% tukey_values
  )
  ```
]

#block(breakable: false, above: 0.65em, below: 0pt)[
Для двумерной проверки составлен вектор $bold(x)_i$ из профессионального стажа и логарифма компенсации. Квадрат расстояния Махаланобиса вычислялся относительно вектора средних $bold(mu)$ и выборочной ковариационной матрицы $bold(S)$:
$ d_i^2 = (bold(x)_i - bold(mu))^T bold(S)^(-1) (bold(x)_i - bold(mu)). $
]
Кандидаты выделены по правилу $d_i^2 > chi_(2; "0,99")^2$ при уровне значимости $alpha = "0,01"$. Диаграмма `mahalanobis_method` из `OutliersLearn` дополняет числовую проверку.

#listing("2.2", "Отметки по расстоянию Махаланобиса")[
  ```r
  pair_values <- as.data.frame(
    select(paired, experience, log_income)
  )
  alpha <- 0.01
  paired$d2 <- mahalanobis(
    pair_values, colMeans(pair_values), cov(pair_values)
  )
  paired$flagged <- paired$d2 > qchisq(1 - alpha, df = 2L)
  ```
]

В многомерный анализ добавлен общий стаж программирования. Пакет `HDoutliers` ищет необычно большие расстояния до ближайшего соседа по верхнему хвосту распределения этих расстояний.

#listing("2.3", "Поиск многомерных выбросов по ближайшим соседям")[
  ```r
  numeric_values <- as.matrix(select(multivariate, -ResponseId))
  hd_positions <- HDoutliers::HDoutliers(numeric_values)
  hd_cases <- multivariate[hd_positions, , drop = FALSE]
  ```
]

#result-table(
  "2.1", "Число наблюдений по методам диагностики", (2.65fr, 1fr, 1fr),
  table.header([*Метод*], [*Полных строк*], [*Отмечено*]),
  [Правило Тьюки], [7~577], [572],
  [Расстояние Махаланобиса], [7~558], [364],
  [Ближайшие соседи в `HDoutliers`], [7~554], [3~248],
)

Доля отметок `HDoutliers` составляет 43,0~%. Методы работают с разными признаками и порогами. Высокая доля отметок в этом наборе показывает, что применение единого правила удаления сильно изменило бы состав выборки. Отмеченные значения сохранены как кандидаты для предметной проверки.

= Задание 3. Гиперграфовый анализ

Вершинами гиперграфа служат анкеты с известными возрастом, образованием и форматом работы. Анкеты с одинаковыми значениями одного или двух фиксированных признаков образуют гиперребро. Для выпуска 2022 года сохраняются рёбра с поддержкой не менее 50 анкет.

По формулам (4.4)–(4.6) методических указаний внутри каждого ребра подсчитываются частоты $S(a)$ значений свободного признака. При $k$ наблюдаемых категориях вычисляются средняя частота, разброс и стандартизованное отклонение:
$ mu = (sum_a S(a)) / k, quad
  sigma = sqrt(sum_a (S(a) - mu)^2) / k, quad
  D(a) = (S(a) - mu) / sigma. $
При нулевом разбросе признак в этом ребре пропускается. Редкими считаются значения с $D(a) < -2$.

#listing("3.1", "Проверка редкости в одном гиперребре")[
  ```r
  counts <- table(observed)
  average <- mean(counts)
  spread <- sqrt(sum((counts - average)^2)) / length(counts)
  if (spread == 0) {
    next
  }
  deviation <- (as.numeric(counts[observed]) - average) / spread
  selected <- deviation < threshold
  ```
]

В контрольном примере из методических указаний получено шесть рёбер. Для записи № 3 в ребре молодых владельцев автомобилей у категории «седан» вычислено $D = -sqrt(2)$.

#listing("3.2", "Построение гиперграфа по трём признакам анкеты")[
  ```r
  categorical <- prepared |>
    select(ResponseId, Age, EdLevel, RemoteWork) |>
    filter(if_all(everything(), ~ !is.na(.x)))
  hypergraph <- hypergraph_outliers(
    as.data.frame(select(categorical, -ResponseId)),
    categorical$ResponseId, 50L, -2
  )
  n_distinct(hypergraph$anomalies$id)
  ```
]

В исследуемой группе из 8~995 полных анкет построено 65 рёбер. Получены 2~094 отметки по сочетаниям анкеты и контекста. После объединения отметок выявлена 951 уникальная анкета. Например, среди 1~011 анкет с незаконченным высшим образованием у возраста менее 18 лет получено $D approx -"3,31"$. Редкость сочетания зависит от контекста гиперребра.

= Задание 4. Преобразование Бокса–Кокса

Преобразование для положительной годовой компенсации $x$ реализовано по § 4.3.5 #link("https://files.stroyinf.ru/Data2/1/4293744/4293744124.pdf")[ГОСТ Р ИСО 16269-4-2017] без сдвига. Использована форма, непрерывная по параметру $lambda$ при $lambda = 0$:
$ y_(lambda)(x) = cases(
  (x^lambda - 1) / lambda quad lambda != 0,
  ln(x) quad lambda = 0,
). $

Параметр выбран максимизацией профильного логарифмического правдоподобия на отрезке $[-2, 2]$ с поправкой Якобиана:
$ ell(lambda) = -n/2 ln(s_lambda^2) + (lambda - 1) sum_(i=1)^n ln(x_i), $
где $s_lambda^2$ — средний квадрат отклонений преобразованных значений от их среднего. Вычисление $x^lambda - 1$ через `expm1` уменьшает потерю точности вблизи нулевого параметра.

#listing("4.1", "Оценка параметра и применение преобразования")[
  ```r
  optimum <- optimize(
    box_cox_likelihood, c(-2, 2),
    values = income$income, maximum = TRUE
  )
  lambda <- optimum$maximum
  transformed_income <- box_cox(income$income, lambda)
  limits <- box_limits(transformed_income)
  ```
]

#block(breakable: false, above: 0.65em, below: 0.65em)[
#par[На 7~577 компенсациях получено $lambda approx -"0,02838"$. По графическому критерию § 4.4, формула (8), использованы границы $Q_1 - k dot "IQR"$ и $Q_3 + k dot "IQR"$, где $k = "1,5"$, а $"IQR" = Q_3 - Q_1$. Границы на преобразованной шкале составили 9,2066 и 10,9825. За ними оказалось 571 наблюдение.]
]

Диаграммы Q–Q на рисунке 4.1 сравнивают эмпирические и теоретические нормальные квантили. У панелей разные вертикальные шкалы, поскольку после преобразования меняются единицы и диапазон значений.

#chart(
  "assets/compensation-qq.png", "4.1",
  "Диаграммы Q–Q до и после преобразования Бокса–Кокса",
  width: 96%,
)

Корреляция выборочных и теоретических квантилей выросла с 0,4645 до 0,8515. После преобразования точки в средней части ближе к прямой. Отклонения в хвостах сохраняются, поэтому преобразование улучшает форму распределения лишь частично.

= Задание 5. Восстановление пропусков

#block(breakable: false, above: 0.65em, below: 0.65em)[
#par[Восстанавливался логарифм годовой компенсации вместе с тремя числовыми признаками стажа. Сравнивались пять заполнений методом сопоставления предсказанных средних (PMM) из `mice` и пять заполнений методом бутстрэп-EM из `Amelia`. При PMM пропуск заполняется наблюдённым значением из числа записей с близким прогнозом. Для проверки скрыты 757 известных компенсаций (около 10~%). Прогнозом служит медиана пяти заполнений, а базовой оценкой — медиана остальных наблюдений.]
]

Средняя абсолютная ошибка вычислялась в логарифмической шкале:
$ "MAE" = 1/m sum_(i=1)^m abs(hat(y)_i - y_i), $
где $m = 757$, $y_i$ — скрытое значение, $hat(y)_i$ — восстановленное.

#listing("5.1", "Маскирование компенсаций для проверки восстановления")[
  ```r
  set.seed(704)
  known <- which(!is.na(imputation_data$log_income))
  holdout <- sample(known, size = floor(0.1 * length(known)))
  masked <- imputation_data
  masked$log_income[holdout] <- NA_real_
  truth <- imputation_data$log_income[holdout]
  baseline <- mean(abs(
    median(masked$log_income, na.rm = TRUE) - truth
  ))
  ```
]

#listing("5.2", "Заполнение маскированной выборки двумя методами")[
  ```r
  pmm <- mice::mice(
    masked, m = 5, method = "pmm", maxit = 5, seed = 704,
    printFlag = FALSE
  )
  pmm_tables <- as.list(mice::complete(pmm, action = "all"))
  set.seed(704)
  bounds <- rbind(c(2, 0, 51), c(3, 0, 51), c(4, 0, 100))
  em <- Amelia::amelia(
    masked, m = 5, bounds = bounds,
    empri = 0.01 * nrow(masked), p2s = 0
  )
  em_tables <- em$imputations
  ```
]

#result-table(
  "5.1", "Ошибка на скрытых значениях компенсации", (2.7fr, 1fr, 1fr),
  table.header([*Метод*], [*Скрыто*], [*MAE*]),
  [Медиана наблюдённых значений], [757], [0,5207],
  [`mice`, сопоставление предсказанных средних], [757], [0,5632],
  [`Amelia`, бутстрэп-EM], [757], [0,7263],
)

#block(breakable: false, above: 0.65em, below: 0pt)[
#par[Медиана дала наименьшую ошибку для точечного прогноза. Среди методов множественной импутации меньшую ошибку дал `mice`. Он использован для пяти заполнений исходной таблицы. Оценки регрессии логарифма компенсации по стажу объединены по правилам Рубина средствами `mitools`.]
]

#listing("5.3", "Объединение регрессий по пяти заполнениям")[
  ```r
  completions <- impute(imputation_data, chosen, 705L)
  analyses <- with(
    mitools::imputationList(completions),
    lm(log_income ~ professional_years + coding_years +
         work_years)
  )
  combined <- mitools::MIcombine(analyses)
  summary(combined)
  ```
]

#result-table(
  "5.2", "Объединённые коэффициенты модели логарифма компенсации",
  (2.5fr, 1fr, 1.9fr),
  table.header([*Признак*], [*Оценка*], [*Интервал 95~%*]),
  [Профессиональный стаж], [0,0189], [[0,0102; 0,0275]],
  [Общий стаж программирования], [0,0041], [[−0,0015; 0,0096]],
  [Опыт работы], [−0,0125], [[−0,0192; −0,0058]],
)

Интервалы описывают условные связи в этой выборке. Проверка со случайным маскированием характеризует восстановление известных ответов. Механизм появления исходных пропусков может отличаться.

= Задание 6. Качество данных до и после обработки

Таблица 6.1 сопоставляет исходную выборку и одну из полных реализаций `mice`. Программно проверено отсутствие пропусков во всех пяти реализациях и сохранение исходных наблюдённых компенсаций.

#listing("6.1", "Проверка полноты таблиц и исходных компенсаций")[
  ```r
  finished <- completions[[1L]]
  observed_income <- !is.na(imputation_data$log_income)
  full_tables <- vapply(
    completions, function(data) !anyNA(data), logical(1)
  )
  stopifnot(
    all(full_tables),
    identical(
      finished$log_income[observed_income],
      imputation_data$log_income[observed_income]
    )
  )
  ```
]

#result-table(
  "6.1", "Сопоставление исходных и обработанных данных",
  (3.6fr, 1fr, 1fr),
  table.header([*Показатель*], [*До*], [*После*]),
  [Число анкет], [9~133], [9~133],
  [Анкеты без идентификатора], [0], [0],
  [Повторные идентификаторы], [0], [0],
  [Пропуски в четырёх числовых полях], [4~717], [0],
  [Непригодные числовые значения], [0], [0],
  [Недоступные ответы в трёх категориальных полях], [140], [140],
)

Годовая компенсация была известна в 7~577 анкетах. Восстановлено 1~556 компенсаций, а медиана изменилась со 146~000 до 145~000 долларов США. Категориальные пропуски сохранены. Это оставляет видимыми ограничения покрытия для анализа гиперграфа.

= Заключение

Числовая диагностика показала зависимость числа отмеченных анкет от пространства признаков и выбранного порога. Гиперграфовый метод выделил 951 анкету с редкими сочетаниями. Преобразование Бокса–Кокса улучшило согласованность квантилей с нормальным распределением. Для числовой таблицы заполнено 4~717 пропусков, а объединение пяти результатов позволило учесть неопределённость заполнения при оценке коэффициентов регрессии.
