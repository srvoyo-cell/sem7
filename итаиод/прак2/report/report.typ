#import "title_page_prac.typ": title_page
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
#show heading: it => block(above: 1.5em, below: 1.5em, sticky: true)[
  #set par(first-line-indent: 0pt)
  #text(size: 14pt, weight: "bold", it.body)
]
#show outline.entry.where(level: 1): strong
#set table(
  stroke: 0.5pt,
  inset: 6pt,
  align: center + horizon,
  fill: (_, row) => if row == 0 { luma(220) } else { white },
)
#let results-table(number, caption, content) = block(
  width: 100%, breakable: false, above: 1em, below: 1em,
)[
  #set par(first-line-indent: 0pt, justify: false)
  #tbl-caption(number, caption)
  #block(width: 100%, above: 0pt, below: 0pt)[
    #align(center, content)
  ]
]
#title_page(
  2,
  "Информационные технологии анализа и обработки данных",
  7,
  проверил: (должность: [доктор технических наук, профессор], фио: [В. Г. Суфиянов]),
)
#pagebreak()
#block(width: 100%)[
  #set par(first-line-indent: 0pt, justify: false)
  #align(center, strong[СОДЕРЖАНИЕ])
  #outline(title: none, indent: auto, depth: 2)
]
#pagebreak()
#set page(numbering: "1")
#counter(page).update(3)

= Введение

Цель практической работы — освоить создание, загрузку и обработку таблиц стандартными средствами R и функциями пакета `dplyr`. Сначала выполнены операции из методических указаний. Затем для варианта 7 обработаны квартальный ряд численности населения Австралии `austres` и ответы Stack Overflow Developer Survey за 2022 год. Время выполнения сравнивается на одной и той же операции группировки.

Расчёты выполнены в R 4.6.2.

= Задание 1. Операции с таблицами из методических указаний

== Создание, отбор и объединение строк

В примерах 2.1–2.6 построена таблица из четырёх строк. В `data.frame` столбцы могут иметь разные типы, при этом их длины совпадают. В учебном примере столбцы `x`, `y`, `z` имеют целочисленный, строковый и логический типы соответственно. При отборе по условию `x > 2` остаются строки с ключами 3 и 4. При объединении по ключу `x` сохраняются только значения 1, 2 и 3, присутствующие в обеих таблицах.

#listing("1.1", "Создание таблицы и отбор строк")[
  ```r
  df <- data.frame(x = 1:4, y = LETTERS[1:4], z = c(TRUE, FALSE))
  df[3:4, -1, drop = FALSE]
  subset(df, x > 2, select = c(x, y))
  df_key <- data.frame(x = c(3, 2, 6, 1),
                       s = c(100, 1000, 300, 500))
  merge(df, df_key, by = "x")
  ```
]

Операция `rbind()` добавляет наблюдения с теми же полями, `cbind()` объединяет столбцы по позиции строки, а `merge()` сопоставляет строки по значению ключа. В примере 2.7 таблица `mtcars` из 32 наблюдений и 11 переменных дополнена сведениями о шкалах. Форма двигателя и тип коробки передач переведены в факторы, а количество цилиндров, передач и карбюраторов — в упорядоченные факторы.

#listing("1.2", "Шкалы признаков в таблице автомобилей")[
  ```r
  cars_with_scales <- within(datasets::mtcars, {
    vs <- factor(vs, levels = c(0, 1), labels = c("V", "S"))
    am <- factor(am, levels = c(0, 1),
                 labels = c("automatic", "manual"))
    cyl <- ordered(cyl)
    gear <- ordered(gear)
    carb <- ordered(carb)
  })
  ```
]

== Преобразование таблицы рейсов

#block(width: 100%, sticky: true, above: 0pt, below: 0.65em)[
  #par[На таблице `nycflights13::flights` (336 776 строк, 19 столбцов) проверены отбор, сортировка, выбор и переименование полей, удаление повторов, вычисление признаков, агрегирование и случайная выборка. Разность задержек при прилёте и вылете выражена в минутах, её отношение ко времени полёта — в минутах за час.]
]

#listing("1.3", "Выбор рейсов по дате и выделение маршрутов")[
  ```r
  january_1 <- filter(flights, month == 1, day == 1)
  january_february <- filter(flights, month %in% c(1, 2))
  routes <- distinct(select(flights, origin, dest))
  nrow(january_1)
  nrow(january_february)
  ```
]

#listing("1.4", "Расчёт разности задержек за время полёта")[
  ```r
  mutate(flights,
    gain = arr_delay - dep_delay,
    gain_per_hour = gain / (air_time / 60)
  )
  summarise(flights, delay = mean(dep_delay, na.rm = TRUE))
  ```
]

В `mutate()` поле `gain_per_hour` использует только что вычисленное `gain`. При аналогичном вызове `transform()` обращение к `gain` завершается ошибкой, поскольку такого столбца в исходной таблице ещё нет. На 1 января приходится 842 рейса, на январь и февраль вместе — 51 955. Без пропусков средняя задержка вылета составляет около $"12,6" "мин"$. Выборка в $1 "%"$ содержит 3 368 строк.

= Задание 2. Обработка встроенных данных

Ряд `austres` содержит 89 наблюдений со II квартала 1971 года по II квартал 1993 года.

#results-table("2.1", "Поля квартальной таблицы")[
  #table(
    columns: (1fr, 1.6fr, 1.4fr),
    table.header([Поле], [Смысл и тип], [Единица]),
    [Год], [Целочисленный год], [Год],
    [Квартал], [Порядковый номер от 1 до 4], [Квартал],
    [Население], [Вещественная численность], [Тыс. человек],
    [Прирост], [Вещественная разность соседних значений], [Тыс. человек],
  )
]

#listing("2.1", "Квартальные приращения и отбор полных лет")[
  ```r
  population_ts <- datasets::austres
  pop <- as.numeric(population_ts)
  population <- data.frame(
    year = as.integer(time(population_ts)),
    quarter = as.integer(cycle(population_ts)),
    population_thousands = pop,
    quarter_change_thousands = c(NA_real_, diff(pop))
  )
  quarters_by_year <- table(population$year)
  full_years <- names(quarters_by_year[quarters_by_year == 4L])
  complete_years <- subset(population, year %in% full_years)
  december <- subset(complete_years, quarter == 4L)
  ```
]

Численность за весь период выросла с 13 067,3 до $"17 661,5" "тыс. человек"$. По 21 полному году вычислены приращение $N_4 - N_3$ и средний уровень $(N_1 + N_2 + N_3 + N_4)/4$, где $N_i$ — численность в квартале $i$. Результаты объединены по календарному году.

#listing("2.2", "Группировка и соединение по году")[
  ```r
  yearly_average <- aggregate(
    population_thousands ~ year, data = complete_years, FUN = mean
  )
  yearly <- merge(
    december[c("year", "population_thousands",
               "quarter_change_thousands")],
    yearly_average, by = "year"
  )
  ```
]

#results-table("2.2", "Примеры годовых сводок в тысячах человек")[
  #table(
    columns: (0.7fr, 1.1fr, 1.3fr, 1.3fr),
    table.header([Год], [Уровень в IV квартале], [Изменение за IV квартал], [Средний уровень]),
    [1972], [13 409,3], [55,4], [13 330,27],
    [1992], [17 568,7], [42,7], [17 506,15],
  )
]

= Задание 3. Обработка внешних данных

Выбрана одна волна опроса, чтобы использовать единый набор полей и одинаковые определения признаков. CSV за 2022 год в сборке Kaggle повреждён, поэтому использован CSV из официального выпуска Stack Overflow. Проверены наличие четырёх выбранных полей и уникальность идентификаторов всех 73 268 анкет. Пустые значения представлены как `NA`.

#results-table("3.1", "Признаки опроса, использованные в обработке")[
  #table(
    columns: (1.0fr, 1.35fr, 1.1fr),
    table.header([Признак], [Смысл], [Тип и единица]),
    [Идентификатор], [Уникальный номер анкеты], [Целое, безразмерный ключ],
    [Страна], [Страна по ответу участника], [Строка, категория],
    [Стаж], [Стаж работы по ответу участника], [Числовой, годы],
    [Языки], [Языки, указанные за последний год], [Строка, список категорий],
  )
]

С помощью `select()` оставлены нужные поля, `rename()` присваивает рабочие имена, а `mutate()` добавляет число языков в анкете. Список языков разделён по разделителю из CSV. Пустой ответ остаётся пропуском. Страна отсутствует в 1 497 анкетах, стаж — в 36 499. Средний стаж по заполненным ответам составляет 10,24 года.

#listing("3.1", "Переименование признаков и подсчёт языков")[
  ```r
  survey_fields <- survey |>
    select(ResponseId, Country, WorkExp,
           LanguageHaveWorkedWith) |>
    rename(country = Country, work_years = WorkExp) |>
    mutate(language_count = if_else(
      is.na(LanguageHaveWorkedWith),
      NA_integer_,
      lengths(strsplit(LanguageHaveWorkedWith, ";", fixed = TRUE))
    ))
  ```
]

#listing("3.2", "Сводка по странам для стажа от пяти лет")[
  ```r
  survey_fields |>
    filter(!is.na(country), work_years >= 5) |>
    group_by(country) |>
    summarise(
      responses = n(),
      mean_work_years = mean(work_years),
      .groups = "drop"
    ) |>
    arrange(desc(responses), country)
  ```
]

#results-table("3.2", "Страны с наибольшим числом анкет после отбора")[
  #table(
    columns: (1.1fr, 1fr, 1.5fr),
    table.header([Страна], [Число анкет], [Средний стаж, лет]),
    [США], [5 877], [16,45],
    [Германия], [1 961], [12,79],
    [Индия], [1 882], [9,26],
  )
]

= Задание 4. Сравнение времени выполнения

В обеих реализациях отбираются анкеты с известной страной и стажем от пяти лет. Средние по странам вычисляются через `aggregate()` или `group_by()` и `summarise()`. После согласования порядка строк результаты для 162 стран совпали с учётом числовой точности.

#listing("4.1", "Средний стаж в двух реализациях группировки")[
  ```r
  summarise_base <- function(data) {
    selected <- subset(data, !is.na(Country) & WorkExp >= 5)
    aggregate(WorkExp ~ Country, data = selected, FUN = mean)
  }
  summarise_dplyr <- function(data) data |>
    filter(!is.na(Country), WorkExp >= 5) |>
    group_by(Country) |>
    summarise(WorkExp = mean(WorkExp), .groups = "drop")
  ```
]

#listing("4.2", "Проверка совпадения средних по странам")[
  ```r
  base_result <- summarise_base(survey)
  dplyr_result <- summarise_dplyr(survey)
  row_order <- match(base_result$Country, dplyr_result$Country)
  dplyr_aligned <- dplyr_result[row_order, ]
  stopifnot(
    nrow(base_result) == nrow(dplyr_result),
    identical(base_result$Country, dplyr_aligned$Country),
    isTRUE(all.equal(base_result$WorkExp, dplyr_aligned$WorkExp))
  )
  ```
]

#listing("4.3", "Оценка времени одного вызова")[
  ```r
  measure_ms <- function(method, data) {
    replicate(5L, {
      elapsed <- system.time(
        for (i in seq_len(20L)) invisible(method(data))
      )[["elapsed"]]
      elapsed * 1000 / 20L
    })
  }
  ```
]

Каждая серия включает 20 вызовов. Если длительность серии $t$ выражена в секундах, время вызова в миллисекундах равно $1000 dot t / 20$. Чтение CSV и вывод результатов не входят в замер.

#block(width: 100%, breakable: false, above: 1em, below: 0pt)[
  #results-table("4.1", "Время вызова по пяти сериям, мс")[
    #table(
      columns: (1fr, 1fr, 1fr, 1fr),
      table.header([Реализация], [Медиана], [Минимум], [Максимум]),
      [base R], [61,5], [49,5], [73,5],
      [`dplyr`], [28,5], [22,5], [32,0],
    )
  ]

  Отношение медиан составляет $"61,5" / "28,5" approx "2,16"$. Это результат конкретной группировки на одном наборе данных и в одной среде запуска.
]

= Заключение

Команды создания, выбора и объединения таблиц проверены на учебных данных. По квартальному ряду получена сводка по 21 полному году, а по анкетам — распределение числа ответов и среднего стажа по странам. Обе реализации группировки дали совпадающие результаты. В выполненном замере медиана времени составила $"28,5" "мс"$ для `dplyr` и $"61,5" "мс"$ для base R.
