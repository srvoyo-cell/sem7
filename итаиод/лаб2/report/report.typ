#import "title_page_lab.typ": title_page
#import "listings.typ": setup-listings, listing, tbl-caption

#show: setup-listings
#set page(paper: "a4", margin: (top: 2cm, bottom: 2cm, left: 3cm, right: 1.5cm), numbering: none, number-align: center)
#set text(font: "Times New Roman", size: 14pt, lang: "ru")
#set par(leading: 0.65em, spacing: 0.65em, justify: true, first-line-indent: (amount: 1.25cm, all: true))
#set heading(numbering: none)
#show heading: it => block(above: 1.5em, below: 1.5em, sticky: true)[
  #set par(first-line-indent: 0pt)
  #text(size: 14pt, weight: "bold", it.body)
]

#let report-table(number, title, columns, header, rows) = {
  set par(first-line-indent: 0pt, justify: false)
  block(width: 100%, breakable: false, above: 1em, below: 1em)[
    #tbl-caption(number, title)
    #block(above: 0pt, below: 0pt)[
      #align(center)[
        #table(
          columns: columns,
          align: (col, _) => if col == 0 { left + horizon } else { center + horizon },
          stroke: 0.5pt,
          inset: 6pt,
          fill: (_, row) => if row == 0 { luma(220) } else { white },
          table.header(..header.map(strong)),
          ..rows,
        )
      ]
    ]
  ]
}

#let report-plot(number, caption, path, width) = {
  set par(first-line-indent: 0pt, justify: false)
  block(width: 100%, breakable: false, above: 0.8em, below: 0.8em)[
    #block(width: 100%, above: 0pt, below: 0.35em)[#align(center, image(path, width: width))]
    #block(width: 100%, above: 0pt, below: 0pt)[#align(center)[*Рисунок #number — #caption*]]
  ]
}

#title_page(
  2,
  "Информационные технологии анализа и обработки данных",
  7,
  проверил: (
    должность: "доктор технических наук, профессор",
    фио: "В. Г. Суфиянов",
  ),
)
#pagebreak()
#{
  set par(first-line-indent: 0pt, justify: false)
  show outline.entry.where(level: 1): strong
  block(width: 100%, above: 0pt, below: 1em, sticky: true)[
    #align(center, strong[СОДЕРЖАНИЕ])
  ]
  outline(title: none, indent: auto, depth: 2)
}
#pagebreak()
#set page(numbering: "1")
#counter(page).update(3)


= Введение

Цель работы — развернуть OLAP-куб на данных Stack Overflow Developer Survey и выполнить запросы по времени и географии. Реализация выполнена на R 4.6.2 в Jupyter. Для представления и подготовки таблиц использованы `tibble` и `dplyr`, для схемы хранилища и запросов — `rolap`.

= Задание 1. Загрузка и согласование данных

Источники варианта 7 — двенадцать ежегодных таблиц с анкетами. Волны 2011–2021 годов взяты из архива Kaggle. Повреждённая таблица за 2022 год заменена оригинальным выпуском Stack Overflow. Для каждой анкеты сохранены год волны и географическое поле. Служебные строки в выпусках 2011–2015 годов исключены.

Названия географического поля различаются между выпусками. После чтения удалены краевые пробелы, а несколько явно эквивалентных названий приведены к одной форме. Например, `United States of America` приведено к `United States`. Укрупнённые регионы, например `Other Europe`, сохранены отдельно. Пустые ответы отнесены к категории «Не указано». В 2022 году её доля составила 1~497 из 73~268 анкет, или 2,04~%.

#listing("1.1", "Нормализация места проживания и построение десятилетия")[
  ```r
  responses <- responses_raw |>
    mutate(
      place = trimws(place),
      place = coalesce(na_if(place, ""), "Не указано"),
      place = recode(place, !!!place_names),
      decade = paste0(year %/% 10L * 10L, "-е")
    )
  ```
]

= Задание 2. Построение OLAP-куба

Хранилище организовано по схеме «звезда». Зерно таблицы фактов — пара «год, место проживания». Для каждой встречающейся пары число анкет вычисляется функцией `count`. Измерение `wave` содержит год и десятилетие, измерение `geography` — согласованную географическую категорию. В таблице `survey` эти измерения связаны с фактами ключами `wave_key` и `geography_key`.

Мера $n_(y, g)$ считает анкеты года $y$ в категории $g$. Агрегация `SUM` в `rolap` даёт годовой итог #box[$N_y = sum_g n_(y, g)$]. Анкеты без географии входят в сумму через категорию «Не указано».

#listing("2.1", "Определение схемы и развёртывание хранилища")[
  ```r
  fact_input <- count(
    responses, decade, year, place, name = "responses"
  )
  survey_schema <- star_schema() |>
    define_facts(name = "survey", measures = "responses",
                 agg_functions = "SUM") |>
    define_dimension(name = "wave",
                     attributes = c("decade", "year")) |>
    define_dimension(name = "geography", attributes = "place")
  survey_db <- star_database(survey_schema, fact_input)
  ```
]

#listing("2.2", "Размеры таблиц и контроль общей суммы меры")[
  ```r
  star_tables <- as_tibble_list(survey_db)
  tibble(
    table = names(star_tables),
    rows = vapply(star_tables, nrow, integer(1L)),
    columns = vapply(star_tables, ncol, integer(1L))
  )
  stopifnot(sum(star_tables$survey$responses) == nrow(responses))
  ```
]

#report-table(
  "2.1", "Размеры таблиц хранилища",
  (1fr, 1.65fr, 0.75fr, 0.85fr),
  ([Таблица], [Назначение], [Строк], [Столбцов]),
  (
    [`wave`], [Год и десятилетие], [12], [3],
    [`geography`], [Место проживания], [281], [2],
    [`survey`], [Факты и число анкет], [1~600], [4],
  ),
)

Сумма меры по 1~600 ячейкам равна 568~855. Это совпадает с числом анкет после загрузки и нормализации.

= Задание 3. Агрегация и детализация

При свёртке (`roll-up`) из запроса исключено измерение географии. Мера суммируется сначала по годам, затем по десятилетиям. За 2011–2019 годы получено 347~687 анкет, за 2020–2022 годы — 221~168. Оба итога относятся к годам, представленным в наборе. При детализации (`drill-down`) итог 2020-х раскрывается по трём отдельным волнам.

#report-table(
  "3.1", "Число анкет по годам",
  (1fr, 1.25fr, 1fr, 1.25fr),
  ([Год], [Анкет], [Год], [Анкет]),
  (
    [2011], [2~813], [2017], [51~392],
    [2012], [6~243], [2018], [98~855],
    [2013], [9~742], [2019], [88~883],
    [2014], [7~643], [2020], [64~461],
    [2015], [26~086], [2021], [83~439],
    [2016], [56~030], [2022], [73~268],
  ),
)

#listing("3.1", "Свёртка по десятилетиям")[
  ```r
  decade_query <- survey_db |>
    star_query() |>
    select_dimension(name = "wave", attributes = "decade") |>
    select_fact(name = "survey", measures = "responses")
  decade_tables <- as_single_tibble_list(
    run_query(survey_db, decade_query)
  )
  ```
]

#listing("3.2", "Детализация меры по отдельным годам")[
  ```r
  year_query <- survey_db |>
    star_query() |>
    select_dimension(name = "wave", attributes = "year") |>
    select_fact(name = "survey", measures = "responses")
  year_tables <- run_query(survey_db, year_query) |>
    as_single_tibble_list()
  year_totals <- year_tables$survey |>
    mutate(year = as.integer(year)) |>
    arrange(year)
  filter(year_totals, year >= 2020L)
  ```
]

Сумма итогов по десятилетиям совпадает с общим числом анкет. Годовые итоги запроса также совпали с прямым подсчётом строк исходных данных по годам.

= Задание 4. Географические срезы

Одномерный срез (`slice`) фиксирует 2022 год и показывает число ответов по географическим категориям. Для доли каждой категории знаменателем служит размер всей волны, включая анкеты без заполненной географии. Наибольшее число ответов в срезе поступило из США.

#listing("4.1", "Срез по году и доли ответов по странам")[
  ```r
  geo_query <- survey_db |>
    star_query() |>
    select_dimension(name = "wave", attributes = "year") |>
    select_dimension(name = "geography", attributes = "place") |>
    select_fact(name = "survey", measures = "responses")
  slice_query <- geo_query |>
    filter_dimension(name = "wave", year == "2022")
  latest <- run_query(survey_db, slice_query) |>
    as_single_tibble_list()
  latest$survey |>
    mutate(share_pct = 100 * responses / sum(responses)) |>
    filter(place != "Не указано") |>
    slice_max(responses, n = 5L, with_ties = FALSE)
  ```
]

#report-table(
  "4.1", "Пять стран с наибольшим числом анкет за 2022 год",
  (1.8fr, 1fr, 1.15fr),
  ([Страна], [Анкет], [Доля, %]),
  (
    [США], [13~543], [18,48],
    [Индия], [6~639], [9,06],
    [Германия], [5~395], [7,36],
    [Великобритания], [4~190], [5,72],
    [Канада], [2~490], [3,40],
  ),
)

Многомерный срез (`dice`) ограничивает запрос одновременно тремя волнами и тремя странами. Операция поворота выводит годы строками, а страны столбцами.

#listing("4.2", "Ограничение по двум измерениям и поворот среза")[
  ```r
  dice_query <- geo_query |>
    filter_dimension(
      name = "wave", year %in% c("2020", "2021", "2022")
    ) |>
    filter_dimension(
      name = "geography",
      place %in% c("United States", "India", "Germany")
    )
  recent <- run_query(survey_db, dice_query) |>
    as_single_tibble_list()
  xtabs(responses ~ year + place, data = recent$survey)
  ```
]

#report-table(
  "4.2", "Число ответов в трёх странах за 2020–2022 годы",
  (0.8fr, 1.3fr, 1.2fr, 1fr),
  ([Год], [Германия], [Индия], [США]),
  (
    [2020], [3~890], [8~403], [12~469],
    [2021], [5~625], [10~511], [15~288],
    [2022], [5~395], [6~639], [13~543],
  ),
)

В 2022 году число анкет уменьшилось во всех трёх выбранных странах относительно 2021 года. Самое заметное падение в абсолютных числах — в Индии, на 3~872 ответа.

= Задание 5. Динамика ответов

Максимальная волна пришлась на 2018 год и составила 98~855 анкет. После снижения в 2019–2020 годах число ответов выросло в 2021 году, а в 2022 году уменьшилось с 83~439 до 73~268. Относительное изменение за последний год составило

$ ("73 268" / "83 439" - 1) dot 100 "%" approx -"12,2" "%". $

#listing("5.1", "Поиск наибольшей волны и расчёт изменения")[
  ```r
  peak_wave <- slice_max(year_totals, responses, n = 1L,
                         with_ties = FALSE)
  latest_total <- filter(year_totals, year == 2022L)$responses
  previous_total <- filter(year_totals, year == 2021L)$responses
  relative_change <- 100 * (latest_total / previous_total - 1)
  ```
]

#report-plot("5.1", "Размер волны опроса по годам", "assets/year-counts.png", 14.4cm)

#block(width: 100%, breakable: false, above: 1.5em)[
  = Заключение

  Куб охватывает 568~855 анкет за 2011–2022 годы. Итоги срезов и свёртки проверены прямым подсчётом анкет. Межгодовое сравнение географии зависит от состава вариантов ответа, поскольку ранние волны включают укрупнённые регионы.
]
