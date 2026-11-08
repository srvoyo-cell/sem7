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
  3, "Информационные технологии анализа и обработки данных", 7,
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

Цель работы — освоить трансформацию и визуализацию данных на языке R. По заданию требуется построить сводные таблицы в `pivottabler`, оформить их для Excel и Word и представить результаты анализа средствами `ggplot2`.

Варианту 7 соответствует набор #link("https://www.kaggle.com/datasets/yasirabdaali/stack-overflow-developer-survey-20112022")[Stack Overflow Developer Survey 2011–2022]. Для анализа формата работы выбрана волна 2022 года, в которой есть отдельный вопрос `RemoteWork`. После анкеты с идентификатором около 68~810 в копии Kaggle обнаружены посторонние двоичные данные. Расчёты выполнены по #link("https://survey.stackoverflow.co/2022/")[официальному выпуску Stack Overflow].

= Задание 1. Подготовка данных

Из 73~268 исходных анкет отбирались респонденты, указавшие полную занятость. Поскольку вопрос допускает несколько ответов, признак полной занятости проверялся как отдельная позиция списка `Employment`. Для сравнения исключены записи без страны, распознанного формата работы или идентификатора. Каждая анкета учитывалась один раз по `ResponseId`. Вектор `work_labels` сопоставляет исходным ответам три русских обозначения формата.

#listing("1.1", "Отбор анкет с полной занятостью")[
  ```r
  workers <- survey |>
    filter(grepl(
      "(^|;)Employed, full-time(;|$)", Employment
    )) |>
    mutate(format = factor(
      unname(work_labels[RemoteWork]),
      levels = unname(work_labels)
    )) |>
    filter(
      !is.na(ResponseId), !is.na(Country), !is.na(format)
    ) |>
    distinct(ResponseId, .keep_all = TRUE)
  ```
]

Для первой сводки взяты восемь стран с наибольшим числом ответов после фильтрации. Для второй анкеты распределены по шести интервалам профессионального стажа из `YearsCodePro`. Два текстовых ответа о стаже менее года и свыше 50 лет сохранены как отдельные категории. Пустые значения стажа в этой сводке исключены.

#listing("1.2", "Отбор стран по числу анкет")[
  ```r
  top_countries <- workers |>
    count(Country, name = "responses") |>
    arrange(desc(responses), Country) |>
    slice_head(n = 8L)
  country_data <- workers |>
    filter(Country %in% top_countries$Country) |>
    mutate(country = factor(
      Country, levels = top_countries$Country
    ))
  ```
]

#listing("1.3", "Группировка анкет по стажу программирования")[
  ```r
  experience_levels <- c(
    "<1", "1–4", "5–9", "10–19", "20–50", ">50"
  )
  experience_data <- workers |>
    mutate(
      years = suppressWarnings(as.numeric(YearsCodePro)),
      experience = case_when(
        YearsCodePro == "Less than 1 year" ~ "<1",
        YearsCodePro == "More than 50 years" ~ ">50",
        years >= 1 & years < 5 ~ "1–4",
        years >= 5 & years < 10 ~ "5–9",
        years >= 10 & years < 20 ~ "10–19",
        years >= 20 & years <= 50 ~ "20–50"
      ),
      experience = factor(experience, levels = experience_levels)
    ) |>
    filter(!is.na(experience))
  ```
]

#result-table(
  "1.1", "Число анкет на этапах подготовки", (3fr, 1fr),
  table.header([*Набор*], [*Анкет*]),
  [Исходный выпуск 2022 года], [73~268],
  [Полная занятость, известны страна и формат работы], [49~188],
  [Восемь стран с наибольшим числом анкет], [27~726],
  [Анкеты с известной категорией стажа], [45~566],
)

= Задание 2. Агрегирование данных

Для каждого сочетания признака строк и формата работы вычислялось число анкет `n()`. Строки первой таблицы образованы странами, строки второй — интервалами стажа. Сводки включают итоги по обоим измерениям.

#listing("2.1", "Расчёт численностей по стране и формату работы")[
  ```r
  country_pivot <- pivottabler::PivotTable$new()
  country_pivot$addData(country_data)
  country_pivot$addRowDataGroups(
    "country", totalCaption = "Всего"
  )
  country_pivot$addColumnDataGroups(
    "format", totalCaption = "Всего"
  )
  country_pivot$defineCalculation(
    calculationName = "Responses", summariseExpression = "n()"
  )
  country_pivot$evaluatePivot()
  ```
]

#result-table(
  "2.1", "Число анкет по стране и формату работы",
  (1.9fr, 1fr, 1fr, 1fr, 0.75fr),
  table.header([*Страна*], [*Удалённо*], [*Гибридно*], [*В офисе*], [*Всего*]),
  [США], [6~037], [3~395], [1~065], [10~497],
  [Индия], [1~626], [1~788], [955], [4~369],
  [Германия], [1~077], [2~101], [363], [3~541],
  [Великобритания], [1~232], [1~674], [226], [3~132],
  [Канада], [1~098], [606], [105], [1~809],
  [Бразилия], [1~073], [352], [187], [1~612],
  [Франция], [269], [1~066], [248], [1~583],
  [Испания], [571], [486], [126], [1~183],
  [*Всего*], [*12~983*], [*11~468*], [*3~275*], [*27~726*],
)

В США на удалённый формат приходится 57,5~% ответов, а в Германии гибридный формат составляет 59,3~%. Проценты рассчитаны относительно числа анкет каждой страны из таблицы 2.1.

В таблице 2.2 нулём обозначено отсутствие анкет в сочетании «стаж свыше 50 лет и работа в офисе». В выгрузках это сочетание отображается пустой ячейкой.

#result-table(
  "2.2", "Число анкет по профессиональному стажу и формату работы",
  (1.9fr, 1fr, 1fr, 1fr, 0.75fr),
  table.header([*Стаж, лет*], [*Удалённо*], [*Гибридно*], [*В офисе*], [*Всего*]),
  [$< 1$], [522], [730], [467], [1~719],
  [1–4], [4~721], [5~859], [2~727], [13~307],
  [5–9], [5~542], [5~844], [1~580], [12~966],
  [10–19], [5~274], [5~067], [1~083], [11~424],
  [20–50], [2~847], [2~566], [727], [6~140],
  [$> 50$], [8], [2], [0], [10],
  [*Всего*], [*18~914*], [*20~068*], [*6~584*], [*45~566*],
)

= Задание 3. Форматирование и выгрузка

Для таблиц по стране и по стажу выбраны тёмно-синий и бирюзовый заголовки. Числовые итоги выделены полужирным. Метод `writeToExcelWorksheet` передаёт сводку и её оформление из `pivottabler` в книгу Excel. Сводки записаны на листы «По странам» и «По стажу».

Для Word рассчитанные сводки извлечены методом `asDataFrame` и оформлены через `flextable`. Документ Word содержит обе таблицы с названиями и итоговыми строками. Итоги в выгрузках совпадают с рассчитанными сводками.

#listing("3.1", "Заполнение листа Excel и оформление таблицы Word")[
  ```r
  workbook <- openxlsx::createWorkbook()
  openxlsx::addWorksheet(workbook, "По странам")
  country_pivot$writeToExcelWorksheet(
    wb = workbook, wsName = "По странам", topRowNumber = 2L,
    leftMostColumnNumber = 2L, showRowGroupHeaders = TRUE
  )
  country_table <- country_pivot$asDataFrame(
    rowGroupsAsColumns = TRUE
  )
  word_table <- flextable::flextable(country_table) |>
    flextable::theme_vanilla() |>
    flextable::bg(bg = "#173D62", part = "header")
  ```
]

= Задание 4. Визуализация результатов

На рисунке 4.1 показаны 27~726 анкет из восьми стран. Сегменты разделяют ответы по формату работы. Самый длинный столбец у США, которым соответствуют 10~497 анкет.

#listing("4.1", "Столбчатая диаграмма форматов работы по странам")[
  ```r
  country_counts <- country_data |>
    count(country, format, name = "responses")
  ggplot(country_counts, aes(
    x = responses, y = factor(country, rev(levels(country))),
    fill = format
  )) +
    geom_col() +
    scale_fill_manual(values = work_colours) +
    scale_y_discrete(labels = scales::label_wrap(26))
  ```
]

#chart(
  "assets/country-formats.png", "4.1",
  "Распределение анкет по формату работы в восьми странах",
  width: 80%,
)

#block(breakable: false, above: 0.65em, below: 0pt)[
На рисунке 4.2 сравниваются доли форматов работы. Для группы стажа $g$ и формата $r$ доля равна
$ p_(g,r) = n_(g,r) / (sum_j n_(g,j)), $
где $n_(g,r)$ — число анкет в ячейке сводки. Проверка в R подтвердила, что $sum_r p_(g,r) = 1$ для каждой группы.
]

#listing("4.2", "Расчёт и проверка долей по группам стажа")[
  ```r
  experience_shares <- experience_data |>
    count(experience, format, name = "responses") |>
    group_by(experience) |>
    mutate(share = responses / sum(responses)) |>
    ungroup()
  share_check <- experience_shares |>
    group_by(experience) |>
    summarise(total = sum(share))
  stopifnot(all(abs(share_check$total - 1) < 1e-12))
  ```
]

#chart(
  "assets/experience-shares.png", "4.2",
  "Доли форматов работы внутри групп профессионального стажа",
  width: 86%,
)

= Заключение

Среди 49~188 отобранных анкет гибридный формат оказался наиболее частым (21~585 ответов, 43,9~%). Удалённый формат составил 40,7~%, офисный — 15,4~%. Среди участников со стажем свыше 50 лет удалённо работают 8 из 10 респондентов. Доли относятся к участникам опроса.
