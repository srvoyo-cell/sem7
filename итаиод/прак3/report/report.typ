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
#let chart(path, number, caption, width: 100%) = block(
  width: 100%, breakable: false, above: 1em, below: 1em,
)[
  #set par(first-line-indent: 0pt, justify: false)
  #block(width: 100%, above: 0pt, below: 0.3em)[
    #align(center, image(path, width: width))
  ]
  #block(width: 100%, above: 0pt, below: 0pt)[
    #align(center, strong[Рисунок #number — #caption])
  ]
]
#title_page(
  3,
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

Цель работы — освоить базовую графику R и выбрать подходящие способы визуализации для данных варианта 7. Приёмы из методических указаний воспроизведены на статистике торговли и сброса сточных вод. В индивидуальной части исследованы квартальные оценки населения Австралии `austres` и результаты Stack Overflow Developer Survey за 2022 год.

Расчёты и построения выполнены в R 4.6.2.

= Задание 1. Графические команды из методических указаний

== Подготовка данных и диаграммы рассеяния

Таблица торговли за 2013 год прочитана из Excel. Она содержит 96 строк и 13 полей. Пропуски, включая прочерки, переведены в `NA`. Экспорт и импорт измерены в миллионах долларов США. Общероссийский итог, федеральные округа и субъекты анализируются раздельно. Из CSV загружен временной ряд сброса сточных вод по морям. Единица объёма сброса — $"млрд" "м"^3$.

На диаграммах торговли металлами экспорт отложен по горизонтали, импорт — по вертикали. Одинаковый масштаб обеих осей задан параметром `asp = 1`. По сравнению со стандартными символами заполненные точки заметнее в местах скопления наблюдений.

#chart("assets/trade.png", "1.1", "Экспорт и импорт металлов по субъектам РФ")

#block(width: 100%, breakable: false, sticky: true,
       above: 0pt, below: 0.65em)[
  #par[Размер точки отражает отношение импорта к экспорту. В расчёт вошли округа с положительными объёмами. Нижняя граница размера сохраняет видимость точек с малой долей импорта.]
]

#listing("1.1", "Размер точки по отношению импорта к экспорту")[
  ```r
  trade_points <- subset(districts,
                         МетЭкспорт > 0 & МетИмпорт > 0)
  ratio <- trade_points$МетИмпорт / trade_points$МетЭкспорт
  point_size <- pmax(0.6, 2 + log(ratio))
  plot(trade_points$МетЭкспорт, trade_points$МетИмпорт,
       asp = 1, pch = 20, cex = point_size,
       col = rgb(1, 0, 0, 0.5))
  ```
]

== Линии, общие оси и легенда

Для динамики сброса использованы точки, ломаные и ступенчатые линии. Параметры `type`, `lty` и `lwd` изменяют способ соединения наблюдений, тип и толщину линий. На общих осях сначала задаются пределы по всем морям, затем добавляются ряды. Такой порядок позволяет сохранить точки, лежащие за пределами первого ряда.

#listing("1.2", "Построение нескольких рядов в общих пределах")[
  ```r
  plot(range(water$Год), range(water[sea_names]), type = "n")
  for (i in seq_along(sea_names)) {
    lines(water$Год, water[[sea_names[i]]], type = "o",
          col = sea_colors[i], pch = 20)
  }
  legend("topright", sea_names,
         col = sea_colors, pch = 20, lty = 1)
  ```
]


#chart("assets/legend.png", "1.2", "Общие оси и два способа оформления легенды")

== Гистограммы, столбцы и сектора

#block(width: 100%, sticky: true, above: 0pt, below: 0.65em)[
  #par[На гистограмме продовольственного экспорта число субъектов подсчитано в интервалах по $100 "млн долл. США"$. Для сравнения химического экспорта округов выбрана горизонтальная столбчатая диаграмма с подписями категорий вдоль вертикальной оси.]
]

#listing("1.3", "Интервалы гистограммы и горизонтальные столбцы")[
  ```r
  food_exports <- regions$ПродЭкспорт
  upper <- 100 * ceiling(max(food_exports, na.rm = TRUE) / 100)
  hist(food_exports, breaks = seq(0, upper, by = 100))
  barplot(districts$ХимЭкспорт, names.arg = district_labels,
          horiz = TRUE, las = 1)
  ```
]

#chart("assets/export.png", "1.3", "Химический экспорт федеральных округов", width: 80%)

На круговой диаграмме доля округа вычисляется как
$ p_i = 100 dot x_i / (sum_j x_j), $
где $x_i$ — объём химического экспорта округа. Проверены подписи, поворот и готовые палитры. Собственная палитра получена интерполяцией опорных цветов в `colorRampPalette()`.

#chart("assets/sectors.png", "1.4", "Доли округов при двух направлениях обхода секторов", width: 84%)

= Задание 2. Визуализация встроенного набора

Ряд `austres` содержит 89 наблюдений со II квартала 1971 года по II квартал 1993 года. Уровень изображён линией, изменения — вертикальными отрезками. Приращение между соседними кварталами равно #box[$Delta N_t = N_t - N_(t-1)$], где $N_t$ — численность в тысячах человек.

#listing("2.1", "Уровень ряда и квартальное приращение")[
  ```r
  population_ts <- datasets::austres
  population <- data.frame(
    time = as.numeric(time(population_ts)),
    level = as.numeric(population_ts)
  ) |>
    mutate(change = level - lag(level))
  plot(population$time, population$level, type = "l")
  plot(population$time, population$change, type = "h")
  ```
]

#chart("assets/population.png", "2.1", "Уровень численности и изменение за квартал")

Численность выросла с 13 067,3 до $"17 661,5" "тыс. человек"$. Прирост за весь период составил $"4 594,2" "тыс. человек"$.

= Задание 3. Обработка и визуализация опроса

== Источник и подготовка

Использована волна опроса Stack Overflow за 2022 год. CSV этой волны из сборки Kaggle повреждён, поэтому для расчётов взят официальный выпуск. Проверены наличие нужных полей и уникальность идентификатора в 73 268 анкетах. Для распределения по странам исключены пропуски в `Country`. В распределение стажа включены только конечные неотрицательные значения `WorkExp`.

#listing("3.1", "Подготовка групп для визуализации")[
  ```r
  countries <- survey |>
    filter(!is.na(Country)) |>
    count(Country, name = "responses", sort = TRUE) |>
    slice_head(n = 8)

  work_experience <- survey |>
    filter(is.finite(WorkExp), WorkExp >= 0) |>
    pull(WorkExp)
  ```
]

== Число анкет по странам и распределение стажа

По числу ответов лидируют США, Индия и Германия. Ранжирование отражает состав участников опроса. Для 36 769 анкет с известным стажем гистограмма построена по пятилетним интервалам. Медиана равна 8 годам, квартили — 4 и 15 годам, среднее — 10,24 года.

#listing("3.2", "Число анкет по странам и интервалы стажа")[
  ```r
  country_labels <- sub("^United Kingdom.*", "United Kingdom",
                        countries$Country)
  country_labels <- sub("^United States.*", "United States",
                        country_labels)
  barplot(rev(countries$responses),
          names.arg = rev(country_labels), horiz = TRUE, las = 1)
  upper <- max(5, ceiling(max(work_experience) / 5) * 5)
  hist(work_experience, breaks = seq(0, upper, by = 5),
       right = FALSE, include.lowest = TRUE)
  ```
]

#chart("assets/countries.png", "3.1", "Восемь стран по числу анкет", width: 90%)
#chart("assets/experience.png", "3.2", "Распределение стажа участников", width: 80%)

== Языки программирования

#block(width: 100%, sticky: true, above: 0pt, below: 0.65em)[
  #par[Строковый ответ разделён на отдельные категории. Повтор одного языка внутри анкеты учитывается один раз. Доля языка рассчитана от 70 975 анкет с заполненным ответом на этот вопрос. Участники могли выбрать несколько языков, поэтому сумма долей может превышать $100 "%"$.]
]

#listing("3.3", "Подсчёт ответов с учётом множественного выбора")[
  ```r
  languages <- survey$LanguageHaveWorkedWith
  answered <- languages[!is.na(languages)]
  languages_per_response <- strsplit(answered, ";", fixed = TRUE)
  unique_languages <- lapply(languages_per_response, unique)
  counts <- sort(
    table(unlist(unique_languages, use.names = FALSE)),
    decreasing = TRUE
  )
  top_languages <- head(counts, 8)
  shares <- 100 * as.numeric(top_languages) / length(answered)
  ```
]

#block(width: 100%, breakable: false, above: 1em, below: 0pt)[
  #chart("assets/languages.png", "3.3", "Доли наиболее часто указанных языков", width: 83%)

  #results-table("3.1", "Языки с наибольшей долей ответов")[
    #table(
      columns: (1.1fr, 1fr, 1.4fr),
      table.header([Язык], [Анкеты], [Доля ответивших, %]),
      [JavaScript], [46 443], [65,4],
      [HTML/CSS], [39 142], [55,1],
      [SQL], [35 127], [49,5],
      [Python], [34 155], [48,1],
    )
  ]
]

= Контрольные вопросы

*1–4.* `plot()` строит стандартный график, `barplot()` — столбцы, `pie()` — сектора, `hist()` — гистограмму. Интервалы задаёт `breaks`, ориентацию столбцов — `horiz`. *5–8.* Параметры устройства задаёт `par()`. Порядок полей `mar` — низ, лево, верх, право. `las = 2` разворачивает подписи, а положительный `scipen` повышает предпочтение обычной записи чисел.

*9–12.* `col` задаёт цвет, `lwd` и `lty` — толщину и тип линии, `cex` и `pch` — размер и форму символа. Точки и линии добавляют `points()` и `lines()`. *13–16.* `box()` строит рамку, `grid()` — сетку, `asp = 1` выравнивает масштаб осей. Размеры подписей и заголовков задают `cex.axis`, `cex.lab`, `cex.main` и `cex.sub`.

*17–20.* Цвет задаётся именем, HEX-кодом, номером в палитре или функциями `rgb()`, `hsv()`, `hcl()`. Прозрачность регулируют `alpha` и `adjustcolor()`. Доступны палитры `rainbow()`, `heat.colors()`, `topo.colors()`, `terrain.colors()` и `hcl.colors()`. `colorRampPalette()` интерполирует опорные цвета. *21–22.* `legend()` строит легенду. Символы задаёт `pch`, линии — `lty` и `lwd`, плашки — `fill`.

= Заключение

На `austres` сопоставлены уровень ряда и квартальные приращения. В опросе 2022 года лидируют США по числу анкет, медианный стаж составил 8 лет, JavaScript указан в $"65,4" "%"$ заполненных ответов о языках.
