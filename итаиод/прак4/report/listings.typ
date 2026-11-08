// ── listings.typ ─────────────────────────────────────────────────────────────
// Модуль оформления листингов и подписей к таблицам.
//
// Использование в основном документе:
//   #import "listings.typ": setup-listings, listing, tbl-caption
//   #show: setup-listings
// ─────────────────────────────────────────────────────────────────────────────

/// Применяет правила оформления блочного кода ко всему документу.
/// Использовать как: #show: setup-listings
#let setup-listings(body) = {
  show raw.where(block: true): it => {
    set text(font: "Courier New", size: 11pt)
    show raw.where(block: false): set text(font: "Courier New", size: 11pt, top-edge: "ascender", bottom-edge: "descender")
    set par(first-line-indent: 0pt, leading: 0.5em, spacing: 0.6em)
    let lines = it.text.split("\n")
    block(
      fill: luma(245),
      inset: (x: 10pt, y: 8pt),
      radius: 3pt,
      width: 100%,
      above: 0pt, below: 1.5em,
      stroke: (left: 2pt + luma(180)),
      grid(
        columns: (auto, 1fr),
        column-gutter: 8pt,
        row-gutter: 4pt,
        ..lines.enumerate(start: 1).map(((i, line)) => (
          align(right, text(fill: luma(160), size: 9.5pt, str(i))),
          raw(line, lang: it.lang),
        )).flatten()
      )
    )
  }
  body
}

#let listing(num, caption, body) = {
  set par(first-line-indent: 0pt, justify: false)
  block(width: 100%, breakable: false, above: 1.5em, below: 1.5em)[
    #block(above: 0pt, below: 0.25em)[
      #align(left)[#strong[Листинг #num] — #caption]
    ]
    #body
  ]
}

#let tbl-caption(num, caption) = {
  set par(first-line-indent: 0pt, justify: false)
  block(above: 1.5em, below: 0.3em, sticky: true)[
    #align(left)[#strong[Таблица #num] — #caption]
  ]
}
