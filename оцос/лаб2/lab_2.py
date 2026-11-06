import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray
    from scipy import signal

    type RealArray = NDArray[np.float64]
    type ComplexArray = NDArray[np.complex128]
    type IntArray = NDArray[np.int64]
    type Row = dict[str, str | int | float]


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # ОЦОС. Лабораторная работа №2. Вариант 7
    **Исследование корреляционных и спектральных свойств случайных сигналов
    при прохождении через линейные системы**

    Цель: оценить АКФ и СПМ случайных сигналов, исследовать действие линейного
    фильтра и когерентность независимых шумов. Выполняются пункты 1–5,
    контрольные вопросы 1–8 и дополнительный пункт 6 с ФНЧ Чебышёва I рода.
    Исследуемый аспект варианта — влияние длинной реализации.

    Источники: «ЛР2_ОЦОС_2026_Задание.pdf» и
    «ЛР2_ОЦОС_2026_структура_отчета.pdf» в каталоге этого ноутбука.
    По указанию пользователя реализация выполнена в marimo.
    Формулы в задании не нумерованы; ниже используются ссылки на пункты.
    """)


@app.cell
def _():
    sample_count: int = 8192
    sampling_rate: float = 1.0
    cutoff: float = 0.20 * np.pi
    filter_order: int = 2
    segment_length: int = 1024
    overlap_fraction: float = 0.5
    window_name: str = "hann"
    realization_count: int = 1
    seed: int = 7
    overlap_count: int = int(segment_length * overlap_fraction)
    segment_count: int = 1 + (sample_count - segment_length) // (
        segment_length - overlap_count
    )
    mo.ui.table(
        [
            {
                "N": sample_count,
                "ωc/π": cutoff / np.pi,
                "Порядок": filter_order,
                "L": segment_length,
                "Перекрытие": overlap_fraction,
                "Окно": window_name,
                "K": realization_count,
                "Сегментов": segment_count,
            }
        ],
        selection=None,
    )
    return (
        cutoff,
        filter_order,
        overlap_count,
        sample_count,
        sampling_rate,
        seed,
        segment_count,
        segment_length,
        window_name,
    )


@app.cell
def _():
    def white_noise(length: int, random_seed: int) -> RealArray:
        """Сгенерировать БГШ по п. 1.1 и разделу 5.1 структуры работы.

        Args:
            length: Количество отсчётов.
            random_seed: Зерно независимого генератора.
        Returns:
            Реализация с выборочными средним 0 и дисперсией 1 (ddof=0).
        """
        values: RealArray = np.random.default_rng(random_seed).standard_normal(length)
        centered: RealArray = values - np.mean(values)
        normalized: RealArray = centered / np.std(centered)
        return normalized

    def acf_direct(values: RealArray) -> RealArray:
        """Оценить АКФ прямой корреляцией по п. 2.1.

        Args:
            values: Вещественная реализация длины N.
        Returns:
            Центрированная смещённая АКФ, делённая на N, лаги -(N-1)..N-1.
        """
        centered: RealArray = values - np.mean(values)
        result: RealArray = np.correlate(centered, centered, mode="full") / values.size
        return result

    def acf_fft(values: RealArray) -> RealArray:
        """Оценить линейную АКФ по Винеру–Хинчину, п. 2.2.

        Args:
            values: Вещественная реализация длины N.
        Returns:
            Та же смещённая АКФ, что в acf_direct, на лагах -(N-1)..N-1.
            Дополнение до 2N-1 устраняет циклическое наложение.
        """
        centered: RealArray = values - np.mean(values)
        fft_length: int = 2 * values.size - 1
        spectrum: ComplexArray = np.fft.fft(centered, n=fft_length)
        circular: RealArray = np.fft.ifft(np.abs(spectrum) ** 2).real / values.size
        result: RealArray = np.fft.fftshift(circular)
        return result

    def periodogram(values: RealArray, fs: float) -> tuple[RealArray, RealArray]:
        """Вычислить двустороннюю периодограмму по п. 3.1.

        Args:
            values: Вещественная реализация длины N.
            fs: Частота дискретизации, в этой работе 1.
        Returns:
            Частоты в рад/отсчёт и СПМ |FFT(x-mean(x))|²/(N*fs).
            При fs=1 дисперсия равна сумме СПМ, делённой на N.
        """
        centered: RealArray = values - np.mean(values)
        spectrum: ComplexArray = np.fft.fft(centered)
        omega: RealArray = np.fft.fftshift(np.fft.fftfreq(values.size, d=1 / fs))
        density: RealArray = np.fft.fftshift(np.abs(spectrum) ** 2 / (values.size * fs))
        return 2 * np.pi * omega / fs, density

    def welch_density(
        values: RealArray, fs: float, length: int, overlap: int, window: str
    ) -> tuple[RealArray, RealArray]:
        """Получить двустороннюю СПМ Уэлча по п. 3.2.

        Args:
            values: Исследуемый сигнал.
            fs: Частота дискретизации.
            length: Длина сегмента L.
            overlap: Число общих отсчётов соседних сегментов.
            window: Оконная функция (периодическое окно SciPy).
        Returns:
            Частоты в рад/отсчёт и СПМ density без удвоения положительной части.
            Среднее удаляется один раз, а не отдельно из каждого сегмента.
        """
        centered: RealArray = values - np.mean(values)
        frequencies: RealArray
        density: RealArray
        frequencies, density = signal.welch(
            centered,
            fs=fs,
            window=window,
            nperseg=length,
            noverlap=overlap,
            detrend=False,
            return_onesided=False,
            scaling="density",
        )
        return np.fft.fftshift(2 * np.pi * frequencies / fs), np.fft.fftshift(density)

    def power_db(values: RealArray) -> RealArray:
        """Перевести мощность в дБ для пп. 3.4, 4.2 и 6.

        Args:
            values: Неотрицательная СПМ или квадрат модуля АЧХ.
        Returns:
            10 log10(values); уровень ограничен снизу -160 дБ для отображения.
        """
        result: RealArray = 10 * np.log10(np.maximum(values, 1e-16))
        return result

    def spectral_metrics(
        omega_p: RealArray,
        psd_p: RealArray,
        omega_w: RealArray,
        psd_w: RealArray,
        theory_w: RealArray,
        band_edge: float,
    ) -> tuple[float, float, float]:
        """Оценить разброс СПМ и ошибку относительно теории по пп. 3.3, 4.2.

        Args:
            omega_p: Частотная сетка периодограммы.
            psd_p: Периодограмма.
            omega_w: Частотная сетка Уэлча.
            psd_w: Оценка Уэлча.
            theory_w: Теоретическая СПМ на сетке Уэлча.
            band_edge: Граница полосы пропускания в рад/отсчёт.
        Returns:
            Дисперсии по частотным отсчётам обеих оценок в линейном масштабе
            и RMSE в дБ относительно теории. DC исключён: среднее удалено.
            Это частотный разброс одной реализации, не ансамблевая дисперсия.
        """
        mask_p: NDArray[np.bool_] = (np.abs(omega_p) <= band_edge) & (omega_p != 0)
        mask_w: NDArray[np.bool_] = (np.abs(omega_w) <= band_edge) & (omega_w != 0)
        variance_p: float = float(np.var(psd_p[mask_p]))
        variance_w: float = float(np.var(psd_w[mask_w]))
        error: RealArray = power_db(psd_w[mask_w]) - power_db(theory_w[mask_w])
        rmse_db: float = float(np.sqrt(np.mean(error**2)))
        return variance_p, variance_w, rmse_db

    return (
        acf_direct,
        acf_fft,
        periodogram,
        power_db,
        spectral_metrics,
        welch_density,
        white_noise,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. Генерация сигналов и фильтр
    Исходный шум нормируется по выборочной дисперсии (`ddof=0`), как требует
    структура работы. Эта нормировка вводит слабую зависимость отсчётов;
    теоретический эталон — идеальный БГШ с $\sigma_x^2=1$.
    Используется причинный `lfilter` с нулевым начальным состоянием.
    Начальный переходный процесс сохранён; для длинной реализации его вклад мал.

    В `butter` частота задаётся долей частоты Найквиста: $W_n=\omega_c/\pi=0.2$.
    Для спектров используется двусторонняя нормировка:
    $$R[0]=\frac{1}{2\pi}\int_{-\pi}^{\pi}S(\omega)\,d\omega.$$
    Поэтому теоретическая СПМ равна $|H|^2$, без множителя 2.
    Сетка БПФ $[-\pi,\pi)$ эквивалентна заданному интервалу $(-\pi,\pi]$.
    """)


@app.cell
def _(cutoff, filter_order, sample_count, seed, white_noise):
    x: RealArray = white_noise(sample_count, seed)
    b: RealArray
    a: RealArray
    b, a = signal.butter(filter_order, cutoff / np.pi, btype="low")
    y: RealArray = signal.lfilter(b, a, x)
    response_omega: RealArray
    response: ComplexArray
    response_omega, response = signal.freqz(b, a, worN=2048)
    signal_statistics: list[Row] = [
        {"Сигнал": "x", "Среднее": float(np.mean(x)), "Дисперсия": float(np.var(x))},
        {"Сигнал": "y", "Среднее": float(np.mean(y)), "Дисперсия": float(np.var(y))},
    ]
    mo.vstack(
        [
            mo.md("Коэффициенты фильтра и выборочные характеристики:"),
            mo.as_html({"b": b, "a": a, "x[:10]": x[:10], "y[:10]": y[:10]}),
            mo.ui.table(signal_statistics, selection=None),
        ]
    )
    return a, b, response, response_omega, signal_statistics, x, y


@app.cell
def _(cutoff, response, response_omega, sample_count, x, y):
    _fig: Figure
    _axes: NDArray[np.object_]
    _fig, _axes = plt.subplots(3, 1, figsize=(10, 8), layout="constrained")
    _index: IntArray = np.arange(sample_count, dtype=np.int64)
    _axes[0].plot(_index, x, linewidth=0.5, label="x[n], БГШ")
    _axes[1].plot(_index, y, linewidth=0.5, label="y[n], окрашенный шум")
    _axis: Axes
    for _axis in _axes[:2]:
        _axis.set(xlabel="n, отсчёт", ylabel="Амплитуда, усл. ед.")
        _axis.legend()
    _axes[0].set_title("Временные реализации, все 8192 отсчёта")
    _axes[2].plot(response_omega / np.pi, np.abs(response), label="Баттерворт")
    _axes[2].axvline(
        cutoff / np.pi, color="gray", linestyle="--", label="Частота среза"
    )
    _axes[2].set(xlabel="ω/π, рад/отсчёт / π", ylabel="|H|", title="АЧХ ФНЧ")
    _axes[2].legend()
    plt.show()
    mo.output.append(_fig)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. АКФ и ВКФ
    Для центрированного $u[n]$ используется смещённая оценка
    $$\widehat R_{uu}[m]=\frac1N\sum_n u[n+m]u[n],$$
    где вне реализации $u[n]=0$. Прямой метод и БПФ имеют одинаковую нормировку.
    Дополнение БПФ до $2N-1$ обязательно для линейной корреляции.

    Соглашение ВКФ: $R_{xy}[m]=E\{x[n]y[n+m]\}$.
    Поэтому для белого входа $R_{xy}[m]=\sigma_x^2h[m]$,
    а в NumPy нужно вызвать `correlate(y, x)`. Положительный лаг означает
    запаздывание выхода. Для сравнения ширины АКФ дополнительно нормируются на $R[0]$.
    """)


@app.cell
def _(a, acf_direct, acf_fft, b, sample_count, x, y):
    acf_x_direct: RealArray = acf_direct(x)
    acf_y_direct: RealArray = acf_direct(y)
    acf_x: RealArray = acf_fft(x)
    acf_y: RealArray = acf_fft(y)
    acf_error_x: float = float(np.max(np.abs(acf_x_direct - acf_x)))
    acf_error_y: float = float(np.max(np.abs(acf_y_direct - acf_y)))
    lags: IntArray = np.arange(1 - sample_count, sample_count, dtype=np.int64)
    cross_xy: RealArray = (
        np.correlate(y - np.mean(y), x - np.mean(x), "full") / sample_count
    )
    impulse: RealArray = np.zeros(128)
    impulse[0] = 1.0
    impulse_response: RealArray = signal.lfilter(b, a, impulse)
    cross_peak: int = int(lags[np.argmax(cross_xy)])
    half_width_x: int = int(
        np.flatnonzero(acf_x[sample_count - 1 :] / acf_x[sample_count - 1] <= 0.5)[0]
    )
    half_width_y: int = int(
        np.flatnonzero(acf_y[sample_count - 1 :] / acf_y[sample_count - 1] <= 0.5)[0]
    )
    mo.ui.table(
        [
            {
                "Сигнал": "x",
                "max |АКФ direct − FFT|": acf_error_x,
                "Первый лаг R/R0 ≤ 0.5": half_width_x,
            },
            {
                "Сигнал": "y",
                "max |АКФ direct − FFT|": acf_error_y,
                "Первый лаг R/R0 ≤ 0.5": half_width_y,
            },
        ],
        selection=None,
    )
    return (
        acf_error_x,
        acf_error_y,
        acf_x,
        acf_x_direct,
        acf_y,
        acf_y_direct,
        cross_peak,
        cross_xy,
        half_width_x,
        half_width_y,
        impulse_response,
        lags,
    )


@app.cell
def _(
    acf_x,
    acf_x_direct,
    acf_y,
    acf_y_direct,
    cross_xy,
    impulse_response,
    lags,
    sample_count,
):
    _fig: Figure
    _axes: NDArray[np.object_]
    _fig, _axes = plt.subplots(4, 1, figsize=(10, 12), layout="constrained")
    _mask: NDArray[np.bool_] = np.abs(lags) <= 40
    _axes[0].plot(lags[_mask], acf_x_direct[_mask], label="x: прямая")
    _axes[0].plot(lags[_mask], acf_x[_mask], ".", label="x: БПФ")
    _axes[0].plot(lags[_mask], acf_y_direct[_mask], label="y: прямая")
    _axes[0].plot(lags[_mask], acf_y[_mask], ".", label="y: БПФ")
    _axes[0].set(title="Совпадение реализаций АКФ", ylabel="R, усл. ед.²")
    _axes[1].plot(lags[_mask], acf_x[_mask], label="Rxx")
    _axes[1].plot(lags[_mask], acf_y[_mask], label="Ryy")
    _axes[1].set(title="АКФ входа и выхода", ylabel="R, усл. ед.²")
    _axes[2].plot(
        lags[_mask], acf_x[_mask] / acf_x[sample_count - 1], label="Rxx/Rxx[0]"
    )
    _axes[2].plot(
        lags[_mask], acf_y[_mask] / acf_y[sample_count - 1], label="Ryy/Ryy[0]"
    )
    _axes[2].set(title="Сравнение ширины АКФ", ylabel="Нормированная АКФ")
    _axes[3].plot(lags[_mask], cross_xy[_mask], label="Rxy, оценка")
    _axes[3].plot(
        np.arange(41), impulse_response[:41], "--", label="h[m], теория при σ²=1"
    )
    _axes[3].set(
        title="ВКФ вход–выход и импульсная характеристика", ylabel="Rxy, усл. ед.²"
    )
    for _axis in _axes:
        _axis.set_xlabel("m, отсчёт")
        _axis.legend()
        _axis.grid(alpha=0.25)
    plt.show()
    mo.output.append(_fig)


@app.cell(hide_code=True)
def _(cross_peak, half_width_x, half_width_y):
    mo.md(f"""
    Ширина до первого пересечения уровня 0,5: x — **{half_width_x}**, y —
    **{half_width_y}** отсчёта. ФНЧ подавляет быстрые изменения, соседние
    выходные отсчёты становятся коррелированными, АКФ расширяется.
    Максимум ВКФ при лаге **{cross_peak}** соответствует максимуму импульсного
    отклика причинного фильтра. Малые расхождения с h[m] связаны с конечной
    длиной реализации, центрированием и начальным переходным процессом.
    """)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3–4. СПМ и сравнение с теорией
    Периодограмма: $\widehat S(\omega_k)=|\mathrm{FFT}(u-\bar u)_k|^2/N$.
    В формулировке п. 3.1 порядок операций описан неоднозначно; СПМ вычисляется
    как **квадрат модуля БПФ**, а не БПФ квадрата сигнала.
    Метод Уэлча усредняет оконные периодограммы с нормировкой на энергию окна.
    Теория п. 4.2: $S_{yy}(\omega)=|H(e^{j\omega})|^2\sigma_x^2$.

    В таблице дисперсия считается **по частотным отсчётам в полосе**
    $0<|\omega|\le\omega_c$, в линейном масштабе. Это требуемая описательная
    характеристика при K=1; она включает наклон СПМ и не равна ансамблевой
    дисперсии оценки. DC исключён из метрик из-за удаления среднего.
    RMSE в дБ вычисляется на собственной сетке Уэлча.
    """)


@app.cell
def _(
    a,
    b,
    cutoff,
    overlap_count,
    periodogram,
    sampling_rate,
    segment_length,
    spectral_metrics,
    welch_density,
    window_name,
    x,
    y,
):
    omega_p: RealArray
    psd_x_p: RealArray
    omega_p, psd_x_p = periodogram(x, sampling_rate)
    psd_y_p: RealArray
    _, psd_y_p = periodogram(y, sampling_rate)
    omega_w: RealArray
    psd_x_w: RealArray
    omega_w, psd_x_w = welch_density(
        x, sampling_rate, segment_length, overlap_count, window_name
    )
    psd_y_w: RealArray
    _, psd_y_w = welch_density(
        y, sampling_rate, segment_length, overlap_count, window_name
    )
    theory_response: ComplexArray
    _, theory_response = signal.freqz(b, a, worN=omega_w)
    theory_y: RealArray = np.abs(theory_response) ** 2
    variance_p: float
    variance_w: float
    rmse_db: float
    variance_p, variance_w, rmse_db = spectral_metrics(
        omega_p, psd_y_p, omega_w, psd_y_w, theory_y, cutoff
    )
    x_metrics: tuple[float, float, float] = spectral_metrics(
        omega_p, psd_x_p, omega_w, psd_x_w, np.ones_like(omega_w), cutoff
    )
    psd_rows: list[Row] = [
        {
            "Сигнал": "x",
            "Var периодограммы": x_metrics[0],
            "Var Уэлча": x_metrics[1],
            "RMSE Уэлча, дБ": x_metrics[2],
        },
        {
            "Сигнал": "y",
            "Var периодограммы": variance_p,
            "Var Уэлча": variance_w,
            "RMSE Уэлча, дБ": rmse_db,
        },
    ]
    mo.ui.table(psd_rows, selection=None)
    return (
        omega_p,
        omega_w,
        psd_x_p,
        psd_x_w,
        psd_y_p,
        psd_y_w,
        rmse_db,
        theory_y,
        variance_p,
        variance_w,
    )


@app.cell
def _(omega_p, omega_w, power_db, psd_x_p, psd_x_w, psd_y_p, psd_y_w, theory_y):
    _fig: Figure
    _axes: NDArray[np.object_]
    _fig, _axes = plt.subplots(3, 1, figsize=(10, 10), layout="constrained")
    _axes[0].plot(
        omega_p / np.pi, power_db(psd_x_p), alpha=0.6, label="Периодограмма x"
    )
    _axes[0].plot(omega_w / np.pi, power_db(psd_x_w), label="Уэлч x")
    _axes[0].set_title("СПМ белого шума")
    _axes[1].plot(
        omega_p / np.pi, power_db(psd_y_p), alpha=0.6, label="Периодограмма y"
    )
    _axes[1].plot(omega_w / np.pi, power_db(psd_y_w), label="Уэлч y")
    _axes[1].set_title("СПМ окрашенного шума")
    _axes[2].plot(omega_w / np.pi, power_db(theory_y), label="Теория |H|²")
    _axes[2].plot(omega_w / np.pi, power_db(psd_y_w), label="Уэлч y")
    _axes[2].set_title("Сравнение с теоретической СПМ")
    for _axis in _axes:
        _axis.set(xlabel="ω/π, рад/отсчёт / π", ylabel="СПМ, дБ", ylim=(-100, 15))
        _axis.legend()
        _axis.grid(alpha=0.25)
    plt.show()
    mo.output.append(_fig)


@app.cell
def _(
    overlap_count,
    sample_count,
    sampling_rate,
    seed,
    segment_length,
    white_noise,
    window_name,
):
    noise_u: RealArray = white_noise(sample_count, seed + 1)
    noise_v: RealArray = white_noise(sample_count, seed + 2)
    csd_frequency: RealArray
    cross_spectrum: ComplexArray
    csd_frequency, cross_spectrum = signal.csd(
        noise_u,
        noise_v,
        fs=sampling_rate,
        window=window_name,
        nperseg=segment_length,
        noverlap=overlap_count,
        return_onesided=False,
        scaling="density",
        detrend=False,
    )
    coherence_frequency: RealArray
    coherence_uv: RealArray
    coherence_frequency, coherence_uv = signal.coherence(
        noise_u,
        noise_v,
        fs=sampling_rate,
        window=window_name,
        nperseg=segment_length,
        noverlap=overlap_count,
        detrend=False,
    )
    coherence_mean: float = float(np.mean(coherence_uv))
    coherence_max: float = float(np.max(coherence_uv))
    _fig: Figure
    _axes: NDArray[np.object_]
    _fig, _axes = plt.subplots(2, 1, figsize=(10, 6), layout="constrained")
    _axes[0].plot(
        np.fft.fftshift(csd_frequency) * 2 / sampling_rate,
        np.abs(np.fft.fftshift(cross_spectrum)),
        label="|Suv|, Уэлч",
    )
    _axes[0].set(title="Взаимный спектр независимых БГШ", ylabel="|Suv|, усл. ед.²")
    _axes[1].plot(
        coherence_frequency * 2 / sampling_rate, coherence_uv, label="Когерентность Cuv"
    )
    _axes[1].set(title="Когерентность независимых БГШ", ylabel="Cuv", ylim=(0, 1))
    for _axis in _axes:
        _axis.set_xlabel("ω/π, рад/отсчёт / π")
        _axis.legend()
    plt.show()
    mo.output.append(_fig)
    return coherence_max, coherence_mean, coherence_uv, cross_spectrum


@app.cell(hide_code=True)
def _(coherence_max, coherence_mean, rmse_db, segment_count, variance_p, variance_w):
    mo.md(f"""
    RMSE Уэлча относительно теории в полосе: **{rmse_db:.3f} дБ**.
    Частотный разброс периодограммы / Уэлча: **{variance_p / variance_w:.2f}**.
    Усреднение {segment_count} перекрывающихся сегментов сглаживает спектр ценой
    разрешения. Перекрывающиеся сегменты не полностью независимы.

    Средняя когерентность независимых БГШ **{coherence_mean:.4f}**, максимум
    **{coherence_max:.4f}**. Теоретически она нулевая, но оценка по конечному
    числу сегментов положительна и флуктуирует. Отдельный ненулевой пик
    не доказывает связь. Нулевая когерентность означает отсутствие линейной
    связи на частоте, а не отсутствие любой нелинейной зависимости.
    """)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Влияние длинной реализации
    Сравним первые 1024 отсчёта с полной реализацией, сохраняя **L=1024**,
    окно и перекрытие. Для короткой записи получается один сегмент — это
    диагностическое сравнение, оно не удовлетворяет требованию ≥8 сегментов
    и не заменяет основной расчёт. Длинная запись даёт 15 сегментов.
    Изменение N улучшает разрешение периодограммы ($2\pi/N$), но само по себе
    не устраняет её случайный разброс. Разрешение Уэлча определяется L;
    увеличение N при фиксированном L позволяет усреднить больше сегментов.
    """)


@app.cell
def _(
    a,
    b,
    cutoff,
    overlap_count,
    periodogram,
    sample_count,
    sampling_rate,
    segment_length,
    spectral_metrics,
    welch_density,
    window_name,
    y,
):
    length_rows: list[Row] = []
    _length: int
    for _length in (1024, sample_count):
        _wp: RealArray
        _pp: RealArray
        _ww: RealArray
        _pw: RealArray
        _h: ComplexArray
        _wp, _pp = periodogram(y[:_length], sampling_rate)
        _ww, _pw = welch_density(
            y[:_length], sampling_rate, segment_length, overlap_count, window_name
        )
        _, _h = signal.freqz(b, a, worN=_ww)
        _metrics: tuple[float, float, float] = spectral_metrics(
            _wp, _pp, _ww, _pw, np.abs(_h) ** 2, cutoff
        )
        length_rows.append(
            {
                "N": _length,
                "L": segment_length,
                "Сегментов": 1
                + (_length - segment_length) // (segment_length - overlap_count),
                "Δω периодограммы": 2 * np.pi / _length,
                "Var периодограммы": _metrics[0],
                "Var Уэлча": _metrics[1],
                "RMSE Уэлча, дБ": _metrics[2],
            }
        )
    mo.ui.table(length_rows, selection=None)
    return (length_rows,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 6*. Альтернативный фильтр Чебышёва I рода
    Порядок и параметр частоты границы полосы те же, что у Баттерворта.
    Дополнительный выбранный параметр — пульсации **1 дБ**.
    У Чебышёва заданная граница соответствует уровню −1 дБ, у Баттерворта
    −3,01 дБ: одинаковый Wn не означает одинаковую частоту уровня −3 дБ.
    Оба фильтра обрабатывают одну реализацию x с нулевым начальным состоянием.
    """)


@app.cell
def _(
    acf_fft,
    cutoff,
    filter_order,
    omega_p,
    omega_w,
    overlap_count,
    periodogram,
    sampling_rate,
    segment_length,
    spectral_metrics,
    welch_density,
    window_name,
    x,
):
    ripple_db: float = 1.0
    b_alt: RealArray
    a_alt: RealArray
    b_alt, a_alt = signal.cheby1(filter_order, ripple_db, cutoff / np.pi, btype="low")
    y_alt: RealArray = signal.lfilter(b_alt, a_alt, x)
    acf_alt: RealArray = acf_fft(y_alt)
    psd_alt_p: RealArray
    psd_alt_w: RealArray
    _, psd_alt_p = periodogram(y_alt, sampling_rate)
    _, psd_alt_w = welch_density(
        y_alt, sampling_rate, segment_length, overlap_count, window_name
    )
    response_alt: ComplexArray
    _, response_alt = signal.freqz(b_alt, a_alt, worN=omega_w)
    theory_alt: RealArray = np.abs(response_alt) ** 2
    alt_metrics: tuple[float, float, float] = spectral_metrics(
        omega_p, psd_alt_p, omega_w, psd_alt_w, theory_alt, cutoff
    )
    mo.as_html({"b Чебышёва": b_alt, "a Чебышёва": a_alt, "y_alt[:10]": y_alt[:10]})
    return acf_alt, alt_metrics, psd_alt_p, psd_alt_w, response_alt, theory_alt, y_alt


@app.cell
def _(
    acf_alt,
    acf_y,
    alt_metrics,
    lags,
    omega_p,
    omega_w,
    power_db,
    psd_alt_p,
    psd_alt_w,
    psd_y_w,
    rmse_db,
    theory_alt,
    theory_y,
    variance_p,
    variance_w,
):
    _fig: Figure
    _axes: NDArray[np.object_]
    _fig, _axes = plt.subplots(4, 1, figsize=(10, 12), layout="constrained")
    _axes[0].plot(omega_w / np.pi, power_db(theory_y), label="Баттерворт")
    _axes[0].plot(omega_w / np.pi, power_db(theory_alt), label="Чебышёв I")
    _axes[0].set(title="АЧХ фильтров, 20 log10|H|", ylabel="АЧХ, дБ", ylim=(-80, 5))
    _mask: NDArray[np.bool_] = np.abs(lags) <= 40
    _axes[1].plot(lags[_mask], acf_y[_mask], label="Баттерворт")
    _axes[1].plot(lags[_mask], acf_alt[_mask], label="Чебышёв I")
    _axes[1].set(
        title="АКФ выходных сигналов", xlabel="m, отсчёт", ylabel="Ryy, усл. ед.²"
    )
    _axes[2].plot(omega_w / np.pi, power_db(psd_y_w), label="Уэлч, Баттерворт")
    _axes[2].plot(omega_w / np.pi, power_db(psd_alt_w), label="Уэлч, Чебышёв I")
    _axes[2].set(title="Сравнение СПМ Уэлча", ylabel="СПМ, дБ", ylim=(-100, 10))
    _axes[3].plot(
        omega_p / np.pi, power_db(psd_alt_p), alpha=0.5, label="Периодограмма"
    )
    _axes[3].plot(omega_w / np.pi, power_db(psd_alt_w), label="Уэлч")
    _axes[3].plot(
        omega_w / np.pi, power_db(theory_alt), "--", label="Теория, Чебышёв I"
    )
    _axes[3].set(
        title="Оценки и теория СПМ для Чебышёва I", ylabel="СПМ, дБ", ylim=(-100, 10)
    )
    for _axis in (_axes[0], _axes[2], _axes[3]):
        _axis.set_xlabel("ω/π, рад/отсчёт / π")
    for _axis in _axes:
        _axis.legend()
        _axis.grid(alpha=0.25)
    plt.show()
    _comparison_table: mo.ui.table = mo.ui.table(
        [
            {
                "Фильтр": "Баттерворт",
                "Var периодограммы": variance_p,
                "Var Уэлча": variance_w,
                "RMSE, дБ": rmse_db,
            },
            {
                "Фильтр": "Чебышёв I",
                "Var периодограммы": alt_metrics[0],
                "Var Уэлча": alt_metrics[1],
                "RMSE, дБ": alt_metrics[2],
            },
        ],
        selection=None,
    )
    mo.vstack([_fig, _comparison_table])


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    У Баттерворта АЧХ монотонная и максимально плоская у нуля; у Чебышёва I
    пульсации до 1 дБ переходят в пульсации теоретической СПМ. Они изменяют
    АКФ и дисперсию выхода. Чебышёв даёт более резкий переход относительно
    своей границы полосы; при выбранных 1 дБ его подавление не обязано быть
    сильнее Баттерворта в каждой отдельной частотной точке.
    Большая дисперсия по частотам может быть следствием формы АЧХ,
    поэтому сама по себе не означает худшую статистическую оценку.

    Оба выбранных БИХ-фильтра имеют нелинейную фазу. Фаза определяет ВКФ
    и задержку выхода; теоретическая АКФ выхода определяется только $|H|^2$.

    ## Контрольные вопросы
    **1. Почему периодограмма несостоятельна?** Увеличение N сужает спектральное
    окно и повышает разрешение, но без усреднения дисперсия оценки в фиксированной
    частоте не стремится к нулю (вне особых частот она порядка квадрата СПМ).

    **2. Смысл когерентности.** $C_{uv}=|S_{uv}|^2/(S_{uu}S_{vv})\in[0,1]$
    характеризует силу линейной связи на частоте. Для независимых шумов теория
    даёт нуль, конечная оценка имеет положительное смещение.

    **3. Ширина АКФ и полоса.** АКФ и СПМ — преобразования Фурье друг друга.
    Узкая полоса обычно даёт более широкую АКФ и более длительную корреляцию.
    Точная форма определяется всей СПМ, включая переходную полосу.

    **4. Максимум ВКФ.** При принятом соглашении ВКФ белого входа с выходом
    равна $\sigma_x^2h[m]$. Положение максимума зависит от импульсного отклика,
    задержки и фазы фильтра; у фильтра без задержки пик может быть в нуле.
    Перестановка аргументов корреляции меняет знак лага.

    **5. Преимущества Уэлча.** Усреднение снижает разброс, окно уменьшает боковые
    лепестки и утечку, перекрытие использует данные полнее. Цена — более широкая
    главная полоса спектрального окна и худшее разрешение, чем у полной записи.
    Для состоятельности нужны рост длины сегмента и числа усреднений.

    **6. Пульсации АЧХ.** Для белого входа $S_{yy}=|H|^2$; пульсации усиления
    становятся пульсациями СПМ. В дБ $10\log_{10}S_{yy}=20\log_{10}|H|$
    при единичной входной СПМ.

    **7. Линейная фаза КИХ.** Постоянная групповая задержка сохраняет взаимное
    расположение спектральных составляющих с учётом общей задержки.
    Амплитудная фильтрация всё равно меняет сигнал. Общий сдвиг не меняет АКФ;
    форма АКФ выхода определяется $|H|^2$, а не фазой.

    **8. Выбор фильтра.** Баттерворт удобен для гладкой полосы без пульсаций;
    Чебышёв — для более узкой переходной области при допустимых пульсациях;
    эллиптический — для строгого подавления при малом порядке и допустимых
    пульсациях в обеих полосах; линейно-фазовый КИХ — когда важна постоянная
    групповая задержка, обычно ценой большего порядка.
    """)


@app.cell
def _(a, acf_x, acf_y, b, omega_w, psd_x_p, psd_y_p, sample_count, x, y):
    parseval_error_x: float = abs(float(np.mean(psd_x_p)) - float(np.var(x)))
    parseval_error_y: float = abs(float(np.mean(psd_y_p)) - float(np.var(y)))
    acf_zero_error_x: float = abs(float(acf_x[sample_count - 1]) - float(np.var(x)))
    acf_zero_error_y: float = abs(float(acf_y[sample_count - 1]) - float(np.var(y)))
    _poles: ComplexArray = np.asarray(np.roots(a), dtype=np.complex128)
    pole_radius: float = float(np.max(np.abs(_poles)))
    _h_dense: ComplexArray
    _, _h_dense = signal.freqz(b, a, worN=sample_count, whole=True)
    theoretical_variance_y: float = float(np.mean(np.abs(_h_dense) ** 2))
    mo.ui.table(
        [
            {"Проверка": "Парсеваль x", "Значение": parseval_error_x},
            {"Проверка": "Парсеваль y", "Значение": parseval_error_y},
            {"Проверка": "АКФ x[0] − Var(x), модуль", "Значение": acf_zero_error_x},
            {"Проверка": "АКФ y[0] − Var(y), модуль", "Значение": acf_zero_error_y},
            {"Проверка": "Максимальный модуль полюса (<1)", "Значение": pole_radius},
            {
                "Проверка": "Теоретическая дисперсия выхода",
                "Значение": theoretical_variance_y,
            },
            {"Проверка": "Выборочная дисперсия выхода", "Значение": float(np.var(y))},
            {
                "Проверка": "Δω Уэлча, рад/отсчёт",
                "Значение": float(omega_w[1] - omega_w[0]),
            },
        ],
        selection=None,
    )
    return parseval_error_x, parseval_error_y, pole_radius, theoretical_variance_y


@app.cell(hide_code=True)
def _(
    acf_error_x,
    acf_error_y,
    rmse_db,
    segment_count,
    theoretical_variance_y,
    variance_p,
    variance_w,
    y,
):
    mo.md(f"""
    ## Итоги вычислений
    Максимальные расхождения двух алгоритмов АКФ: **{acf_error_x:.2e}** для x
    и **{acf_error_y:.2e}** для y. ФНЧ расширяет АКФ и уменьшает дисперсию:
    получено **{float(np.var(y)):.5f}**, теоретически **{theoretical_variance_y:.5f}**.
    Оценка Уэлча по {segment_count} сегментам имеет меньший частотный разброс:
    **{variance_w:.5f}** против **{variance_p:.5f}** у периодограммы;
    RMSE относительно теории **{rmse_db:.3f} дБ**.
    Длинная реализация увеличивает число усреднений при фиксированном L.
    Сравнение с Чебышёвым показывает влияние формы АЧХ на СПМ и АКФ.
    Все результаты и графики отображаются в ноутбуке без экспорта файлов.
    """)


if __name__ == "__main__":
    app.run()
