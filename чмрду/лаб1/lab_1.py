import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    from math import isfinite, log10
    from typing import Literal

    import marimo as mo
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    type Method = Literal["rk3", "ab1", "am3"]
    type History = list[tuple[int, float, float, float | None]]
    type GridResult = tuple[History, list[float], list[float]]
    type TableRow = dict[str, str | int | float | None]


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Лабораторная работа №1. Вариант 7
    **Численное решение обыкновенных дифференциальных уравнений**

    По табл. 1.1: $u'=f(t,u)=u+tu^2$, $u(0)=1$, $t\in[0;0{,}8]$.
    По табл. 1.2 используются:

    | Метод | Параметр методички | Порядок $p$ | Формула |
    |---|---|---|---|
    | Рунге–Кутта | $m=3$ | 3 | (1.8) |
    | Явный Адамс (Эйлер) | $m=1$ | 1 | (1.11) |
    | Неявный Адамс | $m=2$ | 3 | (1.17) |

    Начальная сетка: $N=10$, $\tau=T/N$. Далее $N$ удваивается.
    Условие остановки (1.19):
    $$R=\frac{|y_{\tau}(0{,}8)-y_{\tau/2}(0{,}8)|}
    {(2^p-1)|y_{\tau/2}(0{,}8)|}\le 10^{-3}.$$
    Неявное уравнение решается методом Ньютона (1.21), начиная с $y_n$,
    до относительного изменения менее $10^{-4}$.
    Первое значение для двухшагового Адамса вычисляется методом Рунге–Кутта (1.8).
    """)


@app.cell
def _():
    t_start: float = 0.0
    duration: float = 0.8
    u_initial: float = 1.0
    initial_intervals: int = 10
    runge_tolerance: float = 1e-3
    newton_tolerance: float = 1e-4
    methods: list[tuple[Method, str, int]] = [
        ("rk3", "Рунге–Кутта, m=3", 3),
        ("ab1", "Явный Адамс, m=1", 1),
        ("am3", "Неявный Адамс, m=2", 3),
    ]
    return (
        duration,
        initial_intervals,
        methods,
        newton_tolerance,
        runge_tolerance,
        t_start,
        u_initial,
    )


@app.cell
def _():
    def rhs(t: float, u: float) -> float:
        """Вычислить f(t,u) из табл. 1.1, вариант 7, для задачи (1.1).

        Args:
            t: Независимая переменная.
            u: Значение решения.
        Returns:
            Правая часть u + t*u**2.
        """
        return u + t * u * u

    def rhs_derivative(t: float, u: float) -> float:
        """Вычислить производную f по u для Ньютона (1.21).

        Args:
            t: Независимая переменная.
            u: Значение решения.
        Returns:
            Производная 1 + 2*t*u функции из табл. 1.1, вариант 7.
        """
        return 1.0 + 2.0 * t * u

    def exact_solution(t: float) -> float:
        """Вычислить аналитическое решение варианта 7 для контроля.

        Подстановка v=1/u даёт v'+v=-t, v(0)=1, откуда v=1-t.
        Формула выведена самостоятельно, в методичке не пронумерована.

        Args:
            t: Точка интервала [0; 0.8].
        Returns:
            Точное решение 1/(1-t), удовлетворяющее (1.1).
        """
        return 1.0 / (1.0 - t)

    return exact_solution, rhs, rhs_derivative


@app.cell
def _(rhs, rhs_derivative):
    def rk3_step(t: float, u: float, step: float) -> float:
        """Выполнить шаг Рунге–Кутты третьего порядка по (1.8).

        Args:
            t: Время в текущем узле.
            u: Значение в текущем узле.
            step: Шаг сетки.
        Returns:
            Приближение в следующем узле.
        """
        k1: float = rhs(t, u)
        k2: float = rhs(t + step / 2.0, u + step * k1 / 2.0)
        k3: float = rhs(t + step, u - step * k1 + 2.0 * step * k2)
        return u + step * (k1 + 4.0 * k2 + k3) / 6.0

    def implicit_adams_step(
        t: float,
        u: float,
        previous_t: float,
        previous_u: float,
        step: float,
        tolerance: float,
    ) -> float:
        """Решить (1.17) методом Ньютона (1.21).

        Для (1.17): F(x)=x-u-step*(5*f(t+step,x)+8*f(t,u)
        -f(previous_t,previous_u))/12. Производная берётся по x.
        Начальное приближение и критерий остановки соответствуют разделу 6.

        Args:
            t: Текущий момент времени.
            u: Текущее решение.
            previous_t: Предыдущий момент времени.
            previous_u: Предыдущее решение.
            step: Шаг сетки.
            tolerance: Допуск относительного изменения итераций Ньютона.
        Returns:
            Значение решения в момент t+step.
        """
        next_t: float = t + step
        known_part: float = 8.0 * rhs(t, u) - rhs(previous_t, previous_u)
        estimate: float = u
        for _ in range(100):
            residual: float = (
                estimate - u - step * (5.0 * rhs(next_t, estimate) + known_part) / 12.0
            )
            derivative: float = (
                1.0 - 5.0 * step * rhs_derivative(next_t, estimate) / 12.0
            )
            if abs(derivative) < 1e-14:
                raise ArithmeticError("Нулевая производная в методе Ньютона")
            updated: float = estimate - residual / derivative
            if not isfinite(updated):
                raise ArithmeticError("Нечисловое приближение Ньютона")
            if abs(updated - estimate) <= tolerance * abs(updated):
                return updated
            estimate = updated
        raise ArithmeticError("Метод Ньютона не сошёлся за 100 итераций")

    return implicit_adams_step, rk3_step


@app.cell
def _(implicit_adams_step, rhs, rk3_step):
    def solve_grid(
        method: Method,
        intervals: int,
        start: float,
        length: float,
        initial: float,
        newton_tol: float,
    ) -> tuple[list[float], list[float]]:
        """Решить задачу (1.1) методами (1.8), (1.11), (1.17).

        Для (1.17) стартовое значение y_1 вычисляется по (1.8).

        Args:
            method: Идентификатор выбранного метода.
            intervals: Число интервалов сетки.
            start: Начало интервала.
            length: Длина интервала интегрирования.
            initial: Начальное значение решения.
            newton_tol: Допуск метода Ньютона.
        Returns:
            Узлы сетки и численное решение, включая начальный узел.
        """
        if intervals < 2:
            raise ValueError("Необходимо минимум два интервала")
        step: float = length / intervals
        times: list[float] = [start + index * step for index in range(intervals + 1)]
        values: list[float] = [initial]
        for index in range(intervals):
            t: float = times[index]
            u: float = values[index]
            next_u: float
            if method == "rk3" or (method == "am3" and index == 0):
                next_u = rk3_step(t, u, step)
            elif method == "ab1":
                next_u = u + step * rhs(t, u)
            elif method == "am3":
                next_u = implicit_adams_step(
                    t, u, times[index - 1], values[index - 1], step, newton_tol
                )
            else:
                raise ValueError("Неизвестный метод")
            if not isfinite(next_u):
                raise ArithmeticError("Получено нечисловое решение")
            values.append(next_u)
        return times, values

    return (solve_grid,)


@app.cell
def _(solve_grid):
    def refine_grid(
        method: Method,
        order: int,
        start: float,
        length: float,
        initial: float,
        first_intervals: int,
        tolerance: float,
        newton_tol: float,
    ) -> GridResult:
        """Удваивать число интервалов до первого выполнения (1.19).

        Args:
            method: Численный метод.
            order: Теоретический порядок точности.
            start: Начало интервала.
            length: Длина интервала.
            initial: Начальное значение.
            first_intervals: Число интервалов первой сетки.
            tolerance: Требуемая относительная точность.
            newton_tol: Допуск Ньютона.
        Returns:
            История (N, шаг, конечное значение, оценка Рунге), последняя
            сетка и решение на ней. Для первой сетки оценка отсутствует.
        """
        intervals: int = first_intervals
        times: list[float]
        values: list[float]
        times, values = solve_grid(
            method, intervals, start, length, initial, newton_tol
        )
        history: History = [(intervals, length / intervals, values[-1], None)]
        previous: float = values[-1]
        for _ in range(18):
            intervals *= 2
            times, values = solve_grid(
                method, intervals, start, length, initial, newton_tol
            )
            if values[-1] == 0.0:
                raise ArithmeticError("Нулевой знаменатель в критерии (1.19)")
            runge_error: float = abs(previous - values[-1]) / (
                (2**order - 1) * abs(values[-1])
            )
            history.append((intervals, length / intervals, values[-1], runge_error))
            if runge_error <= tolerance:
                return history, times, values
            previous = values[-1]
        raise ArithmeticError("Критерий Рунге не выполнен за 18 сгущений")

    def fit_order(history: History) -> tuple[list[float], list[float], float, float]:
        """Оценить порядок по (1.20), погрешности взять из п. 3 задания.

        Args:
            history: Последовательность рассчитанных сеток.
        Returns:
            lg(шага), lg(относительной ошибки), наклон p и свободный член B.
            Эталон y_* — конечное значение на последней сетке этого метода.
        """
        reference: float = history[-1][2]
        if len(history) < 3 or reference == 0.0:
            raise ValueError("Для МНК нужны две точки и ненулевой эталон")
        xs: list[float] = [log10(row[1]) for row in history[:-1]]
        errors: list[float] = [
            abs((row[2] - reference) / reference) for row in history[:-1]
        ]
        if any(error <= 0.0 for error in errors):
            raise ValueError("Логарифм нулевой ошибки не определён")
        zs: list[float] = [log10(error) for error in errors]
        count: int = len(xs)
        sum_x: float = sum(xs)
        sum_z: float = sum(zs)
        sum_xz: float = sum(x * z for x, z in zip(xs, zs, strict=True))
        sum_x2: float = sum(x * x for x in xs)
        slope: float = (count * sum_xz - sum_x * sum_z) / (count * sum_x2 - sum_x**2)
        intercept: float = (sum_z - slope * sum_x) / count
        return xs, zs, slope, intercept

    return fit_order, refine_grid


@app.cell
def _(
    duration,
    initial_intervals,
    methods,
    newton_tolerance,
    refine_grid,
    runge_tolerance,
    t_start,
    u_initial,
):
    solutions: dict[Method, GridResult] = {
        method: refine_grid(
            method,
            order,
            t_start,
            duration,
            u_initial,
            initial_intervals,
            runge_tolerance,
            newton_tolerance,
        )
        for method, _name, order in methods
    }
    return (solutions,)


@app.cell
def _(
    duration,
    fit_order,
    methods,
    newton_tolerance,
    solutions,
    solve_grid,
    t_start,
    u_initial,
):
    order_solutions: dict[Method, GridResult] = {}
    for _method, _name, _order in methods:
        _history, _times, _values = solutions[_method]
        _study_history: History = list(_history)
        # При L=2 для МНК есть только одна точка: нужна ещё одна сетка.
        if len(_study_history) < 3:
            _extra_n: int = 2 * _study_history[-1][0]
            _times, _values = solve_grid(
                _method, _extra_n, t_start, duration, u_initial, newton_tolerance
            )
            _extra_runge: float = abs(_study_history[-1][2] - _values[-1]) / (
                (2**_order - 1) * abs(_values[-1])
            )
            _study_history.append(
                (_extra_n, duration / _extra_n, _values[-1], _extra_runge)
            )
        order_solutions[_method] = (_study_history, _times, _values)
    regressions: dict[Method, tuple[list[float], list[float], float, float]] = {
        method: fit_order(order_solutions[method][0])
        for method, _name, _order in methods
    }
    return order_solutions, regressions


@app.cell(hide_code=True)
def _(
    exact_solution, methods, order_solutions, regressions, runge_tolerance, solutions
):
    summary_rows: list[TableRow] = []
    for _method, _name, _order in methods:
        _history, _times, _values = solutions[_method]
        _n, _step, _last, _runge = _history[-1]
        summary_rows.append(
            {
                "Метод": _name,
                "p теор.": _order,
                "Сеток L": len(_history),
                "N": _n,
                "Шаг τ": _step,
                "y(0.8)": _last,
                "Оценка Рунге": _runge,
                "p МНК": regressions[_method][2],
                "N эталона МНК": order_solutions[_method][0][-1][0],
                "Отн. ошибка к u(0.8)=5": abs(_last - exact_solution(0.8))
                / exact_solution(0.8),
            }
        )
    mo.vstack(
        [
            mo.md(
                "## Результаты\nПоследняя сетка каждого метода — первая, выполнившая критерий Рунге."
            ),
            mo.ui.table(
                summary_rows, selection=None, pagination=False, show_download=False
            ),
            mo.md(f"Требуемая оценка Рунге: **не более {runge_tolerance:g}**."),
        ]
    )
    return (summary_rows,)


@app.cell(hide_code=True)
def _(methods, solutions):
    mo.vstack(
        [
            mo.vstack(
                [
                    mo.md(f"### {_name}: последовательность сеток"),
                    mo.ui.table(
                        [
                            {
                                "N": _n,
                                "Шаг τ": _step,
                                "y(0.8)": _last,
                                "Оценка Рунге": _runge,
                            }
                            for _n, _step, _last, _runge in solutions[_method][0]
                        ],
                        selection=None,
                        pagination=False,
                        show_download=False,
                    ),
                ]
            )
            for _method, _name, _order in methods
        ]
    )


@app.cell(hide_code=True)
def _(duration, exact_solution, methods, solutions, t_start):
    solution_figure: Figure = plt.figure(figsize=(9, 5), layout="constrained")
    solution_axes: Axes = solution_figure.subplots()
    for _method, _name, _order in methods:
        _history, _times, _values = solutions[_method]
        solution_axes.plot(_times, _values, label=f"{_name}, N={_history[-1][0]}")
    _reference_times: list[float] = [t_start + duration * _i / 400 for _i in range(401)]
    solution_axes.plot(
        _reference_times,
        [exact_solution(_t) for _t in _reference_times],
        "k--",
        linewidth=1,
        label="Точное решение 1/(1-t)",
    )
    solution_axes.set(
        title="Решения на последних сетках", xlabel="t", ylabel="u(t), y(t)"
    )
    solution_axes.grid(True, alpha=0.3)
    solution_axes.legend()
    plt.show()


@app.cell(hide_code=True)
def _(methods, regressions):
    error_figure: Figure = plt.figure(figsize=(10, 4), layout="constrained")
    for _index, (_method, _name, _order) in enumerate(methods, start=1):
        _axes: Axes = error_figure.add_subplot(1, 3, _index)
        _xs, _zs, _slope, _intercept = regressions[_method]
        _axes.plot(_xs, _zs, "o", label="Расчёт")
        _axes.plot(
            _xs,
            [_slope * _x + _intercept for _x in _xs],
            label=f"МНК: p={_slope:.3f}",
        )
        _axes.set(title=_name, xlabel="lg τ", ylabel="lg ψ")
        _axes.grid(True, alpha=0.3)
        _axes.legend()
    plt.show()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Особенность исходных данных.** Если критерий Рунге выполнен уже
    на второй сетке, остаётся только одна точка для МНК, тогда как (1.20)
    требует хотя бы две. В этом случае **только для исследования порядка**
    рассчитывается ещё одна сетка с удвоенным N. Остановка по точности
    и график решения сохраняются на первой подходящей сетке.
    Эталоном МНК служит последняя сетка расширенной последовательности;
    её N указан отдельно в итоговой таблице. Это дополнительное сгущение
    необходимо для выполнения пункта 4 задания.

    Для оценки практического порядка взяты **все сетки, кроме последней**:
    $$\psi_l=\left|\frac{y_l(0{,}8)-y_*(0{,}8)}{y_*(0{,}8)}\right|,
    \qquad y_*=y_L.$$
    По (1.20) строится прямая $\lg\psi=\bar p\lg\tau+B$.
    Последняя сетка не входит в МНК: её ошибка относительно самой себя равна нулю.
    Для каждого метода используется собственное значение $y_*$.

    Практический порядок может отличаться от теоретического: эталон $y_L$
    также приближённый, а начальные сетки могут ещё не давать асимптотической
    зависимости ошибки от шага. Оценка Рунге относится к правому концу интервала.

    Дополнительный аналитический контроль: при $v=1/u$ уравнение принимает вид
    $v'+v=-t$, $v(0)=1$. Поэтому $v=1-t$, $u(t)=1/(1-t)$ и $u(0{,}8)=5$.
    На всём заданном отрезке точное решение положительно и возрастает.
    """)


if __name__ == "__main__":
    app.run()
