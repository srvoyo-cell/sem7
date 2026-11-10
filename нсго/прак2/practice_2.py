import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from collections.abc import Callable
    from itertools import pairwise

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset, TensorDataset

    torch.set_num_threads(min(4, torch.get_num_threads()))
    seed: int = 7
    return (
        Callable,
        DataLoader,
        Dataset,
        TensorDataset,
        mo,
        nn,
        np,
        pairwise,
        plt,
        seed,
        torch,
    )


@app.cell
def _(mo):
    mo.md("""
    # Практическая работа № 2 · Полносвязные нейронные сети в PyTorch

    Цель работы — построить и обучить MLP для регрессии и классификации,
    сравнить оптимизаторы и исследовать регуляризацию и затухание градиентов.
    В экспериментах сравниваем модели на одних и тех же данных и фиксируем
    случайное зерно.
    """)
    return


@app.cell
def _(Callable, nn, pairwise, torch):
    def build_mlp(
        widths: tuple[int, ...],
        activation: Callable[[], nn.Module],
        seed: int,
        dropout: Callable[[], nn.Module] | None = None,
    ) -> nn.Sequential:
        """Создать MLP из §1 с аффинным выходом и заданными скрытыми слоями.

        Args:
            widths: Ширины всех слоёв, включая вход и выход.
            activation: Конструктор активации скрытого слоя.
            seed: Зерно для начальных весов.
            dropout: Конструктор слоя регуляризации после активации.

        Returns:
            Последовательность слоёв со свежими параметрами.
        """
        layers: list[nn.Module] = []
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            for index, (inputs, outputs) in enumerate(pairwise(widths)):
                layers.append(nn.Linear(inputs, outputs))
                if index < len(widths) - 2:
                    layers.append(activation())
                    if dropout is not None:
                        layers.append(dropout())
        return nn.Sequential(*layers)

    def fit(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        train: tuple[torch.Tensor, torch.Tensor],
        validation: tuple[torch.Tensor, torch.Tensor],
        epochs: int,
        batch_size: int,
        seed: int,
        l2: float = 0.0,
    ) -> tuple[list[float], list[float]]:
        """Минимизировать эмпирический риск по §1 со штрафом из §4.1 при l2 > 0.

        Args:
            model: Обучаемая модель.
            optimizer: Правило обновления параметров.
            criterion: Функция потерь по одному пакету.
            train: Обучающие входы и цели.
            validation: Входы и цели для контроля обобщения.
            epochs: Число проходов по обучающей выборке.
            batch_size: Число объектов в мини-пакете.
            seed: Зерно перестановок и стохастических слоёв.
            l2: Коэффициент ручного L2-штрафа для весов.

        Returns:
            Истории чистых потерь на обучении и валидации с эпохи 0.
        """
        inputs, targets = train
        train_loss: list[float] = []
        validation_loss: list[float] = []
        weights: list[nn.Parameter] = [
            parameter
            for name, parameter in model.named_parameters()
            if name.endswith("weight")
        ]
        generator: torch.Generator = torch.Generator().manual_seed(seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            for epoch in range(epochs + 1):
                if epoch:
                    model.train()
                    order: torch.Tensor = torch.randperm(
                        len(inputs), generator=generator
                    )
                    for indices in order.split(batch_size):
                        optimizer.zero_grad()
                        loss: torch.Tensor = criterion(
                            model(inputs[indices]), targets[indices]
                        )
                        if l2:
                            squared_norm: torch.Tensor = torch.stack(
                                [weight.square().sum() for weight in weights]
                            ).sum()
                            loss = loss + (l2 / 2) * squared_norm
                        loss.backward()
                        optimizer.step()

                model.eval()
                with torch.no_grad():
                    train_loss.append(float(criterion(model(inputs), targets).item()))
                    validation_loss.append(
                        float(criterion(model(validation[0]), validation[1]).item())
                    )
        return train_loss, validation_loss

    return build_mlp, fit


@app.cell
def _(build_mlp, fit, mo, nn, seed: int, torch):
    _generator: torch.Generator = torch.Generator().manual_seed(seed)
    _x: torch.Tensor = torch.linspace(-torch.pi, torch.pi, 1000).reshape(-1, 1)
    _noise: torch.Tensor = 0.1 * torch.randn(_x.shape, generator=_generator)
    _y: torch.Tensor = _x.sin() + _noise
    _order: torch.Tensor = torch.randperm(len(_x), generator=_generator)
    _train: torch.Tensor = _order[:800]
    _valid: torch.Tensor = _order[800:]
    _model: nn.Sequential = build_mlp((1, 32, 32, 1), nn.ReLU, seed)
    _train_loss, _valid_loss = fit(
        _model,
        torch.optim.Adam(_model.parameters(), lr=1e-3),
        nn.MSELoss(),
        (_x[_train], _y[_train]),
        (_x[_valid], _y[_valid]),
        epochs=200,
        batch_size=800,
        seed=seed,
    )
    mo.md(
        f"""
        ## §2 · Типовые задачи

        Для регрессии использованы 1 000 точек и нормальный шум со стандартным
        отклонением 0,1, как в примере кода. MLP 1 → 32 → 32 → 1 с Adam за 200 эпох
        снизила MSE на обучении с **{_train_loss[0]:.4f}** до **{_train_loss[-1]:.4f}**.
        Итоговая MSE на 200 отложенных точках **{_valid_loss[-1]:.4f}**.

        Многоклассовая классификация на логитах показана в аудиторной задаче 3.
        """
    )
    return


@app.cell
def _(mo, nn):
    _shapes: tuple[tuple[int, int], ...] = ((784, 256), (256, 128), (128, 10))
    _multiplies: int = sum(inputs * outputs for inputs, outputs in _shapes)
    _biases: int = sum(outputs for _, outputs in _shapes)
    _network: nn.Sequential = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    _count: int = sum(parameter.numel() for parameter in _network.parameters())
    _multiplies_text, _total_text, _count_text = (
        f"{value:_}".replace("_", " ")
        for value in (_multiplies, _multiplies + _biases, _count)
    )
    mo.md(
        rf"""
        ## §3.1 · Параметры и умножения

        Веса по слоям $784 \cdot 256 = 200\,704$, $256 \cdot 128 = 32\,768$ и
        $128 \cdot 10 = 1\,280$. Суммарно **{_multiplies_text}** умножения на объект.
        С учётом **{_biases}** смещений получаем **{_total_text}**
        обучаемых параметров. Подсчёт через PyTorch даёт **{_count_text}**.
        Сложения и сравнения ReLU в число умножений не входят.
        """
    )
    return


@app.cell
def _(mo, nn, seed: int, torch):
    class GatedMLP(nn.Module):
        """Реализовать сеть с поэлементным вентилем из §3.2."""

        def __init__(self, inputs: int, hidden: int, outputs: int) -> None:
            """Создать параметры для входа, скрытого слоя и выхода.

            Args:
                inputs: Число входных признаков.
                hidden: Размер скрытого вектора и вентиля.
                outputs: Число выходных признаков.
            """
            super().__init__()
            self.first: nn.Linear = nn.Linear(inputs, hidden)
            self.gate: nn.Linear = nn.Linear(hidden, hidden)
            self.output: nn.Linear = nn.Linear(hidden, outputs)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Применить tanh, сигмоидный вентиль и выходное преобразование (§3.2).

            Args:
                x: Пакет входных векторов.

            Returns:
                Пакет выходных векторов.
            """
            h1: torch.Tensor = self.first(x).tanh()
            h2: torch.Tensor = h1 * self.gate(h1).sigmoid()
            return self.output(h2)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        _gated: GatedMLP = GatedMLP(inputs=4, hidden=8, outputs=3)
        _result: torch.Tensor = _gated(torch.randn(2, 4))

    mo.md(
        rf"""
        ## §3.2 · Сеть с сигмоидным вентилем

        $h_1 = \tanh(W_1x+b_1)$, $h_2 = h_1 \odot \sigma(W_2h_1+b_2)$,
        $y = W_3h_2+b_3$. Два входных вектора длины 4 дают
        выход размерности **{tuple(_result.shape)}**.
        """
    )
    return


@app.cell
def _(DataLoader, TensorDataset, build_mlp, mo, nn, seed: int, torch):
    _rng: torch.Generator = torch.Generator().manual_seed(seed)
    _centers: torch.Tensor = torch.randn(10, 20, generator=_rng)
    _labels: torch.Tensor = torch.arange(10, dtype=torch.long).repeat_interleave(120)
    _features: torch.Tensor = _centers[_labels] + 0.35 * torch.randn(
        len(_labels), 20, generator=_rng
    )
    _order: torch.Tensor = torch.randperm(len(_labels), generator=_rng)
    _training: torch.Tensor = _order[:1000]
    _holdout: torch.Tensor = _order[1000:]
    _loader: DataLoader[tuple[torch.Tensor, ...]] = DataLoader(
        TensorDataset(_features[_training], _labels[_training]),
        batch_size=32,
        shuffle=True,
        generator=_rng,
    )
    _classifier: nn.Sequential = build_mlp((20, 50, 10), nn.ReLU, seed)
    _criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
    _optimizer: torch.optim.SGD = torch.optim.SGD(_classifier.parameters(), lr=0.01)

    with torch.no_grad():
        _before: float = float(
            _criterion(_classifier(_features[_holdout]), _labels[_holdout]).item()
        )
    _classifier.train()
    for _epoch in range(10):
        for _x_batch, _y_batch in _loader:
            _optimizer.zero_grad()
            _out: torch.Tensor = _classifier(_x_batch)
            _loss: torch.Tensor = _criterion(_out, _y_batch)
            _loss.backward()
            _optimizer.step()

    _classifier.eval()
    with torch.no_grad():
        _logits: torch.Tensor = _classifier(_features[_holdout])
        _true_labels: torch.Tensor = _labels[_holdout]
        _cross_entropy: torch.Tensor = _criterion(_logits, _true_labels)
        _nll: torch.Tensor = nn.NLLLoss()(_logits.log_softmax(dim=1), _true_labels)
        _delta: float = float((_cross_entropy - _nll).abs().item())
        _accuracy: float = float(
            (_logits.argmax(dim=1) == _true_labels).float().mean().item()
        )

    mo.md(
        f"""
        ## §3.3 · Исправление цикла классификации

        1. Добавляем zero_grad() перед backward(). Иначе градиенты соседних
           мини-пакетов суммируются.
        2. Заменяем Softmax в скрытом слое на ReLU. Softmax связывает все 50
           активаций и нормирует их сумму до единицы, что сужает представление.
           Последний линейный слой исходного фрагмента уже выдаёт логиты.
        3. Проверяем контракт DataLoader. Для обычной классификации передаём
           тензор индексов классов формы (B,), с типом torch.long и значениями
           0–9. В условии подготовка меток не показана, поэтому ошибка в них
           остаётся гипотезой.

        На отложенной части синтетического набора из 10 классов cross-entropy
        уменьшилась с **{_before:.4f}** до **{_cross_entropy.item():.4f}**,
        accuracy составила **{_accuracy:.1%}**. Разность между
        CrossEntropyLoss и NLLLoss после LogSoftmax **{_delta:.2e}**.
        """
    )
    return


@app.cell
def _(mo, nn, torch):
    class Swish(nn.Module):
        """Реализовать активацию Swish из §3.4."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Вычислить произведение x и σ(x) по формуле из §3.4.

            Args:
                x: Тензор активаций.

            Returns:
                Swish для каждого элемента входа.
            """
            return x * x.sigmoid()

    _x: torch.Tensor = torch.linspace(-3, 3, 13, dtype=torch.float64).requires_grad_()
    _verified: bool = torch.autograd.gradcheck(Swish(), (_x,))
    mo.md(
        rf"""
        ## §3.4 · Swish

        $f(x) = x\sigma(x)$, $f'(x)=\sigma(x)+x\sigma(x)(1-\sigma(x))$.
        Проверка autograd на 13 точках в двойной точности
        **{"пройдена" if _verified else "выявила расхождение"}**.
        """
    )
    return


@app.cell
def _(DataLoader, Dataset, mo, seed: int, torch):
    class CubicDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
        """Хранить воспроизводимую выборку из §3.5."""

        def __init__(self, size: int, seed: int) -> None:
            """Сгенерировать равномерные входы на интервале (−2, 2).

            Args:
                size: Количество объектов.
                seed: Зерно генератора.
            """
            generator: torch.Generator = torch.Generator().manual_seed(seed)
            self.inputs: torch.Tensor = torch.rand(size, 1, generator=generator) * 4 - 2

        def __len__(self) -> int:
            """Вернуть количество пар."""
            return len(self.inputs)

        def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
            """Получить пару (x, x³ − 2x) по индексу согласно §3.5.

            Args:
                index: Позиция объекта.

            Returns:
                Вход и соответствующее значение кубического многочлена.
            """
            x: torch.Tensor = self.inputs[index]
            return x, x.pow(3) - 2 * x

    _loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        CubicDataset(128, seed),
        batch_size=32,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    _batch_x, _batch_y = next(iter(_loader))
    mo.vstack(
        [
            mo.md("## §3.5 · Собственный Dataset"),
            mo.md(
                f"Размер пакета **{len(_batch_x)}**. Первые пять пар после перемешивания."
            ),
            mo.ui.table(
                [
                    {"x": float(x.item()), "y": float(y.item())}
                    for x, y in zip(_batch_x[:5], _batch_y[:5], strict=True)
                ],
                selection=None,
            ),
        ]
    )
    return


@app.cell
def _(seed: int, torch):
    _rng: torch.Generator = torch.Generator().manual_seed(seed)
    _x: torch.Tensor = torch.linspace(-2.5, 2.5, 1024).reshape(-1, 1)
    _y: torch.Tensor = (5 * _x).sin() * (-_x.square()).exp()
    _order: torch.Tensor = torch.randperm(len(_x), generator=_rng)
    oscillation_train: tuple[torch.Tensor, torch.Tensor] = (
        _x[_order[:768]],
        _y[_order[:768]],
    )
    oscillation_validation: tuple[torch.Tensor, torch.Tensor] = (
        _x[_order[768:]],
        _y[_order[768:]],
    )
    return oscillation_train, oscillation_validation


@app.cell
def _(
    build_mlp,
    fit,
    mo,
    nn,
    oscillation_train: "tuple[torch.Tensor, torch.Tensor]",
    oscillation_validation: "tuple[torch.Tensor, torch.Tensor]",
    plt,
    render_figure,
    seed: int,
    torch,
):
    _epochs: int = 200
    _threshold: float = 0.03
    _lr: float = 0.01
    _results: list[dict[str, str]] = []
    _first_epochs: dict[str, int] = {}
    _figure, _axis = plt.subplots(figsize=(8, 4), layout="constrained")

    for _name in ("SGD", "SGD + momentum 0,9", "Adam"):
        _model: nn.Sequential = build_mlp((1, 64, 64, 64, 64, 64, 1), nn.ReLU, seed)
        _optimizer: torch.optim.Optimizer
        if _name == "Adam":
            _optimizer = torch.optim.Adam(_model.parameters(), lr=_lr)
        else:
            _optimizer = torch.optim.SGD(
                _model.parameters(), lr=_lr, momentum=0.0 if _name == "SGD" else 0.9
            )
        _training, _validation = fit(
            _model,
            _optimizer,
            nn.MSELoss(),
            oscillation_train,
            oscillation_validation,
            epochs=_epochs,
            batch_size=64,
            seed=seed,
        )
        _first: int | None = next(
            (epoch for epoch, value in enumerate(_validation) if value <= _threshold),
            None,
        )
        if _first is not None:
            _first_epochs[_name] = _first
        _results.append(
            {
                "Оптимизатор": _name,
                "MSE обучения": f"{_training[-1]:.6f}",
                "MSE валидации": f"{_validation[-1]:.6f}",
                "Эпоха первого достижения MSE ≤ 0,03": "—"
                if _first is None
                else str(_first),
            }
        )
        _axis.semilogy(range(_epochs + 1), _validation, label=_name)

    _axis.axhline(_threshold, color="0.5", linestyle=":", label="Порог MSE = 0,03")
    _axis.set(
        title="Сходимость оптимизаторов", xlabel="Эпоха", ylabel="MSE на валидации"
    )
    _axis.legend()
    _conclusion: str = "За выделенное число эпох порог остался недостигнутым."
    if _first_epochs:
        _fastest: str = min(_first_epochs, key=lambda name: _first_epochs[name])
        _conclusion = f"При этих настройках первым порога достиг **{_fastest}**, эпоха **{_first_epochs[_fastest]}**."

    mo.vstack(
        [
            mo.md(
                r"""
            ## §3.6 · Сравнение оптимизаторов

            Аппроксимируем $y=\sin(5x)\exp(-x^2)$ на $[-2{,}5;2{,}5]$.
            У моделей пять скрытых слоёв по 64 нейрона с ReLU, общие начальные
            веса, порядок мини-пакетов и шаг 0,01. На графике MSE по 256
            отложенным точкам после каждой эпохи.
            """
            ),
            render_figure(_figure, "Сходимость SGD, Momentum и Adam"),
            mo.ui.table(_results, selection=None),
            mo.md(_conclusion),
        ]
    )
    return


@app.cell
def _(
    build_mlp,
    fit,
    mo,
    nn,
    oscillation_train: "tuple[torch.Tensor, torch.Tensor]",
    oscillation_validation: "tuple[torch.Tensor, torch.Tensor]",
    plt,
    render_figure,
    seed: int,
    torch,
):
    _lambda: float = 0.02
    _epochs: int = 100
    _baseline: nn.Sequential = build_mlp((1, 64, 64, 1), nn.ReLU, seed)
    _decoupled: nn.Sequential = build_mlp((1, 64, 64, 1), nn.ReLU, seed)
    _weights: list[nn.Parameter] = [
        p for name, p in _decoupled.named_parameters() if name.endswith("weight")
    ]
    _biases: list[nn.Parameter] = [
        p for name, p in _decoupled.named_parameters() if name.endswith("bias")
    ]
    _adam_w: torch.optim.AdamW = torch.optim.AdamW(
        [
            {"params": _weights, "weight_decay": _lambda},
            {"params": _biases, "weight_decay": 0.0},
        ],
        lr=0.003,
    )
    _train_l2, _valid_l2 = fit(
        _baseline,
        torch.optim.Adam(_baseline.parameters(), lr=0.003),
        nn.MSELoss(),
        oscillation_train,
        oscillation_validation,
        _epochs,
        64,
        seed,
        l2=_lambda,
    )
    _train_aw, _valid_aw = fit(
        _decoupled,
        _adam_w,
        nn.MSELoss(),
        oscillation_train,
        oscillation_validation,
        _epochs,
        64,
        seed,
    )
    _figure, _axis = plt.subplots(figsize=(8, 4), layout="constrained")
    _axis.semilogy(_valid_l2, label="Adam + ручной L2")
    _axis.semilogy(_valid_aw, label="AdamW")
    _axis.set(
        title="Сходимость при двух способах регуляризации",
        xlabel="Эпоха",
        ylabel="MSE на валидации",
    )
    _axis.legend()
    mo.vstack(
        [
            mo.md(
                rf"""
            ## §4.1 · Ручной L2 и AdamW

            При $\lambda={_lambda:g}$ ручной штраф
            $\frac{{\lambda}}{{2}}\sum_l\|W_l\|_F^2$ добавляет $\lambda W_l$
            к градиенту. Adam включает эту добавку в оценки обоих моментов.
            AdamW вычисляет моменты по градиентам MSE и отдельно уменьшает
            веса на $\eta\lambda W_l$. Поэтому масштаб уменьшения веса
            управляется напрямую. Смещения исключены из регуляризации в
            обоих случаях.

            | Метод | MSE валидации после 10 эпох | MSE обучения после 100 эпох | MSE валидации после 100 эпох |
            |---|---:|---:|---:|
            | Adam + ручной L2 | {_valid_l2[10]:.6f} | {_train_l2[-1]:.6f} | {_valid_l2[-1]:.6f} |
            | AdamW | {_valid_aw[10]:.6f} | {_train_aw[-1]:.6f} | {_valid_aw[-1]:.6f} |

            В этом запуске меньшая MSE валидации на 10-й эпохе у
            **{"AdamW" if _valid_aw[10] < _valid_l2[10] else "Adam с ручным L2"}**,
            на последней эпохе у **{"AdamW" if _valid_aw[-1] < _valid_l2[-1] else "Adam с ручным L2"}**.
            Для сравнения использована MSE без штрафа в обеих моделях.
            """
            ),
            render_figure(_figure, "Сходимость Adam с L2 и AdamW"),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Домашняя задача 2. Линейный автоэнкодер и PCA

    Пусть строки матрицы $X\in\mathbb R^{n\times d}$ центрированы средним
    обучающей выборки. Линейный автоэнкодер без смещений реконструирует
    $X$ как $XED$, где $E\in\mathbb R^{d\times k}$ и
    $D\in\mathbb R^{k\times d}$. Минимизация MSE равносильна минимизации
    $\|X-XED\|_F^2$, поскольку MSE делит эту сумму квадратов на $nd$.

    Ранг $XED$ не превосходит $k$. Если $X=U\Sigma V^\top$, то по
    теореме Эккарта — Янга — Мирского лучшая аппроксимация ранга $k$
    равна $U_k\Sigma_k V_k^\top$. Она достижима при $E=V_k$ и
    $D=V_k^\top$, поскольку $XV_kV_k^\top=U_k\Sigma_k V_k^\top$.
    Столбцы $V_k$ — главные направления PCA, так как
    $X^\top X=V\Sigma^2V^\top$.

    Веса обученного кодировщика допускают смену базиса
    $E\mapsto ER$, $D\mapsto R^{-1}D$ для любой обратимой матрицы $R$.
    Поэтому сравниваем подпространства по главным углам. Косинусы
    углов — сингулярные числа $Q^\top V_k$, где столбцы $Q$ образуют
    ортонормированный базис пространства кодировщика. Меньший угол
    соответствует более близким направлениям.

    В эксперименте связываем веса декодировщика с транспонированными
    весами кодировщика. Решение $E=V_k$, $D=V_k^\top$ достижимо и при
    этом ограничении.
    """)
    return


@app.cell
def _():
    import gzip
    import struct
    from typing import cast
    from urllib.request import urlopen

    from matplotlib.figure import Figure
    from numpy.typing import NDArray
    from sklearn.decomposition import PCA
    from threadpoolctl import threadpool_limits

    return Figure, NDArray, PCA, cast, gzip, struct, threadpool_limits, urlopen


@app.cell
def _(NDArray, gzip, np, struct, urlopen):
    def download_mnist_images(filename: str) -> NDArray[np.float32]:
        """Загрузить изображения MNIST из IDX и привести яркости к [0, 1].

        Args:
            filename: Имя gzip-файла изображений из набора MNIST.

        Returns:
            Матрица изображений с 784 признаками в порядке следования IDX.
        """
        mirrors: tuple[str, ...] = (
            "https://ossci-datasets.s3.amazonaws.com/mnist/",
            "https://storage.googleapis.com/cvdf-datasets/mnist/",
        )
        error: Exception | None = None
        for mirror in mirrors:
            try:
                with urlopen(mirror + filename, timeout=45) as response:
                    compressed: bytes = response.read()
                payload: bytes = gzip.decompress(compressed)
                magic: int
                count: int
                rows: int
                columns: int
                magic, count, rows, columns = struct.unpack_from(">4I", payload)
                if (magic, rows, columns) != (2051, 28, 28):
                    raise ValueError("Некорректный заголовок изображений MNIST.")
                if count == 0 or len(payload) != 16 + count * rows * columns:
                    raise ValueError("Размер файла не совпадает с заголовком IDX.")
                pixels: NDArray[np.uint8] = np.frombuffer(
                    payload, dtype=np.uint8, offset=16
                ).reshape(count, rows * columns)
                images: NDArray[np.float32] = pixels.astype(np.float32)
                images /= 255.0
                return images
            except (OSError, ValueError, EOFError, struct.error) as exc:
                error = exc

        raise RuntimeError(f"Не удалось загрузить MNIST: {filename}") from error

    return (download_mnist_images,)


@app.cell
def _(NDArray, download_mnist_images, np):
    pca_train_images: NDArray[np.float32] = download_mnist_images(
        "train-images-idx3-ubyte.gz"
    )
    pca_test_images: NDArray[np.float32] = download_mnist_images(
        "t10k-images-idx3-ubyte.gz"
    )
    pca_dimension: int = 10
    return pca_dimension, pca_test_images, pca_train_images


@app.cell
def _(cast, nn, torch):
    def fit_linear_autoencoder(
        train: torch.Tensor,
        dimension: int,
        rng_seed: int,
        epochs: int,
    ) -> tuple[nn.Linear, list[float]]:
        """Обучить линейный автоэнкодер минимизацией MSE из домашней задачи 2.

        Args:
            train: Матрица изображений, центрированная по обучающему среднему.
            dimension: Размерность линейного скрытого представления.
            rng_seed: Начальное состояние генератора весов и порядка батчей.
            epochs: Число проходов по обучающей выборке.

        Returns:
            Обученный кодировщик и история средней MSE по батчам каждой эпохи.
        """
        feature_count: int = train.shape[1]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(rng_seed)
            encoder: nn.Linear = nn.Linear(feature_count, dimension, bias=False)
        generator: torch.Generator = torch.Generator().manual_seed(rng_seed)
        optimizer: torch.optim.Adam = torch.optim.Adam(encoder.parameters(), lr=0.002)
        scheduler: torch.optim.lr_scheduler.CosineAnnealingLR = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=0.0002
            )
        )
        criterion: nn.MSELoss = nn.MSELoss()
        history: list[float] = []
        batch_size: int = 512
        sample_count: int = train.shape[0]

        batch_indices: torch.Tensor
        for _ in range(epochs):
            permutation: torch.Tensor = torch.randperm(
                sample_count, generator=generator
            )
            weighted_loss: float = 0.0
            for batch_indices in permutation.split(batch_size):
                batch: torch.Tensor = train[batch_indices]
                optimizer.zero_grad()
                code: torch.Tensor = cast(torch.Tensor, encoder(batch))
                prediction: torch.Tensor = nn.functional.linear(code, encoder.weight.T)
                loss: torch.Tensor = cast(torch.Tensor, criterion(prediction, batch))
                loss.backward()
                optimizer.step()
                weighted_loss += loss.item() * batch.shape[0]
            scheduler.step()
            history.append(weighted_loss / sample_count)

        encoder.eval()
        return encoder, history

    return (fit_linear_autoencoder,)


@app.cell
def _(
    NDArray,
    PCA,
    cast,
    fit_linear_autoencoder,
    nn,
    np,
    pca_dimension: int,
    pca_test_images: "NDArray[np.float32]",
    pca_train_images: "NDArray[np.float32]",
    seed: int,
    threadpool_limits,
    torch,
):
    _train_tensor: torch.Tensor = torch.from_numpy(pca_train_images)
    _test_tensor: torch.Tensor = torch.from_numpy(pca_test_images)
    _train_mean: torch.Tensor = _train_tensor.mean(dim=0)
    _encoder: nn.Linear
    pca_autoencoder_history: list[float]
    _encoder, pca_autoencoder_history = fit_linear_autoencoder(
        _train_tensor - _train_mean, pca_dimension, seed, epochs=200
    )
    pca_encoder_weights: NDArray[np.float32] = _encoder.weight.detach().numpy()
    with torch.no_grad():
        _test_code: torch.Tensor = cast(
            torch.Tensor, _encoder(_test_tensor - _train_mean)
        )
        _reconstructed: torch.Tensor = (
            nn.functional.linear(_test_code, _encoder.weight.T) + _train_mean
        )
        pca_autoencoder_mse: float = (
            (_reconstructed - _test_tensor).square().mean().item()
        )

    with threadpool_limits(limits=4, user_api="blas"):
        _pca: PCA = PCA(n_components=pca_dimension, svd_solver="covariance_eigh")
        _pca.fit(pca_train_images)
        pca_components: NDArray[np.float32] = _pca.components_
        _pca_reconstructed: NDArray[np.float32] = _pca.inverse_transform(
            _pca.transform(pca_test_images)
        )
        pca_baseline_mse: float = float(
            np.mean(np.square(_pca_reconstructed - pca_test_images))
        )
        _encoder_basis: NDArray[np.float32] = np.linalg.qr(
            pca_encoder_weights.T, mode="reduced"
        )[0]
        _cosines: NDArray[np.float32] = np.linalg.svd(
            _encoder_basis.T @ pca_components.T, compute_uv=False
        )
        pca_principal_angles: NDArray[np.float32] = np.degrees(
            np.arccos(np.clip(_cosines, 0.0, 1.0))
        )
    return (
        pca_autoencoder_history,
        pca_autoencoder_mse,
        pca_baseline_mse,
        pca_components,
        pca_encoder_weights,
        pca_principal_angles,
    )


@app.cell
def _(
    mo,
    pca_autoencoder_history: list[float],
    pca_autoencoder_mse: float,
    pca_baseline_mse: float,
    pca_dimension: int,
    pca_principal_angles: "NDArray[np.float32]",
    pca_test_images: "NDArray[np.float32]",
    pca_train_images: "NDArray[np.float32]",
):
    _angles: str = ", ".join(f"{angle:.2f}°" for angle in pca_principal_angles)
    mo.md(
        f"""
        Обучение на {len(pca_train_images):,} изображениях MNIST, оценка на
        {len(pca_test_images):,} тестовых изображениях. Размерность скрытого
        пространства — {pca_dimension}.

        | Метод | Тестовая MSE на пиксель |
        | --- | ---: |
        | Линейный автоэнкодер | {pca_autoencoder_mse:.6f} |
        | PCA | {pca_baseline_mse:.6f} |

        Средняя MSE по батчам автоэнкодера изменилась с
        {pca_autoencoder_history[0]:.6f} до {pca_autoencoder_history[-1]:.6f}.
        Главные углы между подпространствами — {_angles}.
        Разность тестовых ошибок отражает конечное число шагов обучения,
        выбранный оптимизатор и перенос на отдельную тестовую выборку.
        """
    )
    return


@app.cell
def _(
    Figure,
    NDArray,
    np,
    pca_components: "NDArray[np.float32]",
    pca_dimension: int,
    pca_encoder_weights: "NDArray[np.float32]",
    plt,
    render_figure,
):
    _normalized: NDArray[np.float32] = pca_encoder_weights / np.linalg.norm(
        pca_encoder_weights, axis=1, keepdims=True
    )
    _limit: float = float(max(np.abs(_normalized).max(), np.abs(pca_components).max()))
    _figure: Figure
    _figure, _axes = plt.subplots(
        2, pca_dimension, figsize=(14, 3.4), squeeze=False, layout="constrained"
    )
    _row_images: tuple[NDArray[np.float32], NDArray[np.float32]] = (
        _normalized,
        pca_components,
    )
    for _row, _components in enumerate(_row_images):
        for _column, _component in enumerate(_components):
            _axis = _axes[_row, _column]
            _axis.imshow(
                _component.reshape(28, 28), cmap="RdBu_r", vmin=-_limit, vmax=_limit
            )
            _axis.set_xticks([])
            _axis.set_yticks([])
            if _row == 0:
                _axis.set_title(str(_column))
    _axes[0, 0].set_ylabel("Кодировщик")
    _axes[1, 0].set_ylabel("PCA")
    _figure.suptitle("Нормированные строки весов и главные компоненты MNIST")
    render_figure(_figure, "Веса линейного энкодера и компоненты PCA для MNIST")
    return


@app.cell
def _(
    build_mlp,
    fit,
    mo,
    nn,
    oscillation_train: "tuple[torch.Tensor, torch.Tensor]",
    oscillation_validation: "tuple[torch.Tensor, torch.Tensor]",
    plt,
    render_figure,
    seed: int,
    torch,
):
    class InvertedDropout(nn.Module):
        """Случайно обнулять активации с вероятностью p по §4.3."""

        def __init__(self, probability: float) -> None:
            """Запомнить вероятность удаления активации.

            Args:
                probability: Вероятность удаления в диапазоне [0, 1].
            """
            super().__init__()
            if not 0.0 <= probability <= 1.0:
                raise ValueError("Вероятность удаления должна быть в диапазоне [0, 1].")
            self.probability: float = probability

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Применить маску и компенсировать изменение среднего по §4.3.

            Args:
                x: Тензор активаций.

            Returns:
                В режиме обучения — масштабированные активации, иначе исходный x.
            """
            if not self.training or self.probability == 0:
                return x
            keep: float = 1 - self.probability
            if keep == 0:
                return x * 0
            mask: torch.Tensor = torch.empty_like(x).bernoulli_(keep)
            return x * mask / keep

    _probability: float = 0.15
    _epochs: int = 120
    _custom: nn.Sequential = build_mlp(
        (1, 64, 64, 1), nn.ReLU, seed, lambda: InvertedDropout(_probability)
    )
    _builtin: nn.Sequential = build_mlp(
        (1, 64, 64, 1), nn.ReLU, seed, lambda: nn.Dropout(_probability)
    )
    _custom_train, _custom_valid = fit(
        _custom,
        torch.optim.Adam(_custom.parameters(), lr=0.003),
        nn.MSELoss(),
        oscillation_train,
        oscillation_validation,
        _epochs,
        64,
        seed,
    )
    _builtin_train, _builtin_valid = fit(
        _builtin,
        torch.optim.Adam(_builtin.parameters(), lr=0.003),
        nn.MSELoss(),
        oscillation_train,
        oscillation_validation,
        _epochs,
        64,
        seed,
    )

    _dropout: InvertedDropout = InvertedDropout(_probability)
    _ones: torch.Tensor = torch.ones(10000)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        _masked: torch.Tensor = _dropout(_ones)
    _dropout.eval()
    _identity: bool = torch.equal(_ones, _dropout(_ones))

    _figure, _axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
    for _axis, _custom_history, _builtin_history, _title in zip(
        _axes,
        (_custom_train, _custom_valid),
        (_builtin_train, _builtin_valid),
        ("Обучающая выборка", "Валидация"),
        strict=True,
    ):
        _axis.semilogy(_custom_history, label="Ручной Dropout")
        _axis.semilogy(_builtin_history, label="nn.Dropout", linestyle="--")
        _axis.set(title=_title, xlabel="Эпоха", ylabel="MSE в режиме eval")
        _axis.legend()
    mo.vstack(
        [
            mo.md(
                rf"""
            ## §4.3 · Inverted Dropout

            При вероятности удаления $p={_probability:g}$ используем маску
            $m\sim\operatorname{{Bernoulli}}(1-p)$, поэтому
            $\mathbb E[xm/(1-p)]=x$. В методичке параметр Бернулли записан
            как $p$, хотя масштаб $1/(1-p)$ соответствует вероятности
            **сохранения** $1-p$.

            На входе из 10 000 единиц доля нулей **{(_masked == 0).float().mean().item():.1%}**,
            среднее выхода **{_masked.mean().item():.3f}**. В режиме eval
            слой **{"тождественен" if _identity else "изменил вход"}**.
            Итоговые MSE валидации **{_custom_valid[-1]:.6f}** у ручного
            слоя и **{_builtin_valid[-1]:.6f}** у nn.Dropout.
            """
            ),
            render_figure(_figure, "Кривые обучения для ручного и встроенного Dropout"),
        ]
    )
    return


@app.cell
def _(
    build_mlp,
    fit,
    mo,
    nn,
    oscillation_train: "tuple[torch.Tensor, torch.Tensor]",
    oscillation_validation: "tuple[torch.Tensor, torch.Tensor]",
    seed: int,
    torch,
):
    _results: list[dict[str, str]] = []
    _widths: tuple[int, ...] = (1, *(64,) * 20, 1)
    _inputs, _targets = oscillation_train
    _batch: torch.Tensor = torch.randperm(
        len(_inputs), generator=torch.Generator().manual_seed(seed)
    )[:64]

    for _activation in (nn.Sigmoid, nn.ReLU):
        _model: nn.Sequential = build_mlp(_widths, _activation, seed)
        _first_layer = _model[0]
        if not isinstance(_first_layer, nn.Linear):
            raise TypeError("Первый слой должен быть линейным.")
        _loss: torch.Tensor = nn.MSELoss()(_model(_inputs[_batch]), _targets[_batch])
        _loss.backward()
        _grad: torch.Tensor | None = _first_layer.weight.grad
        if _grad is None:
            raise RuntimeError("Градиент первого слоя не вычислен.")
        _norm: float = float(_grad.norm().item())
        _training, _validation = fit(
            _model,
            torch.optim.SGD(_model.parameters(), lr=0.01),
            nn.MSELoss(),
            oscillation_train,
            oscillation_validation,
            epochs=30,
            batch_size=64,
            seed=seed,
        )
        _results.append(
            {
                "Активация": _activation.__name__,
                "Норма градиента первого слоя": f"{_norm:.3e}",
                "Начальная MSE валидации": f"{_validation[0]:.6f}",
                "Итоговая MSE валидации": f"{_validation[-1]:.6f}",
            }
        )

    mo.vstack(
        [
            mo.md(
                r"""
            ## §4.4 · Затухание градиентов

            У обеих сетей 20 скрытых слоёв по 64 нейрона, одинаковые начальные
            веса и первый пакет из 64 объектов. Норма измерена после первого
            backward и до обновления весов. В цепном правиле
            $\sigma'(z)=\sigma(z)(1-\sigma(z))\le1/4$. Произведение таких
            множителей по глубине уменьшает градиент, особенно в насыщении.
            Для активных нейронов производная ReLU равна 1, для неактивных 0.
            В обеих сетях на градиенты влияют также произведения матриц весов.
            """
            ),
            mo.ui.table(_results, selection=None),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## §5 · Вопросы для обсуждения

    1. Композиция аффинных слоёв эквивалентна одному отображению $Ax+b$.
       Нелинейности позволяют моделировать нелинейную зависимость и строить
       нелинейные границы классов.
    2. Для индекса класса $c$ cross-entropy на логитах $z$ равна
       $-z_c+\operatorname{logsumexp}(z)$.
       CrossEntropyLoss вычисляет устойчивый LogSoftmax внутри.
       NLLLoss получает логарифмы вероятностей и берёт $-\log p_c$.
    3. Backward прибавляет новые градиенты к полям grad. Zero grad очищает
       эти поля перед следующим шагом. Каждый новый forward строит свой граф,
       а backward по умолчанию освобождает промежуточные данные графа.
    4. Autograd записывает операции над тензорами с requires_grad=True
       и проходит по этому графу в обратном порядке, применяя цепное правило.
       У листовых параметров, участвовавших в вычислении потерь, результат
       сохраняется в поле grad.
    5. ReLU обнуляет примерно половину симметрично распределённых входов.
       He использует дисперсию весов $2/\mathrm{fan}_{in}$, чтобы компенсировать
       сокращение второго момента при распространении вперёд. Xavier с
       дисперсией $2/(\mathrm{fan}_{in}+\mathrm{fan}_{out})$ для одинаковых
       ширин даёт вдвое меньшую дисперсию.
    """)
    return


@app.cell
def _(Figure, mo, plt):
    from io import BytesIO

    from matplotlib.backends.backend_agg import FigureCanvasAgg

    def render_figure(figure: Figure, description: str) -> mo.Html:
        """Показать график Matplotlib в marimo, используя буфер в памяти.

        Args:
            figure: График для отображения.
            description: Текстовое описание графика.

        Returns:
            Изображение для области вывода ячейки.
        """
        if not isinstance(figure.canvas, FigureCanvasAgg):
            plt.show()
        with BytesIO() as buffer:
            FigureCanvasAgg(figure).print_png(buffer)
            display: mo.Html = mo.image(buffer.getvalue(), alt=description)
        plt.close(figure)
        return display

    return (render_figure,)


if __name__ == "__main__":
    app.run()
