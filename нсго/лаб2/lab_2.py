import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    from collections.abc import Callable, Iterable
    from io import BytesIO
    from math import isfinite
    from os import getenv
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from time import perf_counter
    from typing import overload

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from numpy.typing import NDArray
    from torch import Tensor, nn
    from torch.optim import Optimizer
    from torchvision.datasets import CIFAR100
    from torchvision.datasets.utils import download_and_extract_archive


@app.cell
def _():
    mo.md(r"""
    # Лабораторная работа № 2. Стохастическая оптимизация

    **Вариант 7.** Сравниваем SGD, SGD с инерцией, Adam и AdamW на функции Леви
    и при обучении свёрточной сети на CIFAR-100. Вторая часть измеряет качество
    обобщения на независимых выборках.

    При обучении минимизируется эмпирический риск (2.3). Для оценки его градиента
    используем среднее по мини-пакету (2.4). Это уменьшает стоимость шага.
    При равномерном отборе оценка несмещённая, а при увеличении пакета её
    дисперсия уменьшается. В опыте с функцией двух переменных берём точный градиент.
    """)
    return


@app.cell
def _():
    seed: int = 42
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu_threads: int = min(4, torch.get_num_threads())
    torch.set_num_threads(cpu_threads)
    steps_2d: int = 300
    start_2d: tuple[float, float] = (2.0, 2.0)
    rates_2d: dict[str, float] = {
        "SGD": 0.001,
        "Momentum": 0.001,
        "Adam": 0.001,
        "AdamW": 0.001,
    }
    rates_cnn: dict[str, float] = {
        "SGD": 0.05,
        "Momentum": 0.05,
        "Adam": 0.001,
        "AdamW": 0.001,
    }
    return cpu_threads, device, rates_2d, rates_cnn, seed, start_2d, steps_2d


@app.cell
def _():
    class CustomSGD(Optimizer):
        """Стохастический градиентный спуск по формуле (2.5)."""

        def __init__(self, params: Iterable[Tensor], lr: float) -> None:
            """Принять обучаемые параметры и скорость обучения lr."""
            super().__init__(params, {"lr": lr})

        @overload
        def step(self, closure: None = None) -> None: ...

        @overload
        def step(self, closure: Callable[[], float]) -> float: ...

        @torch.no_grad()
        def step(self, closure: Callable[[], float] | None = None) -> float | None:
            """Обновить веса по (2.5), вернуть значение closure, если она передана."""
            loss: float | None = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
            for group in self.param_groups:
                rate: float = group["lr"]
                parameters: list[Tensor] = group["params"]
                for param in parameters:
                    gradient: Tensor | None = param.grad
                    if gradient is not None:
                        param.add_(gradient, alpha=-rate)
            return loss

    class CustomMomentum(Optimizer):
        """SGD с инерцией по формулам (2.6)–(2.7)."""

        def __init__(
            self, params: Iterable[Tensor], lr: float, momentum: float = 0.9
        ) -> None:
            """Принять параметры, скорость обучения lr и коэффициент инерции."""
            super().__init__(params, {"lr": lr, "momentum": momentum})

        @overload
        def step(self, closure: None = None) -> None: ...

        @overload
        def step(self, closure: Callable[[], float]) -> float: ...

        @torch.no_grad()
        def step(self, closure: Callable[[], float] | None = None) -> float | None:
            """Накопить скорость, обновить веса и вернуть результат closure, если он есть."""
            loss: float | None = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
            for group in self.param_groups:
                rate: float = group["lr"]
                inertia: float = group["momentum"]
                parameters: list[Tensor] = group["params"]
                for param in parameters:
                    gradient: Tensor | None = param.grad
                    if gradient is None:
                        continue
                    state = self.state[param]
                    if "velocity" not in state:
                        state["velocity"] = torch.zeros_like(param)
                    velocity: Tensor = state["velocity"]
                    velocity.mul_(inertia).add_(gradient)
                    param.add_(velocity, alpha=-rate)
            return loss

    class CustomAdamW(Optimizer):
        """Adam по (2.8)–(2.12) и AdamW с отдельным затуханием по (2.14) из §1.5."""

        def __init__(
            self,
            params: Iterable[Tensor],
            lr: float,
            weight_decay: float,
            betas: tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-8,
        ) -> None:
            """Принять веса, шаг, затухание, коэффициенты моментов и стабилизатор."""
            super().__init__(
                params,
                {"lr": lr, "weight_decay": weight_decay, "betas": betas, "eps": eps},
            )

        @overload
        def step(self, closure: None = None) -> None: ...

        @overload
        def step(self, closure: Callable[[], float]) -> float: ...

        @torch.no_grad()
        def step(self, closure: Callable[[], float] | None = None) -> float | None:
            """Обновить моменты и веса, затем вернуть результат closure, если он есть."""
            loss: float | None = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
            for group in self.param_groups:
                rate: float = group["lr"]
                decay: float = group["weight_decay"]
                betas: tuple[float, float] = group["betas"]
                eps: float = group["eps"]
                parameters: list[Tensor] = group["params"]
                for param in parameters:
                    gradient: Tensor | None = param.grad
                    if gradient is None:
                        continue
                    state = self.state[param]
                    if not state:
                        state["step"] = 0
                        state["first"] = torch.zeros_like(param)
                        state["second"] = torch.zeros_like(param)
                    previous_step: int = state["step"]
                    step: int = previous_step + 1
                    state["step"] = step
                    first: Tensor = state["first"]
                    second: Tensor = state["second"]
                    first.mul_(betas[0]).add_(gradient, alpha=1.0 - betas[0])
                    second.mul_(betas[1]).addcmul_(
                        gradient, gradient, value=1.0 - betas[1]
                    )
                    first_corrected: Tensor = first / (1.0 - betas[0] ** step)
                    second_corrected: Tensor = second / (1.0 - betas[1] ** step)
                    if decay != 0.0:
                        param.mul_(1.0 - rate * decay)
                    param.addcdiv_(
                        first_corrected, second_corrected.sqrt().add_(eps), value=-rate
                    )
            return loss

    def make_optimizer(name: str, params: Iterable[Tensor], rate: float) -> Optimizer:
        """Создать оптимизатор для переданных параметров и скорости обучения."""
        if name == "SGD":
            return CustomSGD(params, rate)
        if name == "Momentum":
            return CustomMomentum(params, rate)
        if name == "Adam":
            return CustomAdamW(params, rate, weight_decay=0.0)
        if name == "AdamW":
            return CustomAdamW(params, rate, weight_decay=0.01)
        raise ValueError(f"Неизвестный оптимизатор {name}")

    return (make_optimizer,)


@app.cell
def _(device: torch.device, make_optimizer):
    def levy(x: Tensor, y: Tensor) -> Tensor:
        """Вычислить функцию Леви из §5.1, пункт 7, для координат x и y."""
        return (
            (3 * torch.pi * x).sin().square()
            + (x - 1).square() * (1 + (3 * torch.pi * y).sin().square())
            + (y - 1).square() * (1 + (2 * torch.pi * y).sin().square())
        )

    def run_trajectory(
        name: str, rate: float, start: tuple[float, float], steps: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Вернуть координаты и значения функции Леви на шагах от 0 до steps.

        Args:
            name: Пользовательский оптимизатор.
            rate: Скорость обучения.
            start: Начальные координаты.
            steps: Число обновлений.

        Returns:
            Массив координат и массив значений целевой функции.
        """
        point: Tensor = torch.tensor(
            start, dtype=torch.float64, device=device, requires_grad=True
        )
        optimizer: Optimizer = make_optimizer(name, [point], rate)
        path: NDArray[np.float64] = np.empty((steps + 1, 2), dtype=np.float64)
        values: NDArray[np.float64] = np.empty(steps + 1, dtype=np.float64)
        for step in range(steps):
            optimizer.zero_grad()
            value: Tensor = levy(point[0], point[1])
            path[step] = point.detach().cpu().numpy()
            values[step] = float(value.item())
            value.backward()
            optimizer.step()
        path[-1] = point.detach().cpu().numpy()
        values[-1] = float(levy(point[0], point[1]).item())
        return path, values

    return levy, run_trajectory


@app.cell
def _(
    rates_2d: dict[str, float],
    run_trajectory,
    start_2d: tuple[float, float],
    steps_2d: int,
):
    trajectories: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]] = {
        name: run_trajectory(name, rate, start_2d, steps_2d)
        for name, rate in rates_2d.items()
    }
    return (trajectories,)


@app.cell
def _(
    rates_2d: dict[str, float],
    trajectories: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]],
):
    trajectory_rows: list[dict[str, str | float]] = []
    for _name, (_points, _values) in trajectories.items():
        _path_length: float = float(
            np.linalg.norm(np.diff(_points, axis=0), axis=1).sum()
        )
        trajectory_rows.append(
            {
                "Оптимизатор": _name,
                "Шаг η": rates_2d[_name],
                "x": round(float(_points[-1, 0]), 7),
                "y": round(float(_points[-1, 1]), 7),
                "f в конце": round(float(_values[-1]), 9),
                "Длина пути": round(_path_length, 6),
            }
        )
    mo.vstack(
        [mo.md("## Функция Леви"), mo.ui.table(trajectory_rows, pagination=False)]
    )
    return


@app.function
def show_figure(figure: Figure, alt: str) -> mo.Html:
    """Отобразить график в marimo из буфера в памяти.

    Args:
        figure: Фигура Matplotlib.
        alt: Текстовое описание графика.

    Returns:
        Изображение графика для вывода в ячейке.
    """
    plt.show()
    with BytesIO() as buffer:
        canvas: FigureCanvasAgg = FigureCanvasAgg(figure)
        canvas.print_png(buffer)
        buffer.seek(0)
        rendered: mo.Html = mo.image(buffer, alt=alt)
    plt.close(figure)
    return rendered


@app.cell
def _(
    levy,
    trajectories: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]],
):
    trajectory_fig: Figure = plt.figure(figsize=(12, 7), layout="constrained")
    _global = trajectory_fig.add_subplot(2, 2, 1)
    _local = trajectory_fig.add_subplot(2, 2, 2)
    _loss_ax = trajectory_fig.add_subplot(2, 1, 2)
    for _ax, _limits in [(_global, (0.9, 2.1)), (_local, (1.98, 2.006))]:
        _axis = torch.linspace(*_limits, 220, dtype=torch.float64)
        _gx, _gy = torch.meshgrid(_axis, _axis, indexing="xy")
        _ax.contour(
            _gx.numpy(), _gy.numpy(), levy(_gx, _gy).numpy(), levels=18, alpha=0.5
        )
        _ax.set(xlabel="x", ylabel="y", xlim=_limits, ylim=_limits)
        _ax.set_aspect("equal")
        for _name, (_path, _values) in trajectories.items():
            (_line,) = _ax.plot(_path[:, 0], _path[:, 1], label=_name)
            _ax.scatter(*_path[-1], color=_line.get_color(), s=18)
        _ax.scatter(2, 2, c="black", marker="x", label="Старт")
    _global.scatter(1, 1, c="black", marker="*", label="Глобальный минимум")
    _global.set_title("Функция Леви и область старта")
    _local.set_title("Траектории в окрестности старта")
    _local.legend()
    _global.legend(fontsize=7)
    for _name, (_, _values) in trajectories.items():
        _loss_ax.plot(range(_values.size), _values, label=_name)
    _loss_ax.set(
        xlabel="Шаг", ylabel="f(x, y)", title="Значение функции вдоль траектории"
    )
    _loss_ax.set_xscale("symlog", linthresh=10)
    _loss_ax.grid(alpha=0.25)
    _loss_ax.legend()
    show_figure(
        trajectory_fig, "Линии уровня функции Леви, траектории и значения функции"
    )
    return


@app.cell
def _():
    mo.md(r"""
    Все слагаемые функции Леви неотрицательны, и при $(1,1)$ они равны нулю.
    При заданном старте и выбранных шагах четыре метода остаются около одного
    локального минимума с $f \approx 1{,}981$. SGD монотонно уменьшает функцию без накопления
    скорости, а инерция даёт колебания вокруг минимума. Adam и AdamW тоже осциллируют
    на первых шагах. У AdamW затухание немного сдвигает конечную точку, поэтому
    конечное значение исходной функции отличается от результата Adam.

    Для постоянного ненулевого градиента $g$ на шаге $t$ получаем
    $m_t=(1-\beta_1^t)g$ и $v_t=(1-\beta_2^t)g^2$ по (2.8)–(2.9).
    Если пренебречь $\varepsilon$, отношение некорректированного шага к
    $\eta$ равно $(1-\beta_1^t)/\sqrt{1-\beta_2^t}$. При $t=1$ и
    $(\beta_1,\beta_2)=(0{,}9,0{,}999)$ это отношение равно $3{,}162$.
    Следовательно, замечание об обязательно меньшем начальном шаге в §1.4
    для указанных параметров не выполняется. Поправки (2.10)–(2.11)
    восстанавливают оценки $g$ и $g^2$, давая шаг около $\eta$.

    Добавление штрафа $\lambda\|\theta\|_2^2$ в (2.13) даёт градиент
    $2\lambda\theta$. Запись с $\lambda\theta$ соответствует штрафу
    $\frac{\lambda}{2}\|\theta\|_2^2$. В AdamW затухание
    $\theta\gets(1-\eta\lambda)\theta$ применяется отдельно от моментов.
    """)
    return


@app.cell
def _():
    def sample_by_class(
        targets: list[int], count: int, generator: np.random.Generator
    ) -> NDArray[np.int64]:
        """Вернуть по count случайных индексов без повторов для каждого класса."""
        labels: NDArray[np.int64] = np.asarray(targets, dtype=np.int64)
        choices: list[NDArray[np.int64]] = []
        for label in np.unique(labels):
            candidates: NDArray[np.int64] = np.flatnonzero(labels == label)
            choices.append(generator.choice(candidates, size=count, replace=False))
        return np.stack(choices)

    def to_tensors(
        dataset: CIFAR100, indices: NDArray[np.int64]
    ) -> tuple[Tensor, Tensor]:
        """Вернуть отобранные изображения в формате NCHW и целые метки классов."""
        selected: NDArray[np.int64] = indices.reshape(-1)
        pixels: Tensor = torch.from_numpy(dataset.data[selected])
        images: Tensor = pixels.permute(0, 3, 1, 2).contiguous().float().div_(255)
        labels: Tensor = torch.as_tensor(
            [dataset.targets[i] for i in selected], dtype=torch.long
        )
        return images, labels

    def load_subsets(
        root: str | None,
        seed: int,
        train_per_class: int,
        val_per_class: int,
        test_per_class: int,
    ) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        """Загрузить CIFAR-100 и вернуть стратифицированные train, validation и test.

        Args:
            root: Каталог CIFAR-100 или None для временной загрузки.
            seed: Начальное состояние генератора случайных чисел.
            train_per_class: Число обучающих изображений на класс.
            val_per_class: Число валидационных изображений на класс.
            test_per_class: Число тестовых изображений на класс.

        Returns:
            Три пары тензоров с изображениями и метками. Все изображения
            стандартизированы по среднему и отклонению обучающей части.
        """
        if root is None:
            with TemporaryDirectory(prefix="cifar100-lab2-") as temp:
                download_and_extract_archive(
                    "https://zenodo.org/records/10089977/files/cifar-100-python.tar.gz?download=1",
                    temp,
                    filename=CIFAR100.filename,
                    md5=CIFAR100.tgz_md5,
                    remove_finished=True,
                )
                official_train: CIFAR100 = CIFAR100(temp, train=True)
                official_test: CIFAR100 = CIFAR100(temp, train=False)
        else:
            local_root: Path = Path(root).expanduser()
            official_train = CIFAR100(local_root, train=True, download=True)
            official_test = CIFAR100(local_root, train=False)

        generator: np.random.Generator = np.random.default_rng(seed)
        train_val_ids: NDArray[np.int64] = sample_by_class(
            official_train.targets, train_per_class + val_per_class, generator
        )
        test_ids: NDArray[np.int64] = sample_by_class(
            official_test.targets, test_per_class, generator
        )
        train_x, train_y = to_tensors(
            official_train, train_val_ids[:, :train_per_class]
        )
        val_x, val_y = to_tensors(official_train, train_val_ids[:, train_per_class:])
        test_x, test_y = to_tensors(official_test, test_ids)

        mean: Tensor = train_x.mean(dim=(0, 2, 3), keepdim=True)
        std: Tensor = train_x.std(dim=(0, 2, 3), keepdim=True, correction=0)
        return (
            ((train_x - mean) / std, train_y),
            ((val_x - mean) / std, val_y),
            ((test_x - mean) / std, test_y),
        )

    return (load_subsets,)


@app.cell
def _(cpu_threads: int, device: torch.device, seed: int):
    mo.md(f"""
    ## Обучение CNN на CIFAR-100

    Из 100 классов берём по 100 обучающих, 20 валидационных и 30 тестовых
    изображений на класс. Train и validation не пересекаются и отобраны из
    исходной обучающей выборки. Test отобран из исходной тестовой выборки.
    Средние и стандартные отклонения каналов вычисляем только на train.

    CNN получает RGB-изображения $32\\times32$ и выдаёт 100 логитов.
    Сравниваем все методы на одинаковых начальных весах и порядке объектов
    (seed {seed}), по 5 эпох с пакетом 128. Потери и точность на train усредняются
    по пакетам в ходе обучения, а на test измеряются с весами конца эпохи.
    Устройство вычислений: **{device}**, число потоков CPU: {cpu_threads}.
    Для запуска с локальной копией CIFAR-100
    можно задать переменную среды **CIFAR100_ROOT**. При первом запуске без
    локальной копии архив (~169 МБ) временно загружается и проверяется
    по контрольной сумме torchvision. [Описание CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html),
    [копия архива на Zenodo](https://zenodo.org/records/10089977).
    """)
    return


@app.cell
def _(load_subsets, seed: int):
    train_data, val_data, test_data = load_subsets(
        getenv("CIFAR100_ROOT"), seed, 100, 20, 30
    )
    return test_data, train_data, val_data


@app.cell
def _(device: torch.device, make_optimizer):
    def make_cnn(seed: int) -> nn.Sequential:
        """Создать CNN с фиксированной инициализацией для 100 классов."""
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            return nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(64 * 4 * 4, 100),
            )

    def epoch_metrics(
        model: nn.Sequential,
        data: tuple[Tensor, Tensor],
        indices: Tensor,
        batch_size: int,
        optimizer: Optimizer | None,
    ) -> tuple[float, float]:
        """Вычислить среднюю кросс-энтропию (2.3) и долю правильных ответов.

        Args:
            model: Оцениваемая сеть.
            data: Тензоры изображений и меток.
            indices: Порядок предъявления объектов.
            batch_size: Максимальный размер пакета.
            optimizer: Оптимизатор для обучения или None для оценки.

        Returns:
            Средняя потеря по объектам и доля правильных ответов.
        """
        images, labels = data
        training: bool = optimizer is not None
        model.train(training)
        criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
        total_loss: float = 0.0
        correct: int = 0
        with torch.set_grad_enabled(training):
            for batch in indices.split(batch_size):
                if optimizer is not None:
                    optimizer.zero_grad()
                logits: Tensor = model(images[batch].to(device))
                targets: Tensor = labels[batch].to(device)
                loss: Tensor = criterion(logits, targets)
                batch_loss: float = float(loss.item())
                if not isfinite(batch_loss):
                    return float("nan"), float("nan")
                if optimizer is not None:
                    loss.backward()
                    optimizer.step()
                total_loss += batch_loss * batch.numel()
                correct += int((logits.argmax(dim=1) == targets).sum().item())
        return total_loss / indices.numel(), correct / indices.numel()

    def fit(
        name: str,
        rate: float,
        batch_size: int,
        epochs: int,
        seed: int,
        train: tuple[Tensor, Tensor],
        holdout: tuple[Tensor, Tensor],
    ) -> tuple[dict[str, list[float]], float]:
        """Обучить CNN на train и оценить её на holdout после каждой эпохи.

        Args:
            name: Название пользовательского оптимизатора.
            rate: Скорость обучения.
            batch_size: Размер обучающего мини-пакета.
            epochs: Число эпох.
            seed: Начальное состояние для весов и перемешивания.
            train: Обучающая часть.
            holdout: Валидационная или тестовая часть для оценки.

        Returns:
            История потерь и точности для обеих выборок и время обучения.
        """
        model: nn.Sequential = make_cnn(seed).to(device)
        optimizer: Optimizer = make_optimizer(name, model.parameters(), rate)
        rng: torch.Generator = torch.Generator().manual_seed(seed)
        evaluation_order: Tensor = torch.arange(len(holdout[1]))
        history: dict[str, list[float]] = {
            "train_loss": [],
            "train_accuracy": [],
            "holdout_loss": [],
            "holdout_accuracy": [],
        }
        train_time: float = 0.0
        for _epoch in range(epochs):
            order: Tensor = torch.randperm(len(train[1]), generator=rng)
            started: float = perf_counter()
            train_loss, train_accuracy = epoch_metrics(
                model, train, order, batch_size, optimizer
            )
            train_time += perf_counter() - started
            eval_loss, eval_accuracy = epoch_metrics(
                model, holdout, evaluation_order, 256, None
            )
            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_accuracy)
            history["holdout_loss"].append(eval_loss)
            history["holdout_accuracy"].append(eval_accuracy)
            if not isfinite(train_loss) or not isfinite(eval_loss):
                break
        return history, train_time

    return (fit,)


@app.cell
def _(fit, rates_cnn: dict[str, float], seed: int, test_data, train_data):
    cnn_results: dict[str, tuple[dict[str, list[float]], float]] = {
        name: fit(name, rate, 128, 5, seed, train_data, test_data)
        for name, rate in rates_cnn.items()
    }
    return (cnn_results,)


@app.cell
def _(cnn_results: dict[str, tuple[dict[str, list[float]], float]]):
    learning_fig: Figure = plt.figure(figsize=(11, 7), layout="constrained")
    for _position, _key, _title, _ylabel in [
        (1, "train_loss", "Train: потери", "Кросс-энтропия"),
        (2, "holdout_loss", "Test: потери", "Кросс-энтропия"),
        (3, "train_accuracy", "Train: точность", "Доля правильных ответов"),
        (4, "holdout_accuracy", "Test: точность", "Доля правильных ответов"),
    ]:
        _ax = learning_fig.add_subplot(2, 2, _position)
        for _name, (_history, _seconds) in cnn_results.items():
            _metric: list[float] = _history[_key]
            _ax.plot(range(1, len(_metric) + 1), _metric, marker="o", label=_name)
        _ax.set(xlabel="Эпоха", ylabel=_ylabel, title=_title, xticks=range(1, 6))
        _ax.grid(alpha=0.25)
        _ax.legend()
    show_figure(learning_fig, "Потери и точность четырёх оптимизаторов по эпохам")
    return


@app.cell
def _(
    cnn_results: dict[str, tuple[dict[str, list[float]], float]],
    rates_cnn: dict[str, float],
):
    cnn_rows: list[dict[str, str | float]] = []
    for _name, (_history, _seconds) in cnn_results.items():
        _train_accuracy: float = _history["train_accuracy"][-1]
        _test_accuracy: float = _history["holdout_accuracy"][-1]
        cnn_rows.append(
            {
                "Оптимизатор": _name,
                "Шаг η": rates_cnn[_name],
                "Train accuracy, %": round(100 * _train_accuracy, 2),
                "Train loss": round(_history["train_loss"][-1], 4),
                "Test accuracy, %": round(100 * _test_accuracy, 2),
                "Test error, %": round(100 * (1 - _test_accuracy), 2),
                "Test loss": round(_history["holdout_loss"][-1], 4),
                "Train − test, п.п.": round(
                    100 * (_train_accuracy - _test_accuracy), 2
                ),
                "Обучение, с": round(_seconds, 2),
            }
        )
    _valid: list[str] = [
        name
        for name, (history, _) in cnn_results.items()
        if len(history["holdout_accuracy"]) == 5
        and isfinite(history["train_loss"][-1])
        and isfinite(history["holdout_accuracy"][-1])
    ]
    if _valid:
        _best: str = max(
            _valid, key=lambda name: cnn_results[name][0]["holdout_accuracy"][-1]
        )
        _fastest: str = min(_valid, key=lambda name: cnn_results[name][1])
        _smoothest: str = min(
            _valid,
            key=lambda name: sum(
                max(0.0, after - before)
                for before, after in zip(
                    cnn_results[name][0]["holdout_loss"],
                    cnn_results[name][0]["holdout_loss"][1:],
                )
            ),
        )
        _conclusion = mo.md(
            f"Лучшее качество после пятой эпохи у **{_best}** "
            f"({cnn_results[_best][0]['holdout_accuracy'][-1]:.2%} на test). "
            f"Наименьшее время обучения у **{_fastest}**. "
            f"Наименьшая сумма повышений тестовой потери между эпохами у **{_smoothest}**. "
            "Точность измерена при фиксированных архитектуре, данных и бюджете эпох."
        )
    else:
        _conclusion = mo.md("Все прогоны прерваны из-за нечисловых значений потерь.")
    mo.vstack([mo.ui.table(cnn_rows, pagination=False), _conclusion])
    return


@app.cell
def _():
    mo.md("""
    ## Размер мини-пакета и скорость обучения

    Для SGD перебираем сетку из листинга 2.15 по две эпохи на настройку.
    Качество каждой конфигурации измеряем на validation. Сравнивая размеры
    пакетов, учитываем, что за одинаковое число эпох меньший пакет даёт
    больше обновлений весов.
    """)
    return


@app.cell
def _(fit, seed: int, train_data, val_data):
    batch_sizes: tuple[int, ...] = (16, 64, 256, 1024)
    learning_rates: tuple[float, ...] = (0.1, 0.01, 0.001, 0.0001)
    grid_results: dict[tuple[int, float], tuple[dict[str, list[float]], float]] = {
        (size, rate): fit("SGD", rate, size, 2, seed, train_data, val_data)
        for size in batch_sizes
        for rate in learning_rates
    }
    return batch_sizes, grid_results, learning_rates


@app.cell
def _(
    grid_results: dict[tuple[int, float], tuple[dict[str, list[float]], float]],
):
    grid_rows: list[dict[str, int | str | float | None]] = []
    for (_size, _rate), (_history, _seconds) in grid_results.items():
        _loss: float = _history["holdout_loss"][-1]
        _accuracy: float = _history["holdout_accuracy"][-1]
        _finite: bool = (
            len(_history["holdout_loss"]) == 2
            and isfinite(_history["train_loss"][-1])
            and isfinite(_loss)
        )
        grid_rows.append(
            {
                "Размер пакета": _size,
                "Шаг η": _rate,
                "Validation accuracy, %": round(_accuracy * 100, 2)
                if _finite
                else None,
                "Validation loss": round(_loss, 4) if _finite else None,
                "Δ loss за вторую эпоху": (
                    round(_loss - _history["holdout_loss"][0], 4) if _finite else None
                ),
                "Обучение, с": round(_seconds, 2),
                "Потери конечны": "Да" if _finite else "Нет",
            }
        )
    mo.ui.table(grid_rows, pagination=False)
    return


@app.cell
def _(
    batch_sizes: tuple[int, ...],
    grid_results: dict[tuple[int, float], tuple[dict[str, list[float]], float]],
    learning_rates: tuple[float, ...],
):
    _quality: NDArray[np.float64] = np.array(
        [
            [
                100 * grid_results[(size, rate)][0]["holdout_accuracy"][-1]
                if len(grid_results[(size, rate)][0]["train_loss"]) == 2
                and isfinite(grid_results[(size, rate)][0]["train_loss"][-1])
                else np.nan
                for rate in learning_rates
            ]
            for size in batch_sizes
        ]
    )
    grid_fig: Figure = plt.figure(figsize=(7, 5), layout="constrained")
    _ax = grid_fig.add_subplot()
    _heatmap = _ax.imshow(_quality, cmap="viridis")
    grid_fig.colorbar(_heatmap, ax=_ax, label="Точность на validation, %")
    _ax.set(
        xlabel="Скорость обучения η",
        ylabel="Размер пакета",
        title="SGD после двух эпох",
        xticks=range(len(learning_rates)),
        xticklabels=[f"{rate:g}" for rate in learning_rates],
        yticks=range(len(batch_sizes)),
        yticklabels=[str(size) for size in batch_sizes],
    )
    for (_row, _col), _value in np.ndenumerate(_quality):
        _label: str = f"{_value:.1f}" if isfinite(_value) else "Сбой"
        _ax.text(
            _col,
            _row,
            _label,
            ha="center",
            va="center",
            color="white",
            bbox={"facecolor": "black", "alpha": 0.5, "edgecolor": "none"},
        )
    show_figure(grid_fig, "Тепловая карта точности SGD на валидационной выборке")
    return


@app.cell
def _(
    batch_sizes: tuple[int, ...],
    grid_results: dict[tuple[int, float], tuple[dict[str, list[float]], float]],
):
    _usable: list[tuple[int, float]] = [
        (size, rate)
        for (size, rate), (history, _) in grid_results.items()
        if len(history["holdout_loss"]) == 2
        and isfinite(history["train_loss"][-1])
        and isfinite(history["holdout_loss"][-1])
        and history["holdout_loss"][-1] < history["holdout_loss"][0]
    ]
    _nondescending: int = len(grid_results) - len(_usable)
    if _usable:
        _best_size, _best_rate = max(
            _usable, key=lambda config: grid_results[config][0]["holdout_accuracy"][-1]
        )
        _best_accuracy: float = grid_results[(_best_size, _best_rate)][0][
            "holdout_accuracy"
        ][-1]
        _smallest: int = min(batch_sizes)
        _small_accuracy: float = grid_results[(_smallest, _best_rate)][0][
            "holdout_accuracy"
        ][-1]
        _good_rates: str = ", ".join(
            f"{rate:g}" for rate in sorted({rate for _, rate in _usable})
        )
        _grid_note: str = (
            f"Потери на validation уменьшились на второй эпохе при шагах **{_good_rates}** "
            f"хотя бы для одного размера пакета. Для **{_nondescending}** сочетаний снижения "
            f"не наблюдалось. Среди сочетаний со снижением потерь максимальная точность "
            f"**{_best_accuracy:.2%}** получилась при пакете **{_best_size}** "
            f"и шаге **{_best_rate:g}**. При том же шаге минимальный пакет "
            f"**{_smallest}** дал **{_small_accuracy:.2%}**. "
            "Очень малый шаг ограничивает продвижение за две эпохи, а чрезмерно "
            "большой шаг может вызвать колебания или расходимость."
        )
    else:
        _grid_note = (
            "Во всех сочетаниях потери на validation не снизились на второй эпохе."
        )
    mo.md(_grid_note)
    return


if __name__ == "__main__":
    app.run()
