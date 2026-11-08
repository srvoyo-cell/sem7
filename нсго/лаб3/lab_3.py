import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import gzip
    from hashlib import sha256
    from io import BytesIO
    from urllib.request import urlopen

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, TensorDataset

    return (
        Axes,
        BytesIO,
        DataLoader,
        F,
        Figure,
        NDArray,
        TensorDataset,
        gzip,
        mo,
        nn,
        np,
        plt,
        sha256,
        torch,
        urlopen,
    )


@app.cell
def _(mo):
    mo.md("""
    # Лабораторная работа № 3. Анализ тензорных данных с использованием свёрточных нейронных сетей

    **Вариант 7.** [USPS](https://hastie.su.domains/ElemStatLearn/datasets/).
    Вход — одно изображение размером 1 × 16 × 16, выход — оценки 10 цифр.

    **Цель.** Проверить формулы для свёрточных слоёв и обучить сеть распознавать
    рукописные цифры. Для малых изображений используем только два слоя MaxPool2d,
    а в третьем блоке сводим каждую карту признаков к одному числу средним пулингом.
    """)


@app.cell
def _(mo):
    mo.md(r"""
    ## Математическая постановка

    Пакет изображений имеет форму $(N,C,H,W)$. Для USPS это $(N,1,16,16)$.
    В свёрточном слое вычисляется кросс-корреляция (§1.2). Для одного выходного
    канала и одной позиции суммируются произведения весов ядра и локального
    окна по всем входным каналам. К результату добавляется смещение канала.

    Для каждой пространственной оси размер выхода (§1.3) равен

    $$
    n_{\mathrm{out}}
    =
    \left\lfloor\frac{n_{\mathrm{in}}+2p-d(k-1)-1}{s}\right\rfloor+1,
    $$

    где $k$ — ширина ядра на данной оси, $s$ — шаг, $p$ — дополнение
    с каждой стороны, $d$ — дилатация. Число обучаемых параметров
    свёртки со смещением (§1.4) равно $C_{\mathrm{out}}(C_{\mathrm{in}}k_hk_w+1)$.

    Рецептивное поле и шаг относительно входа (§1.5) начинаются с
    $r_0=J_0=1$. Для свёртки или пулинга без дилатации
    $r_l=r_{l-1}+(k_l-1)J_{l-1}$ и $J_l=J_{l-1}s_l$.
    У границ часть теоретического поля может приходиться на дополнение.
    """)


@app.cell
def _():
    seed: int = 42
    side: int = 16
    channels: tuple[int, ...] = (16, 32, 64)
    classes: int = 10
    epochs: int = 5
    learning_rate: float = 0.001
    return channels, classes, epochs, learning_rate, seed, side


@app.cell
def _(F, nn, torch):
    def output_length(
        size: int, kernel: int, stride: int = 1, padding: int = 0, dilation: int = 1
    ) -> int:
        """Вычислить длину выходной оси по формуле из §1.3.

        Args:
            size: Размер входа по оси.
            kernel: Размер ядра.
            stride: Шаг перемещения ядра.
            padding: Заполнение с каждой стороны.
            dilation: Расстояние между соседними коэффициентами ядра.

        Returns:
            Размер выхода по оси после округления вниз.
        """
        return (size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1

    def manual_correlation(
        inputs: torch.Tensor,
        weights: torch.Tensor,
        bias: torch.Tensor | None,
        stride: int,
        padding: int,
    ) -> torch.Tensor:
        """Вычислить кросс-корреляцию по §1.2 алгоритмом unfold из шага 2.

        Args:
            inputs: Пакет тензоров формы (N, C_in, H, W).
            weights: Ядра формы (C_out, C_in, kH, kW).
            bias: Смещение каждого выходного канала или None.
            stride: Одинаковый шаг по обеим осям.
            padding: Одинаковое нулевое заполнение по обеим осям.

        Returns:
            Выход формы (N, C_out, H_out, W_out). Окна перемножаются с ядрами
            матрично без отражения.
        """
        batch, _, height, width = inputs.shape
        outputs, _, kernel_h, kernel_w = weights.shape
        out_h: int = output_length(height, kernel_h, stride, padding)
        out_w: int = output_length(width, kernel_w, stride, padding)
        windows: torch.Tensor = F.unfold(
            inputs, (kernel_h, kernel_w), stride=stride, padding=padding
        ).transpose(1, 2)
        flattened_weights: torch.Tensor = weights.flatten(1).transpose(0, 1)
        product: torch.Tensor = windows @ flattened_weights
        if bias is not None:
            product = product + bias
        return product.transpose(1, 2).reshape(batch, outputs, out_h, out_w)

    def make_cnn(channels: tuple[int, ...], classes: int) -> nn.Sequential:
        """Собрать CNN из шага 3 для малых изображений.

        Args:
            channels: Число выходных каналов каждого свёрточного блока.
            classes: Число классов.

        Returns:
            Последовательность Conv2d–BatchNorm2d–ReLU с двумя MaxPool2d,
            глобальным средним пулингом и линейным классификатором.
        """
        layers: list[nn.Module] = []
        in_channels: int = 1
        for index, out_channels in enumerate(channels):
            layers.extend(
                (
                    nn.Conv2d(in_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                )
            )
            layers.append(
                nn.AdaptiveAvgPool2d(1)
                if index == len(channels) - 1
                else nn.MaxPool2d(2)
            )
            in_channels = out_channels
        layers.extend((nn.Flatten(), nn.Linear(in_channels, classes)))
        return nn.Sequential(*layers)

    return make_cnn, manual_correlation, output_length


@app.cell
def _(channels, classes, make_cnn, nn, output_length, side, torch):
    _model: nn.Sequential = make_cnn(channels, classes)
    _model.eval()
    _length: int = side
    _in_channels: int = 1
    _field: int = 1
    _jump: int = 1
    _expected: list[tuple[str, tuple[int, ...], int, int | str]] = []
    for _block, _out_channels in enumerate(channels, start=1):
        _length = output_length(_length, kernel=3, padding=1)
        _field += (3 - 1) * _jump
        _shape: tuple[int, ...] = (1, _out_channels, _length, _length)
        _conv_parameters: int = _out_channels * _in_channels * 3 * 3 + _out_channels
        _expected.extend(
            (
                (f"Conv2d {_block}", _shape, _conv_parameters, _field),
                (f"BatchNorm2d {_block}", _shape, 2 * _out_channels, _field),
                (f"ReLU {_block}", _shape, 0, _field),
            )
        )
        if _block < len(channels):
            _length = output_length(_length, kernel=2, stride=2)
            _field += (2 - 1) * _jump
            _jump *= 2
            _expected.append(
                (f"MaxPool2d {_block}", (1, _out_channels, _length, _length), 0, _field)
            )
        else:
            _field += (_length - 1) * _jump
            _expected.append(("AdaptiveAvgPool2d", (1, _out_channels, 1, 1), 0, _field))
        _in_channels = _out_channels
    _expected.extend(
        (
            ("Flatten", (1, channels[-1]), 0, "—"),
            ("Linear", (1, classes), channels[-1] * classes + classes, "—"),
        )
    )

    architecture_rows: list[dict[str, str | int | bool]] = []
    _activation: torch.Tensor = torch.zeros(1, 1, side, side)
    with torch.no_grad():
        for _row, _layer in zip(_expected, _model, strict=True):
            _name, _expected_shape, _expected_params, _receptive_field = _row
            _activation = _layer(_activation)
            _actual_params: int = sum(
                parameter.numel()
                for parameter in _layer.parameters()
                if parameter.requires_grad
            )
            _matches: bool = (
                tuple(_activation.shape) == _expected_shape
                and _actual_params == _expected_params
            )
            architecture_rows.append(
                {
                    "Слой": _name,
                    "Размер по формуле": " × ".join(map(str, _expected_shape)),
                    "Размер PyTorch": " × ".join(map(str, _activation.shape)),
                    "Параметры по формуле": _expected_params,
                    "Параметры PyTorch": _actual_params,
                    "Рецептивное поле, сторона": _receptive_field,
                    "Совпало": _matches,
                }
            )
    architecture_matches: bool = all(bool(row["Совпало"]) for row in architecture_rows)
    parameter_count: int = sum(
        parameter.numel()
        for parameter in _model.parameters()
        if parameter.requires_grad
    )
    return architecture_matches, architecture_rows, parameter_count


@app.cell
def _(architecture_matches, architecture_rows, mo, parameter_count):
    mo.vstack(
        [
            mo.md(
                rf"""
                ## Контрольные точки 1 и 3. Расчёт и устройство CNN

                Для свёртки используем формулу размера выхода из §1.3 и считаем
                параметры по §1.4 как $C_{{out}}(C_{{in}}k_hk_w+1)$.
                BatchNorm2d добавляет по масштабу и смещению на канал.
                Для рецептивного поля из §1.5 начинаем с $r_0=J_0=1$ и
                обновляем $r \gets r+(k-1)J$, $J \gets Js$.
                В последнем блоке глобальный пулинг усредняет все позиции 4 × 4,
                поэтому формальная сторона поля достигает 30 пикселей с учётом
                заполнения у границ. Поле вычислено для режима оценки, где BatchNorm
                использует накопленные статистики.

                Пространственные размеры после MaxPool2d и глобального пулинга —
                **16 × 16 → 8 × 8 → 4 × 4 → 1 × 1**.
                Всего **{parameter_count}** обучаемых параметров.
                Совпадение по каждому слою: **{"да" if architecture_matches else "нет"}**.
                """
            ),
            mo.ui.table(architecture_rows, selection=None, show_download=False),
        ]
    )


@app.cell
def _(F, manual_correlation, mo, seed, torch):
    _generator: torch.Generator = torch.Generator().manual_seed(seed)
    _inputs: torch.Tensor = torch.randn(
        2, 1, 8, 8, generator=_generator, dtype=torch.float64
    )
    _weights: torch.Tensor = torch.randn(
        4, 1, 3, 3, generator=_generator, dtype=torch.float64
    )
    _bias: torch.Tensor = torch.randn(4, generator=_generator, dtype=torch.float64)
    _manual: torch.Tensor = manual_correlation(
        _inputs, _weights, _bias, stride=1, padding=1
    )
    _reference: torch.Tensor = F.conv2d(_inputs, _weights, _bias, stride=1, padding=1)
    correlation_error: float = float((_manual - _reference).abs().max().item())
    correlation_matches: bool = torch.allclose(_manual, _reference, atol=1e-5)
    mo.md(
        f"""
        ## Контрольная точка 2. Двумерная кросс-корреляция

        Для тензора 2 × 1 × 8 × 8 применены четыре ядра 3 × 3.
        Шаг равен 1, нулевое заполнение равно 1. Размер результата —
        {" × ".join(map(str, _manual.shape))}.

        Максимальная абсолютная ошибка относительно F.conv2d —
        **{correlation_error:.3e}**. Совпадение по критерию
        **torch.allclose(atol=1e-5)**: **{"да" if correlation_matches else "нет"}**.
        """
    )
    return correlation_error, correlation_matches


@app.cell
def _(BytesIO, NDArray, gzip, np, sha256, torch, urlopen):
    def download_usps(
        filename: str, row_count: int, checksum: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Загрузить исходную выборку USPS из архива книги ESL в память.

        Args:
            filename: Имя сжатого файла официального архива данных ESL.
            row_count: Ожидаемое число изображений в выборке.
            checksum: Проверенная SHA-256 сумма сжатого файла.

        Returns:
            Тензор изображений NCHW с яркостями в [0, 1] и метки классов 0–9.
        """
        base_url: str = "https://hastie.su.domains/ElemStatLearn/datasets/"
        with urlopen(base_url + filename, timeout=60) as response:
            compressed: bytes = response.read()
        if sha256(compressed).hexdigest() != checksum:
            raise ValueError(f"Контрольная сумма USPS не совпала: {filename}")
        data: NDArray[np.float32] = np.loadtxt(
            BytesIO(gzip.decompress(compressed)), dtype=np.float32
        )
        if data.shape != (row_count, 257):
            raise ValueError(f"Неверный размер таблицы USPS: {filename}")
        images: NDArray[np.float32] = ((data[:, 1:] + 1.0) / 2.0).reshape(
            row_count, 1, 16, 16
        )
        labels: NDArray[np.int64] = data[:, 0].astype(np.int64)
        return torch.from_numpy(images), torch.from_numpy(labels)

    return (download_usps,)


@app.cell
def _(download_usps, torch):
    usps_train_images: torch.Tensor
    usps_train_labels: torch.Tensor
    usps_train_images, usps_train_labels = download_usps(
        "zip.train.gz",
        7291,
        "66ebf231ab51f2cf6ac1a892d0f98ccffee1079d7a5db638c1f11f366046ca16",
    )
    usps_test_images: torch.Tensor
    usps_test_labels: torch.Tensor
    usps_test_images, usps_test_labels = download_usps(
        "zip.test.gz",
        2007,
        "8dacc3c6e136af6c3488de9e5113a7daf140c8c2f1f0b61c54ee02f03f1a962f",
    )
    return usps_test_images, usps_test_labels, usps_train_images, usps_train_labels


@app.cell
def _(DataLoader, nn, torch):
    def measure_epoch(
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.CrossEntropyLoss,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> tuple[float, float]:
        """Измерить метрики эпохи по шагу 4 и при необходимости обновить веса.

        Args:
            model: Модель классификации.
            loader: Пакеты тензоров изображений и меток.
            criterion: Средняя кросс-энтропия пакета.
            optimizer: Оптимизатор для обучения или None для оценки.

        Returns:
            Кросс-энтропия по всем образцам и доля верных ответов.
        """
        training: bool = optimizer is not None
        model.train(training)
        loss_sum: float = 0.0
        correct: int = 0
        seen: int = 0
        with torch.set_grad_enabled(training):
            for images, labels in loader:
                batch_size: int = labels.shape[0]
                if optimizer is not None:
                    optimizer.zero_grad()
                logits: torch.Tensor = model(images)
                loss: torch.Tensor = criterion(logits, labels)
                if optimizer is not None:
                    loss.backward()
                    optimizer.step()
                loss_sum += float(loss.item()) * batch_size
                correct += int(logits.argmax(1).eq(labels).sum().item())
                seen += batch_size
        return loss_sum / seen, correct / seen

    return (measure_epoch,)


@app.cell
def _(
    DataLoader,
    TensorDataset,
    channels,
    classes,
    epochs,
    learning_rate,
    make_cnn,
    measure_epoch,
    nn,
    seed,
    torch,
    usps_test_images,
    usps_test_labels,
    usps_train_images,
    usps_train_labels,
):
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    torch.manual_seed(seed)
    trained_model: nn.Sequential = make_cnn(channels, classes)
    _training_loader: DataLoader = DataLoader(
        TensorDataset(usps_train_images, usps_train_labels),
        batch_size=128,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    _testing_loader: DataLoader = DataLoader(
        TensorDataset(usps_test_images, usps_test_labels), batch_size=256
    )
    _criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
    _optimizer: torch.optim.AdamW = torch.optim.AdamW(
        trained_model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    history: list[tuple[float, float, float, float]] = []
    for _ in range(epochs):
        _train_loss, _train_accuracy = measure_epoch(
            trained_model, _training_loader, _criterion, _optimizer
        )
        _test_loss, _test_accuracy = measure_epoch(
            trained_model, _testing_loader, _criterion
        )
        history.append((_train_loss, _train_accuracy, _test_loss, _test_accuracy))
    return history, trained_model


@app.cell
def _(
    render_figure,
    Axes,
    Figure,
    history,
    mo,
    parameter_count,
    plt,
    usps_test_labels,
    usps_train_labels,
):
    _figure: Figure = plt.figure(figsize=(9, 3.6), layout="constrained")
    _axes: list[Axes] = _figure.subplots(1, 2).ravel().tolist()
    _epoch_numbers: list[int] = list(range(1, len(history) + 1))
    _axes[0].plot(_epoch_numbers, [row[0] for row in history], "-o", label="Обучение")
    _axes[0].plot(_epoch_numbers, [row[2] for row in history], "-o", label="Тест")
    _axes[0].set(
        title="Кросс-энтропия", xlabel="Эпоха", ylabel="Потери", xticks=_epoch_numbers
    )
    _axes[1].plot(
        _epoch_numbers, [100 * row[1] for row in history], "-o", label="Обучение"
    )
    _axes[1].plot(_epoch_numbers, [100 * row[3] for row in history], "-o", label="Тест")
    _axes[1].set(
        title="Точность",
        xlabel="Эпоха",
        ylabel="Верные ответы, %",
        xticks=_epoch_numbers,
    )
    for _axis in _axes:
        _axis.grid(alpha=0.25)
        _axis.legend()
    _final = history[-1]
    _best_test_epoch: int = (
        min(range(len(history)), key=lambda index: history[index][2]) + 1
    )
    _accuracy_gap: float = 100 * (_final[1] - _final[3])
    _summary: list[dict[str, str | int]] = [
        {
            "Набор": "USPS",
            "Архитектура": "CNN 16–32–64, GAP",
            "Параметры": parameter_count,
            "Train loss": f"{_final[0]:.4f}",
            "Test loss": f"{_final[2]:.4f}",
            "Train accuracy": f"{_final[1]:.2%}",
            "Test accuracy": f"{_final[3]:.2%}",
        }
    ]
    _trend: str = (
        "Минимум тестовых потерь достигнут в последнюю эпоху. Устойчивого роста тестовой ошибки при падении обучающей на этом интервале не наблюдается."
        if _best_test_epoch == len(history)
        else f"Тестовые потери минимальны на эпохе {_best_test_epoch}, затем к концу обучения увеличиваются. Разрыв кривых помогает оценить возможное начало переобучения."
    )
    mo.vstack(
        [
            mo.md(
                f"""
                ## Контрольная точка 4. Обучение и оценка

                Обучение: **{len(usps_train_labels)}** изображение. Тест: **{len(usps_test_labels)}**
                изображений из официального тестового набора USPS. Использованы AdamW,
                кросс-энтропия, фиксированные 5 эпох и шаг оптимизатора 0,001.
                Яркости масштабированы из [−1, 1] в [0, 1]. Тестовые метрики
                используются только для оценки.
                """
            ),
            render_figure(_figure, "Кривые потерь и точности CNN на USPS"),
            mo.ui.table(_summary, selection=None, show_download=False),
            mo.md(
                f"**Переобучение.** {_trend} Разница итоговой точности "
                f"«обучение − тест» составляет **{_accuracy_gap:+.2f} п. п.**"
            ),
        ]
    )


@app.cell
def _(
    render_figure,
    Axes,
    Figure,
    mo,
    nn,
    plt,
    torch,
    trained_model,
    usps_test_images,
    usps_test_labels,
):
    _example: torch.Tensor = usps_test_images[:1]
    _first = trained_model[0]
    assert isinstance(_first, nn.Conv2d)
    _filters: torch.Tensor = _first.weight.detach()[:, 0]
    _limit: float = float(_filters.abs().max().item())
    _filter_figure: Figure = plt.figure(figsize=(7.2, 5.6), layout="constrained")
    _filter_axes: list[Axes] = _filter_figure.subplots(4, 4).ravel().tolist()
    for _index, _axis in enumerate(_filter_axes):
        _axis.imshow(_filters[_index].numpy(), cmap="RdBu_r", vmin=-_limit, vmax=_limit)
        _axis.set_title(str(_index + 1))
        _axis.axis("off")
    _filter_figure.suptitle("Ядра первого Conv2d, красное — плюс, синее — минус")

    _maps: list[torch.Tensor] = []
    _activation: torch.Tensor = _example
    with torch.no_grad():
        for _layer in trained_model:
            _activation = _layer(_activation)
            if isinstance(_layer, nn.Conv2d):
                _maps.append(_activation[0].detach())
    _predicted: int = int(_activation.argmax(1).item())
    _truth: int = int(usps_test_labels[0].item())

    _map_figure: Figure = plt.figure(figsize=(8.8, 6.1), layout="constrained")
    _map_axes = _map_figure.subplots(3, 5, squeeze=False)
    for _block, _maps_block in enumerate(_maps):
        _input_axis: Axes = _map_axes[_block, 0]
        _input_axis.imshow(_example[0, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        _input_axis.set_title(f"Вход • блок {_block + 1}")
        _input_axis.axis("off")
        _map_channels: list[int] = (
            _maps_block.flatten(1).std(dim=1).topk(4).indices.tolist()
        )
        _map_limit: float = float(_maps_block[_map_channels].abs().max().item())
        for _column, _channel in enumerate(_map_channels, start=1):
            _map_axis: Axes = _map_axes[_block, _column]
            _map_axis.imshow(
                _maps_block[_channel].numpy(),
                cmap="RdBu_r",
                vmin=-_map_limit,
                vmax=_map_limit,
            )
            _map_axis.set_title(
                f"Канал {_channel + 1} • {_maps_block.shape[-1]}×{_maps_block.shape[-1]}"
            )
            _map_axis.axis("off")
    _map_figure.suptitle(
        f"Карты до BatchNorm и ReLU. Метка {_truth}, прогноз {_predicted}"
    )
    mo.vstack(
        [
            mo.md("## Контрольная точка 5. Ядра и карты признаков"),
            render_figure(_filter_figure, "Ядра первого свёрточного слоя"),
            render_figure(_map_figure, "Карты признаков свёрточных слоёв"),
            mo.md(
                """
                Ядра первого слоя с соседними положительными и отрицательными
                весами реагируют на переходы между фоном и штрихом. Противоположные
                знаки откликов показывают разные стороны перехода. На картах
                16 × 16 ещё различимы положение и контур цифры. В блоках 2 и 3
                карты становятся грубее (8 × 8 и 4 × 4), зато отклик одной позиции
                зависит от более широкой части изображения. Показаны четыре канала
                с наибольшим пространственным разбросом отдельно в каждом слое.

                Сеть собирает локальные признаки штриха в признаки более крупных
                частей цифры. Глобальный пулинг переносит силу откликов в классификатор
                и ослабляет зависимость от точного положения цифры.
                """
            ),
        ]
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Ответы на контрольные вопросы

    1. **Свёртка и кросс-корреляция.** Математическая свёртка отражает ядро
       по обеим пространственным осям перед суммированием. Кросс-корреляция
       берёт веса в порядке их хранения.
    2. **Реализация в библиотеках.** При обучении веса подбираются по данным.
       Отражение произвольного обучаемого ядра взаимно однозначно меняет
       его параметризацию. Кросс-корреляция даёт тот же класс отображений
       и позволяет индексировать ядро напрямую.
    3. **Размер выхода.** Рост шага уменьшает число позиций окна, дополнение
       увеличивает доступную длину входа, дилатация увеличивает эффективную
       длину ядра до $1+d(k-1)$.
    4. **Параметры.** У каждого из $C_{\mathrm{out}}$ фильтров
       $C_{\mathrm{in}}k_hk_w$ весов. При наличии смещения добавляется один
       параметр на фильтр.
    5. **Сравнение с полносвязным слоем.** Один выходной элемент свёртки
       зависит от локального окна. Те же веса используются во всех позициях,
       поэтому число весов свёртки не растёт вместе с площадью входа.
    6. **Рецептивное поле.** Это область исходного входа, которая может влиять
       на выходной элемент. Следующее ядро объединяет поля соседних элементов,
       а увеличение шага повышает расстояние между их центрами.
    7. **Индуктивные предположения.** Соседние значения связаны и образуют
       полезные локальные структуры. Детектор полезного признака применим
       во многих пространственных позициях, поэтому его веса можно разделять.
    8. **Цветовые каналы.** В RGB каждый пиксель представлен тремя значениями
       интенсивности красного, зелёного и синего. У полутонового USPS один канал.
    9. **MaxPool2d.** Максимум вычисляется отдельно в каждом канале, поэтому
       их число сохраняется. Ядро $2\times2$ с шагом 2 без дополнения и с
       округлением вниз уменьшает каждую пространственную ось примерно вдвое.
    10. **Первые слои.** Они могут выделять локальный контраст, границы,
        штрихи, углы и текстуры. Конкретный отклик зависит от обученных весов.
    11. **Глобальный средний пулинг.** Он превращает каждую карту в одно среднее
        и даёт классификатору $C$ признаков вместо $CHW$. Это сокращает число
        параметров и зависимость от позиции отклика, сохраняя его среднюю силу.
    12. **Переобучение.** Если потери на обучении продолжают падать, а на
        валидации растут и разрыв увеличивается, сеть подстраивается к обучающей
        выборке. Для выбора эпохи используют валидацию, а на тесте проверяют
        итоговую модель.
    """)


@app.cell
def _(BytesIO, Figure, mo, plt):
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
