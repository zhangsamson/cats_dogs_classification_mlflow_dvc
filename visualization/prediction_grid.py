import io
import math

import matplotlib.pyplot as plt
from PIL import Image


def prediction_grid(
    images: list[str], probas, pred_labels=None, labels=None, show=True, ncols=1
) -> Image:
    if labels:
        assert len(images) == len(labels)
    else:
        labels = [None] * len(images)

    if pred_labels:
        assert len(images) == len(pred_labels)
    else:
        pred_labels = [None] * len(images)

    nrows = math.ceil(len(images) / ncols)
    fig, axis_arr = plt.subplots(
        nrows,
        ncols,
        figsize=[3.5 * ncols, 4 * nrows],
        squeeze=False,
    )

    for plot_num, (img_fp, proba) in enumerate(zip(images, probas)):
        img = Image.open(img_fp).convert("RGB")
        cur_col = plot_num % ncols
        cur_row = plot_num // ncols
        ax = axis_arr[cur_row][cur_col]
        ax.imshow(img)

        title = f"cat : {1 - proba:.3f}, dog : {proba:.3f}"
        if labels:
            title += f"\ntrue label : {labels[plot_num]}"

        if pred_labels:
            title += f"\npred label : {pred_labels[plot_num]}"

        ax.set_title(title)
        ax.axis("off")

    io_buf = io.BytesIO()
    plt.savefig(
        io_buf,
        bbox_inches="tight",
        pad_inches=0,
    )

    figure_pil = Image.open(io_buf).copy().convert("RGB")
    io_buf.close()
    if show:
        plt.show()
    plt.ion()
    plt.close()

    return figure_pil
