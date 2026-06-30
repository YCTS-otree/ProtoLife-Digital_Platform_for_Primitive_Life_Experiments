from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backend_bases import KeyEvent, MouseEvent

from scripts.plot_agent_count_v1 import (
    LogData,
    color_contrast_ratio,
    contrasting_colors,
    plot_agent_count,
    power_of_two_tick_interval,
)


def _log(name: str, steps: list[int], counts: list[int], order: int) -> LogData:
    return LogData(Path(name), steps, counts, {}, input_order=order)


def test_power_of_two_ticks_adapt_to_visible_width():
    intervals = [power_of_two_tick_interval((0, width)) for width in (8, 64, 512)]
    assert intervals == sorted(intervals)
    assert all(interval > 0 and interval & (interval - 1) == 0 for interval in intervals)


def test_curve_colors_have_contrast_against_background():
    colors = contrasting_colors("white", 20)
    assert len(set(colors)) == 20
    assert all(color_contrast_ratio(color, "white") >= 3.0 for color in colors)


def test_overlapping_logs_are_drawn_as_independent_curves_and_view_can_reset():
    older = _log("older.jsonl", [0, 4, 8], [2, 3, 4], 0)
    newer = _log("newer.jsonl", [4, 8, 12], [5, 6, 7], 1)
    original_show = plt.show
    plt.show = lambda: None
    try:
        fig, ax = plot_agent_count([older, newer])
        data_lines = [line for line in ax.lines if line.get_label() != "_nolegend_"]
        assert len(data_lines) == 2
        assert list(data_lines[0].get_xdata()) == older.steps
        assert list(data_lines[1].get_xdata()) == newer.steps
        assert ax.get_xlim()[0] == 0
        assert ax.get_ylim() == (0.0, 9.0)

        ax.set_xlim(4, 8)
        fig.canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", fig.canvas, key="r"))
        assert ax.get_xlim() == (0.0, 12.0)

        ax.set_xlim(4, 8)
        x, y = ax.transData.transform((6, 4))
        event = MouseEvent("button_press_event", fig.canvas, x, y, button=1, dblclick=True)
        fig.canvas.callbacks.process("button_press_event", event)
        assert ax.get_xlim() == (0.0, 12.0)
    finally:
        plt.show = original_show
        plt.close("all")
