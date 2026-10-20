# Rendering

`sekai.rendering`. See the [Rendering guide](../guide/rendering.md); the classic-control and
finance renderers are documented under [Environments](envs.md#renderers), and the market
charts under [Finance](finance.md#rendering).

::: sekai.rendering.Renderer

::: sekai.rendering.Canvas

::: sekai.rendering.rotate

::: sekai.rendering.text_width

::: sekai.rendering.RecordFrames

::: sekai.rendering.save_gif

::: sekai.rendering.save_video

## Dashboards

::: sekai.rendering.Dashboard

::: sekai.rendering.Panel

::: sekai.rendering.View

::: sekai.rendering.Row

::: sekai.rendering.Column

::: sekai.rendering.Series

::: sekai.rendering.Stat

::: sekai.rendering.LineChart

::: sekai.rendering.BarChart

::: sekai.rendering.StackedArea

## Themes

::: sekai.rendering.Theme

::: sekai.rendering.theme.LIGHT

::: sekai.rendering.theme.DARK

::: sekai.rendering.get_theme

## Chart helpers

`sekai.rendering.charts`: what the charts above are drawn with, for drawing your own.

::: sekai.rendering.charts
    options:
      show_root_heading: false
      members:
        - Box
        - Plot
        - nice_ticks
        - nice_step
        - tick_labels
        - format_number
        - value_range
        - legend
