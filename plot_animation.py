"""Render synchronized Plotly snapshots without racing mixed SVG/WebGL traces."""

import json

import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder


def write_animated_html(fig, destination):
    """Frames carry trace updates and optional layout/meta/annotation updates.

    Each frame's layout.meta.time is the physical time. Native controls use
    method='skip', with frame names for slider args and play/pause for buttons.
    """
    frames = [frame.to_plotly_json() for frame in fig.frames]
    controller = r"""
    (() => {
        const plot = document.getElementById('{plot_id}');
        const frames = __HEAT_FRAMES__;
        const indices = new Map(frames.map((frame, index) => [frame.name, index]));
        let generation = 0;
        let requested = null;
        let drawing = null;

        async function draw(index) {
            const frame = frames[index];
            const traces = plot.data.map(trace => ({...trace}));
            frame.traces.forEach((traceIndex, j) => {
                Object.assign(traces[traceIndex], frame.data[j]);
            });
            const layout = {...plot.layout, ...frame.layout,
                meta:{...plot.layout.meta, ...frame.layout.meta},
                sliders:plot.layout.sliders.map(slider => ({...slider, active:index}))
            };
            await Plotly.react(plot, traces, layout, {responsive:true, displaylogo:false});
            await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
            plot.dispatchEvent(new CustomEvent('heat-frame-rendered', {
                detail:{index, time:layout.meta.time}
            }));
        }

        function requestFrame(index) {
            // Coalesce rapid slider changes; never redraw concurrently.
            requested = index;
            if (!drawing) {
                drawing = (async () => {
                    while (requested !== null) {
                        const next = requested;
                        requested = null;
                        await draw(next);
                    }
                })().finally(() => { drawing = null; });
            }
            return drawing;
        }

        plot.on('plotly_sliderchange', event => {
            generation += 1;
            const index = indices.get(event.step.args[0]);
            if (index !== undefined) requestFrame(index).catch(console.error);
        });
        plot.on('plotly_buttonclicked', async event => {
            const token = ++generation;
            if (event.button.args[0] !== 'play') return;
            for (let index=0; index<frames.length && token===generation; index++) {
                await requestFrame(index);
                await new Promise(resolve => setTimeout(resolve, 80));
            }
        });
    })();
    """.replace("__HEAT_FRAMES__", json.dumps(frames, cls=PlotlyJSONEncoder))
    # Keep frame data only once, in the controller, instead of also registering
    # it with Plotly's independent animation queue.
    html_fig = go.Figure(data=fig.data, layout=fig.layout)
    html_fig.write_html(destination, include_plotlyjs=True, auto_play=False,
                        post_script=controller, config=dict(responsive=True, displaylogo=False))
