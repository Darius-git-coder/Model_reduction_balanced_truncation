"""Explore the full finite-difference heat system before model reduction.

Run: python visualize_system.py
The HTML is self-contained and can be opened without a Python server.
"""

import argparse
from pathlib import Path

import numpy as np

from helpers import fdm_2d_poisson, implicit_euler


def simulate(d=17, tf=1.0, steps=1000, input_kind="sine"):
    """Return the LTI matrices, time snapshots, input and averaged output.

    Each row of x belongs to one inner grid point, ordered row by row
    as in np.meshgrid(...).flatten(). Boundary values are fixed at zero
    and therefore are not independent states.
    """
    A, B, C, X, Y = fdm_2d_poisson(d)
    if input_kind == "sine":
        u = lambda t: 10 * np.sin(2 * np.pi * t)
    else:
        # Turn the source off after the first fifth of the simulation.
        u = lambda t: 10.0 if t < 0.2 * tf else 0.0
    times = np.linspace(0, tf, steps + 1)
    x = implicit_euler(A, B, tf, u, np.zeros((d * d, 1)), steps)
    return A, B, C, X, Y, times, x, np.array([u(t) for t in times]), (C @ x)[0]


def estimate_error(data, tf, steps, input_kind):
    """Estimate spatial error using a nested grid with half the spacing.

    Both grids use the same Euler time steps. The fine-grid values are
    restricted to coincident coarse nodes, without spatial interpolation.
    This difference is an error estimate, not an exact PDE error or bound.
    A separate run with half the time step measures temporal sensitivity.
    """
    d = data[3].shape[0]
    reference_d = 2 * d + 1
    reference = simulate(reference_d, tf, steps, input_kind)
    restricted = reference[6].reshape(reference_d, reference_d, steps + 1)[1::2, 1::2, :]
    absolute = np.abs(data[6] - restricted.reshape(d * d, steps + 1))
    half_time_step = simulate(d, tf, 2 * steps, input_kind)
    time_difference = np.abs(data[6] - half_time_step[6][:, ::2])
    return dict(reference_d=reference_d, absolute=absolute,
                maximum=np.max(absolute, axis=0),
                time_difference_max=float(np.max(time_difference)))


def save_overview(data, destination, error):
    """Save the grid, stencil matrix, forcing and simulated behavior."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    A, B, C, X, Y, times, x, inputs, output = data
    d = X.shape[0]
    h = 1 / (d + 1)
    grid = np.linspace(0, 1, d + 2)
    GX, GY = np.meshgrid(grid, grid)
    source = np.sin(1.5 * np.pi * GX) * np.sin((4 / 3) * np.pi * GY)
    peak = np.argmax(np.max(np.abs(x), axis=0))
    field = np.pad(x[:, peak].reshape(d, d), 1)
    limit = np.max(np.abs(x)) or 1.0

    fig, axes = plt.subplots(2, 4, figsize=(19, 9), layout="constrained")
    fig.suptitle(f"Wärmeleitung auf dem Einheitsquadrat · d={d}, n={d*d}, h={h:.4f}")
    ax = axes[0, 0]
    ax.plot(GX, GY, color="0.85", lw=0.5)
    ax.plot(GX.T, GY.T, color="0.85", lw=0.5)
    ax.scatter(X, Y, s=8, label=f"{d*d} innere Punkte: Zustandsvariablen")
    edge = (GX == 0) | (GX == 1) | (GY == 0) | (GY == 1)
    ax.scatter(GX[edge], GY[edge], s=8, color="0.3", label="Rand: z = 0")
    ax.set(title="Räumliches Gitter", xlabel="$x_1$", ylabel="$x_2$", aspect="equal")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17))

    for ax, values, title, bound in [
        (axes[0, 1], source, "Räumliche Anregung b(x₁, x₂)", 1.0),
        (axes[0, 2], field, f"Zustand z bei t={times[peak]:.3f}", limit),
    ]:
        im = ax.pcolormesh(GX, GY, values, shading="gouraud", cmap="RdBu_r", vmin=-bound, vmax=bound)
        ax.set(title=title, xlabel="$x_1$", ylabel="$x_2$", aspect="equal")
        fig.colorbar(im, ax=ax, shrink=0.8)
    error_field = np.pad(error["absolute"][:, peak].reshape(d, d), 1)
    error_limit = float(np.max(error["absolute"])) or 1.0
    im = axes[0, 3].pcolormesh(GX, GY, error_field, shading="gouraud", cmap="magma", vmin=0, vmax=error_limit)
    axes[0, 3].set(title=f"Gitterabweichung |zₕ − zₕ/₂| bei t={times[peak]:.3f}", xlabel="$x_1$", ylabel="$x_2$", aspect="equal")
    fig.colorbar(im, ax=axes[0, 3], shrink=0.8)
    axes[1, 0].spy(A, markersize=0.7)
    axes[1, 0].set(title="Besetzungsmuster von A (5-Punkt-Stern)", xlabel="Spalte", ylabel="Zeile")
    axes[1, 1].plot(times, inputs)
    axes[1, 1].set(title="Zeitlicher Eingang u(t)", xlabel="Zeit t", ylabel="u(t)")
    axes[1, 2].plot(times, output)
    axes[1, 2].set(title="Ausgang y₁(t) ≈ ∫Ω z(t, x) dx", xlabel="Zeit t", ylabel="y₁(t)")
    axes[1, 3].plot(times, error["maximum"])
    axes[1, 3].set(title="Größte Gitterabweichung je Zeitpunkt", xlabel="Zeit t", ylabel="max |zₕ − zₕ/₂|")
    for ax in axes[1, 1:]:
        ax.axhline(0, color="0.7", lw=0.6)
        ax.grid(alpha=0.2)
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def build_interactive(data, error):
    """Animate the computed snapshots, including the zero Dirichlet boundary."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    A, B, C, X, Y, times, x, inputs, output = data
    d = X.shape[0]
    grid = np.linspace(0, 1, d + 2)
    GX, GY = np.meshgrid(grid, grid)
    source = np.sin(1.5 * np.pi * GX) * np.sin((4 / 3) * np.pi * GY)
    limit = np.max(np.abs(x)) or 1.0
    # Only animation frames are sampled; every time step enters the simulation.
    indices = np.unique(np.linspace(0, len(times) - 1, min(81, len(times)), dtype=int))
    initial = int(indices[np.argmax(np.max(np.abs(x[:, indices]), axis=0))])
    field = lambda k: np.pad(x[:, k].reshape(d, d), 1)
    error_field = lambda k: np.pad(error["absolute"][:, k].reshape(d, d), 1)
    error_limit = float(np.max(error["absolute"])) or 1.0
    fig = make_subplots(
        rows=2, cols=3, specs=[[{"type": "surface"}, {"type": "surface"}, {}], [{}, {}, {}]],
        subplot_titles=("Räumliche Anregung b(x₁, x₂)", "Zustand z(t, x₁, x₂)", "Errormap |zₕ − zₕ/₂|",
                        "Eingang u(t)", "Ausgang y₁(t) ≈ ∫Ω z dx", "Größte Gitterabweichung"),
        row_heights=[0.7, 0.3], vertical_spacing=0.15, horizontal_spacing=0.10,
    )
    fig.add_trace(go.Surface(x=grid, y=grid, z=source, colorscale="RdBu_r", cmin=-1, cmax=1, showscale=False), row=1, col=1)
    fig.add_trace(go.Surface(x=grid, y=grid, z=field(initial), colorscale="RdBu_r", cmin=-limit, cmax=limit,
                            colorbar=dict(title="z", len=0.5, y=0.75, x=0.65, thickness=12)), row=1, col=2)
    fig.add_trace(go.Heatmap(x=grid, y=grid, z=error_field(initial), colorscale="Magma", zmin=0, zmax=error_limit,
                            colorbar=dict(title="|e|", len=0.5, y=0.75, x=1.02, thickness=12),
                            hovertemplate="x₁=%{x:.3f}<br>x₂=%{y:.3f}<br>|e|=%{z:.6g}<extra></extra>"), row=1, col=3)
    fig.add_trace(go.Scatter(x=times, y=inputs, mode="lines", name="u(t)", line=dict(color="#2563eb")), row=2, col=1)
    fig.add_trace(go.Scatter(x=times, y=output, mode="lines", name="y₁(t)", line=dict(color="#2563eb")), row=2, col=2)
    fig.add_trace(go.Scatter(x=times, y=error["maximum"], mode="lines", name="max |e|", line=dict(color="#9333ea")), row=2, col=3)
    for col, values in [(1, inputs), (2, output), (3, error["maximum"])]:
        fig.add_trace(go.Scatter(x=[times[initial]], y=[values[initial]], mode="markers", marker=dict(size=10, color="#dc2626"), showlegend=False), row=2, col=col)
    frames = []
    for k in indices:
        frames.append(go.Frame(name=str(k), data=[
            go.Surface(z=field(k)),
            go.Heatmap(z=error_field(k)),
            go.Scatter(x=[times[k]], y=[inputs[k]]),
            go.Scatter(x=[times[k]], y=[output[k]]),
            go.Scatter(x=[times[k]], y=[error["maximum"][k]]),
        ], traces=[1, 2, 6, 7, 8]))
    fig.frames = frames
    fig.update_layout(
        title=dict(text=f"Wärmeleitung · d={d} · n={d*d} · h={1/(d+1):.4f} · Rand z=0"
                   f"<br><sup>Errormap: Vergleich mit d={error['reference_d']} bei gleichem Δt · geschätzte räumliche Abweichung</sup>"),
        template="plotly_white", height=820, margin=dict(l=50, r=80, t=110, b=140), showlegend=False,
        uirevision="heat-system",
        scene=dict(xaxis_title="x₁", yaxis_title="x₂", zaxis_title="b", zaxis=dict(range=[-1, 1])),
        scene2=dict(xaxis_title="x₁", yaxis_title="x₂", zaxis_title="z", zaxis=dict(range=[-limit, limit])),
        sliders=[dict(active=list(indices).index(initial), currentvalue=dict(prefix="Zeit t = "), pad=dict(t=40), steps=[
            dict(method="skip", label=f"{times[k]:.3f}", args=[str(k)]) for k in indices
        ])],
        updatemenus=[dict(type="buttons", direction="left", x=0, y=-0.20, buttons=[
            dict(label="Abspielen", method="skip", args=["play"]),
            dict(label="Pause", method="skip", args=["pause"]),
        ])],
    )
    fig.update_xaxes(title_text="Zeit t", row=2)
    fig.update_yaxes(title_text="u(t)", row=2, col=1)
    fig.update_yaxes(title_text="y₁(t)", row=2, col=2)
    fig.update_yaxes(title_text="max |e|", row=2, col=3)
    fig.update_xaxes(title_text="x₁", row=1, col=3)
    fig.update_yaxes(title_text="x₂", row=1, col=3)
    fig.layout.annotations[2].text = (
        f"Errormap |zₕ − zₕ/₂|<br><sup>t={times[initial]:.3f} · max |e|={error['maximum'][initial]:.3g}</sup>"
    )
    return fig


def save_interactive(data, destination, error):
    from plot_animation import write_animated_html

    fig = build_interactive(data, error)
    fig.update_layout(meta=dict(error_traces=[2], error_markers=[8]))
    for frame in fig.frames:
        k = int(frame.name)
        annotations = [annotation.to_plotly_json() for annotation in fig.layout.annotations]
        annotations[2]["text"] = (
            f"Errormap |zₕ − zₕ/₂|<br><sup>t={data[5][k]:.3f} · max |e|={error['maximum'][k]:.3g}</sup>"
        )
        frame.layout = dict(annotations=annotations, meta=dict(time=float(data[5][k])))
    write_animated_html(fig, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d", type=int, default=17, help="innere Gitterpunkte pro Raumrichtung")
    parser.add_argument("--tf", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--input", choices=["sine", "pulse"], default="sine")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs")
    args = parser.parse_args()
    if args.d < 1 or args.tf <= 0 or args.steps < 1:
        parser.error("d, tf und steps müssen positiv sein")
    data = simulate(args.d, args.tf, args.steps, args.input)
    error = estimate_error(data, args.tf, args.steps, args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"heat_system_{args.input}"
    png = args.output_dir / f"{stem}.png"
    html = args.output_dir / f"{stem}.html"
    save_overview(data, png, error)
    save_interactive(data, html, error)
    A, B, C, *_ = data
    print(f"d={args.d}, h={1/(args.d+1):.6f}, n={args.d**2}")
    print(f"A: {A.shape}, B: {B.shape}, C₁: {C.shape}; C₂ = Identität")
    print(f"Räumlicher Vergleich: d_ref={error['reference_d']}, max |z_h-z_h/2|={np.max(error['absolute']):.6g}")
    print(f"Zeitvergleich auf dem ursprünglichen Gitter: max |z_dt-z_dt/2|={error['time_difference_max']:.6g}")
    print(f"Übersicht: {png}\nInteraktiv: {html}")


if __name__ == "__main__":
    main()
