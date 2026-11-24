"""Compare coarse finite differences and balanced truncation of a fine model.

Run: python compare_methods.py
Both error maps use the same fine FDM reference at common spatial nodes.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from helpers import balancing, implicit_euler
from plot_animation import write_animated_html
from visualize_system import simulate


def compare(d_fine=17, d_coarse=8, r=4, tf=1., steps=1000, input_kind="sine"):
    if not (0 < d_coarse < d_fine) or (d_fine + 1) % (d_coarse + 1):
        raise ValueError("Das grobe Gitter muss im feinen liegen: d_coarse+1 muss d_fine+1 teilen")
    fine = simulate(d_fine, tf, steps, input_kind)
    coarse = simulate(d_coarse, tf, steps, input_kind)
    A, B = fine[:2]
    # Full-state observation makes the reconstructed spatial field the output.
    Ab, Bb, Cb, sigma = balancing(A, B, np.eye(d_fine**2), 1e-14)
    if not 1 <= r <= len(sigma):
        raise ValueError(f"r muss zwischen 1 und {len(sigma)} liegen")
    if input_kind == "sine":
        u = lambda t: 10*np.sin(2*np.pi*t)
    else:
        u = lambda t: 10. if t < 0.2*tf else 0.
    xr = implicit_euler(Ab[:r, :r], Bb[:r], tf, u, np.zeros((r, 1)), steps)
    bt = Cb[:, :r] @ xr
    stride = (d_fine + 1) // (d_coarse + 1)

    def restrict(values):
        field = values.reshape(d_fine, d_fine, steps+1)
        return field[stride-1::stride, stride-1::stride, :].reshape(d_coarse**2, steps+1)

    reference_common = restrict(fine[6])
    bt_common = restrict(bt)
    ef = np.abs(coarse[6] - reference_common)
    eb = np.abs(bt_common - reference_common)
    return dict(fine=fine, coarse=coarse, bt=bt, sigma=sigma, r=r,
                d_fine=d_fine, d_coarse=d_coarse, input_kind=input_kind,
                reference_common=reference_common, bt_common=bt_common,
                error_fdm=ef, error_bt=eb, error_difference=ef-eb,
                max_fdm=np.max(ef, axis=0), max_bt=np.max(eb, axis=0))


def snapshot_fields(result, k):
    """Pad zero boundary values; keep both errors on identical coarse nodes."""
    df, dc = result["d_fine"], result["d_coarse"]
    return [
        np.pad(result["fine"][6][:, k].reshape(df, df), 1),
        np.pad(result["coarse"][6][:, k].reshape(dc, dc), 1),
        np.pad(result["bt"][:, k].reshape(df, df), 1),
        np.pad(result["error_fdm"][:, k].reshape(dc, dc), 1),
        np.pad(result["error_bt"][:, k].reshape(dc, dc), 1),
        np.pad(result["error_difference"][:, k].reshape(dc, dc), 1),
    ]


def error_titles(result, k):
    return [
        f"FDM: |z_grob − z_ref|<br><sup>max = {result['max_fdm'][k]:.3g}</sup>",
        f"BT: |z_BT − z_ref|<br><sup>max = {result['max_bt'][k]:.3g}</sup>",
        "Δe = |e_FDM| − |e_BT|<br><sup>Positiv: BT genauer · Negativ: FDM genauer</sup>",
    ]


def build_comparison(result):
    df, dc, r = result["d_fine"], result["d_coarse"], result["r"]
    times = result["fine"][5]
    gf, gc = np.linspace(0, 1, df+2), np.linspace(0, 1, dc+2)
    indices = np.unique(np.linspace(0, len(times)-1, min(81, len(times)), dtype=int))
    initial = int(indices[np.argmax(np.max(np.abs(result["fine"][6][:, indices]), axis=0))])
    fields = snapshot_fields(result, initial)
    zmax = max(float(np.max(np.abs(values))) for values in (result["fine"][6], result["coarse"][6], result["bt"])) or 1.
    emax = max(float(np.max(result["error_fdm"])), float(np.max(result["error_bt"]))) or 1.
    delta_max = float(np.max(np.abs(result["error_difference"]))) or 1.
    titles = [f"Referenz: feine FDM · n={df**2}", f"Grobe FDM · n={dc**2}", f"Balanced Truncation · r={r}"]
    fig = make_subplots(rows=3, cols=3,
        specs=[[{"type":"surface"}]*3, [{}, {}, {}], [{"colspan":3}, None, None]],
        subplot_titles=titles + error_titles(result, initial) + ["Maximaler Fehler auf denselben räumlichen Knoten"],
        row_heights=[0.42, 0.34, 0.24], vertical_spacing=0.11, horizontal_spacing=0.09)
    for col, grid in enumerate((gf, gc, gf), 1):
        fig.add_trace(go.Surface(x=grid, y=grid, z=fields[col-1], coloraxis="coloraxis3"), row=1, col=col)
    for col in range(1, 4):
        fig.add_trace(go.Heatmap(x=gc, y=gc, z=fields[col+2], coloraxis="coloraxis" if col<3 else "coloraxis2",
            hovertemplate="x₁=%{x:.3f}<br>x₂=%{y:.3f}<br>Wert=%{z:.6g}<extra></extra>"), row=2, col=col)
    for key, name, color in [("max_fdm", "Grobe FDM", "#2563eb"), ("max_bt", "Balanced Truncation", "#c026d3")]:
        values = result[key]
        # Zero errors at the initial time are omitted on this logarithmic axis.
        fig.add_trace(go.Scatter(x=times, y=np.where(values>0, values, np.nan), name=name,
                                mode="lines", line=dict(color=color)), row=3, col=1)
    for key, color in [("max_fdm", "#2563eb"), ("max_bt", "#c026d3")]:
        fig.add_trace(go.Scatter(x=[times[initial]], y=[result[key][initial]], mode="markers",
                                marker=dict(size=9, color=color), showlegend=False), row=3, col=1)
    frames = []
    for k in indices:
        fields = snapshot_fields(result, k)
        annotations = [a.to_plotly_json() for a in fig.layout.annotations]
        for j, text in enumerate(error_titles(result, k), 3):
            annotations[j]["text"] = text
        frames.append(go.Frame(name=str(k), data=[
            *(go.Surface(z=fields[j]) for j in range(3)),
            *(go.Heatmap(z=fields[j]) for j in range(3, 6)),
            go.Scatter(x=[times[k]], y=[result["max_fdm"][k]]),
            go.Scatter(x=[times[k]], y=[result["max_bt"][k]]),
        ], traces=[0, 1, 2, 3, 4, 5, 8, 9],
            layout=dict(annotations=annotations, meta=dict(time=float(times[k])))))
    fig.frames = frames
    fig.update_layout(
        title=dict(text=f"Finite Differenzen versus Balanced Truncation · t={times[initial]:.3f}"
            f"<br><sup>Referenz {df**2} Zustände · grobe FDM {dc**2} · BT {r} · Eingang: {result['input_kind']}"
            f"<br>Errormaps: gleiche {dc}×{dc} innere Vergleichsknoten · identisches Δt · gemeinsame Fehlerskala · C₂=I</sup>"),
        template="plotly_white", height=1150, margin=dict(l=65, r=85, t=135, b=145),
        uirevision="method-comparison", meta=dict(error_traces=[3,4], error_markers=[8,9]),
        coloraxis=dict(cmin=0, cmax=emax, colorscale="Magma", colorbar=dict(title="|e|", x=0.65, y=0.45, len=0.28, thickness=12)),
        coloraxis2=dict(cmin=-delta_max, cmax=delta_max, colorscale="PiYG", colorbar=dict(title="Δe", x=1.02, y=0.45, len=0.28, thickness=12)),
        coloraxis3=dict(cmin=-zmax, cmax=zmax, colorscale="RdBu_r", colorbar=dict(title="z", x=1.02, y=0.85, len=0.25, thickness=12)),
        legend=dict(orientation="h", y=-0.04, x=0),
        sliders=[dict(active=list(indices).index(initial), currentvalue=dict(prefix="Zeit t = "), pad=dict(t=55),
            steps=[dict(method="skip", label=f"{times[k]:.3f}", args=[str(k)]) for k in indices])],
        updatemenus=[dict(type="buttons", direction="left", x=0, y=-0.16, buttons=[
            dict(label="Abspielen", method="skip", args=["play"]),
            dict(label="Pause", method="skip", args=["pause"]),
        ])],
    )
    for scene in ("scene", "scene2", "scene3"):
        fig.update_layout(**{scene:dict(xaxis_title="x₁", yaxis_title="x₂", zaxis_title="z", zaxis=dict(range=[-zmax,zmax]))})
    for col in range(1, 4):
        fig.update_xaxes(title_text="x₁", row=2, col=col)
        fig.update_yaxes(title_text="x₂", row=2, col=col)
    fig.update_xaxes(title_text="Zeit t", row=3, col=1)
    fig.update_yaxes(title_text="max |e| (log.)", type="log", row=3, col=1)
    # Keep the physical time in the main heading synchronized as well.
    base_title = fig.layout.title.text
    for frame in fig.frames:
        k = int(frame.name)
        frame.layout.title = dict(text=base_title.replace(f"t={times[initial]:.3f}", f"t={times[k]:.3f}", 1))
    return fig


def save_static(result, destination, errors_only=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = int(np.argmax(np.max(np.abs(result["fine"][6]), axis=0)))
    fields = snapshot_fields(result, k)
    time = result["fine"][5][k]
    df, dc = result["d_fine"], result["d_coarse"]
    gf, gc = np.linspace(0, 1, df+2), np.linspace(0, 1, dc+2)
    zmax = max(np.max(np.abs(v)) for v in (result["fine"][6],result["coarse"][6],result["bt"])) or 1.
    emax = max(np.max(result["error_fdm"]),np.max(result["error_bt"])) or 1.
    delta = np.max(np.abs(result["error_difference"])) or 1.
    fig, axes = plt.subplots(1 if errors_only else 2, 3, figsize=(16, 4.8 if errors_only else 9), layout="constrained", squeeze=False)
    fig.suptitle(f"FDM und Balanced Truncation · t={time:.3f} · Referenz n={df**2}, FDM n={dc**2}, BT r={result['r']}")
    settings = [
        (gf, fields[0], "Feine FDM: Referenz", "RdBu_r", -zmax, zmax),
        (gc, fields[1], "Grobe FDM", "RdBu_r", -zmax, zmax),
        (gf, fields[2], "Balanced Truncation: rekonstruiertes Feld", "RdBu_r", -zmax, zmax),
        (gc, fields[3], f"FDM-Fehler · max {result['max_fdm'][k]:.3g}", "magma", 0, emax),
        (gc, fields[4], f"BT-Fehler · max {result['max_bt'][k]:.3g}", "magma", 0, emax),
        (gc, fields[5], "|e_FDM| − |e_BT| · positiv: BT genauer", "PiYG", -delta, delta),
    ]
    if errors_only:
        settings = settings[3:]
    for ax, (grid, values, title, cmap, low, high) in zip(axes.flat, settings):
        im = ax.pcolormesh(grid, grid, values, shading="nearest", cmap=cmap, vmin=low, vmax=high)
        ax.set(title=title, xlabel="x₁", ylabel="x₂", aspect="equal", xlim=(0,1), ylim=(0,1))
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-fine", type=int, default=17)
    parser.add_argument("--d-coarse", type=int, default=8)
    parser.add_argument("--r", type=int, default=4)
    parser.add_argument("--tf", type=float, default=1.)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--input", choices=["sine", "pulse"], default="sine")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent/"outputs")
    args = parser.parse_args()
    try:
        result = compare(args.d_fine, args.d_coarse, args.r, args.tf, args.steps, args.input)
    except ValueError as error:
        parser.error(str(error))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"comparison_{args.input}"
    write_animated_html(build_comparison(result), args.output_dir/f"{stem}.html")
    save_static(result, args.output_dir/f"{stem}.png")
    save_static(result, args.output_dir/f"{stem}_errors.png", errors_only=True)
    metrics = dict(reference_states=args.d_fine**2, fdm_states=args.d_coarse**2,
        bt_states=args.r, balanced_states=len(result["sigma"]),
        max_error_fdm=float(np.max(result["error_fdm"])), max_error_bt=float(np.max(result["error_bt"])),
        common_inner_nodes=args.d_coarse**2, time_steps=args.steps,
        reference="fine finite-difference model; not an exact PDE solution")
    (args.output_dir/f"{stem}_metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
    print(json.dumps(metrics, indent=2))
    print(f"Interaktiv: {args.output_dir / (stem + '.html')}")


if __name__ == "__main__":
    main()
