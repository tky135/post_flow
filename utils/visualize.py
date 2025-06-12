import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def build_normalized_histogram_fig(
    data_tensor: torch.Tensor,
    num_bins: int = 100,
    title: str = "Distribution Evolution (Area=1)",
    x_range: tuple[float, float] = None,
):
    """
    Animated histogram where each frame's bars integrate to 1.
    """
    data_array = data_tensor.detach().cpu().numpy()
    T, N = data_array.shape

    # decide fixed x‐range
    if x_range is not None:
        mn, mx = x_range
    else:
        mn, mx = data_array.min(), data_array.max()

    bins = np.linspace(mn, mx, num_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    width = bins[1] - bins[0]

    # compute per‐frame densities (so ∑ density * bin_width = 1)
    hists = [np.histogram(data_array[t], bins=bins, density=True)[0]
             for t in range(T)]

    # find the tallest bin across all frames for y‐axis limits
    # Need to scale by width since we're plotting height * width
    max_density = max((h * width).max() for h in hists)

    # build the plotly figure and frames
    # Scale the heights by bin width so the bars visually integrate to 1
    fig = go.Figure([go.Bar(x=centers, y=hists[0] * width, width=width, opacity=0.8)])
    fig.frames = [
        go.Frame(data=[go.Bar(x=centers, y=hists[t] * width, width=width)], name=str(t))
        for t in range(T)
    ]

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis=dict(title="Value", range=[mn, mx]),
        yaxis=dict(title="Density × Width", range=[0, max_density]),
        updatemenus=[dict(
            type="buttons", showactive=False, y=1.05,
            buttons=[dict(label="► Play", method="animate", args=[
                None,
                {
                    "frame": {"duration": 30, "redraw": False},
                    "transition": {"duration": 0},
                    "fromcurrent": True,
                    "mode": "immediate",
                }
            ])]
        )],
        sliders=[dict(
            pad={"t": 50},
            steps=[dict(method="animate",
                        args=[[str(t)], dict(mode="immediate")],
                        label=str(t)) for t in range(T)]
        )]
    )
    return fig

def build_1d_trajectory_fig(
    trajs: dict[str, list[torch.Tensor]],
    D1_gt_samples: torch.Tensor = None,
    num_particles: int = 20,
    title: str = "1D Trajectories",
    x_range: tuple[float, float] = None,
):
    """
    trajs[name] = list of T tensors of shape [P,1].
    D1_gt_samples: optional tensor [M,1], plotted at y=1.
    x_range: optional (min, max) to fix the x-axis range.
    """
    T = max(len(L) for L in trajs.values())

    # collect global x-values
    all_x = []
    for L in trajs.values():
        for t in L:
            all_x.append(t.detach().cpu().numpy().ravel())
    if D1_gt_samples is not None:
        all_x.append(D1_gt_samples.detach().cpu().numpy().ravel())
    all_x = np.hstack(all_x)

    # Use explicit range if given, else compute from data
    if x_range is not None:
        mn, mx = x_range
    else:
        mn, mx = all_x.min(), all_x.max()

    fig = go.Figure()

    # initial particles at y=0
    for name, L in trajs.items():
        x0 = L[0].detach().cpu().numpy().ravel()
        fig.add_trace(go.Scatter(
            x=x0, y=np.zeros_like(x0),
            mode="markers", marker=dict(size=6),
            name=name
        ))

    # ground-truth at y=1
    if D1_gt_samples is not None:
        gt = D1_gt_samples.detach().cpu().numpy().ravel()
        fig.add_trace(go.Scatter(
            x=gt, y=np.ones_like(gt),
            mode="markers", marker=dict(symbol="x", size=6),
            name="GT", showlegend=True
        ))

    # Create traces for each particle - both the trail and the marker
    trace_indices = {"trails": [], "markers": []}
    for name, L in trajs.items():
        stacked = np.stack([t.detach().cpu().numpy().ravel() for t in L])  # [T,P]
        P = stacked.shape[1]
        
        # Add trail traces (initially just the starting point)
        for i in range(min(num_particles, P)):
            fig.add_trace(go.Scatter(
                x=[stacked[0, i]], 
                y=[0],
                mode="lines",
                line=dict(color="gray", width=1.5),
                opacity=0.5,
                showlegend=False
            ))
            trace_indices["trails"].append(len(fig.data) - 1)
        
        # Add marker traces
        for i in range(min(num_particles, P)):
            fig.add_trace(go.Scatter(
                x=[stacked[0, i]], 
                y=[0],
                mode="markers",
                marker=dict(size=8),
                opacity=1,
                showlegend=False
            ))
            trace_indices["markers"].append(len(fig.data) - 1)
    
    # Create frames that update both trails and markers
    frames = []
    for t in range(T):
        frame_data = []
        
        # Update trails - add points up to current time
        particle_idx = 0
        for name, L in trajs.items():
            if t < len(L):
                stacked = np.stack([L[ti].detach().cpu().numpy().ravel() for ti in range(t+1)])
                P = stacked.shape[1]
                for i in range(min(num_particles, P)):
                    y_vals = [ti / (len(L) - 1) if len(L) > 1 else 0 for ti in range(t+1)]
                    frame_data.append(go.Scatter(
                        x=stacked[:, i],
                        y=y_vals,
                        mode="lines",
                        line=dict(color="gray", width=1.5),
                        opacity=0.5,
                        showlegend=False
                    ))
                    particle_idx += 1
        
        # Update markers - current position only
        particle_idx = 0
        for name, L in trajs.items():
            if t < len(L):
                x = L[t].detach().cpu().numpy().ravel()
                P = len(x)
                for i in range(min(num_particles, P)):
                    y_val = t / (len(L) - 1) if len(L) > 1 else 0
                    frame_data.append(go.Scatter(
                        x=[x[i]],
                        y=[y_val],
                        mode="markers",
                        marker=dict(size=8),
                        opacity=1,
                        showlegend=False
                    ))
                    particle_idx += 1
        
        # Only update the animated traces, not the static ones
        frame = go.Frame(
            data=frame_data,
            name=str(t),
            traces=trace_indices["trails"] + trace_indices["markers"]
        )
        frames.append(frame)
    
    fig.frames = frames

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis=dict(title="Value", range=[mn, mx]),
        yaxis=dict(title="Level", range=[0, 1]),
        updatemenus=[dict(
            type="buttons", showactive=False, y=-0.1,
            buttons=[dict(label="► Play", method="animate", args=[
                None,
                {
                    "frame": {"duration": 30, "redraw": False},
                    "transition": {"duration": 0},
                    "fromcurrent": True,
                    "mode": "immediate",
                }
            ])]
        )],
        sliders=[dict(
            pad={"t": 20},
            steps=[dict(method="animate",
                        args=[[str(t)], dict(mode="immediate")],
                        label=str(t)) for t in range(T)]
        )]
    )
    return fig

def visualize_combined(
    data_tensor: torch.Tensor,
    trajs: dict[str, list[torch.Tensor]],
    D1_gt_samples: torch.Tensor = None,
    save_path: str = "combined.html",
    x_range: tuple[float, float] = None,
    titles: tuple[str, str] = ("Normalized Histogram", "1D Trajectories")
):
    """
    Vertically stacks the normalized-histogram and 1D-trajectory animations.
    x_range: optional (min, max) to apply to both subplots.
    """
    # Pass the same x_range into both builders
    fh = build_normalized_histogram_fig(data_tensor, x_range=x_range)
    ft = build_1d_trajectory_fig(trajs, D1_gt_samples, x_range=x_range)

    combined = make_subplots(
        rows=2, cols=1, vertical_spacing=0.12,
        subplot_titles=titles
    )
    
    # Track how many traces belong to histogram
    histogram_trace_count = len(fh.data)
    
    # add all static traces
    for tr in fh.data:
        combined.add_trace(tr, row=1, col=1)
    for tr in ft.data:
        combined.add_trace(tr, row=2, col=1)

    # merge frames - need to adjust trace indices for trajectory animation
    frames = []
    for t in range(len(fh.frames)):
        # Get trajectory frame and adjust trace indices if present
        traj_frame = ft.frames[t]
        
        # Adjust trace indices for the combined figure
        if hasattr(traj_frame, 'traces') and traj_frame.traces is not None:
            # Offset trajectory trace indices by the number of histogram traces
            adjusted_traces = [histogram_trace_count + idx for idx in traj_frame.traces]
            frame = go.Frame(
                data=fh.frames[t].data + traj_frame.data,
                name=str(t),
                traces=[0] + adjusted_traces  # [0] for histogram, adjusted indices for trajectory
            )
        else:
            frame = go.Frame(
                data=fh.frames[t].data + traj_frame.data,
                name=str(t)
            )
        
        frames.append(frame)
    combined.frames = frames

    # enforce the same x_range on both panels
    mn, mx = x_range if x_range is not None else fh.layout.xaxis.range
    combined.update_xaxes(range=[mn, mx], row=1, col=1)
    combined.update_xaxes(range=[mn, mx], row=2, col=1)
    # y-ranges are fixed
    # combined.update_yaxes(range=[0, 1], row=1, col=1, title_text="Density")
    combined.update_yaxes(range=[0, 1], row=2, col=1, title_text="Level")

    combined.update_layout(
        height=650, width=600,
        updatemenus=[dict(type="buttons", showactive=False,
            buttons=[dict(label="► Play", method="animate", args=[
                None,
                {
                    "frame": {"duration": 30, "redraw": False},
                    "transition": {"duration": 0},
                    "fromcurrent": True,
                    "mode": "immediate",
                }
            ])]
        )],
        sliders=[dict(
            pad={"t": 40},
            steps=[dict(method="animate",
                        args=[[str(t)], dict(mode="immediate")],
                        label=str(t)) for t in range(len(fh.frames))]
        )]
    )

    combined.write_html(save_path, include_plotlyjs="cdn", auto_play=False)
    print(f"Saved animation to {save_path}")