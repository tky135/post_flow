import numpy as np
import torch
import plotly.graph_objects as go


def visualize_distribution_evolution(
    data_tensor: torch.Tensor,
    save_path: str,
    num_bins: int = 30,
    title: str = "Distribution Evolution Over Time",
    x_range: tuple = None,
    y_range: tuple = None,
    alpha: float = 0.7,
    color: str = "#1E90FF",
    auto_open: bool = False,
    include_plotlyjs: str = 'cdn'
):
    """
    Visualize the evolution of a distribution over time as an animated histogram.
    
    Parameters:
    -----------
    data_tensor : torch.Tensor
        Tensor of shape [T, N] where T is timesteps and N is samples per timestep
    save_path : str
        Path where to save the HTML file (e.g., "output/animation.html")
    num_bins : int
        Number of histogram bins
    title : str
        Title of the visualization
    x_range : tuple
        Fixed x-axis range (min, max). If None, auto-scale based on data
    y_range : tuple
        Fixed y-axis range (min, max). If None, auto-scale based on data
    alpha : float
        Opacity of histogram bars (0-1)
    color : str
        Color of histogram bars (hex color or color name)
    auto_open : bool
        Whether to automatically open the HTML file in browser
    include_plotlyjs : str
        'cdn' for online viewing, 'inline' for offline viewing
    
    Returns:
    --------
    str : Path where the HTML file was saved
    """
    
    # Convert tensor to numpy
    data_np = data_tensor.detach().cpu().numpy()
    T, N = data_np.shape
    
    # Find global min/max for consistent binning
    global_min = np.min(data_np)
    global_max = np.max(data_np)
    
    # Set x_range if not provided
    if x_range is None:
        padding = 0.1 * (global_max - global_min)
        x_range = (global_min - padding, global_max + padding)
    
    # Create bins
    bin_edges = np.linspace(x_range[0], x_range[1], num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    
    # Calculate histograms for all time steps
    histograms = []
    max_density = 0
    
    for t in range(T):
        counts, _ = np.histogram(data_np[t], bins=bin_edges, density=True)
        histograms.append(counts)
        max_density = max(max_density, np.max(counts))
    
    # Set y_range if not provided
    if y_range is None:
        y_range = (0, max_density * 1.1)
    
    # Create figure
    fig = go.Figure()
    
    # Add initial histogram (t=0)
    fig.add_trace(
        go.Bar(
            x=bin_centers,
            y=histograms[0],
            name="Distribution",
            width=bin_width,
            marker_color=color,
            opacity=alpha,
            hovertemplate='Value: %{x:.3f}<br>Density: %{y:.3f}<extra></extra>'
        )
    )
    
    # Create frames for animation
    frames = []
    for t in range(T):
        frame = go.Frame(
            data=[
                go.Bar(
                    x=bin_centers,
                    y=histograms[t],
                    width=bin_width,
                    marker_color=color,
                    opacity=alpha,
                    hovertemplate='Value: %{x:.3f}<br>Density: %{y:.3f}<extra></extra>'
                )
            ],
            name=str(t),
            traces=[0]
        )
        frames.append(frame)
    
    # Create slider steps
    slider_steps = []
    for t in range(T):
        step = dict(
            method="animate",
            args=[
                [str(t)],
                dict(
                    mode="immediate",
                    frame=dict(duration=0, redraw=True),
                    transition=dict(duration=0),
                )
            ],
            label=f"t={t}"
        )
        slider_steps.append(step)
    
    # Create slider
    sliders = [
        dict(
            active=0,
            currentvalue=dict(
                prefix="Time Step: ",
                visible=True,
                xanchor="right"
            ),
            pad=dict(b=10, t=50),
            len=0.9,
            x=0.1,
            xanchor="left",
            y=0,
            yanchor="top",
            steps=slider_steps
        )
    ]
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            font=dict(size=20)
        ),
        xaxis=dict(
            title="Value",
            range=x_range,
            showgrid=True,
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='gray',
            zerolinewidth=1
        ),
        yaxis=dict(
            title="Density",
            range=y_range,
            showgrid=True,
            gridcolor='lightgray'
        ),
        sliders=sliders,
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.1,
                xanchor="right",
                y=0.02,
                yanchor="bottom",
                pad=dict(t=0, r=10),
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=500, redraw=True),
                                transition=dict(duration=300),
                                fromcurrent=True,
                                mode="immediate"
                            )
                        ]
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                mode="immediate",
                                transition=dict(duration=0)
                            )
                        ]
                    )
                ]
            )
        ],
        margin=dict(l=80, r=80, t=100, b=100),
        height=600,
        width=900,
        template="plotly_white",
        showlegend=False
    )
    
    # Add frames to figure
    fig.frames = frames
    
    # Add annotation with statistics
    stats_text = f"Samples per timestep: {N}<br>Total timesteps: {T}"
    fig.add_annotation(
        text=stats_text,
        xref="paper", yref="paper",
        x=0.98, y=0.98,
        xanchor="right", yanchor="top",
        showarrow=False,
        bordercolor="gray",
        borderwidth=1,
        borderpad=4,
        bgcolor="white",
        opacity=0.8,
        font=dict(size=12)
    )
    
    # Save to HTML
    fig.write_html(
        save_path,
        auto_open=auto_open,
        include_plotlyjs=include_plotlyjs,
        config={
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'distribution_frame',
                'height': 600,
                'width': 900,
                'scale': 2
            }
        }
    )
    
    print(f"Animation saved to: {save_path}")
    return save_path


# Example usage
if __name__ == "__main__":
    # Create example data: normal distribution shifting mean over time
    T = 30  # timesteps
    N = 1000  # samples per timestep
    
    # Generate synthetic data
    data_list = []
    for t in range(T):
        # Mean shifts from -2 to 2 over time
        mean = -2 + (4 * t / (T - 1))
        # Generate samples from normal distribution
        samples = torch.randn(N) + mean
        data_list.append(samples)
    
    # Stack into tensor of shape [T, N]
    data_tensor = torch.stack(data_list)
    
    # Visualize and save
    output_path = visualize_distribution_evolution(
        data_tensor=data_tensor,
        save_path="distribution_animation.html",
        num_bins=40,
        title="Normal Distribution with Shifting Mean",
        color="#FF6B6B",
        alpha=0.8
    )
    
    # Example with custom range
    output_path2 = visualize_distribution_evolution(
        data_tensor=data_tensor,
        save_path="distribution_fixed_range.html",
        num_bins=50,
        title="Distribution Evolution (Fixed Range)",
        x_range=(-5, 5),
        y_range=(0, 0.5),
        color="#4ECDC4",
        include_plotlyjs='inline'  # For offline viewing
    )