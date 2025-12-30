import numpy as np
import matplotlib.pyplot as plt
import os

# Standard color mapping for HSI classification
colors = [
    [0, 0, 1], [255, 128, 0], [127, 255, 0], [0, 255, 0], [0, 0, 255],
    [46, 139, 87], [255, 0, 255], [0, 255, 255], [255, 255, 255],
    [160, 82, 45], [160, 32, 240], [255, 127, 80], [218, 112, 214],
    [255, 0, 0], [255, 255, 0], [127, 255, 212], [216, 191, 216]
]

def data_to_colormap(data):
    """Convert class indices to RGB colors."""
    assert len(data.shape) == 2
    x_list = data.reshape((-1,))
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if item < len(colors):
            y[index] = np.array(colors[item]) / 255
    return y

def plot_classification_map(prediction_map, ground_truth, save_path, dpi=300):
    """Generate and save a classification map."""
    prediction_map = prediction_map.astype(np.int8)
    h, w = ground_truth.shape
    color_map = data_to_colormap(prediction_map).reshape((h, w, 3))
    
    fig = plt.figure(frameon=False)
    fig.set_size_inches(w*2.0/dpi, h*2.0/dpi)
    
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)
    
    
    ax.imshow(color_map)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

def save_classification_maps(dataset_name, model_name, predictions, ground_truth, save_dir='classification_maps_new'):
    """Save both prediction and ground truth classification maps."""
    os.makedirs(f'{save_dir}/{dataset_name}', exist_ok=True)
    
    # Check if ground truth map already exists
    gt_map_path = f'{save_dir}/{dataset_name}/{dataset_name}_ground_truth.png'
    
    # Generate ground truth map only if it doesn't exist
    if not os.path.exists(gt_map_path):
        print(f"Generating ground truth map for {dataset_name}...")
        plot_classification_map(
            ground_truth,
            ground_truth,
            gt_map_path
        )

    # Save predicted Classification map
    plot_classification_map(
        predictions,
        ground_truth,
        f'{save_dir}/{dataset_name}/{dataset_name}_{model_name}_prediction.png'
    ) 