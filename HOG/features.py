import numpy as np
from PIL import Image

def compute_hog(image: np.ndarray, cell_size: int, block_size: int, bins: int) -> np.ndarray:
    """Compute a basic HOG descriptor for a single grayscale image."""
    if image.ndim != 2:
        raise ValueError("HOG expects a single-channel (grayscale) image")

    h = (image.shape[0] // cell_size) * cell_size
    w = (image.shape[1] // cell_size) * cell_size
    cropped = image[:h, :w].astype(np.float32) / 255.0
    
    if h == 0 or w == 0:
        return np.zeros(0, dtype=np.float32)

    gy, gx = np.gradient(cropped)
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = (np.rad2deg(np.arctan2(gy, gx)) + 180.0) % 180.0

    n_cells_y = h // cell_size
    n_cells_x = w // cell_size
    hist = np.zeros((n_cells_y, n_cells_x, bins), dtype=np.float32)

    for cell_y in range(n_cells_y):
        row_start = cell_y * cell_size
        row_end = row_start + cell_size
        for cell_x in range(n_cells_x):
            col_start = cell_x * cell_size
            col_end = col_start + cell_size
            cell_mag = magnitude[row_start:row_end, col_start:col_end]
            cell_ori = orientation[row_start:row_end, col_start:col_end]
            hist_vals, _ = np.histogram(
                cell_ori, bins=bins, range=(0.0, 180.0), weights=cell_mag
            )
            hist[cell_y, cell_x, :] = hist_vals

    n_blocks_y = n_cells_y - block_size + 1
    n_blocks_x = n_cells_x - block_size + 1
    
    if n_blocks_x <= 0 or n_blocks_y <= 0:
        return np.zeros(0, dtype=np.float32)

    features = []
    eps = 1e-6
    for by in range(n_blocks_y):
        for bx in range(n_blocks_x):
            block = hist[by : by + block_size, bx : bx + block_size, :]
            block_vector = block.ravel()
            norm = np.linalg.norm(block_vector) + eps
            features.append(block_vector / norm)

    return np.concatenate(features, axis=0) if features else np.zeros(0, dtype=np.float32)

def compute_hog_spm(image: np.ndarray, cell_size: int, block_size: int, bins: int, spm_levels: list) -> np.ndarray:
    """Compute HOG with Spatial Pyramid Matching."""
    h, w = image.shape
    features = []

    for level in spm_levels:
        num_cells = 2**level
        step_y = h // num_cells
        step_x = w // num_cells

        for i in range(num_cells):
            for j in range(num_cells):
                y0 = i * step_y
                y1 = h if i == num_cells - 1 else (i + 1) * step_y
                x0 = j * step_x
                x1 = w if j == num_cells - 1 else (j + 1) * step_x

                region = image[y0:y1, x0:x1]
                hog_region = compute_hog(region, cell_size, block_size, bins)
                features.append(hog_region)

    return np.concatenate(features, axis=0)

def process_single_image(img_path, cell_size, block_size, bins, spm_levels):
    """Worker function for parallel processing."""
    try:
        img = Image.open(img_path).convert("L").resize((224, 224))
        img = np.asarray(img, dtype=np.float32)

        if spm_levels:
            return compute_hog_spm(img, cell_size, block_size, bins, spm_levels)
        else:
            return compute_hog(img, cell_size, block_size, bins)
    except Exception as e:
        print(f"[WARN] Corrupt image {img_path}: {e}")
        return None