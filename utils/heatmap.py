import numpy as np


def eval_gaussian(mu, sigma=20., h=480, w=640):

    mu_np = np.array(mu, dtype=np.float32).reshape(1, 2)
    x = np.linspace(0, w-1, w)
    y = np.linspace(0, h-1, h)

    grid = np.array(np.meshgrid(y, x)).reshape(2, -1)
    diff = grid - mu_np.T
    norm_squared = np.sum(diff ** 2, axis=0)
    tmp = norm_squared / (sigma ** 2)
    proba = np.exp(-tmp).reshape(w, h)
    return proba.T


def __gaussian(norm_squared, sigma=20.):
    return np.exp(-norm_squared / (sigma ** 2))


def __circular_mask(grid, center, radius):

    dist_from_center = np.sqrt((grid[0, :] - center[0]) ** 2 + (grid[1, :] - center[1]) ** 2)
    mask = dist_from_center <= radius
    return mask


def eval_line(pt1, pt2, m, b, h=576, w=720, sigma=20):
    # (x, y)

    x = np.linspace(0, w - 1, w)
    y = np.linspace(0, h - 1, h)

    grid = np.array(np.meshgrid(y, x)).reshape(2, -1)
    distance = np.abs(m * grid[1, :] + -1 * grid[0, :] + b) / np.sqrt(m ** 2 + 1)
    heatmap = __gaussian(distance ** 2, sigma=sigma).reshape(w, h)
    center = ((pt1[0] + pt2[0]) / 2, (pt1[1] + pt2[1]) / 2)
    dist = np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
    mask = __circular_mask(grid, center, dist//2 + sigma//2)
    heatmap *= mask.reshape(w, h)

    return heatmap.T


if __name__ == "__main__":
    pass
