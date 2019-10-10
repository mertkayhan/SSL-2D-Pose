import numpy as np


def compute_integral(pt1, pt2, connectivity):
    # (y, x)

    # get the points on the connecting line
    points, num_points = get_points(pt1, pt2)

    # integral
    try:
        score = connectivity[points].sum()
    except IndexError:  # basically no connectivity
        score = -200

    return score


def get_line(pt1, pt2):
    # (y, x)

    m = float(pt1[0] - pt2[0]) / float(pt1[1] - pt2[1] + 1e-5)
    b = pt2[0] - m * pt2[1]
    return m, b


def get_points(pt1, pt2):
    # (y, x)

    m, b = get_line(pt1, pt2)
    number_of_samples = abs(pt2[1] - pt1[1]) + 1
    x_s = np.linspace(start=min(pt1[1], pt2[1]), stop=max(pt1[1], pt2[1]), num=number_of_samples)
    y_s = m * x_s + b
    points = [y_s.astype(np.int32), x_s.astype(np.int32)]
    return points, number_of_samples


if __name__ == "__main__":
    pass