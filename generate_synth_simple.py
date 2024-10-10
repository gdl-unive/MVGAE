import numpy as np


def simple():
    g = np.array([[1, 2], [2, 3]])
    g2 = np.array([[1, 2], [2, 3], [1, 3]])

    with open('data/SYNTH_SIMPLE/raw/SYNTH_SIMPLE_A.txt', 'w') as f:
        for row in np.array([[g + (i * 3)] for i in range(100)]).reshape(100 * g.shape[0], 2):
            f.write(f'{row[0]},{row[1]}\n')
        offset = 100 * 3
        for row in np.array([[g2 + (i * 3) + offset] for i in range(100)]).reshape(100 * g2.shape[0], 2):
            f.write(f'{row[0]},{row[1]}\n')

    with open('data/SYNTH_SIMPLE/raw/SYNTH_SIMPLE_graph_indicator.txt', 'w') as f:
        for i in range(1, 101):
            f.write(f'{i}\n{i}\n{i}\n')
        for i in range(101, 201):
            f.write(f'{i}\n{i}\n{i}\n')

    with open('data/SYNTH_SIMPLE/raw/SYNTH_SIMPLE_graph_labels.txt', 'w') as f:
        for i in range(100):
            f.write(f'1\n')
        for i in range(100):
            f.write(f'2\n')

    with open('data/SYNTH_SIMPLE/raw/SYNTH_SIMPLE_node_labels.txt', 'w') as f:
        for i in range(100):
            f.write(f'1\n2\n1\n')
        for i in range(100):
            f.write(f'2\n2\n2\n')


def simple_diff_node_feat():
    g = np.array([[1, 2], [2, 3]])
    g2 = np.array([[1, 2], [2, 3], [1, 3]])

    with open('data/SYNTH_SIMPLE/raw/SYNTH_SIMPLE_A.txt', 'w') as f:
        for row in np.array([[g + (i * 3)] for i in range(100)]).reshape(100 * g.shape[0], 2):
            f.write(f'{row[0]},{row[1]}\n')
        offset = 100 * 3
        for row in np.array([[g2 + (i * 3) + offset] for i in range(100)]).reshape(100 * g2.shape[0], 2):
            f.write(f'{row[0]},{row[1]}\n')

    with open('data/SYNTH_SIMPLE/raw/SYNTH_SIMPLE_graph_indicator.txt', 'w') as f:
        for i in range(1, 101):
            f.write(f'{i}\n{i}\n{i}\n')
        for i in range(101, 201):
            f.write(f'{i}\n{i}\n{i}\n')

    with open('data/SYNTH_SIMPLE/raw/SYNTH_SIMPLE_graph_labels.txt', 'w') as f:
        for i in range(100):
            f.write(f'1\n')
        for i in range(100):
            f.write(f'2\n')

    with open('data/SYNTH_SIMPLE/raw/SYNTH_SIMPLE_node_labels.txt', 'w') as f:
        for i in range(100):
            f.write(f'3\n4\n3\n')
        for i in range(100):
            f.write(f'4\n4\n4\n')


def simple_star_square():
    g = np.array([[1, 2], [1, 3], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5], [3, 4], [3, 5], [4, 5]])
    g2 = np.array([[1, 2], [2, 3], [3, 4], [1, 4]])

    with open('data/SYNTH_SIMPLE/raw/SYNTH_SIMPLE_A.txt', 'w') as f:
        for row in np.array([[g + (i * 5)] for i in range(100)]).reshape(100 * g.shape[0], 2):
            f.write(f'{row[0]},{row[1]}\n')
        offset = 100 * 5
        for row in np.array([[g2 + (i * 4) + offset] for i in range(100)]).reshape(100 * g2.shape[0], 2):
            f.write(f'{row[0]},{row[1]}\n')

    with open('data/SYNTH_SIMPLE/raw/SYNTH_SIMPLE_graph_indicator.txt', 'w') as f:
        for i in range(1, 101):
            f.write(f'{i}\n' * 5)
        for i in range(101, 201):
            f.write(f'{i}\n' * 4)

    with open('data/SYNTH_SIMPLE/raw/SYNTH_SIMPLE_graph_labels.txt', 'w') as f:
        for i in range(100):
            f.write(f'1\n')
        for i in range(100):
            f.write(f'2\n')

    with open('data/SYNTH_SIMPLE/raw/SYNTH_SIMPLE_node_labels.txt', 'w') as f:
        for i in range(100):
            f.write(f'1\n' * len(g))
        for i in range(100):
            f.write(f'2\n' * len(g2))


if __name__ == '__main__':
    simple_star_square()
