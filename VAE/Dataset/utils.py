import matplotlib.pyplot as plt

def visualize_dataloader(batch_array):
    fig, axes = plt.subplots(3, 3)
    for i, ax in enumerate(axes.flat):
        print(batch_array[i].shape)
        ax.imshow(batch_array[i])
        ax.axis('off')
    #TODO: show on the plot info like img size
    plt.tight_layout()
    plt.show()
    