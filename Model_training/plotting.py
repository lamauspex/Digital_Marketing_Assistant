
""" Отображение графиков """


import matplotlib.pyplot as plt


def plot_metrics(train_losses, val_losses):
    """Строит график изменений потерь."""
    epochs = range(len(train_losses))

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 1, 1)
    plt.plot(
        epochs,
        train_losses,
        label='Train Loss',
        color='blue'
    )
    plt.plot(
        epochs,
        val_losses,
        label='Validation Loss',
        color='orange'
    )
    plt.title('Loss during Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
