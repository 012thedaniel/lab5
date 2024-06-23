import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class NeuralNetwork:
    # Ініціалізація ваг та зсувів
    def __init__(self):
        self.W1 = np.random.normal(size=(9, 784)) * np.sqrt(1 / 784)
        self.B1 = np.random.normal(size=(9, 1)) * np.sqrt(1 / 9)
        self.W2 = np.random.normal(size=(3, 9)) * np.sqrt(1 / 9)
        self.B2 = np.random.normal(size=(3, 1)) * np.sqrt(1 / 3)

    # Функція активації ReLU
    @staticmethod
    def __relu(Z):
        return np.maximum(Z, 0)

    # Похідна функції активації ReLU
    @staticmethod
    def __relu_derivative(Z):
        return (Z > 0).astype(int)

    # Функція активації softmax
    @staticmethod
    def __softmax(Z):
        exps = np.exp(Z)
        return exps / np.sum(exps, axis=0)

    # Пряме поширення
    def __forward_propagation(self, X):
        # Лінійна комбінація ваг та входів для першого шару
        Z1 = self.W1 @ X + self.B1
        # Активація першого шару
        A1 = self.__relu(Z1)
        # Лінійна комбінація ваг та виходів першого шару для другого шару
        Z2 = self.W2 @ A1 + self.B2
        # Активація другого шару
        A2 = self.__softmax(Z2)
        return Z1, A1, Z2, A2

    # Зворотне поширення
    def __backward_propagation(self, X, Z1, A1, A2, A2E):
        # Кількість тренувальних наборів
        num_of_training_sets = A2E.shape[1]
        # Помилка виходу
        dZ2 = A2 - A2E
        # Градієнт для ваг другого шару
        dW2 = (dZ2 @ A1.T) / num_of_training_sets
        # Градієнт для зсувів другого шару
        db2 = np.sum(dZ2, axis=1, keepdims=True) / num_of_training_sets
        # Помилка першого шару
        dZ1 = self.W2.T @ dZ2 * self.__relu_derivative(Z1)
        # Градієнт для ваг першого шару
        dW1 = (dZ1 @ X.T) / num_of_training_sets
        # Градієнт для зсуву першого шару
        db1 = np.sum(dZ1, axis=1, keepdims=True) / num_of_training_sets
        return dW1, db1, dW2, db2

    # Оновлення ваг та зсувів
    def __update_weights(self, dW1, dB1, dW2, dB2, alpha):
        self.W1 = self.W1 - alpha * dW1
        self.B1 = self.B1 - alpha * dB1
        self.W2 = self.W2 - alpha * dW2
        self.B2 = self.B2 - alpha * dB2

    # Формування послідовності очікуваних результатів
    def __get_sequence_of_expected_prediction(self, Y):
        sequence = np.zeros((Y.size, self.B2.size))
        sequence[np.arange(Y.size), Y - 1] = 1
        return sequence.T

    # Тренування нейронної мережі
    def train(self, X, Y, alpha, iterations):
        # Формування послідовності очікуваних результатів
        A2E = self.__get_sequence_of_expected_prediction(Y)
        for _ in range(iterations):
            # Пряме поширення
            Z1, A1, Z2, A2 = self.__forward_propagation(X)
            # Зворотне поширення
            dW1, dB1, dW2, dB2 = self.__backward_propagation(X, Z1, A1, A2, A2E)
            # Оновлення ваг та зсувів
            self.__update_weights(dW1, dB1, dW2, dB2, alpha)

    # Прогнозування наступного значення часового ряду та обрахунок точності класифікації
    def test_nn_accuracy(self, X, Y):
        # Прогнозування наступного значення часового ряду (пряме поширення)
        _, _, _, A2 = self.__forward_propagation(X)
        predictions = np.argmax(A2, 0) + 1
        # Обрахунок точності класифікації
        accuracy = np.sum(predictions == Y) / Y.size
        return accuracy, predictions[:10]


# Візуалізація зображення
def get_img(X, index):
    current_image = X[:, index, None].reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()


if __name__ == '__main__':
    # Завантаження та обробка даних
    data = pd.read_csv('data.csv').to_numpy()
    np.random.shuffle(data)

    # Формування набору тренувальних даних (входи та виходи)
    training_set = data[:-1000].T
    Y_training, X_training = np.split(training_set, [1])
    # Формування набору тестувальних даних (входи та виходи)
    testing_set = data[-1000:].T
    Y_testing, X_testing = np.split(testing_set, [1])

    # Ініціалізація нейронної мережі
    nn = NeuralNetwork()

    # Візуалізація тестового набору даних
    print(f'\nЧисла на зображеннях: {", ".join(list(map(str, Y_testing[0][:10])))}.')
    for i in range(10):
        get_img(X_testing, i)

    # Тестування нейронної мережі до навчання та вивід результатів
    nn_accuracy, predicted_nums = nn.test_nn_accuracy(X_testing / 255., Y_testing)
    print(f'\n\tРозпізнані значення до навчання: {", ".join(list(map(str, predicted_nums)))}')
    print(f'\tТочність розпізнавання чисел до навчання: {round(nn_accuracy * 100, 1)}%')

    # Тренування нейронної мережі
    nn.train(X_training / 255., Y_training, 0.1, 250)

    # Тестування нейронної мережі після навчання та вивід результатів
    nn_accuracy, predicted_nums = nn.test_nn_accuracy(X_testing / 255., Y_testing)
    print(f'\n\tРозпізнані значення після навчання: {", ".join(list(map(str, predicted_nums)))}')
    print(f'\tТочність розпізнавання чисел після навчання: {round(nn_accuracy * 100, 1)}%')
