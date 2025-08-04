# 🧠 Machine Learning Demo Application

An interactive terminal application demonstrating the four fundamental categories of machine learning, including **supervised**, **unsupervised**, **reinforcement**, and **semi-supervised learning**. Each demo includes step-by-step explanations, live data processing, and performance analysis—all from your terminal!

---

## 🚀 Features

- **Supervised Learning**  
  Logistic Regression for breast cancer detection

- **Unsupervised Learning**  
  K-Means clustering on the Iris dataset

- **Reinforcement Learning**  
  Q-Learning to solve the CartPole balancing challenge

- **Semi-Supervised Learning**  
  Label Propagation on partially-labeled Iris samples

- **User-Friendly Menu**  
  Simple numbered menu to select your learning journey

- **Educational Output**  
  Clear terminal feedback on loading, preprocessing, training, evaluation, and interpretation

---

## 📸 Screenshots or Illustrations

*(Optional: Insert screenshots or ASCII art samples of menus or output for visual impact.)*

---

## 🛠️ Installation

**Requirements:**
- Python 3.7 or later
- numpy
- pandas
- scikit-learn
- gymnasium **OR** gym (for reinforcement learning)

**Install dependencies (recommended):**


---

## ▶️ Usage

1. **Clone or download this repository:**
    ```
    git clone https://github.com/yourusername/ml-demo.git
    cd ml-demo
    ```

2. **Run the application:**
    ```
    python ml-demo.py
    ```

3. **Choose a demo:**  
   Use keys `1` to `4` to explore, `5` to exit.

---

## 📚 Machine Learning Demos Included

| Type                     | Example Task       | Dataset/Env         | Algorithm      | Goal                                    |
|--------------------------|-------------------|---------------------|---------------|-----------------------------------------|
| Supervised Learning      | Cancer detection  | Breast cancer       | LogisticReg.  | Classify benign/malignant tumors        |
| Unsupervised Learning    | Flower grouping   | Iris                | K-Means       | Discover natural clusters               |
| Reinforcement Learning   | CartPole balance  | Gym CartPole        | Q-Learning    | Balance pole via rewards                |
| Semi-Supervised Learning | Partial labeling  | Iris (masked labels)| LabelProp.    | Learn with sparse labels                |

---

## 💡 Educational Purpose

This app is for educational use — for anyone wanting to see, in real time, how core ML tasks are set up, tested, and evaluated!

---

## 🛟 Troubleshooting / Help

- **Missing dependencies?**  
  The program checks for required libraries and notifies which demos might be unavailable.

- **To install missing libraries:**
    ```
    pip install numpy pandas scikit-learn gymnasium
    pip install gym   # if gymnasium unavailable
    ```

- **KeyboardInterrupt (Ctrl+C):**  
   Gracefully exits at any point.

- **Python errors?**  
   Make sure your Python version is 3.7+ and all required packages are installed.

---

## 📝 License

This demo application is open source under the [MIT License](LICENSE).

---

## ✒️ Author

Agile Creative Labs Inc, 2025

---

## 🙏 Acknowledgements

- [scikit-learn](https://scikit-learn.org/)
- [OpenAI Gym](https://www.gymlibrary.dev/) / [Gymnasium](https://gymnasium.farama.org/)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/) (datasets)



