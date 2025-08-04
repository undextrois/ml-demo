#!/usr/bin/env python3
"""
==========================================================================================
Machine Learning Demo Application
==========================================================================================

Description:
    An interactive terminal application demonstrating four fundamental categories of 
    machine learning through hands-on, educational examples:

      1. Supervised Learning (Binary Classification)
            - Logistic Regression on the breast cancer dataset
            - Goal: Predict malignant vs. benign tumors

      2. Unsupervised Learning (Clustering)
            - K-Means clustering on the iris flower dataset
            - Goal: Discover natural groupings without using original labels

      3. Reinforcement Learning (Q-Learning)
            - Teach an agent to solve the CartPole environment from OpenAI Gym/Gymnasium
            - Goal: Learn via trial and error to maximize cumulative reward

      4. Semi-Supervised Learning
            - Label propagation with partially-labeled iris data
            - Goal: Use a small fraction of labeled data to infer labels on unlabeled samples

Features:
    - Textual user interface with explanations at every step
    - Walkthrough of data loading, preprocessing, model training, evaluating, and interpretation
    - Robust error handling and library availability checks
    - Educational notes and detailed metric reporting

Requirements:
    - Python 3.7 or later
    - numpy
    - pandas
    - scikit-learn
    - gymnasium OR gym (for reinforcement learning demo)
    - (All dependencies are open-source and can be installed via pip)

    To install requirements:
      pip install numpy pandas scikit-learn gymnasium

      # If gymnasium is not available, gym can be used instead for RL part:
      pip install gym

Usage:
    Run the application from the terminal:

        python ml-demo.py

    Navigate the menu using numbers 1-5 to select the demo or exit.
    Each demo will display progress, metrics, and conceptual explanations.

Author:
   Agile Creative Labs Inc. (c) 2025

License:
    MIT License

Educational Purpose:
    This script is intended for teaching and demonstration of core ML approaches.
    It is not intended for production use. Please consult the official documentation of libraries
    and research best practices for real-world machine learning projects.

Remember: Each type of ML solves different problems:
   • Supervised: Learn from labeled examples
   • Unsupervised: Find hidden patterns in data
   • Reinforcement: Learn through trial and error
   • Semi-supervised: Learn from mixed labeled/unlabeled data

==========================================================================================
"""

import sys
import warnings
warnings.filterwarnings('ignore')

# Core libraries
import numpy as np
import pandas as pd

# Try importing required libraries with graceful error handling
try:
    from sklearn.datasets import load_breast_cancer, load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.cluster import KMeans
    from sklearn.semi_supervised import LabelPropagation
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                               f1_score, silhouette_score)
    SKLEARN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: scikit-learn not available: {e}")
    SKLEARN_AVAILABLE = False

try:
    import gymnasium as gym
    GYM_AVAILABLE = True
    GYM_LIBRARY = "gymnasium"
except ImportError:
    try:
        import gym
        GYM_AVAILABLE = True
        GYM_LIBRARY = "gym"
    except ImportError:
        print("Warning: Neither gymnasium nor gym available for reinforcement learning demo")
        GYM_AVAILABLE = False


def supervised_learning_demo():
    """
    Demonstrates supervised learning using binary classification on the breast cancer dataset.
    Uses logistic regression to predict malignant vs benign tumors.
    """
    print("\n" + "="*60)
    print("SUPERVISED LEARNING DEMONSTRATION")
    print("="*60)
    print("Task: Binary Classification (Breast Cancer Detection)")
    print("Algorithm: Logistic Regression")
    
    if not SKLEARN_AVAILABLE:
        print("❌ Scikit-learn is not available. Cannot run supervised learning demo.")
        return
    
    try:
        # Data Loading
        print("\n📊 Step 1: Loading Dataset...")
        data = load_breast_cancer()
        X, y = data.data, data.target
        print(f"✅ Dataset loaded successfully!")
        print(f"   - Shape: {X.shape} (samples x features)")
        print(f"   - Classes: {data.target_names}")
        print(f"   - Features: {len(data.feature_names)} features")
        
        # Preprocessing
        print("\n🔧 Step 2: Preprocessing...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print("✅ Features standardized using StandardScaler")
        print(f"   - Mean of scaled features: {np.mean(X_scaled, axis=0)[:3].round(3)}... (first 3)")
        print(f"   - Std of scaled features: {np.std(X_scaled, axis=0)[:3].round(3)}... (first 3)")
        
        # Split Data
        print("\n🔀 Step 3: Splitting Data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"✅ Data split completed!")
        print(f"   - Training set: {X_train.shape[0]} samples")
        print(f"   - Test set: {X_test.shape[0]} samples")
        
        # Model Creation & Training
        print("\n🤖 Step 4: Model Training...")
        print("📝 Creating Logistic Regression model...")
        model = LogisticRegression(random_state=42, max_iter=1000)
        
        print("🎯 Training model on training data...")
        model.fit(X_train, y_train)
        print("✅ Model training completed!")
        
        # Prediction
        print("\n🔮 Step 5: Making Predictions...")
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        print(f"✅ Predictions made for {len(y_test)} test samples")
        
        # Evaluation & Logging
        print("\n📈 Step 6: Model Evaluation...")
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print("🎯 PERFORMANCE METRICS:")
        print(f"   ├─ Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   ├─ Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"   ├─ Recall:    {recall:.4f} ({recall*100:.2f}%)")
        print(f"   └─ F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
        
        print("\n💡 INTERPRETATION:")
        print(f"   The model correctly classified {accuracy*100:.1f}% of breast cancer cases.")
        print(f"   Precision: {precision*100:.1f}% of predicted malignant cases were actually malignant.")
        print(f"   Recall: {recall*100:.1f}% of actual malignant cases were correctly identified.")
        
    except Exception as e:
        print(f"❌ Error in supervised learning demo: {e}")


def unsupervised_learning_demo():
    """
    Demonstrates unsupervised learning using K-means clustering on the iris dataset.
    Ignores the original labels and attempts to discover natural groupings.
    """
    print("\n" + "="*60)
    print("UNSUPERVISED LEARNING DEMONSTRATION")
    print("="*60)
    print("Task: Clustering (Iris Flower Species)")
    print("Algorithm: K-Means Clustering")
    
    if not SKLEARN_AVAILABLE:
        print("❌ Scikit-learn is not available. Cannot run unsupervised learning demo.")
        return
    
    try:
        # Data Loading
        print("\n📊 Step 1: Loading Dataset...")
        data = load_iris()
        X = data.data  # Ignoring labels (y) for unsupervised learning
        print(f"✅ Iris dataset loaded successfully!")
        print(f"   - Shape: {X.shape} (samples x features)")
        print(f"   - Features: {data.feature_names}")
        print("   - Note: Ignoring original labels for unsupervised learning")
        
        # Preprocessing
        print("\n🔧 Step 2: Preprocessing...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print("✅ Features standardized using StandardScaler")
        print(f"   - Original feature ranges varied significantly")
        print(f"   - After scaling: mean ≈ 0, std ≈ 1 for all features")
        
        # Model Creation & Training
        print("\n🤖 Step 3: Clustering...")
        print("📝 Creating K-Means model with 3 clusters...")
        print("   (We expect 3 clusters since iris has 3 species)")
        
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        print("🎯 Training K-Means clustering algorithm...")
        cluster_labels = kmeans.fit_predict(X_scaled)
        print("✅ Clustering completed!")
        
        # Evaluation & Logging
        print("\n📈 Step 4: Cluster Quality Evaluation...")
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        
        print("🎯 CLUSTERING RESULTS:")
        print(f"   ├─ Number of clusters found: {len(np.unique(cluster_labels))}")
        print(f"   ├─ Silhouette Score: {silhouette_avg:.4f}")
        print(f"   └─ Cluster centers shape: {kmeans.cluster_centers_.shape}")
        
        # Analyze cluster distribution
        unique, counts = np.unique(cluster_labels, return_counts=True)
        print("\n📊 CLUSTER DISTRIBUTION:")
        for cluster_id, count in zip(unique, counts):
            print(f"   ├─ Cluster {cluster_id}: {count} samples ({count/len(X)*100:.1f}%)")
        
        print("\n💡 INTERPRETATION:")
        if silhouette_avg > 0.5:
            print("   ✅ Good clustering quality! Clusters are well-separated.")
        elif silhouette_avg > 0.25:
            print("   ⚠️  Moderate clustering quality. Some overlap between clusters.")
        else:
            print("   ❌ Poor clustering quality. Clusters may be poorly defined.")
            
        print(f"   The algorithm discovered {len(unique)} natural groupings in the data.")
        print("   In a real scenario, we could analyze these clusters to understand")
        print("   hidden patterns in the data without knowing the original labels.")
        
    except Exception as e:
        print(f"❌ Error in unsupervised learning demo: {e}")


def reinforcement_learning_demo():
    """
    Demonstrates reinforcement learning using Q-learning on the CartPole environment.
    Teaches an agent to balance a pole on a cart through trial and error.
    """
    print("\n" + "="*60)
    print("REINFORCEMENT LEARNING DEMONSTRATION")
    print("="*60)
    print("Task: Cart-Pole Balancing")
    print("Algorithm: Q-Learning")
    
    if not GYM_AVAILABLE:
        print("❌ Gymnasium/Gym is not available. Cannot run reinforcement learning demo.")
        return
    
    try:
        # Environment Setup
        print("\n🎮 Step 1: Environment Setup...")
        env = gym.make('CartPole-v1')
        print(f"✅ CartPole-v1 environment created using {GYM_LIBRARY}!")
        print("   - Goal: Keep the pole balanced on the cart")
        print("   - Actions: Move cart left (0) or right (1)")
        print("   - State: Cart position, velocity, pole angle, pole velocity")
        print("   - Reward: +1 for each timestep the pole stays upright")
        
        # Q-Learning Parameters
        print("\n🧠 Step 2: Q-Learning Setup...")
        n_episodes = 1000
        max_steps = 200
        learning_rate = 0.1
        discount_factor = 0.95
        epsilon = 1.0  # Exploration rate
        epsilon_decay = 0.995
        epsilon_min = 0.01
        
        # Discretize the continuous state space for Q-table
        n_bins = 10
        state_bounds = [
            (-2.4, 2.4),      # Cart position
            (-3.0, 3.0),      # Cart velocity  
            (-0.5, 0.5),      # Pole angle
            (-2.0, 2.0)       # Pole velocity
        ]
        
        # Initialize Q-table
        q_table = np.zeros([n_bins] * 4 + [env.action_space.n])
        print(f"✅ Q-Learning parameters initialized:")
        print(f"   ├─ Episodes: {n_episodes}")
        print(f"   ├─ Learning rate: {learning_rate}")
        print(f"   ├─ Discount factor: {discount_factor}")
        print(f"   ├─ Q-table shape: {q_table.shape}")
        print(f"   └─ Initial exploration rate: {epsilon}")
        
        def discretize_state(state):
            """Convert continuous state to discrete state for Q-table indexing"""
            discrete_state = []
            for i, val in enumerate(state):
                low, high = state_bounds[i]
                discrete_val = int(np.digitize(val, np.linspace(low, high, n_bins-1)))
                discrete_val = max(0, min(discrete_val, n_bins-1))
                discrete_state.append(discrete_val)
            return tuple(discrete_state)
        
        # Training Loop
        print("\n🎯 Step 3: Training Agent...")
        print("📝 Starting Q-learning training process...")
        
        episode_rewards = []
        for episode in range(n_episodes):
            state, _ = env.reset()
            discrete_state = discretize_state(state)
            total_reward = 0
            
            for step in range(max_steps):
                # Epsilon-greedy action selection
                if np.random.random() < epsilon:
                    action = env.action_space.sample()  # Explore
                else:
                    action = np.argmax(q_table[discrete_state])  # Exploit
                
                # Take action and observe result
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                next_discrete_state = discretize_state(next_state)
                
                # Q-learning update (Bellman equation)
                current_q = q_table[discrete_state + (action,)]
                max_next_q = np.max(q_table[next_discrete_state])
                new_q = current_q + learning_rate * (reward + discount_factor * max_next_q - current_q)
                q_table[discrete_state + (action,)] = new_q
                
                discrete_state = next_discrete_state
                total_reward += reward
                
                if done:
                    break
            
            episode_rewards.append(total_reward)
            
            # Decay exploration rate
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            
            # Progress logging
            if (episode + 1) % 200 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"   Episode {episode + 1:4d}: Avg reward = {avg_reward:6.2f}, Epsilon = {epsilon:.3f}")
        
        print("✅ Training completed!")
        
        # Evaluation
        print("\n📈 Step 4: Evaluating Learned Policy...")
        print("🔮 Running episodes with learned policy (no exploration)...")
        
        test_rewards = []
        for test_episode in range(5):
            state, _ = env.reset()
            discrete_state = discretize_state(state)
            total_reward = 0
            
            for step in range(max_steps):
                # Use learned policy (no exploration)
                action = np.argmax(q_table[discrete_state])
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                discrete_state = discretize_state(next_state)
                total_reward += reward
                
                if done:
                    break
            
            test_rewards.append(total_reward)
            print(f"   Test Episode {test_episode + 1}: Reward = {total_reward}")
        
        env.close()
        
        # Results Summary
        avg_training_reward = np.mean(episode_rewards[-100:])
        avg_test_reward = np.mean(test_rewards)
        
        print("\n🎯 REINFORCEMENT LEARNING RESULTS:")
        print(f"   ├─ Average training reward (last 100 episodes): {avg_training_reward:.2f}")
        print(f"   ├─ Average test reward: {avg_test_reward:.2f}")
        print(f"   └─ Maximum possible reward per episode: {max_steps}")
        
        print("\n💡 INTERPRETATION:")
        print("   Reinforcement Learning teaches agents through trial and error.")
        print("   The agent learned to balance the pole by receiving rewards (+1)")
        print("   for each timestep the pole stayed upright, and learning from")
        print("   the consequences of its actions using the Q-learning algorithm.")
        
        if avg_test_reward > 150:
            print("   ✅ Great performance! The agent learned to balance the pole well.")
        elif avg_test_reward > 100:
            print("   ⚠️  Moderate performance. The agent learned some balancing skills.")
        else:
            print("   ❌ Poor performance. The agent needs more training or parameter tuning.")
            
    except Exception as e:
        print(f"❌ Error in reinforcement learning demo: {e}")


def semi_supervised_learning_demo():
    """
    Demonstrates semi-supervised learning using label propagation on partially labeled iris data.
    Shows how models can learn from both labeled and unlabeled data.
    """
    print("\n" + "="*60)
    print("SEMI-SUPERVISED LEARNING DEMONSTRATION")
    print("="*60)
    print("Task: Classification with Partial Labels (Iris)")
    print("Algorithm: Label Propagation")
    
    if not SKLEARN_AVAILABLE:
        print("❌ Scikit-learn is not available. Cannot run semi-supervised learning demo.")
        return
    
    try:
        # Data Loading
        print("\n📊 Step 1: Loading Dataset...")
        data = load_iris()
        X, y_full = data.data, data.target
        print(f"✅ Iris dataset loaded successfully!")
        print(f"   - Shape: {X.shape} (samples x features)")
        print(f"   - Classes: {data.target_names}")
        print(f"   - Total labeled samples available: {len(y_full)}")
        
        # Creating Partial Labels
        print("\n🏷️  Step 2: Creating Partially Labeled Dataset...")
        np.random.seed(42)
        y_partial = y_full.copy()
        
        # Randomly set 70% of labels to -1 (unlabeled)
        unlabel_ratio = 0.7
        n_unlabeled = int(len(y_partial) * unlabel_ratio)
        unlabel_indices = np.random.choice(len(y_partial), n_unlabeled, replace=False)
        y_partial[unlabel_indices] = -1
        
        n_labeled = len(y_partial) - n_unlabeled
        print(f"✅ Partial labeling completed!")
        print(f"   ├─ Labeled samples: {n_labeled} ({(1-unlabel_ratio)*100:.0f}%)")
        print(f"   ├─ Unlabeled samples: {n_unlabeled} ({unlabel_ratio*100:.0f}%)")
        print(f"   └─ Unlabeled samples marked with -1")
        
        # Show label distribution
        labeled_mask = y_partial != -1
        if np.sum(labeled_mask) > 0:
            unique_labels, label_counts = np.unique(y_partial[labeled_mask], return_counts=True)
            print("\n📊 LABELED DATA DISTRIBUTION:")
            for label, count in zip(unique_labels, label_counts):
                class_name = data.target_names[int(label)]
                print(f"   ├─ {class_name}: {count} samples")
        
        # Preprocessing
        print("\n🔧 Step 3: Preprocessing...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print("✅ Features standardized using StandardScaler")
        
        # Model Creation & Training
        print("\n🤖 Step 4: Semi-Supervised Learning...")
        print("📝 Creating Label Propagation model...")
        print("   This model will learn from the small amount of labeled data")
        print("   and propagate labels to similar unlabeled samples.")
        
        model = LabelPropagation(kernel='knn', n_neighbors=7, max_iter=1000)
        print("\n🎯 Training model on partially labeled data...")
        model.fit(X_scaled, y_partial)
        print("✅ Model training completed!")
        print("   The model learned patterns from labeled data and inferred")
        print("   labels for unlabeled samples based on similarity.")
        
        # Prediction and Evaluation
        print("\n🔮 Step 5: Label Propagation Results...")
        y_pred = model.predict(X_scaled)
        
        # Calculate accuracy against the original full labels
        accuracy = accuracy_score(y_full, y_pred)
        
        # Analyze predictions on originally unlabeled data
        unlabeled_accuracy = accuracy_score(y_full[unlabel_indices], y_pred[unlabel_indices])
        labeled_accuracy = accuracy_score(y_full[labeled_mask], y_pred[labeled_mask])
        
        print("🎯 SEMI-SUPERVISED LEARNING RESULTS:")
        print(f"   ├─ Overall accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   ├─ Accuracy on originally labeled data: {labeled_accuracy:.4f}")
        print(f"   └─ Accuracy on originally unlabeled data: {unlabeled_accuracy:.4f}")
        
        # Show prediction distribution
        unique_pred, pred_counts = np.unique(y_pred, return_counts=True)
        print("\n📊 FINAL PREDICTION DISTRIBUTION:")
        for pred_label, count in zip(unique_pred, pred_counts):
            class_name = data.target_names[int(pred_label)]
            print(f"   ├─ {class_name}: {count} samples")
        
        # Compare with original distribution
        unique_orig, orig_counts = np.unique(y_full, return_counts=True)
        print("\n📊 ORIGINAL DISTRIBUTION (for comparison):")
        for orig_label, count in zip(unique_orig, orig_counts):
            class_name = data.target_names[int(orig_label)]
            print(f"   ├─ {class_name}: {count} samples")
        
        print("\n💡 INTERPRETATION:")
        print("   Semi-supervised learning leverages both labeled and unlabeled data.")
        print(f"   Using only {(1-unlabel_ratio)*100:.0f}% labeled data, the model achieved")
        print(f"   {accuracy*100:.1f}% accuracy on the complete dataset.")
        print("   This is particularly useful when labeling data is expensive or time-consuming.")
        
        if accuracy > 0.9:
            print("   ✅ Excellent performance! The model successfully propagated labels.")
        elif accuracy > 0.7:
            print("   ⚠️  Good performance. The model learned meaningful patterns.")
        else:
            print("   ❌ Limited performance. More labeled data or different approach needed.")
            
    except Exception as e:
        print(f"❌ Error in semi-supervised learning demo: {e}")


def main_menu():
    """
    Main interactive menu for the machine learning demonstration application.
    Provides options to run different types of ML demos or exit the program.
    """
    while True:
        print("\n" + "="*60)
        print("🤖 MACHINE LEARNING DEMONSTRATION MENU")
        print("="*60)
        print("Select a machine learning type to explore:")
        print()
        print("1. 🎯 Supervised Learning (Binary Classification)")
        print("   └─ Breast Cancer Detection using Logistic Regression")
        print()
        print("2. 🔍 Unsupervised Learning (Clustering)")
        print("   └─ Iris Flower Clustering using K-Means")
        print()
        print("3. 🎮 Reinforcement Learning (Q-Learning)")
        print("   └─ Cart-Pole Balancing using Q-Learning")
        print()
        print("4. 🏷️  Semi-Supervised Learning (Label Propagation)")
        print("   └─ Iris Classification with Partial Labels")
        print()
        print("5. ❌ Exit")
        print()
        
        try:
            choice = input("Enter your choice (1-5): ").strip()
            
            if choice == '1':
                supervised_learning_demo()
            elif choice == '2':
                unsupervised_learning_demo()
            elif choice == '3':
                reinforcement_learning_demo()
            elif choice == '4':
                semi_supervised_learning_demo()
            elif choice == '5':
                print("\n👋 Thank you for exploring machine learning!")
                print("   Remember: Each type of ML solves different problems:")
                print("   • Supervised: Learn from labeled examples")
                print("   • Unsupervised: Find hidden patterns in data")
                print("   • Reinforcement: Learn through trial and error")
                print("   • Semi-supervised: Learn from mixed labeled/unlabeled data")
                print("\n🚀 Keep learning and exploring! Goodbye!")
                break
            else:
                print("❌ Invalid choice! Please enter a number between 1 and 5.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Program interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            print("Please try again or exit the program.")
        
        # Pause before showing menu again
        input("\n📱 Press Enter to return to the main menu...")


if __name__ == "__main__":
    """
    Main entry point of the ML demonstration application.
    """
    print("🤖 Machine Learning Demo Application")
    print("=====================================")
    print("Welcome! This application demonstrates four fundamental types of machine learning.")
    print()
    
    # Check library availability
    missing_libs = []
    if not SKLEARN_AVAILABLE:
        missing_libs.append("scikit-learn")
    if not GYM_AVAILABLE:
        missing_libs.append("gymnasium/gym")
    
    if missing_libs:
        print("⚠️  Warning: Some libraries are missing:")
        for lib in missing_libs:
            print(f"   - {lib}")
        print("\nTo install missing libraries:")
        print("   pip install scikit-learn gymnasium numpy")
        print("\nSome demos may not be available, but others will still work!")
    else:
        print("✅ All required libraries are available!")
    
    print("\n🎓 Educational Note:")
    print("Each demo includes detailed explanations to help you understand")
    print("the concepts, processes, and results of different ML approaches.")
    
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\n👋 Program interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        print("Please check your Python environment and library installations.")
        sys.exit(1)
