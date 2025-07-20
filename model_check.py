import json
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from scipy.optimize import minimize_scalar
from pymongo import MongoClient


class AdaptiveThresholdAnalyzer:
    def __init__(self, OnSubject: str):
        """
        Initialize analyzer and load data.
        """
        self.OnSubject = OnSubject
        self.student_data = self.load_data()

        # Mapping of difficulty labels to weights & IRT b-values
        self.difficulty_weights = {"easy": 1, "moderate": 2, "hard": 3}
        self.difficulty_map = {"easy": -1.0, "moderate": 0.0, "hard": 1.0}

        # Containers for processed data
        self.student_rows = []
        self.theta_scores = []

        self.prepare_data()

    def load_data(self):
        """
        Load student data from JSON file.
        """
        client = MongoClient("mongodb://localhost:27017/")
        db = client["test_db"]                  
        collection = db["student_response"]    
        # 📥 Fetch unique (student_id, practicesetId) combinations
        cursor = collection.find({}, {"student_id": 1, "practicesetId": 1, "_id": 0})
        unique_pairs = set((doc["student_id"], doc["practicesetId"]) for doc in cursor)

        result = []
        for student_id, practiceset_id in unique_pairs:
            # Fetch filtered data
            data = list(
                collection.find(
                    {
                        "student_id": student_id,
                        "practicesetId": practiceset_id,
                        "subject.name": self.OnSubject
                    },
                    {"_id": 0}
                )
            )

            if not data:
                continue  # skip if no data

            # build items & determine module2
            items = []
            module2_val = "unknown"  # default
            for item in data:
                section = item.get("section", "Static")
                # if section is not Static → set as module2
                if section != "Static":
                    module2_val = section.lower()  # e.g., "hard", "easy"
                else:
                    items.append({
                        "question_id": item["question_id"],
                        "correct": item["correct"],
                        "complexity": item["compleixty"]
                    })

            result.append({
                "student_id": student_id,
                "items": items,
                "module2": module2_val
            })
        return result

    def prepare_data(self):
        """
        Process student data to extract features:
        - raw score
        - weighted score
        - IRT theta
        - label (0 = easy, 1 = hard)
        """
        for student in self.student_data:
            responses = [item["correct"] for item in student["items"]]
            b_values = [self.difficulty_map[item["complexity"]] for item in student["items"]]
            theta = self.estimate_theta_1pl(responses, b_values)

            self.theta_scores.append({
                "student_id": student["student_id"],
                "theta": theta,
                "label": 1 if student["module2"].lower() == "hard" else 0
            })

            correct = 0
            weighted = 0
            for item in student["items"]:
                w = self.difficulty_weights.get(item["complexity"].lower(), 0)
                c = int(item["correct"])
                correct += c
                weighted += c * w

            self.student_rows.append({
                "student_id": student["student_id"],
                "raw_score": correct,
                "weighted_score": weighted,
                "label": 1 if student["module2"].lower() == "hard" else 0
            })

    def estimate_theta_1pl(self, responses, difficulties):
        """
        Estimate student ability (theta) using simplified IRT 1PL model.
        """
        responses = np.array(responses)
        b_vals = np.array(difficulties)

        def neg_log_likelihood(theta):
            prob = 1 / (1 + np.exp(-(theta - b_vals)))
            log_likelihood = responses * np.log(prob) + (1 - responses) * np.log(1 - prob)
            return -np.sum(log_likelihood)

        result = minimize_scalar(neg_log_likelihood, bounds=(-3, 3), method='bounded')
        return result.x

    def raw_score_model(self, df):
        """
        Find optimal raw score threshold maximizing accuracy.
        """
        best_acc, best_thresh = 0, None
        for t in range(df['raw_score'].min(), df['raw_score'].max() + 1):
            pred = (df['raw_score'] >= t).astype(int)
            acc = accuracy_score(df['label'], pred)
            if acc > best_acc:
                best_acc, best_thresh = acc, t
        return {'name': 'raw_score', 'threshold': best_thresh, 'accuracy': best_acc, 'complexity': 1}

    def weighted_score_model(self, df):
        """
        Find optimal weighted score threshold maximizing accuracy.
        """
        best_acc, best_thresh = 0, None
        for t in range(df['weighted_score'].min(), df['weighted_score'].max() + 1):
            pred = (df['weighted_score'] >= t).astype(int)
            acc = accuracy_score(df['label'], pred)
            if acc > best_acc:
                best_acc, best_thresh = acc, t
        return {'name': 'difficulty_weighted', 'threshold': best_thresh, 'accuracy': best_acc, 'complexity': 2}

    def percentile_model(self, df, score_column='weighted_score'):
        """
        Find percentile-based threshold maximizing accuracy.
        """
        best_acc, best_percentile, best_threshold = 0, None, None
        for p in range(10, 91, 5):
            threshold = np.percentile(df[score_column], p)
            pred = (df[score_column] >= threshold).astype(int)
            acc = accuracy_score(df['label'], pred)
            if acc > best_acc:
                best_acc, best_percentile, best_threshold = acc, p, threshold
        return {
            'name': 'percentile',
            'threshold': round(best_threshold, 3),
            'accuracy': best_acc,
            'percentile_used': best_percentile,
            'complexity': 2
        }

    def decision_tree_model(self, df):
        """
        Fit simple decision tree to predict label.
        """
        X = df[['raw_score', 'weighted_score']]
        y = df['label']
        model = DecisionTreeClassifier(max_depth=1)
        model.fit(X, y)
        pred = model.predict(X)
        acc = accuracy_score(y, pred)
        return {'name': 'decision_tree', 'threshold': 'tree-based', 'accuracy': acc, 'complexity': 2}

    def IRT_based(self, df):
        """
        Find optimal IRT theta threshold maximizing accuracy.
        """
        best_acc, best_thresh = 0, None
        for t in sorted(df['theta'].unique()):
            pred = (df['theta'] >= t).astype(int)
            acc = accuracy_score(df['label'], pred)
            if acc > best_acc:
                best_acc, best_thresh = acc, t
        return {'name': 'IRT-Based', 'threshold': best_thresh, 'accuracy': best_acc, 'complexity': 2}

    def evaluate_models(self):
        """
        Evaluate all models and print results.
        """
        df = pd.DataFrame(self.student_rows)
        df2 = pd.DataFrame(self.theta_scores)

        models = [
            self.raw_score_model(df),
            self.weighted_score_model(df),
            self.percentile_model(df),
            self.decision_tree_model(df),
            self.IRT_based(df2)
        ]

        models_sorted = sorted(models, key=lambda x: (-x['accuracy'], x['complexity']))
        best_model = models_sorted[0]

        print("All Model Evaluations:")
        for m in models_sorted:
            print(f"Model: {m['name']:20s} | Threshold: {str(m['threshold']):8s} | Accuracy: {m['accuracy']:.2f} | Complexity: {m['complexity']}")

        print("\nBest Model Detected:")
        print(f"Name: {best_model['name']}")
        print(f"Threshold: {best_model['threshold']}")
        print(f"Accuracy: {best_model['accuracy']:.2f}")
        print(f"Complexity: {best_model['complexity']} (lower = simpler)")

        return best_model


if __name__ == "__main__":
    analyzer = AdaptiveThresholdAnalyzer("Reading and Writing")
    analyzer.evaluate_models()
