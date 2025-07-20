# DSAT Adaptive Test Analysis

This project analyzes student responses to DSAT adaptive tests, stores them in MongoDB, and evaluates student performance and question impact using various metrics and models.

## 📂 Project Structure

```
.
├── data_push.py                # Utility to upload JSON data to MongoDB
├── model_check.py              # Analyze thresholds and student performance
├── performance_check.py        # Evaluate and identify high-impact questions
├── scoring_DSAT_v2.json        # Raw → scaled score mapping for DSAT
├── data
    ├── 66fece285a916f0bb5aea9c5user_attempt_v3.json    # Sample student responses (v3)
    ├── 67f2aae2c084263d16dbe462user_attempt_v2.json    # Sample student responses (v2)

```

---

## 🚀 Features

- Load and upload student response data and scoring maps to MongoDB.
- Compute raw, weighted, and IRT-based student scores.
- Train simple classifiers and determine optimal thresholds for assigning module difficulty.
- Identify which incorrect questions have the highest potential impact on overall scores.
- Estimate how changing answers (from wrong → correct) affects the student’s scaled scores and module difficulty.

---

## 📦 Installation

### Requirements

- Python 3.7+
- MongoDB (local instance)
- Python packages:
  - numpy
  - pandas
  - scikit-learn
  - pymongo
  - scipy

Install dependencies:

```bash
pip install numpy pandas scikit-learn pymongo scipy
```

---

## 🗃️ Data

### Sample Data

- `user_attempt_v3.json` and `user_attempt_v2.json`: student response records in JSON format.
- `scoring_DSAT_v2.json`: maps raw scores to scaled scores for Math and Reading & Writing sections.

### Database

- Database: `test_db`
- Collections:
  - `student_response`: student answers
  - `scoring_DSAT`: scaled score mappings

---

## 📋 Usage

### 1️⃣ Push data to MongoDB

```bash
python data_push.py
```

- You will be prompted:
  - `s` for student response
  - `d` for scoring map
- Then provide the path to a `.json` file or a directory containing `.json` files.
- Data is loaded into MongoDB and `_id` fields are removed to avoid duplication.

---

### 2️⃣ Run threshold analysis

``` bash 
python model_check.py
```

This computes:

- Optimal thresholds for raw & weighted scores.
- Percentile-based thresholds.
- IRT-based theta estimates.

---

### 3️⃣ Analyze performance & identify impactful questions

This computes:

- Current total scaled score.
- Module 2 difficulty assigned.
- Top `n` questions that, if answered correctly, would improve scores the most.
- Run it to get recommendation for Student id : 65aafd6d9acfd21d1abbfaae and Practice id: 6fece275a916f0bb5aea97e or can be changed to see for other ids
```bash 
    python performance_check
``` 


---

## 🧪 Methods

- **Raw Score Model:** Find raw score threshold that best predicts if Module 2 was `hard` or `easy`.
- **Weighted Score Model:** Same as above but weights questions by difficulty.
- **IRT 1PL Theta:** Estimate student ability with Item Response Theory.
- **Percentile Model:** Use score percentiles for thresholds.
- **Decision Tree Classifier:** Fits a shallow tree on scores.

In `performance_check.py`, it uses the DSAT adaptive rule (Module 2 depends on Module 1 performance) and computes score changes if specific wrong answers were corrected.

---

## 🔗 MongoDB

You must have MongoDB running locally:

- default connection: `mongodb://localhost:27017/`
- database: `test_db`

You can change the database or collection names in `data_push.py`, `model_check.py`, or `performance_check.py`.

> Note: The model used here can be changed based on whichever model gives the optimal performance for your data.

---

## 📝 Notes

- The student response JSONs have fields such as:
  - `student_id`, `question_id`, `correct`, `time_spent`, `subject`, `unit`, `topic`, `complexity`, `section`, etc.
- Scores are mapped using the `scoring_DSAT_v2.json` file which specifies how raw scores convert to scaled scores for `hard` and `easy` modules.

---

## 📑 References

- SAT and DSAT official score mapping tables.
- Item Response Theory (1PL) for estimating ability levels.
- MongoDB documentation: [https://www.mongodb.com/docs/](https://www.mongodb.com/docs/)

---



