# Evaluation Metrics in Machine Learning (Confusion Matrix, Precision, Recall, F1 Score)

A complete beginner friendly guide, with intuition, examples and the "why" behind every formula.

---

## 1. Why do we even need an evaluation metric?

Imagine you build a model that predicts whether a patient has a disease. Once trained, you cannot just say "it works." You need a number that tells you how well it works. That number is called an evaluation metric.

Here is the twist that beginners miss. Not every metric measures the same kind of "good." One metric might say your model is great while another says it is terrible, on the exact same model. This is why we do not rely on a single number blindly. We look at the full picture, and that full picture starts with something called the confusion matrix.

---

## 2. The Confusion Matrix (the foundation of everything)

Think of a disease test again. There are only two facts that matter for every patient.

1. What the patient's actual condition is (has disease or does not have disease)
2. What the model predicted (predicted disease or predicted no disease)

Combining these two facts gives four possible outcomes.

| | Actual: Yes (has disease) | Actual: No (healthy) |
|---|---|---|
| **Predicted: Yes** | True Positive (TP) | False Positive (FP) |
| **Predicted: No** | False Negative (FN) | True Negative (TN) |

Let us understand each cell like a story, not just a definition.

- **True Positive (TP):** The patient actually has the disease, and the model correctly predicted disease. A correct alarm.
- **True Negative (TN):** The patient is actually healthy, and the model correctly predicted healthy. A correct all clear.
- **False Positive (FP):** The patient is actually healthy, but the model wrongly predicted disease. This is a false alarm. In statistics this is also called a Type 1 error.
- **False Negative (FN):** The patient actually has the disease, but the model wrongly predicted healthy. This is the most dangerous mistake in a medical setting, because a sick person is sent home thinking they are fine. This is also called a Type 2 error.

Once you have these four numbers, every other metric in this guide is simply built by combining them in different ratios.

---

## 3. Accuracy: the first metric everyone learns, and why it can lie to you

**Formula:**

Accuracy = (TP + TN) / (TP + TN + FP + FN)

In plain words, accuracy asks: out of everyone I made a prediction about, how many did I get right, whether they were sick or healthy.

**Example:** Suppose out of 100 patients, 90 are healthy and only 10 actually have the disease. A lazy model that always predicts "healthy" for everyone will still get 90 correct out of 100, giving an accuracy of 90 percent, even though it never once caught the actual disease.

This is called the class imbalance problem, and it is exactly why accuracy alone is dangerous when one class (say, healthy patients) heavily outnumbers the other (sick patients). This is precisely the gap that precision and recall were built to fill.

---

## 4. Precision: how much can you trust a positive prediction?

**Formula:**

Precision = TP / (TP + FP)

Precision only looks at the cases where the model raised an alarm (predicted "Yes, disease"). Out of all those alarms, how many were actually correct?

Think of precision as answering this question: "When my model says yes, how often is it actually right?"

**When does precision become 1 (perfect)?**
Precision becomes 1 only when FP is 0. This means every single time the model predicted "Yes," it was actually correct. There were zero false alarms. The model may still be missing some real cases (FN can still exist), but whatever it does flag as positive is always genuinely positive.

**When does precision become 0 (worst case)?**
Precision becomes 0 when TP is 0, meaning every positive prediction the model made was wrong. In other words, every time it raised an alarm, it was a false alarm.

**Where precision matters most:** Spam email detection. If precision is low, genuine important emails (like a job offer) get wrongly marked as spam and moved out of your inbox. You care deeply that a "spam" prediction is actually spam.

---

## 5. Recall: how many real positives did you actually catch?

**Formula:**

Recall = TP / (TP + FN)

Recall only looks at the actual real positive cases (people who truly have the disease). Out of all those real cases, how many did the model successfully catch?

Think of recall as answering this question: "Out of everyone who truly needed to be caught, how many did I actually find?"

**When does recall become 1 (perfect)?**
Recall becomes 1 only when FN is 0. This means the model successfully caught every single real positive case, and missed none. It may have raised some false alarms along the way (FP can still exist), but it never let a real case slip through.

**When does recall become 0 (worst case)?**
Recall becomes 0 when TP is 0, meaning the model missed every single real positive case. Every actual disease case was wrongly told "you are healthy."

**Where recall matters most:** Disease screening or fraud detection. Missing a real disease case (a false negative) can cost a life, so you would rather flag a few extra healthy people for a second test (some false positives are acceptable) than miss even one real patient.

---

## 6. The tug of war between precision and recall

Here is something every beginner should sit with for a moment. Precision and recall usually pull in opposite directions.

If you make your model extremely cautious and only predicts "disease" when it is almost completely sure, you will get very few false positives, so precision goes up. But in becoming so cautious, it will also miss several real disease cases that were not obvious, so recall goes down.

If instead you make your model very aggressive and flags "disease" for almost anyone showing the slightest symptom, you will catch almost every real case, so recall goes up. But you will also raise a lot of false alarms on healthy people, so precision goes down.

You cannot usually maximize both at the same time. This trade off is exactly the problem the F1 score was invented to solve.

---

## 7. F1 Score: the real intuition behind the formula

**Formula:**

F1 = 2 × (Precision × Recall) / (Precision + Recall)

### Why not just take a simple average of precision and recall?

This is the most important question, and most beginners never get a satisfying answer, so let us build it from scratch.

Suppose Precision = 1.0 and Recall = 0.0. A model like this is actually useless, since it caught zero real cases. But a simple average would give:

(1.0 + 0.0) / 2 = 0.5

A score of 0.5 sounds decent, like the model is "half good." But it is not half good. It is completely broken for the purpose of catching real cases. A simple average hides this weakness because it lets one large value cover up one terrible value.

### Why the harmonic mean fixes this

F1 uses something called the harmonic mean instead of the simple mean. The harmonic mean has a special property: it is always pulled down heavily by the smaller of the two numbers. It punishes imbalance.

Using the same numbers, Precision = 1.0 and Recall = 0.0:

F1 = 2 × (1.0 × 0.0) / (1.0 + 0.0) = 0 / 1.0 = 0

Now the score correctly reflects reality. If either precision or recall is zero, F1 becomes zero, no matter how good the other one is. This forces a model to be reasonably good at both catching real cases and avoiding false alarms, not just excellent at one while ignoring the other.

### Why the number 2 in the formula

The general harmonic mean formula for two numbers a and b is:

Harmonic mean = 2ab / (a + b)

The 2 in the numerator is simply a scaling constant that comes from the mathematical definition of the harmonic mean itself. It exists so that when a and b are equal (say precision = recall = 0.8), the harmonic mean also comes out to exactly 0.8, matching intuition. Without that 2, the formula would not reduce back to the original number when both inputs are already equal.

You can check this yourself: if Precision = Recall = 0.8,

F1 = 2 × (0.8 × 0.8) / (0.8 + 0.8) = 2 × 0.64 / 1.6 = 1.28 / 1.6 = 0.8

The formula behaves exactly as expected when both scores agree, and it becomes stricter and more punishing the moment they disagree. That is the entire intuition. F1 rewards balance and punishes lopsidedness.

### What counts as a good F1 score?

F1 score always lies between 0 and 1. A value close to 1 means the model has both strong precision and strong recall together. A value close to 0 means at least one of them is very weak. There is no single universal "good" number since it depends on the problem, but generally the closer to 1, the better the balance between catching real cases and avoiding false alarms.

---

## 8. Where is this actually used in real machine learning?

These metrics matter most whenever a dataset has class imbalance, meaning one outcome is far more common than the other. Some real examples:

- **Medical diagnosis:** Detecting cancer or a disease from scans or blood tests, where missing a real case (false negative) is far more costly than a false alarm.
- **Spam detection:** Deciding whether an email is spam or not, where wrongly blocking a genuine email (false positive) is annoying and costly.
- **Fraud detection in banking:** Catching fraudulent transactions among millions of genuine ones.
- **Search and recommendation systems:** Measuring whether the results shown to a user are actually relevant (precision) and whether all relevant results were retrieved (recall).
- **Any classification problem with rare events**, such as detecting a rare manufacturing defect on a factory line.

Whenever both types of mistakes (false alarms and missed cases) have different real world costs, you cannot rely on accuracy alone. You need precision, recall and F1 together to see the true picture.

---

## 9. How to calculate this in Python (using scikit learn)

You almost never need to compute these manually in real projects. The scikit learn library already has ready made functions.

```python
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# y_true = actual labels, y_pred = model's predicted labels
y_true = [1, 0, 1, 1, 0, 1, 0, 0]
y_pred = [1, 0, 0, 1, 0, 1, 1, 0]

print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))
print("F1 Score:", f1_score(y_true, y_pred))

# This one prints all metrics together in a neat table
print(classification_report(y_true, y_pred))
```

`classification_report` is the most commonly used function in practice, since it prints precision, recall, F1 score and support (number of samples) for every class at once, saving you from calling each function separately.

---

## 10. Quick recap table (for revision before an exam or interview)

| Metric | Formula | What it really answers |
|---|---|---|
| Accuracy | (TP + TN) / Total | Out of everyone, how many did I get right overall |
| Precision | TP / (TP + FP) | When I say Yes, how often am I actually right |
| Recall | TP / (TP + FN) | Out of all real Yes cases, how many did I catch |
| F1 Score | 2PR / (P + R) | A balanced score that punishes weakness in either precision or recall |

Remember the two key extremes:

- Precision is 1 when there are zero false positives. Precision is 0 when there are zero true positives (every alarm was wrong).
- Recall is 1 when there are zero false negatives (nothing real was missed). Recall is 0 when there are zero true positives (every real case was missed).
- F1 becomes 0 the moment either precision or recall becomes 0, because the harmonic mean refuses to hide a weak score behind a strong one.
