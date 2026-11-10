"""An inspectable toy example, not an accuracy evaluation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spam_classifier import nb_test, nb_train


def main():
    training = ["project meeting", "team meeting", "win prize", "claim prize"]
    labels = [0, 0, 1, 1]
    names = {0: "HAM", 1: "SPAM"}
    print("Synthetic training messages")
    for text, label in zip(training, labels):
        print(f"  {names[label]:4}  {text}")
    model = nb_train(training, labels)
    messages = ["team project", "win claim prize", "meeting", "prize"]
    predictions = nb_test(messages, model, use_log=True, smoothing=True)
    print("\nPredictions (log scores + Laplace smoothing)")
    for text, label in zip(messages, predictions):
        print(f"  {text:<20} -> {names[label]}")
    print("\nToy examples only; no real-email accuracy is measured.")


if __name__ == "__main__":
    main()
