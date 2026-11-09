import glob
import os
import math
import re


def load_data(directory):
    x = []
    y = []
    for f in sorted(glob.glob(os.path.join(directory, "HAM.*.txt"))):
        with open(f, 'r', encoding='utf-8', errors='replace') as file:
            x.append(file.read())
            y.append(0)
    for f in sorted(glob.glob(os.path.join(directory, "SPAM.*.txt"))):
        with open(f, 'r', encoding='utf-8', errors='replace') as file:
            x.append(file.read())
            y.append(1)
    return x, y

def nb_train(x, y):
    if not x or len(x) != len(y) or any(label not in (0, 1) for label in y):
        raise ValueError("Supply equally sized nonempty documents and binary labels")
    if set(y) != {0, 1}:
        raise ValueError("Training requires examples of both classes")
    model = {
        'ham_count': 0,
        'spam_count': 0,
        'ham_fd': {},
        'spam_fd': {}
    }
    for document, label in zip(x, y):
        words = re.findall(r'\b\w+\b', document.lower())
        if label == 0:
            model['ham_count'] += 1
            update_frequency_distribution(words, model['ham_fd'])
        else:
            model['spam_count'] += 1
            update_frequency_distribution(words, model['spam_fd'])
    if not model['ham_fd'] or not model['spam_fd']:
        raise ValueError("Each class requires at least one token")
    return model

def update_frequency_distribution(words, freq_dict):
    for word in words:
        if word in freq_dict:
            freq_dict[word] += 1
        else:
            freq_dict[word] = 1


def nb_test(docs, trained_model, use_log=False, smoothing=False):
    predictions = []
    # Calculate the prior probabilities
    total_emails = trained_model['ham_count'] + trained_model['spam_count']
    prior_ham = trained_model['ham_count'] / total_emails
    prior_spam = trained_model['spam_count'] / total_emails
    
    # Pre-calculate total words in ham and spam for smoothing
    total_ham_words = sum(trained_model['ham_fd'].values())
    total_spam_words = sum(trained_model['spam_fd'].values())
    vocabulary = set(trained_model['ham_fd']) | set(trained_model['spam_fd'])
    vocab_size = len(vocabulary)

    for doc in docs:
        words = re.findall(r'\b\w+\b', doc.lower())
        # Calculate the log probabilities or direct probabilities
        ham_prob = math.log(prior_ham) if use_log else prior_ham
        spam_prob = math.log(prior_spam) if use_log else prior_spam
        
        for word in words:
            if word not in vocabulary:
                continue
            if smoothing:
                # Add-one Laplace smoothing
                ham_word_prob = (trained_model['ham_fd'].get(word, 0) + 1) / (total_ham_words + vocab_size)
                spam_word_prob = (trained_model['spam_fd'].get(word, 0) + 1) / (total_spam_words + vocab_size)
            else:
                # No smoothing
                ham_word_prob = trained_model['ham_fd'].get(word, 0) / total_ham_words if trained_model['ham_fd'].get(word, 0) > 0 else 0.0
                spam_word_prob = trained_model['spam_fd'].get(word, 0) / total_spam_words if trained_model['spam_fd'].get(word, 0) > 0 else 0.0
            
            if use_log:
                ham_prob += (math.log(ham_word_prob) if ham_word_prob else float("-inf"))
                spam_prob += (math.log(spam_word_prob) if spam_word_prob else float("-inf"))
            else:
                ham_prob *= ham_word_prob
                spam_prob *= spam_word_prob
        
        # Compare the probabilities to classify
        if use_log:
            prediction = 0 if ham_prob >= spam_prob else 1
        else:
            prediction = 0 if ham_prob >= spam_prob else 1
        
        predictions.append(prediction)
    
    return predictions
    
def f_score(y_true, y_pred):
    true_positives = sum(1 for true, pred in zip(y_true, y_pred) if true == pred == 1)
    false_positives = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 1)
    false_negatives = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 0)

    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)






def main():
    import argparse
    parser = argparse.ArgumentParser(description="Coursework Naive Bayes spam classifier")
    parser.add_argument("--train", help="Optional directory containing authorized HAM.*.txt / SPAM.*.txt")
    parser.add_argument("--test", help="Optional test directory; used only with --train")
    args = parser.parse_args()
    if args.test and not args.train:
        parser.error("--test requires --train")
    if args.train:
        x, y = load_data(args.train)
        model = nb_train(x, y)
        if args.test:
            docs, labels = load_data(args.test)
            print("F1:", f_score(labels, nb_test(docs, model, use_log=True, smoothing=True)))
    else:
        model = nb_train(["project meeting", "team meeting", "win prize", "claim prize"], [0, 0, 1, 1])
        docs = ["team project", "win claim prize"]
        print("Synthetic demonstration:", nb_test(docs, model, use_log=True, smoothing=True))


if __name__ == "__main__":
    main()
