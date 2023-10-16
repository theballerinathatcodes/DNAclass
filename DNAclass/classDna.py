import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Sample DNA sequence data with labels
data = {
    'sequence': ['ATCGATCGATCG', 'GGGGTTAGTTAC', 'AGCTTAGCGGCT', 'CCCTAGGCGGAC', 'TAGGCTAAGCTA'],
    'label': ['coding', 'noncoding', 'coding', 'noncoding', 'coding']
}

# Create a DataFrame to store the data
df = pd.DataFrame(data)

# One-hot encode the DNA sequences
encoded_sequences = []
for seq in df['sequence']:
    encoded_seq = []
    for base in seq:
        if base == 'A':
            encoded_seq.extend([1, 0, 0, 0])  # A: 1000
        elif base == 'T':
            encoded_seq.extend([0, 1, 0, 0])  # T: 0100
        elif base == 'C':
            encoded_seq.extend([0, 0, 1, 0])  # C: 0010
        elif base == 'G':
            encoded_seq.extend([0, 0, 0, 1])  # G: 0001
    encoded_sequences.append(encoded_seq)

df['encoded_sequence'] = encoded_sequences

# Split the data into training and testing sets
X = list(df['encoded_sequence'])  # Features (one-hot encoded DNA sequences)
y = list(df['label'])             # Labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a random forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
