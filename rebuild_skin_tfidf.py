import pickle
from pathlib import Path

import pandas as pd
from scipy.io import mmwrite
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / 'datasets' / 'skin_data_final.csv'
MODEL_DIR = BASE_DIR / 'models'
TFIDF_PATH = MODEL_DIR / 'tfidf_rebuild.pkl'
MATRIX_PATH = MODEL_DIR / 'Tfidf_skin_data_rebuild.mtx'


def load_training_data():
    df = pd.read_csv(DATA_PATH)
    if 'cleaned_question' not in df.columns:
        raise ValueError("Missing required column: 'cleaned_question'")

    df = df.dropna(subset=['cleaned_question']).copy()
    df['cleaned_question'] = df['cleaned_question'].astype(str).str.strip()
    df = df[df['cleaned_question'] != '']

    if df.empty:
        raise ValueError('No valid cleaned_question rows found.')
    return df


def verify_self_similarity(tfidf, tfidf_matrix, df, sample_count=5):
    sample_indices = [0, len(df) // 4, len(df) // 2, (len(df) * 3) // 4, len(df) - 1]
    sample_indices = list(dict.fromkeys(sample_indices))[:sample_count]

    print('\nSelf similarity check:')
    all_ok = True
    for idx in sample_indices:
        sample_vec = tfidf.transform([df.iloc[idx]['cleaned_question']])
        similarities = linear_kernel(sample_vec, tfidf_matrix)[0]
        top_index = int(similarities.argmax())
        self_similarity = float(similarities[idx])
        ok = top_index == idx and self_similarity > 0.999
        all_ok = all_ok and ok
        print(
            f'  sample_index={idx}, top_index={top_index}, '
            f'self_similarity={self_similarity:.6f}, ok={ok}'
        )

    if not all_ok:
        raise RuntimeError('Self similarity check failed.')


def main():
    df = load_training_data()

    print(f'Loaded: {DATA_PATH}')
    print(f'Rows: {len(df)}')

    tfidf = TfidfVectorizer(sublinear_tf=True)
    tfidf_matrix = tfidf.fit_transform(df['cleaned_question'])

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with TFIDF_PATH.open('wb') as model_file:
        pickle.dump(tfidf, model_file)
    mmwrite(MATRIX_PATH, tfidf_matrix)

    print(f'Saved TF-IDF model: {TFIDF_PATH}')
    print(f'Saved TF-IDF matrix: {MATRIX_PATH}')
    print(f'Matrix shape: {tfidf_matrix.shape}')
    print(f'Vocabulary size: {len(tfidf.vocabulary_)}')

    verify_self_similarity(tfidf, tfidf_matrix, df)
    print('\nRebuild complete.')


if __name__ == '__main__':
    main()
