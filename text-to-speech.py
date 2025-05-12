import spacy
import pyttsx3


class TextToSpeech:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.nlp = spacy.load("en_core_web_sm")

    def preprocess_text(self, text):
        """Preprocess text using spaCy only."""
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]

        print("Part of Speech Tags:")
        for token in doc:
            if not token.is_punct:
                print(f"{token.text}: {token.pos_}")

        return ' '.join(sentences)

    def synthesize_speech(self, text, output_file='output.mp3'):
        processed_text = self.preprocess_text(text)
        self.engine.save_to_file(processed_text, output_file)
        self.engine.runAndWait()
        print(f"Speech synthesis complete: saved to {output_file}")


def analyze_text_complexity(text):
    words = text.split()
    unique_words = set(words)
    complexity = {
        "total_words": len(words),
        "unique_words": len(unique_words),
        "unique_word_ratio": len(unique_words) / len(words)
    }
    return complexity


def main():
    sample_text = "Natural Language Processing enables machines to understand human language. It includes tasks like tokenization, part-of-speech tagging, and named entity recognition."

    print("Text Complexity Analysis:")
    complexity = analyze_text_complexity(sample_text)
    for k, v in complexity.items():
        print(f"{k}: {v}")

    print("\nGenerating Speech:")
    tts = TextToSpeech()
    tts.synthesize_speech(sample_text, output_file='nlp_speech_output.mp3')


if __name__ == "__main__":
    main()
