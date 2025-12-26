"""
Video captioning evaluation metrics: BLEU-4 and METEOR.

Both metrics support multiple reference captions and average the scores.
"""

from typing import Dict, List

import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu

try:
    from nltk.translate.meteor_score import meteor_score
except ImportError:
    meteor_score = None


class CaptioningMetrics:
    """Calculate BLEU-4 and METEOR metrics for video captioning."""

    def __init__(self):
        """Initialize nltk tokenizers."""
        # Download required NLTK data
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)

        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet", quiet=True)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenize text into words.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens (lowercase)
        """
        return word_tokenize(text.lower())

    def bleu_4(
        self,
        generated: str,
        reference_captions: List[str],
    ) -> float:
        """Calculate BLEU-4 score for one sample.

        BLEU-4 evaluates up to 4-gram precision with brevity penalty.

        Args:
            generated: Generated caption string
            reference_captions: List of reference captions

        Returns:
            BLEU-4 score (0.0 to 1.0)
        """
        generated_tokens = self._tokenize(generated)

        # All reference captions should be tokenized
        reference_tokens = [self._tokenize(ref) for ref in reference_captions]

        # Calculate BLEU-4 (weights for 1-gram, 2-gram, 3-gram, 4-gram)
        weights = (0.25, 0.25, 0.25, 0.25)
        smoothing_function = SmoothingFunction().method1

        score = sentence_bleu(
            reference_tokens,
            generated_tokens,
            weights=weights,
            smoothing_function=smoothing_function,
        )

        return score

    def meteor(
        self,
        generated: str,
        reference_captions: List[str],
    ) -> float:
        """Calculate METEOR score for one sample.

        METEOR is an improved F1-score accounting for synonyms and stemming.
        Averages score across multiple references.

        Args:
            generated: Generated caption string
            reference_captions: List of reference captions

        Returns:
            METEOR score (0.0 to 1.0)
        """
        if meteor_score is None:
            raise ImportError(
                "NLTK meteor_score not available. " "Ensure NLTK is properly installed."
            )

        # Tokenize generated caption
        generated_tokens = self._tokenize(generated)

        # Calculate METEOR for each reference and average
        scores = []
        for reference in reference_captions:
            try:
                reference_tokens = self._tokenize(reference)
                score = meteor_score(
                    [reference_tokens],  # References must be list of token lists
                    generated_tokens,
                )
                scores.append(score)
            except Exception as e:
                # If meteor calculation fails, use 0
                print(f"Warning: METEOR calculation failed: {e}")
                scores.append(0.0)

        return sum(scores) / len(scores) if scores else 0.0

    def evaluate_batch(
        self,
        generated_captions: List[str],
        reference_captions_list: List[List[str]],
    ) -> Dict[str, float]:
        """Calculate metrics for a batch of captions.

        Args:
            generated_captions: List of generated captions
            reference_captions_list: List of lists of reference captions
                                    (each sample has multiple references)

        Returns:
            Dictionary with 'bleu_4' and 'meteor' average scores
        """
        if len(generated_captions) != len(reference_captions_list):
            raise ValueError("Number of generated captions must match reference list size")

        bleu_scores = []
        meteor_scores = []

        for generated, references in zip(generated_captions, reference_captions_list):
            bleu_scores.append(self.bleu_4(generated, references))
            meteor_scores.append(self.meteor(generated, references))

        return {
            "bleu_4": sum(bleu_scores) / len(bleu_scores),
            "meteor": sum(meteor_scores) / len(meteor_scores),
        }

    def evaluate_corpus(
        self,
        generated_captions: List[str],
        reference_captions_list: List[List[str]],
    ) -> Dict[str, float]:
        """Calculate corpus-level metrics (BLEU-4 only).

        BLEU-4 can be calculated at corpus level for more robust results.

        Args:
            generated_captions: List of generated captions
            reference_captions_list: List of lists of reference captions

        Returns:
            Dictionary with 'bleu_4' (corpus-level) and 'meteor' (average)
        """
        if len(generated_captions) != len(reference_captions_list):
            raise ValueError("Number of generated captions must match reference list size")

        # Corpus-level BLEU-4
        generated_tokens_list = [self._tokenize(cap) for cap in generated_captions]
        # Flatten reference tokens for corpus BLEU
        references_corpus = [
            [self._tokenize(ref) for ref in refs] for refs in reference_captions_list
        ]

        smoothing_function = SmoothingFunction().method1
        weights = (0.25, 0.25, 0.25, 0.25)

        bleu_4_score = corpus_bleu(
            references_corpus,
            generated_tokens_list,
            weights=weights,
            smoothing_function=smoothing_function,
        )

        # Average METEOR across samples
        meteor_scores = []
        for generated, references in zip(generated_captions, reference_captions_list):
            meteor_scores.append(self.meteor(generated, references))

        return {
            "bleu_4": bleu_4_score,
            "meteor": (sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0),
        }
