"""
Sentence Generator Module using T5 for Sign Language Translation

This module takes broken word sequences (glosses) from sign language detection
and converts them into natural, grammatically correct English sentences.
"""

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch


class SentenceGenerator:
    """
    Uses a T5 model to convert sign language glosses into fluent sentences.
    
    Example:
        >>> sg = SentenceGenerator()
        >>> sg.generate_sentence("dog bark loud night scare thief")
        "The dog barked loudly at night and scared the thief."
    """
    
    def __init__(self, model_name: str = "vennify/t5-base-grammar-correction"):
        """
        Initialize the T5 model and tokenizer.
        
        Args:
            model_name: HuggingFace model name. Options:
                - "vennify/t5-base-grammar-correction" (~900MB, best for grammar)
                - "google/flan-t5-base" (~900MB, general purpose)
        """
        print(f"🔄 Loading T5 model: {model_name}...")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   Using device: {self.device}")
        
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            print(f"✅ T5 model loaded successfully!")
            self.is_ready = True
            
        except Exception as e:
            print(f"❌ Failed to load T5 model: {e}")
            self.is_ready = False
            self.tokenizer = None
            self.model = None
    
    def generate_sentence(self, glosses: str) -> str:
        """
        Convert broken word sequences into a natural English sentence.
        
        Args:
            glosses: Space-separated words from sign detection
                     (e.g., "dog bark loud night scare thief")
        
        Returns:
            A grammatically correct sentence
            (e.g., "The dog barked loudly at night and scared the thief.")
        """
        if not self.is_ready:
            print("⚠️ T5 model not ready, returning original text.")
            return glosses
        
        # Clean up the input
        cleaned_glosses = glosses.strip().lower()
        
        if not cleaned_glosses:
            return ""
        
        # For grammar correction model, use "grammar: " prefix
        # The model expects: "grammar: [broken text]" -> outputs corrected text
        prompt = f"grammar: {cleaned_glosses}"

        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=256,
                truncation=True
            ).to(self.device)
            
            # Generate output with beam search (no sampling for consistency)
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=100,
                    num_beams=5,
                    early_stopping=True,
                    do_sample=False
                )
            
            # Decode the generated text
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up the output
            generated_text = generated_text.strip()
            
            # Ensure proper capitalization and punctuation
            if generated_text and not generated_text[0].isupper():
                generated_text = generated_text[0].upper() + generated_text[1:]
            
            if generated_text and generated_text[-1] not in '.!?':
                generated_text += '.'
            
            print(f"🤖 T5 generated: '{generated_text}'")
            return generated_text
            
        except Exception as e:
            print(f"❌ T5 generation error: {e}")
            return glosses


# Singleton instance for lazy loading
_generator_instance = None


def get_sentence_generator(model_name: str = "vennify/t5-base-grammar-correction") -> SentenceGenerator:
    """
    Get or create a singleton SentenceGenerator instance.
    This ensures the model is only loaded once.
    """
    global _generator_instance
    
    if _generator_instance is None:
        _generator_instance = SentenceGenerator(model_name)
    
    return _generator_instance


# For direct testing
if __name__ == "__main__":
    print("\n--- Testing Sentence Generator ---\n")
    
    sg = SentenceGenerator()
    
    test_cases = [
        "dog bark loud night scare thief",
        "i go store buy milk",
        "weather cold need jacket",
        "hello how you",
        "thank you help me"
    ]
    
    for test in test_cases:
        print(f"\nInput:  '{test}'")
        result = sg.generate_sentence(test)
        print(f"Output: '{result}'")
