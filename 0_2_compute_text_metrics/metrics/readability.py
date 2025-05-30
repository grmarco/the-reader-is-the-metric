import textstat
import re

class ReadabilityAnalysis:
    def __init__(self, language='en'):
        """
        Initializes the ReadabilityAnalysis class.
        
        :param language: Language of the text ('en' for English, 'es' for Spanish).
        """
        self.language = language
        textstat.set_lang(language)
    
    def analyze(self, text):
        """
        Analyzes the text and returns a dictionary with various readability metrics.
        
        :param text: The text to analyze.
        :return: Dictionary with readability metrics.
        """
        results = {
            "flesch_reading_ease": textstat.flesch_reading_ease(text),
            "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
            "smog_index": textstat.smog_index(text),
            "coleman_liau_index": textstat.coleman_liau_index(text),
            "automated_readability_index": textstat.automated_readability_index(text),
            "dale_chall_readability_score": textstat.dale_chall_readability_score(text),
            "difficult_words": textstat.difficult_words(text),
            "linsear_write_formula": textstat.linsear_write_formula(text),
            "gunning_fog": textstat.gunning_fog(text),
            "text_standard": self._convert_text_standard(textstat.text_standard(text)),
        }
        
        if self.language == 'es':
            results.update({
                "fernandez_huerta": textstat.fernandez_huerta(text),
                "szigriszt_pazos": textstat.szigriszt_pazos(text),
                "gutierrez_polini": textstat.gutierrez_polini(text),
                "crawford": textstat.crawford(text),
            })
        
        return results

    def _convert_text_standard(self, text_standard):
        """
        Converts the text_standard value to a numeric value.
        
        :param text_standard: The text_standard value returned by textstat.
        :return: The numeric value of text_standard.
        """
        match = re.search(r'\d+', text_standard)
        return int(match.group()) if match else None