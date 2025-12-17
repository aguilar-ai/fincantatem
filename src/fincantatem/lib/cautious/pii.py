from functools import lru_cache
from typing import Sequence

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities.engine.recognizer_result import (
    RecognizerResult as AnonymizerRecognizerResult,
)

from fincantatem.domain.values import SourceCode


@lru_cache(maxsize=1)
def _analyzer() -> AnalyzerEngine:
    return AnalyzerEngine()


@lru_cache(maxsize=1)
def _anonymizer() -> AnonymizerEngine:
    return AnonymizerEngine()


def strip_pii(
    source_code: SourceCode,
    *,
    entities: Sequence[str] | None = None,
    score_threshold: float = 0.7,
    allow_list: Sequence[str] | None = None,
    language: str = "en",
) -> SourceCode:
    text = str(source_code)

    analyzer_results = _analyzer().analyze(
        text=text,
        language=language,
        entities=list(entities) if entities is not None else None,
        score_threshold=score_threshold,
        allow_list=list(allow_list) if allow_list is not None else None,
    )
    anonymizer_results = [
        AnonymizerRecognizerResult(
            entity_type=r.entity_type,
            start=r.start,
            end=r.end,
            score=r.score,
        )
        for r in analyzer_results
    ]
    anonymized = _anonymizer().anonymize(text=text, analyzer_results=anonymizer_results)

    return SourceCode(anonymized.text)
