import asyncio
import json
import logging
import re
import sys
from textwrap import dedent
from typing import Dict, List, Tuple

from openai import AsyncOpenAI

import settings
from services.cache.app_cache_service import app_cache_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
FALLBACK_VALUE = settings.FALLBACK_VALUE


class BusinessAnalyzer:
    """Handles ChatGPT-based business complementarity analysis"""

    def __init__(self, openai_client: AsyncOpenAI = None, cache=None):
        self.openai_client = openai_client or AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY, timeout=settings.OPENAI_TIMEOUT
        )
        self.cache = cache or app_cache_service

    def _parse_chatgpt_response(
        self, result_text: str, comparison_tags: List[str]
    ) -> Dict[str, float]:
        """Parse ChatGPT response with multiple fallback strategies"""

        # Strategy 1: Try direct JSON parsing
        try:
            return json.loads(result_text)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Remove markdown formatting
        try:
            if "```json" in result_text:
                json_part = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                json_part = result_text.split("```")[1].split("```")[0].strip()
            else:
                json_part = result_text

            return json.loads(json_part)
        except (json.JSONDecodeError, IndexError):
            pass

        # Strategy 3: Find JSON-like content with regex
        try:
            json_match = re.search(r"\{[^{}]*\}", result_text, re.DOTALL)
            if json_match:
                json_content = json_match.group(0)
                return json.loads(json_content)
        except (json.JSONDecodeError, AttributeError):
            pass

        # Strategy 4: Extract key-value pairs manually
        try:
            scores = {}
            lines = result_text.split("\n")
            for line in lines:
                if ":" in line and any(tag in line for tag in comparison_tags):
                    for tag in comparison_tags:
                        if tag in line:
                            # Try to extract number after colon
                            parts = line.split(":")
                            if len(parts) >= 2:
                                score_text = parts[1].strip().rstrip(",").rstrip("}")
                                try:
                                    score = float(score_text)
                                    if 0.0 <= score <= 1.0:
                                        scores[tag] = score
                                        break
                                except ValueError:
                                    pass

            if scores:
                # Fill in missing tags with default score
                for tag in comparison_tags:
                    if tag not in scores:
                        scores[tag] = FALLBACK_VALUE
                return scores

        except Exception:
            pass

        logger.info("Could not parse ChatGPT response, using fallback scores")
        return {tag: FALLBACK_VALUE for tag in comparison_tags}

    def _parse_batch_chatgpt_response(
        self,
        result_text: str,
        target_profiles: List[str],
        comparison_profiles: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """Parse batch ChatGPT response with fallback strategies"""

        try:
            return json.loads(result_text)
        except json.JSONDecodeError:
            pass

        try:
            if "```json" in result_text:
                json_part = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                json_part = result_text.split("```")[1].split("```")[0].strip()
            else:
                json_part = result_text

            return json.loads(json_part)
        except (json.JSONDecodeError, IndexError):
            pass

        # Strategy 3: Find complete JSON object with better regex
        try:
            brace_count = 0
            start_pos = result_text.find("{")
            if start_pos != -1:
                for i, char in enumerate(result_text[start_pos:], start_pos):
                    if char == "{":
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            json_content = result_text[start_pos : i + 1]
                            return json.loads(json_content)
        except (json.JSONDecodeError, ValueError):
            pass

        # Strategy 4: Try to find and parse nested JSON structures
        try:
            json_match = re.search(
                r"{[^{}]*(?:{[^{}]*}[^{}]*)*}", result_text, re.DOTALL
            )
            if json_match:
                json_content = json_match.group(0)
                return json.loads(json_content)
        except (json.JSONDecodeError, AttributeError):
            pass

        # Strategy 5: Manual parsing of key-value pairs from truncated response
        try:
            scores = {}
            lines = result_text.split("\n")
            current_target = None

            for line in lines:
                line = line.strip()
                # Look for target profile names
                for target in target_profiles:
                    if target[:50] in line and '"' in line:
                        current_target = target
                        break

                # Look for comparison scores
                if current_target and ":" in line:
                    for comp in comparison_profiles:
                        if comp[:30] in line:
                            try:
                                score_match = re.search(r":\s*([0-9]*\.?[0-9]+)", line)
                                if score_match:
                                    score = float(score_match.group(1))
                                    if 0.0 <= score <= 1.0:
                                        if current_target not in scores:
                                            scores[current_target] = {}
                                        scores[current_target][comp] = score
                            except (ValueError, AttributeError):
                                pass

            if scores:
                for target in target_profiles:
                    if target not in scores:
                        scores[target] = {}
                    for comp in comparison_profiles:
                        if comp not in scores[target]:
                            scores[target][comp] = FALLBACK_VALUE
                return scores

        except Exception:
            pass

        logger.info(
            f"Could not parse batch ChatGPT response, using fallback scores. Result: {result_text[:100]}"
        )
        return {
            target: {comp: FALLBACK_VALUE for comp in comparison_profiles}
            for target in target_profiles
        }

    async def _process_single_batch(
        self, batch_targets: List[str], comparison_profiles: List[str], category: str
    ) -> Dict[str, Dict[str, float]]:
        """Process a single batch of targets concurrently"""
        targets_list = "\n".join(
            [f"{j + 1}. {profile}" for j, profile in enumerate(batch_targets)]
        )
        comparison_list = "\n".join([f"- {profile}" for profile in comparison_profiles])

        prompt = dedent(
            f"""You are analyzing complementarity between multiple {category} target profiles and comparison profiles.
                            TARGET PROFILES:
                            {targets_list}
                            
                            COMPARISON PROFILES:
                            {comparison_list}
                            
                            For EACH target profile (1-{len(batch_targets)}), rate its complementarity (0.0-1.0) against
                             ALL comparison profiles.
                            
                            Scoring criteria (0.0 to 1.0):
                            - 0.9-1.0: Highly complementary profiles that create significant strategic value together
                            - 0.7-0.8: Strong complementarity with clear synergistic potential
                            - 0.5-0.6: Moderate complementarity with some collaboration opportunities
                            - 0.3-0.4: Limited complementarity, different but not particularly synergistic
                            - 0.1-0.2: Minimal complementarity, too similar or conflicting
                            - 0.0: No strategic value, identical or directly competing profiles
                            
                            Return a JSON object where each target profile maps to its scores:
                            {{
                              "Target Profile 1 Name": {{"Comparison 1": 0.8, "Comparison 2": 0.6}},
                              "Target Profile 2 Name": {{"Comparison 1": 0.4, "Comparison 2": 0.9}}
                            }}
                            
                            CRITICAL: Return ONLY valid JSON, no explanations, no ```json or anything added to the json
                            answer. Use exact target profile names as keys."""
        )

        try:
            raw = await self.openai_client.chat.completions.with_raw_response.create(
                model=settings.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=settings.TEMPERATURE,
                max_tokens=settings.MAX_TOKENS,
            )

            headers = raw.headers
            remaining_requests = headers.get("x-ratelimit-remaining-requests")
            remaining_tokens = headers.get("x-ratelimit-remaining-tokens")
            reset_requests = headers.get("x-ratelimit-reset-requests")
            reset_tokens = headers.get("x-ratelimit-reset-tokens")
            processing_ms = headers.get("openai-processing-ms")
            request_id = headers.get("x-request-id")
            logger.info(
                f"single batch response received. {remaining_requests} remaining requests, {remaining_tokens}, reset_requests: {reset_requests}, reset_tokens: {reset_tokens}, processing_ms: {processing_ms}, request_id: {request_id}"
            )

            response = raw.parse()

            result_text = response.choices[0].message.content.strip()
            logger.debug(f"Raw ChatGPT response length: {len(result_text)} chars")
            logger.debug(f"Response starts with: {result_text[:200]}")
            logger.debug(
                f"Response ends with: {result_text[-200:] if len(result_text) > 200 else result_text}"
            )

            batch_results = self._parse_batch_chatgpt_response(
                result_text, batch_targets, comparison_profiles
            )

            logger.info(
                f"  ✅ Processed batch of {len(batch_targets)} {category} profiles"
            )
            return batch_results

        except Exception as e:
            logger.error(f"  ❌ Batch complementarity failed for {category}: {e}")
            logger.info(
                f"  🔄 Falling back to individual requests for {len(batch_targets)} profiles"
            )

            individual_results = {}
            for target in batch_targets:
                individual_results[target] = {
                    profile: FALLBACK_VALUE for profile in comparison_profiles
                }

            return individual_results

    async def get_profile_complementarity(
        self,
        target_profiles: List[str],
        comparison_profiles: List[str],
        category: str,
        batch_size: int = 8,
    ) -> Dict[str, Dict[str, float]]:
        """Score every target against every comparison, asking the model only for
        the pairs the cache does not already hold."""
        status = self.cache.get_complementarity_cache_status(
            target_profiles, comparison_profiles, category
        )
        results = {target: dict(scores) for target, scores in status.cached.items()}

        if not status.missing:
            return results

        for comparisons, targets in self._group_by_missing(status.missing).items():
            scored = await self._score_group(
                list(targets), list(comparisons), category, batch_size
            )
            for target, scores in scored.items():
                results.setdefault(target, {}).update(scores)
            self.cache.cache_complementarity_scores(scored, category)

        return results

    def _group_by_missing(
        self, missing: Dict[str, List[str]]
    ) -> Dict[Tuple[str, ...], List[str]]:
        """Group targets that need the same comparisons so each batch asks one question.

        Adding one person to a scored roster leaves every existing target missing
        exactly that person, which collapses into a single group."""
        groups: Dict[Tuple[str, ...], List[str]] = {}
        for target, comparisons in missing.items():
            groups.setdefault(tuple(comparisons), []).append(target)
        return groups

    async def _score_group(
        self,
        targets: List[str],
        comparisons: List[str],
        category: str,
        batch_size: int,
    ) -> Dict[str, Dict[str, float]]:
        logger.info(
            f"Feature {category}: scoring {len(targets)} targets against "
            f"{len(comparisons)} comparisons in batches of {batch_size}"
        )

        batches = [
            targets[start : start + batch_size]
            for start in range(0, len(targets), batch_size)
        ]
        batch_results = await asyncio.gather(
            *[
                self._process_single_batch(batch, comparisons, category)
                for batch in batches
            ],
            return_exceptions=True,
        )

        scored: Dict[str, Dict[str, float]] = {}
        for batch, batch_result in zip(batches, batch_results):
            if isinstance(batch_result, Exception):
                logger.error(f"Feature {category}: batch failed: {batch_result}")
                for target in batch:
                    scored[target] = {
                        comparison: FALLBACK_VALUE for comparison in comparisons
                    }
                continue

            for target in batch:
                row = batch_result.get(target)
                if row is None:
                    logger.warning(
                        f"Feature {category}: no scores returned for a target, "
                        f"using the fallback value"
                    )
                    row = {comparison: FALLBACK_VALUE for comparison in comparisons}
                scored[target] = row

        return scored
