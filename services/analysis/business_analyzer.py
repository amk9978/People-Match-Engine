import asyncio
import json
import logging
import re
import sys
from textwrap import dedent
from typing import Dict, List, Union, overload

from openai import AsyncOpenAI

import settings
from services.redis.app_cache_service import app_cache_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
FALLBACK_VALUE = settings.FALLBACK_VALUE


class BusinessAnalyzer:
    """Handles ChatGPT-based business complementarity analysis"""

    def __init__(self):
        self.openai_client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY, timeout=settings.OPENAI_TIMEOUT
        )
        self.cache = app_cache_service
        self.batch_delay = settings.ANALYZER_BATCH_DELAY

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

    async def _process_single_batch_with_delay(
        self, batch_targets: List[str], comparison_profiles: List[str], category: str, delay: float
    ) -> Dict[str, Dict[str, float]]:
        """Process a single batch with initial delay for rate limiting"""
        if delay > 0:
            await asyncio.sleep(delay)
        return await self._process_single_batch(batch_targets, comparison_profiles, category)

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

    @overload
    async def get_profile_complementarity(
        self,
        target_profiles: str,
        comparison_profiles: List[str],
        category: str,
        batch_size: int = 4,
    ) -> Dict[str, float]: ...

    @overload
    async def get_profile_complementarity(
        self,
        target_profiles: List[str],
        comparison_profiles: List[str],
        category: str,
        batch_size: int = 4,
    ) -> Dict[str, Dict[str, float]]: ...

    async def get_profile_complementarity(
        self,
        target_profiles: Union[str, List[str]],
        comparison_profiles: List[str],
        category: str,
        batch_size: int = 4,
    ) -> Union[Dict[str, float], Dict[str, Dict[str, float]]]:
        """Get complementarity scores for single or multiple target profiles in batched requests"""

        single_profile = isinstance(target_profiles, str)
        if single_profile:
            target_profiles = [target_profiles]

        cache_status = self.cache.get_dataset_complementarity_cache_status(
            target_profiles, comparison_profiles, category
        )
        results = cache_status["cached_results"]
        uncached_targets = cache_status["uncached_targets"]

        if not uncached_targets:
            if single_profile:
                return results[list(results.keys())[0]]
            return results

        logger.info(
            f"Processing {len(uncached_targets)} uncached {category} profiles in batches of {batch_size}"
        )

        tasks = []
        for i in range(0, len(uncached_targets), batch_size):
            batch_targets = uncached_targets[i : i + batch_size]
            delay = i // batch_size * self.batch_delay
            task = asyncio.create_task(
                self._process_single_batch_with_delay(batch_targets, comparison_profiles, category, delay)
            )
            tasks.append((task, batch_targets))

        batch_results_list = await asyncio.gather(*[task for task, _ in tasks], return_exceptions=True)

        for (task, batch_targets), batch_result in zip(tasks, batch_results_list):
            if isinstance(batch_result, Exception):
                logger.error(f"  ❌ Batch task failed: {batch_result}")
                for target in batch_targets:
                    results[target] = {
                        profile: FALLBACK_VALUE for profile in comparison_profiles
                    }
            else:
                # Cache individual results and merge
                batch_cache_results = {}
                for target in batch_targets:
                    if isinstance(batch_result, dict) and target in batch_result:
                        results[target] = batch_result[target]
                        batch_cache_results[target] = batch_result[target]
                        logger.info(f"  ✓ Got batch result for {target[:50]}...")
                    else:
                        results[target] = {
                            profile: FALLBACK_VALUE for profile in comparison_profiles
                        }
                        logger.info(
                            f"  ⚠️ No result for {target[:50]}..., using fallback"
                        )

                if batch_cache_results:
                    self.cache.cache_dataset_complementarity_results(
                        batch_cache_results, comparison_profiles, category
                    )

        logger.info(f"Batch processing complete: {len(results)} {category} profiles processed")

        if single_profile:
            return results[list(results.keys())[0]]

        return results
