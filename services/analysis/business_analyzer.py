import asyncio
import hashlib
import json
import logging
import os
import re
from typing import Dict, List, Set, Union, overload

import pandas as pd
from openai import AsyncOpenAI

from services.redis.app_cache_service import app_cache_service

logger = logging.getLogger(__name__)


class BusinessAnalyzer:
    """Handles ChatGPT-based business complementarity analysis"""

    def __init__(self):
        self.openai_client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"), timeout=30.0
        )
        self.cache = app_cache_service

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
                        scores[tag] = 0.5
                return scores

        except Exception:
            pass

        # Final fallback: Return moderate scores for all
        logger.info(f"  ⚠️ Could not parse ChatGPT response, using fallback scores")
        return {tag: 0.5 for tag in comparison_tags}

    def _parse_batch_chatgpt_response(
        self,
        result_text: str,
        target_profiles: List[str],
        comparison_profiles: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """Parse batch ChatGPT response with fallback strategies"""

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
            json_match = re.search(r"\{.*\}", result_text, re.DOTALL)
            if json_match:
                json_content = json_match.group(0)
                return json.loads(json_content)
        except (json.JSONDecodeError, AttributeError):
            pass

        # Final fallback: Return moderate scores for all combinations
        logger.info(
            f"  ⚠️ Could not parse batch ChatGPT response, using fallback scores"
        )
        return {
            target: {comp: 0.5 for comp in comparison_profiles}
            for target in target_profiles
        }

    async def _process_single_batch(
        self, batch_targets: List[str], comparison_profiles: List[str], category: str
    ) -> Dict[str, Dict[str, float]]:
        """Process a single batch of targets concurrently"""
        targets_list = "\n".join(
            [f"{j+1}. {profile}" for j, profile in enumerate(batch_targets)]
        )
        comparison_list = "\n".join([f"- {profile}" for profile in comparison_profiles])

        prompt = f"""You are analyzing complementarity between multiple {category} target profiles and comparison profiles.

TARGET PROFILES:
{targets_list}

COMPARISON PROFILES:
{comparison_list}

For EACH target profile (1-{len(batch_targets)}), rate its complementarity (0.0-1.0) against ALL comparison profiles.

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

CRITICAL: Return ONLY valid JSON, no explanations. Use exact target profile names as keys."""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=6000,
            )

            result_text = response.choices[0].message.content.strip()
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
                    profile: 0.5 for profile in comparison_profiles
                }

            return individual_results

    @overload
    async def get_profile_complementarity(
        self,
        target_profiles: str,
        comparison_profiles: List[str],
        category: str,
        batch_size: int = 16,
    ) -> Dict[str, float]: ...

    @overload
    async def get_profile_complementarity(
        self,
        target_profiles: List[str],
        comparison_profiles: List[str],
        category: str,
        batch_size: int = 16,
    ) -> Dict[str, Dict[str, float]]: ...

    async def get_profile_complementarity(
        self,
        target_profiles: Union[str, List[str]],
        comparison_profiles: List[str],
        category: str,
        batch_size: int = 16,
    ) -> Union[Dict[str, float], Dict[str, Dict[str, float]]]:
        """Get complementarity scores for single or multiple target profiles in batched requests"""

        # Handle single profile input
        single_profile = isinstance(target_profiles, str)
        if single_profile:
            target_profiles = [target_profiles]

        # Check cache status for the dataset
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

        # Create all batch tasks concurrently
        tasks = []
        for i in range(0, len(uncached_targets), batch_size):
            batch_targets = uncached_targets[i : i + batch_size]
            task = asyncio.create_task(
                self._process_single_batch(batch_targets, comparison_profiles, category)
            )
            tasks.append((task, batch_targets))

        # Execute all batches concurrently and gather results
        batch_results_list = await asyncio.gather(
            *[task for task, _ in tasks], return_exceptions=True
        )

        # Process results and handle exceptions
        for (task, batch_targets), batch_result in zip(tasks, batch_results_list):
            if isinstance(batch_result, Exception):
                logger.error(f"  ❌ Batch task failed: {batch_result}")
                for target in batch_targets:
                    results[target] = {profile: 0.5 for profile in comparison_profiles}
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
                            profile: 0.5 for profile in comparison_profiles
                        }
                        logger.info(
                            f"  ⚠️ No result for {target[:50]}..., using fallback"
                        )

                # Cache all batch results at once
                if batch_cache_results:
                    self.cache.cache_dataset_complementarity_results(
                        batch_cache_results, comparison_profiles, category
                    )

        logger.info(
            f"✅ Batch processing complete: {len(results)} {category} profiles processed"
        )

        # Return single profile result if input was single profile
        if single_profile:
            return results[list(results.keys())[0]]

        return results
