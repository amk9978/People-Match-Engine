import hashlib
import json
import logging
import os
import re
from typing import Dict, List, Set

import pandas as pd
from openai import AsyncOpenAI

from services.redis.redis_cache import RedisEmbeddingCache

logger = logging.getLogger(__name__)


class BusinessAnalyzer:
    """Handles ChatGPT-based business complementarity analysis"""

    def __init__(self):
        self.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.cache = RedisEmbeddingCache()

    def extract_business_tags_from_dataset(self, csv_path: str) -> Dict[str, Set[str]]:
        """Extract all unique business tags from dataset"""
        df = pd.read_csv(csv_path)

        business_columns = {
            "industry": "Company Identity - Industry Classification",
            "market": "Company Market - Market Traction",
            "offering": "Company Offering - Value Proposition",
        }

        business_tags = {category: set() for category in business_columns.keys()}

        for category, column_name in business_columns.items():
            for _, row in df.iterrows():
                if pd.notna(row[column_name]):
                    # Split by pipe and clean
                    tags = [tag.strip() for tag in str(row[column_name]).split("|")]
                    business_tags[category].update(tag for tag in tags if tag)

        logger.info(f"Extracted business tags:")
        for category, tags in business_tags.items():
            logger.info(f"  {category}: {len(tags)} unique tags")

        return business_tags

    async def get_causal_relationships_for_tag(
        self, target_tag: str, comparison_tags: List[str], category: str
    ) -> Dict[str, float]:
        """Get causal relationship scores for one tag against a list of others"""

        # Check cache first
        cache_key = f"causal_{category}_{target_tag}_vs_{len(comparison_tags)}"
        cached_result = self.cache.get(cache_key)
        if cached_result and isinstance(cached_result, str):
            return json.loads(cached_result)

        # Prepare ChatGPT prompt
        comparison_list = "\n".join([f"- {tag}" for tag in comparison_tags])

        prompt = f"""
You are a business strategy expert analyzing complementary relationships between {category} characteristics.

TARGET: {target_tag}

Rate the STRATEGIC VALUE of business connections between "{target_tag}" and each of these other {category} characteristics:

{comparison_list}

Scoring criteria (0.0 to 1.0):
- 0.9-1.0: Highly complementary, creates significant strategic value (e.g., "Early-stage" + "Growth-stage" for mentorship/investment)
- 0.7-0.8: Strong complementary value (e.g., "B2B SaaS" + "Enterprise Hardware" for integration opportunities)  
- 0.5-0.6: Moderate complementary value (some synergies possible)
- 0.3-0.4: Limited complementary value (different but not particularly synergistic)
- 0.1-0.2: Minimal complementary value (too similar or unrelated)
- 0.0: No strategic value (identical or conflicting)

IMPORTANT: Rate COMPLEMENTARY VALUE, not similarity. Different industries/stages often create MORE strategic value.

CRITICAL: You MUST respond with ONLY a valid JSON object. No explanations, no markdown, no text before or after. Start your response with {{ and end with }}. Example format:
{{"Artificial Intelligence": 0.85, "Manufacturing": 0.92, "Healthcare": 0.76}}

Your response:"""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=1000,
            )

            result_text = response.choices[0].message.content.strip()

            # Enhanced JSON parsing with multiple fallback strategies
            causal_scores = self._parse_chatgpt_response(result_text, comparison_tags)

            # Cache the result
            self.cache.set(cache_key, json.dumps(causal_scores))

            logger.info(
                f"  ✓ Got causal scores for {target_tag} vs {len(causal_scores)} tags"
            )
            return causal_scores

        except Exception as e:
            logger.info(f"  ✗ Error getting causal scores for {target_tag}: {e}")
            # Return default moderate scores
            return {tag: 0.5 for tag in comparison_tags}

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

    def extract_persona_tags_from_dataset(self, csv_path: str) -> Set[str]:
        """Extract all unique persona tags from the dataset"""
        df = pd.read_csv(csv_path)

        persona_tags = set()
        for _, row in df.iterrows():
            if pd.notna(row["All Persona Titles"]):
                # Split by pipe and clean tags
                raw_tags = str(row["All Persona Titles"]).split("|")
                for tag in raw_tags:
                    clean_tag = tag.strip()
                    if clean_tag and len(clean_tag) > 2:
                        persona_tags.add(clean_tag)

        logger.info(f"Extracted {len(persona_tags)} unique persona tags from dataset")
        return persona_tags

    def extract_role_tags_from_dataset(self, csv_path: str) -> Set[str]:
        """Extract all unique role specification tags from the dataset"""
        df = pd.read_csv(csv_path)

        role_tags = set()
        for _, row in df.iterrows():
            if pd.notna(row["Professional Identity - Role Specification"]):
                # Split by pipe and clean tags
                raw_tags = str(row["Professional Identity - Role Specification"]).split(
                    "|"
                )
                for tag in raw_tags:
                    clean_tag = tag.strip()
                    if clean_tag and len(clean_tag) > 2:
                        role_tags.add(clean_tag)

        logger.info(f"Extracted {len(role_tags)} unique role tags from dataset")
        return role_tags

    async def analyze_role_complementarity(self, role1: str, role2: str) -> float:
        """Analyze complementarity between two specific roles using ChatGPT"""
        if role1 == role2:
            return 0.0

        cache_key = f"role_complementarity_{role1}_{role2}"
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return float(cached_result)

        prompt = f"""
You are analyzing professional role complementarity for networking value.

Role 1: "{role1}"
Role 2: "{role2}"

Rate the STRATEGIC NETWORKING VALUE (0.0 to 1.0) if these two roles connected:

- 0.9-1.0: Highly complementary (e.g., "Software Engineer" + "Product Manager", "Founder" + "Investor")
- 0.7-0.8: Strong complementary value (e.g., "Sales Director" + "Marketing Director")
- 0.5-0.6: Moderate complementary value (related but different focus)
- 0.3-0.4: Limited complementary value (somewhat related)
- 0.1-0.2: Minimal complementary value (very similar or unrelated)
- 0.0: No strategic value (identical roles)

RESPOND WITH ONLY A NUMBER BETWEEN 0.0 AND 1.0:"""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10,
            )

            result_text = response.choices[0].message.content.strip()

            # Try to extract a float
            try:
                score = float(result_text)
                if 0.0 <= score <= 1.0:
                    # Cache the result
                    self.cache.set(cache_key, score)
                    return score
            except ValueError:
                pass

            # Fallback: return moderate score
            self.cache.set(cache_key, 0.5)
            return 0.5

        except Exception as e:
            logger.info(
                f"Error analyzing role complementarity for {role1} vs {role2}: {e}"
            )
            self.cache.set(cache_key, 0.5)
            return 0.5

    async def get_profile_complementarity(
        self, target_profile: str, comparison_profiles: List[str], category: str
    ) -> Dict[str, float]:
        """Get complementarity scores between complete profile vectors"""
        target_hash = hashlib.md5(target_profile.encode()).hexdigest()[:8]
        comparison_hash = hashlib.md5(
            str(sorted(comparison_profiles)).encode()
        ).hexdigest()[:8]
        cache_key = (
            f"profile_complementarity_{category}_{target_hash}_vs_{comparison_hash}"
        )
        cached_result = self.cache.get(cache_key)
        if cached_result and isinstance(cached_result, str):
            return json.loads(cached_result)

        comparison_list = "\n".join([f"- {profile}" for profile in comparison_profiles])

        prompt = f"""
You are a business strategy expert analyzing complementary relationships between complete {category} profiles.

TARGET PROFILE: {target_profile}

Rate the STRATEGIC COMPLEMENTARITY between the target profile and each of these other {category} profiles:

{comparison_list}

Scoring criteria (0.0 to 1.0):
- 0.9-1.0: Highly complementary profiles that create significant strategic value together
- 0.7-0.8: Strong complementarity with clear synergistic potential
- 0.5-0.6: Moderate complementarity with some collaboration opportunities  
- 0.3-0.4: Limited complementarity, different but not particularly synergistic
- 0.1-0.2: Minimal complementarity, too similar or conflicting
- 0.0: No strategic value, identical or directly competing profiles

IMPORTANT: Rate COMPLEMENTARY VALUE between complete profiles, not similarity. Consider how the full profile combinations would create strategic business value.

CRITICAL: You MUST respond with ONLY a valid JSON object. No explanations, no markdown, no text before or after. Start your response with {{ and end with }}.

Your response:"""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=1500,
            )

            result_text = response.choices[0].message.content.strip()

            # Enhanced JSON parsing
            complementarity_scores = self._parse_chatgpt_response(
                result_text, comparison_profiles
            )

            # Cache the result
            self.cache.set(cache_key, json.dumps(complementarity_scores))

            logger.info(
                f"  ✓ Got profile complementarity for {target_profile[:50]}... vs {len(complementarity_scores)} profiles"
            )
            return complementarity_scores

        except Exception as e:
            logger.info(
                f"⚠️ Error getting profile complementarity for {target_profile}: {e}"
            )
            return {profile: 0.5 for profile in comparison_profiles}
