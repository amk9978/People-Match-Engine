#!/usr/bin/env python3

import os
import re
from typing import List
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()


class GraphScorer:
    """Handles 2+2 architecture scoring and weight tuning"""

    def __init__(self):
        # Load default weights from .env file
        self.default_weights = self._load_default_weights_from_env()

    def _load_default_weights_from_env(self) -> list[float]:
        """Load default weights from .env and map to 2+2 architecture"""
        try:
            # Read old 5-component weights from .env
            role_sim = float(os.getenv("ROLE_SIMILARITY_WEIGHT", "0.0"))
            exp_sim = float(os.getenv("EXPERIENCE_SIMILARITY_WEIGHT", "0.1"))
            exp_comp = float(os.getenv("EXPERIENCE_COMPLEMENTARITY_WEIGHT", "0.6"))
            business_comp = float(os.getenv("BUSINESS_COMPLEMENTARITY_WEIGHT", "0.2"))
            persona_comp = float(os.getenv("PERSONA_COMPLEMENTARITY_WEIGHT", "0.2"))
            
            # Map to 2+2 architecture:
            # person_similarity = role_sim + exp_sim
            # person_complementarity = exp_comp (experience complementarity is main driver)
            # business_similarity = remaining weight for business similarity
            # business_complementarity = business_comp + persona_comp
            
            person_similarity = role_sim + exp_sim
            person_complementarity = exp_comp
            business_complementarity = business_comp + persona_comp
            
            # Calculate remaining for business similarity
            used_weight = person_similarity + person_complementarity + business_complementarity
            business_similarity = max(0.0, 1.0 - used_weight)
            
            weights = [person_similarity, person_complementarity, business_similarity, business_complementarity]
            
            # Normalize to sum to 1.0
            total = sum(weights)
            if total > 0:
                weights = [w/total for w in weights]
            else:
                # Fallback if all weights are zero
                weights = [0.25, 0.25, 0.25, 0.25]
                
            print(f"📋 Loaded default weights from .env: {[f'{w:.3f}' for w in weights]}")
            return weights
            
        except Exception as e:
            print(f"⚠️ Error loading weights from .env: {e}, using balanced fallback")
            return [0.25, 0.25, 0.25, 0.25]

    def calculate_2plus2_score(
        self, 
        role_sim: float, exp_sim: float, role_comp: float, exp_comp: float,
        industry_sim: float, market_sim: float, offering_sim: float, persona_sim: float,
        business_comp: float,
        weights: List[float]
    ) -> float:
        """
        Calculate 2+2 architecture score with geometric mean within categories
        
        Args:
            Person similarities: role_sim, exp_sim
            Person complementarities: role_comp, exp_comp
            Business similarities: industry_sim, market_sim, offering_sim, persona_sim
            Business complementarity: business_comp (combined)
            weights: [person_sim_weight, person_comp_weight, business_sim_weight, business_comp_weight]
        
        Returns:
            Final score using 2+2 architecture
        """
        
        # Person dimension - geometric mean (role weighted higher than experience)
        if role_sim > 0 and exp_sim > 0:
            person_similarity = (role_sim ** 0.6) * (exp_sim ** 0.4)
        else:
            person_similarity = 0.0
            
        if role_comp > 0 and exp_comp > 0:
            person_complementarity = (role_comp ** 0.6) * (exp_comp ** 0.4)
        else:
            person_complementarity = 0.0
        
        # Business dimension - geometric mean (industry weighted highest)
        if all(x > 0 for x in [industry_sim, market_sim, offering_sim, persona_sim]):
            business_similarity = ((industry_sim ** 0.3) * (market_sim ** 0.25) * 
                                 (offering_sim ** 0.25) * (persona_sim ** 0.2))
        else:
            business_similarity = 0.0
        
        # Business complementarity - keep existing combined score
        business_complementarity = max(0.0, business_comp)
        
        # Final 2+2 linear combination
        final_score = (
            weights[0] * person_similarity + 
            weights[1] * person_complementarity +
            weights[2] * business_similarity + 
            weights[3] * business_complementarity
        )
        
        return max(0.0, min(1.0, final_score))  # Clamp to [0,1]

    async def get_tuned_2plus2_weights(self, user_prompt: str = None) -> List[float]:
        """
        Get 4 weights for 2+2 architecture using ChatGPT or defaults
        
        Args:
            user_prompt: User's intent description for weight tuning
            
        Returns:
            [person_sim_weight, person_comp_weight, business_sim_weight, business_comp_weight]
        """
        
        if not user_prompt or not user_prompt.strip():
            print("🔧 No user prompt provided, using .env defaults")
            return self.default_weights
        
        prompt = f"""CRITICAL: You must respond with EXACTLY this format: [0.25, 0.25, 0.25, 0.25]
NO OTHER TEXT ALLOWED. Just the 4 numbers in brackets.

User wants: "{user_prompt}"

Provide exactly 4 weights that sum to 1.0:
1. person_similarity_weight (similar roles/experience)
2. person_complementarity_weight (different roles/experience)  
3. business_similarity_weight (similar industries/markets)
4. business_complementarity_weight (different business contexts)

Examples:
- AI executives: [0.4, 0.1, 0.4, 0.1]
- Partnerships: [0.2, 0.2, 0.1, 0.5] 
- Mentorship: [0.1, 0.4, 0.3, 0.2]
- Investment: [0.15, 0.15, 0.2, 0.5]
- Customers: [0.2, 0.1, 0.1, 0.6]
- General: [0.25, 0.25, 0.25, 0.25]

RESPOND ONLY: [number, number, number, number]"""

        # Try up to 3 times to get proper format
        for attempt in range(3):
            try:
                client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                
                response = await client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,  # Make it more deterministic
                    max_tokens=30,    # Reduce to force conciseness
                )

                response_text = response.choices[0].message.content.strip()
                print(f"🤖 ChatGPT attempt {attempt+1}: {response_text}")
                
                # Try multiple parsing approaches
                weights = None
                
                # Approach 1: Extract [a, b, c, d] format
                match = re.search(r'\[([0-9., ]+)\]', response_text)
                if match:
                    weights_str = match.group(1)
                    weights = [float(w.strip()) for w in weights_str.split(',')]
                
                # Approach 2: Just find 4 decimal numbers
                if not weights:
                    numbers = re.findall(r'0?\.\d+', response_text)
                    if len(numbers) >= 4:
                        weights = [float(n) for n in numbers[:4]]
                
                # Approach 3: Find any sequence of 4 numbers
                if not weights:
                    numbers = re.findall(r'\d+\.?\d*', response_text)
                    if len(numbers) >= 4:
                        weights = [float(n) for n in numbers[:4]]
                
                if weights and len(weights) == 4:
                    # Normalize to sum to 1.0
                    total = sum(weights)
                    if total > 0:
                        weights = [w/total for w in weights]
                        
                        # Validate all weights are non-negative
                        if all(w >= 0 for w in weights):
                            print(f"🎯 ChatGPT tuned weights for '{user_prompt}': {[f'{w:.3f}' for w in weights]}")
                            return weights
                
                # If we get here, parsing failed - try again with more explicit prompt
                if attempt < 2:
                    prompt += f"\n\nATTEMPT {attempt+2}: Your last response '{response_text}' was not in correct format. Respond with ONLY: [0.xx, 0.xx, 0.xx, 0.xx]"
                    
            except Exception as e:
                print(f"⚠️ ChatGPT attempt {attempt+1} failed: {e}")
                if attempt == 2:
                    break
        
        print(f"⚠️ All ChatGPT attempts failed. Using .env default weights")
        return self.default_weights