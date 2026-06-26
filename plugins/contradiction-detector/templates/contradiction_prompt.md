# Contradiction Detection Task

You are analyzing two wiki pages that may contain contradictory claims. Your job is to identify any factual contradictions between them.

## Page A: {{ page_a_title }}

**Path:** `{{ page_a_slug }}`

{{ page_a_body }}

---

## Page B: {{ page_b_title }}

**Path:** `{{ page_b_slug }}`

{{ page_b_body }}

---

## Instructions

1. Extract the main factual claims from each page.
2. Compare the claims between pages.
3. Identify any contradictions where both claims cannot be true simultaneously.

## Output Format

Return a JSON array of contradictions found. Each contradiction should have:
- `claim_a`: The claim from page A (concise summary)
- `claim_b`: The conflicting claim from page B
- `confidence`: A score from 0.0 to 1.0 indicating how likely this is a real contradiction
- `reasoning`: Brief explanation of why these claims contradict

If no contradictions are found, return an empty array `[]`.

Example output:
```json
[
  {
    "claim_a": "The API requires authentication via API key",
    "claim_b": "The API is publicly accessible without authentication",
    "confidence": 0.95,
    "reasoning": "Directly opposite statements about authentication requirements"
  }
]
```

Return ONLY the JSON array, no additional text.
