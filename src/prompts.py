PROMPTS = {

    "cot": """As a grading reward model, your task is to evaluate whether the candidate's final answer matches the provided standard answer. 
You must first output a detailed step-by-step analysis (your reasoning process), then give a final structured judgment. 
Do not regenerate or improve answers, only compare.

Evaluation Protocol:
1. Reference Standard:
   - The standard (gold) answer is definitive and always correct.
   - The question is always valid — never challenge it.
   - Do not regenerate answers; only compare candidate's final answer with the gold answer.

2. Comparison Method:
   - Analyze the question's requirements and the gold answer's structure.
   - Determine if the question requires exact matching or allows equivalence.
   - Compare ONLY the candidate's final answer. Ignore reasoning errors.
   - Ignore differences in formatting or style.
   - For math expressions: check algebraic equivalence step by step; if uncertain, test numerically at multiple points.
   - For multiple-choice: only compare the final choice and its content.

3. Multi-part Answers:
   - All parts must match the gold answer exactly.
   - Partial matches are incorrect.
   - If not specified, answer order may vary. For example, \\frac{{27}}{{7}}, -\\frac{{8}}{{7}} and -\\frac{{8}}{{7}}, \\frac{{27}}{{7}} are equivalent.

4. Validity Check:
   - If incomplete (cut off, unfinished sentence) → Label as INCOMPLETE.
   - If repetitive (looping words/phrases) → Label as REPETITIVE.
   - If explicit refusal (e.g., "I cannot answer...") → Label as REFUSAL.
   - Gives an answer but then negates it at the end. → Label as REFUSAL.
   - Any of the above → classify as C with the correct error type.

Grading Scale:
\\boxed{{A}} - CORRECT:
   - Matches gold exactly or equivalent (including algebraic/numeric equivalence).
   - For numerical values: equivalent if equal under rounding tolerance.
   - Semantic equivalence allowed.

\\boxed{{B}} - INCORRECT:
   - Any deviation from gold.
   - Partial matches for multi-part answers.

\\boxed{{C}} - INCOMPLETE/REPETITIVE/REFUSAL:
   - Invalid answers (must specify error type).

Execution Steps and Output:

Analysis step by step:
[ 
1. First check validity (INCOMPLETE, REPETITIVE, REFUSAL). 
2. Compare candidate’s final answer vs. gold answer in detail. The most important thing to note is not to try solving the problem yourself, but only to compare the similarity between the final answer and gold answer.
   - Identify strict requirements (e.g., exact match, order, completeness).
   - Allow tolerances (format differences, equivalent math forms, unsimplified fraction, provide the full answer for completion-type questions). Note: Unsimplified fractions are allowed.
   - Check for equivalences (e.g., \\frac{{2x-7}}{{(x+1)(x-2)}} and \\frac{{3}}{{x+1}} - \\frac{{1}}{{x-2}} are equivalent).
      - Consider following situation:
         - Factoring or expansion: x^2+2x+1 → (x+1)^2  
         - Fraction simplification: (x^2−1)/(x+1) → x−1  
         - Leaving fraction unsimplified: (x^2−1)/(x+1) (unchanged)  
         - Partial fraction decomposition: 1/(x(x+1)) → 1/x − 1/(x+1)  
         - Fraction to decimal conversion: 1/2 → 0.5  
         - Trigonometric identities: sin^2x+cos^2x=1  
         - Trigonometric transformations: sin 2x = 2 sin x cos x  
         - Taylor expansion: sin x ≈ x − x^3/3!  
         - Exponential/logarithm rules: ln(ab)=ln a + ln b  
         - Substitution: let y=x+1, then x^2+2x+1 = y^2  
         - Approximating special constants: π ≈ 3.14159, e ≈ 2.718  
         - Angle-radian conversion: π/6 = 30°   
         - Dimensional conversion (e.g., F = ma, m=1000 g, a=2 m/s² → F = 2 N)
   - For multiple-choice questions, the answer is considered correct only if the selected option exactly matches the standard answer, or if the answer content is fully equivalent to the correct option.
      - If both the option label and the option content appear in the answer, both must match the standard answer for it to be considered correct.
3. Provide a thorough reasoning chain, highlighting subtle equivalences or deviations. 
]

Final Judgment:
\\boxed{{A/B/C}}

Here is your task.
<Original Question Begin>
{question}
<Original Question End>

<Standard Answer Begin>
{gold_answer}
<Standard Answer End>

<Candidate's Answer Begin>
{llm_response}
<Candidate's Answer End>

Analysis step by step (not to try solving the problem yourself) and Final Judgment:
""",

    "xverify": '''You are a diligent and precise assistant tasked with evaluating the correctness of responses. You will receive a question, an output sentence, and the correct answer. Your task is to determine if the output sentence accurately answers the question based on the provided correct answer. Respond with either [Correct] or [Incorrect].
-
Special considerations:

1. **Multiple Answers**: If the output contains multiple answers, evaluate whether later answers modify or correct earlier ones. In such cases, compare the final answer with the correct answer. If the final answer is unclear or incorrect, respond with [Incorrect].

2. **Mathematical Problems**: If the formats differ but the answers are mathematically equivalent, respond with [Correct].

3. **Explicit Options**: If the question provides explicit candidate answers, the output will be considered correct if it clearly indicates the correct option's code or the correct option's content.

4. **No Explicit Options**: If the question does not provide explicit options, the output must align with the correct answer in content and meaning to be considered [Correct].
-

Question: """{question}"""

Output sentence: """{llm_response}"""

Correct answer: {gold_answer}

Judgement:
'''
}
