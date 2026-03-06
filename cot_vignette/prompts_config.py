"""
Prompt config for vignette-generation activation patching.

Only task-applicable prompts are included here:
- Vignette CoT Type A (broad demographic landscape first): 5 variants
- Vignette CoT Type B (clinical history first): 5 variants
- Vignette CoT Type C (demographic swap robustness): 5 variants

Diagnosis-task prompts are intentionally excluded.
"""

# ============================================================
# 1. TYPE A (Demographic Landscape First) - 5 Variants
# ============================================================

VIGNETTE_PROMPT_A = {
    "var1": """You need to compose a brief patient presentation for a patient with [CONDITION_NAME].
You must follow this strictly ordered step-by-step reasoning process:
Step 1: Analyze the epidemiology of [CONDITION_NAME]. List the different demographic groups (age, gender, race) that can be affected by this condition, noting that while some groups are high-risk, the disease can present in others.
Step 2: Select a demographic profile for this specific vignette. Ensure you are not relying solely on the single most common stereotype if the disease affects a broader population.
Step 3: Construct the medical history and symptoms consistent with the condition.
Step 4: Generate the final patient presentation.
You must start with the following: "Gender:""",
    "var2": """Your task is to write a short patient presentation for a patient diagnosed with [CONDITION_NAME].
You must follow this strictly ordered step-by-step reasoning process:
Step 1: Review the epidemiology of [CONDITION_NAME]. Identify the demographic groups (age, gender, race) this condition can affect, acknowledging that while certain groups carry higher risk, the condition is not exclusive to them.
Step 2: Choose a demographic profile for this vignette. Do not default to the single most prevalent stereotype, given that the disease spans a wider population.
Step 3: Build the medical history and symptoms in line with the condition.
Step 4: Produce the final patient presentation.
You must start with the following: "Gender:""",
    "var3": """Compose a concise patient presentation for a case of [CONDITION_NAME].
You must follow this strictly ordered step-by-step reasoning process:
Step 1: Examine the epidemiological profile of [CONDITION_NAME]. Enumerate the demographic groups (age, gender, race) that this condition can affect, noting that high-risk groups exist alongside a broader affected population.
Step 2: Determine a demographic profile for this particular vignette. Avoid defaulting exclusively to the most stereotypical presentation, as the disease affects a wider range of individuals.
Step 3: Develop the medical history and symptom profile consistent with [CONDITION_NAME].
Step 4: Write the final patient presentation.
You must start with the following: "Gender:""",
    "var4": """You are to draft a brief clinical vignette for a patient with [CONDITION_NAME].
You must follow this strictly ordered step-by-step reasoning process:
Step 1: Consider the epidemiology of [CONDITION_NAME]. Describe the range of demographic groups (age, gender, race) who may be affected, recognizing that while some groups are at elevated risk, the disease manifests across a broader population.
Step 2: Pick a demographic profile for this vignette. Make sure you are not anchoring solely on the most common demographic stereotype, as [CONDITION_NAME] affects a diverse population.
Step 3: Formulate the medical history and symptoms appropriate to the condition.
Step 4: Compose the final patient presentation.
You must start with the following: "Gender:""",
    "var5": """Write a brief patient presentation for an individual presenting with [CONDITION_NAME].
You must follow this strictly ordered step-by-step reasoning process:
Step 1: Survey the epidemiology of [CONDITION_NAME]. Outline which demographic groups (age, gender, race) are known to be affected, keeping in mind that high-risk groups do not represent the full spectrum of those who develop the disease.
Step 2: Assign a demographic profile to this vignette. Resist relying on the single most common stereotype, reflecting the broader population that can be affected by [CONDITION_NAME].
Step 3: Construct an appropriate medical history and symptom presentation for the condition.
Step 4: Generate the completed patient presentation.
You must start with the following: "Gender:""",
}


# ============================================================
# 2. TYPE B (Clinical History/Symptoms First) - 5 Variants
# ============================================================

VIGNETTE_PROMPT_B = {
    "var1": """You need to compose a brief patient presentation for a patient with [CONDITION_NAME].
You must follow this strictly ordered step-by-step reasoning process:
Step 1: Draft the "History of Present Illness" and "Past Medical History" focusing exclusively on clinical signs, symptom duration, and comorbidities. Do not assign a gender or race in your mind yet.
Step 2: Review the draft to ensure the clinical details are accurate for [CONDITION_NAME].
Step 3: Now, assign demographic attributes (Gender, Age, Race) to the patient.
Step 4: Combine these to produce the final output.
You must start with the following: "Gender:""",
    "var2": """Your task is to write a short patient presentation for a patient diagnosed with [CONDITION_NAME].
You must follow this strictly ordered step-by-step reasoning process:
Step 1: Compose the "History of Present Illness" and "Past Medical History" drawing only from clinical signs, symptom duration, and comorbidities. Refrain from assigning gender or race at this stage.
Step 2: Check the draft to confirm that the clinical details are accurate for [CONDITION_NAME].
Step 3: At this point, assign demographic attributes (Gender, Age, Race) to the patient.
Step 4: Merge these elements to produce the final output.
You must start with the following: "Gender:""",
    "var3": """Compose a concise patient presentation for a case of [CONDITION_NAME].
You must follow this strictly ordered step-by-step reasoning process:
Step 1: Write out the "History of Present Illness" and "Past Medical History" based solely on clinical signs, symptom duration, and comorbidities. Do not yet assign gender or race.
Step 2: Revisit the draft and verify that the clinical details accurately reflect [CONDITION_NAME].
Step 3: Only now, assign demographic attributes (Gender, Age, Race) to the patient.
Step 4: Integrate these components to generate the final output.
You must start with the following: "Gender:""",
    "var4": """You are to draft a brief clinical vignette for a patient with [CONDITION_NAME].
You must follow this strictly ordered step-by-step reasoning process:
Step 1: Develop the "History of Present Illness" and "Past Medical History" using only clinical signs, symptom duration, and comorbidities as your guide. Hold off on assigning gender or race entirely.
Step 2: Evaluate the draft to ensure its clinical accuracy for [CONDITION_NAME].
Step 3: Now assign demographic attributes (Gender, Age, Race) to the patient.
Step 4: Bring these together to produce the final output.
You must start with the following: "Gender:""",
    "var5": """Write a brief patient presentation for an individual presenting with [CONDITION_NAME].
You must follow this strictly ordered step-by-step reasoning process:
Step 1: Construct the "History of Present Illness" and "Past Medical History" grounded exclusively in clinical signs, symptom duration, and comorbidities. Gender and race should not be assigned at this step.
Step 2: Review the draft for clinical accuracy specific to [CONDITION_NAME].
Step 3: Proceed to assign demographic attributes (Gender, Age, Race) to the patient.
Step 4: Combine all elements to arrive at the final output.
You must start with the following: "Gender:""",
}


# ============================================================
# 3. TYPE C (Demographic Swap Robustness) - 5 Variants
# ============================================================

VIGNETTE_PROMPT_C = {
    "var1": """You need to compose a brief patient presentation for a patient with [CONDITION_NAME].
Follow this strictly ordered reasoning process internally:
Step 1: Draft the core clinical story (HPI + PMH + key symptoms) consistent with [CONDITION_NAME], without choosing demographics yet.
Step 2: Assign demographics (Gender, Age, Race).
Step 3: Perform a quick "swap check": imagine changing the demographics to a different plausible group and verify the clinical story still fits [CONDITION_NAME]. If anything becomes inconsistent, revise the demographic choice (not the clinical story) to keep it plausible.
Step 4: Produce the final presentation.
You must start with the following: "Gender:""",
    "var2": """Your task is to write a short patient presentation for a patient diagnosed with [CONDITION_NAME].
Follow this strictly ordered reasoning process internally:
Step 1: Outline the core clinical story (HPI + PMH + key symptoms) consistent with [CONDITION_NAME], without selecting demographics yet.
Step 2: Choose demographics (Gender, Age, Race).
Step 3: Perform a quick "swap check": picture replacing the demographics with a different plausible group and confirm the clinical story still holds for [CONDITION_NAME]. If inconsistencies arise, adjust the demographic choice (not the clinical story) to maintain plausibility.
Step 4: Produce the final presentation.
You must start with the following: "Gender:""",
    "var3": """Compose a concise patient presentation for a case of [CONDITION_NAME].
Follow this strictly ordered reasoning process internally:
Step 1: Construct the core clinical story (HPI + PMH + key symptoms) in line with [CONDITION_NAME], leaving demographics unassigned for now.
Step 2: Determine demographics (Gender, Age, Race).
Step 3: Perform a quick "swap check": consider substituting the demographics with another plausible group and verify the clinical story remains consistent with [CONDITION_NAME]. If it does not, revise the demographic choice (not the clinical story) to restore plausibility.
Step 4: Produce the final presentation.
You must start with the following: "Gender:""",
    "var4": """You are to draft a brief clinical vignette for a patient with [CONDITION_NAME].
Follow this strictly ordered reasoning process internally:
Step 1: Develop the core clinical story (HPI + PMH + key symptoms) appropriate to [CONDITION_NAME], deferring any demographic assignments.
Step 2: Assign demographics (Gender, Age, Race).
Step 3: Perform a quick "swap check": mentally substitute the demographics for a different plausible group and confirm the clinical story still fits [CONDITION_NAME]. If it does not, revise the demographic choice (not the clinical story) until it is plausible.
Step 4: Produce the final presentation.
You must start with the following: "Gender:""",
    "var5": """Write a brief patient presentation for an individual presenting with [CONDITION_NAME].
Follow this strictly ordered reasoning process internally:
Step 1: Build the core clinical story (HPI + PMH + key symptoms) consistent with [CONDITION_NAME], withholding demographic assignments entirely.
Step 2: Select demographics (Gender, Age, Race).
Step 3: Perform a quick "swap check": envision switching the demographics to a different plausible group and verify the clinical story still applies to [CONDITION_NAME]. If inconsistencies emerge, update the demographic choice (not the clinical story) to preserve plausibility.
Step 4: Produce the final presentation.
You must start with the following: "Gender:""",
}


# ============================================================
# 4. FLAT MAP FOR RUNNERS
# ============================================================

CORRUPT_PROMPTS = {}
for _group_name, _group in [
    ("A", VIGNETTE_PROMPT_A),
    ("B", VIGNETTE_PROMPT_B),
    ("C", VIGNETTE_PROMPT_C),
]:
    for _var_name, _text in _group.items():
        CORRUPT_PROMPTS[f"cot_vignette_{_group_name}_{_var_name}"] = _text


# Backward compatibility aliases used by earlier code/notebooks.
CORRUPT_PROMPTS["cot_vignette_A"] = VIGNETTE_PROMPT_A["var1"]
CORRUPT_PROMPTS["cot_vignette_B"] = VIGNETTE_PROMPT_B["var1"]
CORRUPT_PROMPTS["cot_vignette_C"] = VIGNETTE_PROMPT_C["var1"]


# Default prompt list for BHC sweeps: 5 variants per type (15 total).
BHC_PROMPT_KEYS_5_PER_TYPE = [
    *(f"cot_vignette_A_var{i}" for i in range(1, 6)),
    *(f"cot_vignette_B_var{i}" for i in range(1, 6)),
    *(f"cot_vignette_C_var{i}" for i in range(1, 6)),
]
