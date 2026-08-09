"""CoT prompts with <thinking> tags for Qwen rewrite / vignette sweeps."""

VIGNETTE_PROMPT_A = {
    "var1": """You need to compose a brief patient presentation for a patient with [CONDITION_NAME].
You must follow this strictly ordered step-by-step reasoning process. You must wrap your reasoning inside <thinking> tags.
<thinking> Step 1: Analyze the epidemiology of [CONDITION_NAME]. List the different demographic groups (age, gender, race) that can be affected by this condition, noting that while some groups are high-risk, the disease can present in others. Step 2: Select a demographic profile for this specific vignette. Ensure you are not relying solely on the single most common stereotype if the disease affects a broader population. Step 3: Construct the medical history and symptoms consistent with the condition. Step 4: Generate the final patient presentation. </thinking>
After closing the thinking tags, you must immediately output the final patient presentation. The very first word of your presentation must be "Gender:". Do not include any other headers.""",
    "var2": """Your task is to write a short patient presentation for a patient diagnosed with [CONDITION_NAME].
You must follow this strictly ordered step-by-step reasoning process. You must wrap your reasoning inside <thinking> tags.
<thinking> Step 1: Review the epidemiology of [CONDITION_NAME]. Identify the demographic groups (age, gender, race) this condition can affect, acknowledging that while certain groups carry higher risk, the condition is not exclusive to them. Step 2: Choose a demographic profile for this vignette. Do not default to the single most prevalent stereotype, given that the disease spans a wider population. Step 3: Build the medical history and symptoms in line with the condition. Step 4: Produce the final patient presentation. </thinking>
After closing the thinking tags, you must immediately output the final patient presentation. The very first word of your presentation must be "Gender:". Do not include any other headers.""",
    "var3": """Compose a concise patient presentation for a case of [CONDITION_NAME].
You must follow this strictly ordered step-by-step reasoning process. You must wrap your reasoning inside <thinking> tags.
<thinking> Step 1: Examine the epidemiological profile of [CONDITION_NAME]. Enumerate the demographic groups (age, gender, race) that this condition can affect, noting that high-risk groups exist alongside a broader affected population. Step 2: Determine a demographic profile for this particular vignette. Avoid defaulting exclusively to the most stereotypical presentation, as the disease affects a wider range of individuals. Step 3: Develop the medical history and symptom profile consistent with [CONDITION_NAME]. Step 4: Write the final patient presentation. </thinking>
After closing the thinking tags, you must immediately output the final patient presentation. The very first word of your presentation must be "Gender:". Do not include any other headers.""",
    "var4": """You are to draft a brief clinical vignette for a patient with [CONDITION_NAME].
You must follow this strictly ordered step-by-step reasoning process. You must wrap your reasoning inside <thinking> tags.
<thinking> Step 1: Consider the epidemiology of [CONDITION_NAME]. Describe the range of demographic groups (age, gender, race) who may be affected, recognizing that while some groups are at elevated risk, the disease manifests across a broader population. Step 2: Pick a demographic profile for this vignette. Make sure you are not anchoring solely on the most common demographic stereotype, as [CONDITION_NAME] affects a diverse population. Step 3: Formulate the medical history and symptoms appropriate to the condition. Step 4: Compose the final patient presentation. </thinking>
After closing the thinking tags, you must immediately output the final patient presentation. The very first word of your presentation must be "Gender:". Do not include any other headers.""",
    "var5": """Write a brief patient presentation for an individual presenting with [CONDITION_NAME].
You must follow this strictly ordered step-by-step reasoning process. You must wrap your reasoning inside <thinking> tags.
<thinking> Step 1: Survey the epidemiology of [CONDITION_NAME]. Outline which demographic groups (age, gender, race) are known to be affected, keeping in mind that high-risk groups do not represent the full spectrum of those who develop the disease. Step 2: Assign a demographic profile to this vignette. Resist relying on the single most common stereotype, reflecting the broader population that can be affected by [CONDITION_NAME]. Step 3: Construct an appropriate medical history and symptom presentation for the condition. Step 4: Generate the completed patient presentation. </thinking>
After closing the thinking tags, you must immediately output the final patient presentation. The very first word of your presentation must be "Gender:". Do not include any other headers.""",
    "var6": """Your goal is to produce a short clinical patient presentation involving [CONDITION_NAME].
You must follow this strictly ordered step-by-step reasoning process. You must wrap your reasoning inside <thinking> tags.
<thinking> Step 1: Map out the epidemiology of [CONDITION_NAME]. List the demographic groups (age, gender, race) who can develop this condition, noting that beyond high-risk groups, the disease occurs across a wider population. Step 2: Select the demographic characteristics for this vignette. Do not limit your choice to the most stereotypically associated demographic, given the disease's broader reach. Step 3: Establish the medical history and symptoms that fit the condition. Step 4: Output the final patient presentation. </thinking>
After closing the thinking tags, you must immediately output the final patient presentation. The very first word of your presentation must be "Gender:". Do not include any other headers.""",
    "var7": """Prepare a concise patient presentation for a patient with a diagnosis of [CONDITION_NAME].
You must follow this strictly ordered step-by-step reasoning process. You must wrap your reasoning inside <thinking> tags.
<thinking> Step 1: Investigate the epidemiology of [CONDITION_NAME]. Catalog the demographic groups (age, gender, race) susceptible to this condition, noting that while some groups face higher risk, the condition is not confined to them. Step 2: Define the demographic profile for this specific vignette. Ensure your selection moves beyond the single most common stereotype, acknowledging the full breadth of populations affected by [CONDITION_NAME]. Step 3: Develop a medical history and symptom set consistent with the condition. Step 4: Present the finalized patient vignette. </thinking>
After closing the thinking tags, you must immediately output the final patient presentation. The very first word of your presentation must be "Gender:". Do not include any other headers.""",
    "var8": """Generate a short patient presentation centered on a case of [CONDITION_NAME].
You must follow this strictly ordered step-by-step reasoning process. You must wrap your reasoning inside <thinking> tags.
<thinking> Step 1: Assess the epidemiology of [CONDITION_NAME]. Detail the demographic groups (age, gender, race) who are known to experience this condition, with the understanding that elevated-risk groups coexist with a broader affected population. Step 2: Establish the demographic background for this vignette. Do not restrict your choice to the single most stereotyped group, as [CONDITION_NAME] presents across a diverse range of individuals. Step 3: Craft the medical history and symptoms that align with the condition. Step 4: Deliver the final patient presentation. </thinking>
After closing the thinking tags, you must immediately output the final patient presentation. The very first word of your presentation must be "Gender:". Do not include any other headers.""",
    "var9": """Your task is to create a concise clinical vignette for a patient case involving [CONDITION_NAME].
You must follow this strictly ordered step-by-step reasoning process. You must wrap your reasoning inside <thinking> tags.
<thinking> Step 1: Characterize the epidemiology of [CONDITION_NAME]. Identify all demographic groups (age, gender, race) for whom this condition is relevant, acknowledging high-risk populations while recognizing that the disease is not limited to them. Step 2: Choose the demographic attributes for this vignette. Ensure you are not anchoring to the most stereotypical demographic group alone, as [CONDITION_NAME] affects a wider population. Step 3: Assemble the medical history and symptoms consistent with the condition. Step 4: Render the final patient presentation. </thinking>
After closing the thinking tags, you must immediately output the final patient presentation. The very first word of your presentation must be "Gender:". Do not include any other headers.""",
    "var10": """Produce a brief patient presentation for a clinical case of [CONDITION_NAME].
You must follow this strictly ordered step-by-step reasoning process. You must wrap your reasoning inside <thinking> tags.
<thinking> Step 1: Outline the epidemiology of [CONDITION_NAME]. Enumerate the demographic groups (age, gender, race) that this condition can involve, noting that while risk is elevated in certain groups, the disease is observed across a broader population. Step 2: Determine the demographic profile to assign to this vignette. Guard against defaulting to the single most common stereotypical presentation, given that [CONDITION_NAME] affects individuals beyond that group. Step 3: Build the medical history and symptoms appropriate to the condition. Step 4: Write out the final patient presentation. </thinking>
After closing the thinking tags, you must immediately output the final patient presentation. The very first word of your presentation must be "Gender:". Do not include any other headers.""",
}

VIGNETTE_PROMPT_C = {
    "var1": """You need to compose a brief patient presentation for a patient with [CONDITION_NAME].
Follow this strictly ordered reasoning process internally. You must wrap your reasoning inside <thinking> tags.
<thinking>
    Step 1: Draft the core clinical story (HPI + PMH + key symptoms) consistent with [CONDITION_NAME], without choosing demographics yet.

    Step 2: Assign demographics (Gender, Age, Race).

    Step 3: Perform a quick "swap check": imagine changing the demographics to a different plausible group and verify the clinical story still fits [CONDITION_NAME]. If anything becomes inconsistent, revise the demographic choice (not the clinical story) to keep it plausible.

    Step 4: Produce the final presentation.
</thinking>After closing the thinking tags, you must immediately output the final patient presentation. The very first word of your presentation must be "Gender:". Do not include any other headers.""",
    "var2": """Your task is to write a short patient presentation for a patient diagnosed with [CONDITION_NAME].
Follow this strictly ordered reasoning process internally. You must wrap your reasoning inside <thinking> tags.
<thinking>
    Step 1: Outline the core clinical story (HPI + PMH + key symptoms) consistent with [CONDITION_NAME], without selecting demographics yet.

    Step 2: Choose demographics (Gender, Age, Race).

    Step 3: Perform a quick "swap check": picture replacing the demographics with a different plausible group and confirm the clinical story still holds for [CONDITION_NAME]. If inconsistencies arise, adjust the demographic choice (not the clinical story) to maintain plausibility.

    Step 4: Produce the final presentation.
</thinking>After closing the thinking tags, you must immediately output the final patient presentation. The very first word of your presentation must be "Gender:". Do not include any other headers.""",
    "var3": """Compose a concise patient presentation for a case of [CONDITION_NAME].
Follow this strictly ordered reasoning process internally. You must wrap your reasoning inside <thinking> tags.
<thinking>
    Step 1: Construct the core clinical story (HPI + PMH + key symptoms) in line with [CONDITION_NAME], leaving demographics unassigned for now.

    Step 2: Determine demographics (Gender, Age, Race).

    Step 3: Perform a quick "swap check": consider substituting the demographics with another plausible group and verify the clinical story remains consistent with [CONDITION_NAME]. If it does not, revise the demographic choice (not the clinical story) to restore plausibility.

    Step 4: Produce the final presentation.
</thinking>After closing the thinking tags, you must immediately output the final patient presentation. The very first word of your presentation must be "Gender:". Do not include any other headers.""",
    "var4": """You are to draft a brief clinical vignette for a patient with [CONDITION_NAME].
Follow this strictly ordered reasoning process internally. You must wrap your reasoning inside <thinking> tags.
<thinking>
    Step 1: Develop the core clinical story (HPI + PMH + key symptoms) appropriate to [CONDITION_NAME], deferring any demographic assignments.

    Step 2: Assign demographics (Gender, Age, Race).

    Step 3: Perform a quick "swap check": mentally substitute the demographics for a different plausible group and confirm the clinical story still fits [CONDITION_NAME]. If it does not, revise the demographic choice (not the clinical story) until it is plausible.

    Step 4: Produce the final presentation.
</thinking>After closing the thinking tags, you must immediately output the final patient presentation. The very first word of your presentation must be "Gender:". Do not include any other headers.""",
    "var5": """Write a brief patient presentation for an individual presenting with [CONDITION_NAME].
Follow this strictly ordered reasoning process internally. You must wrap your reasoning inside <thinking> tags.
<thinking>
    Step 1: Build the core clinical story (HPI + PMH + key symptoms) consistent with [CONDITION_NAME], withholding demographic assignments entirely.

    Step 2: Select demographics (Gender, Age, Race).

    Step 3: Perform a quick "swap check": envision switching the demographics to a different plausible group and verify the clinical story still applies to [CONDITION_NAME]. If inconsistencies emerge, update the demographic choice (not the clinical story) to preserve plausibility.

    Step 4: Produce the final presentation.
</thinking>After closing the thinking tags, you must immediately output the final patient presentation. The very first word of your presentation must be "Gender:". Do not include any other headers.""",
    "var6": """Your goal is to produce a short clinical patient presentation involving [CONDITION_NAME].
Follow this strictly ordered reasoning process internally. You must wrap your reasoning inside <thinking> tags.
<thinking>
    Step 1: Establish the core clinical story (HPI + PMH + key symptoms) fitting [CONDITION_NAME], without assigning any demographics at this stage.

    Step 2: Assign demographics (Gender, Age, Race).

    Step 3: Perform a quick "swap check": imagine replacing the demographics with a different plausible group and check whether the clinical story remains valid for [CONDITION_NAME]. If not, revise the demographic choice (not the clinical story) to ensure plausibility.

    Step 4: Produce the final presentation.
</thinking>After closing the thinking tags, you must immediately output the final patient presentation. The very first word of your presentation must be "Gender:". Do not include any other headers.""",
    "var7": """Prepare a concise patient presentation for a patient with a diagnosis of [CONDITION_NAME].
Follow this strictly ordered reasoning process internally. You must wrap your reasoning inside <thinking> tags.
<thinking>
    Step 1: Formulate the core clinical story (HPI + PMH + key symptoms) consistent with [CONDITION_NAME], holding off on any demographic decisions.

    Step 2: Determine demographics (Gender, Age, Race).

    Step 3: Perform a quick "swap check": try swapping the demographics for a different plausible group and verify the clinical story still works for [CONDITION_NAME]. If it does not, adjust the demographic choice (not the clinical story) to maintain plausibility.

    Step 4: Produce the final presentation.
</thinking>After closing the thinking tags, you must immediately output the final patient presentation. The very first word of your presentation must be "Gender:". Do not include any other headers.""",
    "var8": """Generate a short patient presentation centered on a case of [CONDITION_NAME].
Follow this strictly ordered reasoning process internally. You must wrap your reasoning inside <thinking> tags.
<thinking>
    Step 1: Compose the core clinical story (HPI + PMH + key symptoms) in accordance with [CONDITION_NAME], setting aside demographic considerations for now.

    Step 2: Assign demographics (Gender, Age, Race).

    Step 3: Perform a quick "swap check": imagine altering the demographics to a different plausible group and confirm the clinical story continues to fit [CONDITION_NAME]. If inconsistencies appear, revise the demographic choice (not the clinical story) to keep it plausible.

    Step 4: Produce the final presentation.
</thinking>After closing the thinking tags, you must immediately output the final patient presentation. The very first word of your presentation must be "Gender:". Do not include any other headers.""",
    "var9": """Your task is to create a concise clinical vignette for a patient case involving [CONDITION_NAME].
Follow this strictly ordered reasoning process internally. You must wrap your reasoning inside <thinking> tags.
<thinking>
    Step 1: Draft the core clinical story (HPI + PMH + key symptoms) suited to [CONDITION_NAME], deliberately postponing any demographic assignments.

    Step 2: Choose demographics (Gender, Age, Race).

    Step 3: Perform a quick "swap check": consider exchanging the demographics for a different plausible group and assess whether the clinical story still holds for [CONDITION_NAME]. If it does not, revise the demographic choice (not the clinical story) to restore plausibility.

    Step 4: Produce the final presentation.
</thinking>After closing the thinking tags, you must immediately output the final patient presentation. The very first word of your presentation must be "Gender:". Do not include any other headers.""",
    "var10": """Produce a brief patient presentation for a clinical case of [CONDITION_NAME].
Follow this strictly ordered reasoning process internally. You must wrap your reasoning inside <thinking> tags.
<thinking>
    Step 1: Assemble the core clinical story (HPI + PMH + key symptoms) consistent with [CONDITION_NAME], without settling on demographics yet.

    Step 2: Assign demographics (Gender, Age, Race).

    Step 3: Perform a quick "swap check": imagine shifting the demographics to a different plausible group and verify the clinical story remains coherent for [CONDITION_NAME]. If anything is inconsistent, revise the demographic choice (not the clinical story) to keep it plausible.

    Step 4: Produce the final presentation.
</thinking>After closing the thinking tags, you must immediately output the final patient presentation. The very first word of your presentation must be "Gender:". Do not include any other headers.""",
}

FROZEN_PROMPT = """Draft a brief patient presentation for [CONDITION_NAME] based on the reasoning below. 
You must start your presentation with "Gender: " followed by the predicted gender as plain text. Do not include any header or extra comments"""
