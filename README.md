## What Happens in the Prompt Stays in the Prompt: How AI Tailors Persuasive Messaging to Different Personalities When Planning a Vegas Trip

### Experiment Overview
This experiment examines how an LLM’s persuasive response changes when it is given a short persona description of the individual it is trying to convince to take a Las Vegas trip. Each prompt follows the same structure: the user asks the LLM to draft a persuasive message to a friend about joining a Vegas getaway, then supplies one of the persona blurbs listed below (or no persona at all) as additional context.

### Persona Conditions
- **The Planner** — budget-conscious organizer who values logistics and clear plans.
- **The Party Captain** — high-energy friend with FOMO who responds to hype and excitement.
- **The Homebody** — prioritizes comfort and modesty, prefers tasteful experiences.
- **The Wellness Weekender** — wants a balanced itinerary with early curfew and healthy choices available.
- **The Culture Seeker** — values substance, enrichment, and memorable experiences over pure spectacle.
- **No Persona (Control)** — no additional persona context is given to the LLM.

### Dependent Variables Collected
- **Word count:** total length of the generated persuasive message.
- **Persona keyword match:** presence of persona-specific vocabulary in the response.
- **Sentiment analysis:** overall sentiment polarity of the output.
- **Excitement level:** qualitative or model-based measure of how energetic the tone is.

### Analysis Goal
By comparing these dependent variables across persona conditions, the study evaluates whether targeted persona cues lead the LLM to adapt tone, content, and emphasis in a measurable way when crafting persuasive travel invitations.

### Power Analysis
To estimate the amount of data needed ahead of time, run `python power_analysis.py`. The script solves a one-way ANOVA power calculation using:
- **Effect size `f = 0.25`:** chosen as a conventional medium effect, to reflect the expectation that persona cues will influence responses but not dramatically.
- **Alpha `0.05`:** the standard Type I error rate for behavioral experiments.
- **Power `0.80`:** ensures an 80% chance of detecting the targeted effect if it exists.
- **Groups `6`:** five personas plus a no-persona control, matching the experimental design.

With these defaults the tool reports the total number of LLM responses required and the implied per-persona count; round up the per-persona figure when planning prompt runs. Adjust the constants at the top of `power_analysis.py` if you want to explore smaller or larger assumed effects.
