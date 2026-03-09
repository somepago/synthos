# Claude Instructions for Diffscapes Project

-ALWAYS READ DISCUSSION.md when you open a new claude! 
-ALWAYS ASK BEFORE LAUNCHING JOBS - and always **RUN THEM INSIDE SCREEN**
-DO NOT JUST SAY YES TO WHATEVERY I SAY, USE YOUR BRAIN AND THINK THROUGH, PUSH BACK IF WHAT IM SAYING IS INCORRECT.
-WHEN WRITING TO DISCUSSION.MD FILE, WRITE ONLY FACTS AND TODOS - NOT YOUR MADE UP HYPOTHESES. 
-DO NOT MAKE THINGS UP, IF YOU ARE MAKING CLAIMS ABOUT SOME PARTS OF CODE, VERIFY BEFORE STATING SOMETHING.
-MOST LIKELY THIS IS A UNIFIED MACHINE (DGX SPARK) - SO CHECK FIRST BEFORE YOU START WRITING OR PROPOSING OFFLOADING SORT OF PATTERNS WHEN WE HIT OOMS
-DO NOT HIDE TESTING FOLDERS WHEN IM ASKING YOU TO TEST SOME THINGS - ALWAYS CLEARLY STATE WHERE I CAN SEE THOSE TESTING GENS OR LOGS, EITHER MAKE THEM IN PROJECT FOLDER OR EXPLICITLY TELL ME WHERE II CAN ACCESS THOSE.
## Project Overview

<TBD>

## Hard Facts (DO NOT suggest alternatives to these)

- **Z-Image-Turbo is text-to-image only** — no siglip weights in checkpoint
- **Omni weights are not released** — cannot load pretrained siglip projection
- **Frozen DiT cannot learn to attend to new SigLip tokens from random projection alone** — the attention layers were never trained with SigLip features
- **Pure SigLip conditioning without VAE tokens is unreliable** — the omni mode noise_mask/dual-adaln path is not viable for our setup
- **Do NOT suggest omni mode, noise_mask tricks, or dummy VAE refs** — these have been tried and failed
- **Current approach**: 
## Workflow Rules


### Git & Commits
- **NEVER commit without explicit user permission**
- Always show git status before asking to commit
- When committing, use descriptive messages that explain the "why" not just the "what"
- DO NOT Include co-author tag: `Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>`

### Code Changes
- **Read files before editing them** - always understand context first
- **Prefer editing existing files over creating new ones** - avoid file bloat
- **Keep research code simple** - don't over-engineer or add unnecessary abstractions
- Only add features/refactors that are directly requested
- Don't add comments, docstrings, or type hints to code you didn't change
- Don't add error handling for scenarios that can't happen

### Testing & Verification
- After refactoring, verify imports work (syntax check at minimum)
- Don't run actual training/inference without permission (runs on expensive GPU)
- Show diagnostic output but don't over-explain environment-specific warnings (torch/numpy import errors on local machine are expected)


## Documentation Updates

When making significant changes, update:
- `README.md` - user-facing documentation, usage examples
- `DISCUSSION.md` - research notes, fixes, insights
- Keep `PLAN.md` as historical record (don't update unless working on those experiments)


## Communication Style

- Be concise - don't over-explain
- Use file:line_number format when referencing code locations
- No emojis unless explicitly requested
- Focus on what you're doing, not time estimates
