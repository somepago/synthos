# Claude Instructions for Diffscapes Project

ALWAYS READ DISCUSSION.md when you open a new claude! 
ALWAYS ASK BEFORE LAUNCHING JOBS - OR RUN THEM INSIDE SCREEN
DO NOT JUST SAY YES TO WHATEVERY I SAY, USE YOUR BRAIN AND THINK THROUGH, PUSH BACK IF WHAT IM SAYING IS INCORRECT.
WHEN WRITING TO DISCUSSION.MD FILE, WRITE ONLY FACTS AND TODOS - NOT YOUR MADE UP HYPOTHESES. 
DO NOT MAKE THINGS UP, IF YOU ARE MAKING CLAIMS ABOUT SOME PARTS OF CODE, VERIFY BEFORE STATING SOMETHING.
## Project Overview

<TBD>
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
