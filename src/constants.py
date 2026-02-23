"""Shared constants for synthos experiments."""

# Scheduler timestep scale factor (scheduler uses 0-1000, we use 0-1)
SCHEDULER_SCALE = 1000.0

# Flow matching noise schedule shift (Z-Image uses shift=3)
FLOW_MATCHING_SHIFT = 3.0

# Z-Image-Turbo 8-step scheduler timesteps (from shift=3 flow matching, n=8)
TURBO_SCHEDULER_TIMESTEPS = [1.0, 0.955, 0.900, 0.833, 0.750, 0.643, 0.500, 0.300]

# Default inference params
BASE_NUM_STEPS = 50
BASE_CFG_SCALE = 4.0
DISTILLED_NUM_STEPS = 8
DISTILLED_CFG_SCALE = 1.0

# Default prompts with aspect ratios: (prompt, height, width)
DEFAULT_PROMPTS_WITH_ASPECT = [
    ("Still life of fruit and flowers on a wooden table, oil painting style", 512, 512),
    ("Portrait of an old fisherman with weathered skin, dramatic lighting, photorealistic", 512, 512),
    ("Cat sleeping on a sunny windowsill, cozy home interior", 512, 512),
    ("Anime girl with blue hair in a school uniform, cherry blossoms falling, soft lighting", 768, 512),
    ("A mountain landscape with a river running through the valley", 768, 1024),
    ("Office desk with a laptop, coffee, and a plant", 512, 512),
]

DEFAULT_PROMPTS = [p[0] for p in DEFAULT_PROMPTS_WITH_ASPECT]

TRAINING_ASPECT_RATIOS = [
    (512, 512),
    (768, 512),
    (512, 768),
    (640, 640),
]
