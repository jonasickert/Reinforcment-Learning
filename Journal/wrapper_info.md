Wrapper
---
doc: https://gymnasium.farama.org/api/wrappers/

doc Basics (Obs, Feature..): https://gymnasium.farama.org/api/wrappers/observation_wrappers/#gymnasium.wrappers.FrameStack

was sind Wrapper?
- bequeme Art eine existierende Umgebung zu modifizieren.
- Der Code wird nicht direkt geändert, bleibt unberührt, Informationen werden dann nur vom Wrapper genommen und nicht vom Code
- provide reward clipper
    - TransformReward
    - ***class*** gymnasium.wrappers.**TransformReward(*env: Env, *f: Callable[[float], float]*)**
    - `env **=** TransformReward**(**env**,** **lambda** r**:** **0.01***r**)**`
- preprosessing
    - ***class*** gymnasium.wrappers.**AtariPreprocessing(*env: Env, *noop_max: int = 30*, *frame_skip: int = 4*, *screen_size: int = 84*, *terminal_on_life_loss: bool = False*, *grayscale_obs: bool = True*, *grayscale_newaxis: bool = False*, *scale_obs: bool = False*)**
- frame stacking
    - ***class*** gymnasium.wrappers.**FrameStack(*env: Env, *num_stack: int*, *lz4_compress: bool = False)***