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


### Grundsätzliche Idee der File (28.06, Jonas)

---

- 2 Möglichkeiten:
  - File besteht aus 3 methoden die jeweils das Detail der übergeben 
  bekommenen Environment "ändert"
  - File besteht aus einer Klasse, die bei Erstellung eine Environment 
  übergeben bekommt und die Environment nach allen Veränderungen 
  zurückgibt
- reward recht einfach, TransformReward()
- preprocessing schwieriger, viele parameter unerklärt.
  - kein frame skipping da frame stack = 4,
  - oder frame skipping = 4 da frame stack = 4 und 4 frames das gleiche 
  frame da ist, deswegen den frame skippen?
  - *** atariPreprocessing nicht sinnvoll da von 210x180 auf 84x84 gescaled wird.
  - scaling muss selber passieren => sleber schreiben
  - greyscale ebenfalls selber schreiben.
- Framestacing recht einfach, FrameStack.
