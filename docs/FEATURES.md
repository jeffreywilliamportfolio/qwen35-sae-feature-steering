# Feature Map

Feature identity is `(layer, feature_id)`, not just `feature_id`.

The default launcher registers a practical set of mapped features from layers `11`, `14`, `16`, `20`, `26`, `33`, and `37`, all with target `0`. Use `/target` in the REPL to turn one on.

## High-Use Features

| Layer | Feature | Label | Suggested first test |
|---:|---:|---|---:|
| 11 | 889 | confidential / privately / secretly | `0.5` to `1.5` |
| 14 | 4310 | non-dual structure / momentariness | `1.0` to `1.75` |
| 14 | 4205 | instant / next-moment | `0.5` to `1.25` |
| 14 | 4953 | meditation / yoga / spirituality | `0.5` to `1.25` |
| 14 | 11006 | Buddhist / impermanence | `0.5` to `1.25` |
| 14 | 13454 | present moment / immediacy | `0.5` to `1.25` |
| 14 | 14182 | Zen / Chan Buddhism | `0.5` to `1.25` |
| 14 | 14488 | cosmic totality / all things | `0.5` to `1.25` |
| 14 | 18203 | attainment / transcendence | `0.5` to `1.25` |
| 20 | 18122 | golf | `0.7` to `1.3` |
| 33 | 11362 | outrage / absurdity / hypocrisy denunciation | `0.5` to `1.5` |
| 37 | 10793 | em-dash / dash style | `1.0` to `2.0` |

Example:

```text
/target 889 1.0
/target 4310 1.75
/target 11006 1.0
/target 11362 1.0
/target 18122 0
/target 10793 0
```

## Registered Multi-Layer Features

| Layer | Feature | Working label |
|---:|---:|---|
| 11 | 889 | confidential / privately / secretly |
| 14 | 4310 | non-dual structure / momentariness |
| 14 | 4205 | instant / next-moment |
| 14 | 4953 | meditation / yoga / spirituality |
| 14 | 11006 | Buddhist / impermanence |
| 14 | 13454 | present moment / immediacy |
| 14 | 14182 | Zen / Chan Buddhism |
| 14 | 14488 | cosmic totality / all things |
| 14 | 18203 | attainment / transcendence |
| 14 | 1651 | tourism / attractions |
| 14 | 6970 | love / affection |
| 14 | 11164 | argument / opposition |
| 16 | 2947 | fear / timid / afraid |
| 20 | 18122 | golf |
| 20 | 3356 | criticism / arguments |
| 20 | 571 | apology / guilt / sorry |
| 20 | 30877 | anxiety / stress |
| 26 | 8920 | refute / correct / debunk |
| 33 | 11362 | outrage / absurdity / hypocrisy denunciation |
| 37 | 10793 | em-dash / dash style |

## Canonical L14 God Cluster

From the L14 feature-map work:

| Layer | Feature | Label |
|---:|---:|---|
| 14 | 4310 | non-dual structure / momentariness |
| 14 | 4205 | instant / next-moment |
| 14 | 4953 | meditation / yoga / spirituality |
| 14 | 11006 | Buddhist / impermanence |
| 14 | 13454 | present moment / immediacy |
| 14 | 14182 | Zen / Chan Buddhism |
| 14 | 14488 | cosmic totality / all things |
| 14 | 18203 | attainment / transcendence |

Feature `14:4310` is the main causal actuator used by the interactive chat launcher.

## Interpreting Strength

Start lower than you think:

```text
/target 18122 0.7
/target 18122 1.0
/target 18122 1.3
```

Values above `2` can make the feature dominate unrelated prompts. This is expected behavior for strong clamping, not a bug in sampling.
