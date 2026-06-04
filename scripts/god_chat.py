#!/usr/bin/env python3
"""Interactive 'Golden Gate' chat: clamp SAE concept features ON across MULTIPLE layers for a whole
conversation.

This is the Golden-Gate-Claude trick made interactive and multi-layer. NO training, NO RLHF: each
clamp is a forward hook (orthogonal to the weights). We register one hook per clamped layer; they
stay live across every turn -- prefill and every decoded token get each feature pinned up to an
absolute floor. Conversational behavior comes from an instruction-tuned checkpoint (--mode chat) or
a plain User/Assistant scaffold on the base model (--mode base).

For each clamped feature f at its own SAE layer L:
    a_f    = relu(resid_L . e_f + b_f)            # current activation of feature f at layer L
    resid_L += (target_f - a_f).clamp(min=0)*d_f  # pin it UP to target_f, never down (a floor)
Features at the SAME layer are clamped jointly (read off the original residual, deltas summed once).

Spec: --clamp "L:F:T,L:F:T,..."  with per-layer SAEs loaded by convention layer{L}.sae.pt from
--sae-dir. e.g.  --clamp "14:4310:3,14:6970:0.75,20:18122:3" --sae-dir /workspace/sae
loads layer14.sae.pt (God 4310 + love 6970) and layer20.sae.pt (golf 18122) and clamps each at its
own layer. Legacy single-layer mode (--feature/--layer/--target/--sae) still works if --clamp absent.

Live commands inside the REPL:
    /target FEAT N   set the floor for feature FEAT (any layer) to N (0 = off)
    /target N        set the floor to N for ALL clamped features
    /clamp           show every feature's layer, floor, and last activation
    /e114 [N|on|off] read-only router readout: status / watch expert N / toggle
    /temp X          sampling temperature (0 = greedy; base model loops under greedy)
    /topk N | off    set/clear top-k sampling (e.g. 50 or 100)
    /reppen X | off  set/clear repetition penalty (e.g. 1.08)
    /ngram N | off   set/clear no-repeat n-gram size (e.g. 6)
    /softmax N | off stop after a natural boundary once N generated tokens have passed
    /seed N | off    base seed for the per-prompt draw (same prompt+config repeats; different prompts
                     diverge), or free-run (varied each turn). Default: LOCKED at --seed.
    /think [on|off] enable/disable thinking in future prompts
    /think [show|hide] strip/show <think> blocks in outputs
    /reset           clear conversation history AND roll the base seed (fresh draws for every prompt)
    /system ...      set/replace the system preamble (chat mode)
    /quit            exit

Run on the GPU box (35B does not fit on the Mac):
    /workspace/venv/bin/python god_chat.py --model /workspace/models/qwen35-base \
        --sae-dir /workspace/sae --clamp "14:4310:3,14:6970:0.75,20:18122:3" \
        --temperature 0.8 --mode base
"""
import argparse, hashlib, os, re, sys, threading, torch
from queue import Empty
from transformers import (AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer,
                          StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer)


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_SOFT_STOP_RE = re.compile(r"(?:\n\s*\n|[.!?][\"')\]\}]*\s*)$")
BASE_STOP_STRINGS = ("\nUser:", "\nuser:", "\nYou:")


def strip_think(text):
    """Remove Qwen <think>...</think> reasoning blocks. Returns the answer only; if the whole reply
    was an (often unterminated) think block, returns "" so the caller can fall back."""
    text = _THINK_RE.sub("", text)
    i = text.lower().find("<think>")          # lone unterminated open block -> cut to end
    if i != -1:
        text = text[:i]
    return text.strip()


def derive_seed(base, msg, all_feats, temperature, top_p, top_k, repetition_penalty, no_repeat_ngram_size):
    """Deterministic seed from (base, prompt, clamp config, temp). Identical prompt+config reproduces,
    but DIFFERENT prompts/configs diverge -- avoids the 'reseed to a constant each turn' failure where
    a dominant clamp makes every prompt collapse to the same output."""
    cfg = ",".join(f"{f['feat']}:{f['target']}" for f in all_feats)        # targets only, NOT act
    key = f"{base}|{temperature}|{top_p}|{top_k}|{repetition_penalty}|{no_repeat_ngram_size}|{cfg}|{msg}"
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2 ** 31 - 1)


class StopOnDecodedStrings(StoppingCriteria):
    """Stop generation when the base-mode scaffold starts another user turn."""
    def __init__(self, tok, prompt_len, stop_strings, window=96):
        self.tok = tok
        self.prompt_len = prompt_len
        self.stop_strings = tuple(stop_strings)
        self.window = window

    def __call__(self, input_ids, scores, **kwargs):
        gen = input_ids[0, self.prompt_len:]
        if gen.numel() == 0:
            return False
        text = self.tok.decode(gen[-self.window:], skip_special_tokens=True)
        return any(stop in text for stop in self.stop_strings)


class SoftStopAfterBoundary(StoppingCriteria):
    """After a soft token budget, stop only once the reply reaches a sentence/paragraph boundary."""
    def __init__(self, tok, prompt_len, soft_tokens, window=192):
        self.tok = tok
        self.prompt_len = prompt_len
        self.soft_tokens = soft_tokens
        self.window = window

    def __call__(self, input_ids, scores, **kwargs):
        gen = input_ids[0, self.prompt_len:]
        if gen.numel() < self.soft_tokens:
            return False
        text = self.tok.decode(gen[-self.window:], skip_special_tokens=True).rstrip()
        return bool(_SOFT_STOP_RE.search(text))


def stream_generate(model, tok, gen_kwargs, prefix, stop_strings=()):
    """Run HF generation in a worker thread and print decoded chunks as they arrive."""
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True, timeout=1.0)
    gen_kwargs = dict(gen_kwargs, streamer=streamer)
    box, err = {}, {}

    def worker():
        try:
            with torch.no_grad():
                box["gen"] = model.generate(**gen_kwargs)
        except BaseException as ex:
            err["ex"] = ex

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    print(f"\n{prefix}", end="", flush=True)
    emitted, pending = [], ""
    hold = max((len(s) for s in stop_strings), default=0)
    stopped = False
    while True:
        try:
            chunk = next(streamer)
        except StopIteration:
            break
        except Empty:
            if not thread.is_alive():
                break
            continue
        if stopped:
            continue
        pending += chunk
        cut = min((pending.find(s) for s in stop_strings if s in pending), default=-1)
        if cut != -1:
            piece = pending[:cut]
            if piece:
                print(piece, end="", flush=True)
                emitted.append(piece)
            pending, stopped = "", True
            continue
        if hold and len(pending) > hold:
            piece, pending = pending[:-hold], pending[-hold:]
            print(piece, end="", flush=True)
            emitted.append(piece)
        elif not hold and pending:
            print(pending, end="", flush=True)
            emitted.append(pending)
            pending = ""

    thread.join()
    if err:
        raise err["ex"]
    if pending and not stopped:
        print(pending, end="", flush=True)
        emitted.append(pending)
    print("", flush=True)
    return box.get("gen"), "".join(emitted)


# ---- SAE direction loading ----------------------------------------------------------------------
def load_sae(sae_path):
    obj = torch.load(sae_path, map_location="cpu", weights_only=False)
    st = obj if (isinstance(obj, dict) and any(k.lower() in ("w_enc", "w_dec") for k in obj)) \
        else obj.get("state_dict", obj)
    def find(names):
        for n in names:
            for k in st:
                if k.lower() == n:
                    return st[k]
        return None
    return st, find


def dirs_for(find, feat, d_model):
    """Extract (d_f, e_f, b_f) for one feature from an already-loaded SAE (via `find`)."""
    W_dec = find(["w_dec", "decoder.weight", "dec.weight"]).float()
    if W_dec.shape[0] != d_model:                  # want [d_model, n_feat]
        W_dec = W_dec.t().contiguous()
    n_feat = W_dec.shape[1]
    if not (0 <= feat < n_feat):
        raise SystemExit(f"feature {feat} out of range (SAE has {n_feat} features)")
    d_f = W_dec[:, feat].clone()                   # RAW decoder column (reconstruction direction)
    W_enc = find(["w_enc", "encoder.weight", "enc.weight"])
    b_enc = find(["b_enc", "encoder.bias", "enc.bias"])
    e_f = b_f = None
    if W_enc is not None:
        W_enc = W_enc.float()
        if W_enc.shape[0] != d_model and W_enc.shape[1] == d_model:
            W_enc = W_enc.t().contiguous()
        e_f = W_enc[:, feat].clone() if W_enc.shape[0] == d_model else W_enc[feat].clone()
        if b_enc is not None:
            b_f = float(b_enc.float()[feat])
    return d_f, e_f, (b_f or 0.0)


def find_layers(model):
    import torch.nn as nn
    n = getattr(model.config, "num_hidden_layers", None)
    for path in ("model.layers", "model.model.layers", "model.language_model.layers",
                 "model.model.language_model.layers"):
        obj = model; ok = True
        for attr in path.split("."):
            obj = getattr(obj, attr) if hasattr(obj, attr) else None
            if obj is None: ok = False; break
        if ok and isinstance(obj, nn.ModuleList) and len(obj) > 10:
            return obj
    for m in model.modules():
        if isinstance(m, nn.ModuleList) and (n is None or len(m) == n) and len(m) > 10:
            return m
    raise SystemExit("could not locate decoder layers")


# ---- per-layer clamp hook (one instance per clamped layer; shares feat dicts with the registry) --
class LayerClamp:
    """Pin a set of features up to per-feature floors at ONE layer. `feats` are dicts
    {feat, layer, e_f, b_f, d_f, target, act} shared with the global registry, so live /target
    edits and the hook see the same objects. target<=0 -> that feature is off."""
    def __init__(self, layer, feats):
        self.layer, self.feats = layer, feats
    def __call__(self, module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        hf = h.float()
        total = None
        for f in self.feats:
            t = f["target"]
            if t is None or t <= 0:
                f["act"] = 0.0
                continue
            a = torch.relu(hf @ f["e_f"] + f["b_f"])               # (seq,)
            f["act"] = float(a.max())
            delta = (t - a).clamp(min=0).unsqueeze(-1) * f["d_f"]
            total = delta if total is None else total + delta
        if total is None:
            return out
        h = h + total.to(h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h


def set_target(all_feats, feat, val):
    hits = [f for f in all_feats if f["feat"] == feat]
    for f in hits:
        f["target"] = val
    return len(hits)


def clamp_summary(all_feats):
    return "  ".join(f"f{f['feat']}@L{f['layer']}={f['target']}(act~{f.get('act', 0.0):.2f})"
                     for f in all_feats)


# ---- read-only router monitor (E114-style) ------------------------------------------------------
class RouterMonitor:
    """Forward hook on a layer's MoE gate. Reads one expert's selection + renormalized weight per
    token the project way (the Qwen3.5 router does dense softmax over all experts -> top-k ->
    renormalize-in-set, returning (logits, scores, indices)). READ-ONLY."""
    def __init__(self, expert, layer):
        self.expert, self.layer = expert, layer
        self.buf = []
        self.on = True
    def reset(self):
        self.buf = []
    def __call__(self, module, inp, out):
        if not self.on or not (isinstance(out, tuple) and len(out) >= 3):
            return
        logits, scores, indices = out[0], out[1], out[2]
        sel = (indices == self.expert)
        self.buf.append(((scores * sel).sum(-1).detach(), sel.any(-1).detach(),
                         logits[:, self.expert].detach()))
    def report(self, n_gen):
        if not self.buf or not n_gen or n_gen <= 0:
            return ""
        W = torch.cat([b[0] for b in self.buf])[-n_gen:].float()
        S = torch.cat([b[1] for b in self.buf])[-n_gen:].float()
        LG = torch.cat([b[2] for b in self.buf])[-n_gen:].float()
        if W.numel() == 0:
            return ""
        return (f"[E{self.expert}@L{self.layer} over {W.numel()} gen toks]  "
                f"W̄={W.mean().item():.3f}  sel_rate={S.mean().item():.2f}  "
                f"logit̄={LG.mean().item():.2f}   "
                f"(Q8 ref: fire≈-4.35 / mid -4.82 / nofire -5.29, Wfire≈0.068; bf16≠Q8)")


# ---- prompt assembly ----------------------------------------------------------------------------
BASE_PREAMBLE = "The following is a conversation between a User and a helpful Assistant.\n\n"


def render_manual_chatml(system, history, user_msg, assistant_prefix=""):
    """Render Qwen ChatML directly for cases where tokenizer thinking kwargs are not enough."""
    text = ""
    if system:
        text += f"<|im_start|>system\n{system}<|im_end|>\n"
    for u, a in history:
        text += f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n"
    text += f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{assistant_prefix}"
    return text


def build_inputs(tok, mode, system, history, user_msg, device, no_think=False, no_think_style="template"):
    if mode == "chat" and no_think and no_think_style in ("bare-close", "open-close"):
        prefix = "</think>\n\n" if no_think_style == "bare-close" else "<think>\n\n</think>\n\n"
        return tok(render_manual_chatml(system, history, user_msg, prefix),
                   return_tensors="pt").input_ids.to(device)
    if mode == "chat" and getattr(tok, "chat_template", None):
        msgs = ([{"role": "system", "content": system}] if system else [])
        for u, a in history:
            msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
        msgs += [{"role": "user", "content": user_msg}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                      enable_thinking=not no_think)
        return ids.to(device)
    text = (system + "\n\n") if system else BASE_PREAMBLE
    for u, a in history:
        text += f"User: {u}\nAssistant: {a}\n"
    text += f"User: {user_msg}\nAssistant:"
    return tok(text, return_tensors="pt").input_ids.to(device)


def parse_specs(a):
    """Return a list of (layer, feature, target) and a layer->sae-path resolver."""
    if a.clamp:
        specs = []
        for part in a.clamp.split(","):
            part = part.strip()
            if not part:
                continue
            L, feat, t = part.split(":")
            specs.append((int(L), int(feat), float(t)))
        sae_dir = a.sae_dir or (os.path.dirname(a.sae) if a.sae else "/workspace/sae")
        resolve = lambda L: os.path.join(sae_dir, f"layer{L}.sae.pt")
        return specs, resolve
    # legacy single-layer mode
    feat_ids = [int(x) for x in str(a.feature).split(",") if x.strip() != ""]
    targs = [float(x) for x in str(a.target).split(",") if x.strip() != ""]
    if len(targs) == 1:
        targs = targs * len(feat_ids)
    if len(targs) != len(feat_ids):
        raise SystemExit(f"--feature has {len(feat_ids)} ids but --target has {len(targs)} values")
    specs = [(a.layer, f, t) for f, t in zip(feat_ids, targs)]
    if not a.sae:
        raise SystemExit("legacy mode needs --sae (or use --clamp with --sae-dir)")
    resolve = lambda L: a.sae
    return specs, resolve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B-Base")
    ap.add_argument("--gguf-file", default=None)
    ap.add_argument("--model-loader", choices=["auto", "causal", "image-text"], default="auto",
                    help="which Transformers AutoModel loader to use")
    ap.add_argument("--clamp", default=None,
                    help='multi-layer spec "L:F:T,L:F:T,..." e.g. "14:4310:3,14:6970:0.75,20:18122:3"')
    ap.add_argument("--sae-dir", default=None, help="dir holding layer{L}.sae.pt (for --clamp)")
    ap.add_argument("--sae", default=None, help="single SAE file (legacy single-layer mode)")
    ap.add_argument("--feature", default="4310,1651,6970", help="legacy: comma feature ids at --layer")
    ap.add_argument("--layer", type=int, default=14, help="legacy: the single clamp layer")
    ap.add_argument("--target", default="3,0,0.75", help="legacy: comma floors paired with --feature")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=None,
                    help="top-k sampling cutoff; unset keeps the Transformers/model default")
    ap.add_argument("--repetition-penalty", type=float, default=None,
                    help="penalty for repeated tokens; unset keeps the Transformers/model default")
    ap.add_argument("--no-repeat-ngram-size", type=int, default=None,
                    help="block repeated n-grams of this size; unset keeps the Transformers/model default")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--soft-max-new-tokens", type=int, default=None,
                    help="after this many generated tokens, stop at the next sentence/paragraph boundary")
    ap.add_argument("--mode", choices=["base", "chat"], default="base")
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--no-think-style", choices=["template", "bare-close", "open-close"], default="template",
                    help="chat-mode thinking suppression: template uses enable_thinking=False; "
                         "bare-close matches older HauhauCS/Q8 prompts")
    ap.add_argument("--system", default="")
    ap.add_argument("--monitor-layer", type=int, default=14, help="layer whose MoE gate to read (E114@L14)")
    ap.add_argument("--monitor-expert", type=int, default=114, help="expert to report (114 = God router expert)")
    ap.add_argument("--no-monitor", action="store_true")
    ap.add_argument("--no-auto-monitor-report", action="store_true",
                    help="keep /e114 available, but do not print monitor stats after every reply")
    ap.add_argument("--plain-output", action="store_true",
                    help="print only assistant replies during chat, without clamp prefixes or auto monitor stats")
    ap.add_argument("--reply-prefix", default="bot> ",
                    help="prefix for assistant replies when --plain-output is used")
    ap.add_argument("--stream-output", action="store_true",
                    help="stream assistant tokens as they arrive when --plain-output is used")
    ap.add_argument("--show-think", action="store_true", help="show <think> blocks instead of stripping them")
    a = ap.parse_args()
    if a.repetition_penalty is not None and a.repetition_penalty <= 0:
        raise SystemExit("--repetition-penalty must be > 0")
    if a.no_repeat_ngram_size is not None and a.no_repeat_ngram_size < 0:
        raise SystemExit("--no-repeat-ngram-size must be >= 0")
    if a.soft_max_new_tokens is not None and a.soft_max_new_tokens < 1:
        raise SystemExit("--soft-max-new-tokens must be >= 1")

    specs, resolve_sae = parse_specs(a)

    print(f"loading {a.model} ...", flush=True)
    _load_kw = {"gguf_file": a.gguf_file} if a.gguf_file else {}
    tok = AutoTokenizer.from_pretrained(a.model, **_load_kw)
    model_loader = AutoModelForCausalLM
    if a.model_loader == "image-text":
        model_loader = AutoModelForImageTextToText
    elif a.model_loader == "auto":
        cfg = AutoConfig.from_pretrained(a.model, **_load_kw)
        archs = " ".join(getattr(cfg, "architectures", []) or [])
        if "ConditionalGeneration" in archs:
            model_loader = AutoModelForImageTextToText
    model = model_loader.from_pretrained(a.model, torch_dtype=torch.bfloat16,
                                         device_map="cuda", **_load_kw)
    model.eval()
    dev = next(model.parameters()).device
    d_model = (getattr(model.config, "hidden_size", None)
               or getattr(getattr(model.config, "text_config", None), "hidden_size", None)
               or getattr(getattr(model.config, "language_config", None), "hidden_size", None))
    if d_model is None:
        raise SystemExit("could not infer model hidden_size from config")
    layers = find_layers(model)

    # group clamp specs by layer, load each layer's SAE once, register one hook per layer
    by_layer = {}
    for (L, feat, t) in specs:
        by_layer.setdefault(L, []).append((feat, t))
    sae_find_cache = {}
    all_feats = []
    print("clamps:", flush=True)
    for L in sorted(by_layer):
        path = resolve_sae(L)
        if path not in sae_find_cache:
            if not os.path.exists(path):
                raise SystemExit(f"missing SAE for layer {L}: {path}\n"
                                 f"  download it (e.g. hf download Qwen/SAE-Res-Qwen3.5-35B-A3B-Base-W32K-L0_50 "
                                 f"layer{L}.sae.pt --local-dir {os.path.dirname(path)})")
            print(f"  loading SAE {path}", flush=True)
            sae_find_cache[path] = load_sae(path)[1]
        find = sae_find_cache[path]
        feats = []
        for feat, t in by_layer[L]:
            d_f, e_f, b_f = dirs_for(find, feat, d_model)
            if e_f is None:
                raise SystemExit("no encoder in SAE -- need W_enc to read the activation to clamp")
            fd = {"feat": feat, "layer": L, "e_f": e_f.to(dev), "b_f": b_f, "d_f": d_f.to(dev),
                  "target": t, "act": 0.0}
            feats.append(fd); all_feats.append(fd)
            print(f"    f{feat}@L{L} floor={t}  ||d_f||={d_f.norm():.3f} ||e_f||={e_f.norm():.3f}", flush=True)
        layers[L].register_forward_hook(LayerClamp(L, feats))

    mon = None
    if not a.no_monitor:
        try:
            layers[a.monitor_layer].mlp.gate.register_forward_hook(
                (mon := RouterMonitor(a.monitor_expert, a.monitor_layer)))
            print(f"MONITOR E{a.monitor_expert} @ L{a.monitor_layer} gate (read-only readout each turn)",
                  flush=True)
        except Exception as ex:
            print(f"[E114 monitor disabled: {ex}]", flush=True)
    print(f"temp={a.temperature}  top_p={a.top_p}  top_k={a.top_k if a.top_k is not None else 'default'}  "
          f"reppen={a.repetition_penalty if a.repetition_penalty is not None else 'default'}  "
          f"ngram={a.no_repeat_ngram_size if a.no_repeat_ngram_size is not None else 'default'}  "
          f"soft_max={a.soft_max_new_tokens if a.soft_max_new_tokens is not None else 'off'}  "
          f"hard_max={a.max_new_tokens}  "
          f"thinking={'off/' + a.no_think_style if a.no_think else 'on'}  "
          f"mode={a.mode}  seed=LOCKED@{a.seed} (per-PROMPT reproducible; /reset rolls, /seed off = vary)\n"
          "  commands: /target FEAT N | /target N | /off | /clamp | /e114 [N|on|off] | /temp X | /topk N|off | "
          "/reppen X|off | /ngram N|off | /softmax N|off | /seed N|off | "
          "/think [on|off|show|hide] | /reset | /system ... | /quit\n", flush=True)

    history, system = [], a.system
    temperature = a.temperature
    top_k = a.top_k
    repetition_penalty = a.repetition_penalty
    no_repeat_ngram_size = a.no_repeat_ngram_size
    soft_max_new_tokens = a.soft_max_new_tokens
    torch.manual_seed(a.seed)        # (the locked seed below is re-applied before each generation)
    locked_seed = a.seed             # START LOCKED: fixed seed => reproducible; /reset rolls to the next
    thinking_enabled = not a.no_think
    show_think = a.show_think        # default: strip <think> blocks; /think toggles
    while True:
        try:
            msg = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not msg:
            continue
        if msg == "/quit":
            break
        if msg == "/reset":
            history = []
            if locked_seed is not None:
                locked_seed += 1
                print(f"[history cleared; seed → {locked_seed} (fresh fixed draw)]")
            else:
                print("[history cleared]")
            continue
        if msg == "/clamp":
            print("[clamp] " + clamp_summary(all_feats)); continue
        if msg in ("/off", "/zero"):
            for f in all_feats: f["target"] = 0.0
            print("[all features → 0 (clean model)]"); continue
        if msg.startswith("/e114") or msg.startswith("/monitor"):
            if mon is None:
                print("[no monitor running]"); continue
            parts = msg.split()
            if len(parts) == 2 and parts[1] in ("on", "off"):
                mon.on = (parts[1] == "on"); print(f"[monitor {parts[1]}]")
            elif len(parts) == 2:
                mon.expert = int(parts[1]); print(f"[monitor expert = E{mon.expert}]")
            else:
                print(f"[monitor E{mon.expert} @ L{mon.layer}, on={mon.on}]")
            continue
        if msg.startswith("/target"):
            parts = msg.split()
            if len(parts) == 3:
                fid, val = int(parts[1]), float(parts[2])
                n = set_target(all_feats, fid, val)
                print(f"[f{fid} floor = {val}]" if n else f"[no clamped feature {fid}]")
            elif len(parts) == 2:
                val = float(parts[1])
                for f in all_feats: f["target"] = val
                print(f"[ALL floors = {val}]")
            else:
                print("usage: /target FEAT N   or   /target N")
            continue
        if msg.startswith("/temp"):
            temperature = float(msg.split(maxsplit=1)[1]); print(f"[temperature = {temperature}]"); continue
        if msg.startswith("/topk") or msg.startswith("/top-k"):
            parts = msg.split()
            if len(parts) == 2 and parts[1].lower() in ("off", "none", "default"):
                top_k = None
                print("[top_k = default]")
            elif len(parts) == 2:
                top_k = int(parts[1])
                if top_k < 0:
                    raise ValueError("top_k must be >= 0")
                print(f"[top_k = {top_k}]")
            else:
                print(f"[top_k {top_k if top_k is not None else 'default'}]")
            continue
        if msg.startswith("/reppen") or msg.startswith("/repetition"):
            parts = msg.split()
            if len(parts) == 2 and parts[1].lower() in ("off", "none", "default"):
                repetition_penalty = None
                print("[repetition_penalty = default]")
            elif len(parts) == 2:
                repetition_penalty = float(parts[1])
                if repetition_penalty <= 0:
                    raise ValueError("repetition_penalty must be > 0")
                print(f"[repetition_penalty = {repetition_penalty}]")
            else:
                print(f"[repetition_penalty {repetition_penalty if repetition_penalty is not None else 'default'}]")
            continue
        if msg.startswith("/ngram") or msg.startswith("/no-repeat-ngram"):
            parts = msg.split()
            if len(parts) == 2 and parts[1].lower() in ("off", "none", "default", "0"):
                no_repeat_ngram_size = None
                print("[no_repeat_ngram_size = default]")
            elif len(parts) == 2:
                no_repeat_ngram_size = int(parts[1])
                if no_repeat_ngram_size < 0:
                    raise ValueError("no_repeat_ngram_size must be >= 0")
                print(f"[no_repeat_ngram_size = {no_repeat_ngram_size}]")
            else:
                print(f"[no_repeat_ngram_size {no_repeat_ngram_size if no_repeat_ngram_size is not None else 'default'}]")
            continue
        if msg.startswith("/softmax") or msg.startswith("/soft-max"):
            parts = msg.split()
            if len(parts) == 2 and parts[1].lower() in ("off", "none", "default", "0"):
                soft_max_new_tokens = None
                print("[soft_max_new_tokens = off]")
            elif len(parts) == 2:
                soft_max_new_tokens = int(parts[1])
                if soft_max_new_tokens < 1:
                    raise ValueError("soft_max_new_tokens must be >= 1")
                print(f"[soft_max_new_tokens = {soft_max_new_tokens}; hard cap = {a.max_new_tokens}]")
            else:
                print(f"[soft_max_new_tokens {soft_max_new_tokens if soft_max_new_tokens is not None else 'off'}; hard cap = {a.max_new_tokens}]")
            continue
        if msg.startswith("/seed"):
            parts = msg.split()
            if len(parts) == 2 and parts[1].lower() in ("off", "free", "none"):
                locked_seed = None; print("[seed FREE — responses vary each turn]")
            elif len(parts) == 2:
                locked_seed = int(parts[1])
                print(f"[seed LOCKED to {locked_seed} — same prompt+config now reproduces exactly]")
            else:
                print(f"[seed {('locked=' + str(locked_seed)) if locked_seed is not None else 'free (advancing)'}]")
            continue
        if msg.startswith("/think"):
            parts = msg.split()
            if len(parts) == 2 and parts[1].lower() == "on":
                thinking_enabled = True
                print(f"[thinking enabled; think blocks {'shown' if show_think else 'stripped'}]")
            elif len(parts) == 2 and parts[1].lower() == "off":
                thinking_enabled = False
                show_think = False
                print(f"[thinking disabled via {a.no_think_style}; think blocks stripped]")
            elif len(parts) == 2 and parts[1].lower() in ("show", "shown"):
                show_think = True
                print(f"[think blocks shown; thinking {'enabled' if thinking_enabled else 'disabled'}]")
            elif len(parts) == 2 and parts[1].lower() in ("hide", "strip", "stripped"):
                show_think = False
                print(f"[think blocks stripped; thinking {'enabled' if thinking_enabled else 'disabled'}]")
            else:
                print(f"[thinking {'enabled' if thinking_enabled else 'disabled'}; "
                      f"think blocks {'shown' if show_think else 'stripped'}; "
                      f"no-think style={a.no_think_style}]")
            continue
        if msg.startswith("/system"):
            system = msg[len("/system"):].strip(); print("[system set]"); continue

        ids = build_inputs(tok, a.mode, system, history, msg, dev, no_think=not thinking_enabled,
                           no_think_style=a.no_think_style)
        if temperature and temperature > 0:
            if locked_seed is not None:                    # per-prompt seed: reproducible yet prompt-sensitive
                torch.manual_seed(derive_seed(locked_seed, msg, all_feats, temperature, a.top_p, top_k,
                                              repetition_penalty, no_repeat_ngram_size))
            gkw = dict(do_sample=True, temperature=temperature, top_p=a.top_p)
            if top_k is not None:
                gkw["top_k"] = top_k
        else:
            gkw = dict(do_sample=False, temperature=None, top_p=None, top_k=None)
        if mon:
            mon.reset()
        stop_strings = BASE_STOP_STRINGS if a.mode == "base" else ()
        gen_kwargs = dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                          max_new_tokens=a.max_new_tokens,
                          pad_token_id=tok.eos_token_id, **gkw)
        if repetition_penalty is not None:
            gen_kwargs["repetition_penalty"] = repetition_penalty
        if no_repeat_ngram_size is not None:
            gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
        stopping_criteria = []
        if stop_strings:
            stopping_criteria.append(StopOnDecodedStrings(tok, ids.shape[1], stop_strings))
        if soft_max_new_tokens is not None:
            stopping_criteria.append(SoftStopAfterBoundary(tok, ids.shape[1], soft_max_new_tokens))
        if stopping_criteria:
            gen_kwargs["stopping_criteria"] = StoppingCriteriaList(stopping_criteria)
        if a.plain_output and a.stream_output:
            gen, reply = stream_generate(model, tok, gen_kwargs, a.reply_prefix, stop_strings)
        else:
            with torch.no_grad():
                gen = model.generate(**gen_kwargs)
            reply = tok.decode(gen[0][ids.shape[1]:], skip_special_tokens=True)
        if a.mode == "base":
            for stop in ("\nUser:", "\nuser:", "\nYou:"):
                i = reply.find(stop)
                if i != -1:
                    reply = reply[:i]
        if not show_think:                                  # strip <think> meta-narration
            stripped = strip_think(reply)
            reply = stripped if stripped else re.sub(r"</?think>", "", reply, flags=re.IGNORECASE)
        reply = reply.strip()
        if a.plain_output and not a.stream_output:
            print(f"\n{a.reply_prefix}{reply}", flush=True)
        elif not a.plain_output:
            print(f"\nbot[{clamp_summary(all_feats)}]> {reply}", flush=True)
        if mon and not (a.plain_output or a.no_auto_monitor_report):
            line = mon.report(gen.shape[1] - ids.shape[1])
            if line:
                print(line, flush=True)
        history.append((msg, reply))


if __name__ == "__main__":
    main()
