import os
import time
import torch

from tqdm.auto import tqdm
from torch.utils.data import Dataset

from config import (
    EMBEDDING_DIM as DIM,
    EMBEDDING_MODEL,
    GEN_BATCH_SIZE,
    CACHE_VERSION,
)

from benchmarks.arc import load_arc

from models.generator import HypothesisGenerator, LLM_NAME
from models.encoder import HypothesisEncoder


class QAIRDataset(Dataset):
    """
    Cached (question, hypotheses, options) embeddings for one ARC split.

    Each sample carries:
        Q  (D,)    question embedding      -- added in cache v33
        H  (K, D)  hypothesis embeddings
        O  (N, D)  option embeddings
        y  int     index of the correct option
        is_fallback (K,) bool              -- added in cache v33

    The question embedding is the single highest-impact field here. It
    was absent for the project's whole history even though the question
    TEXT was already cached: measured on these exact embeddings, a plain
    MLP scores 0.3585 from options alone and 0.4849 once the question is
    available. Caches built before v33 are migrated in place on load --
    that only needs the encoder, not the LLM, so it takes a minute
    rather than hours.
    """

    def __init__(self, split="train", max_samples=None, cache_dir="./cache"):

        self.samples = []

        os.makedirs(cache_dir, exist_ok=True)

        cache_path = os.path.join(cache_dir, f"arc_{split}.pt")

        resume_raw_index = 0

        # ====================================================
        # LOAD CACHE (possibly partial)
        # ====================================================

        if os.path.exists(cache_path):

            print(f"[CACHE FOUND] {cache_path}")

            try:
                loaded = torch.load(cache_path, map_location="cpu")

            except Exception as e:
                # Almost always a build that was interrupted mid-write by
                # an older version of _save() (which wrote in place rather
                # than atomically). The partial file is unreadable and so
                # cannot be resumed from -- move it aside rather than
                # delete it, and rebuild, instead of dying with a raw
                # miniz stack trace.
                salvage = cache_path + ".corrupt"

                print(
                    f"\n[CACHE UNREADABLE] {cache_path}\n"
                    f"  {type(e).__name__}: {e}\n"
                    f"  This is what a cache build interrupted mid-write "
                    f"looks like (truncated zip archive).\n"
                    f"  Moving it to {salvage} and rebuilding from scratch.\n"
                    f"  Saves are atomic now, so this cannot recur.\n"
                )

                if os.path.exists(salvage):
                    os.remove(salvage)

                os.replace(cache_path, salvage)

                loaded = None

            if loaded is None:
                self.samples = []
                is_complete = False
                resume_raw_index = 0
                version = CACHE_VERSION

            elif isinstance(loaded, dict):
                self.samples = loaded.get("samples", [])
                is_complete = loaded.get("complete", False)
                resume_raw_index = loaded.get("raw_index", 0)
                version = loaded.get("version", 0)
            else:
                # Old-format cache (plain list). We have no way of knowing
                # whether it was finished, so treat it as complete to avoid
                # silently re-triggering a full rebuild of a cache someone
                # already relied on. New caches will always carry the flag.
                self.samples = loaded
                is_complete = True
                resume_raw_index = 0
                version = 0

            # Refuse to silently mix embeddings built with a different
            # encoder/dim than the one config.py currently points at. This
            # is the exact "mat1 and mat2 shapes cannot be multiplied"
            # failure this project already hit once (384 -> 768); without
            # this guard, moving the dim back down (768 -> 384) would
            # silently load the old cache and hit it again, just deeper
            # into training instead of at dataset construction.
            if self.samples:

                cached_dim = self.samples[0]["H"].shape[-1]

                if cached_dim != DIM:

                    raise RuntimeError(
                        f"{cache_path} was built with {cached_dim}-dim "
                        f"embeddings, but config.EMBEDDING_DIM is now {DIM} "
                        f"(config.EMBEDDING_MODEL={EMBEDDING_MODEL}). Move, "
                        f"rename, or delete {cache_path} and rebuild before "
                        f"training -- refusing to silently mix incompatible "
                        f"cached embeddings."
                    )

            print(f"[CACHE LOADED] {len(self.samples)} samples "
                  f"(complete={is_complete}, version={version})")

            if self.samples and is_complete:
                self._migrate(cache_path, loaded if isinstance(loaded, dict) else {})

            # Only the paths below may return early. A cache that failed to
            # load leaves samples empty, and returning here would hand the
            # caller a silently EMPTY dataset instead of rebuilding.
            if self.samples:

                if max_samples is not None and len(self.samples) >= max_samples:
                    # Caller wants a capped subset and we already have enough.
                    self.samples = self.samples[:max_samples]
                    print(f"[CACHE TRUNCATED TO max_samples] "
                          f"{len(self.samples)} samples")
                    return

                if is_complete and max_samples is None:
                    self.report_quality(split)
                    return

            print(
                f"[CACHE INCOMPLETE] Resuming build from raw index "
                f"{resume_raw_index} ({len(self.samples)} samples so far)..."
            )

        else:

            print(f"[CACHE MISSING] Building {split} cache from scratch...")

        # ====================================================
        # BUILD (or RESUME) CACHE
        # ====================================================

        start_time = time.time()

        generator = HypothesisGenerator()

        encoder = HypothesisEncoder()

        ds = load_arc()

        if split not in ds:

            if "validation" in ds:
                split = "validation"

            else:
                split = list(ds.keys())[0]

        raw = ds[split]

        count = len(self.samples)

        last_raw_index = resume_raw_index

        # Examples that passed filtering but haven't been generated/encoded
        # yet: (raw_idx, stem, options, y).
        pending = []

        def flush_pending():

            nonlocal count, last_raw_index

            if not pending:
                return

            questions = [item[1] for item in pending]
            options_list = [item[2] for item in pending]

            try:
                batch_out = generator.generate_batch_with_flags(
                    questions, options_list
                )
            except Exception as e:
                print(f"[BATCH GENERATE FAILED] {e} -- falling back per-item")
                batch_out = [
                    ([generator.fallback(q, o) for o in options],
                     [True] * len(options))
                    for q, options in zip(questions, options_list)
                ]

            all_hyp_texts = [h for hyps, _ in batch_out for h in hyps]
            all_opt_texts = [o for options in options_list for o in options]

            H_all = encoder.encode(all_hyp_texts)
            O_all = encoder.encode(all_opt_texts)
            Q_all = encoder.encode(questions)

            h_off = 0
            o_off = 0

            for qi, ((raw_idx, stem, options, y), (hyps, flags)) in enumerate(
                zip(pending, batch_out)
            ):

                h_len = len(hyps)
                o_len = len(options)

                H = H_all[h_off:h_off + h_len]
                O = O_all[o_off:o_off + o_len]

                h_off += h_len
                o_off += o_len

                sample = {

                    "question": stem,

                    "options": options,

                    "hypotheses": hyps,

                    "is_fallback": torch.tensor(flags, dtype=torch.bool),

                    "option_index": list(range(len(options))),

                    "Q": Q_all[qi].cpu(),

                    "H": H.cpu(),

                    "O": O.cpu(),

                    "y": y,
                }

                self.samples.append(sample)

                count += 1

                last_raw_index = raw_idx + 1

                # ========================================
                # PERIODIC SAVE (marked incomplete)
                # ========================================

                if count % 50 == 0:

                    self._save(cache_path, last_raw_index, complete=False)

                    print(f"[AUTOSAVE] {count} samples "
                          f"(raw_index={last_raw_index})")

            pending.clear()

        for raw_idx, ex in enumerate(
            tqdm(raw, desc=f"Building {split} cache", initial=resume_raw_index)
        ):

            if raw_idx < resume_raw_index:
                continue

            if max_samples is not None and count >= max_samples:
                break

            try:

                question = ex["question"]

                # ============================================
                # ARC FORMAT
                # ============================================

                if isinstance(question, dict):

                    stem = question["stem"]

                    choices = question["choices"]

                    options = choices["text"]

                    labels = choices["label"]

                else:

                    stem = ex["question"]

                    choices = ex["choices"]

                    options = choices["text"]

                    labels = choices["label"]

                # ============================================
                # FILTERS
                # ============================================

                if len(options) < 2:
                    # Resolve anything already queued before this raw_idx
                    # advances past it, so last_raw_index never jumps ahead
                    # of samples that haven't actually been saved yet.
                    flush_pending()
                    last_raw_index = raw_idx + 1
                    continue

                answer = ex["answerKey"]

                if answer not in labels:
                    flush_pending()
                    last_raw_index = raw_idx + 1
                    continue

                y = labels.index(answer)

                # ============================================
                # QUEUE FOR BATCHED GENERATION + ENCODING
                # ============================================

                pending.append((raw_idx, stem, options, y))

                hit_chunk_size = len(pending) >= GEN_BATCH_SIZE

                hit_max_samples = (
                    max_samples is not None
                    and (count + len(pending)) >= max_samples
                )

                if hit_chunk_size or hit_max_samples:
                    flush_pending()

                if hit_max_samples:
                    break

            except Exception as e:

                print(f"[SKIP] {e}")

                flush_pending()

                last_raw_index = raw_idx + 1

                continue

        flush_pending()

        # ====================================================
        # FINAL SAVE (marked complete)
        # ====================================================

        finished_full_split = (max_samples is None)

        self._save(cache_path, last_raw_index, complete=finished_full_split)

        elapsed = (time.time() - start_time) / 60

        print(f"[CACHE SAVED] {cache_path} (complete={finished_full_split})")

        print(f"[TOTAL SAMPLES] {len(self.samples)}")

        print(f"[BUILD TIME] {elapsed:.2f} min")

        self.report_quality(split)

    # ========================================================
    # SAVE / MIGRATE / REPORT
    # ========================================================

    def _save(self, cache_path, raw_index, complete):
        """
        Atomic: write to a temp file in the same directory, then rename.

        The cache is autosaved every 50 samples during a build that takes
        hours, so being interrupted mid-write is a foreseeable state, not
        an exotic one. Writing in place meant a Ctrl-C (or a crash, or a
        reboot) landing inside torch.save left a TRUNCATED zip archive --
        which torch.load reports as an opaque
        "PytorchStreamReader failed reading zip archive: failed finding
        central directory" miniz error, destroying hours of generation
        with no indication of what went wrong. os.replace is atomic on
        both POSIX and Windows, so the real path only ever points at a
        complete file.
        """

        tmp_path = cache_path + ".tmp"

        torch.save(
            {
                "samples": self.samples,
                "raw_index": raw_index,
                "complete": complete,
                "encoder": EMBEDDING_MODEL,
                "generator": LLM_NAME,
                "version": CACHE_VERSION,
            },
            tmp_path,
        )

        os.replace(tmp_path, cache_path)

    def _migrate(self, cache_path, meta):
        """
        Bring a pre-v33 cache forward in place.

        Only the question embedding can be recovered without the LLM, and
        it is by far the most valuable field, so this runs automatically.
        Hypothesis quality CANNOT be repaired by migration -- that needs
        regeneration. report_quality() will tell you if you need it.
        """

        needs_q = "Q" not in self.samples[0]
        needs_flags = "is_fallback" not in self.samples[0]

        if not (needs_q or needs_flags):
            return

        if needs_q:

            print("[CACHE MIGRATE] Adding question embeddings (encoder only, "
                  "no LLM needed)...")

            encoder = HypothesisEncoder()

            texts = [s["question"] for s in self.samples]

            embeddings = []

            for i in tqdm(range(0, len(texts), 128), desc="Encoding questions"):
                embeddings.append(encoder.encode(texts[i:i + 128]).cpu())

            embeddings = torch.cat(embeddings)

            for s, q in zip(self.samples, embeddings):
                s["Q"] = q

        if needs_flags:

            # Detect the old template fallback so downstream code can mask
            # these even on a legacy cache.
            import re

            pattern = re.compile(r"correctly explains the .* in the question\.$")

            for s in self.samples:
                s["is_fallback"] = torch.tensor(
                    [bool(pattern.search(h.strip())) for h in s["hypotheses"]],
                    dtype=torch.bool,
                )

        self._save(cache_path, meta.get("raw_index", 0), complete=True)

        print(f"[CACHE MIGRATED] {cache_path} -> version {CACHE_VERSION}")

    def report_quality(self, split):
        """
        Print the hypothesis-quality numbers that the audit had to
        reverse-engineer from the cache by hand. If the fallback rate is
        high, the hypotheses carry little signal and no amount of
        architecture will help -- regenerate instead.
        """

        if not self.samples or "is_fallback" not in self.samples[0]:
            return

        flags = torch.cat([s["is_fallback"] for s in self.samples])

        total = flags.numel()
        n_fb = int(flags.sum())

        per_sample = torch.tensor(
            [int(s["is_fallback"].all()) for s in self.samples]
        )

        print(
            f"[CACHE QUALITY {split}] fallback hypotheses "
            f"{n_fb}/{total} = {100 * n_fb / max(total, 1):.1f}%   "
            f"fully-fallback samples {int(per_sample.sum())}/{len(self.samples)}"
        )

        if n_fb / max(total, 1) > 0.10:
            print(
                "[CACHE QUALITY WARNING] >10% of hypotheses are fallbacks. "
                "These carry no information about which option is correct. "
                "Delete this cache and rebuild with the current generator."
            )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        return self.samples[idx]


def collate_fn(batch, shuffle_options=True):
    """
    shuffle_options: randomly permutes each sample's option (and matching
    hypothesis) order so the model can't learn positional shortcuts from
    the small dataset. Pass shuffle_options=False for validation/eval so
    metrics are stable and reproducible.

    Ragged option/hypothesis counts (ARC has a handful of 3- and
    5-option questions) are zero-padded, and the masks below say which
    entries are real. Those masks used to be computed here and then
    never passed to the model, so a zero-padded option participated in
    every reduction and could be returned as the prediction.
    """

    max_h = max(x["H"].shape[0] for x in batch)

    max_o = max(x["O"].shape[0] for x in batch)

    dim = batch[0]["H"].shape[-1]

    Hs = []
    Os = []
    Qs = []
    ys = []
    H_masks = []
    O_masks = []
    fallbacks = []

    for sample in batch:

        H = sample["H"]
        O = sample["O"]
        y = sample["y"]

        fb = sample.get(
            "is_fallback", torch.zeros(H.shape[0], dtype=torch.bool)
        )

        if shuffle_options:

            n = O.shape[0]

            perm = torch.randperm(n)

            O = O[perm]

            if H.shape[0] == n:
                H = H[perm]
                fb = fb[perm]

            y = int((perm == y).nonzero(as_tuple=True)[0].item())

        h_len = H.shape[0]
        o_len = O.shape[0]

        H_mask = torch.zeros(max_h, dtype=torch.bool)

        O_mask = torch.zeros(max_o, dtype=torch.bool)

        H_mask[:h_len] = True
        O_mask[:o_len] = True

        fb_pad = torch.zeros(max_h, dtype=torch.bool)
        fb_pad[:h_len] = fb

        if H.shape[0] < max_h:

            pad = H.new_zeros(max_h - h_len, dim)

            H = torch.cat([H, pad], dim=0)

        if O.shape[0] < max_o:

            pad = O.new_zeros(max_o - o_len, dim)

            O = torch.cat([O, pad], dim=0)

        Hs.append(H)
        Os.append(O)
        Qs.append(sample["Q"] if "Q" in sample else torch.zeros(dim))

        ys.append(y)
        H_masks.append(H_mask)
        O_masks.append(O_mask)
        fallbacks.append(fb_pad)

    return {
        "H": torch.stack(Hs),
        "O": torch.stack(Os),
        "Q": torch.stack(Qs),
        "H_mask": torch.stack(H_masks),
        "O_mask": torch.stack(O_masks),
        "is_fallback": torch.stack(fallbacks),
        "y": torch.tensor(ys, dtype=torch.long),
    }
