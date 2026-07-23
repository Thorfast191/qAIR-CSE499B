import os
import time
import torch

from tqdm.auto import tqdm
from torch.utils.data import Dataset

from config import EMBEDDING_DIM as DIM, EMBEDDING_MODEL, GEN_BATCH_SIZE

from benchmarks.arc import load_arc

from models.generator import HypothesisGenerator, LLM_NAME
from models.encoder import HypothesisEncoder


class QAIRDataset(Dataset):

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

            loaded = torch.load(cache_path, map_location="cpu")

            if isinstance(loaded, dict):
                self.samples = loaded.get("samples", [])
                is_complete = loaded.get("complete", False)
                resume_raw_index = loaded.get("raw_index", 0)
            else:
                # Old-format cache (plain list). We have no way of knowing
                # whether it was finished, so treat it as complete to avoid
                # silently re-triggering a full rebuild of a cache someone
                # already relied on. New caches will always carry the flag.
                self.samples = loaded
                is_complete = True
                resume_raw_index = 0

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
                  f"(complete={is_complete})")

            if max_samples is not None:
                # Caller explicitly wants a capped subset. If we already
                # have enough, or more than enough, that's fine either way.
                self.samples = self.samples[:max_samples]
                print(f"[CACHE TRUNCATED TO max_samples] {len(self.samples)} samples")
                return

            if is_complete:
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

            if split == "validation" and "test" in ds:
                split = "test"

            elif "validation" in ds:
                split = "validation"

            else:
                split = list(ds.keys())[0]

        raw = ds[split]

        count = len(self.samples)

        last_raw_index = resume_raw_index

        # Examples that passed filtering but haven't been generated/encoded
        # yet: (raw_idx, stem, options, y). Generation and encoding run in
        # batches of GEN_BATCH_SIZE instead of one example at a time -- an
        # LLM decoding one prompt per call wastes almost all of a GPU's
        # throughput compared to decoding a padded batch of prompts at once.
        pending = []

        def flush_pending():

            nonlocal count, last_raw_index

            if not pending:
                return

            questions = [item[1] for item in pending]
            options_list = [item[2] for item in pending]

            try:
                hyps_list = generator.generate_batch(questions, options_list)
            except Exception as e:
                print(f"[BATCH GENERATE FAILED] {e} -- falling back per-item")
                hyps_list = [
                    generator.fallback(q, o)
                    for q, o in zip(questions, options_list)
                ]

            all_hyp_texts = [h for hyps in hyps_list for h in hyps]
            all_opt_texts = [o for options in options_list for o in options]

            H_all = encoder.encode(all_hyp_texts)
            O_all = encoder.encode(all_opt_texts)

            h_off = 0
            o_off = 0

            for (raw_idx, stem, options, y), hyps in zip(pending, hyps_list):

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

                    "option_index": list(range(len(options))),

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

                    cache = {

                        "samples": self.samples,

                        "raw_index": last_raw_index,

                        "complete": False,

                        "encoder": EMBEDDING_MODEL,

                        "generator": LLM_NAME,

                        "version": 32,
                    }

                    torch.save(cache, cache_path)

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

        cache = {

            "samples": self.samples,

            "raw_index": last_raw_index,

            "complete": finished_full_split,

            "encoder": EMBEDDING_MODEL,

            "generator": LLM_NAME,

            "version": 32,
        }

        torch.save(cache, cache_path)

        elapsed = (time.time() - start_time) / 60

        print(f"[CACHE SAVED] {cache_path} (complete={finished_full_split})")

        print(f"[TOTAL SAMPLES] {len(self.samples)}")

        print(f"[BUILD TIME] {elapsed:.2f} min")

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
    """

    max_h = max(x["H"].shape[0] for x in batch)

    max_o = max(x["O"].shape[0] for x in batch)

    dim = batch[0]["H"].shape[-1]

    Hs = []
    Os = []
    ys = []
    H_masks = []
    O_masks = []

    for sample in batch:

        H = sample["H"]
        O = sample["O"]
        y = sample["y"]

        if shuffle_options:

            n = O.shape[0]

            perm = torch.randperm(n)

            O = O[perm]

            if H.shape[0] == n:
                H = H[perm]

            y = int((perm == y).nonzero(as_tuple=True)[0].item())

        h_len = H.shape[0]
        o_len = O.shape[0]

        H_mask = H.new_zeros(max_h, dtype=torch.bool)

        O_mask = O.new_zeros(max_o, dtype=torch.bool)

        H_mask[:h_len] = True
        O_mask[:o_len] = True

        if H.shape[0] < max_h:

            pad = H.new_zeros(max_h - h_len, dim)

            H = torch.cat([H, pad], dim=0)

        if O.shape[0] < max_o:

            pad = O.new_zeros(max_o - o_len, dim)

            O = torch.cat([O, pad], dim=0)

        Hs.append(H)
        Os.append(O)

        ys.append(y)
        H_masks.append(H_mask)
        O_masks.append(O_mask)

    return {
        "H": torch.stack(Hs),
        "O": torch.stack(Os),
        "H_mask": torch.stack(H_masks),
        "O_mask": torch.stack(O_masks),
        "y": torch.tensor(ys, dtype=torch.long),
    }