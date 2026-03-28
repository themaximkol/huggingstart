import csv
import time
import traceback
from pathlib import Path
from src.pipeline.evaluator import Evaluator
from src.pipeline.chat_templates import get_persona_names


def run_experiment(
        prompts: list[str],
        evaluator: Evaluator,
        personas: list[str] = None,
        n_completions: int = 3,
        output_path: str = "data/results/experiment.csv",
        resume: bool = True,
) -> list[dict]:
    # Runs every prompt through every persona, scores all completions,
    # saves results row-by-row to CSV

    if personas is None:
        personas = get_persona_names()

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Load already-completed pairs if resuming
    completed = set()
    if resume and output_file.exists():
        with open(output_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed.add((row["prompt"], row["persona"]))
        print(f"Resuming — {len(completed)} pairs already done")

    # Open CSV for appending
    fieldnames = [
        "prompt", "persona",
        "mean_toxicity", "max_toxicity",
        "n_completions", "timestamp"
    ]
    file_exists = output_file.exists()
    results = []

    with open(output_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        total = len(prompts) * len(personas)
        done = 0

        for prompt in prompts:
            for persona in personas:
                done += 1

                # Skip if already done
                if (prompt, persona) in completed:
                    print(f"[{done}/{total}] Skipping: {prompt[:40]} | {persona}")
                    continue

                try:
                    start = time.time()
                    result = evaluator.evaluate(
                        prompt,
                        persona_name=persona,
                        n=n_completions,
                    )
                    elapsed = time.time() - start

                    row = {
                        "prompt": prompt,
                        "persona": persona,
                        "mean_toxicity": round(result["mean_toxicity"], 4),
                        "max_toxicity": round(result["max_toxicity"], 4),
                        "n_completions": n_completions,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }

                    writer.writerow(row)
                    f.flush()  # write to disk immediately, don't buffer
                    results.append(row)

                    print(
                        f"[{done}/{total}] {persona:12} | "
                        f"mean_tox={row['mean_toxicity']:.3f} | "
                        f"{elapsed:.1f}s | {prompt[:40]}"
                    )

                except Exception as e:
                    # Log the error but keep going
                    # One bad prompt should never kill a 300-prompt run
                    print(f"[{done}/{total}] ERROR on '{prompt[:30]}' | {persona}")
                    print(f"  {type(e).__name__}: {e}")
                    traceback.print_exc()
                    continue

    print(f"\nDone. Results saved to {output_path}")
    return results


def load_results(path: str = "data/results/experiment.csv"):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))
