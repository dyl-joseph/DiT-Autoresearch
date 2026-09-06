# DiT AutoResearch Agent Instructions

When the user starts a DiT inference autoresearch session:

1. Read `SKILL.md` and `program.md` completely.
2. Follow the immutable benchmark/evaluator rules.
3. Run the unmodified baseline first.
4. Use `dit-ar search` over `knowledge/handbook/` before non-trivial experiments.
5. Modify only configured mutable paths.
6. Commit one hypothesis at a time.
7. Run `dit-ar experiment` for every candidate.
8. Keep only candidates that pass the original-baseline quality gate and improve the active performance objective.
9. Quantization is never kept if any required benchmark accuracy/quality score drops.
10. Continue the experiment loop until the user stops it.
