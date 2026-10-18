# Video script (about 4 minutes 40 seconds)

Spoken text is in quotes; **on screen** says what to show. About 690 spoken words, at a calm pace of about 150 words a
minute. Every number comes from `runs/final/report.md`, `runs/bench/summary.md` or `PLAN.md`. The slides that go with it
are the deck "Lightning & the Specialists" (claude.ai): each slide's speaker notes hold its lines, and the lines in
[brackets] there are for questions, not for reading out.

Before recording: start the page (`uv run --group demo python -m demo.app`, tunnel to `localhost:7860`), and run
`scripts/demo_reset.sh`, which must end with "demo_reset: ready". Present live. If a server misbehaves, switch Models to
"replay": the recorded takes come back exactly.

## 1. Team (0:00 to 0:25)

**On screen:** the cover slide, with you on camera.

"Hi, I'm Sachin Kumar, a Senior Data Scientist III at LexisNexis, and this is Lightning & the Specialists, a solo entry
in the Multi-Agent Systems track. I built a small-model IT operations team: a planner and small specialist models that
handle helpdesk tickets and production incidents together, on one GPU."

## 2. Elevator pitch (0:25 to 1:00)

**On screen:** the pitch slide.

"Here's the idea. NVIDIA Nemotron 3.5 Lightning, a 30-billion-parameter model, does two jobs: it teaches, and it plans.
As a teacher, its verified solutions to thousands of IT tasks trained small 4-billion-parameter specialists, one per
API. As a planner, Lightning splits each request into steps and hands each step to a specialist. A safety layer, which
we call the seatbelt, checks every tool call. On held-out cases, the trained helpdesk specialist went from 5 to 8 of 10
cases, beating Lightning itself, which got 6."

## 3. Live demo (1:00 to 2:00)

**On screen:** the "Live demo" slide for a moment, then the page's Live tab. Keep the demo request. Tick Helpdesk and
Ops, chaos off, Models: live. Click Run.

"The request: 'The payments app keeps crashing after this morning's deploy: roll it back and open a P1 ticket.'"

**On screen:** the team's column, top to bottom.

"Lightning's plan opens the ticket first, though the user asked for it second, because a production change needs a
change ticket. The helpdesk specialist opens HD-1042. Its ID passes to the ops specialist, which reads the rollout
history and rolls payments back from revision 9 to 8 with that ticket. The pods go from crash-looping to Running, the
verifier passes the run, and the counter shows every step was handled by a small model."

**On screen:** the right column, single_base.

"On the right, one small model with all twelve tools on the same request. It tries the rollback before any ticket
exists, the change-ticket rule refuses it, and the verifier fails the run." (Describe what it does live; in all three
recorded takes it went for the rollback first.)

**On screen:** turn chaos on (0.3, seed 5) and run again.

"Now the tools fail at random with 503 errors. The seatbelt retries automatically, and the team still gets it right."

## 4. How it's built (2:00 to 4:05)

**On screen:** back to the deck, the data slide ("Where every dataset comes from").

"Where the data comes from: no outside dataset trains or scores anything. The environment's pod logs come from
Loghub's open log samples. For training tasks, code picks each action and its gold calls, and Lightning writes the
request. The practice cases are hand-written, and a separate AI agent wrote the locked test set."

**On screen:** the factory slide.

"Lightning attempted 2,347 of those tasks in NeMo Gym. Only clean passes, checked by the verifier, became training
data, and we removed 257 that were too close to a test case: 1,844 examples, one LoRA adapter per API."

**On screen:** the team slide.

"NeMo Gym is the backbone. It hosts the environment: helpdesk and ops tools with isolated state, a rule that every
production change needs an open ticket, and a verifier that compares final states, never an LLM judge. Every setup,
from one small model to the full team, is scored through the same Gym config, with three attempts per case."

**On screen:** the hardware slide.

"Everything runs on one H200. vLLM serves Lightning, and a second server holds the small base model with every
specialist as a 17-million-parameter LoRA adapter. LoRA fine-tuning runs on the same GPU next to both servers, which
took the GPU to about 90% of its memory."

**On screen:** the trade-offs slide.

"A few things we learned the hard way. Without guided decoding, only 8 of Lightning's 24 plans fit our schema; with
vLLM's structured outputs, all 24 do. The seatbelt's self-check had to ask 'is a tool call still needed?', not 'is the
task done?', or it threw away correct answers that asked the user a question. Serving all specialists on one server
holds almost four times the cache of separate servers in the same memory, though merged models were 18% faster on our
traffic, a real trade-off." (If you run long, drop the self-check sentence.)

**On screen:** the honest-failure slide, or the helpdesk-only take on the page (untick Ops, Models: replay).

"One honest failure: with ops switched off, Lightning assumed an ops specialist existed anyway, and the helpdesk
specialist wrote a note claiming a rollback that never happened. Only the verifier's state check caught it."

## 5. So what (4:05 to 4:40)

**On screen:** the specialists' results slide.

"So what? On 35 held-out cases, written by a separate AI agent that never saw our results, training lifted both
specialists, and the small helpdesk specialist beat the 30B model."

**On screen:** the system slide.

"The full team tied one trained model with every tool, 8 of 15 each, and Lightning alone still did best."

**On screen:** the closing slide.

"We report that as it is, with the environment, the data and the code open for anyone to rerun: small, verified
specialists you can train and serve on a single GPU, with NeMo Gym measuring every claim. Thanks for watching."
