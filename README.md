# CSE515 Project: WiFi Sensing with Channel State Information

Notes for me, written so that I can pick this up again after a gap and remember what we are doing and why.

**Team:** Saarthak Gupta (22134024), Shreya Sangma (22134025), Vaibhav Jain (22164020)
**Course:** CSE515, Advanced Communications Networks and Systems, Dr. Ajay Pratap

---

## 1. What the project is, in plain language

Every WiFi device constantly measures how the radio signal changes on its way from the transmitter to the receiver. That measurement is called Channel State Information, and everyone shortens it to CSI. It is not a special sensor reading. It is ordinary information that the WiFi chip already computes so that it can decode packets correctly.

The interesting part is what happens when a person moves through the room. Their body reflects some of the radio waves and blocks others, so the CSI values change. More importantly, they change in a pattern that depends on what the person did. Walking produces a different pattern from sitting down, and sitting down produces a different pattern from falling.

If you record enough of those patterns and label them, you can train a model to recognise them. At that point an ordinary WiFi link has become a sensor. Nobody has to wear a device, no camera is involved, it works in complete darkness, and it works through walls.

Our project takes real CSI recordings that other researchers have already collected and released, trains deep learning models on them, and then studies how well those models actually work. The important word there is "actually", and section 5 explains why.

## 2. Why we picked this topic

The course slides set the constraints, and this topic satisfies all of them.

Slide 8 lists "Wireless Based Indoor Localization Technologies" as an approved topic, so we do not have to argue that it is in scope. Slide 10 says the work must come from the last one to three years and must not be textbook material, and the IEEE published the 802.11bf standard for WiFi sensing in September 2025, which is about as recent as a standard can be. Slide 10 also asks for research papers, industry standards, and commercial products together, and this topic has all three, because companies including Cognitive Systems, Origin AI, and Linksys already sell WiFi sensing features.

On the practical side, the project is entirely software. It needs Python, PyTorch, and NumPy, and nothing else. There is no Arduino, no soldering, and no lab time. The datasets are public and are already downloaded into this folder, so there is no data collection work either. There is also a solid open source baseline to start from, which means we get our first working result cheaply.

## 3. What data we have

All four datasets are already downloaded, extracted, and checked. They live in `data/Data/` and take about 29 GB in total, of which 8.5 GB is the original zip files that can be deleted once we are confident.

| Dataset | What it contains | Shape of one sample | Classes | Train / Test |
|---|---|---|---|---|
| UT-HAR | Seven everyday activities performed by six people. The first CSI dataset released publicly. | 1 x 250 x 90 | 7 | 3977 / 996 |
| NTU-Fi HAR | Six activities recorded at much finer resolution, 114 subcarriers per antenna pair. | 3 x 114 x 500 | 6 | 936 / 264 |
| NTU-Fi HumanID | Fourteen people walking, used to identify who is walking rather than what they are doing. | 3 x 114 x 500 | 14 | 294 / 546 |
| Widar3.0 | Twenty two hand gestures, recorded across every combination of 17 people, 8 standing positions, and 5 body orientations. | 22 x 20 x 20 | 22 | 34926 / 8726 |

A few things are worth understanding about this data.

The activity classes are ordinary things like lying down, falling, walking, picking up, running, sitting down, and standing up. The Widar3.0 gestures are things like push and pull, sweep, clap, slide, and drawing letters and digits in the air.

The numbers in each sample are not pictures, even though they are shaped a little like small images. For UT-HAR, one sample is 250 time steps by 90 subcarriers, so one axis is time and the other is radio frequency. For NTU-Fi it is three antenna pairs by 114 subcarriers by 500 time steps. Widar3.0 is different again, because its authors already converted the raw signal into something called a Body coordinate Velocity Profile, which is a 20 by 20 grid of velocities captured at 22 time steps.

Widar3.0 matters most to us, and the reason is hidden in its filenames. A file named `user10-1-1-1-1-1-...csv` encodes, in order, the volunteer number, the gesture, where the person was standing, which way they were facing, and which repetition it was. That means we know the exact recording conditions for every single sample, and that is what makes section 5 possible.

Two small traps that I have already hit and do not want to hit again. The UT-HAR files end in `.csv` but are actually NumPy `.npy` dumps, so they must be opened with `np.load` and not `np.loadtxt`. And the benchmark's own README lists the NTU-Fi HumanID split as 546 train and 294 test, while the folders on disk are the other way round.

Running `python3 code/check_data.py` verifies all of this. It needs only NumPy, so it works before PyTorch is installed.

## 4. What the existing code already does

The folder `code/WiFi-CSI-Sensing-Benchmark/` holds the SenseFi benchmark, published in the journal Patterns by Cell Press. It is the standard starting point in this area. Using it is not cheating, it is the sensible baseline that the research papers themselves compare against.

Here is what it gives us for free.

It implements eleven models in PyTorch and lets us train any of them on any of the four datasets with a single command, for example `python run.py --model ResNet18 --dataset NTU-Fi_HAR`. The eleven models are MLP, LeNet, ResNet18, ResNet50, ResNet101, RNN, GRU, LSTM, BiLSTM, CNN+GRU, and a Vision Transformer. It handles loading each dataset, normalising it, training, and printing the test accuracy. There is also a second script, `self_supervised.py`, which trains a model on NTU-Fi HAR without using any labels and then tests what it learned on NTU-Fi HumanID.

Here is what it does not do, which is where our work begins.

It reports accuracy and nothing else. It never records how many parameters a model has or how long it takes to make one prediction, even though those numbers decide whether a model could ever run on a real access point.

It loads data on a single CPU thread. There is no `num_workers` setting anywhere in the repository. I measured the cost, and parsing one Widar sample takes 5.67 milliseconds, so a single pass over the training set spends 3.3 minutes simply reading files before any learning happens. The shipped configuration trains for 100 epochs, which works out to roughly five and a half hours of pure file parsing while the GPU sits idle. Some epoch counts are stranger still, with one UT-HAR configuration set to 3000 epochs and one Widar configuration set to 500.

Most importantly, it splits Widar3.0 in a way that flatters the results, which is the next section.

## 5. The gap we found, and what makes this a real project

We cannot simply run someone else's code and submit the output. So I went looking for something genuinely worth doing, and found it in the way the benchmark divides Widar3.0 into a training half and a testing half.

I parsed every filename in both halves and compared them. All seventeen volunteers appear in the training set and all seventeen appear again in the test set. All eight standing positions appear on both sides. All five body orientations appear on both sides.

This means the split is what researchers call domain mixed. When the model is tested, it is being asked to recognise gestures performed by people it has already trained on, standing where it has already seen them stand, facing the way it has already seen them face. The reported accuracy is therefore an in domain number. It tells us that the model learned the training conditions well. It does not tell us whether the model would work for a person it has never met, standing somewhere it has never seen, which is the only thing that matters if this technology is ever going to ship.

This is exactly the criticism that the 2024 to 2026 research literature makes about the field, and the papers making it are already sitting in our `references/` folder. The honest experiment has not been run in this repository, and because Widar3.0 records the conditions in its filenames, we are in a position to run it.

So our contribution is to rebuild the split so that whole conditions are held out, then measure how far the accuracy falls, and then try to win it back.

Building those splits turned up a second thing that we did not expect, and it changed the design. The volunteers did not all perform the same gestures. Users 1 and 2 performed all twenty two and hold 48 percent of the data between them, while ten volunteers performed only nine gestures each, and gestures 13 to 22 were performed by just two people. That means a cross user split cannot use all twenty two classes, because holding out the wrong volunteer would delete entire classes from the training set. Standing positions have a similar problem, since positions 6, 7 and 8 cover only six gestures and about 600 samples each. Orientation turned out to be the clean factor, because all five orientations cover all twenty two gestures with roughly 8730 samples each.

We only found this because the split code refuses to return a split that has lost a class. That check is worth keeping, because a silently broken split would have produced a dramatic accuracy drop that meant nothing at all.

The three protocols we settled on, all of which pass the leak check, are these.

| Protocol | Held out of training | Classes | Train | Test |
|---|---|---|---|---|
| cross-orientation | body orientation 5 | 22 | 34927 | 8725 |
| cross-position | standing position 5 | 22 | 35280 | 8372 |
| cross-user | volunteers 15, 16 and 17 | 9 | 26648 | 3370 |

Each protocol also carries a matched in domain split, which is a random split over exactly the same samples with exactly the same test size. This matters more than it looks. Without the matched partner we would be comparing a nine class cross user number against a twenty two class benchmark number, and the difference would tell us nothing. With it, the only thing that changes between the two columns of our results table is whether the test conditions were seen during training, which is precisely the question we are asking.

## 6. What we will do for the midterm

The midterm submission is the literature survey plus the groundwork, and it is the cheaper half of the project. Judging by last year's reports, the midterm does not require any experimental results at all, but we will include some anyway, because it makes the submission clearly stronger and because it makes the end term much easier.

The midterm will contain the following four things.

First, a written survey of WiFi sensing, covering the IEEE 802.11bf standard published in September 2025, the research papers from 2024 onwards that are already collected in `references/`, and the commercial products shipping today. This satisfies the slide 10 requirement for papers, standards, and products together.

Second, a fixed and profiled data pipeline. This one is **done**. `code/build_widar_cache.py` converts all 43652 Widar CSV files into a single 1.54 GB memory mapped array plus a metadata table, using all 28 CPU threads, and the whole conversion takes about six seconds. The measured effect on the data path is large. Reading a sample with the loader the benchmark actually uses costs 5.67 milliseconds when the file is not in the operating system cache and about 2.6 milliseconds when it is, which works out to somewhere between 1.5 and 3.3 minutes of parsing per training epoch. Reading the same sample from the cache costs 0.002 milliseconds, so the parsing cost per epoch effectively disappears. That is a speedup of roughly a thousand times on the data path, and it is what makes the rest of the experiments affordable on a laptop.

Third, the honest split code. This one is **done** as well. `code/splits.py` builds the three protocols in the table above, refuses any split that leaks a condition or loses a gesture class, and prints the evidence saved in `results/widar_split_evidence.txt`.

Fourth, reproduced baseline results. This is now **done**. All 44 runs finished, meaning eleven models on each of the four datasets, and the generated tables are in `results/midterm_tables.md`. Three things came out of it that are worth reporting in their own right.

The first is that the benchmark's training recipe is unstable. Ten of the 44 runs ended more than five accuracy points below their own best epoch, and ResNet101 on UT-HAR is the extreme case, reaching 0.972 at its best epoch and then collapsing to 0.099 by the last one. Since the benchmark only ever evaluates after the final epoch, its published numbers inherit whatever the last epoch happened to produce. We record both figures, which is why we can show this rather than repeat it.

The second is that bigger models are not better here, and on Widar3.0 they are consistently worse. LeNet at 299 thousand parameters is the most accurate model on that dataset at 0.704, while ResNet101 at 42.6 million parameters manages 0.658 and takes 67 minutes to train against LeNet's 5. The likely reason is that a Widar sample is a 20 by 20 velocity grid rather than a photograph, so architectures designed for ImageNet are simply mismatched to it. The recurrent models tell the same story on NTU-Fi, where a 79 thousand parameter GRU reaches 1.000 and beats every ResNet.

The third is the size of the spread. On NTU-Fi HAR the MLP holds 175 million parameters and 701 MB of weights to reach 0.981, while the GRU reaches 1.000 with 79 thousand parameters and 0.3 MB. That is a factor of 2200 in size for a worse result.

The size and speed half of this was collected separately. `code/model_profile.py` measures the parameter count and inference latency of all eleven models on all three datasets, and the table is in `results/model_profile.txt`. The spread is striking and is a result in its own right. On NTU-Fi the models range from 26 thousand parameters for the RNN to 175 million for the MLP, a factor of six thousand, and the Vision Transformer is the slowest model on that dataset at 54 milliseconds per batch despite holding fewer than a million parameters. With the accuracy numbers now beside them, the report says which of the eleven are actually worth their cost. The midterm report prints both latency columns, the GPU figure at batch 64 that the literature usually quotes and the single sample CPU figure that an access point would actually see.

Fifth, the domain mixing finding from section 5, written up with the evidence behind it. The evidence itself is already generated and sitting in `results/`.

**The midterm is now complete.** The literature survey covering nineteen papers is written into section 4 of the report, and the report itself is `CSE515_Midterm_Report.docx`, which is 8 pages with three figures and five tables. It is generated by `code/make_report.py` from the recorded results, so editing an experiment and rebuilding keeps the document honest. If I hand edit the Word file directly I should remember that rerunning the generator will overwrite it.

## 7. The end term work

The end term submission is the midterm document extended with the real experiments. Last year one group reused their midterm text word for word and simply appended to it, so nothing written for the midterm is wasted. The generator handles this with a flag, so `python code/make_report.py --stage midterm` still rebuilds the submitted midterm unchanged and `--stage endsem` builds the longer document.

Two things found while planning this phase changed its shape, and both make the result stronger than we expected.

**The leakage is total, not partial.** The midterm reported that all 17 volunteers, all 8 positions and all 5 orientations appear on both sides of the shipped split. Measuring it per recording condition is far more damning. Group every sample by the combination of gesture, volunteer, position and orientation and there are 4265 such conditions holding 10.2 recordings each. 4259 of them straddle the shipped split, and every one of the 8726 test samples has a sibling in training recorded under the identical condition. The benchmark is largely testing on repeat takes of recordings it has already learned.

One trap worth remembering. The repetition number in the filename is not a unique identifier. 4370 files carry a bracketed suffix from a name collision and reuse a repetition number another file already holds, so grouping by repetition would still leak. The condition group deliberately leaves repetition out.

**The condition matters more than the label in the raw data.** Two recordings sharing a condition have a mean cosine similarity of 0.565. Two recordings of the same gesture under different conditions reach only 0.449, against 0.437 for two recordings of different gestures. So sharing a gesture is worth 0.012 and sharing a condition is worth 0.129, about ten times more. A nearest neighbour lookup with no training at all scores 0.306 on the benchmark's split against a chance level of 0.045. That is measured by `code/evidence.py` in seconds and no model can be blamed for it.

This gives us three arms instead of two, and the decomposition is the contribution.

| Arm | How it splits | What it measures |
|---|---|---|
| Benchmark style | random over samples | what SenseFi reports today |
| Grouped in domain | random over whole recording conditions, every factor still shared | in domain accuracy once repeat takes cannot leak |
| Cross domain | one whole orientation, position or volunteer group held out | accuracy on a condition never seen |

The first gap is the leakage the benchmark gives itself. The second is the real domain gap. For LeNet with an orientation held out the three numbers are 0.687, 0.638 and 0.326, so of the 0.361 total fall about 0.049 is leakage and 0.313 is the genuine domain shift.

The end term contains the following.

First, the cross domain experiment across all three protocols and three arms, on four models. The drop is the headline result.

Second, the leave one out sweep, holding out each orientation and each position in turn. This turned out to matter more than expected. Accuracy ranges from about 0.62 down to about 0.32 depending on which orientation is held out, because holding out an end of the range asks the model to extrapolate while holding out a middle value only asks it to interpolate. Any single cross domain number is therefore a choice that moves the headline a long way.

Third, the recovery attempts, using physics motivated augmentation and mixup, reported honestly including whatever fails to help.

Fourth, the efficiency study, extended with CPU latency for a single sample on four threads, because an access point has no GPU and classifies one sample at a time, which makes the midterm's batch 64 GPU figure the wrong measurement for the deployment question.

Fifth, a measured bug report. The benchmark's CNN+GRU ends in `nn.Softmax` and its training loop then applies `nn.CrossEntropyLoss`, which softmaxes again. The same mistake is in the UT-HAR and NTU-Fi versions. We do not patch the vendored copy, we measure what removing it is worth.

Sixth, a conclusion that says plainly what works, what does not, and what we would do next.

One thing was dropped from the original plan. Self supervised pre training was going to be the second recovery method, but the benchmark's own implementation is hardcoded for the NTU-Fi tensor shape and cannot be reused for Widar, so it would have meant writing a contrastive stage from scratch for a method that rarely beats good supervised augmentation at this data scale. Few shot adaptation is the better use of that effort and is named as the next step in the conclusion.

### Running the end term sweep

```bash
$PY code/evidence.py      # model-free leakage evidence, seconds
$PY code/run_endsem.py    # 61 runs, about 9 hours, resumable
$PY code/model_profile.py # rerun after the sweep, adds CPU latency
$PY code/make_endsem_tables.py
$PY code/make_endsem_figures.py
$PY code/make_report.py --stage endsem
```

`run_endsem.py` skips anything already recorded, so if it is interrupted it is restarted with the same command. Jobs are ordered by how much the report needs them rather than by cost, so the whole LeNet headline result is finished in the first half hour and the expensive models come last.

### Status, 22 September 2026

The end term experiments are finished. 65 runs, no failures. Everything in both reports is generated from recorded results, so rebuilding after any change keeps the documents honest.

```bash
$PY code/make_endsem_tables.py && $PY code/make_endsem_figures.py
$PY code/make_report.py --stage endsem     # 14 pages
$PY code/make_report.py --stage midterm    # 8 pages
```

What the end term found, in short. Holding out a body orientation costs about 0.30 accuracy on every architecture tried, holding out a standing position costs nothing at all, and holding out volunteers costs about 0.13. Near duplicate leakage in the benchmark's own split accounts for a further 0.05 that has nothing to do with generalisation. Augmentation wins back at most 0.044 of the orientation gap without costing in domain accuracy. Few shot adaptation does not rescue it, and retraining the whole network on a few hundred labelled examples from the new condition is markedly worse than not adapting at all.

One thing to be careful about if the job matrix is extended. Recovery runs are keyed by protocol name, and a run on `cross-orientation/in_domain_grouped` shares its base name with one on `cross-orientation`. Any code that aggregates them has to filter on the arm and the model as well as the name, or the control runs silently overwrite the numbers they exist to check. That bug was live in `make_endsem_tables.py` for one generation and is fixed.

### Submission format, taken from last year's reports

The reports in `sample_submissions/` are useful only for their format. Their content is unrelated to what we are doing.

Everyone used the same section order: Introduction, Motivation, Background on the topic, Related Works, Problem Statement, Proposed Solution, System Design, Results, Conclusion, References, with the code in an appendix at the very end. The midterm ran about 8 pages and 1700 words. The end term ran about 12 pages and 2600 words, and longer once the code appendix was included. Everything was written in Word or Google Docs and exported to PDF, with no LaTeX and no IEEE template. Results were presented as a small table with the baseline in one column and the improved version in the next, followed by plots with captions.

### Our page budget and house style

The midterm report must fit 7 to 8 pages and the end term report 12 to 14. Both are currently inside that, at 8 and 11, with the end term leaving room for the sections still waiting on the sweep. Check a rebuild with

```bash
soffice --headless --convert-to pdf CSE515_Endsem_Report.docx && pdfinfo CSE515_Endsem_Report.pdf | grep Pages
```

Three things keep the length down and should not be undone without checking the page count afterwards. The body text is 10pt Calibri at 1.05 line spacing with 0.75 inch margins. The title is a compact header at the top of page one rather than a separate title sheet, which is worth a full page on a report this short, and the page break can be put back at the end of `title_page` if a separate sheet is wanted. The appendix lists the code files as a single sentence rather than a table.

On the writing itself. Keep it concise and do not pad it with context the reader does not need. Use complete sentences in plain language, and never use long dashes. After any edit, confirm with

```bash
pdftotext CSE515_Endsem_Report.pdf - | grep -c '—\|–'     # must print 0
```

Avoid the stock phrases that make writing read as machine generated, in particular "not only ... but", "it is worth noting", "moreover", "furthermore", "crucial" and "ultimately". `code/make_report.py` is the only place the prose lives, so fix it there rather than in the Word file, because rebuilding overwrites the document.

### Rebuilding the proposal PDF

```bash
google-chrome --headless --disable-gpu --no-sandbox --no-pdf-header-footer \
  --print-to-pdf=Proposal.pdf proposal.html
pdfinfo Proposal.pdf | grep Pages     # must say 1
```

The page is tuned to fill exactly one A4 sheet. If text is added and it spills onto a second page, the `html { zoom: 0.98 }` rule near the top of the stylesheet is the dial to turn down.

### Rules from slide 10 that we must not break

Every figure in the report has to be redrawn by us. Every idea has to be written in our own words. Short quotes are acceptable, copied text and copied figures are not, and the slide says plainly that any infringement forfeits the grade.

### To confirm with the instructor

Slide 7 says teams of four while slide 13 says a maximum of three students. We are three, which is safe either way, but it is worth asking. Slide 13 also lists a Project Homework 1, a list of five references, due on 8 August 2026, and we should check whether that is still outstanding. Finally, slide 7 requires the project to be approved before we start, so the proposal goes first.

---

## What is in this folder

```
.
├── CSE515_Midterm_Report.docx   the midterm report, generated by make_report.py
├── Proposal.pdf     the one page proposal for the instructor
├── proposal.html    source for the PDF, edit this and re-render
├── PROPOSAL.md      the same text in markdown for copying into other documents
├── README.md        this file
├── intro.pptx       the course intro slides
├── references/      19 paper PDFs, plus INDEX.md explaining what each one is for
├── data/Data/       the four CSI datasets, extracted and verified
├── data/cache/      the cached Widar array and metadata table, built by build_widar_cache.py
├── results/         generated evidence and result tables
│   ├── midterm_tables.md        the midterm tables, regenerate with make_tables.py
│   ├── endsem_tables.md         the end term tables, from make_endsem_tables.py
│   ├── baselines.jsonl          one line per training run, midterm and end term
│   ├── evidence.json            the model free similarity and lookup numbers
│   ├── model_profile.txt        parameter counts, GPU latency and CPU latency
│   ├── endsem_sweep.log         the running log of the end term sweep
│   ├── ckpt/                    saved weights from the cross domain runs
│   └── widar_split_evidence.txt the leakage evidence, five sections of it
├── sample_submissions/   last year's reports, kept for format reference only
└── code/
    ├── check_data.py                 dataset sanity check, run this first
    ├── build_widar_cache.py          one time conversion of the Widar CSVs into a cache
    ├── splits.py                     the three split arms, and the evidence for them
    ├── evidence.py                   leakage measured with no model, similarity and 1-NN
    ├── model_profile.py              parameters, GPU latency and single sample CPU latency
    ├── train.py                      trains one model on one dataset, protocol and arm
    ├── run_baselines.py              drives the midterm sweep, skips what is recorded
    ├── run_endsem.py                 drives the 61 end term runs, also resumable
    ├── make_tables.py                the midterm tables
    ├── make_endsem_tables.py         the end term tables
    ├── make_figures.py               the six midterm figures
    ├── make_endsem_figures.py        the end term figures
    ├── make_report.py                builds either report, --stage midterm or endsem
    └── WiFi-CSI-Sensing-Benchmark/   the SenseFi baseline repository
```
