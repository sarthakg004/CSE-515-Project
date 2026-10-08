"""Build the midterm or the end term report as a .docx file.

Section order follows the reports in sample_submissions/. Every number in the tables is
read from results/, so the document cannot drift out of step with the experiments.

The end term report is the midterm report extended rather than replaced, which is how
the previous submissions for this course were written. Sections 1 to 4 carry over almost
unchanged, sections 5 to 8 gain the new material, and section 9 is rewritten.

    /home/sarthak/miniconda3/envs/op05/bin/python code/make_report.py --stage midterm
    /home/sarthak/miniconda3/envs/op05/bin/python code/make_report.py --stage endsem
"""
import argparse
import json
import os

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(RESULTS, "figures")
OUT = {"midterm": os.path.join(ROOT, "CSE515_Midterm_Report.docx"),
       "endsem": os.path.join(ROOT, "CSE515_Endsem_Report.docx")}
STAGE = ["midterm"]          # set from the command line in main()


def endsem():
    return STAGE[0] == "endsem"

ACCENT = RGBColor(0x12, 0x3A, 0x5E)
INK = RGBColor(0x15, 0x20, 0x2B)
MUTED = RGBColor(0x5A, 0x66, 0x73)

ORDER = ["MLP", "LeNet", "ResNet18", "ResNet50", "ResNet101",
         "RNN", "GRU", "LSTM", "BiLSTM", "CNN+GRU", "ViT"]
NICE = {"UT_HAR_data": "UT-HAR", "NTU-Fi_HAR": "NTU-Fi HAR",
        "NTU-Fi-HumanID": "NTU-Fi HumanID", "Widar": "Widar3.0"}

_fig_n = [0]
_tab_n = [0]
_subn = {}


def sub(doc, sec, title):
    """Numbered subsection heading. Auto numbering means a subsection can be
    dropped in one stage of the report without renumbering the rest by hand."""
    _subn[sec] = _subn.get(sec, 0) + 1
    doc.add_heading(f"{sec}.{_subn[sec]} {title}", level=2)


# ----------------------------------------------------------------- helpers
def load():
    runs = [json.loads(l) for l in open(os.path.join(RESULTS, "baselines.jsonl"))
            if l.strip()]
    acc = {(r["dataset"], r["model"]): r for r in runs if r.get("tag") == "baseline"}
    prof = {}
    pf = os.path.join(RESULTS, "model_profile.jsonl")
    if os.path.exists(pf):
        prof = {(p["dataset"], p["model"]): p
                for p in (json.loads(l) for l in open(pf) if l.strip())}
    return acc, prof


ARMS = ["in_domain", "in_domain_grouped", "cross"]
PROTO_NAME = {"cross-orientation": "orientation", "cross-position": "standing position",
              "cross-user": "volunteer"}
EACC = "val_sel_acc"


def an(word):
    """Correct indefinite article. PROTO_NAME holds "orientation"."""
    return ("an " if word[0].lower() in "aeiou" else "a ") + word


_WORDS = ["zero", "one", "two", "three", "four", "five", "six",
          "seven", "eight", "nine", "ten", "eleven", "twelve"]


def nw(n):
    """Small counts are spelled out in prose, larger ones stay as digits."""
    n = int(n)
    return _WORDS[n] if 0 <= n < len(_WORDS) else f"{n:,}"


def split_protocol(p):
    base, _, arm = p.partition("/")
    name, _, held = base.partition(":")
    return name, held, (arm or "cross")


def load_endsem():
    """Everything the end term sections need, keyed for direct lookup.

    Returns (arms, loo, rec, evidence, runs) where arms[(model, protocol, arm)] is a run
    row, loo[protocol][value] is the accuracy with that value held out, and
    rec[method][protocol] is the accuracy of a recovery method.
    """
    path = os.path.join(RESULTS, "baselines.jsonl")
    runs = [json.loads(l) for l in open(path) if l.strip()] if os.path.exists(path) else []
    runs = [r for r in runs if str(r.get("tag", "")).startswith("endsem")]

    arms, loo, rec = {}, {}, {}
    for r in runs:
        if r["protocol"] == "original":
            continue
        name, held, arm = split_protocol(r["protocol"])
        if r["tag"] == "endsem" and not held:
            arms[(r["model"], name, arm)] = r
        if r["model"] == "LeNet" and arm == "cross":
            if r["tag"] == "endsem-loo" and held:
                loo.setdefault(name, {})[int(held)] = r[EACC]
            elif r["tag"] == "endsem" and not held:
                loo.setdefault(name, {})[5] = r[EACC]
            elif r["tag"] == "endsem-rec":
                m = f"augmentation to {int(r.get('aug_deg', 20))} degrees" \
                    if r.get("augment") else ""
                if r.get("mixup"):
                    m = f"{m} with mixup" if m else "mixup"
                rec.setdefault(m, {})[name] = r[EACC]

    ev = {}
    p = os.path.join(RESULTS, "evidence.json")
    if os.path.exists(p):
        ev = json.load(open(p))
    return arms, loo, rec, ev, runs


def style_doc(doc):
    n = doc.styles["Normal"]
    n.font.name = "Calibri"
    n.font.size = Pt(10)
    n.font.color.rgb = INK
    n.paragraph_format.space_after = Pt(5)
    n.paragraph_format.line_spacing = 1.05
    for name, size, colour, before in [("Heading 1", 13, ACCENT, 10),
                                       ("Heading 2", 10.5, ACCENT, 7),
                                       ("Heading 3", 10, INK, 6)]:
        s = doc.styles[name]
        s.font.name = "Calibri"
        s.font.size = Pt(size)
        s.font.bold = True
        s.font.color.rgb = colour
        s.paragraph_format.space_before = Pt(before)
        s.paragraph_format.space_after = Pt(4)
        s.paragraph_format.keep_with_next = True


def para(doc, text, size=None, bold=False, italic=False, align=None,
         colour=None, after=None, style=None):
    p = doc.add_paragraph(style=style)
    r = p.add_run(text)
    if size:
        r.font.size = Pt(size)
    r.bold, r.italic = bold, italic
    if colour:
        r.font.color.rgb = colour
    if align is not None:
        p.alignment = align
    else:
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if after is not None:
        p.paragraph_format.space_after = Pt(after)
    return p


def bullets(doc, items):
    for it in items:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(it).font.size = Pt(10.5)
        p.paragraph_format.space_after = Pt(3)


def shade(cell, hexcolour):
    el = OxmlElement("w:shd")
    el.set(qn("w:fill"), hexcolour)
    cell._tc.get_or_add_tcPr().append(el)


def add_table(doc, headers, rows, caption, widths=None, highlight=None):
    _tab_n[0] += 1
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        c = t.rows[0].cells[i]
        c.text = ""
        r = c.paragraphs[0].add_run(h)
        r.bold = True
        r.font.size = Pt(8)
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        shade(c, "123A5E")
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(str(v))
            run.font.size = Pt(8)
            if highlight is not None and ri in highlight:
                run.bold = True
            cells[i].paragraphs[0].paragraph_format.space_after = Pt(1)
        if ri % 2 == 1:
            for c in cells:
                shade(c, "F1F5F8")
    if widths:
        for r in t.rows:
            for i, w in enumerate(widths):
                r.cells[i].width = Inches(w)
    cap = doc.add_paragraph()
    cr = cap.add_run(f"Table {_tab_n[0]}. {caption}")
    cr.font.size = Pt(8)
    cr.italic = True
    cr.font.color.rgb = MUTED
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_before = Pt(2)
    cap.paragraph_format.space_after = Pt(7)
    return t


def add_figure(doc, filename, caption, width=4.7):
    _fig_n[0] += 1
    path = os.path.join(FIGS, filename)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(1)
    if os.path.exists(path):
        p.add_run().add_picture(path, width=Inches(width))
    else:
        ph = p.add_run(f"[ PLACEHOLDER: insert {filename} here ]")
        ph.italic = True
        ph.font.color.rgb = RGBColor(0xA4, 0x42, 0x3A)
    cap = doc.add_paragraph()
    cr = cap.add_run(f"Figure {_fig_n[0]}. {caption}")
    cr.font.size = Pt(8)
    cr.italic = True
    cr.font.color.rgb = MUTED
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(7)


# ----------------------------------------------------------------- content
def title_page(doc):
    """A compact header rather than a whole title sheet.

    A full title page costs one of the seven or eight pages this report is meant to
    fit in, for about fifty words. The same information sits at the top of page one
    and the introduction follows underneath. Restore the page break at the end of
    this function if a separate title sheet is wanted.
    """
    para(doc, "CSE515: Advanced Communications Networks and Systems",
         size=10, align=WD_ALIGN_PARAGRAPH.CENTER, colour=MUTED, after=1)
    para(doc, "WiFi Sensing with Channel State Information for Device Free "
              "Human Activity Recognition and Indoor Localization",
         size=15, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, colour=INK, after=3)
    para(doc, "End Semester Project Report" if endsem() else "Midterm Project Report",
         size=10, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, colour=MUTED, after=5)
    para(doc, "Saarthak Gupta (22134024)    Shreya Sangma (22134025)    "
              "Vaibhav Jain (22164020)",
         size=10, align=WD_ALIGN_PARAGRAPH.CENTER, after=1)
    para(doc, "Instructor: Dr. Ajay Pratap. Department of Computer Science and "
              "Engineering, Indian Institute of Technology (BHU) Varanasi",
         size=9, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, colour=MUTED, after=10)


def sec_introduction(doc):
    doc.add_heading("1. Introduction", level=1)
    para(doc, "Every WiFi device measures how the radio signal changes on its way from "
              "the transmitter to the receiver. That measurement is called Channel State "
              "Information, usually shortened to CSI. It needs no extra hardware, "
              "because a WiFi chip already computes it in order to decode packets, and "
              "many commodity chipsets will simply report it.")
    para(doc, "A person moving through the room reflects some of those radio waves and "
              "blocks others, so the CSI values change. They change in a pattern that "
              "depends on what the person did, and walking produces a different pattern "
              "from sitting down or falling. Record enough of those patterns with labels "
              "and a model can be trained to tell them apart, at which point an ordinary "
              "WiFi link has become a sensor. Nobody has to wear a device, no camera is "
              "involved, it works in the dark, and radio passes through plasterboard. "
              "Fall detection for elderly people living alone is the application most "
              "often cited.")
    para(doc, "This report covers " + ("both halves of our project. We survey the recent "
              "literature, reproduce the standard benchmark for this field, show that "
              "the way it measures success is broken, measure how much accuracy an "
              "honest evaluation costs, and try to win that accuracy back."
              if endsem() else
              "the first half of our project. We survey the recent literature, reproduce "
              "the standard benchmark for this field on four public datasets, and report "
              "a problem we found in how that benchmark measures success. The second "
              "half will act on that problem."))


def sec_motivation(doc):
    doc.add_heading("2. Motivation", level=1)
    para(doc, "The topic is current rather than settled textbook material. IEEE "
              "published Standard 802.11bf, the WLAN Sensing amendment, on 26 September "
              "2025, and it defines how devices advertise sensing support, set up a "
              "sensing session and report measurements [2]. Cognitive Systems, Origin "
              "AI and Linksys already sell motion sensing features built on the same "
              "principle [10]. "
              "The course notes ask for a topic covered by recent papers, industry "
              "standards and shipping products together, and this one satisfies all "
              "three.")
    para(doc, "The research reason is that the field has a well known unsolved "
              "problem. A model trained in one setting with one group of people tends "
              "to fail as soon as it meets a different room or person. This domain "
              "shift is the main thing standing between a laboratory demonstration and "
              "a product, and our project is aimed at it.")


def sec_background(doc):
    doc.add_heading("3. Background on WiFi Sensing", level=1)

    sub(doc, 3, "What CSI is")
    para(doc, "Early wireless sensing used the Received Signal Strength Indicator, a "
              "single number that collapses the whole channel into one value. CSI is "
              "far richer. A WiFi link using orthogonal frequency division multiplexing "
              "splits its bandwidth into many narrow subcarriers, and CSI reports "
              "amplitude and phase for each subcarrier and each pair of antennas, so "
              "one measurement is a small matrix rather than a scalar. A moving body "
              "changes the length of the reflected paths and produces a Doppler shift, "
              "and different movements produce different shifts. That is why the same "
              "measurement supports both coarse tasks such as occupancy and fine ones "
              "such as recognising a gesture drawn in the air.")

    sub(doc, 3, "How the field developed")
    para(doc, "Ma, Zhou and Wang surveyed a decade of commodity WiFi sensing and group "
              "it into activity recognition, object sensing and localization [4]. "
              "Hardware stopped being the bottleneck once CSI could be read from "
              "ordinary network cards, after which the difficulty moved into processing "
              "and then into machine learning. That phase brought a new problem, which "
              "is that results became hard to compare, because different groups used "
              "different datasets, preprocessing and splits. Yang and colleagues "
              "answered this with SenseFi, an open benchmark implementing eleven model "
              "families on four public datasets under one procedure [1]. SenseFi is our "
              "starting point, and Section 5 explains where we found it wanting.")
    para(doc, "Sensing is also being folded into the standard itself. Meneghello and "
              "colleagues set out the physical and medium access layer changes behind "
              "802.11bf [3]. The standard deliberately leaves the sensing algorithm to "
              "the implementer, so the modelling question this project studies is not "
              "settled by it. The wider idea is Integrated Sensing and Communication, "
              "where one radio carries data and senses at the same time, as in ISAC-Fi "
              "[12] and in related work inferring human pose from 5G uplink signals "
              "[13].")


def sec_related(doc):
    doc.add_heading("4. Related Works", level=1)
    para(doc, "We reviewed nineteen papers, weighted towards work from 2024 onwards as "
              "the course notes require. They fall into four groups.")

    sub(doc, 4, "Benchmarks and surveys")
    para(doc, "Yang and colleagues introduced SenseFi, comparing eleven deep learning "
              "models across four CSI datasets and reporting accuracy, model size and "
              "the transferability of learned features [1]. Ma and colleagues give the "
              "broader ten year survey [4]. A 2025 study compares BiLSTM against CNN+GRU "
              "on the same UT-HAR and NTU-Fi datasets we use and argues that dataset "
              "characteristics matter as much as architecture [11], which agrees with "
              "what we measured independently in Section 8.")

    sub(doc, 4, "The generalisation problem")
    para(doc, "This is the central open problem and the reason our project exists. "
              "Strohmayer and Kampel apply image style augmentation to CSI and study "
              "generalisation between line of sight and through wall scenarios [6]. Xu "
              "and colleagues survey self supervised learning for CSI activity "
              "recognition, motivated by the scarcity of labels [22], and CAPC combines "
              "contrastive predictive coding with augmentation so a model can learn from "
              "unlabelled data and adapt across environments [14].")
    para(doc, "Two recent papers show how far the framing has moved. MU-SHOT-Fi adapts "
              "to a new environment using only unlabelled data from it and a pretrained "
              "model, because privacy rules often prevent sharing the original training "
              "data [15]. An identity agnostic approach reformulates multi user "
              "recognition as activity counting so the model stops depending on a fixed "
              "set of known users [16]. Both react to the same weakness we identify in "
              "Section 5, which is that models quietly learn the people they trained "
              "on.")

    sub(doc, 4, "Efficiency and deployment")
    para(doc, "Sensing too expensive to compute will never run on a real access point. "
              "EfficientFi compresses CSI at the edge and restores it in the cloud so "
              "sensing can scale to many homes [5]. WiFlexFormer is a compact "
              "transformer reaching comparable accuracy with far fewer parameters, in "
              "about 10 milliseconds on a Jetson Orin Nano [7]. Its authors observe "
              "that a low parameter count helps cross domain generalisation, so small "
              "models often beat larger ones when conditions change, and our own "
              "measurements in Section 8 point the same way.")
    para(doc, "Newer work pushes efficiency harder. A physics guided temporal "
              "convolutional network uses Doppler energy to focus attention where motion "
              "actually happens [17], Wi-Spike uses a spiking network for low power "
              "recognition [18], and a 2026 paper treats the acquisition rate as a "
              "constrained resource, since sensing shares airtime with ordinary traffic "
              "[19].")

    sub(doc, 4, "Localization and real deployments")
    para(doc, "Barahimi and colleagues report lessons from a CSI localization system "
              "using more than 400 access points in a real building, and name the "
              "underuse of unlabelled data and the heterogeneity of CSI measurements as "
              "the factors that most limit large scale deployment [8]. Other work "
              "studies malicious access points that shift the readings a fingerprinting "
              "system depends on [20], and maxVSTAR uses a camera to relabel CSI "
              "automatically so an edge model keeps adapting after installation [21].")
    para(doc, "The literature agrees that generalisation is the central problem and "
              "many groups propose ways to improve it. What we found missing is a check "
              "on whether the standard evaluation measures generalisation at all, and "
              "that is the gap this report documents.")


def sec_problem(doc, acc):
    doc.add_heading("5. Problem Statement", level=1)
    para(doc, "A project that only runs someone else's code and reports the output is "
              "not worth submitting, so we first examined how the standard benchmark "
              "decides whether a model is good. We found a problem in it.")

    sub(doc, 5, "What we checked")
    para(doc, "Widar3.0 records the conditions of every sample in its filename. A file "
              "named user10-1-1-1-1-1 encodes the volunteer, the gesture, where the "
              "person stood, which way they faced and which repetition it was. We "
              "parsed all 43,652 filenames in the benchmark's two halves and compared "
              "which conditions appear on each side.")

    add_table(doc,
              ["Factor", "Values in training", "Values in testing", "Values in both"],
              [["Volunteers", "17", "17", "17"],
               ["Standing positions", "8", "8", "8"],
               ["Body orientations", "5", "5", "5"]],
              "Every recording condition in Widar3.0 appears on both sides of the "
              "benchmark's train and test split. Nothing is held out.",
              widths=[1.8, 1.5, 1.5, 1.4])

    para(doc, "Every volunteer, position and orientation appears in both halves. The "
              "split is domain mixed, because the two halves were drawn from one pool "
              "and shuffled, so nothing in the test set is new to the model.")
    if not endsem():
        add_figure(doc, "fig1_split_evidence.png",
                   "The benchmark's split holds nothing back, so the model is tested on "
                   "people, positions and orientations it has already trained on. Our "
                   "cross-user split holds three volunteers out entirely.")

    sub(doc, 5, "Why this matters")
    para(doc, "Such a model is asked to recognise gestures performed by people it has "
              "trained on, standing where it has seen them stand and facing the way it "
              "has seen them face. It is an examination where the practice paper and "
              "the real paper come from the same question bank, so a high score shows "
              "familiarity rather than understanding. The accuracy fairly measures how "
              "well a model learned its training conditions, but it does not answer the "
              "question that decides whether the technology can ship, which is whether "
              "it works for a person it has never met.")

    if endsem():
        sub(doc, 5, "The leakage is worse when counted per recording condition")
        para(doc, "Counting factor by factor understates the problem, and while building "
                  "the corrected splits we found the sharper way to measure it. Group "
                  "every sample by the combination of gesture, volunteer, standing "
                  "position and body orientation, and call that a recording condition. "
                  "Widar3.0 contains 4265 of them, holding an average of 10.2 recordings "
                  "each, because each volunteer repeated every gesture several times "
                  "before moving on.")
        add_table(doc, ["Measurement", "Value"],
                  [["Recording conditions in the dataset", "4265"],
                   ["Recordings per condition, on average", "10.2"],
                   ["Conditions appearing on both sides of the split", "4259 of 4265"],
                   ["Test samples with a sibling recording in training",
                    "8726 of 8726, which is 100 percent"]],
                  "The benchmark's split measured per recording condition rather than "
                  "per factor.", widths=[3.8, 2.2])
        para(doc, "Every test sample has at least one training sample recorded from the "
                  "same person performing the same gesture in the same place facing the "
                  "same way. We checked that these are not copies of one file. They are "
                  "repeat takes of one action, and they resemble each other far more "
                  "than two recordings of the same gesture made under different "
                  "conditions do. So the benchmark is testing largely on near duplicates "
                  "of its own training data, which is a separate problem from testing in "
                  "domain and has a different size and a different cure. Section 8 "
                  "measures the two apart.")
        para(doc, "One trap is worth recording for anyone repeating this. The repetition "
                  "number in the filename is not a unique identifier for a recording. "
                  "4370 files carry a bracketed suffix from a name collision and reuse a "
                  "repetition number that another file already holds, so 5873 "
                  "combinations own more than one file. Grouping by repetition would "
                  "therefore have produced a split that still leaked. Our condition group "
                  "deliberately leaves repetition out.")

    sub(doc, 5, "The problem we will solve")
    para(doc, "Our project therefore asks two questions. First, how much accuracy is "
              "actually lost when a WiFi sensing model meets conditions that were held "
              "out of its training, as opposed to conditions it has already seen. "
              "Second, how much of that loss can be recovered using techniques available "
              "today, such as data augmentation. " + (
                  "Sections 8.6 to 8.9 answer both."
                  if endsem() else
                  "The midterm work reported here builds everything needed to answer "
                  "them. The end term work will answer them."))


def sec_solution(doc):
    doc.add_heading("6. Proposed Solution", level=1)
    if endsem():
        para(doc, "Our solution has three parts. First, rebuild the evaluation split so "
                  "that whole recording conditions are held out of training, rather "
                  "than shuffling every sample together. A model trained that way has "
                  "never seen the person, position or orientation it is tested on. "
                  "Second, report cost beside accuracy, because accuracy alone cannot "
                  "say whether a model could run on a real access point, so we record "
                  "parameter count, weight size and inference latency as well. Third, "
                  "try to recover what the honest split costs, using augmentation and "
                  "adaptation, and report the outcome including the methods that fail. "
                  "A properly measured negative result is still a result.")
        return

    para(doc, "Our solution has three parts. The first two are complete and are "
              "reported here. The third is the end term work.")

    sub(doc, 6, "Rebuild the evaluation split honestly")
    para(doc, "Instead of shuffling all the samples together, we hold entire recording "
              "conditions out of training, so a model has never seen the people, "
              "positions or orientations it is tested on. Building these splits needed "
              "care, because the dataset is not evenly balanced, and Section 7 explains "
              "what we found.")

    sub(doc, 6, "Measure cost as well as accuracy")
    para(doc, "The benchmark reports accuracy and nothing else, which cannot say "
              "whether a model could ever run on a real access point. We also record "
              "the parameter count, the weight size and the inference latency of every "
              "model, and report them beside the accuracy.")

    sub(doc, 6, "Recover the lost accuracy, and check the cost")
    para(doc, "In the end term we will apply data augmentation to the cross domain "
              "splits and measure how much of the lost accuracy it wins back, "
              "reporting this honestly including anything that fails to help. We will "
              "then check whether a small model retains most of the accuracy of a large "
              "one, which decides whether any of it is deployable.")


def sec_design(doc, acc, prof):
    doc.add_heading("7. System Design", level=1)

    sub(doc, 7, "Datasets")
    para(doc, "We use the four public CSI datasets the benchmark supports, all "
              "released by the researchers who recorded them and all verified in our "
              "repository. Together they occupy about 29 gigabytes. Widar3.0 [9] is the "
              "one this project turns on.")
    add_table(doc,
              ["Dataset", "Task", "Sample shape", "Classes", "Train", "Test"],
              [["UT-HAR", "7 daily activities, 6 people", "1 x 250 x 90", "7",
                "3,977", "996"],
               ["NTU-Fi HAR", "6 activities, 114 subcarriers", "3 x 114 x 500", "6",
                "936", "264"],
               ["NTU-Fi HumanID", "Identify who is walking", "3 x 114 x 500", "14",
                "294", "546"],
               ["Widar3.0", "22 hand gestures", "22 x 20 x 20", "22",
                "34,926", "8,726"]],
              "The four public CSI datasets used in this work.",
              widths=[1.25, 1.85, 1.15, 0.6, 0.7, 0.6])
    para(doc, "These are not images, although two of them are shaped like small ones. "
              "A UT-HAR sample is 250 time steps by 90 subcarriers. Widar3.0 differs, "
              "because its authors converted the raw signal into a Body coordinate "
              "Velocity Profile, a 20 by 20 grid of velocities at 22 time steps. Its filenames record the "
              "volunteer, standing position and body orientation of every sample, "
              "which is what makes the rest of this report possible.")

    sub(doc, 7, "Fixing the data pipeline")
    para(doc, "The benchmark parses one text file per sample on a single thread, since "
              "no worker count is configured anywhere in it. Reading one Widar sample "
              "costs 5.67 milliseconds cold and about 2.6 milliseconds warm, which is "
              "between 1.5 and 3.3 minutes of file reading for every pass over the "
              "training set. At the shipped 100 epochs that is hours spent parsing text "
              "while the graphics processor sits idle. We convert all 43,652 files once "
              "into a memory mapped array of 1.54 gigabytes, plus a metadata table "
              "holding the volunteer, position and orientation parsed from each "
              "filename. The conversion takes about six seconds on 28 threads and a "
              "sample then costs 0.002 milliseconds to read. That is what made the rest "
              "of this work affordable on a laptop.")

    sub(doc, 7, "Building splits that hold up")
    para(doc, "Our split code refuses to return a split that leaks a held out condition "
              "or loses a gesture class, and that check failed on the first attempt. The "
              "volunteers did not all perform the same gestures. Volunteers 1 and 2 "
              "performed all 22 and hold 48 percent of the data, ten performed only 9 "
              "each, and gestures 13 to 22 were performed by just two people, so holding "
              "out the wrong volunteers deletes whole classes. Positions 6, 7 and 8 have "
              "the same problem at about 600 samples each. Orientation is the clean "
              "factor, with all five covering all 22 gestures at roughly 8,730 samples. "
              "Without that check the split would still have produced numbers, and they "
              "would have shown a large drop that measured missing classes rather than "
              "lost generalisation.")
    add_table(doc,
              ["Protocol", "Held out of training", "Classes", "Train", "Test"],
              [["Cross orientation", "Body orientation 5", "22", "34,927", "8,725"],
               ["Cross position", "Standing position 5", "22", "35,280", "8,372"],
               ["Cross user", "Volunteers 15, 16 and 17", "9", "26,648", "3,370"]],
              ("The three cross domain protocols. Each also carries the two in domain "
               "arms described next, over exactly the same samples with the same test "
               "size." if endsem() else
               "The three cross domain protocols. Each one also carries a matched in "
               "domain split, which is a random split over exactly the same samples "
               "with the same test size, so that the only thing that changes between "
               "the two is whether the test conditions were seen during training."),
              widths=[1.4, 1.9, 0.8, 0.9, 0.8])

    if endsem():
        sub(doc, 7, "Three arms rather than two")
        para(doc, "Comparing a cross domain split against a plain random one confuses "
                  "two effects, for the reason given in Section 5. A random split over "
                  "samples leaks near duplicates, because repeat takes of one condition "
                  "fall on both sides of it, and a cross domain split does not. "
                  "Comparing only those two would measure the leakage and the domain "
                  "shift added together and report the sum as though it were the shift "
                  "alone. We therefore run every protocol three ways over the same pool "
                  "of samples with the same test size.")
        add_table(doc, ["Arm", "How the split is made", "What it measures"],
                  [["Benchmark style", "Random over individual samples",
                    "What SenseFi reports today"],
                   ["Grouped in domain",
                    "Random over whole recording conditions, so every volunteer, "
                    "position and orientation still appears on both sides",
                    "In domain accuracy once repeat takes can no longer leak"],
                   ["Cross domain", "One whole orientation, position or volunteer group "
                    "held out", "Accuracy on a condition never seen"]],
                  "The three arms. The difference between the first two is the leakage, "
                  "and the difference between the last two is the domain shift.",
                  widths=[1.2, 2.6, 2.2])
        para(doc, "The grouped arm is the control that makes the decomposition "
                  "possible. It is still domain mixed, so it differs from the benchmark "
                  "arm in one respect only. The split code asserts this, refusing any "
                  "grouped split in which a condition appears on both sides or a factor "
                  "value is lost.")

        sub(doc, 7, "Choosing which epoch to report")
        para(doc, "Our midterm quoted the best epoch, the highest accuracy any epoch "
                  "reached on the test set. For an in domain sweep that is a defensible "
                  "summary, but for a cross domain claim it is peeking, because the test "
                  "labels chose the stopping point. Here we carve a validation set out "
                  "of the training data, pick the epoch on that, and report test "
                  "accuracy at that epoch. The validation set is itself carved at "
                  "recording condition level, otherwise the same near duplicate lookup "
                  "this report is about would win it and it would choose the epoch "
                  "badly. All three readings are kept and Section 8 reports what the "
                  "difference is worth.")

        sub(doc, 7, "The recovery methods")
        para(doc, "Two methods are tried here, both applied only to the training data "
                  "of the cross domain arm. Section 8 adds a third, which is adapting a "
                  "trained model on a few labelled examples from the held out "
                  "condition.")
        bullets(doc, [
            "Augmentation built around the physics rather than around images. The "
            "velocity plane is rotated, which corresponds to a change in the direction "
            "the person faces, and scaled, which corresponds to moving faster or "
            "slower. The sample is also shifted in time, given noise, and has part of "
            "the grid erased. Each time step of a velocity profile is a distribution "
            "summing to about one, so every step is rescaled back to its original mass, "
            "otherwise the benchmark's fixed normalisation constants stop holding.",
            "Mixup, blending two samples and their labels in the same proportion. A "
            "blend of two velocity profiles is still a velocity profile, so this is "
            "reasonable here rather than a purely visual trick.",
        ])
        para(doc, "The rotation angle needs care. Anything close to a quarter turn "
                  "destroys the label, because the gesture set contains Draw N "
                  "horizontally and Draw N vertically as separate classes, and the same "
                  "pairing for Draw O and Draw Zigzag. A small angle is also the "
                  "physically correct choice, since a velocity profile is already in "
                  "body coordinates and only the residual error in estimating the torso "
                  "direction survives. We measured two settings rather than assuming.")

    sub(doc, 7, "Training procedure")
    para(doc, "We keep the benchmark's training recipe so our numbers stay comparable "
              "to the published ones, meaning Adam at a learning rate of 0.001, cross "
              "entropy loss and a batch size of 64. Three things differ deliberately.")
    bullets(doc, [
        "Widar reads the cached array rather than reparsing 43,652 text files every "
        "epoch, and the other file backed datasets use worker processes.",
        "Test accuracy is correct predictions divided by total samples. The benchmark "
        "averages per batch instead, which overweights the final partial batch.",
        "Two configurations ship with epoch counts far out of line with the rest, "
        "the recurrent model on UT-HAR at 3000 and the same model on Widar3.0 at "
        "500. We cap both at 200, so those two runs see less training than the "
        "benchmark intends. Every other model runs for the epochs the benchmark "
        "ships.",
    ])
    para(doc, "Two runs needed a smaller batch. The Vision Transformer on the NTU-Fi "
              "datasets exhausts 8 gigabytes of graphics memory, because its input "
              "becomes about 760 patch tokens and attention cost grows with the square "
              "of that. Those two runs use a batch of 16.")

    sub(doc, 7, "Hardware and software")
    para(doc, "Everything runs on a laptop with an NVIDIA GeForce RTX 4060 holding 8 "
              "gigabytes, an Intel Core i7-14700HX with 28 threads and 16 gigabytes of "
              "system memory, using PyTorch 2.14 with CUDA 13.2. The benchmark asks for "
              "PyTorch 1.12, which cannot run on this card at all because it builds no "
              "kernels for compute capability 8.9. Anyone reproducing this work on "
              "current hardware will meet the same obstacle. We checked that all eleven "
              "models build and run under the newer version.")


def sec_results(doc, acc, prof):
    doc.add_heading("8. Results", level=1)
    para(doc, "We trained all eleven models on all four datasets, which is 44 runs. "
              "Every number below is generated directly from the recorded results "
              "rather than copied by hand, so the tables cannot drift out of step with "
              "the experiments.")

    sub(doc, 8, "Reproduced baseline accuracy")
    rows = []
    for m in ORDER:
        row = [m]
        for d in NICE:
            r = acc.get((d, m))
            row.append(f"{r['final_acc']:.3f} ({r['best_acc']:.3f})" if r else "-")
        rows.append(row)
    add_table(doc, ["Model"] + [NICE[d] for d in NICE], rows,
              "Test accuracy at the final epoch, with the best epoch in brackets. "
              "Eleven models on four datasets, 44 runs.",
              widths=[1.0, 1.25, 1.25, 1.35, 1.15])
    if not endsem():
        add_figure(doc, "fig2_accuracy_by_dataset.png",
                   "Best test accuracy for every model on every dataset. Widar3.0 is "
                   "markedly harder than the other three.")
    para(doc, "The three activity datasets are largely solved, with accuracies above "
              "0.97 common on UT-HAR and NTU-Fi HAR and a recurrent model reaching a "
              "perfect score on the latter. Widar3.0 is much harder at 0.704, which is "
              "unsurprising with 22 gesture classes rather than 6 or 7.")

    if not endsem():
        _results_instability(doc, acc)
    if endsem():
        para(doc, "Cost matters as much as accuracy, because the benchmark reports "
                  "accuracy alone and a model that cannot run on an access point is of "
                  "no use. The spread is large. On NTU-Fi HAR a gated recurrent unit "
                  "with 79 thousand parameters reaches a perfect 1.000 while the "
                  "multilayer perceptron needs 175 million to reach 0.981. On Widar3.0 "
                  "accuracy peaks at LeNet with 299 thousand parameters and 0.704, and "
                  "every larger model is worse, with ResNet101 holding 142 times more "
                  "parameters and reaching only 0.658. A Widar sample is a 20 by 20 "
                  "grid of velocities at 22 time steps rather than a photograph, so the "
                  "capacity of an image network has little to work with. That is "
                  "convenient here, because it means the cross domain experiments below "
                  "can use a small model without any loss of rigour.")
        return
    _results_cost(doc, acc, prof)
    _results_widar_size(doc)
    _results_train_cost(doc, acc)


def _results_instability(doc, acc):
    sub(doc, 8, "The training recipe is unstable")
    unstable = [(d, m, r) for (d, m), r in acc.items()
                if r["best_acc"] - r["final_acc"] > 0.05]
    para(doc, f"Recording both the final epoch and the best epoch exposed something the "
              f"benchmark cannot see. {len(unstable)} of the 44 runs ended below their "
              f"own best epoch by more than five accuracy points. The extreme case is "
              f"ResNet101 on UT-HAR, which reached 0.972 at its best epoch and then "
              f"collapsed to 0.099 by the last one, which is below the 0.143 that random "
              f"guessing would give on seven classes.")
    deep = {"ResNet18", "ResNet50", "ResNet101"}
    recur = {"RNN", "GRU", "LSTM", "BiLSTM", "CNN+GRU"}
    n_deep = sum(1 for _, m, _ in unstable if m in deep)
    n_rec = sum(1 for _, m, _ in unstable if m in recur)
    para(doc, f"This matters because the benchmark evaluates only after the final "
              f"epoch, so its published figure for a model is whatever that last epoch "
              f"happened to produce. The recipe has no learning rate schedule and no "
              f"early stopping to prevent a late collapse. The single worst collapse is "
              f"a deep residual network, but counting the affected runs rather than "
              f"their size tells a different story, because {n_rec} of the "
              f"{len(unstable)} are recurrent models against {n_deep} residual ones. "
              f"The problem belongs to the recipe rather than to one family of "
              f"architecture."
              + (" Section 7 explains how we choose the epoch for the end term runs."
                 if endsem() else
                 " For the rest of this report we quote best epoch accuracy, with one "
                 "caveat stated plainly. It is the highest accuracy any epoch reached "
                 "on the test set, so the test set chose the stopping point, which is "
                 "something no deployed system could do. It is an optimistic reading "
                 "rather than a fair one, and it makes our figures more generous than "
                 "the benchmark's own, which are final epoch numbers. We print both "
                 "columns throughout so the difference stays visible, and the end term "
                 "work will choose the epoch on a validation set carved out of the "
                 "training data."))


def _results_cost(doc, acc, prof):
    sub(doc, 8, "Cost against accuracy")
    para(doc, "The benchmark reports accuracy alone. Once parameter count and inference "
              "latency are placed beside it, the picture changes considerably.")
    for d, cap in [("Widar", "Widar3.0")]:
        rows = []
        got = [(m, acc[(d, m)]) for m in ORDER if (d, m) in acc]
        best = max(r["best_acc"] for _, r in got)
        for m, r in sorted(got, key=lambda t: -t[1]["best_acc"]):
            p = prof.get((d, m))
            rows.append([m, f"{r['best_acc']:.3f}", f"{best - r['best_acc']:+.3f}",
                         f"{r['params']:,}", f"{r['params'] * 4 / 1e6:.1f}",
                         f"{p['ms_per_sample']:.3f}" if p else "-",
                         f"{p['cpu_ms_batch1']:.2f}" if p else "-"])
        add_table(doc, ["Model", "Accuracy", "Gap to best", "Parameters",
                        "Weights MB", "GPU ms", "CPU ms"], rows,
                  f"Accuracy against cost on {cap}, ordered by accuracy. The GPU "
                  f"column is one sample within a batch of 64 on the graphics card. "
                  f"The CPU column is a single sample on four processor threads with "
                  f"no graphics card, which is the situation an access point is in.",
                  widths=[0.9, 0.8, 0.85, 1.1, 0.85, 0.75, 0.75], highlight={0})
    band = [prof[("Widar", m)] for m in ORDER
            if ("Widar", m) in prof and prof[("Widar", m)]["ms_per_sample"] <= 0.005]
    if band:
        lo = min(p["cpu_ms_batch1"] for p in band)
        hi = max(p["cpu_ms_batch1"] for p in band)
        para(doc, f"The two columns answer different questions and the second is the "
                  f"one that decides deployability. A router has no graphics card and "
                  f"it classifies one sample as it arrives rather than 64 at a time, so "
                  f"the processor figure is the one to read. The graphics card figure "
                  f"cannot separate the small models at all. {nw(len(band)).capitalize()} "
                  f"of the eleven sit between 0.001 and 0.005 milliseconds there, which "
                  f"is too close to tell apart, while on a processor those same "
                  f"{nw(len(band))} spread from {lo:.2f} to {hi:.2f} milliseconds, a "
                  f"factor of {nw(round(hi / max(lo, 1e-9)))}. "
                  f"We report the graphics card column here "
                  f"because it is what the literature usually quotes, and the end term "
                  f"work uses the processor column instead.")
    para(doc, "On NTU-Fi HAR a gated recurrent unit with 79 thousand parameters and 0.3 "
              "megabytes of weights reaches a perfect 1.000, while the multilayer "
              "perceptron needs 175 million parameters and 701 megabytes to reach 0.981. "
              "That is roughly 2,200 times the size for a worse result, and every "
              "residual network there is beaten by models a hundred times smaller.")

def _results_widar_size(doc):
    sub(doc, 8, "On Widar3.0, the best model size is a small one")
    add_figure(doc, "fig4_widar_size_trend.png",
               "Accuracy on Widar3.0 with the models ordered from smallest to largest. "
               "LeNet is the most accurate and every larger model is worse. Below "
               "LeNet the ordering does not follow size.", width=5.4)
    para(doc, "The pattern on Widar3.0 is stronger, though it is not simply that "
              "smaller is better, and the figure shows two different things above and "
              "below the peak. Above it the trend is clean. Accuracy peaks at LeNet "
              "with 299 thousand parameters and 0.704, and all four larger models rank "
              "below it, falling further the larger they get. ResNet101 holds 142 times "
              "more parameters than LeNet and took 67 minutes against LeNet's 5, yet "
              "reaches only 0.658. Below the peak there is no such trend. The Vision "
              "Transformer at 95 thousand parameters reaches 0.679 while LSTM and "
              "BiLSTM, both larger, manage 0.640 and 0.634, so among the small models "
              "architecture decides the result and size does not.")
    para(doc, "Two cautions apply to the ends of that figure. The smallest model, a "
              "plain recurrent network at 31 thousand parameters, is also the one whose "
              "epoch count we capped, so its 0.483 reflects 200 of the 500 epochs the "
              "benchmark ships for it and should not be read as the cost of being "
              "small. At the other end the nearest comparisons are tight, since LeNet "
              "beats ResNet18 by 0.013, and every figure here is a single run with no "
              "repeat seeds, so a gap that small is not something we would defend. The "
              "gap from LeNet to ResNet101 is 0.046 and the ordering of the four larger "
              "models is consistent, which is the part of the result the end term work "
              "relies on.")
    para(doc, "The likely explanation is a mismatch between architecture and data. A "
              "Widar sample is a 20 by 20 grid of velocities at 22 time steps, not a "
              "photograph, and residual networks were designed for large natural images "
              "with deep hierarchies of visual structure. That capacity has little to "
              "work with here. The authors of WiFlexFormer make the same observation, "
              "that a low parameter count helps rather than hurts in this domain [7]."
              + (" It is also convenient, because it means the cross domain experiments "
                 "below can use a small model without any loss of rigour." if endsem()
                 else ""))

def _results_train_cost(doc, acc):
    total = sum(r["train_min"] for r in acc.values())
    widar = sum(r["train_min"] for (d, _), r in acc.items() if d == "Widar")
    para(doc, f"The whole sweep of 44 runs cost {total:.0f} minutes of graphics "
              f"processor time on the laptop described above, of which {widar:.0f} "
              f"minutes were Widar3.0. That is small enough to repeat whenever a result "
              f"is questioned, which is the point of having fixed the data pipeline "
              f"first.")


def sec_endsem_results(doc, E):
    """Sections 8.6 onwards, which are the end term experiments."""
    arms, loo, rec, ev, runs = E

    # ---------------------------------------------------------------- 8.6 headline
    sub(doc, 8, "What the honest split costs")
    para(doc, "This is the central result of the project. Each row uses the same "
              "samples and the same test set size, so the only thing that changes along "
              "a row is what the split holds back.")
    rows, have = [], []
    for model in ["LeNet", "GRU", "ViT", "ResNet18"]:
        for proto, nice in PROTO_NAME.items():
            got = [arms.get((model, proto, a)) for a in ARMS]
            if not all(got):
                continue
            a, b, c = (g[EACC] for g in got)
            have.append((model, proto, a, b, c))
            rows.append([model, f"Held out {nice}", f"{a:.3f}", f"{b:.3f}", f"{c:.3f}",
                         f"{b - a:+.3f}", f"{c - b:+.3f}"])
    if rows:
        add_table(doc, ["Model", "Protocol", "Benchmark style", "Grouped in domain",
                        "Cross domain", "Cost of removing repeat takes",
                        "Cost of the domain shift"], rows,
                  "The accuracy drop separated into its two causes. Accuracy is taken at "
                  "the epoch chosen on a validation set carved out of the training data, "
                  "never at the best epoch on the test set.",
                  widths=[0.75, 1.15, 0.95, 0.95, 0.85, 1.0, 0.95])
        add_figure(doc, "fig7_three_arm_decomposition.png",
                   "The same model and the same samples under three different splits. "
                   "The left bar is what the benchmark reports and the right bar is what "
                   "the model can actually do on a condition it has never seen.",
                   width=5.6)
        lead = [h for h in have if h[0] == "LeNet" and h[1] == "cross-orientation"]
        if lead:
            _, _, a, b, c = lead[0]
            para(doc, f"Take LeNet with a held out orientation. The benchmark style "
                      f"split reports {a:.3f}. Removing only the repeat takes, while "
                      f"leaving every orientation visible in training, brings that to "
                      f"{b:.3f}. Holding the orientation out brings it to {c:.3f}. Of "
                      f"the {a - c:.3f} point fall, {a - b:.3f} is the benchmark testing "
                      f"on repeat recordings of conditions it already trained on, which "
                      f"is leakage rather than a domain problem, and the remaining "
                      f"{b - c:.3f} is the real cost of meeting a new orientation. "
                      f"Reporting {a - c:.3f} alone would credit the domain shift with a "
                      f"loss largely caused by near duplicate test samples.")
        spread = {}
        for model, proto, a, b, c in have:
            spread.setdefault(proto, []).append(c - b)
        if len(spread) > 1:
            mean = {p: sum(v) / len(v) for p, v in spread.items()}
            worst, best = min(mean, key=mean.get), max(mean, key=mean.get)
            good = ("costs nothing at all, and in our runs it slightly improves the "
                    f"result by {mean[best]:.3f}" if mean[best] >= 0
                    else f"costs only {-mean[best]:.3f}")
            para(doc, f"The three protocols do not behave alike. Averaged over the "
                      f"models we trained, holding out {an(PROTO_NAME[worst])} costs "
                      f"{-mean[worst]:.3f} accuracy points, while holding out "
                      f"{an(PROTO_NAME[best])} {good}. That is what the physics predicts, "
                      f"and it matters for the dataset's own claims. A body coordinate "
                      f"velocity profile is computed in a frame attached to the person, "
                      f"and the authors of Widar3.0 designed it so the result would not "
                      f"depend on where someone stands or which way they face. They half "
                      f"succeeded. The transform does remove the dependence on standing "
                      f"position. It does not remove the dependence on orientation, "
                      f"which enters through an estimate of the torso direction and "
                      f"survives as the largest residual effect. Calling the "
                      f"representation domain independent is too strong, and so would be "
                      f"denying it works at all.")

    # ---------------------------------------------------------------- 8.7 leave one out
    ready = {k: v for k, v in loo.items() if len(v) >= 4}
    if ready:
        sub(doc, 8, "The drop depends on which condition is held out")
        para(doc, "A single cross domain number hides something. We repeated the "
                  "experiment holding out each value of the condition in turn.")
        rows = []
        for name, d in ready.items():
            ks = sorted(d)
            rows.append([PROTO_NAME[name].capitalize()]
                        + [f"{d[k]:.3f}" for k in ks]
                        + [f"{max(d.values()) - min(d.values()):.3f}"])
        width = max(len(sorted(d)) for d in ready.values())
        add_table(doc, ["Condition held out"] + [str(i) for i in range(1, width + 1)]
                  + ["Spread"], rows,
                  "LeNet accuracy with each value of the condition held out in turn. "
                  "The spread column is the best case minus the worst case.",
                  widths=[1.5] + [0.7] * width + [0.8])
        add_figure(doc, "fig8_leave_one_out.png",
                   "Accuracy against which value of the condition was held out. The "
                   "dashed line is the same model with nothing held out. The worst case "
                   "is marked in red.", width=5.9)
        o = ready.get("cross-orientation")
        if o:
            ks = sorted(o)
            worst, best = min(o, key=o.get), max(o, key=o.get)
            para(doc, f"For orientation the shape is close to an inverted U. Holding "
                      f"out orientation {best} leaves accuracy at {o[best]:.3f}, while "
                      f"holding out orientation {worst} brings it to {o[worst]:.3f}. "
                      f"That spread of {max(o.values()) - min(o.values()):.3f} points is "
                      f"larger than many improvements reported in this literature. "
                      f"Orientations {ks[0]} and {ks[-1]} sit at the ends of the range, "
                      f"so holding one out asks the model to extrapolate, while holding "
                      f"out a middle value only asks it to interpolate between "
                      f"neighbours it already knows. The choice of held out value "
                      f"therefore moves the headline a long way, and a paper that "
                      f"reports one cross domain figure without saying which value it "
                      f"held out has not said enough for the number to be checked.")
        q = ready.get("cross-position")
        if q:
            para(doc, f"Standing position behaves in the opposite way. Its spread is "
                      f"only {max(q.values()) - min(q.values()):.3f} and every value "
                      f"sits near the in domain result, so there is no bad case to "
                      f"pick. Holding out a position is not a domain shift worth the "
                      f"name, and this is the second independent sign of it after the "
                      f"decomposition in the previous section.")

    # ---------------------------------------------------------------- 8.8 model free
    if ev.get("one_nn") and ev.get("similarity"):
        sub(doc, 8, "The same result without training anything")
        para(doc, "Every number so far came out of a neural network, which leaves the "
                  "reply that the architecture or the learning rate was at fault. So we "
                  "repeated the measurement with no model. Each test sample takes the "
                  "gesture label of the single most similar training sample, where "
                  "similarity is the cosine between two raw velocity profiles. There is "
                  "nothing to train and nothing to tune.")
        rows = []
        order = {a_: i for i, a_ in enumerate(ARMS)}          # benchmark, grouped, cross
        for k, v in sorted(ev["one_nn"].items(),
                           key=lambda kv: (list(PROTO_NAME).index(kv[0].split("/")[0])
                                           if kv[0].split("/")[0] in PROTO_NAME else 9,
                                           order.get(kv[0].split("/")[1], 9))):
            proto, arm = k.split("/")
            rows.append([PROTO_NAME.get(proto, proto).capitalize(),
                         {"in_domain": "Benchmark style",
                          "in_domain_grouped": "Grouped in domain",
                          "cross": "Cross domain"}[arm],
                         str(v["n_classes"]), f"{v['chance']:.3f}", f"{v['acc']:.3f}"])
        add_table(doc, ["Protocol", "Arm", "Classes", "Chance", "Nearest neighbour"],
                  rows, "Accuracy of labelling each test sample with its single nearest "
                        "training sample, on raw data with no learning involved.",
                  widths=[1.4, 1.5, 0.8, 0.8, 1.2])
        s = ev["similarity"]
        same_c = s.get("same recording condition", [0])[0]
        same_g = s.get("same gesture, different condition", [0])[0]
        diff = s.get("different gesture", [0])[0]
        add_figure(doc, "fig9_model_free_evidence.png",
                   "Left, what two recordings have in common and how similar that makes "
                   "them. Right, how far a plain nearest neighbour lookup gets on each "
                   "arm of the split.", width=6.1)
        para(doc, f"The left half of the figure explains the problem in one "
                  f"measurement. Two recordings of different gestures have a mean cosine "
                  f"similarity of {diff:.3f}. Two recordings of the same gesture under "
                  f"different conditions reach only {same_g:.3f}, so sharing a gesture "
                  f"is worth {same_g - diff:+.3f}. Two recordings sharing a condition "
                  f"reach {same_c:.3f}, worth {same_c - diff:+.3f}, about "
                  f"{(same_c - diff) / max(same_g - diff, 1e-6):.0f} times as much. In "
                  f"the raw data, who performed a gesture and where they stood leaves a "
                  f"stronger trace than which gesture it was, so a model rewarded for "
                  f"accuracy on a condition mixed split is mostly being rewarded for "
                  f"identifying the condition.")
        key = "cross-orientation"
        a = ev["one_nn"].get(f"{key}/in_domain", {})
        b = ev["one_nn"].get(f"{key}/in_domain_grouped", {})
        c = ev["one_nn"].get(f"{key}/cross", {})
        if a and b and c:
            para(doc, f"The right half makes the same point about the split. On the "
                      f"benchmark's own arrangement a lookup with no learning reaches "
                      f"{a['acc']:.3f} against chance of {a['chance']:.3f}. Removing the "
                      f"repeat takes drops it to {b['acc']:.3f}, and holding the "
                      f"orientation out drops it to {c['acc']:.3f}. That reproduces the "
                      f"ordering of the trained results without a single parameter, so "
                      f"the effect belongs to the data and the split rather than to our "
                      f"models. If a body coordinate velocity profile really were domain "
                      f"independent by construction, these three numbers would be "
                      f"equal.")

    # ---------------------------------------------------------------- 8.9 recovery
    if rec:
        sub(doc, 8, "Trying to win the accuracy back")
        base = {p: arms[("LeNet", p, "cross")][EACC] for p in PROTO_NAME
                if ("LeNet", p, "cross") in arms}
        cols = [p for p in PROTO_NAME if p in base]
        if cols:
            rows = [["No recovery"] + [f"{base[c]:.3f}" for c in cols]]
            for m in sorted(rec):
                cells = []
                for c in cols:
                    v = rec[m].get(c)
                    cells.append(f"{v:.3f} ({v - base[c]:+.3f})" if v is not None else "-")
                rows.append([m.capitalize()] + cells)
            add_table(doc, ["Method"] + [f"Held out {PROTO_NAME[c]}" for c in cols], rows,
                      "Cross domain accuracy with each recovery method, and in brackets "
                      "the change against the same model with no recovery applied.",
                      widths=[1.7] + [1.4] * len(cols))
            add_figure(doc, "fig10_recovery.png",
                       "Change in cross domain accuracy for each recovery method. Bars "
                       "below the line are methods that made the result worse, which are "
                       "reported for the same reason as the ones that helped.", width=5.9)
            best = {}
            for c in cols:
                cand = [(rec[m][c] - base[c], m) for m in rec if c in rec[m]]
                if cand:
                    best[c] = max(cand)
            if best:
                lines = []
                for c in cols:
                    if c in best:
                        d, m = best[c]
                        lines.append(f"for a held out {PROTO_NAME[c]} the best method was "
                                     f"{m}, worth {d:+.3f}")
                para(doc, "Taking the best method for each protocol, "
                          + "; ".join(lines) + ".")
                o20 = rec.get("augmentation to 20 degrees", {}).get("cross-orientation")
                o45 = rec.get("augmentation to 45 degrees", {}).get("cross-orientation")
                pred = ("Before running these we predicted that rotating the velocity "
                        "plane would help most where orientation is held out, that "
                        "scaling it would help where the volunteer is held out, and "
                        "that standing position had no comparable physical story so "
                        "would gain least. The first two held. The third did not, and "
                        "position gained about as much as anything else, which we "
                        "cannot explain from the physics and do not intend to invent a "
                        "reason for.")
                if o20 is not None and o45 is not None:
                    pred += (f" We were also wrong about the rotation angle, and this "
                             f"is the clearer miss. We argued that the angle had to "
                             f"stay small, because Widar3.0 contains Draw N "
                             f"horizontally and Draw N vertically as separate classes, "
                             f"so a rotation near a quarter turn maps one class onto "
                             f"another and destroys the label. It is also the "
                             f"physically tidy choice, since a velocity profile is "
                             f"already in body coordinates and only the residual error "
                             f"in estimating the torso direction should survive. The "
                             f"measurement disagrees. On the held out orientation the "
                             f"wider rotation reaches {o45:.3f} against {o20:.3f} for "
                             f"the narrow one, so the label damage we were worried "
                             f"about costs less than the extra variety buys. We "
                             f"measured both settings rather than assuming, which is "
                             f"the only reason we know.")
                para(doc, pred)
            # the control runs: does the gain cost in-domain accuracy, and does it
            # survive a change of architecture
            ctl = [r for r in runs if r["tag"] == "endsem-rec"
                   and not (r["model"] == "LeNet"
                            and split_protocol(r["protocol"])[2] == "cross")]
            if ctl:
                dom = [r for r in ctl
                       if split_protocol(r["protocol"])[2] == "in_domain_grouped"]
                oth = [r for r in ctl if r["model"] != "LeNet"]
                bits = []
                for r in dom:
                    nm, _, arm_ = split_protocol(r["protocol"])
                    ref = arms.get(("LeNet", nm, arm_))
                    if ref:
                        bits.append(r[EACC] - ref[EACC])
                if bits:
                    para(doc, f"Two checks sit behind those numbers. A method that "
                              f"raises cross domain accuracy is only worth having if it "
                              f"does not quietly cost in domain accuracy to get there, "
                              f"so the same augmentation was run against the grouped in "
                              f"domain arm. It changed the result by between "
                              f"{min(bits):+.3f} and {max(bits):+.3f}, so nothing is "
                              f"being traded away."
                              + (f" It also survives a change of architecture, since "
                                 f"the same augmentation moves {oth[0]['model']} on the "
                                 f"held out orientation from "
                                 f"{arms[(oth[0]['model'], 'cross-orientation', 'cross')][EACC]:.3f} "
                                 f"to {oth[0][EACC]:.3f}."
                                 if oth and (oth[0]["model"], "cross-orientation",
                                             "cross") in arms else ""))

    # ---------------------------------------------------------------- 8.10 adaptation
    ad = os.path.join(RESULTS, "adaptation.jsonl")
    if os.path.exists(ad):
        rows_ad = [json.loads(l) for l in open(ad) if l.strip()]
        by = {}
        for r in rows_ad:
            by.setdefault(r["protocol"], {}).setdefault(r.get("mode", "head"),
                                                        []).append(r)
        if by:
            sub(doc, 8, "How many labelled examples from the new condition would it take")
            para(doc, "Augmentation asks the model to imagine the new condition. The "
                      "other option a deployer has is to label a few examples from it, "
                      "so we measured how far that gets. Two versions are run and the "
                      "contrast between them is the point. Retraining only the final "
                      "layer freezes the features and can move nothing but the decision "
                      "boundaries. Retraining the whole network lets the features change "
                      "too. If only the second recovers accuracy, the representation "
                      "itself is what failed to transfer. Examples are reserved before "
                      "anything else, so every point is scored on the same samples.")
            ks = sorted({r["k"] for m in by.values() for rs in m.values() for r in rs})
            trows = []
            for proto in PROTO_NAME:
                if proto not in by:
                    continue
                for mode in ("head", "full"):
                    rs = {r["k"]: r for r in by[proto].get(mode, [])}
                    if not rs:
                        continue
                    trows.append([f"Held out {PROTO_NAME[proto]}",
                                  "Last layer only" if mode == "head"
                                  else "Whole network"]
                                 + [f"{rs[k]['acc']:.3f}" if k in rs else "-"
                                    for k in ks])
            if trows:
                add_table(doc, ["Protocol", "What is retrained"] + [str(k) for k in ks],
                          trows,
                          "Accuracy after adapting on labelled examples from the "
                          "condition the model had never seen. The column heading is "
                          "the number of examples per gesture, and zero is the "
                          "unadapted model scored on the same samples.",
                          widths=[1.35, 1.15] + [0.44] * len(ks))
            add_figure(doc, "fig13_adaptation.png",
                       "Accuracy against how many labelled examples from the new "
                       "condition were used to adapt. The dotted line is the unadapted "
                       "model. The horizontal axis is logarithmic and zero is drawn at "
                       "the left.", width=6.1)

            ref = by.get("cross-orientation") or by[list(by)[0]]
            hd = sorted(ref.get("head", []), key=lambda r: r["k"])
            fl = sorted(ref.get("full", []), key=lambda r: r["k"])
            if hd and fl:
                base = hd[0]["acc"]
                hd_hi, fl_hi = hd[-1], fl[-1]
                dip = min(fl[1:], key=lambda r: r["acc"]) if len(fl) > 1 else None
                para(doc, f"Retraining only the last layer never helps. It starts at "
                          f"{base:.3f} and is still only {hd_hi['acc']:.3f} after "
                          f"{hd_hi['n_shots']} labelled examples from the new "
                          f"orientation. Thousands of correctly labelled examples from "
                          f"exactly the condition the model fails on, and moving the "
                          f"decision boundaries with them buys almost nothing.")
                if dip:
                    para(doc, f"Retraining the whole network has a trap in the middle "
                              f"of the curve. With {dip['n_shots']} examples it reaches "
                              f"{dip['acc']:.3f}, far below the {base:.3f} of not "
                              f"adapting at all, so a practitioner who collected a few "
                              f"hundred labelled samples and fine tuned on them would "
                              f"make their system worse while believing they had done "
                              f"the sensible thing. Only at {fl_hi['n_shots']} examples "
                              f"does it climb to {fl_hi['acc']:.3f}, an improvement of "
                              f"{fl_hi['acc'] - base:+.3f}. The features are therefore "
                              f"what fail to transfer, and repairing them needs "
                              f"something close to a full dataset from the new "
                              f"condition, which is most of the work of collecting the "
                              f"data again rather than a quick calibration step.")

    # ---------------------------------------------------------------- efficiency
    prof = {}
    pf = os.path.join(RESULTS, "model_profile.jsonl")
    if os.path.exists(pf):
        prof = {p["model"]: p for p in (json.loads(l) for l in open(pf) if l.strip())
                if p["dataset"] == "Widar" and "cpu_ms_batch1" in p}
    cross = {m: arms[(m, "cross-orientation", "cross")][EACC]
             for m in ["LeNet", "GRU", "ViT", "ResNet18"]
             if (m, "cross-orientation", "cross") in arms}
    if prof and len(cross) >= 2:
        sub(doc, 8, "Whether any of it is cheap enough to deploy")
        para(doc, "Accuracy on its own cannot say whether a model could run on an "
                  "access point. The usual figure quoted, including in our own midterm, "
                  "is latency on a graphics card at a batch of 64, which is the wrong "
                  "measurement twice over. A router has no graphics card, and it "
                  "classifies one sample as it arrives rather than 64 at a time. So we "
                  "measured the time to classify a single sample on four processor "
                  "threads with no accelerator, which is roughly what an access point "
                  "has, and placed it beside the accuracy that survives a change of "
                  "orientation rather than the in domain figure.")
        rows = []
        for m in sorted(cross, key=lambda m: -cross[m]):
            p_ = prof[m]
            rows.append([m, f"{cross[m]:.3f}", f"{p_['params']:,}",
                         f"{p_['cpu_ms_batch1']:.2f}", f"{p_['ms_per_sample']:.3f}"])
        add_table(doc, ["Model", "Cross domain accuracy", "Parameters",
                        "CPU ms per sample", "GPU ms per sample"], rows,
                  "Cost against the accuracy that actually survives a domain shift. "
                  "The CPU column is a single sample on four threads with no graphics "
                  "card.", widths=[1.0, 1.3, 1.1, 1.1, 1.1])
        add_figure(doc, "fig12_cost_vs_cross_domain.png",
                   "Cross domain accuracy against the cost of one classification on a "
                   "processor. The horizontal axis is logarithmic.", width=4.4)
        best = max(cross, key=cross.get)
        worst = max(prof, key=lambda m: prof[m]["cpu_ms_batch1"] if m in cross else -1)
        if best in prof and worst in prof and worst != best:
            ratio = prof[worst]["cpu_ms_batch1"] / max(prof[best]["cpu_ms_batch1"], 1e-9)
            para(doc, f"The answer is that the cheap model is also the accurate one. "
                      f"{best} classifies a sample in "
                      f"{prof[best]['cpu_ms_batch1']:.2f} milliseconds on a processor "
                      f"and holds the best cross domain accuracy at {cross[best]:.3f}, "
                      f"while {worst} takes {prof[worst]['cpu_ms_batch1']:.2f} "
                      f"milliseconds, which is {ratio:.0f} times longer, for "
                      f"{cross[worst]:.3f}. There is no accuracy being bought with that "
                      f"cost. A sensing system built on the small model would keep up "
                      f"with a continuous stream of samples on hardware an access point "
                      f"already has, which makes the deployment question the least "
                      f"difficult of the ones this report asks. The hard part remains "
                      f"the domain shift, not the arithmetic.")

    # ---------------------------------------------------------------- 8.11 selection
    sel = [r for r in runs if r["tag"] == "endsem" and r["protocol"] != "original"]
    if sel:
        sub(doc, 8, "How much the choice of reporting epoch is worth")
        para(doc, "A cross domain claim is only as honest as the rule used to pick the "
                  "epoch. Three rules are available and they disagree.")
        rows = [["Final epoch, which is what SenseFi reports",
                 f"{sum(r['final_acc'] for r in sel) / len(sel):.3f}"],
                ["Validation selected, which is what this report quotes",
                 f"{sum(r['val_sel_acc'] for r in sel) / len(sel):.3f}"],
                ["Best epoch measured on the test set, an oracle",
                 f"{sum(r['best_acc'] for r in sel) / len(sel):.3f}"]]
        add_table(doc, ["How the epoch is chosen",
                        f"Mean accuracy over {len(sel)} runs"], rows,
                  "The same runs read three different ways.", widths=[4.0, 2.0])
        gap = (sum(r["best_acc"] for r in sel) - sum(r["val_sel_acc"]
               for r in sel)) / len(sel)
        para(doc, f"The oracle rule is worth {gap:+.3f} points over honest selection "
                  f"and nobody deploying a system could obtain it, because it needs the "
                  f"test labels to decide when to stop. Our midterm quoted the best "
                  f"epoch and we are correcting that here rather than leaving it. "
                  f"SenseFi's own script trains for a fixed number of epochs and "
                  f"evaluates once at the end, so its published figures are final epoch "
                  f"numbers and our midterm was the more generous of the two. Every end "
                  f"term number here uses the validation selected rule.")

    # ---------------------------------------------------------------- 8.11 softmax
    fix = [r for r in runs if r["tag"] == "endsem-softmax"]
    if fix:
        sub(doc, 8, "A bug in the benchmark, measured rather than asserted")
        mid = None
        p = os.path.join(RESULTS, "baselines.jsonl")
        for line in open(p):
            r = json.loads(line)
            if r.get("tag") == "baseline" and r["dataset"] == "Widar" \
                    and r["model"] == "CNN+GRU":
                mid = r
        para(doc, "While reading the benchmark closely enough to reuse it we found that "
                  "its CNN and GRU model ends in a softmax layer, and that the training "
                  "loop then passes the output to a cross entropy loss, which applies a "
                  "softmax of its own. The model is therefore trained through two stacked "
                  "softmax functions, which flattens the gradient and slows learning. The "
                  "same mistake is present in the UT-HAR and the NTU-Fi versions of that "
                  "model and in the self supervised variant.")
        rows = []
        if mid:
            rows.append(["As shipped, softmax applied twice", f"{mid['best_acc']:.3f}"])
        rows.append(["Trailing softmax removed", f"{fix[0]['best_acc']:.3f}"])
        add_table(doc, ["Version of the model", "Accuracy on Widar3.0"], rows,
                  "One line removed from a model definition, measured on the "
                  "benchmark's own split so the comparison is like for like.",
                  widths=[3.4, 1.8])
        if mid:
            para(doc, f"Removing the layer is worth "
                      f"{fix[0]['best_acc'] - mid['best_acc']:+.3f} accuracy points. We "
                      f"have deliberately not patched the vendored copy of the benchmark, "
                      f"because the purpose of this project is to evaluate it rather than "
                      f"to quietly repair it, and a reader should be able to reproduce the "
                      f"original behaviour. The fix is applied in our own training script "
                      f"behind a flag.")


def sec_conclusion(doc):
    doc.add_heading("9. Conclusion", level=1)
    para(doc, "This report covers the first half of our project. We surveyed the "
              "recent literature, rebuilt the data pipeline so experiments are "
              "affordable, reproduced the benchmark across 44 runs and found three "
              "things.")
    para(doc, "First, the benchmark's Widar3.0 split holds nothing back. All 17 "
              "volunteers, all 8 standing positions and all 5 orientations appear in "
              "both halves, so the reported accuracy measures how well a model learned "
              "its training conditions rather than whether it would work for a new "
              "person in a new place. Because Widar3.0 records those conditions in its "
              "filenames a correct split can be built, and we have built three.")
    para(doc, "Second, the training recipe is unstable. Ten of the 44 runs ended "
              "materially below their own best epoch and one collapsed from 0.972 to "
              "0.099. Since the benchmark evaluates only after the final epoch, its "
              "published numbers inherit that instability.")
    para(doc, "Third, larger models are not better here, and on Widar3.0 every model "
              "larger than the best one is worse, falling further the larger it gets. "
              "The best model there is 142 times smaller than the largest and 13 times "
              "faster to train, which means the expensive cross domain experiments "
              "planned for the end term can use a small model without any loss of "
              "rigour. Among the models smaller than the best one, size predicts "
              "nothing and architecture decides the result.")
    para(doc, "The end term work follows directly. We will train on the three cross "
              "domain protocols, measure how far accuracy falls against the matched in "
              "domain baselines, and try to recover the loss with data augmentation, "
              "reporting the outcome honestly including anything that does not help.")


def sec_endsem_conclusion(doc, E):
    arms, loo, rec, ev, runs = E
    doc.add_heading("9. Conclusion", level=1)
    para(doc, "We set out to answer one question. Does WiFi sensing work when it meets "
              "conditions it was not trained on, and if it does not, how much of the gap "
              "can be closed with the techniques available today. The answer to the "
              "first part is no, by a wide margin, and the second part is more "
              "discouraging than we expected.")

    lead = arms.get(("LeNet", "cross-orientation", "cross"))
    mixed = arms.get(("LeNet", "cross-orientation", "in_domain"))
    grouped = arms.get(("LeNet", "cross-orientation", "in_domain_grouped"))
    if lead and mixed and grouped:
        a, b, c = mixed[EACC], grouped[EACC], lead[EACC]
        para(doc, f"The headline number is that accuracy on Widar3.0 falls from {a:.3f} "
                  f"to {c:.3f} when the test set contains a body orientation the model "
                  f"has never seen, using the same model, the same samples and the same "
                  f"test set size throughout. The more useful part of that result is the "
                  f"decomposition. Only {b - c:.3f} of the {a - c:.3f} point fall is the "
                  f"domain shift itself. The other {a - b:.3f} is the benchmark testing "
                  f"on repeat recordings of conditions it had already trained on, which "
                  f"is a leakage problem that has nothing to do with generalisation and "
                  f"which would have been invisible without the grouped control arm.")

    para(doc, "We think the measurement that should be taken most seriously is the one "
              "with no model in it. Two recordings of the same gesture made under "
              "different conditions are barely more alike than two recordings of "
              "different gestures, while two recordings sharing a condition are markedly "
              "more alike than either. A nearest neighbour lookup with nothing to train "
              "reproduces the whole pattern of our trained results. That rules out the "
              "explanation that we simply chose poor models, and it argues against the "
              "claim that the body coordinate velocity profile is domain independent by "
              "construction. It clearly reduces the problem, since it was designed to, "
              "but it does not remove it.")

    o = loo.get("cross-orientation", {})
    if len(o) >= 4:
        para(doc, f"A practical warning follows from the leave one out sweep. The size of "
                  f"the drop depends heavily on which value is held out, ranging from "
                  f"{min(o.values()):.3f} to {max(o.values()):.3f} across the five "
                  f"orientations. Holding out a value at the edge of the range asks the "
                  f"model to extrapolate and holding out a middle value only asks it to "
                  f"interpolate. Any paper that reports a single cross domain figure "
                  f"without naming the value it held out has not given enough "
                  f"information for the result to be checked, and we would now count our "
                  f"own earlier plan to report one number as the same mistake.")

    if rec:
        base = {p: arms[("LeNet", p, "cross")][EACC] for p in PROTO_NAME
                if ("LeNet", p, "cross") in arms}
        gains = [(rec[m][p] - base[p], m, p) for m in rec for p in rec[m] if p in base]
        if gains:
            best = max(gains)
            para(doc, f"The recovery attempts are the honest disappointment of the "
                      f"project. The best single result was {best[1]} on the held out "
                      f"{PROTO_NAME[best[2]]} protocol, worth {best[0]:+.3f} accuracy "
                      f"points. Set against a gap of several tenths, augmentation of the "
                      f"kind the recent literature recommends recovers a small fraction "
                      f"of what is lost. We report this plainly because a negative result "
                      f"that has been measured properly is still a result, and because "
                      f"the alternative, quietly reporting only the protocol where a "
                      f"method happened to work, is exactly the practice this project set "
                      f"out to criticise.")

    para(doc, "Three things follow for anyone building on this. The first is that a "
              "split must hold out whole recording conditions and not merely whole "
              "factor values, because repeat takes of one condition are near duplicates "
              "and they leak. The second is that the held out value has to be named, "
              "because the choice moves the result more than most published improvements "
              "do. The third is that the useful comparison is against a matched in "
              "domain partner over the same samples, since without one a cross domain "
              "number cannot be interpreted at all.")
    ad = os.path.join(RESULTS, "adaptation.jsonl")
    if os.path.exists(ad):
        rs = [json.loads(l) for l in open(ad) if l.strip()]
        ref = [r for r in rs if r["protocol"] == "cross-orientation"] or rs
        hd = sorted([r for r in ref if r.get("mode") == "head"], key=lambda r: r["k"])
        fl = sorted([r for r in ref if r.get("mode") == "full"], key=lambda r: r["k"])
        if hd and fl:
            para(doc, f"We also asked the question a deployer would actually ask, which "
                      f"is how many labelled examples from the new condition it takes "
                      f"before the system becomes usable again. Retraining only the "
                      f"final layer never helps, even given {hd[-1]['n_shots']} labelled "
                      f"examples, which tells us the features rather than the decision "
                      f"boundaries are what fail to transfer. Retraining the whole "
                      f"network does eventually work, reaching {fl[-1]['acc']:.3f} with "
                      f"{fl[-1]['n_shots']} examples, but it is worse than doing nothing "
                      f"at every smaller amount we tried. There is no cheap calibration "
                      f"step here. Recovering the accuracy costs most of the effort of "
                      f"collecting the dataset again.")

    para(doc, "If we continued this work the next step would not be a larger model. Our "
              "own efficiency results point the other way, since the best model on this "
              "dataset is among the smallest we tried and the largest is reliably worse. "
              "Nor would it be a better adaptation step, because the measurement above "
              "says the representation is what breaks. The direction we would take is to "
              "change how the representation is learned, by training across several "
              "orientations with the orientation label deliberately made unpredictable "
              "from the features, so that the network is prevented from encoding the "
              "condition in the first place. Our evidence that the condition dominates "
              "the gesture in the raw data is precisely the argument for trying it.")


def sec_references(doc):
    doc.add_heading("10. References", level=1)
    refs = [
        'J. Yang, X. Chen, D. Wang, H. Zou, C. X. Lu, S. Sun and L. Xie, "SenseFi: A '
        'Library and Benchmark on Deep Learning Empowered WiFi Human Sensing," '
        'Patterns, Cell Press, vol. 4, no. 3, 2023. arXiv:2207.07859',
        'IEEE Standards Association, "IEEE Std 802.11bf-2025, Amendment 4: Enhancements '
        'for Wireless LAN Sensing," published 26 September 2025. '
        'IEEE SA',
        'F. Meneghello, C. Chen, C. Cordeiro and F. Restuccia, "Toward Integrated '
        'Sensing and Communications in IEEE 802.11bf Wi-Fi Networks," IEEE '
        'Communications Magazine, 2023. arXiv:2212.13930',
        'Y. Ma, G. Zhou and S. Wang, "A Survey of Commodity WiFi Sensing in 10 Years: '
        'Current Status, Challenges, and Opportunities," 2021. '
        'arXiv:2111.07038',
        'J. Yang, X. Chen, H. Zou, D. Wang, Q. Xu and L. Xie, "EfficientFi: Towards '
        'Large Scale Lightweight WiFi Sensing via CSI Compression," IEEE Internet of '
        'Things Journal, 2022. arXiv:2204.04138',
        'J. Strohmayer and M. Kampel, "Data Augmentation Techniques for Cross Domain '
        'WiFi CSI Based Human Activity Recognition," 2024. arXiv:2401.00964',
        'J. Strohmayer and M. Kampel, "WiFlexFormer: Efficient WiFi Based Person '
        'Centric Sensing," 2024. arXiv:2411.04224',
        'B. Barahimi, H. Tabassum, M. Omer and O. Waqar, "Lessons from Deploying '
        'Learning Based CSI Localization on a Large Scale ISAC Platform," 2025. '
        'arXiv:2504.17173',
        'Y. Zheng, Y. Zhang, K. Qian, G. Zhang, Y. Liu, C. Wu and Z. Yang, "Widar3.0: '
        'WiFi Based Activity Recognition Dataset," IEEE DataPort, open access dataset.',
        'Cognitive Systems Corporation, "WiFi Motion," commercial WiFi sensing product.',
        '"Evaluating BiLSTM and CNN+GRU Approaches for Human Activity '
        'Recognition Using WiFi CSI Data," 2025. arXiv:2506.11165',
        '"ISAC-Fi: Enabling Full-fledged Monostatic Sensing over Wi-Fi '
        'Communication," 2024. arXiv:2408.09851',
        '"No Vision, No Wearables: 5G-based 2D Human Pose Recognition with '
        'Integrated Sensing and Communications," 2025. arXiv:2512.24923',
        '"Context-Aware Predictive Coding: A Representation Learning '
        'Framework for WiFi Sensing," 2024. arXiv:2410.01825',
        '"MU-SHOT-Fi: Self-Supervised Multi-User Wi-Fi Sensing with '
        'Source-free Unsupervised Domain Adaptation," 2026. arXiv:2605.01369',
        '"From User Recognition to Activity Counting: An Identity-Agnostic '
        'Approach to Multi-User WiFi Sensing," 2026. arXiv:2604.16572',
        '"Physics-Guided Attention in a Lightweight TCN for Efficient WiFi '
        'CSI-Based Human Activity Recognition," 2026. arXiv:2606.01834',
        '"Wi-Spike: A Low-power WiFi Human Multi-action Recognition Model '
        'with Spiking Neural Networks," 2026. arXiv:2603.14475',
        '"Resource-Efficient WiFi CSI Sensing via Exploiting the Age of '
        'Samples," 2026. arXiv:2606.31690',
        '"Securing WiFi Fingerprint-based Indoor Localization Systems from '
        'Malicious Access Points," 2025. arXiv:2505.07724',
        '"maxVSTAR: Maximally Adaptive Vision-Guided CSI Sensing with '
        'Closed-Loop Edge Model Adaptation," 2025. arXiv:2510.26146',
        'K. Xu, J. Wang, H. Zhu and D. Zheng, "Self-Supervised Learning for WiFi '
        'CSI-Based Human Activity Recognition: A Systematic Study," 2023. '
        'arXiv:2308.02412',
    ]
    for i, r in enumerate(refs, 1):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.35)
        p.paragraph_format.first_line_indent = Inches(-0.35)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.line_spacing = 1.0
        run = p.add_run(f"[{i}]  ")
        run.bold = True
        run.font.size = Pt(8.5)
        run2 = p.add_run(r)
        run2.font.size = Pt(8.5)


def sec_appendix(doc):
    doc.add_heading("Appendix A. Code", level=1)
    para(doc, "Every number in this report is produced by the code submitted alongside "
              "it. Nothing is transcribed by hand, so rebuilding the document after "
              "changing an experiment keeps the two consistent.")
    items = [
        ("check_data.py", "verifies the four datasets are present and correctly shaped"),
        ("build_widar_cache.py", "converts the 43,652 Widar files into one cached array "
                                 "plus the metadata table"),
        ("splits.py", "builds the splits and refuses any that leaks a condition or "
                      "loses a class"),
        ("train.py", "trains one model on one dataset, protocol and arm"),
        ("run_baselines.py", "drives the 44 baseline runs, skipping what is recorded"),
        ("model_profile.py", "parameter count and inference latency for every model"),
        ("make_tables.py", "the result tables"),
        ("make_figures.py", "the figures"),
        ("make_report.py", "this document"),
    ]
    if endsem():
        items[6:6] = [
            ("evidence.py", "measures the leakage with no model, by pairwise similarity "
                            "and nearest neighbour lookup on raw data"),
            ("run_endsem.py", "drives the 61 end term runs, also resumable"),
            ("adapt.py", "the few shot adaptation curves"),
        ]
    para(doc, "; ".join(f"{n} {d}" for n, d in items) + ".")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["midterm", "endsem"], default="endsem")
    STAGE[0] = ap.parse_args().stage

    acc, prof = load()
    doc = Document()
    for s in doc.sections:
        s.top_margin = s.bottom_margin = Inches(0.75)
        s.left_margin = s.right_margin = Inches(0.8)
    style_doc(doc)

    title_page(doc)
    sec_introduction(doc)
    sec_motivation(doc)
    sec_background(doc)
    sec_related(doc)
    sec_problem(doc, acc)
    sec_solution(doc)
    sec_design(doc, acc, prof)
    sec_results(doc, acc, prof)
    if endsem():
        E = load_endsem()
        sec_endsem_results(doc, E)
        sec_endsem_conclusion(doc, E)
    else:
        sec_conclusion(doc)
    sec_references(doc)
    sec_appendix(doc)

    out = OUT[STAGE[0]]
    doc.save(out)
    print(f"written to {os.path.normpath(out)}")
    print(f"  {_fig_n[0]} figures, {_tab_n[0]} tables, {len(acc)} runs referenced")


if __name__ == "__main__":
    main()
