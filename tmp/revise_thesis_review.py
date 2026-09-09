from pathlib import Path

root = Path(__file__).resolve().parents[1] / 'docs/thesis'

def replace(text, old, new):
    assert old in text, old[:100]
    return text.replace(old, new)

p = root / 'chapter6_experimental_evaluation.tex'
s = p.read_text(encoding='utf-8')
s = s.replace('% Existing references.bib entries are used without modification.\n', '')
a = s.index('The literature-oriented subset')
b = s.index('The causal short suite', a)
s = s[:a] + r'''The offline motion-distribution subset freezes four music identities per genre.
Each identity is paired with its lexicographically first preflight-passed
ground-truth motion. Five seeds yield 200 reconstructed outputs, evaluated as
five separate sets of 40 against the same 40 references. These outputs use
preanalysed musical timing and trace-based SMPL reconstruction; they characterise
motion features and do not measure online beat delivery or control deadlines.
The evaluation excludes six seconds of warm-up and reconstructs the following
20 seconds at 60 frames per second. The deployed catalogue is used without a
leave-one-music-out exclusion, so this is not another held-out retrieval test.

''' + s[b:]
a = s.index('The causal formal experiments are serial')
b = s.index('The recorded\nplatform', a)
s = s[:a] + r'''Table~\ref{tab:ch6_protocol_matrix} states the experimental units and run counts.
The 135 formal runs are exactly $13\times2\times5+1\times1\times5$.
The separate 120 Hz capacity pilot comprises 18 short runs: three inputs
(stitched AIST++, silence/recovery, and a 90-to-150 BPM tempo jump), two
conditions (full and authored timing), and seeds 0--2. Each uses 60 seconds of
input, including six seconds of warm-up. Three full-system runs of the repeated
stitched input use the same seeds and 306 seconds each, including warm-up.
The 21-run denominator excludes the ten-second smoke check.

\begin{table}[tb]
\centering\small
\begin{tabular}{p{0.29\linewidth}p{0.36\linewidth}rr}
\toprule
Evaluation & Inputs and conditions & Seeds & Count \\
\midrule
Offline retrieval & 60 held-out music identities & -- & 60 \\
Offline rejection & 30 recordings; overlapping windows & -- & 750 \\
Offline SMPL features & 40 music/reference pairs; reconstructed output & 0--4 & 200 \\
120 Hz pilot, 60 s & 3 inputs; full and authored & 0--2 & 18 \\
120 Hz pilot, 306 s & 1 stitched input; full & 0--2 & 3 \\
60 Hz formal, 40 s & 13 inputs; full & 0--4 & 65 \\
60 Hz formal, 40 s & Same 13 inputs; authored & 0--4 & 65 \\
60 Hz formal, 606 s & 1 stitched input; full & 0--4 & 5 \\
\bottomrule
\end{tabular}
\caption{Evaluation matrix. Counts are queries, windows, reconstructed outputs,
or runtime runs as indicated. Runtime durations include six seconds of warm-up.
Only the final three rows constitute the 135 formal runs.}
\label{tab:ch6_protocol_matrix}
\end{table}

The runtime experiments are serial and headless, with a six-second matching
window, one-second matching interval, collision checking and final-output
auditing enabled. The pilot tests the 120 Hz design target stated in
Section~\ref{sec:intro_problem}: control-work p99 below 8.33 ms and a
deadline-miss ratio below 1\% for each run. Only six of its 21 runs pass all
gates; the worst p99 is 15.572 ms and the worst miss ratio is 0.18544.
The formal operating point was consequently set to 60 Hz before the formal
matrix was executed. Its p99 budget is 16.67 ms; the 1\% miss threshold and
final-output constraints remain the same. This is a reduced operating target,
not evidence that the initial 120 Hz requirement was met. Different source
coverage and durations also prevent treating the pilot and formal matrix as
a paired estimate of the effect of frequency. Their results are not pooled.
''' + s[b:]
s = s.replace('librosa 0.10.2.post1,', r'librosa 0.10.2.post1 \cite{mcfee2024librosa},')
a = s.index('Every supplemental run preserves')
b = s.index(r'\section{Evaluation Metrics}', a)
s = s[:a] + '''Every formal run preserves its command, source identity, seed, trace,
timing report, final-pose archive and log. The audit checks the declared run
matrix, artifact hashes, strictly increasing output clocks and consistent robot
identity. All 538 frozen input-file hashes matched during the post-run audit,
and the formal run index contained no failed attempts. Source-file hashes
supplement the Git commit identifier to capture uncommitted implementation
changes. Pilot and formal artifacts have separate manifests.

''' + s[b:]
s = replace(s, 'averages with its missingness retained. Preanalysed replay uses its original beat\nrecords and is reported separately.', 'averages with its missingness retained.')
a = s.index('Kinetic and geometric FID/Div')
b = s.index('Robot-domain diversity additionally', a)
s = s[:a] + r'''Fr\'echet Inception Distance (FID) compares Gaussian approximations to reference
and output feature distributions \cite{heusel2017fid}. Following the
AIST++/FACT and Bailando evaluation convention \cite{li2021ai,siyao2022bailando},
this chapter uses motion descriptors in place of image Inception features.
The subscripts $k$ and $g$ identify kinetic and geometric feature spaces:
the pinned AIST++ extractor returns 72 kinetic components (horizontal and
vertical kinetic energy and an acceleration-based expenditure statistic for
each of 24 joints) and 32 geometric components summarising body-configuration
predicates. The extractor is fixed to
\texttt{google/aistplusplus\_api} commit
\texttt{2dd7b3e946b794fd0081c98e2e2433545abf8b87}; local changes concern imports
and formatting only.

For feature space $h\in\{k,g\}$, let $\boldsymbol{\mu}_{R,h}$ and
$\boldsymbol{\Sigma}_{R,h}$ be the reference sample mean and covariance, and
let $\boldsymbol{\mu}_{O,h}$ and $\boldsymbol{\Sigma}_{O,h}$ be their output
counterparts. The reported distance is
\begin{equation}
\begin{split}
\mathrm{FID}_h={}&\|\boldsymbol{\mu}_{R,h}-\boldsymbol{\mu}_{O,h}\|_2^2\\
&+\operatorname{tr}\!\left(\boldsymbol{\Sigma}_{R,h}
+\boldsymbol{\Sigma}_{O,h}
-2\left(\boldsymbol{\Sigma}_{R,h}^{1/2}\boldsymbol{\Sigma}_{O,h}
\boldsymbol{\Sigma}_{R,h}^{1/2}\right)^{1/2}\right).
\end{split}
\label{eq:ch6_fid}
\end{equation}
Lower FID indicates closer feature distributions, not necessarily better
musical correspondence. Diversity (Div) is the mean Euclidean distance among
all distinct output feature-vector pairs, in the same evaluation family
\cite{siyao2022bailando}:
\begin{equation}
\mathrm{Div}_h=\frac{2}{n(n-1)}\sum_{a<b}
\|\mathbf{z}_{a,h}-\mathbf{z}_{b,h}\|_2,\qquad n=40.
\label{eq:ch6_div}
\end{equation}
Thus each seed uses all $\binom{40}{2}=780$ unordered pairs, with the reference
Div calculated identically on the 40 ground-truth items. Larger Div means
greater spread; it is interpreted relative to reference spread rather than
maximised without regard to plausibility.

Both metrics use reconstructed AIST++/SMPL motion, not G1 joint angles.
Raw results use the extractor's native scale. Standardised results transform
both sets using reference means and population standard deviations, with
each denominator floored at $10^{-8}$. FID itself uses sample covariances.
The kinetic covariance is necessarily rank deficient with only 40 samples in
72 dimensions; the implementation uses a symmetric eigendecomposition with
negative numerical eigenvalues clipped to zero. Finite-sample FID can be biased
\cite{chong2020fid}, so neither this stabilisation nor averaging five seeds
establishes comparability with larger published benchmarks.

''' + s[b:]
s = replace(s, 'The old cohort lacks such frame clocks, so its nominal-rate derivatives are\ndiagnostics only. They are not reconstructed physical-time measurements.\n', '')
s = replace(s, "\\cite{tseng2023edge}. The old COM/foot-speed formula is retained only as a legacy\nproxy. The G1 adaptation follows the official evaluator's algebra on a", "\\cite{tseng2023edge}. The G1 adaptation follows the official evaluator's algebra on a")
s = replace(s, 'identities}\n\\label{tab:ch6_genres}', r'''identities. Codes: BR = breaking; HO = house; JB = ballet jazz;
JS = street jazz; KR = krump; LH = LA-style hip-hop; LO = locking;
MH = middle hip-hop; PO = popping; WA = waacking
\cite{tsuchida2019aist,li2021ai}}
\label{tab:ch6_genres}''')
s = s.replace('cohorts.}', 'the three formal conditions.}')
s = replace(s, 'and measurement instruments are matched. Comparing the original replay cohort\nwith the causal cohort would simultaneously change beat information, output-clock\nmeasurement and safety-audit overhead.', 'and measurement instruments are matched. The offline reconstructed feature\nsubset supplies a separate description of motion spread and is not included\nin this timing comparison.')
s = s.replace('The corrected pipeline measures', 'The output pipeline measures')
s = replace(s, 'The supplemental final-output audit performs an additional clearance check,\nincluding forward kinematics. Its overhead is reported separately and remains\ninside total control work. Thus differences from the old headless measurements\ncannot be attributed solely to causal beat analysis.', 'The final-output audit performs an additional clearance check, including\nforward kinematics. Its overhead is reported separately and remains inside\ntotal control work.')
a = s.index('The replay ablations cover')
b = s.index(r'\begin{table}', a)
s = s[:a] + '''The formal ablation compares full beat-synchronised control with authored
timing on the same 13 inputs and five matched seeds. Authored timing disables
beat-driven retiming while retaining retrieval, motion selection, transitions,
collision checking and output constraints. The formal evaluation does not
isolate the contributions of retrieval scoring, diversity selection, entry
search or output limiting; those component benefits remain unestablished.

''' + s[b:]
a = s.index('Table~\\ref{tab:ch6_ablation} evaluates')
b = s.index('Conditions were executed serially', a)
s = s[:a] + r'''Table~\ref{tab:ch6_ablation} reports paired timing comparisons after averaging
matched seeds within each source. The 65 source/seed combinations are not
treated as 65 independent musical inputs. The displayed tests retain Holm
correction over the full set of recorded condition-by-metric tests in the
short suite, rather than recomputing correction over only the displayed rows.
There is no corrected evidence of a BAS difference. Absence of significance
is not proof of equivalence. The source-mean deadline-miss ratio differs from
full by \ChSixMissDifference{} (adjusted $p=\ChSixMissHolm{}$), and both
conditions pass the absolute 60 Hz gates. RQ5 is therefore answered for the
timing condition and sustained operation, with the other component effects
left open.
''' + s[b:]
s = replace(s, 'They do not rescue the separate 120 Hz pilot, whose 15/21 failures show\nthat the original operating target is not reliable on this platform.', 'The separate 120 Hz pilot has 15/21 failures, so the initial design target\nremains unmet on this platform.')
s = replace(s, 'statistics delimit the component claims that can be made. Replay ablations and\ncausal timing ablations answer different experimental questions.', 'statistics delimit the timing claim; the other component contributions have\nnot been isolated by this formal matrix.')
p.write_text(s, encoding='utf-8')

p = root / 'chapter7_conclusions_future_work.tex'
s = p.read_text(encoding='utf-8').replace('% Historical DSP-only, smoke and nominal-clock reports are not pooled with it.\n','')
s = replace(s, 'is a contribution of the work, but the replay ablations do not establish a\nstatistically supported benefit for every component after multiplicity\ncorrection.', 'is a contribution of the work, but the formal experiment isolates only the\nfull versus authored-timing comparison. It does not establish a separate\nbenefit for retrieval scoring, selection policy or entry search.')
s = replace(s, 'final poses and actual output clocks. It also separates older preanalysed\nfile replay from the causal supplement, which contains 130 formal short runs\nand five long runs.', 'final poses and actual output clocks. The formal matrix in\nTable~\\ref{tab:ch6_protocol_matrix} contains 130 matched short runs and five\nlong runs.')
s = replace(s, 'causal long runs repeat only one stitched source. The causal supplement also\nretests only full versus authored timing, so the complete replay ablation matrix\ncannot be treated as causal online validation.', 'causal long runs repeat only one stitched source. The formal ablation covers\nfull versus authored timing; causal component tests for retrieval, diversity\nand transition entry remain future work.')
s = s.replace('1.000 for house to 0.000 for ballet jazz, street jazz and\nkrump', '1.000 for house (HO) to 0.000 for ballet jazz (JB), street jazz (JS) and\nkrump (KR), using the genre key in Table~\\ref{tab:ch6_genres},')
s = s.replace('while also identifying an unmet 120 Hz target.', 'while leaving the 120 Hz design target in\nSection~\\ref{sec:intro_problem} unmet.')
p.write_text(s, encoding='utf-8')

p = root / 'scripts/export_chapter6.py'
s = p.read_text(encoding='utf-8')
s = s.replace("    old = subset('preanalysed_replay', 'full')\n", '')
s = s.replace("groups = [('Replay full', old), ('Causal full', full), ('Causal authored', authored), ('Causal long', long)]", "groups = [('Causal full', full), ('Causal authored', authored), ('Causal long', long)]")
s = s.replace("key = 'preanalysed_replay' if label.startswith('Replay') else 'online_causal'", "key = 'online_causal'")
s = s.replace("suite = 'full' if label.startswith('Replay') else ('long' if label.endswith('long') else 'short')", "suite = 'long' if label.endswith('long') else 'short'")
s = '\n'.join(line for line in s.split('\n') if "('preanalysed_replay','ablation'" not in line)
s = s.replace('groups[1:3]', 'groups[:2]').replace('groups[1:]', 'groups')
s = s.replace("ax.set_xticks(range(1,5),['Replay\\nfull','Causal\\nfull','Causal\\nauthored','Causal\\nlong'])", "ax.set_xticks(range(1,4),['Causal\\nfull','Causal\\nauthored','Causal\\nlong'])")
p.write_text(s, encoding='utf-8')
print('Revised evaluation, conclusions and exporter.')
