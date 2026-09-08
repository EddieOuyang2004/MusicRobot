"""Generate the English thesis architecture as matching vector PDF and SVG."""
from pathlib import Path
from html import escape
from math import atan2, cos, sin

from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor


OUT = Path(__file__).resolve().parents[1] / "figures"
STEM = "chapter3_end_to_end_architecture"
W, H = 1000, 950
INK, MUTED = "#203348", "#526477"
BLUE, TEAL, AMBER = "#32679B", "#247B78", "#A36A24"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUT / (STEM + ".pdf")), pagesize=(W, H))
    
    c.setAuthor("MusicRobot thesis")
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
           
           '<desc>Offline audio and motion preparation supplies a frozen catalogue. Online retrieval, preparation and beat-synchronous commitment feed continuous G1 kinematic playback.</desc>']

    def rect(x, y, w, h, fill, stroke, radius=10):
        c.setFillColor(HexColor(fill))
        c.setStrokeColor(HexColor(stroke))
        c.setLineWidth(1)
        c.roundRect(x, H-y-h, w, h, radius, fill=1, stroke=1)
        svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}"/>')

    def text(x, y, value, size=14, color=INK, bold=False, anchor="start"):
        c.setFillColor(HexColor(color))
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        getattr(c, "drawCentredString" if anchor == "middle" else "drawString")(x, H-y, value)
        svg.append(f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" text-anchor="{anchor}" fill="{color}">{escape(value)}</text>')

    def arrow(points, color=MUTED, dashed=False):
        c.setStrokeColor(HexColor(color))
        c.setLineWidth(1.7)
        c.setDash([5, 4] if dashed else [])
        p = c.beginPath()
        p.moveTo(points[0][0], H-points[0][1])
        for x, y in points[1:]:
            p.lineTo(x, H-y)
        c.drawPath(p)
        c.setDash([])
        x, y = points[-1]
        px, py = points[-2]
        a = atan2(y-py, x-px)
        head = [(x, y), (x-8*cos(a)+3.5*sin(a), y-8*sin(a)-3.5*cos(a)),
                (x-8*cos(a)-3.5*sin(a), y-8*sin(a)+3.5*cos(a))]
        p = c.beginPath()
        p.moveTo(head[0][0], H-head[0][1])
        for hx, hy in head[1:]:
            p.lineTo(hx, H-hy)
        p.close()
        c.setFillColor(HexColor(color))
        c.drawPath(p, fill=1, stroke=0)
        dash = ' stroke-dasharray="5 4"' if dashed else ''
        svg.append(f'<polyline points="{" ".join(f"{x},{y}" for x,y in points)}" fill="none" stroke="{color}" stroke-width="1.7"{dash}/>')
        svg.append(f'<polygon points="{" ".join(f"{x},{y}" for x,y in head)}" fill="{color}"/>')

    def box(x, y, title, lines, color, h=82, fill="#FFFFFF"):
        rect(x, y, 200, h, fill, color)
        text(x+100, y+26, title, 15, color, True, "middle")
        for i, line in enumerate(lines):
            text(x+100, y+48+i*18, line, 13, INK, anchor="middle")

    rect(0, 0, W, H, "#FFFFFF", "#FFFFFF", 0)
    
    

    rect(20, 78, 960, 280, "#F3F7FB", "#CFDCE9", 14)
    text(40, 106, "01  OFFLINE PREPARATION", 17, BLUE, True)
    text(610, 106, "Whole-clip analysis and catalogue admission", 13, MUTED)
    box(40, 130, "AIST++ music", ["Paired audio tracks", "Track and segment identity"], BLUE)
    box(280, 130, "Audio descriptors", ["Embedding + rhythm / timbre", "Optional style activations"], BLUE)
    box(520, 130, "Searchable audio bank", ["Segments linked to tracks", "Reference normalisation"], BLUE)
    box(40, 250, "SMPL motion", ["Paired human trajectories", "Source motion identity"], BLUE)
    box(280, 250, "Motion analysis", ["Key poses and motion profiles", "Activity and regularity"], BLUE)
    box(520, 250, "GMR to Unitree G1", ["Canonical named-joint motion", "Kinematic preflight"], BLUE)
    rect(760, 161, 200, 143, "#DFECF8", BLUE)
    text(860, 190, "FROZEN CATALOGUE", 15, BLUE, True, "middle")
    for i, line in enumerate(["Audio index + statistics", "Motion profiles + key poses", "Validated G1 trajectories", "Joined IDs + provenance"]):
        text(860, 215+i*21, line, 13, INK, anchor="middle")
    for y in [171, 291]:
        arrow([(240,y),(280,y)], BLUE)
        arrow([(480,y),(520,y)], BLUE)
    arrow([(720,171),(740,171),(740,202),(760,202)], BLUE)
    arrow([(720,291),(740,291),(740,265),(760,265)], BLUE)

    rect(20, 393, 960, 306, "#F1F8F6", "#C9DFD9", 14)
    text(40, 423, "02  ONLINE PERCEPTION AND DECISION", 17, TEAL, True)
    text(650, 423, "Causal audio and bounded background work", 13, MUTED)
    box(40, 450, "Live audio", ["Microphone or file stream", "Rolling audio buffer"], TEAL)
    box(280, 450, "Query features", ["Matched offline extractor", "Rolling-window descriptors"], TEAL)
    box(520, 450, "Hierarchical retrieval", ["Segments to tracks to motions", "Similarity + compatibility"], TEAL)
    box(760, 450, "Stable selection", ["Evidence and hysteresis", "Relevance / diversity policy"], TEAL)
    for x in [240,480,720]:
        arrow([(x,491),(x+40,491)], TEAL)
    box(40, 584, "Rhythm analysis", ["Short-block onset / activity", "Accepted beat events"], TEAL)
    box(280, 584, "Commit + entry search", ["Beat / readiness gate", "Entry pose + blend duration"], TEAL)
    box(520, 584, "Ready motion pool", ["Completed candidates only", "Bounded cache"], TEAL)
    box(760, 584, "Candidate preparation", ["Load, ground and adapt", "Build entry-state features"], TEAL)
    arrow([(140,532),(140,584)], TEAL)
    arrow([(860,532),(860,584)], TEAL)
    text(870,562,"Pending ID",11,TEAL)
    arrow([(760,625),(720,625)], TEAL)
    arrow([(520,625),(480,625)], TEAL)
    arrow([(240,625),(280,625)], TEAL)

    # The immutable catalogue supplies search data and motion assets separately.
    arrow([(805,304),(805,377),(620,377),(620,450)], BLUE, True)
    text(465,380,"Index + profiles",12,BLUE)
    arrow([(960,232),(990,232),(990,625),(960,625)], BLUE, True)

    rect(20, 744, 960, 180, "#FCF8F0", "#E7D8BF", 14)
    text(280,774,"03  CONTINUOUS PLAYBACK AND OUTPUT",17,AMBER,True)
    box(40, 793, "Beat-phase playback", ["Active / target sampling", "Gradual speed / phase control"], AMBER)
    box(280, 793, "Continuous transition", ["Root alignment + pose blend", "Atomic state handover"], AMBER)
    box(520, 793, "Output conditioning", ["Stateful joint limits", "Optional collision handling"], AMBER)
    box(760, 793, "Unitree G1 / MuJoCo", ["29-DoF pose output", "Kinematic preview"], AMBER)
    for x in [240,480,720]:
        arrow([(x,834),(x+40,834)], AMBER)
    arrow([(380,666),(380,722),(200,722),(200,793)], TEAL)
    text(397,725,"Committed motion, entry phase and transition duration",12,TEAL)
    arrow([(100,666),(100,793)], TEAL, True)
    text(38,719,"Beat events",11,TEAL)
    
    c.save()
    svg.append("</svg>")
    (OUT / (STEM + ".svg")).write_text("\n".join(svg), encoding="utf-8")


if __name__ == "__main__":
    main()
