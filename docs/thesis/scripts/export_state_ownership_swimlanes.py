"""English logical-owner swimlanes; a representative asynchronous switch."""
from pathlib import Path
from html import escape
from math import atan2, cos, sin
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor

OUT = Path(__file__).resolve().parents[1] / "figures"
STEM = "chapter3_state_ownership_swimlanes"
W, H = 1190, 960
INK, MUTED = "#203348", "#596A7B"
BLUE, TEAL, GOLD = "#32679B", "#247B78", "#A36A24"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUT / (STEM + ".pdf")), pagesize=(W, H))
    c.setTitle("State Ownership and Asynchronous Execution")
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
           '<title>State Ownership and Asynchronous Execution</title>',
           '<desc>Six logical-owner swimlanes show background retrieval and preparation while the foreground continues playback and output, followed by a prepared switch and atomic handover.</desc>']

    def rect(x,y,w,h,fill,stroke,r=8):
        c.setFillColor(HexColor(fill)); c.setStrokeColor(HexColor(stroke)); c.setLineWidth(1)
        c.roundRect(x,H-y-h,w,h,r,fill=1,stroke=1)
        svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}"/>')

    def text(x,y,s,size=13,color=INK,bold=False,center=False):
        c.setFillColor(HexColor(color)); c.setFont("Helvetica-Bold" if bold else "Helvetica",size)
        (c.drawCentredString if center else c.drawString)(x,H-y,s)
        svg.append(f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" text-anchor="{"middle" if center else "start"}" fill="{color}">{escape(s)}</text>')

    def line(points,color=MUTED,dash=False,head=True):
        c.setStrokeColor(HexColor(color)); c.setLineWidth(1.5); c.setDash([5,4] if dash else [])
        p=c.beginPath(); p.moveTo(points[0][0],H-points[0][1])
        for x,y in points[1:]: p.lineTo(x,H-y)
        c.drawPath(p); c.setDash([])
        d=' stroke-dasharray="5 4"' if dash else ''
        svg.append(f'<polyline points="{" ".join(f"{x},{y}" for x,y in points)}" fill="none" stroke="{color}" stroke-width="1.5"{d}/>')
        if head:
            x,y=points[-1]; px,py=points[-2]; a=atan2(y-py,x-px)
            pts=[(x,y),(x-7*cos(a)+3*sin(a),y-7*sin(a)-3*cos(a)),(x-7*cos(a)-3*sin(a),y-7*sin(a)+3*cos(a))]
            p=c.beginPath(); p.moveTo(x,H-y)
            for hx,hy in pts[1:]: p.lineTo(hx,H-hy)
            p.close(); c.setFillColor(HexColor(color)); c.drawPath(p,fill=1,stroke=0)
            svg.append(f'<polygon points="{" ".join(f"{x},{y}" for x,y in pts)}" fill="{color}"/>')

    xs=[40,230,420,610,800,990]
    colors=[BLUE,BLUE,BLUE,TEAL,TEAL,GOLD]
    def event(lane,y,title,body=(),h=56):
        x=xs[lane]; color=colors[lane]
        rect(x,y,160,h,"#FFFFFF",color)
        text(x+80,y+22,title,13,color,True,True)
        for i,s in enumerate(body): text(x+80,y+41+17*i,s,12,INK,center=True)

    rect(0,0,W,H,"#FFFFFF","#FFFFFF",0)
    
    
    rect(40,80,540,29,"#EAF1F8","#CFDCE9")
    text(310,100,"AUDIO INGEST AND BACKGROUND EXECUTORS",13,BLUE,True,True)
    rect(610,80,540,29,"#EAF5F1","#CADFD8")
    text(880,100,"FOREGROUND LOOP: THREE LOGICAL OWNERS",13,TEAL,True,True)
    owners=[("Audio source",["Rolling buffer", "Audio / feature frames"]),
            ("Retrieval worker",["In-flight Future", "Completed ranking", "Rejection status"]),
            ("Preparation pool",["Load / prepare Futures", "Ready cache", "Failed candidate IDs"]),
            ("Selection policy",["Evidence / pending ID", "Hold and recency", "Selection reason"]),
            ("Playback / transition",["Phase / beat history", "Active + target state", "Blend progress"]),
            ("Output layer",["Last emitted pose", "Limiter state", "Collision correction"])]
    for i,(title,lines) in enumerate(owners):
        x=xs[i]
        rect(x-7,116,174,804,"#F8FAFC" if i<3 else "#F7FAF8","#E1E7EB")
        rect(x,124,160,97,"#EAF1F8" if i<3 else "#EDF5F1",colors[i])
        text(x+80,145,title,13,colors[i],True,True)
        for j,s in enumerate(lines): text(x+80,166+j*17,s,12,INK,center=True)
        line([(x+80,225),(x+80,910)],"#C3CED6",True,False)

    # The playback and output lanes keep making progress during worker jobs.
    for y in [239,321,403,485]:
        event(4,y,"Advance active motion",["Beat / phase update"],54)
        event(5,y,"Condition and emit",["Update limiter state"],54)
        line([(960,y+27),(990,y+27)],GOLD)
    event(0,239,"Audio snapshot",["Copy recent window"],54)
    line([(200,266),(230,266)],BLUE)
    event(1,239,"Submit if idle",["Extract query features", "Rank tracks / motions"],84)
    text(310,340,"Busy: skip submission",11,BLUE,center=True)
    event(1,354,"Future completed",["Ranking or rejection"],56)
    line([(390,382),(610,382)],BLUE,True)
    text(500,371,"poll(): done only",11,BLUE,center=True)
    event(3,354,"Observe result",["Update selection state"],56)
    line([(690,410),(690,430),(500,430),(500,450)],BLUE)
    text(595,421,"Ranked candidate IDs",11,BLUE,center=True)
    event(2,450,"Prepare candidates",["Load, ground, adapt", "Build entry features"],78)
    event(2,554,"Publish readiness",["Ready or failed IDs"],56)
    line([(580,582),(610,582)],BLUE,True)
    event(3,554,"Eligible selection",["Policy + ready candidate"],56)
    event(4,554,"Accepted beat",["Eligible switch boundary"],56)
    line([(690,610),(690,699),(800,699)],TEAL)
    text(695,648,"Selection intent",11,TEAL)
    line([(880,610),(880,662)],TEAL)
    text(890,643,"Timing gate",11,TEAL)
    event(4,662,"Commit prepared plan",["Entry pose + duration", "Current plan / generation"],74)
    event(2,630,"Background entry score",["Ready set + state snapshot", "Generation-tagged plan"],74)
    line([(500,610),(500,630)],BLUE)
    line([(500,704),(500,730),(800,730)],BLUE,True)
    text(545,749,"Completed plan; reject stale generation",11,BLUE)
    event(4,760,"Blend source / target",["Root alignment + phase"],56)
    event(5,760,"Condition and emit",["Limits + optional collision"],56)
    line([(960,788),(990,788)],GOLD)
    event(4,844,"Atomic handover",["Target becomes active", "Sampler, phase, root, ID"],70)
    event(5,844,"Continue output",["Preserve limiter history"],56)
    line([(960,872),(990,872)],GOLD)
    line([(880,736),(880,760)],TEAL)
    line([(880,816),(880,844)],TEAL)

    
    c.save()
    svg.append("</svg>")
    (OUT/(STEM+".svg")).write_text("\n".join(svg),encoding="utf-8")


if __name__ == "__main__":
    main()
