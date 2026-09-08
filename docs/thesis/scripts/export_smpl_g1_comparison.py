"""Assemble rendered same-frame pairs with English labels and provenance."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.colors import HexColor

ROOT=Path(__file__).resolve().parents[3]
TMP=ROOT/'tmp/pdfs/smpl_g1'
OUT=ROOT/'docs/thesis/figures'
STEM='chapter4_smpl_g1_same_frame'
W,H=990,800


def main():
    meta=json.loads((TMP/'provenance.json').read_text())
    c=canvas.Canvas(str(OUT/(STEM+'.pdf')),pagesize=(W,H))
    c.setTitle('SMPL-to-G1 Retargeting: Same-Frame Comparison')
    def text(x,y,s,size=14,bold=False,color='#203348'):
        c.setFillColor(HexColor(color)); c.setFont('Helvetica-Bold' if bold else 'Helvetica',size)
        c.drawString(x,H-y,s)
    frames=meta['frames_zero_based']
    for col,frame in enumerate(frames):
        x=120+col*285
        text(x+35,35,f'Frame {frame}  |  {frame/meta["fps"]:.2f} s',15,True)
        for row,kind in enumerate(['smpl','g1']):
            a=np.load(TMP/f'{kind}_{frame}.npy')
            c.drawImage(ImageReader(Image.fromarray(a)),x,H-(50+row*353)-326,width=285,height=326)
    text(25,77,'SOURCE',12,True,color='#32679B')
    text(25,98,'SMPL',18,True,color='#32679B')
    text(25,119,'24 joints',12)
    text(25,430,'TARGET',12,True,color='#247B78')
    text(25,451,'Unitree G1',16,True,color='#247B78')
    text(25,472,'29 DoF',12)
    text(28,770,'Motion: '+meta['source_motion_id']+'  |  60 fps  |  Zero-based frame indices',13)
    c.save()
    (OUT/(STEM+'.json')).write_text(json.dumps(meta,indent=2))


if __name__=='__main__': main()
