from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path


OUTPUT = Path(__file__).with_name("music_feature_motion_mapping.xlsx")

HEADERS = [
    "Feature",
    "Pipeline / Source",
    "Current Use",
    "How It Can Further Influence Motion",
    "Implementation Status",
]

ROWS = [
    ["Tempo BPM / beat times", "Legacy offline: librosa.beat.beat_track", "Generates beat-aligned phase and open/clap keypoints for the trajectory.", "Drive global playback speed, align major poses to beats, and choose slower, heavier gestures for low BPM versus lighter gestures for high BPM.", "Already used"],
    ["Phase", "Legacy CSV and realtime controller", "Represents normalized progress through the motion cycle; clap/open poses are sampled from it.", "Use phase to place secondary gestures at predictable beat subdivisions, such as half-beat nods or quarter-beat sways.", "Already used"],
    ["RMS loudness / amp_scale / rms_norm", "Legacy RMS and realtime microphone block RMS", "Scales yaw and pitch amplitude; realtime silence/noise gating reduces amplitude to zero.", "Map louder sections to larger reach, stronger clap closure, faster pose transitions, or broader body sway.", "Already used"],
    ["Onset envelope / onset_strength / energy_delta", "Legacy librosa onset_strength and realtime high-pass energy delta", "Strong onsets create short yaw/pitch accent pulses and contribute to realtime accent strength.", "Trigger sharp micro-gestures, hits, nods, or percussive closures; onset strength can control accent duration and sharpness.", "Already used"],
    ["PLP beat events / is_beat", "Realtime librosa.beat.plp pulse peaks", "Triggers phase correction, beat keypoint alignment, tempo updates, and accent starts.", "Use beat events to schedule pose locks, footstep-like shifts, or clap contact frames exactly on musical beats.", "Already used"],
    ["Beat period / estimated BPM", "Realtime BeatIntervalEstimator", "Updates target phase rate so the loop retimes to live music.", "Control motion cycle duration, transition speed, and gesture selection by tempo bands.", "Already used"],
    ["Beat confidence", "Realtime PLP peak height and prominence", "Raises phase-correction gain and accent strength for more reliable beats.", "Modulate how strongly the robot commits to a beat: high confidence gives crisp corrections; low confidence keeps motion smoother.", "Already used"],
    ["Brightness / spectral centroid", "Realtime FFT centroid normalized by Nyquist", "Slightly increases accent sensitivity through the brightness scale.", "Make bright music produce sharper, higher, more staccato gestures; darker music can produce rounder and slower motion.", "Partly used"],
    ["Low-frequency energy", "Realtime FFT band energy, 20-250 Hz", "Computed and shown in status output, but not directly mapped to motion.", "Increase heavy body sway, clap force, grounding, or low-frequency pose amplitude when bass is strong.", "Available, not mapped"],
    ["Mid-frequency energy", "Realtime FFT band energy, 250-4000 Hz", "Computed and shown in status output, but not directly mapped to motion.", "Use as the main musical-body signal for melody/vocal-driven amplitude or expressive pose shaping.", "Available, not mapped"],
    ["High-frequency energy", "Realtime FFT band energy, above 4000 Hz", "Computed and shown in status output, but not directly mapped to motion.", "Drive small fast accents, wrist-like flicks, sharper clap release, or increased motion texture.", "Available, not mapped"],
    ["Spectral rolloff", "Realtime 85% spectral energy frequency", "Computed and shown in status output, but not directly mapped to motion.", "Use as a robust brightness measure for choosing soft versus sharp accent profiles.", "Available, not mapped"],
    ["Zero-crossing rate", "Realtime microphone waveform sign changes", "Computed and shown in status output, but not directly mapped to motion.", "Add controlled tremble or jitter for noisy/percussive textures, with strict smoothing and limits.", "Available, not mapped"],
    ["Spectral contrast", "Realtime librosa.feature.spectral_contrast", "Computed and cached, but not directly mapped to motion.", "Use higher contrast for cleaner, more decisive hits; lower contrast for blended, fluid movement.", "Available, not mapped"],
    ["MFCC 1 and MFCC 2", "Realtime librosa.feature.mfcc", "Computed and cached, but not directly mapped to motion.", "Classify broad timbre states and switch between gesture styles, such as percussive, vocal-like, or ambient.", "Available, not mapped"],
    ["Rhythm density", "Realtime onset peak count per second", "Computed and shown in status output, but not directly mapped to motion.", "Increase gesture complexity when density is high; keep sparse music simple and spacious.", "Available, not mapped"],
    ["Tempo stability", "Realtime variation of recent beat intervals", "Computed and shown in status output, but not directly mapped to motion.", "Use stable tempo for stronger phase locking; reduce correction and smooth more when tempo is unstable.", "Available, not mapped"],
    ["Offbeat ratio", "Realtime onset phase relative to recent beat intervals", "Computed and shown in status output, but not directly mapped to motion.", "Add syncopated secondary motion when offbeat activity is high, such as counter-sways or delayed accents.", "Available, not mapped"],
    ["Close envelope", "Legacy generated clap motion", "Shapes the open-to-closed clap pitch curve over each phase cycle.", "Can be modulated by loudness or onset strength to make claps close more tightly on energetic sections.", "Already used"],
    ["Keypoint label / beat weight", "Generated CSV metadata", "Marks important open/clap poses for beat-aligned realtime playback.", "Let more important poses receive stronger correction while lighter poses remain fluid.", "Already used"],
    ["Music active / noise gate", "Realtime RMS compared with calibrated noise floor", "Disables beat detection and smooths amplitude to zero when input is below the gate.", "Use inactive state to return to neutral pose, idle motion, or low-power safe posture.", "Already used"],
]


def xml_escape(value: str) -> str:
    return html.escape(value, quote=False)


def col_name(index: int) -> str:
    result = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result


def ref(row: int, col: int) -> str:
    return f"{col_name(col)}{row}"


def shared_strings(data: list[list[str]]) -> tuple[list[str], dict[str, int]]:
    strings: list[str] = []
    index: dict[str, int] = {}
    for row in data:
        for value in row:
            if value not in index:
                index[value] = len(strings)
                strings.append(value)
    return strings, index


def shared_strings_xml(strings: list[str]) -> str:
    items = []
    for value in strings:
        items.append(f'<si><t xml:space="preserve">{xml_escape(value)}</t></si>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="{len(strings)}" uniqueCount="{len(strings)}">'
        f'{"".join(items)}</sst>'
    )


def worksheet_xml(data: list[list[str]], string_index: dict[str, int]) -> str:
    widths = [30, 38, 62, 78, 24]
    cols = "".join(
        f'<col min="{i}" max="{i}" width="{width}" customWidth="1"/>'
        for i, width in enumerate(widths, start=1)
    )
    row_parts = []
    for row_num, row in enumerate(data, start=1):
        row_style = 1 if row_num == 1 else 0
        height = 30 if row_num == 1 else 72
        cells = []
        for col_num, value in enumerate(row, start=1):
            cells.append(f'<c r="{ref(row_num, col_num)}" s="{row_style}" t="s"><v>{string_index[value]}</v></c>')
        row_parts.append(f'<row r="{row_num}" ht="{height}" customHeight="1">{"".join(cells)}</row>')
    last = ref(len(data), len(data[0]))
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheetPr><outlinePr summaryBelow="1" summaryRight="1"/></sheetPr>
  <dimension ref="A1:{last}"/>
  <sheetViews>
    <sheetView tabSelected="1" workbookViewId="0" showGridLines="0">
      <pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>
      <selection pane="bottomLeft" activeCell="A2" sqref="A2"/>
    </sheetView>
  </sheetViews>
  <sheetFormatPr defaultRowHeight="15"/>
  <cols>{cols}</cols>
  <sheetData>{"".join(row_parts)}</sheetData>
  <autoFilter ref="A1:{last}"/>
  <pageMargins left="0.7" right="0.7" top="0.75" bottom="0.75" header="0.3" footer="0.3"/>
</worksheet>'''


def styles_xml() -> str:
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="2">
    <font><sz val="10"/><color theme="1"/><name val="Calibri"/><family val="2"/></font>
    <font><b/><sz val="10"/><color rgb="FFFFFFFF"/><name val="Calibri"/><family val="2"/></font>
  </fonts>
  <fills count="3">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF1F4E78"/><bgColor indexed="64"/></patternFill></fill>
  </fills>
  <borders count="2">
    <border><left/><right/><top/><bottom/><diagonal/></border>
    <border>
      <left style="thin"><color rgb="FFD9E2EC"/></left>
      <right style="thin"><color rgb="FFD9E2EC"/></right>
      <top style="thin"><color rgb="FFD9E2EC"/></top>
      <bottom style="thin"><color rgb="FFD9E2EC"/></bottom>
      <diagonal/>
    </border>
  </borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="2">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0" applyBorder="1" applyAlignment="1"><alignment vertical="top" wrapText="1"/></xf>
    <xf numFmtId="0" fontId="1" fillId="2" borderId="1" xfId="0" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment horizontal="center" vertical="center" wrapText="1"/></xf>
  </cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
  <dxfs count="0"/>
  <tableStyles count="0" defaultTableStyle="TableStyleMedium2" defaultPivotStyle="PivotStyleLight16"/>
</styleSheet>'''


def write_xlsx() -> None:
    data = [HEADERS] + ROWS
    strings, string_index = shared_strings(data)
    files = {
        "[Content_Types].xml": '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
  <Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>''',
        "_rels/.rels": '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>''',
        "xl/workbook.xml": '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <fileVersion appName="xl" lastEdited="7" lowestEdited="7" rupBuild="23426"/>
  <workbookPr defaultThemeVersion="166925"/>
  <bookViews><workbookView xWindow="0" yWindow="0" windowWidth="28800" windowHeight="16800"/></bookViews>
  <sheets><sheet name="Feature Motion Map" sheetId="1" r:id="rId1"/></sheets>
  <calcPr calcId="191029"/>
</workbook>''',
        "xl/_rels/workbook.xml.rels": '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" Target="sharedStrings.xml"/>
</Relationships>''',
        "xl/styles.xml": styles_xml(),
        "xl/sharedStrings.xml": shared_strings_xml(strings),
        "xl/worksheets/sheet1.xml": worksheet_xml(data, string_index),
        "docProps/core.xml": '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>Music Feature Motion Mapping</dc:title><dc:creator>Codex</dc:creator><cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">2026-06-24T00:00:00Z</dcterms:created><dcterms:modified xsi:type="dcterms:W3CDTF">2026-06-24T00:00:00Z</dcterms:modified>
</cp:coreProperties>''',
        "docProps/app.xml": '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Microsoft Excel</Application><DocSecurity>0</DocSecurity><ScaleCrop>false</ScaleCrop>
  <HeadingPairs><vt:vector size="2" baseType="variant"><vt:variant><vt:lpstr>Worksheets</vt:lpstr></vt:variant><vt:variant><vt:i4>1</vt:i4></vt:variant></vt:vector></HeadingPairs>
  <TitlesOfParts><vt:vector size="1" baseType="lpstr"><vt:lpstr>Feature Motion Map</vt:lpstr></vt:vector></TitlesOfParts>
</Properties>''',
    }
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, content in files.items():
            zf.writestr(name, content)


def verify_package() -> None:
    with zipfile.ZipFile(OUTPUT, "r") as zf:
        names = set(zf.namelist())
        required = {
            "[Content_Types].xml",
            "_rels/.rels",
            "xl/workbook.xml",
            "xl/_rels/workbook.xml.rels",
            "xl/worksheets/sheet1.xml",
            "xl/styles.xml",
            "xl/sharedStrings.xml",
        }
        missing = required - names
        if missing:
            raise RuntimeError(f"Missing XLSX parts: {sorted(missing)}")
        for name in required:
            ET.fromstring(zf.read(name))
        sheet = zf.read("xl/worksheets/sheet1.xml").decode("utf-8")
    row_count = len(re.findall(r"<row ", sheet))
    cell_count = len(re.findall(r"<c ", sheet))
    expected_rows = len(ROWS) + 1
    expected_cells = expected_rows * len(HEADERS)
    if row_count != expected_rows or cell_count != expected_cells:
        raise RuntimeError(f"Unexpected dimensions: rows={row_count}, cells={cell_count}")


if __name__ == "__main__":
    write_xlsx()
    verify_package()
    print(OUTPUT.resolve())
