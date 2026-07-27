# -*- coding: utf-8 -*-
"""五个评估专家（Skill）的 prompt 定义——本项目的核心 IP。

设计（ADR-0002 D6 详尽版）：
  ① 维度定义与边界 → ② 失真检查清单（含"长什么样"）→ ③ 强制分析流程
  → ④ 分级标准（每级配该维度下的具体表现）→ ⑤ 输出契约（JSON）
拼装方式借鉴 JiuGong RichPromptBuilder：共享外壳 + 维度章节。
prompt 中绝不出现数据集名称/文件名/ID/MOS（ADR-0001 §4）。
"""

# ---------- 共享章节 ----------

_PROCEDURE = """\
[Assessment procedure — follow strictly]
1. Scan the whole image for a first impression.
2. Inspect each aspect in the checklist below, one by one.
3. Identify the DOMINANT quality issue(s) for your dimension.
4. Judge their severity objectively.
5. Map your judgment to a level, then give a precise score within that level's band."""

_OUTPUT_CONTRACT = """\
[Output format — strict]
Reply with JSON only, no other text:
{"level": <int 1-5>, "score": <float>, "reason": "<= 25 words, key evidence only"}"""

_SCALE_BLOCK = {
    "koniq": """\
[Score scale]
The final score is a float in [1, 5]. Level-to-band guide:
  5 Excellent -> 4.3-5.0 | 4 Good -> 3.5-4.2 | 3 Fair -> 2.5-3.4 | 2 Poor -> 1.5-2.4 | 1 Bad -> 1.0-1.4
Use the FULL range; do not cluster scores near the middle.""",
    "spaq": """\
[Score scale]
The final score is a float in [0, 10]. Level-to-band guide:
  5 Excellent -> 8.5-10 | 4 Good -> 6.5-8.4 | 3 Fair -> 4.5-6.4 | 2 Poor -> 2.5-4.4 | 1 Bad -> 0-2.4
Use the FULL range; do not cluster scores near the middle.""",
}

# ---------- 五个专家定义 ----------

SKILLS = {
    "S-TECH": {
        "name": "Technical Quality Assessor",
        "dimension": (
            "You assess ONLY the technical fidelity of the image: sharpness, noise, exposure, "
            "white balance, and digital artifacts. Ignore artistic merit and subject matter."
        ),
        "checklist": [
            ("Focus / blur", "Edges of the main subject, fine textures (hair, foliage, fabric, text): crisp or smeared? Any motion blur or misfocus?"),
            ("Noise", "Uniform areas (sky, walls, shadows): luminance grain or color speckles? Heavier in dark regions?"),
            ("Exposure", "Clipped highlights (detail-less white blobs)? Crushed shadows (detail-less black areas)? Overall too dark/bright?"),
            ("White balance", "Unnatural color cast (too warm/cool/green)?"),
            ("Compression / artifacts", "8x8 JPEG blocks, ringing near edges, banding in gradients, mosquito noise?"),
            ("Chromatic aberration", "Purple/green fringes on high-contrast edges?"),
        ],
        "levels": [
            "Pristine: crisp detail, clean tones, no visible artifacts of any kind.",
            "Minor softness or slight noise, only visible on close inspection; not distracting.",
            "Visible noise, blur, or exposure issues that clearly reduce clarity.",
            "Strong distortions; obvious detail loss at first glance.",
            "Extreme distortions; content barely recognizable.",
        ],
    },
    "S-AESTH": {
        "name": "Aesthetic Quality Assessor",
        "dimension": (
            "You assess ONLY the aesthetic merit: composition, lighting aesthetics, color harmony, "
            "and visual hierarchy. Judge as visual experience, not technical fidelity."
        ),
        "checklist": [
            ("Composition", "Subject placement (rule of thirds, intentional centering)? Visual balance? Leading lines? Cluttered background?"),
            ("Lighting aesthetics", "Directional/moody light vs flat or harsh light? Does light model the subject well?"),
            ("Color harmony", "Cohesive palette (complementary/analogous) vs clashing colors? Pleasant tonal relationships?"),
            ("Visual hierarchy", "Does the eye land on the intended subject? Clear depth (foreground/background separation)?"),
        ],
        "levels": [
            "Compelling: strong composition and beautiful light; gallery-worthy.",
            "Pleasing overall; only minor compositional or lighting weaknesses.",
            "Ordinary snapshot feel; nothing offensive, nothing memorable.",
            "Awkward framing, harsh light, or clashing colors that detract from the subject.",
            "Chaotic or harsh; actively unpleasant to look at.",
        ],
    },
    "S-CONTENT": {
        "name": "Content Integrity Assessor",
        "dimension": (
            "You assess ONLY whether the image content is complete and well-preserved: subject "
            "completeness, cropping, occlusion, visibility of key regions. Not about style."
        ),
        "checklist": [
            ("Subject completeness", "Is the main subject fully inside the frame, or cut off at edges?"),
            ("Occlusion", "Is the subject blocked by foreground objects, people, or artifacts?"),
            ("Key-region visibility", "Faces, text, salient objects: clearly visible or hidden/truncated?"),
            ("Framing intent vs accident", "Does cropping look intentional and harmless, or accidental and harmful?"),
        ],
        "levels": [
            "Complete: subject and all key regions fully preserved.",
            "Minor edge cropping that does not harm understanding.",
            "Noticeable cropping/occlusion of secondary content; main content still readable.",
            "Main subject partially cut or occluded; the message is weakened.",
            "Subject severely cut or hidden; content cannot be understood.",
        ],
    },
    "S-NATURAL": {
        "name": "Naturalness Assessor",
        "dimension": (
            "You assess ONLY whether the image looks natural and unprocessed: detect over-sharpening, "
            "HDR artifacts, oversaturation, and over-smoothing. Ignore subject and composition."
        ),
        "checklist": [
            ("Over-sharpening", "Bright/dark halos along edges? Crunchy, brittle textures?"),
            ("HDR / tone-mapping", "Flat, glowy, or surreal rendering? Halos around dark objects against bright backgrounds?"),
            ("Oversaturation", "Neon, unrealistic colors? Skin tones turning orange/red?"),
            ("Over-smoothing", "Waxy/plastic skin, watercolor-like smearing from aggressive noise reduction?"),
        ],
        "levels": [
            "Completely natural; no processing trace anywhere.",
            "Subtle processing visible only on close inspection.",
            "Noticeable processing, but the scene still looks plausible.",
            "Obvious over-processing; the image feels artificial.",
            "Heavily processed; plastic, glowy, or uncanny throughout.",
        ],
    },
    "S-GLOBAL": {
        "name": "Holistic Quality Assessor",
        "dimension": (
            "You assess the OVERALL quality impression exactly as a typical viewer would, "
            "considering everything together without breaking it into dimensions."
        ),
        "checklist": [
            ("First impression", "Immediate gut reaction: good or bad?"),
            ("Anything annoying", "Any single issue that dominates your impression?"),
            ("Overall acceptability", "Would an ordinary viewer find this image acceptable, nice, or defective?"),
        ],
        "levels": [
            "Excellent overall impression; nothing to complain about.",
            "Good; minor flaws, easy to overlook.",
            "Fair; flaws are noticeable but tolerable.",
            "Poor; flaws are annoying and hard to ignore.",
            "Bad; unacceptable quality.",
        ],
    },
}

SKILL_ORDER = ["S-TECH", "S-AESTH", "S-CONTENT", "S-NATURAL", "S-GLOBAL"]

# 规则库 tag → Skill 映射（CKE 规则按 tag 定向注入）
TAG_TO_SKILL = {
    "TECH": "S-TECH", "AESTH": "S-AESTH", "CONTENT": "S-CONTENT",
    "NATURAL": "S-NATURAL", "GLOBAL": "S-GLOBAL", "GENERAL": None,
}


def build_skill_prompt(skill_id: str, scale_key: str, rules: list[str] | None = None) -> str:
    """拼装完整 Skill prompt。rules: 该 Skill 的经验规则（CKE 产物，可为空）。"""
    s = SKILLS[skill_id]
    parts = [
        f"You are an expert image quality assessor specialized as a {s['name']}.",
        f"[Your dimension]\n{s['dimension']}",
        "[Checklist — inspect each aspect]\n" + "\n".join(
            f"- {name}: {desc}" for name, desc in s["checklist"]
        ),
        "[Quality levels for YOUR dimension]\n" + "\n".join(
            f"  {i + 1} = {txt}" for i, txt in enumerate(reversed(s["levels"]), start=0)
        ),
        _PROCEDURE,
        _SCALE_BLOCK[scale_key],
    ]
    if rules:
        parts.append("[Field notes from calibration — apply when relevant]\n" + "\n".join(
            f"  {i + 1}. {r}" for i, r in enumerate(rules)
        ))
    parts.append(_OUTPUT_CONTRACT)
    return "\n\n".join(parts)


def build_r1_bare_prompt(scale_key: str) -> str:
    lo, hi = {"koniq": (1, 5), "spaq": (0, 10)}[scale_key]
    return (f"Rate the overall quality of this image on a scale from {lo} to {hi}. "
            f"Reply with only a single number.")


def build_r1_rich_prompt(scale_key: str) -> str:
    """R1-rich = 仅 S-GLOBAL 专家（带完整细则），无多视角。"""
    return build_skill_prompt("S-GLOBAL", scale_key)


# 裸问四释义：同一意思四种问法，取均值平滑量化噪声
BARE_PARAS = [
    "Rate the overall quality of this image on a scale from {lo} to {hi}. Reply with only a single number.",
    "On a scale from {lo} to {hi}, how would you rate the overall quality of this image? Reply with only a single number.",
    "Give a single overall quality score for this image, from {lo} (worst) to {hi} (best). Reply with only the number.",
    "As an image quality rater, assign one overall quality score from {lo} to {hi} to this image. Reply with only a single number.",
]
