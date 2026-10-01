"""Build data/sample_textbook.pdf: a short, original biology chapter used for demos and tests."""
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

CHAPTER = {
    "Chapter 5: Photosynthesis and Plant Nutrition": [],
    "5.1 Introduction": [
        "All living organisms need energy to survive. Green plants are autotrophs because they make their "
        "own food from simple inorganic substances. Animals are heterotrophs because they depend on other "
        "organisms for food.",
        "Photosynthesis is the process by which green plants convert light energy into chemical energy stored "
        "in glucose. The raw materials of photosynthesis are carbon dioxide and water. Oxygen is released as a "
        "by-product of photosynthesis.",
        "The overall equation shows that six molecules of carbon dioxide react with six molecules of water to "
        "form one molecule of glucose and six molecules of oxygen.",
    ],
    "5.2 The Leaf and the Chloroplast": [
        "The leaf is the main site of photosynthesis in most plants. A leaf has a broad, flat lamina that "
        "provides a large surface area for absorbing sunlight.",
        "Chloroplasts are the organelles in which photosynthesis takes place. Each chloroplast is surrounded "
        "by a double membrane and contains stacks of thylakoids called grana.",
        "Chlorophyll is a green pigment that absorbs light energy. Chlorophyll absorbs mainly red and blue "
        "light and reflects green light, which is why leaves look green.",
        "The stroma is the fluid-filled space of the chloroplast that surrounds the thylakoids. Most cells of "
        "the palisade mesophyll contain many chloroplasts because this layer receives the most light.",
        "Stomata are tiny pores on the lower surface of the leaf. Gas exchange takes place through the stomata. "
        "Each stoma is surrounded by two guard cells that control its opening and closing.",
    ],
    "5.3 Light-Dependent Reactions": [
        "The light-dependent reactions take place in the thylakoid membranes of the chloroplast. In these "
        "reactions, light energy is used to split water molecules.",
        "Photolysis is the splitting of water molecules by light energy. Photolysis releases oxygen, protons "
        "and electrons.",
        "The energy carriers ATP and NADPH are produced during the light-dependent reactions. ATP is the main "
        "energy currency of the cell.",
        "The electron transport chain passes electrons between proteins in the thylakoid membrane and pumps "
        "protons into the thylakoid space.",
    ],
    "5.4 The Calvin Cycle": [
        "The Calvin cycle is a series of light-independent reactions that take place in the stroma. The Calvin "
        "cycle uses ATP and NADPH to convert carbon dioxide into sugar.",
        "Carbon fixation is the first stage of the Calvin cycle. During carbon fixation, carbon dioxide is "
        "attached to a five-carbon sugar called ribulose bisphosphate.",
        "Rubisco is the enzyme that catalyses carbon fixation. Rubisco is believed to be the most abundant "
        "protein on Earth.",
        "The glucose produced by photosynthesis may be converted into starch for storage. Starch is stored in "
        "the leaves and in storage organs such as tubers.",
    ],
    "5.5 Factors Affecting the Rate of Photosynthesis": [
        "A limiting factor is a factor that is in shortest supply and therefore restricts the rate of a "
        "process. Light intensity, carbon dioxide concentration and temperature are the main limiting factors "
        "of photosynthesis.",
        "As light intensity increases, the rate of photosynthesis increases until another factor becomes "
        "limiting. Greenhouse growers often add carbon dioxide to increase crop yield.",
        "Temperature affects photosynthesis because the reactions are controlled by enzymes. At very high "
        "temperatures, enzymes are denatured and the rate of photosynthesis falls sharply.",
        "The compensation point is the light intensity at which the rate of photosynthesis equals the rate of "
        "respiration. At the compensation point there is no net exchange of gases.",
    ],
    "5.6 Mineral Nutrition": [
        "Plants absorb mineral ions from the soil through their root hair cells. Root hair cells have a long "
        "extension that increases the surface area for absorption.",
        "Nitrogen is needed to make amino acids and proteins. A plant lacking nitrogen shows stunted growth "
        "and yellow older leaves.",
        "Magnesium is an essential part of the chlorophyll molecule. A shortage of magnesium causes "
        "chlorosis, which is the yellowing of leaves between the veins.",
        "Xylem vessels transport water and mineral ions from the roots to the leaves. Phloem tissue "
        "transports sugars from the leaves to other parts of the plant.",
        "Translocation is the movement of sucrose and amino acids through the phloem. Transpiration refers to "
        "the loss of water vapour from the leaves through the stomata.",
    ],
    "Summary": [
        "Photosynthesis converts light energy into chemical energy in chloroplasts. The light-dependent "
        "reactions produce ATP, NADPH and oxygen, while the Calvin cycle fixes carbon dioxide into sugars. "
        "The rate of photosynthesis depends on limiting factors such as light intensity, carbon dioxide "
        "concentration and temperature.",
    ],
}


def build(path: Path) -> Path:
    st = getSampleStyleSheet()
    story = []
    for i, (heading, paras) in enumerate(CHAPTER.items()):
        if i in (3, 5):  # spread the chapter over several pages
            story.append(PageBreak())
        story.append(Paragraph(heading, st["Title"] if i == 0 else st["Heading2"]))
        for p in paras:
            story.append(Paragraph(p, st["BodyText"]))
            story.append(Spacer(1, 6))
    path.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(path), pagesize=A4, title="Photosynthesis and Plant Nutrition").build(story)
    return path


if __name__ == "__main__":
    out = build(Path(__file__).resolve().parent.parent / "data" / "sample_textbook.pdf")
    print(f"wrote {out}")
