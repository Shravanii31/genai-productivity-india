"""Headless smoke test for app/app.py using streamlit.testing.v1.AppTest.

Three distinct pages behind st.session_state.page ("intro" | "inputs" | "results"):
intro (hero + 5 About cards + "Get started") -> inputs (AI footprint, digitization,
what matters most + "Back" / "See my results") -> results (gauge + charts +
summary + "Edit my inputs"). Only one page's content is ever visible at a time.
"""
from streamlit.testing.v1 import AppTest

at = AppTest.from_file("app/app.py", default_timeout=30)
at.run()
assert not at.exception, f"Exception on initial render: {at.exception}"
assert at.session_state["page"] == "intro", f"Expected default page 'intro', got {at.session_state['page']}"
print("Initial render OK -- page = 'intro'")


def walk(node):
    children = getattr(node, "children", None)
    if children:
        for c in children.values():
            yield c
            yield from walk(c)


def has_gauge():
    for node in walk(at.main):
        body = getattr(getattr(node, "proto", None), "body", "")
        if 'gauge-num">' in body and "%" in body:
            return True
    return False


def about_card_count():
    n = 0
    for node in walk(at.main):
        body = getattr(getattr(node, "proto", None), "body", "")
        if 'class="about-card"' in body:
            n += 1
    return n


# --- Page 1: intro ---
assert len(at.pills) == 0 and len(at.slider) == 0, "No inputs should render on the intro page"
assert not has_gauge(), "No gauge should render on the intro page"
assert about_card_count() == 5, f"Expected 5 About cards on the intro page, found {about_card_count()}"
get_started = [b for b in at.button if "Get started" in b.label]
assert len(get_started) == 1, f"Expected one 'Get started' button, found {len(get_started)}"
print("Page 1 (intro) OK -- 5 About cards, 0 inputs, Get started button present")

get_started[0].click().run()
assert not at.exception, f"Exception after clicking 'Get started': {at.exception}"
assert at.session_state["page"] == "inputs"
print("Navigated intro -> inputs OK")

# --- Page 2: inputs ---
assert len(at.pills) == 2, f"Expected 2 pills widgets, found {len(at.pills)}"
assert len(at.slider) == 5, f"Expected 5 sliders, found {len(at.slider)}"
assert about_card_count() == 0, "About cards should not render on the inputs page"
back_btn = [b for b in at.button if b.label == "← Back"]
assert len(back_btn) == 1, f"Expected one '← Back' button, found {len(back_btn)}"
see_results = [b for b in at.button if "See my results" in b.label]
assert len(see_results) == 1, f"Expected one 'See my results' button, found {len(see_results)}"
print("Page 2 (inputs) OK -- 2 pills, 5 sliders, Back + See my results buttons present")

# Set non-default values before navigating away, to confirm they persist
at.pills[0].set_value(["Generative AI", "Machine learning"]).run()
at.pills[1].set_value(["Summarizing", "Drafting"]).run()
at.slider[2].set_value(40).run()  # reskilling
assert not at.exception, f"Exception interacting with inputs: {at.exception}"
print("Set non-default input values OK")

see_results[0].click().run()
assert not at.exception, f"Exception after clicking 'See my results': {at.exception}"
assert at.session_state["page"] == "results"
print("Navigated inputs -> results OK")

# --- Page 3: results ---
assert len(at.pills) == 0 and len(at.slider) == 0, "Inputs should be hidden on the results page"
assert about_card_count() == 0, "About cards should not render on the results page"
assert has_gauge(), "No prediction number found on the results page"
edit_btn = [b for b in at.button if "Edit my inputs" in b.label]
assert len(edit_btn) == 1, f"Expected one 'Edit my inputs' button, found {len(edit_btn)}"
print("Page 3 (results) OK -- inputs/About hidden, gauge rendered, Edit button present")

# --- Back to inputs: confirm previously-set values were retained, not reset ---
edit_btn[0].click().run()
assert not at.exception, f"Exception after clicking 'Edit my inputs': {at.exception}"
assert at.session_state["page"] == "inputs"
assert len(at.pills) == 2 and len(at.slider) == 5, "Inputs should reappear after going back"
assert at.pills[0].value == ["Generative AI", "Machine learning"], f"Pills value not retained: {at.pills[0].value}"
assert at.slider[2].value == 40, f"Slider value not retained: {at.slider[2].value}"
print("Navigated results -> inputs OK -- previous values retained, not reset")

# --- Back to intro from inputs ---
back_btn = [b for b in at.button if b.label == "← Back"][0]
back_btn.click().run()
assert not at.exception, f"Exception after clicking '← Back': {at.exception}"
assert at.session_state["page"] == "intro"
assert about_card_count() == 5, "About cards should be back on the intro page"
print("Navigated inputs -> intro OK")

print("\nALL CHECKS PASSED.")
