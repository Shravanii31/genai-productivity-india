"""Headless smoke test for app/app.py using streamlit.testing.v1.AppTest.

Two distinct views behind a single session_state.show_results flag:
input page (hero + AI footprint + digitization + what matters most + "See my
results" CTA) -> results page (gauge + charts + summary, with "Edit my inputs"
to go back). Inputs and results are never both visible at once.
"""
from streamlit.testing.v1 import AppTest

at = AppTest.from_file("app/app.py", default_timeout=30)
at.run()
assert not at.exception, f"Exception on initial render: {at.exception}"
print("Initial render OK")

# Input page: all inputs present, no results yet, no "Edit" button
assert len(at.pills) == 2, f"Expected 2 pills widgets, found {len(at.pills)}"
assert len(at.slider) == 5, f"Expected 5 sliders, found {len(at.slider)}"
cta = [b for b in at.button if "See my results" in b.label]
assert len(cta) == 1, f"Expected one 'See my results' button, found {len(cta)}"
edit_btn = [b for b in at.button if "Edit my inputs" in b.label]
assert len(edit_btn) == 0, "Edit button should not exist on the input page"
print(f"Input page OK: {len(at.pills)} pills, {len(at.slider)} sliders, CTA present, no Edit button")


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


assert not has_gauge(), "Result gauge should not render on the input page"

# Set some non-default values before switching pages, to confirm they persist
at.pills[0].set_value(["Generative AI", "Machine learning"]).run()
at.pills[1].set_value(["Summarizing", "Drafting"]).run()
at.slider[2].set_value(40).run()  # reskilling
assert not at.exception, f"Exception interacting with inputs: {at.exception}"
print("Set non-default input values OK")

cta[0].click().run()
assert not at.exception, f"Exception after clicking 'See my results': {at.exception}"
print("Clicked 'See my results' OK")

# Results page: inputs are gone entirely, gauge is present, Edit button is present
assert len(at.pills) == 0, f"Pills should be hidden on the results page, found {len(at.pills)}"
assert len(at.slider) == 0, f"Sliders should be hidden on the results page, found {len(at.slider)}"
assert has_gauge(), "No prediction number found on the results page"
edit_btn = [b for b in at.button if "Edit my inputs" in b.label]
assert len(edit_btn) == 1, f"Expected one 'Edit my inputs' button, found {len(edit_btn)}"
cta_gone = [b for b in at.button if "See my results" in b.label]
assert len(cta_gone) == 0, "'See my results' button should not exist on the results page"
print("Results page OK: inputs hidden, gauge rendered, Edit button present")

# Go back and confirm inputs retained their previously-set values (not reset)
edit_btn[0].click().run()
assert not at.exception, f"Exception after clicking 'Edit my inputs': {at.exception}"
assert len(at.pills) == 2 and len(at.slider) == 5, "Inputs should reappear after going back"
assert at.pills[0].value == ["Generative AI", "Machine learning"], f"Pills value not retained: {at.pills[0].value}"
assert at.slider[2].value == 40, f"Slider value not retained: {at.slider[2].value}"
print("Back on input page OK: previous values retained, not reset")

print("\nALL CHECKS PASSED.")
