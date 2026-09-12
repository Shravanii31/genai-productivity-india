"""Headless smoke test for app/app.py using streamlit.testing.v1.AppTest."""
from streamlit.testing.v1 import AppTest

at = AppTest.from_file("app/app.py", default_timeout=30)
at.run()
assert not at.exception, f"Exception on intro step: {at.exception}"
print("Step 0 (Intro) OK")

at.button[0].click().run()
assert not at.exception, f"Exception after clicking Start: {at.exception}"
print("Step 1 (Firm Profile) OK -- widgets:", len(at.slider) + len(at.number_input))

# Exercise real interactivity: move a slider and flip the segmented control
at.slider[0].set_value(80).run()
assert not at.exception, f"Exception setting slider: {at.exception}"
at.segmented_control[0].set_value("No").run()
assert not at.exception, f"Exception setting segmented_control: {at.exception}"
print("Interacted with slider + segmented_control OK")

# Step 1: Firm Profile -> Continue
next_btn = [b for b in at.button if b.label == "Continue"][0]
next_btn.click().run()
assert not at.exception, f"Exception after Firm Profile: {at.exception}"
print("Step 2 (AI Usage) OK")

# Step 2: AI Usage -- exercise the pills multi-select widgets
assert len(at.pills) >= 2, f"Expected 2 pills widgets, found {len(at.pills)}"
at.pills[0].set_value(["Generative AI", "Machine learning"]).run()
assert not at.exception, f"Exception setting AI-tech pills: {at.exception}"
at.pills[1].set_value(["Summarizing", "Drafting"]).run()
assert not at.exception, f"Exception setting AI-task pills: {at.exception}"
print("Pills interaction OK:", at.pills[0].value, at.pills[1].value)
next_btn = [b for b in at.button if b.label == "Continue"][0]
next_btn.click().run()
assert not at.exception, f"Exception after AI Usage: {at.exception}"
print("Step 3 (Digitization) OK")

next_btn = [b for b in at.button if b.label == "Continue"][0]
next_btn.click().run()
assert not at.exception, f"Exception after Digitization: {at.exception}"
print("Step 4 (Operations) OK")

next_btn = [b for b in at.button if b.label == "Continue"][0]
next_btn.click().run()
assert not at.exception, f"Exception after Operations: {at.exception}"
print("Step 5 (Policy) OK")

next_btn = [b for b in at.button if b.label == "See my result"][0]
next_btn.click().run()
assert not at.exception, f"Exception reaching Result: {at.exception}"
print("Step 6 (Result) OK")

# Confirm a real prediction number rendered
found_pct = False
for md in at.markdown:
    if "gauge-num" in md.value and "%" in md.value:
        found_pct = True
        print("Found prediction markup:", [line for line in md.value.splitlines() if "gauge-num" in line])
assert found_pct, "No prediction number found on result screen"

print("\nALL STEPS PASSED -- no exceptions, prediction rendered.")
