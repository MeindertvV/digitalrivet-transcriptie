"""The languages of the translations, in one place (REFACTOR_PLAN.md step 69).

TRANSLATED is what translate.py makes, named as its prompts name them; the duplicates gate (common/dubbel.py) counts
a translation by it. LABELS is how the (Dutch) reader shows a language; other codes are shown as they are.
frontend/assets/zoeken.js has its own names, for the browser.
"""

TRANSLATED = {"en": "English", "nl": "Dutch"}
LABELS = {"nl": "Nederlands", "en": "Engels", "de": "Duits", "fr": "Frans", "oc": "Opera Calvini"}
