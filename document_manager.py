# document_manager.py
import os
import re
from datetime import datetime

# --- Document Template Configuration ---
TEMPLATE_DIR = "./document_templates"
DOCUMENT_TEMPLATES = {
    "nda": "simple_nda_template.txt",
    "non-disclosure agreement": "simple_nda_template.txt",
    # Add more mappings as you create more templates (e.g., "lease agreement": "lease_agreement_template.txt")
}

# --- Placeholder Descriptions ---
# This dictionary maps placeholder names (from your templates) to user-friendly explanations.
# You will need to expand this as you add more templates and placeholders.
PLACEHOLDER_DESCRIPTIONS = {
    "PARTY_ONE_NAME": "The full legal name of the Disclosing Party (the one sharing confidential information)",
    "PARTY_ONE_ADDRESS": "The complete address of the Disclosing Party",
    "PARTY_TWO_NAME": "The full legal name of the Receiving Party (the one receiving confidential information)",
    "PARTY_TWO_ADDRESS": "The complete address of the Receiving Party",
    "CONFIDENTIAL_INFO_DESCRIPTION": "A brief description of the type of confidential information being shared (e.g., business plans, product designs, customer lists)",
    "CONFIDENTIAL_INFO_EXAMPLES": "Specific examples of confidential information (e.g., 'technical data, formulas, marketing strategies')",
    "agreement_term_months": "The duration of the agreement in months (e.g., 12 for one year)",
    # Add descriptions for other placeholders here, if you add them to your templates
    # "CLIENT_NAME": "Full legal name of the client",
    # "SERVICE_DESCRIPTION": "Detailed description of the services to be provided",
}


def handle_document_filling(template_key):
    """
    Guides the user through filling out a document template.
    """
    template_filename = DOCUMENT_TEMPLATES.get(template_key)
    if not template_filename:
        print(f"Aleks: Error: No template found for '{template_key}'.")
        return

    template_path = os.path.join(TEMPLATE_DIR, template_filename)

    if not os.path.exists(template_path):
        print(f"Aleks: Error: Template file '{template_filename}' not found at '{template_path}'.")
        return

    print(f"\nAleks: Okay, let's fill out your '{template_key}' template.")
    
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            template_content = f.read()
    except Exception as e:
        print(f"Aleks: Error reading template file: {e}")
        return

    placeholders = set(re.findall(r'\[(.*?)\]|\{\{(.*?)\}\}', template_content))
    placeholders = {p.strip() for tup in placeholders for p in tup if p.strip()}
    
    if 'current_date' in placeholders:
        placeholders.remove('current_date') # 'current_date' is handled automatically

    filled_data = {}
    print("\nAleks: Please provide the following details:")

    if 'current_date' in template_content:
        filled_data['current_date'] = datetime.now().strftime("%B %d, %Y")
        print(f"Aleks: Setting current date to: {filled_data['current_date']}")

    for placeholder in sorted(list(placeholders)):
        description = PLACEHOLDER_DESCRIPTIONS.get(placeholder, placeholder.replace('_', ' ').title())
        user_input = input(f"Aleks: {description}: ") # Use description in prompt
        filled_data[placeholder] = user_input

    filled_document = template_content
    for placeholder, value in filled_data.items():
        filled_document = re.sub(rf"\[{re.escape(placeholder)}\]|" + r"\{\{" + re.escape(placeholder) + r"\}\}", value, filled_document)

    print("\n" + "="*50)
    print("Aleks: Here is your filled document preview:")
    print("="*50)
    print(filled_document)
    print("="*50 + "\n")

    print("Aleks: Please review the document carefully.")
    review_correct = input("Aleks: Is everything correct? (yes/no): ").strip().lower()

    if review_correct == 'yes':
        print("Aleks: Great! The document is finalized.")
        output_filename = f"filled_{template_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        # Output saved in the same TEMPLATE_DIR for now, could be changed to an 'output' folder
        output_path = os.path.join(TEMPLATE_DIR, output_filename)
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(filled_document)
            print(f"Aleks: Document saved as '{output_filename}' in the '{TEMPLATE_DIR}' folder.")
            print("Aleks: (Mock process for sending to government agency complete.)")
        except Exception as e:
            print(f"Aleks: Error saving document: {e}")
    else:
        print("Aleks: Okay, please indicate what needs to be changed for future improvements.")
        print("Aleks: For now, you can manually edit the content from the preview above.")