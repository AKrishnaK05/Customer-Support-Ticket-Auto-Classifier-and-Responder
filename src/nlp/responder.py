def generate_response(category, subject, action):

    billing_template = "We're sorry for the issue with {subject}. We will process your {action} shortly."
    technical_template = "We understand your issue with {subject}. Our team is working to {action} it."
    account_template = "We'll help you resolve your account issue related to {subject}."
    fraud_template = "We take security seriously. We are reviewing the {subject} issue."
    general_template = "Thanks for reaching out regarding {subject}. We'll assist you soon."

    if category == "billing":
        return billing_template.format(subject=subject, action=action)

    elif category == "technical":
        return technical_template.format(subject=subject, action=action)

    elif category == "account":
        return account_template.format(subject=subject)

    elif category == "fraud":
        return fraud_template.format(subject=subject)

    else:
        return general_template.format(subject=subject)
